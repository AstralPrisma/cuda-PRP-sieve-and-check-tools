#!/usr/bin/env python3
"""Bounded CPU-only GPRSV output oracle. Never runs GPRSV, CUDA, or networking.

Default CLI only reads input files and writes JSON to stdout. Optional --report
may create a NEW report strictly inside this audit directory. Native source,
survivor and factor files are never changed. Generated-prime spans <=100000 and
candidate k spans <=4096; this is a correctness oracle, not a production sieve.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PMAX = (1 << 62) - 1
KMAX = 1 << 62
U64MAX = (1 << 64) - 1
MAX_SPAN = 100000
MAX_K_SPAN = 4096
MAX_ROWS = 8192
MAX_N = 100000
MAX_BITS = 2000000
MAX_WORK = 8000000
MAX_FILE_BYTES = 8 * 1024 * 1024
MR_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
MULT = r"(?:\d+\^\d+|\d+[#!])"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prime64(value: int) -> bool:
    """Deterministic Miller-Rabin in the bounded unsigned-64 domain."""
    require(0 <= value <= U64MAX, "prime test outside uint64")
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value == p:
            return True
        if value % p == 0:
            return False
    d, s = value - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        if a % value == 0:
            continue
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def small_primes(n: int) -> list[int]:
    require(0 <= n <= MAX_N, f"product index limited to {MAX_N}")
    composite = bytearray(n + 1)
    for p in range(2, math.isqrt(n) + 1):
        if not composite[p]:
            composite[p * p:n + 1:p] = b"\1" * ((n - p * p) // p + 1)
    return [p for p in range(2, n + 1) if not composite[p]]


def interval_primes(low: int, high: int) -> list[int]:
    require(1 <= low <= PMAX and 1 <= high <= PMAX, "P bounds outside supported range")
    if high <= low:
        return []
    require(high - low <= MAX_SPAN, f"oracle refuses P span above {MAX_SPAN}")
    values = [2] if low < 2 <= high else []
    first = max(3, low + 1)
    first += (first % 2 == 0)
    return values + [p for p in range(first, high + 1, 2) if prime64(p)]


@dataclass(frozen=True)
class Spec:
    family: str
    n: int
    base: int = 0

    def multiplier(self) -> int:
        require(1 <= self.n <= MAX_N, f"n must be in 1..{MAX_N} for this oracle")
        if self.family == "factorial":
            value = math.factorial(self.n)
        elif self.family == "primorial":
            # Actual GPRSV requires n itself to be prime (GFPPS need not).
            require(prime64(self.n), "GPRSV primorial index n must itself be prime")
            value = math.prod(small_primes(self.n))
        else:
            require(self.family == "bn" and 2 <= self.base <= 1 << 31, "invalid b^n definition")
            value = pow(self.base, self.n)
        require(value.bit_length() <= MAX_BITS, f"exact multiplier exceeds {MAX_BITS} oracle bits")
        return value

    def text(self) -> str:
        return f"{self.base}^{self.n}" if self.family == "bn" else f"{self.n}{'!' if self.family == 'factorial' else '#'}"


def parse_multiplier(text: str) -> Spec:
    if "^" in text:
        b, n = map(int, text.split("^"))
        return Spec("bn", n, b)
    return Spec("factorial" if text.endswith("!") else "primorial", int(text[:-1]))


@dataclass
class Terms:
    spec: Spec
    mode: str
    units: set[tuple[int, int]]  # twin: (k,0); independent: (k,-1)/(k,+1).
    sieved_to: int | None
    format: str

    def describe(self):
        rows = sorted(self.units)
        encoded = "".join(f"{k},{c}\n" for k, c in rows).encode("ascii")
        return {"type": self.spec.family, "n": self.spec.n, "base": self.spec.base,
                "mode": self.mode, "format": self.format, "sieved_to": self.sieved_to,
                "units": len(rows), "individual_terms": len(rows) * (2 if self.mode == "twin" else 1),
                "survivor_set_sha256": hashlib.sha256(encoded).hexdigest(), "survivors": rows}


def read_text(path: Path) -> str:
    require(path.stat().st_size <= MAX_FILE_BYTES, "input file exceeds bounded oracle size")
    return path.read_text(encoding="utf-8-sig", errors="strict")


def parse_terms(path: Path, fallback: tuple[Spec, str] | None = None) -> Terms:
    text = read_text(path)
    if not text.strip():
        require(fallback is not None, "empty ABCD file has no metadata; specify expected type/n/mode")
        return Terms(fallback[0], fallback[1], set(), None, "empty")
    lines = text.splitlines()
    header = lines[0].strip()
    boundary = 1
    tail = re.search(r"\s*//\s*Sieved\s+to\s+(\d+)\s*$", header, re.I)
    if tail:
        boundary = int(tail[1])
        header = header[:tail.start()].rstrip()
    twin = re.fullmatch(rf"ABC(D)?\s+\$a\*({MULT})\+1\s*&\s*\$a\*({MULT})-1(?:\s*\[(\d+)\])?", header, re.I)
    independent = re.fullmatch(rf"ABC\s+\$a\*({MULT})\$b", header, re.I)
    npg = re.fullmatch(r"(\d+):T:0:(\d+):3", header, re.I)
    units = set()
    previous = None
    rows = [line.strip() for line in lines[1:] if line.strip()]
    require(len(rows) <= MAX_ROWS, "too many candidate rows for oracle")
    if twin:
        spec = parse_multiplier(twin[2])
        require(spec == parse_multiplier(twin[3]), "twin multiplier headers disagree")
        mode, fmt = "twin", "ABCD" if twin[1] else "ABC"
        require((fmt == "ABCD") == (twin[4] is not None), "ABCD requires exactly one initial k")
        if fmt == "ABCD":
            previous = int(twin[4])
            units.add((previous, 0))
        for row in rows:
            require(re.fullmatch(r"\d+", row), "invalid twin candidate row")
            k = int(row) if fmt == "ABC" else previous + int(row)
            require((k, 0) not in units, "duplicate twin candidate")
            units.add((k, 0))
            previous = k
    elif independent:
        spec, mode, fmt = parse_multiplier(independent[1]), "independent", "ABC"
        for row in rows:
            match = re.fullmatch(r"(\d+)\s+([+-]1)", row)
            require(match is not None, "invalid independent ABC row")
            unit = (int(match[1]), int(match[2]))
            require(unit not in units, "duplicate independent candidate")
            units.add(unit)
    elif npg:
        boundary, base = int(npg[1]), int(npg[2])
        exponent = None
        mode, fmt = "twin", "NewPGen"
        for row in rows:
            match = re.fullmatch(r"(\d+)\s+(\d+)", row)
            require(match is not None, "invalid NewPGen row")
            k, n = int(match[1]), int(match[2])
            require(exponent in (None, n), "NewPGen contains multiple exponents")
            require((k, 0) not in units, "duplicate NewPGen candidate")
            units.add((k, 0))
            exponent = n
        if exponent is None:
            require(fallback is not None and fallback[0].family == "bn", "empty NewPGen needs expected exponent")
            exponent = fallback[0].n
        spec = Spec("bn", exponent, base)
    else:
        raise ValueError("unsupported GPRSV candidate header")
    require(1 <= boundary <= PMAX, "invalid sieve frontier")
    require(all(1 <= k <= KMAX for k, _ in units), "candidate k outside GPRSV bounds")
    return Terms(spec, mode, units, boundary, fmt)


def generated(spec: Spec, mode: str, low: int, high: int, remove_base=False) -> set[tuple[int, int]]:
    require(1 <= low <= high <= KMAX and high - low + 1 <= MAX_K_SPAN,
            f"oracle k span must be at most {MAX_K_SPAN}")
    ks = list(range(low, high + 1))
    if spec.family == "bn" and (spec.base == 2 or spec.base % 2):
        want = 1 if spec.base == 2 else 0
        ks = [k for k in ks if k % 2 == want]
    if remove_base and spec.family == "bn" and spec.base != 2:
        ks = [k for k in ks if k % spec.base]
    require(ks, "empty normalized k range")
    return {(k, c) for k in ks for c in ((0,) if mode == "twin" else (-1, 1))}


def expected_survivors(spec: Spec, mode: str, initial: set[tuple[int, int]], low: int, high: int,
                       *, source_frontier=1, declared_max_k: int | None = None) -> tuple[Terms, dict]:
    require(initial, "empty initial input is not resumable in GPRSV")
    require(len(initial) <= MAX_ROWS, "too many initial units")
    require(all(c == 0 if mode == "twin" else c in (-1, 1) for _, c in initial), "candidate signs disagree with mode")
    if spec.family == "bn" and (spec.base == 2 or spec.base % 2):
        require(all(k % 2 == (1 if spec.base == 2 else 0) for k, _ in initial), "input violates native k parity")
    require(max(k for k, _ in initial) - min(k for k, _ in initial) + 1 <= MAX_K_SPAN,
            "sparse candidate file has an excessively wide k bitmap span")
    multiplier = spec.multiplier()
    require(all(k * multiplier + (c if c else -1) >= 2 for k, c in initial),
            "this regression oracle excludes candidates below 2")
    effective_low = max(low, source_frontier, spec.n if spec.family != "bn" else 1)
    if spec.family == "bn" and (spec.base == 2 or spec.base % 2):
        effective_low = max(effective_low, 2)
    maximum_k = declared_max_k if declared_max_k is not None else max(k for k, _ in initial)
    largest_term = maximum_k * multiplier + 1
    effective_high = min(high, math.isqrt(largest_term)) if largest_term <= U64MAX else high
    require(effective_high >= effective_low, "pmax shrank below pmin; avoid this degenerate native case")
    primes = interval_primes(effective_low, effective_high)
    work = len(primes) * (max(1, multiplier.bit_length() // 1024) + 2 * len(initial))
    require(work <= MAX_WORK, f"oracle cost estimate {work} exceeds {MAX_WORK}; narrow P/k range")
    survivors = set(initial)
    for prime in primes:
        product_mod = multiplier % prime
        if product_mod == 0:
            continue
        for k, sign in tuple(survivors):
            for c in ((-1, 1) if sign == 0 else (sign,)):
                if ((k % prime) * product_mod + c) % prime == 0 and k * multiplier + c != prime:
                    survivors.remove((k, sign))
                    break
    last = primes[-1] if primes else effective_low
    report = {"requested_pmin_exclusive": low, "requested_pmax_inclusive": high,
              "effective_pmin_exclusive": effective_low, "effective_pmax_inclusive": effective_high,
              "initial_units": len(initial), "removed_units": len(initial) - len(survivors),
              "primes_tested": len(primes), "expected_frontier": last,
              "multiplier_bits": multiplier.bit_length(), "oracle_work_estimate": work}
    return Terms(spec, mode, survivors, last, "oracle"), report


def factor_records(path: Path, spec: Spec) -> list[tuple[int, int, int]]:
    records = []
    for index, raw in enumerate(read_text(path).splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(rf"\s*(\d+)\s*\|\s*(\d+)\*({MULT})([+-])1\s*", raw)
        require(match is not None, f"invalid factor record line {index}")
        prime, k, mult, sign = match.groups()
        require(parse_multiplier(mult) == spec, f"factor multiplier mismatch line {index}")
        prime, k, c = int(prime), int(k), 1 if sign == "+" else -1
        require(2 <= prime <= PMAX and 1 <= k <= KMAX, "factor values outside GPRSV domain")
        records.append((prime, k, c))
        require(len(records) <= MAX_ROWS, "too many factor records for bounded oracle")
    return records


def verify_factors(paths: list[Path], spec: Spec, *, mode=None, initial=None,
                   survivors=None, pmin=1, pmax=PMAX) -> dict:
    multiplier = spec.multiplier()
    seen_units = set()
    records = []
    for path in paths:
        records += factor_records(path, spec)
    require(len(records) <= MAX_ROWS, "too many combined factor records")
    residues = {}
    require(len({p for p, _, _ in records}) * max(1, multiplier.bit_length() // 1024) <= MAX_WORK,
            "factor-only verification exceeds bounded oracle work budget")
    for p, k, c in records:
        require(pmin < p <= pmax, f"factor p={p} outside this pass's interval")
        require(prime64(p), f"reported factor {p} is not prime")
        if p not in residues:
            residues[p] = multiplier % p
        value = residues[p]
        require(((k % p) * value + c) % p == 0, f"invalid divisor: {p} | {k}*{spec.text()}{c:+d}")
        require(k * multiplier + c != p, "number itself was incorrectly reported as a proper factor")
        if mode is not None:
            unit = (k, 0 if mode == "twin" else c)
            require(unit in initial, "factor does not belong to this pass's initial candidate set")
            require(unit not in survivors, "reported factor still has a surviving candidate")
            require(unit not in seen_units, "multiple factor records for one removed unit (stale/appended log?)")
            seen_units.add(unit)
    if mode is not None:
        require(seen_units == initial - survivors, "factor log does not cover every removal exactly once")
    return {"records": len(records), "all_modular_divisibility_checks_passed": True,
            "self_factor_records": 0, "removal_coverage_checked": mode is not None,
            "note": "winning primes may differ between runs/streams; factor text is not compared bytewise"}


def same_survivors(left: Terms, right: Terms):
    require((left.spec, left.mode) == (right.spec, right.mode), "candidate definitions/modes differ")
    require(left.units == right.units,
            f"survivor mismatch: left-only={sorted(left.units-right.units)[:12]}, right-only={sorted(right.units-left.units)[:12]}")


def spec_from_args(args) -> Spec:
    return Spec(args.type, args.n, args.base or 0)


def check_command(args):
    spec, mode = spec_from_args(args), args.mode
    if args.initial:
        original = parse_terms(args.initial, (spec, mode))
        require((original.spec, original.mode) == (spec, mode), "input header overrides CLI definition; provide its actual type/n/mode")
        initial = original.units
        frontier = original.sieved_to or 1
        require(initial, "empty survivor file is not resumable by current GPRSV")
        max_k = max(k for k, _ in initial)  # Native resume derives bitmap bounds before -r filtering.
        if args.remove and spec.family == "bn" and spec.base != 2:
            initial = {(k,c) for k,c in initial if k % spec.base}
    else:
        require(args.kmin is not None and args.kmax is not None, "generated test requires kmin/kmax")
        initial = generated(spec, mode, args.kmin, args.kmax, args.remove)
        frontier = 1
        # Parity normalization changes max_k before native sqrt cap, -r does not.
        max_k = max(k for k, _ in generated(spec, mode, args.kmin, args.kmax))
    expected, diagnostics = expected_survivors(spec, mode, initial, args.pmin, args.pmax,
                                               source_frontier=frontier, declared_max_k=max_k)
    outputs = []
    for path, factors in ((args.actual, args.factors), (args.peer, args.peer_factors)):
        if path is None:
            continue
        actual = parse_terms(path, (spec, mode))
        same_survivors(expected, actual)
        if actual.sieved_to is not None:
            require(actual.sieved_to == expected.sieved_to, "output did not reach the expected completed prime frontier")
        result = {"path": str(path), "survivors": actual.describe(), "status": "PASS",
                  "frontier_verified": actual.sieved_to is not None}
        if factors:
            result["factor_checks"] = verify_factors(factors, spec, mode=mode, initial=initial,
                survivors=actual.units, pmin=diagnostics["effective_pmin_exclusive"],
                pmax=diagnostics["effective_pmax_inclusive"])
        outputs.append(result)
    return {"status": "PASS", "oracle": diagnostics, "outputs": outputs}


def plan_command():
    """A small fixed matrix; selecting target primes examines only a few integers."""
    q32 = (1 << 32) + 1
    while not prime64(q32):
        q32 += 2
    q32_below = (1 << 32) - 1
    while not prime64(q32_below):
        q32_below -= 2
    q62 = PMAX
    while not prime64(q62):
        q62 -= 2
    common = ["--cpu-small-prime", "0", "--threads", "256", "--blocks", "64",
              "--batch-primes", "1024", "--prime-threads", "2", "--prime-prefetch", "2",
              "--cuda-streams", "2", "--segment-mib", "1", "--progress-seconds", "60", "--verify"]
    cases = []
    for family, termtype, nlarge in (("factorial", "3", 31), ("primorial", "2", 67)):
        for mode in ("independent", "twin"):
            for tag, n, lo, hi, pmin, pmax, generator, forced in [
                ("self", 3, 1, 128, 1, 2000, "segmented", None),
                ("small32", 97, 1, 256, 97, 20000, "segmented", None)]:
                cases.append(dict(label=f"{family}_{mode}_{tag}", type=family, termtype=termtype,
                                  n=n, kmin=lo, kmax=hi, pmin=pmin, pmax=pmax, mode=mode,
                                  prime_generator=generator, forced_factor=forced))
            for tag, q, pmin, pmax in (("cross32_below", q32_below, (1 << 32)-50000, (1 << 32)+50000),
                                      ("cross32_above", q32, (1 << 32)-50000, (1 << 32)+50000),
                                      ("near62", q62, PMAX-100000, PMAX)):
                product = Spec(family, nlarge).multiplier()
                inverse = pow(product % q, -1, q)
                for c in (-1, 1):
                    target = inverse if c == -1 else q - inverse
                    low = max(1, target - 32)
                    high = min(KMAX, low + 64)
                    cases.append(dict(label=f"{family}_{mode}_{tag}_{'minus' if c < 0 else 'plus'}",
                                      type=family, termtype=termtype, n=nlarge, kmin=low, kmax=high,
                                      pmin=pmin, pmax=pmax, mode=mode, prime_generator="mr",
                                      forced_factor={"p":q,"k":target,"c":c}))
    for case in cases:
        argv = ["-t",case["termtype"],"-n",str(case["n"]),"-k",str(case["kmin"]),"-K",str(case["kmax"]),
                "-p",str(case["pmin"]),"-P",str(case["pmax"]),"-f","A","--prime-generator",case["prime_generator"]]
        if case["mode"] == "independent":
            argv += ["-s"]
        case["gprsv_arguments"] = argv + common + ["-o",case["label"]+".pfgw","-O",case["label"]+".factors.txt"]
    return {"status":"PLAN_ONLY_NO_GPU_RUNS", "cases":cases,
            "repeat_controls":{"cuda_streams":[1,2],"prime_threads":[1,2],"format_twin":["A","D"],
                               "cpu_small_prime_compare":[0,100000]},
            "notes":["Use new output/factor paths per implementation/pass; factors append.",
                     "Split a selected P interval at its midpoint, resume -i pass1 -P originalPmax; compare full survivor set.",
                     "CLI -f is overwritten by an input header on resume; do not assume format conversion.",
                     "Do not use SIGINT durations for performance. Use identical finite P bounds, normal completion, same generator/threads/batch/streams.",
                     "No --prime-generator builtin exists. Explicit segmented/mr bypass optional primesieve.",
                     "Use mr near 2^62; segmented there would generate primes through approximately 2^31."]}


def selftest():
    for p in (2,3,5,97,4294967311,(1 << 61)-1):
        require(prime64(p), "prime selftest")
    for c in (0,1,4,91,341,4294967295,PMAX):
        require(not prime64(c), "composite selftest")
    checks = []
    for family in ("factorial", "primorial"):
        spec = Spec(family,3)
        for mode in ("independent", "twin"):
            initial = generated(spec,mode,1,128)
            expected, diagnostics = expected_survivors(spec,mode,initial,1,2000)
            require((1,0) in expected.units if mode == "twin" else {(1,-1),(1,1)} <= expected.units,
                    "self-prime 5/7 was removed")
            direct = set()
            for k,c in initial:
                signs = (-1,1) if c == 0 else (c,)
                if all(prime64(k*6+sign) for sign in signs):
                    direct.add((k,c))
            require(expected.units == direct, "small complete sieve disagrees with exact primality")
            first, _ = expected_survivors(spec,mode,initial,1,13)
            resumed, _ = expected_survivors(spec,mode,first.units,1,2000,source_frontier=first.sieved_to)
            same_survivors(expected,resumed)
            checks.append({"type":family,"mode":mode,**diagnostics})
    return {"status":"PASS_CPU_ONLY", "checks":checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="optional NEW report below this audit directory")
    subs = parser.add_subparsers(dest="command",required=True)
    subs.add_parser("plan")
    subs.add_parser("selftest")
    inspect = subs.add_parser("parse")
    inspect.add_argument("file",type=Path)
    compare = subs.add_parser("compare")
    compare.add_argument("left",type=Path)
    compare.add_argument("right",type=Path)
    check = subs.add_parser("check")
    factors = subs.add_parser("factors")
    for command in (check,factors):
        command.add_argument("--type",choices=("factorial","primorial","bn"),required=True)
        command.add_argument("--n",type=int,required=True)
        command.add_argument("--base",type=int,default=0)
        command.add_argument("--pmin",type=int,default=1)
        command.add_argument("--pmax",type=int,default=PMAX if command is factors else None,required=command is check)
        command.add_argument("--factors",type=Path,action="append",default=[],required=command is factors)
    check.add_argument("--mode",choices=("twin","independent"),default="twin")
    check.add_argument("--kmin",type=int)
    check.add_argument("--kmax",type=int)
    check.add_argument("--remove",action="store_true")
    check.add_argument("--initial",type=Path)
    check.add_argument("--actual",type=Path,required=True)
    check.add_argument("--peer",type=Path)
    check.add_argument("--peer-factors",type=Path,action="append",default=[])
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan_command()
        elif args.command == "selftest":
            result = selftest()
        elif args.command == "parse":
            result = parse_terms(args.file).describe()
        elif args.command == "compare":
            left,right = parse_terms(args.left),parse_terms(args.right)
            same_survivors(left,right)
            result = {"status":"PASS", "full_survivor_sets_equal":True,"left":left.describe(),"right":right.describe()}
        elif args.command == "factors":
            result = verify_factors(args.factors,spec_from_args(args),pmin=args.pmin,pmax=args.pmax)
        else:
            result = check_command(args)
        encoded = json.dumps(result,indent=2) + "\n"
        if args.report:
            target = args.report.absolute()
            require(not target.exists() and not target.is_symlink(), "report must be a new file")
            target = target.resolve()
            require(ROOT in target.parents, "report must be inside this audit directory")
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("x",encoding="utf-8",newline="\n") as stream:
                stream.write(encoded)
        print(encoded,end="")
        return 0
    except (ValueError,OSError) as exc:
        print(json.dumps({"status":"FAIL","error":str(exc)}),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
