"""Catena Siderum Prime Search / Siever. Exact arithmetic; no online services."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

import gmpy2

VERSION = "1.3.0"
ROOT = Path(__file__).resolve().parent
MAX_U64 = (1 << 64) - 1
MAX_SIEVE_DEPTH = (1 << 62) - 1
MAX_PYTHON_SIEVE_DEPTH = 10**12
GPU_MAX_BITS = (1 << 20)*15-16


def parse_unsigned_integer(text):
    """Exact decimal/scientific integer parsing, with no float rounding."""
    if not text or len(text) > 4096 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text):
        raise argparse.ArgumentTypeError("expected an unsigned integer (decimal or exact scientific notation)")
    parts = re.split("[eE]", text)
    mantissa = parts[0]
    digits = mantissa.replace(".", "").lstrip("0")
    if not digits:
        return 0
    exponent = 0
    if len(parts) == 2:
        raw = parts[1]
        negative = raw.startswith("-")
        for digit in raw.lstrip("+-"):
            exponent = min(100000, exponent*10 + ord(digit)-ord("0"))
        if negative:
            exponent = -exponent
    fractional = len(mantissa.rsplit(".", 1)[1]) if "." in mantissa else 0
    shift = exponent - fractional
    if shift < 0:
        remove = -shift
        if remove >= len(digits) or any(ch != "0" for ch in digits[-remove:]):
            raise argparse.ArgumentTypeError("value is not an exact integer")
        digits = digits[:-remove]
    else:
        if len(digits)+shift > 20:
            raise argparse.ArgumentTypeError("unsigned integer overflow")
        digits += "0"*shift
    if len(digits) > 20 or int(digits) > MAX_U64:
        raise argparse.ArgumentTypeError("unsigned integer overflow")
    return int(digits)


@dataclass(frozen=True)
class Candidate:
    start: int
    terms: int
    step: int
    sign: int

    def __post_init__(self):
        if self.start < 1 or self.terms < 1:
            raise ValueError("start and terms must be positive integers")
        if self.step not in (1, 2) or self.sign not in (-1, 1):
            raise ValueError("step must be 1 or 2; sign must be + or -")
        if self.last >= MAX_U64:
            raise ValueError("last base + 1 must fit uint64")

    @property
    def last(self):
        return self.start + self.step * (self.terms - 1)

    def expression(self):
        if self.terms <= 12:
            return "*".join(f"{a}^{a+1}" for a in range(self.start, self.last+1, self.step)) + ("+1" if self.sign == 1 else "-1")
        return f"product((x+{self.step}*i)^(x+{self.step}*i+1),i=0..{self.terms-1},x={self.start})" + ("+1" if self.sign == 1 else "-1")

    def modulus(self, p):
        # p is a verified prime from our segmented generator.
        if p < 2:
            raise ValueError("modulus must be >= 2")
        product = 1
        for a in range(self.start, self.last + 1, self.step):
            if a % p == 0:
                return self.sign % p
            product = product * pow(a, (a + 1) % (p - 1), p) % p
        return (product + self.sign) % p

    def capped_value(self, cap):
        """Exact value up to cap; cap+1 means larger (no float comparisons)."""
        prod, product_cap = 1, cap + 1
        for a in range(self.start, self.last+1, self.step):
            if a == 1:
                continue
            power, base, exp = 1, a, a+1
            while exp:
                if exp & 1:
                    power *= base
                    if power > product_cap:
                        return cap+1
                exp >>= 1
                if exp:
                    base = min(product_cap+1, base*base)
            prod *= power
            if prod > product_cap:
                return cap+1
        return min(cap+1, prod+self.sign)

    def value(self, max_bits=16_000_000):
        # A rigorous lower bound rejects huge requests before constructing powers.
        if self.terms > 1_000_000:
            raise ValueError("terms exceeds the one-million-term resource guard")
        lower = 0
        for a in range(self.start, self.last+1, self.step):
            lower += (a+1)*(a.bit_length()-1)
            if lower > max_bits+1:
                raise ValueError(f"candidate exceeds max-bits={max_bits}")
        def tree(lo, hi):
            if hi-lo == 1:
                a = self.start+self.step*lo
                return gmpy2.mpz(a) ** (a+1)
            mid = (lo+hi)//2
            return tree(lo, mid)*tree(mid, hi)
        n = tree(0, self.terms)+self.sign
        if n.bit_length() > max_bits:
            raise ValueError(f"candidate exceeds max-bits={max_bits}")
        return n


def primes(lo, hi, segment=1 << 18):
    if hi > MAX_PYTHON_SIEVE_DEPTH:
        raise ValueError("Python sieve supports pmax<=1e12; select a native CPU/GPU backend for deeper sieving")
    if hi < 2 or lo > hi:
        return
    limit = math.isqrt(hi)
    small = bytearray(b"\1")*(limit+1)
    if limit >= 0:
        small[0] = 0
    if limit >= 1:
        small[1] = 0
    for p in range(2, math.isqrt(limit)+1):
        if small[p]:
            first = p*p
            small[first::p] = b"\0"*((limit-first)//p+1)
    base = [p for p in range(2, limit+1) if small[p]]
    for low in range(max(2, lo), hi+1, segment):
        high = min(hi, low+segment-1)
        flags = bytearray(b"\1")*(high-low+1)
        for p in base:
            first = max(p*p, ((low+p-1)//p)*p)
            if first <= high:
                flags[first-low::p] = b"\0"*((high-first)//p+1)
        yield from (low+i for i, flag in enumerate(flags) if flag)


def sieve_python(candidates, pmin, pmax):
    factors = [0]*len(candidates)
    small = [c.capped_value(pmax) for c in candidates]
    groups = {}
    for i, c in enumerate(candidates):
        groups.setdefault((c.start, c.terms, c.step), []).append(i)
    for p in primes(pmin, pmax):
        for indexes in groups.values():
            active = [i for i in indexes if not factors[i] and small[i] > p]
            if not active:
                continue
            c = candidates[active[0]]
            # Any full residue cycle includes a base divisible by p.
            if p != 2 and c.terms >= p:
                continue
            product = (c.modulus(p) - c.sign) % p
            for i in active:
                if (product + candidates[i].sign) % p == 0:
                    factors[i] = p
    return factors


def native_binary(name):
    suffix = ".exe" if os.name == "nt" else ""
    p = ROOT / "bin" / (name+suffix)
    return p if p.is_file() else None


def sieve(candidates, pmin, pmax, backend="auto"):
    if not 2 <= pmin <= pmax <= MAX_SIEVE_DEPTH:
        raise ValueError(f"require 2<=pmin<=pmax<={MAX_SIEVE_DEPTH}")
    if not candidates:
        return [], "none"
    exe = native_binary("CSSV")
    if backend == "python" or (not exe and backend == "auto"):
        return sieve_python(candidates, pmin, pmax), "python-exact"
    if not exe:
        raise ValueError("CSSV executable missing; build it or select --backend python")
    c = candidates[0]
    signs = {item.sign for item in candidates}
    both = signs == {-1, 1}
    width = 2 if both else 1
    expected = [(x, s) for x in range(c.start, candidates[-1].start+1)
                for s in ((-1, 1) if both else (c.sign,))]
    if ([(item.start, item.sign) for item in candidates] != expected
            or any((item.terms, item.step) != (c.terms, c.step) for item in candidates)):
        raise ValueError("native sieve needs contiguous starts, equal terms/step, and minus-then-plus pairs for both")
    with tempfile.TemporaryDirectory(prefix="catena-sieve-") as temp:
        out = Path(temp)/"factors.txt"
        native_backend = "cpu" if backend == "auto" and len(candidates)//width*c.terms <= 256 else backend
        cmd = [str(exe), "--start", str(c.start), "--end", str(candidates[-1].start),
               "--terms", str(c.terms), "--step", str(c.step), "--sign", "both" if both else "+" if c.sign == 1 else "-",
               "--pmin", str(pmin), "--pmax", str(pmax), "--out", str(out),
               "--backend", native_backend, "--format", "factors"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"CSSV failed ({result.returncode}): {result.stderr.strip()}")
        rows = [row.split() for row in out.read_text("ascii").splitlines()]
        if len(rows) != len(candidates)//width or any(len(row) != width for row in rows):
            raise RuntimeError("CSSV output row/column count mismatch")
        values = [int(x) for row in rows for x in row]
        if len(values) != len(candidates):
            raise RuntimeError("CSSV output count mismatch")
        for c, factor in zip(candidates, values):
            if factor and (factor < pmin or factor > pmax or not gmpy2.is_prime(factor)
                           or c.modulus(factor) != 0 or c.capped_value(factor) <= factor):
                raise RuntimeError("CSSV returned an invalid proper factor")
        return values, result.stdout.strip()


def checked_result(c, backend="auto", max_bits=16_000_000, checkpoint=None):
    n = c.value(max_bits)
    record = {**asdict(c), "expression": c.expression(), "bits": n.bit_length(),
              "digits": len(n.digits(10)), "value_sha256": hashlib.sha256(n.digits(16).encode("ascii")).hexdigest(),
              "value_hash_encoding": "ASCII lowercase hexadecimal, no prefix or newline"}
    if n < 2:
        return {**record, "result": "NOT_PRIME", "test": "N<2"}, n
    exe = native_binary("CSPS")
    # Small inputs are dominated by CUDA launch/setup overhead. Explicit GPU
    # selection remains available for cross-checks and hardware benchmarking.
    use_gpu = backend == "gpu" or (backend == "auto" and 1_000_000 <= n.bit_length() <= GPU_MAX_BITS and exe)
    if use_gpu:
        if n.bit_length() > GPU_MAX_BITS:
            raise ValueError(f"GPU candidate limit is {GPU_MAX_BITS} bits; use --backend cpu")
        if not exe:
            raise ValueError("CSPS executable missing; build it or use --backend cpu")
        cmd = [str(exe), "--step", str(c.step), "--start", str(c.start), "--terms", str(c.terms),
               "--sign", "+" if c.sign == 1 else "-", "--progress-every-bits", "10000"]
        if checkpoint:
            ck = Path(checkpoint)
            cmd += ["--checkpoint", str(ck), "--checkpoint-every-bits", "10000"]
            if ck.exists():
                cmd += ["--resume-checkpoint"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        lines, tail = [], []
        try:
            for line in process.stdout:
                print(line, end="", file=sys.stderr, flush=True)
                tail.append(line)
                tail = tail[-30:]
                if line.startswith("csps-check:"):
                    lines.append(line.strip())
            rc = process.wait()
        except KeyboardInterrupt:
            # SIGINT/console Ctrl+C is inherited by the native checkpoint writer.
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                raise RuntimeError("CSPS is still saving; leave it running and inspect its checkpoint")
            raise
        if rc:
            raise RuntimeError(f"CSPS failed ({rc}): {''.join(tail)[-2000:]}")
        if len(lines) != 1:
            raise RuntimeError("CSPS missing unique result")
        outcome = lines[0].rsplit("result=", 1)[-1].split(",", 1)[0].strip()
        if outcome not in ("PRP", "COMPOSITE"):
            raise RuntimeError("CSPS did not finish a complete PRP check")
        # Exact CPU residue check on small GPU regression-sized inputs.
        if n.bit_length() <= 8192:
            expect = n == 2 or (n % 2 and gmpy2.powmod(2, n-1, n) == 1)
            if (outcome == "PRP") != bool(expect):
                raise RuntimeError("GPU/CPU Fermat result mismatch")
        return {**record, "result": outcome, "test": "CUDA-exact-NTT-Fermat-base-2", "engine_output": lines[0]}, n
    outcome = bool(gmpy2.is_prime(n, 25))
    return {**record, "result": "PRP" if outcome else "COMPOSITE", "test": "GMP-probable-prime-reps-25"}, n


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_state(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema": "CatenaSiderum-state-1", "data": data}
    body["sha256"] = hashlib.sha256(encode(body)).hexdigest()
    fd, temp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(encode(body)+b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_state(path):
    body = json.loads(Path(path).read_bytes())
    checksum = body.pop("sha256", None)
    if body.get("schema") != "CatenaSiderum-state-1" or checksum != hashlib.sha256(encode(body)).hexdigest():
        raise ValueError("state format/checksum mismatch")
    return body["data"]


def interactive(kind):
    print("Catena Siderum " + ("Siever" if kind == "sieve" else "Prime Search"))
    print("N = product((x+d*i)^(x+d*i+1), i=0..m-1) +/- 1")
    step = input("类型：1=连续型，2=隔项型 [1]: ").strip() or "1"
    sign = input("符号：+、- 或 both（正负同时） [+]: " if kind == "sieve" else "符号：+ 或 - [+]: ").strip() or "+"
    start = input("起始值 x: ").strip()
    terms = input("项数 m: ").strip()
    return [kind, "--step", step, "--sign", sign, "--start", start, "--terms", terms]


def acquire_locks(paths):
    """Lock both progress and output, even if another job uses another state."""
    stack = ExitStack()
    try:
        for path in sorted({str(p.resolve()) for p in paths if p}):
            lock_path = Path(path+".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(lock_path.open("a+b"))
            handle.seek(0, 2)
            if not handle.tell():
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stack.close()
        raise ValueError("another process owns this state/output, or its lock cannot be opened") from exc
    return stack


def main(argv=None, default_kind="check"):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = interactive(default_kind)
    elif argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, default_kind)
    parser = argparse.ArgumentParser(description="Catena Siderum Prime Search & Catena Siderum Siever")
    parser.add_argument("command", choices=("check", "sieve", "search"))
    parser.add_argument("--step", "--mode", type=int, choices=(1, 2), required=True)
    parser.add_argument("--sign", choices=("+", "-", "both"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--terms", type=int, required=True)
    parser.add_argument("--end", type=int, help="inclusive final start value; default is --start")
    parser.add_argument("--pmin", type=parse_unsigned_integer, default=2)
    parser.add_argument("--pmax", type=parse_unsigned_integer, default=100_000)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu", "python"), default="auto")
    parser.add_argument("--max-bits", type=int, default=16_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--state", type=Path, help="atomic range checkpoint; use with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, help="JSONL output; an existing file is protected")
    parser.add_argument("--print-value", action="store_true")
    args = parser.parse_args(argv)
    if args.sign == "both" and args.command != "sieve":
        parser.error("--sign both currently supports sieve only; use separate signs for check/search")
    end = args.start if args.end is None else args.end
    Candidate(args.start, args.terms, args.step, 1 if args.sign == "+" else -1)
    Candidate(end, args.terms, args.step, 1 if args.sign == "+" else -1)
    if end < args.start or not 2 <= args.pmin <= args.pmax <= MAX_SIEVE_DEPTH:
        parser.error(f"require end>=start and 2<=pmin<=pmax<={MAX_SIEVE_DEPTH}")
    python_sieve = args.backend == "python" or (args.backend == "auto" and not native_binary("CSSV"))
    if args.command != "check" and python_sieve and args.pmax > MAX_PYTHON_SIEVE_DEPTH:
        parser.error("Python sieve supports pmax<=1e12; select a native CPU/GPU backend for deeper sieving")
    if not 1 <= args.batch_size <= 65536 or not 1 <= args.max_bits <= 16_000_000:
        parser.error("batch-size must be 1..65536; max-bits must be 1..16000000")
    if args.resume and not args.state:
        parser.error("--resume requires --state")
    if args.state and args.output and args.state.resolve() == args.output.resolve():
        parser.error("state and output must be different paths")
    config = {k: getattr(args, k) for k in ("command", "step", "sign", "start", "terms", "pmin", "pmax", "max_bits", "backend", "print_value")}
    config["end"] = end
    config["output"] = str(args.output.resolve()) if args.output else None
    state = {"config": config, "next": args.start, "output_bytes": 0, "output_sha256": hashlib.sha256(b"").hexdigest()}
    with acquire_locks((args.state, args.output)):
        return run_range(args, end, config, state)


def run_range(args, end, config, state):
    if args.resume:
        state = load_state(args.state)
        if state["config"] != config or not args.start <= state["next"] <= end+1:
            raise ValueError("resume configuration/progress mismatch")
    elif args.state and args.state.exists():
        raise ValueError("state exists; use --resume or a new path")
    output_hasher = hashlib.sha256()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not args.output.exists() and state["output_bytes"]:
                raise ValueError("output file missing")
            if not args.output.exists():
                args.output.touch(exist_ok=False)
            with args.output.open("r+b") as f:
                remaining = state["output_bytes"]
                while remaining:
                    block = f.read(min(1 << 20, remaining))
                    if not block:
                        break
                    output_hasher.update(block)
                    remaining -= len(block)
                if remaining or output_hasher.hexdigest() != state["output_sha256"]:
                    raise ValueError("output prefix length/checksum mismatch")
                f.truncate(state["output_bytes"])
        else:
            args.output.touch(exist_ok=False)
    if args.state:
        atomic_state(args.state, state)
    started = time.monotonic()
    while state["next"] <= end:
        first = state["next"]
        last = min(end, first+args.batch_size-1)
        signs = (-1, 1) if args.sign == "both" else (1 if args.sign == "+" else -1,)
        candidates = [Candidate(x, args.terms, args.step, sign) for x in range(first, last+1) for sign in signs]
        factors, engine = sieve(candidates, args.pmin, args.pmax, args.backend) if args.command != "check" else ([0]*len(candidates), "disabled")
        pending_lines = []
        for c, factor in zip(candidates, factors):
            record = {**asdict(c), "expression": c.expression()}
            if factor:
                record.update(result="COMPOSITE", factor=factor, test="proper-factor", sieve_depth=args.pmax)
            elif args.command == "sieve":
                record.update(result="NOT_PRIME" if c.capped_value(2)<2 else "SURVIVOR", sieve_depth=args.pmax)
            else:
                ck = str(args.state)+f".x{c.start}.csck" if args.state else None
                record, n = checked_result(c, args.backend, args.max_bits, ck)
                if args.print_value:
                    record["value"] = n.digits(10)
            pending_lines.append(encode(record)+b"\n")
            if len(pending_lines) < len(signs):
                continue
            # A pair is one commit unit: interrupted output tails are truncated
            # on resume, so next=x+1 can never skip an uncommitted second sign.
            line = b"".join(pending_lines)
            if args.output:
                with args.output.open("ab") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                state["output_bytes"] += len(line)
                # Incremental digest avoids rereading the full result log per candidate.
                output_hasher.update(line)
                state["output_sha256"] = output_hasher.hexdigest()
            print(line.decode("utf-8"), end="", flush=True)
            state["next"] = c.start+1
            if args.state:
                atomic_state(args.state, state)
                # Commit the durable range result before discarding its large
                # arithmetic checkpoint. Unfinished candidates keep theirs.
                finished_ck = Path(str(args.state)+f".x{c.start}.csck")
                if finished_ck.exists():
                    finished_ck.unlink()
            pending_lines.clear()
        print(f"progress: {last-args.start+1}/{end-args.start+1}; elapsed={time.monotonic()-started:.3f}s; sieve={engine}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; resume repeats the unfinished candidate.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
