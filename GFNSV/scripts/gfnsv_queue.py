"""Read self-contained GFNSV compact v4 and legacy CUDA v3 snapshots.

Compact v4 authenticates metadata and its ordered survivor list in one file;
no sieve sidecar is needed. Legacy bare GFN/ABC views still require their v3
.sieve-state for migration. Neither authenticated format should be consumed in
place by a checker's pop_candidate(); export_completed_queue preserves a backup.
"""
from __future__ import annotations

import hashlib
from array import array
import os
from pathlib import Path
import re
import shutil
import tempfile

_U64_MAX = (1 << 64) - 1
_MAX_SLOTS = 64 * 1024 * 1024
_HEADER = re.compile(
    r"(#GFNSV_STATE|#GFNSV_COMPLETE) version=3 engine=cuda "
    r"n=(\d+) bmin=(\d+) bmax=(\d+) start_pmin=(\d+) pmin=(\d+) "
    r"pmax=(\d+) next_k=(\d+) slots=(\d+) alive_count=(\d+) format=(base|expr)"
)
_HEADER4 = re.compile(
    r"(#GFNSV_STATE|#GFNSV_COMPLETE) version=4 engine=cuda "
    r"n=(\d+) bmin=(\d+) bmax=(\d+) start_pmin=(\d+) pmin=(\d+) "
    r"pmax=(\d+) next_k=(\d+) slots=(\d+) alive_count=(\d+) format=(G|A|base|expr)"
)
_ABC_HEADER = re.compile(r"ABC \$a\^([0-9]+)\+1 // Sieved to ([0-9]+)")
_GFN_HEADER = re.compile(r"GFN n=([0-9]+) // Sieved to ([0-9]+)")


def _number(text: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise RuntimeError("noncanonical GFNSV checkpoint integer")
    value = int(text)
    if value > _U64_MAX:
        raise RuntimeError("overflowing GFNSV checkpoint integer")
    return value


def _line(stream) -> str | None:
    raw = stream.readline(1027)
    if not raw:
        return None
    raw = raw.removesuffix(b"\n").removesuffix(b"\r")
    if len(raw) > 1024 or any(c < 32 or c > 126 for c in raw):
        raise RuntimeError("invalid GFNSV checkpoint line")
    return raw.decode("ascii")


def _inspect_v3_stream(path: Path, stream, expected_n: int, visit_candidate=None) -> dict:
    """Validate one open snapshot; visitors must not publish before this returns."""
    header = _line(stream)
    match = _HEADER.fullmatch(header or "")
    if not match:
        raise RuntimeError(f"{path}: requires GFNSV CUDA state version 3; legacy CPU states must be finished with the archived CPU sieve")
    fields = match.groups()
    names = ("n", "bmin", "bmax", "start_pmin", "pmin", "pmax", "next_k", "slots", "alive_count")
    info = dict(zip(names, map(_number, fields[1:10])))
    info["format"] = fields[10]
    info["complete"] = fields[0] == "#GFNSV_COMPLETE"
    n, lo, hi = info["n"], info["bmin"], info["bmax"]
    if not 1 <= n <= 20 or n != expected_n:
        raise RuntimeError(f"GFNSV checkpoint n={n} does not match requested n={expected_n}")
    if lo < 2 or hi < lo or not 3 <= info["start_pmin"] <= info["pmax"] <= (1 << 62) - 1:
        raise RuntimeError("invalid GFNSV checkpoint range")
    first_even = lo + (lo & 1)
    slots = max(0, (hi - first_even) // 2 + 1)
    q = 1 << (n + 1)
    first_k = (info["start_pmin"] - 2) // q + 1
    last_k = (info["pmax"] - 1) // q
    complete = slots == 0 or info["next_k"] > last_k
    if (slots != info["slots"] or slots > _MAX_SLOTS or info["alive_count"] > slots
            or not first_k <= info["next_k"] <= last_k + 1
            or info["pmin"] != info["next_k"] * q + 1 or complete != info["complete"]):
        raise RuntimeError("inconsistent GFNSV checkpoint metadata")
    size = os.fstat(stream.fileno()).st_size
    if not slots * 2 + len(header) + 80 <= size <= slots * 96 + 4096:
        raise RuntimeError("GFNSV checkpoint length disagrees with slot count")
    digest = hashlib.sha256((header + "\n").encode("ascii"))
    survivors = 0
    for index in range(slots):
        record = _line(stream)
        b = first_even + 2 * index
        expected = str(b) + (f"^{1 << n}+1" if info["format"] == "expr" else "")
        is_survivor = False
        if record is not None and record.startswith("#FACTOR "):
            parts = record.split(" ")
            if len(parts) != 3 or _number(parts[1]) != b:
                raise RuntimeError("GFNSV factor records are missing, duplicated or reordered")
            p = _number(parts[2])
            if not (info["start_pmin"] <= p <= info["pmax"] and p < info["pmin"] and p % q == 1):
                raise RuntimeError("GFNSV factor is outside completed sieve prefix")
        elif record == expected:
            survivors += 1
            is_survivor = True
        else:
            raise RuntimeError("GFNSV candidate records are truncated, duplicated or reordered")
        digest.update((record + "\n").encode("ascii"))
        if visit_candidate is not None:
            visit_candidate(b, is_survivor)
    checksum = digest.hexdigest()
    footer = f"#GFNSV_END count={slots} survivors={info['alive_count']} sha256={checksum}"
    if _line(stream) != footer or survivors != info["alive_count"] or _line(stream) is not None:
        raise RuntimeError("GFNSV checkpoint count/SHA-256/footer mismatch")
    info["sha256"] = checksum
    return info


def _inspect_v3_file(path: Path, expected_n: int) -> dict:
    # Historical private name retained for callers/tests. Consumption backups
    # may now be either self-contained v3 or v4, never recursively resolved views.
    with path.open("rb") as stream:
        return _inspect_snapshot_stream(path, stream, expected_n)


def _inspect_v4_stream(path: Path, stream, expected_n: int, visit_candidate=None) -> dict:
    """Validate compact survivors in O(alive_count), never O(original slots)."""
    prefix = None
    header = _line(stream)
    if header is not None and header.startswith(("GFN", "ABC")):
        prefix = header
        header = _line(stream)
    match = _HEADER4.fullmatch(header or "")
    if not match:
        raise RuntimeError(f"{path}: invalid self-contained GFNSV compact v4 metadata; not falling back to any sidecar")
    fields = match.groups()
    names = ("n", "bmin", "bmax", "start_pmin", "pmin", "pmax", "next_k", "slots", "alive_count")
    info = dict(zip(names, map(_number, fields[1:10])))
    info.update(format=fields[10], complete=fields[0] == "#GFNSV_COMPLETE", version=4)
    n, lo, hi = info["n"], info["bmin"], info["bmax"]
    if not 1 <= n <= 20 or n != expected_n:
        raise RuntimeError(f"GFNSV checkpoint n={n} does not match requested n={expected_n}")
    if lo < 2 or hi < lo or not 3 <= info["start_pmin"] <= info["pmax"] <= (1 << 62) - 1:
        raise RuntimeError("invalid GFNSV checkpoint range")
    first_even = lo + (lo & 1)
    slots = max(0, (hi - first_even) // 2 + 1)
    q = 1 << (n + 1)
    first_k = (info["start_pmin"] - 2) // q + 1
    last_k = (info["pmax"] - 1) // q
    complete = slots == 0 or info["next_k"] > last_k
    if (slots != info["slots"] or slots > _MAX_SLOTS or info["alive_count"] > slots
            or not first_k <= info["next_k"] <= last_k + 1
            or info["pmin"] != info["next_k"] * q + 1 or complete != info["complete"]):
        raise RuntimeError("inconsistent GFNSV compact checkpoint metadata")
    boundary = min(info["pmax"], info["pmin"] - 1)
    if info["format"] == "G":
        expected_prefix = f"GFN n={n} // format=4 Sieved to {boundary}"
    elif info["format"] == "A":
        expected_prefix = f"ABC $a^{1 << n}+1 // format=4 Sieved to {boundary}"
    else:
        expected_prefix = None
    if prefix != expected_prefix:
        raise RuntimeError("GFNSV compact header/format/n/sieve boundary mismatch")
    initial = (prefix + "\n" if prefix is not None else "") + header + "\n"
    count = info["alive_count"]
    size = os.fstat(stream.fileno()).st_size
    minimum_footer = len(f"#GFNSV_END count={count} sha256=") + 64 + 1
    if not len(initial) + count * 2 + minimum_footer <= size <= count * 96 + 4096:
        raise RuntimeError("GFNSV compact checkpoint length disagrees with survivor count")
    digest = hashlib.sha256(initial.encode("ascii"))
    previous = 0
    expression_suffix = f"^{1 << n}+1"
    for _ in range(count):
        record = _line(stream)
        if record is None:
            raise RuntimeError("truncated GFNSV compact survivor list")
        token = record
        if info["format"] == "expr":
            if not token.endswith(expression_suffix):
                raise RuntimeError("GFNSV compact expression has wrong exponent or suffix")
            token = token[:-len(expression_suffix)]
        base = _number(token)
        if base < first_even or base > hi or base & 1 or base <= previous:
            raise RuntimeError("GFNSV compact survivors are out of range, odd, duplicated or reordered")
        previous = base
        digest.update((record + "\n").encode("ascii"))
        if visit_candidate is not None:
            visit_candidate(base, True)
    checksum = digest.hexdigest()
    footer = f"#GFNSV_END count={count} sha256={checksum}"
    if _line(stream) != footer or _line(stream) is not None:
        raise RuntimeError("GFNSV compact checkpoint count/SHA-256/footer mismatch")
    info["sha256"] = checksum
    return info


def _inspect_snapshot_stream(path: Path, stream, expected_n: int, visit_candidate=None,
                             *, allow_legacy_view: bool = False) -> dict:
    """Dispatch on the same open handle; corrupt compact input never retries."""
    offset = stream.tell()
    first = _line(stream)
    if first is not None and "// format=" in first:
        stream.seek(offset)
        return _inspect_v4_stream(path, stream, expected_n, visit_candidate)
    if first is not None and first.startswith(("GFN", "ABC")):
        second = _line(stream)
        stream.seek(offset)
        if second is not None and not re.fullmatch(r"[0-9]+", second):
            return _inspect_v4_stream(path, stream, expected_n, visit_candidate)
        if not allow_legacy_view:
            raise RuntimeError(f"{path}: requires GFNSV CUDA state version 3 or 4; a bare candidate view is not a snapshot")
        return _inspect_abc(path, stream, expected_n, visit_candidate)
    stream.seek(offset)
    if first is not None and re.match(r"#GFNSV_(?:STATE|COMPLETE) version=4(?: |$)", first):
        return _inspect_v4_stream(path, stream, expected_n, visit_candidate)
    return _inspect_v3_stream(path, stream, expected_n, visit_candidate)


def _abc_bases(stream):
    previous = 0
    while (record := _line(stream)) is not None:
        base = _number(record)
        if base < 2 or base & 1 or base <= previous:
            raise RuntimeError("GFNSV ABC bases must be even, unique and strictly increasing")
        previous = base
        yield base


def _inspect_abc(path: Path, stream, expected_n: int, visit_candidate=None) -> dict:
    """Inspect either export header against one authoritative open sidecar."""
    kind = "GFN/ABC"
    try:
        header = _line(stream) or ""
        match = _GFN_HEADER.fullmatch(header)
        if match:
            kind = "GFN"
            declared_n, sieved_to = map(_number, match.groups())
            if not 1 <= expected_n <= 20 or declared_n != expected_n:
                raise RuntimeError(f"GFNSV GFN n={declared_n} does not match requested n={expected_n}")
        else:
            match = _ABC_HEADER.fullmatch(header)
            if match:
                kind = "ABC"
                exponent, sieved_to = map(_number, match.groups())
                if not 1 <= expected_n <= 20 or exponent != 1 << expected_n:
                    raise RuntimeError(f"GFNSV ABC exponent={exponent} does not match requested n={expected_n}")
        if not match:
            raise RuntimeError("invalid GFNSV GFN/ABC header")
        backup = path.with_name(path.name + ".sieve-state")
        bases = _abc_bases(stream)
        pending = next(bases, None)

        def check_candidate(base: int, is_survivor: bool) -> None:
            nonlocal pending
            if pending is not None and pending < base:
                raise RuntimeError("GFNSV ABC base is outside the checkpoint range")
            if pending == base:
                pending = next(bases, None)
            elif is_survivor:
                raise RuntimeError(f"GFNSV ABC is missing surviving base {base}")
            if visit_candidate is not None:
                visit_candidate(base, is_survivor)

        with backup.open("rb") as checkpoint:
            info = _inspect_v3_stream(backup, checkpoint, expected_n, check_candidate)
        if pending is not None:
            raise RuntimeError("GFNSV ABC base is outside the checkpoint range")
        boundary = info["pmax"] if info["complete"] else min(info["pmax"], info["pmin"] - 1)
        if sieved_to > boundary:
            raise RuntimeError("GFNSV ABC sieve boundary is ahead of its checkpoint")
        return info
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot trust GFNSV {kind} {path}: {exc}; keep its .sieve-state and run "
            "GFNSV --resume to rebuild the candidate output before using this queue"
        ) from exc


def inspect_sieve_file(path: Path, expected_n: int) -> dict | None:
    """Return validated metadata, or None for a plain pre-sieved task queue.

    A plain queue stays plain even when an older .sieve-state exists beside it:
    recovering that snapshot would reintroduce already consumed PRP tasks.
    Neither inspecting nor exporting needs a CUDA binary or a GPU.
    """
    with path.open("rb") as stream:
        first = stream.readline(1027)
        stream.seek(0)
        if b"// format=" in first or first.removeprefix(b"\xef\xbb\xbf").lstrip().startswith((b"ABC", b"GFN", b"#GFNSV_STATE", b"#GFNSV_COMPLETE")):
            return _inspect_snapshot_stream(path, stream, expected_n, allow_legacy_view=True)
        for raw in stream:
            if raw.removeprefix(b"\xef\xbb\xbf").lstrip().startswith((b"#GFNSV_", b"#FACTOR ", b"ABC", b"GFN")):
                raise RuntimeError(f"misplaced or unsupported sieve state in {path}; use the original sieve to finish/export it")
        return None


def read_completed_snapshot(path: Path, expected_n: int) -> tuple[dict, array]:
    """Return validated metadata and survivors without altering any source file.

    Self-contained input is opened once; only a legacy bare export additionally
    opens its authoritative v3 sidecar once.
    Survivors are collected during that same SHA-256-validated read, never by
    reopening a filename after validation. Memory costs eight bytes per
    survivor plus small parsing overhead. Plain task queues are not accepted.
    """
    if not isinstance(expected_n, int) or isinstance(expected_n, bool) or not 1 <= expected_n <= 20:
        raise RuntimeError("expected GFNSV n must be an integer in 1..20")
    path = Path(path)
    survivors = array("Q")

    def collect(base: int, is_survivor: bool) -> None:
        if is_survivor:
            survivors.append(base)

    with path.open("rb") as stream:
        first = stream.readline(1027)
        stream.seek(0)
        if b"// format=" in first or first.removeprefix(b"\xef\xbb\xbf").lstrip().startswith((b"ABC", b"GFN", b"#GFNSV_STATE", b"#GFNSV_COMPLETE")):
            info = _inspect_snapshot_stream(path, stream, expected_n, collect, allow_legacy_view=True)
        else:
            raise RuntimeError("offline conversion requires a complete authenticated compact v4 or CUDA v3 snapshot (or legacy view plus v3 sidecar); plain candidate text is not trusted")
    if not info["complete"]:
        raise RuntimeError("GFNSV snapshot is partial; finish sieving before task conversion")
    if len(survivors) != info["alive_count"]:
        raise RuntimeError("GFNSV snapshot survivor count changed during validation")
    return info, survivors


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_completed_queue(path: Path, expected_n: int) -> dict:
    """Keep an intact .sieve-state before atomically publishing a plain queue."""
    info = inspect_sieve_file(path, expected_n)
    if info is None or not info["complete"]:
        raise RuntimeError("GFNSV did not produce a validated completed checkpoint")
    backup = path.with_name(path.name + ".sieve-state")
    if backup.exists():
        previous = _inspect_v3_file(backup, expected_n)
        if previous["sha256"] != info["sha256"]:
            raise RuntimeError(f"saved sieve snapshot differs: {backup}; choose a new --cand path")
    else:
        fd, temporary = tempfile.mkstemp(prefix=backup.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as destination, path.open("rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            copied = _inspect_v3_file(Path(temporary), expected_n)
            if copied["sha256"] != info["sha256"]:
                raise RuntimeError("GFNSV checkpoint changed during queue export")
            os.replace(temporary, backup)
            _fsync_dir(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as destination, backup.open("rb") as source:
            def emit_candidate(base: int, is_survivor: bool) -> None:
                if is_survivor:
                    destination.write(str(base) + "\n")

            exported = _inspect_snapshot_stream(backup, source, expected_n, emit_candidate)
            if exported["sha256"] != info["sha256"]:
                raise RuntimeError("GFNSV checkpoint changed during queue export")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return info


def load_result_hits(path: Path) -> list[int]:
    """Load the existing count/base result file; malformed files fail closed."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read saved PRP results {path}: {exc}") from exc
    if not lines:
        raise RuntimeError(f"saved PRP results are empty: {path}")
    try:
        count = _number(lines[0])
        hits = [_number(line) for line in lines[1:]]
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"malformed saved PRP results: {path}") from exc
    if count != len(hits) or len(set(hits)) != len(hits) or any(base < 2 for base in hits):
        raise RuntimeError(f"saved PRP result count/bases are invalid: {path}")
    return hits


def write_result_hits(path: Path, hits: list[int]) -> None:
    """Commit unique hits before removing a completed candidate from its queue."""
    unique = list(dict.fromkeys(hits))
    if any(not isinstance(base, int) or not 2 <= base <= _U64_MAX for base in unique):
        raise RuntimeError("invalid PRP base in result checkpoint")
    contents = "\n".join([str(len(unique)), *map(str, unique)]) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as destination:
            destination.write(contents)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
