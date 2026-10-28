"""Validate a CSSV v2 survivor file and split it into one-candidate CSPS tasks.

Usage: python cssv_to_tasks.py -i candidates.txt -o new_task_directory
The destination must not exist. No network access, arithmetic engine, or GPU
is used. A SHA-256 footer is an integrity check, not a primality/sieve proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from csps_tasks import MAX_U64, MAX_TERMS, make_task, parse_task, task_filename

MAX_SIEVE_DEPTH = (1 << 62) - 1
MAX_START_COUNT = 2_000_000
MAX_CANDIDATE_COUNT = MAX_START_COUNT * 2
_UINT = rb"(?:0|[1-9][0-9]{0,19})"
_HEADER = re.compile(
    rb"# CSSV v2 start=(" + _UINT + rb") end=(" + _UINT
    + rb") terms=(" + _UINT + rb") step=([12]) signs=(minus|plus|both) pmin=("
    + _UINT + rb") sieved_to=(" + _UINT + rb") count=(" + _UINT + rb")\n"
)
_ROW = re.compile(rb"([1-9][0-9]{0,19}) ([+-])1\n")
_FOOTER = re.compile(rb"#SHA256 ([0-9a-f]{64})\n")


def parse_candidates(path: str | Path) -> dict:
    """Strictly validate the entire file before returning its in-memory snapshot.

    ``source_sha256`` hashes the complete original file including the footer;
    ``payload_sha256`` hashes the header and rows, the checksum in the footer.
    ``rows`` is a list of ``(start, sign)`` pairs in their original order.
    """
    path = Path(path)
    all_hash, payload_hash = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.readline(513)
        match = _HEADER.fullmatch(header)
        if match is None:
            raise ValueError("invalid CSSV v2 header (ASCII decimal fields, LF, no BOM required)")
        start, end, terms, step = map(int, match.group(1, 2, 3, 4))
        signs = match.group(5).decode("ascii")
        pmin, depth, count = map(int, match.group(6, 7, 8))
        if not 1 <= start <= end < MAX_U64:
            raise ValueError("invalid inclusive start/end range")
        if end - start + 1 > MAX_START_COUNT:
            raise ValueError("start/end range exceeds the native two-million-start limit")
        if not 1 <= terms <= MAX_TERMS or end + step * (terms - 1) >= MAX_U64:
            raise ValueError("terms or last-base uint64 boundary exceeded")
        if not 2 <= pmin <= MAX_SIEVE_DEPTH or not pmin - 1 <= depth <= MAX_SIEVE_DEPTH:
            raise ValueError("expected 2 <= pmin <= 2^62-1 and pmin-1 <= sieved_to <= 2^62-1")
        if count > MAX_CANDIDATE_COUNT or count > (end - start + 1) * (2 if signs == "both" else 1):
            raise ValueError("candidate count exceeds declared range")
        # A valid row is at least 5 bytes. Reject absurd count claims before
        # allocating their rows; large legitimate inputs remain stream-read.
        if count > os.fstat(stream.fileno()).st_size // 5:
            raise ValueError("candidate count exceeds source file size")
        all_hash.update(header)
        payload_hash.update(header)
        rows = []
        previous = None
        sign_counts = Counter()
        for index in range(count):
            row = stream.readline(65)
            found = _ROW.fullmatch(row)
            if found is None:
                raise ValueError(f"invalid or missing candidate row {index + 1}")
            x = int(found.group(1))
            sign = -1 if found.group(2) == b"-" else 1
            key = (x, sign)
            if not start <= x <= end:
                raise ValueError(f"candidate row {index + 1} lies outside the header range")
            if signs != "both" and sign != (1 if signs == "plus" else -1):
                raise ValueError(f"candidate row {index + 1} has an undeclared sign")
            if previous is not None and key <= previous:
                raise ValueError(f"candidate row {index + 1} is duplicate or not in x/sign order")
            if x == 1 and terms == 1 and sign == -1:
                raise ValueError("candidate is less than 2")
            previous = key
            rows.append(key)
            sign_counts["plus" if sign == 1 else "minus"] += 1
            payload_hash.update(row)
            all_hash.update(row)
        footer = stream.readline(81)
        checksum = _FOOTER.fullmatch(footer)
        if checksum is None:
            raise ValueError("invalid checksum footer or candidate count mismatch")
        if checksum.group(1).decode("ascii") != payload_hash.hexdigest():
            raise ValueError("CSSV payload SHA-256 mismatch")
        if stream.read(1):
            raise ValueError("unexpected data after checksum footer")
        all_hash.update(footer)
    return {
        "format": "CSSV", "version": 2, "start": start, "end": end,
        "terms": terms, "step": step, "signs": signs, "pmin": pmin,
        "sieved_to": depth, "count": count, "rows": rows,
        "counts": {"minus": sign_counts["minus"], "plus": sign_counts["plus"]},
        "source_sha256": all_hash.hexdigest(), "payload_sha256": payload_hash.hexdigest(),
    }


def convert(source: str | Path, output: str | Path) -> dict:
    """Create tasks in a private sibling stage, then rename on full success."""
    source, output = Path(source), Path(output)
    # lexists also refuses a broken destination symlink.
    if os.path.lexists(output):
        raise FileExistsError(f"destination already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or output.name in ("", ".", ".."):
        raise ValueError("destination parent must be an existing directory")
    output = parent / output.name
    snapshot = parse_candidates(source)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    stage_identity = stage.stat()
    published = False
    try:
        records = []
        for x, sign in snapshot["rows"]:
            params = {"start": x, "terms": snapshot["terms"], "step": snapshot["step"], "sign": sign}
            name = task_filename(**params)
            data = make_task(**params).encode("utf-8")
            if parse_task(data, filename=name) != params:
                raise AssertionError("internal task round-trip failure")
            with (stage / name).open("xb") as stream:
                stream.write(data)
            records.append({"filename": name, "sha256": hashlib.sha256(data).hexdigest(),
                            "x": x, "sign": sign})
        # Reopen every completed task before publishing the directory. This
        # catches write/truncation defects without requiring GPU arithmetic.
        for item in records:
            with (stage / item["filename"]).open("rb") as stream:
                data = stream.read(257)
            params = parse_task(data, filename=item["filename"])
            if hashlib.sha256(data).hexdigest() != item["sha256"] or params != {
                "start": item["x"], "terms": snapshot["terms"],
                "step": snapshot["step"], "sign": item["sign"],
            }:
                raise ValueError(f"task read-back verification failed: {item['filename']}")
        manifest = {
            "format": "CSPS task manifest", "version": 1,
            "atomic_publish": True, "power_loss_durable": False,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_file": source.name, "source_sha256": snapshot["source_sha256"],
            "source_payload_sha256": snapshot["payload_sha256"],
            "sieve": {key: snapshot[key] for key in (
                "start", "end", "terms", "step", "signs", "pmin", "sieved_to")},
            "count": snapshot["count"], "counts": snapshot["counts"], "tasks": records,
        }
        with (stage / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if os.path.lexists(output):
            raise FileExistsError(f"destination appeared during conversion: {output}")
        # On Windows rename refuses any existing directory. On POSIX use
        # renameat2(RENAME_NOREPLACE) rather than replacing an empty directory.
        _rename_no_replace(stage, output)
        published = True
        return manifest
    finally:
        if not published and stage.exists():
            current = stage.lstat()
            if (current.st_dev, current.st_ino) != (stage_identity.st_dev, stage_identity.st_ino) or stage.is_symlink():
                raise RuntimeError(f"refusing cleanup: staging directory identity changed: {stage}")
            shutil.rmtree(stage)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError(errno.ENOSYS, "atomic no-overwrite directory rename requires renameat2")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path, help="checksummed CSSV v2 candidate file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="new task directory (must not exist)")
    args = parser.parse_args(argv)
    try:
        manifest = convert(args.input, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps({"tasks": manifest["count"], **manifest["counts"],
                      "sieved_to": manifest["sieve"]["sieved_to"],
                      "source_sha256": manifest["source_sha256"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
