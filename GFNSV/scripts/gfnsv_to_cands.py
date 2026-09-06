"""Offline GFNSV snapshot -> GFPS cand_<base>.txt, with no network calls.

Use one writer per output directory. Every target is preflighted before any
write; identical files can be skipped when resuming an interrupted conversion.
Compact v4 input is self-contained; a .sieve-state is read only for a legacy
bare view. Input files are never changed. --dry-run creates nothing, even
temporary files. Publication uses same-directory temporary files and atomic installation.
On POSIX filesystems without hard links, safe no-clobber installation relies on
the documented single-writer rule plus a final nonexistence check before rename.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left
import errno
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

# A dry run must not create an import cache beside either standalone script.
sys.dont_write_bytecode = True
from gfnsv_queue import _fsync_dir, read_completed_snapshot

MANIFEST_NAME = ".gfn-tasks.json"


def _status(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_link(status) -> bool:
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _directory_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        status = _status(part)
        if status is not None and (_is_link(status) or not stat.S_ISDIR(status.st_mode)):
            raise RuntimeError(f"output directory or ancestor is not an ordinary directory: {part}")
    return path


def _existing_bytes(path: Path, limit: int) -> bytes | None:
    status = _status(path)
    if status is None:
        return None
    if _is_link(status) or not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"existing target is a directory, symlink, or nonregular file: {path}")
    if status.st_size > limit:
        raise RuntimeError(f"existing target is too large or has conflicting contents: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"existing target changed while being read: {path}")
    return data


def _manifest(n: int) -> dict:
    return {"format": "GFNSV_TASK_DIRECTORY", "version": 1, "n": n, "exponent": str(1 << n)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest field")
        result[key] = value
    return result


def _install_new(path: Path, contents: bytes) -> None:
    """Never expose a half-written destination or replace a known target."""
    fd, temporary = tempfile.mkstemp(prefix=".gfn-install-", suffix=".tmp", dir=path.parent)
    tmp = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if _status(path) is not None:
            raise RuntimeError(f"target appeared after preflight; refusing to overwrite: {path}")
        if os.name == "nt":
            # Unlike POSIX rename, Windows rename fails if the target exists.
            os.rename(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except OSError as exc:
                unsupported = {errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}
                if exc.errno not in unsupported:
                    raise
                if _status(path) is not None:
                    raise RuntimeError(f"target appeared after preflight: {path}") from exc
                # Supports FAT-like filesystems without depending on hard links.
                # No other writer may create files in this directory meanwhile.
                os.rename(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def convert(input_path: Path, out_dir: Path, *, expected_n: int = 16, dry_run: bool = False) -> dict:
    info, bases = read_completed_snapshot(Path(input_path), expected_n)
    out_dir = _directory_path(Path(out_dir))
    manifest_path = out_dir / MANIFEST_NAME
    expected_manifest = _manifest(info["n"])
    existing_manifest = _existing_bytes(manifest_path, 8192)
    if existing_manifest is not None:
        try:
            manifest = json.loads(existing_manifest, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(f"invalid task-directory manifest: {manifest_path}") from exc
        if (manifest != expected_manifest or not isinstance(manifest, dict)
                or type(manifest.get("n")) is not int or type(manifest.get("version")) is not int):
            raise RuntimeError(f"task directory is bound to a different n or manifest format; choose a new --out-dir: {out_dir}")
    elif out_dir.exists() and any(p.name.startswith("cand_") and p.name.endswith(".txt") for p in out_dir.iterdir()):
        raise RuntimeError(f"existing cand files have no {MANIFEST_NAME}; choose a new empty task directory")

    # A deeper re-sieve can remove bases that were exported by an earlier run.
    # Leaving those old tasks beside the new survivors would silently undo that
    # extra sieving. The manifest binds n, while this check binds this directory's
    # actual task set to a subset of the current authoritative snapshot.
    if existing_manifest is not None:
        for entry in out_dir.iterdir():
            if not (entry.name.startswith("cand_") and entry.name.endswith(".txt")):
                continue
            match = re.fullmatch(r"cand_([1-9][0-9]{0,19})\.txt", entry.name)
            base = int(match[1]) if match else -1
            index = bisect_left(bases, base)
            if index == len(bases) or bases[index] != base:
                raise RuntimeError(f"existing cand is absent from the current survivor snapshot: {entry.name}; choose a new empty task directory (no files were removed)")

    missing = array("Q")
    skipped = 0
    for base in bases:
        target = out_dir / f"cand_{base}.txt"
        wanted = f"{base}\n".encode("ascii")
        existing = _existing_bytes(target, len(wanted))
        if existing is None:
            missing.append(base)
        elif existing != wanted:
            raise RuntimeError(f"conflicting existing candidate contents; refusing to overwrite: {target}")
        else:
            skipped += 1

    report = {"input": str(input_path), "out_dir": str(out_dir), "n": info["n"],
              "exponent": str(1 << info["n"]), "sieve_sha256": info["sha256"],
              "sieved_to": min(info["pmax"], info["pmin"] - 1), "candidates": len(bases),
              "created": 0 if dry_run else len(missing), "would_create": len(missing),
              "skipped_identical": skipped, "dry_run": dry_run}
    if dry_run:
        return report
    out_dir.mkdir(parents=True, exist_ok=True)
    if existing_manifest is None:
        contents = (json.dumps(expected_manifest, sort_keys=True, indent=2) + "\n").encode("ascii")
        _install_new(manifest_path, contents)
    for base in missing:
        _install_new(out_dir / f"cand_{base}.txt", f"{base}\n".encode("ascii"))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=16, help="required GFN n (exponent is 2^n), default 16")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing anything")
    args = parser.parse_args(argv)
    try:
        result = convert(args.input, args.out_dir, expected_n=args.expected_n, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
