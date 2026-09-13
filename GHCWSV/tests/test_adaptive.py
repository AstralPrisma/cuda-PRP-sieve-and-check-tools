"""Bounded GPU scheduling regression; no GPU work without explicit --gpu."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ghcw_to_cands import read_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, help="Optional original 1.1 executable")
    parser.add_argument("--gpu", action="store_true", help="Explicitly permit bounded GPU tests")
    args = parser.parse_args()
    if not args.gpu:
        parser.error("GPU work requires --gpu; stop competing GPU work first")
    binary = args.binary.resolve()
    baseline = args.baseline.resolve() if args.baseline else None
    reports = []
    with tempfile.TemporaryDirectory(prefix="ghcwsv-adaptive-") as directory:
        directory = pathlib.Path(directory)

        def run(name, exe, options):
            output = directory / (name + ".txt")
            command = [str(exe), *map(str, options), "-o", str(output),
                       "--prime-threads", "2", "--prime-generator", "segmented"]
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True, timeout=180)
            text = result.stdout.decode("utf-8")
            if result.returncode:
                raise AssertionError((name, result.returncode, text[-2000:], result.stderr.decode("utf-8")))
            data = output.read_bytes()
            snapshot = read_snapshot(output)
            reports.append({"name": name, "wall_s": time.monotonic() - start,
                            "survivors": snapshot.count, "p": snapshot.p,
                            "rebuilds": text.count("workset-rebuild:"),
                            "sha256": hashlib.sha256(data).hexdigest()})
            return output, data, text

        common = ["-b", 7, "-n", 100001, "-N", 500000, "--sign", "both", "-P", 2000000]
        _, adaptive, text = run("adaptive", binary, common)
        assert "workset-rebuild:" in text and "recent_primes_per_s=" in text
        assert read_snapshot(directory / "adaptive.txt").count == 25096
        _, fixed, _ = run("fixed", binary, [*common, "--work-batching", "fixed"])
        assert adaptive == fixed
        if baseline:
            _, original, _ = run("original", baseline, common)
            assert adaptive == original
        # Deliberately end inside the large producer batch and resume. This is
        # deterministic prefix coverage; actual signal tests are separately recorded.
        partial, _, _ = run("prefix", binary, [*common[:-1], 500000])
        _, resumed, _ = run("resumed", binary, ["-i", partial, "-P", 2000000])
        assert resumed == adaptive
    print(json.dumps({"result": "PASS", "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
