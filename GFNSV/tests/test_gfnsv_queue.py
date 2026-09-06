"""Deterministic CPU-only compatibility tests; no GFNSV executable is needed."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gfnsv_queue as queue


def v3_state(*, n=2, bmin=2, bmax=12, start_pmin=3, pmax=100,
             next_k=None, factors=None, fmt="base"):
    """Generate canonical v3 data independently of the production reader."""
    q = 1 << (n + 1)
    if next_k is None:
        next_k = (pmax - 1) // q + 1
    if factors is None:
        factors = {2: 17, 8: 17, 10: 73} if n == 2 else {}
    bases = list(range(bmin + (bmin & 1), bmax + 1, 2))
    factors = {base: factor for base, factor in factors.items() if base in bases}
    for base, factor in factors.items():
        assert start_pmin <= factor <= pmax and factor < next_k * q + 1
        assert factor % q == 1 and pow(base, 1 << n, factor) == factor - 1
    alive = [base for base in bases if base not in factors]
    complete = not bases or next_k > (pmax - 1) // q
    tag = "#GFNSV_COMPLETE" if complete else "#GFNSV_STATE"
    header = (
        f"{tag} version=3 engine=cuda n={n} bmin={bmin} bmax={bmax} "
        f"start_pmin={start_pmin} pmin={next_k * q + 1} pmax={pmax} "
        f"next_k={next_k} slots={len(bases)} alive_count={len(alive)} format={fmt}"
    )
    records = [
        f"#FACTOR {base} {factors[base]}" if base in factors else
        str(base) + (f"^{1 << n}+1" if fmt == "expr" else "")
        for base in bases
    ]
    body = "\n".join([header, *records]) + "\n"
    checksum = hashlib.sha256(body.encode("ascii")).hexdigest()
    footer = f"#GFNSV_END count={len(bases)} survivors={len(alive)} sha256={checksum}\n"
    return (body + footer).encode("ascii"), alive


def abc_output(bases, *, n=2, sieved_to=100):
    return (f"ABC $a^{1 << n}+1 // Sieved to {sieved_to}\n"
            + "".join(f"{base}\n" for base in bases)).encode("ascii")


def gfn_output(bases, *, n=2, sieved_to=100):
    return (f"GFN n={n} // Sieved to {sieved_to}\n"
            + "".join(f"{base}\n" for base in bases)).encode("ascii")


class QueueCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gfnsv_queue_test_")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cand.txt"
        self.sidecar = self.path.with_name(self.path.name + ".sieve-state")

    def make_abc(self, *, abc_bases=None, abc_n=None, sieved_to=None, **state_args):
        data, alive = v3_state(**state_args)
        n = state_args.get("n", 2)
        pmax = state_args.get("pmax", 100)
        frontier = state_args.get("next_k", (pmax - 1) // (1 << (n + 1)) + 1)
        if sieved_to is None:
            sieved_to = min(pmax, frontier * (1 << (n + 1)))
        self.sidecar.write_bytes(data)
        self.path.write_bytes(abc_output(alive if abc_bases is None else abc_bases,
                                         n=n if abc_n is None else abc_n,
                                         sieved_to=sieved_to))
        return data, alive

    def assert_unchanged_export_failure(self, expected_n=2):
        original = self.path.read_bytes()
        with self.assertRaises(RuntimeError):
            queue.export_completed_queue(self.path, expected_n)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))

    def test_legacy_base_and_expr_export(self):
        for fmt in ("base", "expr"):
            with self.subTest(fmt=fmt):
                if self.sidecar.exists():
                    self.sidecar.unlink()
                data, alive = v3_state(fmt=fmt)
                self.path.write_bytes(data)
                info = queue.inspect_sieve_file(self.path, 2)
                self.assertTrue(info["complete"])
                self.assertEqual(info["format"], fmt)
                self.assertEqual(queue.export_completed_queue(self.path, 2), info)
                self.assertEqual(self.sidecar.read_bytes(), data)
                self.assertEqual(self.path.read_text(), "".join(f"{b}\n" for b in alive))
                self.assertIsNone(queue.inspect_sieve_file(self.path, 2))

    def test_legacy_partial_not_exported(self):
        self.path.write_bytes(v3_state(next_k=8, factors={2: 17, 8: 17})[0])
        self.assertFalse(queue.inspect_sieve_file(self.path, 2)["complete"])
        self.assert_unchanged_export_failure()
        self.assertFalse(self.sidecar.exists())

    def test_legacy_version_corruption_and_truncation(self):
        data, _ = v3_state()
        variants = [data.replace(b"version=3", b"version=2"), data[:-30],
                    data.replace(b"#FACTOR 2 17", b"#FACTOR 2 41"), data + b"2\n"]
        for variant in variants:
            with self.subTest(variant=variant[-100:]):
                self.path.write_bytes(variant)
                with self.assertRaises(RuntimeError):
                    queue.inspect_sieve_file(self.path, 2)

    def test_legacy_existing_different_backup_rejected(self):
        self.path.write_bytes(v3_state()[0])
        self.sidecar.write_bytes(v3_state(factors={2: 17, 8: 17})[0])
        self.assert_unchanged_export_failure()

    def test_abc_returns_identical_v3_metadata(self):
        self.make_abc()
        self.assertEqual(queue.inspect_sieve_file(self.path, 2),
                         queue.inspect_sieve_file(self.sidecar, 2))

    def test_n16_abc_uses_exponent_65536(self):
        self.make_abc(n=16, pmax=1_000_000)
        self.assertTrue(self.path.read_bytes().startswith(b"ABC $a^65536+1"))
        self.assertEqual(queue.inspect_sieve_file(self.path, 16)["n"], 16)

    def test_complete_abc_export_preserves_authoritative_sidecar(self):
        data, alive = self.make_abc(fmt="expr")
        queue.export_completed_queue(self.path, 2)
        self.assertEqual(self.sidecar.read_bytes(), data)
        self.assertEqual(self.path.read_text(), "".join(f"{b}\n" for b in alive))
        self.assertIsNone(queue.inspect_sieve_file(self.path, 2))

    def test_partial_abc_is_resumable_but_not_exportable(self):
        data, _ = self.make_abc(next_k=8, factors={2: 17, 8: 17})
        self.assertFalse(queue.inspect_sieve_file(self.path, 2)["complete"])
        self.assert_unchanged_export_failure()
        self.assertEqual(self.sidecar.read_bytes(), data)

    def test_stale_abc_survivor_superset_exports_latest_survivors(self):
        data, alive = self.make_abc(abc_bases=[4, 6, 10, 12], sieved_to=64)
        self.assertTrue(queue.inspect_sieve_file(self.path, 2)["complete"])
        queue.export_completed_queue(self.path, 2)
        self.assertEqual(self.sidecar.read_bytes(), data)
        self.assertEqual(self.path.read_text(), "".join(f"{b}\n" for b in alive))
        self.assertNotIn("10", self.path.read_text().splitlines())

    def test_stale_abc_with_no_new_factors_is_allowed(self):
        self.make_abc(sieved_to=64)
        self.assertTrue(queue.inspect_sieve_file(self.path, 2)["complete"])

    def test_missing_latest_survivor_fails_with_resume_guidance(self):
        for bases in ([6, 12], [4, 12], [4, 6], [], [2, 4, 6, 8, 10]):
            with self.subTest(bases=bases):
                self.make_abc(abc_bases=bases)
                with self.assertRaisesRegex(RuntimeError, "missing surviving base .*GFNSV --resume"):
                    queue.inspect_sieve_file(self.path, 2)
                self.assert_unchanged_export_failure()

    def test_abc_without_sidecar_fails_closed(self):
        self.path.write_bytes(abc_output([4, 6, 12]))
        with self.assertRaisesRegex(RuntimeError, "GFNSV --resume"):
            queue.inspect_sieve_file(self.path, 2)

    def test_corrupt_or_plain_sidecar_fails_closed(self):
        data, _ = self.make_abc()
        variants = [data[:-20], data.replace(b"sha256=", b"sha256=0"),
                    data.replace(b"#FACTOR 2 17", b"#FACTOR 2 41"), b"4\n6\n12\n"]
        for variant in variants:
            with self.subTest(variant=variant[-100:]):
                self.sidecar.write_bytes(variant)
                with self.assertRaisesRegex(RuntimeError, "GFNSV --resume"):
                    queue.inspect_sieve_file(self.path, 2)

    def test_abc_sidecar_is_never_resolved_recursively(self):
        data, _ = self.make_abc()
        self.sidecar.with_name(self.sidecar.name + ".sieve-state").write_bytes(data)
        self.sidecar.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "requires GFNSV CUDA state version 3"):
            queue.inspect_sieve_file(self.path, 2)

    def test_wrong_abc_exponent_and_sidecar_n(self):
        self.make_abc(abc_n=3)
        with self.assertRaisesRegex(RuntimeError, "exponent"):
            queue.inspect_sieve_file(self.path, 2)
        self.make_abc(n=3, factors={}, abc_n=2)
        with self.assertRaisesRegex(RuntimeError, "n=3.*n=2"):
            queue.inspect_sieve_file(self.path, 2)

    def test_malformed_abc_header_rejected(self):
        self.make_abc()
        headers = [b"ABC $a^04+1 // Sieved to 100", b"ABC $a^4+1 // Sieved to 0100",
                   b" ABC $a^4+1 // Sieved to 100", b"\xef\xbb\xbfABC $a^4+1 // Sieved to 100",
                   b"ABC $a^4+1 // Sieved to -1",
                   b"ABC $a^4+1", b"ABC $a^4+1 // Sieved to 100 ",
                   b"ABC $a^4+1 // Sieved to 18446744073709551616",
                   b"ABC $a^4+1 // Sieved to " + b"1" * 1100]
        for header in headers:
            with self.subTest(header=header[:80]):
                self.path.write_bytes(header + b"\n4\n6\n12\n")
                with self.assertRaises(RuntimeError):
                    queue.inspect_sieve_file(self.path, 2)

    def test_invalid_abc_records_rejected(self):
        self.make_abc()
        records = [b"4\n4\n6\n12\n", b"6\n4\n12\n", b"3\n4\n6\n12\n",
                   b"4\n6\n12\n14\n", b"04\n6\n12\n", b"4^4+1\n6\n12\n",
                   b"4\n\n6\n12\n", b"#FACTOR 2 17\n4\n6\n12\n",
                   b"4\n6\n12\n\xff\n", b"4\n6\n12\n18446744073709551616\n",
                   b"4\n6\n12\n" + b"1" * 1100 + b"\n"]
        for body in records:
            with self.subTest(body=body[:80]):
                self.path.write_bytes(b"ABC $a^4+1 // Sieved to 100\n" + body)
                with self.assertRaises(RuntimeError):
                    queue.inspect_sieve_file(self.path, 2)

    def test_abc_base_below_checkpoint_range_rejected(self):
        self.make_abc(bmin=4, abc_bases=[2, 4, 6, 12])
        with self.assertRaisesRegex(RuntimeError, "outside the checkpoint range"):
            queue.inspect_sieve_file(self.path, 2)

    def test_abc_boundary_never_ahead_of_state(self):
        self.make_abc(next_k=8, factors={2: 17, 8: 17}, sieved_to=64)
        self.assertFalse(queue.inspect_sieve_file(self.path, 2)["complete"])
        self.make_abc(next_k=8, factors={2: 17, 8: 17}, sieved_to=65)
        with self.assertRaisesRegex(RuntimeError, "boundary is ahead"):
            queue.inspect_sieve_file(self.path, 2)
        self.make_abc(sieved_to=101)
        with self.assertRaisesRegex(RuntimeError, "boundary is ahead"):
            queue.inspect_sieve_file(self.path, 2)

    def test_plain_consumed_queue_never_rehydrates_sidecar(self):
        data, _ = self.make_abc()
        for backup in (data, b"damaged checkpoint\n"):
            for remaining in (b"6\n12\n", b""):
                with self.subTest(remaining=remaining, valid_backup=backup == data):
                    self.sidecar.write_bytes(backup)
                    self.path.write_bytes(remaining)
                    self.assertIsNone(queue.inspect_sieve_file(self.path, 2))
                    self.assert_unchanged_export_failure()

    def test_misplaced_abc_header_rejected(self):
        self.make_abc()
        self.path.write_bytes(b"4\n" + self.path.read_bytes())
        with self.assertRaises(RuntimeError):
            queue.inspect_sieve_file(self.path, 2)

    def test_empty_range_and_no_survivors_export_empty_queue(self):
        for args in ({"bmin": 3, "bmax": 3, "factors": {}},
                     {"bmin": 2, "bmax": 2, "factors": {2: 17}}):
            with self.subTest(args=args):
                self.make_abc(**args)
                self.assertTrue(queue.inspect_sieve_file(self.path, 2)["complete"])
                queue.export_completed_queue(self.path, 2)
                self.assertEqual(self.path.read_bytes(), b"")
                self.assertIsNone(queue.inspect_sieve_file(self.path, 2))

    def test_crlf_abc_and_sidecar_are_supported(self):
        self.make_abc()
        for path in (self.path, self.sidecar):
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertTrue(queue.inspect_sieve_file(self.path, 2)["complete"])
        queue.export_completed_queue(self.path, 2)
        self.assertEqual(self.path.read_bytes(), b"4\n6\n12\n")

    def test_export_checks_the_actual_opened_snapshot_before_publish(self):
        self.make_abc()
        real_inspect = queue._inspect_v3_file

        def replace_after_validation(path, expected_n):
            info = real_inspect(path, expected_n)
            path.write_bytes(v3_state(factors={2: 17, 8: 17})[0])
            return info

        with mock.patch.object(queue, "_inspect_v3_file", side_effect=replace_after_validation):
            self.assert_unchanged_export_failure()

    def test_result_helpers_unchanged(self):
        hits = self.path.parent / "hits.txt"
        self.assertEqual(queue.load_result_hits(hits), [])
        queue.write_result_hits(hits, [4, 6, 4])
        self.assertEqual(queue.load_result_hits(hits), [4, 6])
        self.assertEqual(hits.read_text(), "2\n4\n6\n")
        hits.write_text("2\n4\n4\n", encoding="ascii")
        with self.assertRaises(RuntimeError):
            queue.load_result_hits(hits)

    def test_gfn_n_is_log2_exponent_and_exports_same_queue(self):
        data, alive = self.make_abc(n=16, pmax=1_000_000)
        self.path.write_bytes(gfn_output(alive, n=16, sieved_to=1_000_000))
        self.assertEqual(queue.inspect_sieve_file(self.path, 16)["n"], 16)
        info, survivors = queue.read_completed_snapshot(self.path, 16)
        self.assertEqual(list(survivors), alive)
        self.assertTrue(info["complete"])
        queue.export_completed_queue(self.path, 16)
        self.assertEqual(self.path.read_bytes(), b"".join(f"{b}\n".encode() for b in alive))
        self.assertEqual(self.sidecar.read_bytes(), data)

    def test_gfn_wrong_n_or_exponent_written_as_n_rejected(self):
        self.make_abc()
        for n in (0, 3, 16, 65536, 21):
            with self.subTest(n=n):
                self.path.write_bytes(gfn_output([4, 6, 12], n=n))
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    queue.inspect_sieve_file(self.path, 2)

    def test_gfn_missing_sidecar_and_malformed_header_rejected(self):
        self.path.write_bytes(gfn_output([4, 6, 12]))
        with self.assertRaisesRegex(RuntimeError, "sieve-state"):
            queue.inspect_sieve_file(self.path, 2)
        self.sidecar.write_bytes(v3_state()[0])
        for header in (b"GFN n=02 // Sieved to 100", b"GFN n=2 // Sieved to 0100",
                       b"GFN n=-2 // Sieved to 100", b"GFN n=2", b" GFN n=2 // Sieved to 100",
                       b"\xef\xbb\xbfGFN n=2 // Sieved to 100"):
            with self.subTest(header=header):
                self.path.write_bytes(header+b"\n4\n6\n12\n")
                with self.assertRaises(RuntimeError):
                    queue.inspect_sieve_file(self.path, 2)

    def test_gfn_stale_superset_snapshot_uses_sidecar_survivors(self):
        self.make_abc()
        self.path.write_bytes(gfn_output([2, 4, 6, 8, 10, 12], sieved_to=0))
        info, bases = queue.read_completed_snapshot(self.path, 2)
        self.assertTrue(info["complete"])
        self.assertEqual(list(bases), [4, 6, 12])

    def test_gfn_misplaced_header_rejected(self):
        self.path.write_bytes(b"4\n"+gfn_output([6, 12]))
        with self.assertRaises(RuntimeError):
            queue.inspect_sieve_file(self.path, 2)

    def test_readonly_snapshot_opens_authority_once(self):
        data, alive = self.make_abc()
        self.path.write_bytes(gfn_output(alive))
        original = self.path.read_bytes()
        opens = []
        real_open = Path.open

        def record_open(path, *args, **kwargs):
            opens.append(path)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", record_open):
            info, survivors = queue.read_completed_snapshot(self.path, 2)
        self.assertEqual(opens, [self.path, self.sidecar])
        self.assertEqual(list(survivors), alive)
        self.assertTrue(info["complete"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.sidecar.read_bytes(), data)

    def test_readonly_snapshot_rejects_plain_partial_corrupt(self):
        for data in (b"4\n6\n12\n", v3_state(next_k=8, factors={2: 17, 8: 17})[0],
                     v3_state()[0][:-8]):
            with self.subTest(data=data[:50]):
                self.path.write_bytes(data)
                with self.assertRaises(RuntimeError):
                    queue.read_completed_snapshot(self.path, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
