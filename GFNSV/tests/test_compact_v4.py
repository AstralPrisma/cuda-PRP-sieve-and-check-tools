"""Mock self-contained compact v4 fixtures; never run a sieve or use a GPU."""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gfnsv_queue as queue
import gfnsv_to_cands as converter
from test_gfnsv_queue import gfn_output, v3_state


def compact_state(*, fmt="G", n=2, bmin=2, bmax=12, start_pmin=3,
                  pmax=100, next_k=None, bases=(4, 6, 12)):
    """Independent compact encoder; stores only the explicitly given survivors."""
    q = 1 << (n + 1)
    if next_k is None:
        next_k = (pmax - 1) // q + 1
    pmin = next_k * q + 1
    first_even = bmin + (bmin & 1)
    slots = max(0, (bmax-first_even)//2+1)
    complete = slots == 0 or next_k > (pmax-1)//q
    marker = "#GFNSV_COMPLETE" if complete else "#GFNSV_STATE"
    header = (f"{marker} version=4 engine=cuda n={n} bmin={bmin} bmax={bmax} "
              f"start_pmin={start_pmin} pmin={pmin} pmax={pmax} next_k={next_k} "
              f"slots={slots} alive_count={len(bases)} format={fmt}")
    boundary = min(pmax, pmin-1)
    prefix = (f"GFN n={n} // format=4 Sieved to {boundary}" if fmt == "G" else
              f"ABC $a^{1 << n}+1 // format=4 Sieved to {boundary}" if fmt == "A" else None)
    lines = ([prefix] if prefix is not None else []) + [header]
    lines += [str(b)+(f"^{1 << n}+1" if fmt == "expr" else "") for b in bases]
    body = "\n".join(lines)+"\n"
    checksum = hashlib.sha256(body.encode()).hexdigest()
    return (body+f"#GFNSV_END count={len(bases)} sha256={checksum}\n").encode("ascii")


def rehash(data):
    lines = data.splitlines()
    body = b"\n".join(lines[:-1])+b"\n"
    count = lines[-1].split(b"count=", 1)[1].split(b" ", 1)[0]
    return body+b"#GFNSV_END count="+count+b" sha256="+hashlib.sha256(body).hexdigest().encode()+b"\n"


class CompactSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gfnsv_compact_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/"sieved.gfn"
        self.sidecar = self.path.with_name(self.path.name+".sieve-state")

    def test_all_formats_are_self_contained_and_input_unchanged(self):
        for fmt in ("G", "A", "base", "expr"):
            with self.subTest(fmt=fmt):
                data = compact_state(fmt=fmt)
                self.path.write_bytes(data)
                info, bases = queue.read_completed_snapshot(self.path, 2)
                self.assertEqual((info["version"], info["format"], info["complete"]), (4, fmt, True))
                self.assertEqual(list(bases), [4, 6, 12])
                self.assertEqual(queue.inspect_sieve_file(self.path, 2), info)
                self.assertEqual(self.path.read_bytes(), data)
                self.assertFalse(self.sidecar.exists())

    def test_n16_dedicated_header_and_abc_exponent(self):
        for fmt, prefix in (("G", b"GFN n=16 // format=4"), ("A", b"ABC $a^65536+1 // format=4")):
            self.path.write_bytes(compact_state(fmt=fmt, n=16, pmax=1_000_000))
            self.assertTrue(self.path.read_bytes().startswith(prefix))
            self.assertEqual(queue.read_completed_snapshot(self.path, 16)[0]["n"], 16)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                queue.read_completed_snapshot(self.path, 15)

    def test_compact_ignores_even_corrupt_adjacent_sidecar(self):
        self.path.write_bytes(compact_state())
        self.sidecar.write_bytes(b"not a valid old checkpoint\n")
        self.assertEqual(list(queue.read_completed_snapshot(self.path, 2)[1]), [4, 6, 12])
        self.assertEqual(self.sidecar.read_bytes(), b"not a valid old checkpoint\n")

    def test_v4_reads_single_handle_and_does_not_open_sidecar(self):
        self.path.write_bytes(compact_state())
        opened = []
        original = Path.open

        def track(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", track):
            self.assertEqual(list(queue.read_completed_snapshot(self.path, 2)[1]), [4, 6, 12])
        self.assertEqual(opened, [self.path])

    def test_partial_is_readable_but_not_convertible(self):
        self.path.write_bytes(compact_state(next_k=8))
        self.assertFalse(queue.inspect_sieve_file(self.path, 2)["complete"])
        with self.assertRaisesRegex(RuntimeError, "partial"):
            queue.read_completed_snapshot(self.path, 2)
        with self.assertRaisesRegex(RuntimeError, "completed"):
            queue.export_completed_queue(self.path, 2)
        self.assertFalse(self.sidecar.exists())

    def test_empty_survivors_and_empty_original_range(self):
        for args in ({"bases": ()}, {"bmin": 3, "bmax": 3, "bases": ()}):
            for fmt in ("G", "A", "base", "expr"):
                with self.subTest(args=args, fmt=fmt):
                    self.path.write_bytes(compact_state(fmt=fmt, **args))
                    info, bases = queue.read_completed_snapshot(self.path, 2)
                    self.assertTrue(info["complete"])
                    self.assertEqual(len(bases), 0)

    def test_empty_range_boundary_is_frontier_not_automatically_pmax(self):
        self.path.write_bytes(compact_state(bmin=3, bmax=3, bases=(), next_k=1))
        self.assertTrue(self.path.read_bytes().startswith(b"GFN n=2 // format=4 Sieved to 8\n"))
        self.assertTrue(queue.read_completed_snapshot(self.path, 2)[0]["complete"])
        self.path.write_bytes(rehash(self.path.read_bytes().replace(b"Sieved to 8\n", b"Sieved to 100\n")))
        with self.assertRaisesRegex(RuntimeError, "boundary mismatch"):
            queue.read_completed_snapshot(self.path, 2)

    def test_loop_depends_on_survivors_not_64m_original_slots(self):
        self.path.write_bytes(compact_state(bmax=2*64*1024*1024, bases=(4, 6)))
        counts = []

        def guarded_range(count):
            counts.append(count)
            self.assertLessEqual(count, 2, "iterated original slots instead of survivor count")
            return builtins.range(count)

        with mock.patch.object(queue, "range", guarded_range, create=True):
            info, bases = queue.read_completed_snapshot(self.path, 2)
        self.assertEqual(info["slots"], 64*1024*1024)
        self.assertEqual(list(bases), [4, 6])
        self.assertEqual(counts, [2])

    def test_metadata_and_body_hash_tampering_rejected(self):
        data = compact_state()
        for broken in (data.replace(b"bmin=2", b"bmin=4"), data.replace(b"\n6\n", b"\n8\n"),
                       data.replace(b"alive_count=3", b"alive_count=2"), data.replace(b"sha256=", b"sha256=0"),
                       data+b"2\n", data[:-20]):
            with self.subTest(broken=broken[-70:]):
                self.path.write_bytes(broken)
                with self.assertRaises(RuntimeError):
                    queue.read_completed_snapshot(self.path, 2)

    def test_semantic_errors_rejected_even_with_recomputed_hash(self):
        data = compact_state()
        variants = (
            data.replace(b"pmin=105", b"pmin=104"), data.replace(b"next_k=13", b"next_k=14"),
            data.replace(b"slots=6", b"slots=7"), data.replace(b"format=G", b"format=A"),
            data.replace(b"Sieved to 100", b"Sieved to 99"), data.replace(b"GFN n=2", b"GFN n=3"),
            data.replace(b"version=4", b"version=5"), data.replace(b"// format=4", b"// format=5"),
            data.replace(b"bmin=2", b"bmin=02"), data.replace(b"count=3 sha256", b"count=2 sha256"),
            data.replace(b"#GFNSV_COMPLETE", b"#GFNSV_STATE"),
        )
        for broken in variants:
            with self.subTest(broken=broken[:120]):
                self.path.write_bytes(rehash(broken))
                with self.assertRaises(RuntimeError):
                    queue.read_completed_snapshot(self.path, 2)

    def test_compact_survivor_range_order_and_format_validation(self):
        for bases in ((4, 4, 6), (6, 4, 12), (3, 4, 6), (4, 6, 14), (0, 4, 6)):
            self.path.write_bytes(compact_state(bases=bases))
            with self.assertRaises(RuntimeError):
                queue.read_completed_snapshot(self.path, 2)
        data = compact_state(fmt="expr")
        for broken in (data.replace(b"4^4+1", b"4^8+1"), data.replace(b"4^4+1", b"04^4+1"),
                       data.replace(b"4^4+1", b"4")):
            self.path.write_bytes(rehash(broken))
            with self.assertRaises(RuntimeError):
                queue.read_completed_snapshot(self.path, 2)

    def test_compact_header_required_only_for_G_A(self):
        data = compact_state()
        for broken in (b"\n".join(data.splitlines()[1:])+b"\n",
                       rehash(b"GFN n=2 // format=4 Sieved to 100\n"+compact_state(fmt="base"))):
            self.path.write_bytes(broken)
            with self.assertRaises(RuntimeError):
                queue.read_completed_snapshot(self.path, 2)

    def test_crlf_has_same_canonical_lf_hash(self):
        data = compact_state()
        self.path.write_bytes(data)
        info, _ = queue.read_completed_snapshot(self.path, 2)
        self.path.write_bytes(data.replace(b"\n", b"\r\n"))
        self.assertEqual(queue.read_completed_snapshot(self.path, 2)[0], info)

    def test_first_line_only_never_falls_back_to_valid_old_empty_sidecar(self):
        self.sidecar.write_bytes(v3_state(bmin=2, bmax=2, factors={2: 17})[0])
        for fmt in ("G", "A"):
            data = compact_state(fmt=fmt, bmin=2, bmax=2, bases=())
            first = data.splitlines(keepends=True)[0]
            for prefix in (first, first[:-1], first.replace(b"format=4", b"format=5"),
                           first.replace(b"GFN", b"BROKEN").replace(b"ABC", b"BROKEN")):
                self.path.write_bytes(prefix)
                with self.assertRaises(RuntimeError):
                    queue.inspect_sieve_file(self.path, 2)
                with self.assertRaises(RuntimeError):
                    queue.read_completed_snapshot(self.path, 2)

    def test_compact_corrupt_or_truncated_never_uses_legacy_sidecar(self):
        self.sidecar.write_bytes(v3_state()[0])
        data = compact_state()
        for broken in (data[:-4], data[:data.find(b"slots=")+3],
                       data.replace(b"\n6\n", b"\n8\n"), data.replace(b"version=4", b"version=3")):
            self.path.write_bytes(broken)
            with mock.patch.object(queue, "_inspect_abc", side_effect=AssertionError("unsafe legacy fallback")):
                with self.assertRaises(RuntimeError):
                    queue.read_completed_snapshot(self.path, 2)

    def test_v4_export_consumption_backup_and_plain_never_rehydrates(self):
        for fmt in ("G", "A", "base", "expr"):
            if self.sidecar.exists():
                self.sidecar.unlink()
            data = compact_state(fmt=fmt)
            self.path.write_bytes(data)
            info = queue.export_completed_queue(self.path, 2)
            self.assertEqual(self.sidecar.read_bytes(), data)
            self.assertEqual(self.path.read_bytes(), b"4\n6\n12\n")
            self.assertEqual(queue._inspect_v3_file(self.sidecar, 2), info)
            for consumed in (b"6\n12\n", b""):
                self.path.write_bytes(consumed)
                self.assertIsNone(queue.inspect_sieve_file(self.path, 2))

    def test_v4_existing_different_consumption_backup_blocks_export(self):
        data = compact_state()
        self.path.write_bytes(data)
        self.sidecar.write_bytes(v3_state()[0])
        with self.assertRaisesRegex(RuntimeError, "snapshot differs"):
            queue.export_completed_queue(self.path, 2)
        self.assertEqual(self.path.read_bytes(), data)

    def test_v4_offline_converter_no_sidecar_or_input_mutation(self):
        for fmt in ("G", "A", "base", "expr"):
            data = compact_state(fmt=fmt)
            self.path.write_bytes(data)
            out = self.root/("cands-"+fmt)
            report = converter.convert(self.path, out, expected_n=2, dry_run=True)
            self.assertEqual(report["would_create"], 3)
            self.assertFalse(out.exists())
            report = converter.convert(self.path, out, expected_n=2)
            self.assertEqual(report["created"], 3)
            self.assertEqual((out/"cand_4.txt").read_bytes(), b"4\n")
            self.assertEqual(json.loads((out/converter.MANIFEST_NAME).read_bytes())["n"], 2)
            self.assertEqual(self.path.read_bytes(), data)
            self.assertFalse(self.sidecar.exists())

    def test_old_bare_empty_export_remains_legacy_compatible(self):
        self.sidecar.write_bytes(v3_state(bmin=2, bmax=2, factors={2: 17})[0])
        self.path.write_bytes(gfn_output([]))
        self.assertEqual(len(queue.read_completed_snapshot(self.path, 2)[1]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
