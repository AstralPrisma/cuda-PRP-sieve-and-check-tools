"""Offline mock snapshots only; no actual sieve input, network, or GPU."""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gfnsv_to_cands as converter
from test_gfnsv_queue import abc_output, gfn_output, v3_state


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gfnsv_converter_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sieved.gfn"
        self.sidecar = self.source.with_name(self.source.name + ".sieve-state")
        self.out = self.root / "cands"

    def source_fixture(self, kind="GFN", n=2, **kwargs):
        data, bases = v3_state(n=n, **kwargs)
        if kind == "v3":
            self.source.write_bytes(data)
        else:
            self.sidecar.write_bytes(data)
            pmax = kwargs.get("pmax", 100)
            frontier = kwargs.get("next_k", (pmax-1)//(1 << (n+1))+1)
            to = min(pmax, frontier*(1 << (n+1)))
            formatter = gfn_output if kind == "GFN" else abc_output
            self.source.write_bytes(formatter(bases, n=n, sieved_to=to))
        return bases

    def tree(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes() if p.is_file() else None
                for p in self.root.rglob("*") if not p.is_symlink()}

    def run_convert(self, **kwargs):
        return converter.convert(self.source, self.out, expected_n=kwargs.pop("expected_n", 2), **kwargs)

    def assert_failure_without_writes(self, **kwargs):
        before = self.tree()
        with self.assertRaises((OSError, RuntimeError)):
            self.run_convert(**kwargs)
        self.assertEqual(self.tree(), before)

    def test_all_three_formats_and_input_unchanged(self):
        for kind in ("GFN", "ABC", "v3"):
            with self.subTest(kind=kind):
                self.out = self.root / ("cands-"+kind)
                bases = self.source_fixture(kind)
                source = self.source.read_bytes()
                sidecar = self.sidecar.read_bytes() if self.sidecar.exists() else None
                report = self.run_convert()
                self.assertEqual(report["created"], len(bases))
                for base in bases:
                    self.assertEqual((self.out/f"cand_{base}.txt").read_bytes(), f"{base}\n".encode())
                self.assertEqual(self.source.read_bytes(), source)
                if sidecar is not None:
                    self.assertEqual(self.sidecar.read_bytes(), sidecar)
                manifest = json.loads((self.out/converter.MANIFEST_NAME).read_bytes())
                self.assertEqual((manifest["n"], manifest["exponent"]), (2, "4"))

    def test_cli_defaults_n16(self):
        self.source_fixture(n=16, pmax=1_000_000)
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            rc = converter.main([str(self.source), "--out-dir", str(self.out)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(capture.getvalue())["n"], 16)

    def test_wrong_expected_n_and_cli_failure(self):
        self.source_fixture()
        self.assert_failure_without_writes(expected_n=16)
        with contextlib.redirect_stderr(io.StringIO()) as capture:
            rc = converter.main([str(self.source), "--out-dir", str(self.out)])
        self.assertEqual(rc, 1)
        self.assertIn("error:", capture.getvalue())
        self.assertFalse(self.out.exists())

    def test_repeated_conversion_skips_exact_files(self):
        self.source_fixture()
        first = self.run_convert()
        before = self.tree()
        second = self.run_convert()
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["skipped_identical"], first["created"])
        self.assertEqual(self.tree(), before)

    def test_mixed_n_directory_is_rejected(self):
        self.source_fixture()
        self.run_convert()
        self.source_fixture(n=3, factors={})
        self.assert_failure_without_writes(expected_n=3)

    def test_all_targets_preflighted_before_any_write(self):
        self.source_fixture()
        self.out.mkdir()
        (self.out/converter.MANIFEST_NAME).write_text(json.dumps(converter._manifest(2)), encoding="ascii")
        # Earlier targets are absent; a conflict on the final target must not
        # create any earlier file or alter the existing manifest.
        (self.out/"cand_12.txt").write_bytes(b"999\n")
        self.assert_failure_without_writes()
        self.assertFalse((self.out/"cand_4.txt").exists())

    def test_existing_candidate_directory_is_rejected(self):
        self.source_fixture()
        self.out.mkdir()
        (self.out/converter.MANIFEST_NAME).write_text(json.dumps(converter._manifest(2)), encoding="ascii")
        (self.out/"cand_12.txt").mkdir()
        self.assert_failure_without_writes()

    def test_existing_candidates_without_manifest_rejected_even_identical(self):
        self.source_fixture()
        self.out.mkdir()
        (self.out/"cand_4.txt").write_bytes(b"4\n")
        self.assert_failure_without_writes()

    def test_unrelated_files_preserved(self):
        self.source_fixture()
        self.out.mkdir()
        unrelated = self.out/"notes.md"
        unrelated.write_bytes(b"keep this file\n")
        self.run_convert()
        self.assertEqual(unrelated.read_bytes(), b"keep this file\n")

    def test_deeper_snapshot_rejects_old_extraneous_tasks_without_writes(self):
        self.source_fixture()
        self.run_convert()
        (self.out/"cand_6.txt").unlink()
        self.source_fixture(pmax=300, factors={2: 17, 4: 257, 8: 17, 10: 73})
        self.assert_failure_without_writes()
        self.assertTrue((self.out/"cand_4.txt").exists())
        self.assertFalse((self.out/"cand_6.txt").exists())

    def test_unknown_or_noncanonical_cand_name_rejected_with_manifest(self):
        self.source_fixture()
        self.run_convert()
        for name in ("cand_999.txt", "cand_04.txt", "cand_notes.txt"):
            with self.subTest(name=name):
                extra = self.out/name
                extra.write_bytes(b"do not remove\n")
                self.assert_failure_without_writes()
                self.assert_failure_without_writes(dry_run=True)
                extra.unlink()

    def test_partial_is_rejected_before_output_creation(self):
        for kind in ("GFN", "ABC", "v3"):
            with self.subTest(kind=kind):
                self.source_fixture(kind, next_k=8, factors={2: 17, 8: 17})
                self.assert_failure_without_writes()

    def test_corrupt_and_missing_sidecars_are_rejected(self):
        self.source_fixture()
        data = self.sidecar.read_bytes()
        for bad in (data[:-9], data.replace(b"#FACTOR 2 17", b"#FACTOR 2 41"), b"4\n6\n12\n"):
            with self.subTest(bad=bad[:40]):
                self.sidecar.write_bytes(bad)
                self.assert_failure_without_writes()
        self.sidecar.unlink()
        self.assert_failure_without_writes()

    def test_plain_text_is_rejected_even_with_valid_sidecar(self):
        self.source_fixture()
        self.source.write_bytes(b"4\n6\n12\n")
        self.assert_failure_without_writes()

    def test_empty_complete_snapshot_writes_manifest_only(self):
        self.source_fixture(bmin=2, bmax=2, factors={2: 17})
        report = self.run_convert()
        self.assertEqual(report["candidates"], 0)
        self.assertEqual(report["created"], 0)
        self.assertEqual([p.name for p in self.out.iterdir()], [converter.MANIFEST_NAME])

    def test_empty_range_v3_is_valid(self):
        self.source_fixture("v3", bmin=3, bmax=3, factors={})
        self.assertEqual(self.run_convert()["candidates"], 0)

    def test_stale_export_does_not_reintroduce_removed_candidates(self):
        self.source_fixture()
        self.source.write_bytes(gfn_output([2, 4, 6, 8, 10, 12], sieved_to=0))
        self.run_convert()
        self.assertEqual(sorted(p.name for p in self.out.glob("*.txt")), ["cand_12.txt", "cand_4.txt", "cand_6.txt"])

    def test_missing_survivor_in_export_is_rejected(self):
        self.source_fixture()
        self.source.write_bytes(gfn_output([4, 12]))
        self.assert_failure_without_writes()

    def test_dry_run_does_not_create_parent_directory_or_temp_files(self):
        self.source_fixture()
        self.out = self.root / "not-created" / "nested" / "cands"
        before = self.tree()
        with mock.patch.object(converter.tempfile, "mkstemp", side_effect=AssertionError("dry run wrote a temp file")):
            result = self.run_convert(dry_run=True)
        self.assertEqual(result["would_create"], 3)
        self.assertEqual(result["created"], 0)
        self.assertEqual(self.tree(), before)

    def test_dry_run_existing_directory_unchanged(self):
        self.source_fixture()
        self.run_convert()
        before = self.tree()
        self.assertEqual(self.run_convert(dry_run=True)["skipped_identical"], 3)
        self.assertEqual(self.tree(), before)

    def test_dry_run_still_rejects_conflicts(self):
        self.source_fixture()
        self.run_convert()
        (self.out/"cand_6.txt").write_bytes(b"6\r\n")
        self.assert_failure_without_writes(dry_run=True)

    def test_standalone_dry_run_does_not_create_python_cache(self):
        self.source_fixture()
        scripts = self.root/"standalone"
        scripts.mkdir()
        for name in ("gfnsv_to_cands.py", "gfnsv_queue.py"):
            shutil.copyfile(Path(__file__).resolve().parents[1]/"scripts"/name, scripts/name)
        before = self.tree()
        environment = dict(os.environ)
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        proc = subprocess.run([sys.executable, str(scripts/"gfnsv_to_cands.py"), str(self.source),
                               "--out-dir", str(self.out), "--expected-n", "2", "--dry-run"],
                              cwd=scripts, env=environment, capture_output=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["would_create"], 3)
        self.assertEqual(self.tree(), before)

    def test_malformed_or_directory_manifest_rejected(self):
        self.source_fixture()
        self.out.mkdir()
        manifest = self.out/converter.MANIFEST_NAME
        for bad in (b"", b"{}", b"not json", b"[]", b"x"*9000,
                    b'{"format":"GFNSV_TASK_DIRECTORY","version":true,"n":2,"exponent":"4"}',
                    b'{"format":"GFNSV_TASK_DIRECTORY","version":1,"n":3,"n":2,"exponent":"4"}'):
            manifest.write_bytes(bad)
            self.assert_failure_without_writes()
        manifest.unlink()
        manifest.mkdir()
        self.assert_failure_without_writes()

    def test_file_cannot_be_output_directory(self):
        self.source_fixture()
        self.out.write_bytes(b"not a directory")
        self.assert_failure_without_writes()

    def test_symlink_target_and_output_directory_rejected(self):
        self.source_fixture()
        self.run_convert()
        target = self.out/"cand_6.txt"
        target.unlink()
        unrelated = self.root/"elsewhere"
        unrelated.write_bytes(b"6\n")
        try:
            target.symlink_to(unrelated)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"host cannot create test symlinks: {exc}")
        self.assert_failure_without_writes()
        link = self.root/"linked-output"
        link.symlink_to(self.out, target_is_directory=True)
        self.out = link
        self.assert_failure_without_writes()

    def test_snapshot_is_not_reopened_or_exported_in_place(self):
        self.source_fixture()
        real_snapshot = converter.read_completed_snapshot

        def read_then_retarget(*args):
            result = real_snapshot(*args)
            # Simulates a different snapshot replacing the filename only after
            # the first open has been validated. Conversion must use that first
            # immutable survivor set, not reread the changed path.
            self.source.write_bytes(b"untrusted replacement\n")
            self.sidecar.write_bytes(b"untrusted replacement\n")
            return result

        with mock.patch.object(converter, "read_completed_snapshot", side_effect=read_then_retarget) as reader:
            result = self.run_convert()
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(result["created"], 3)
        self.assertEqual((self.out/"cand_4.txt").read_bytes(), b"4\n")

    def test_interrupted_install_can_be_repeated_without_partial_task(self):
        self.source_fixture()
        install = converter._install_new

        def fail_second(path, contents):
            if path.name == "cand_6.txt":
                raise OSError("injected interruption")
            install(path, contents)

        with mock.patch.object(converter, "_install_new", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                self.run_convert()
        self.assertEqual((self.out/"cand_4.txt").read_bytes(), b"4\n")
        self.assertFalse((self.out/"cand_6.txt").exists())
        self.assertFalse(list(self.out.glob("*.tmp")))
        result = self.run_convert()
        self.assertEqual((result["created"], result["skipped_identical"]), (2, 1))

    def test_atomic_install_refuses_target_created_after_preflight(self):
        self.out.mkdir()
        target = self.out/"cand_4.txt"
        target.write_bytes(b"foreign data\n")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            converter._install_new(target, b"4\n")
        self.assertEqual(target.read_bytes(), b"foreign data\n")
        self.assertFalse(list(self.out.glob("*.tmp")))

    @unittest.skipIf(os.name == "nt", "POSIX fallback; Windows uses no-clobber rename")
    def test_no_hardlink_filesystem_uses_atomic_single_writer_fallback(self):
        self.source_fixture()
        with mock.patch.object(converter.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "no hard links")):
            self.assertEqual(self.run_convert()["created"], 3)
        self.assertFalse(list(self.out.glob("*.tmp")))
        self.assertEqual((self.out/"cand_4.txt").read_bytes(), b"4\n")

    def test_windows_reparse_attribute_is_rejected(self):
        status = mock.Mock(st_mode=0o100644, st_file_attributes=0x400)
        self.assertTrue(converter._is_link(status))

    def test_mock_symlinks_rejected_without_host_symlink_privileges(self):
        self.source_fixture()
        self.run_convert()
        real_status = converter._status
        for bad_path in (self.out/"cand_4.txt", self.out/converter.MANIFEST_NAME,
                         self.out, self.root):
            with self.subTest(path=bad_path.name):
                def simulated_status(path):
                    if path == bad_path:
                        return mock.Mock(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
                    return real_status(path)

                with mock.patch.object(converter, "_status", side_effect=simulated_status):
                    self.assert_failure_without_writes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
