# SPDX-License-Identifier: GPL-2.0-or-later
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument('--binary', type=Path, required=True)
ap.add_argument('--report', type=Path, required=True)
args = ap.parse_args()
binary = args.binary.resolve()
checks = []
assert b'SUPPLIER_SELFTEST_PASS checks=12' in subprocess.check_output([str(binary), '--selftest-supplier'])
assert b'prime_sequence_selftest=PASS, cases=32' in subprocess.check_output([str(binary), '--selftest-primes'])
checks += ['supplier starvation/rollback/freeze policy:12 checks', '32 exact prime-sequence comparisons']
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot(path):
    body, footer = path.read_bytes().rsplit(b'sha256 ', 1)
    assert hashlib.sha256(body).hexdigest().encode() == footer.strip()
    return dict(line.split(' ', 1) for line in body.decode().splitlines()[1:9])
with tempfile.TemporaryDirectory(prefix='gfpdivsv-auto-test-') as directory:
    work = Path(directory)
    def run(label, options, ok=True):
        output = work / (label + '.txt')
        result = subprocess.run([str(binary), '-o', str(output), *map(str, options)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        assert (result.returncode == 0) == ok, result.stdout.decode(errors='replace')[-3000:]
        if output.exists(): snapshot(output)
        return output, result.stdout.decode(errors='replace')
    common = ['--n', 1000, '--family', 'factorial', '--sign', 'both', '-P', 1000000]
    reference, _ = run('reference', common + ['--prime-mode', 'array'])
    cache = work / 'profile.tuning'
    cold, log = run('cold', common + ['--tuning-cache', cache])
    assert 'autotune_cache=miss' in log
    hot, log = run('hot', common + ['--tuning-cache', cache])
    assert 'autotune_cache=hit' in log
    assert cold.read_bytes() == hot.read_bytes() == reference.read_bytes()
    checks.append('cold/hot calibration does not advance or alter search state')
    changed, log = run('changed-region', common + ['--tuning-cache', cache, '--prime-span', 1048576])
    assert 'autotune_cache=miss' in log and changed.read_bytes() == reference.read_bytes()
    checks.append('resource-configuration changes invalidate the profile')
    cache.write_bytes(cache.read_bytes().replace(b'batch ', b'Batch ', 1))
    corrupt, log = run('corrupt', common + ['--tuning-cache', cache])
    assert 'autotune_cache=miss' in log and corrupt.read_bytes() == reference.read_bytes()
    checks.append('checksum-corrupt profile is recalibrated')
    data = cache.read_bytes()
    body, _ = data.rsplit(b'sha256 ', 1)
    key_start = body.index(b'key ') + 4
    body = body[:key_start] + (b'0' if body[key_start:key_start+1] != b'0' else b'1') + body[key_start+1:]
    cache.write_bytes(body + b'sha256 ' + hashlib.sha256(body).hexdigest().encode() + b'\n')
    mismatch, log = run('mismatched-key', common + ['--tuning-cache', cache])
    assert 'autotune_cache=miss' in log and mismatch.read_bytes() == reference.read_bytes()
    checks.append('validly checksummed hardware/task mismatch is recalibrated')
    foreign = work / 'foreign.tuning'
    foreign.write_text('unrelated user data\n')
    before = foreign.read_bytes()
    output, log = run('foreign-cache', common + ['--tuning-cache', foreign])
    assert foreign.read_bytes() == before and output.read_bytes() == reference.read_bytes()
    checks.append('unrelated cache target is preserved')
    pinned, log = run('pinned', common + ['--batch-primes', 65536, '--prime-threads', 2])
    assert 'prime_supply=pipeline, prime_threads=2' in log and 'batch=65536' in log
    assert pinned.read_bytes() == reference.read_bytes()
    checks.append('manual batch/thread overrides honored')
    manual, log = run('manual', common + ['--prime-mode', 'pipeline', '--prime-threads', 4, '--batch-primes', 131])
    assert 'autotune_cache=' not in log and manual.read_bytes() == reference.read_bytes()
    checks.append('explicit non-auto modes bypass all calibration')
    before = reference.read_bytes()
    run('bad-overlap', ['-i', reference, '-P', 1000010, '--tuning-cache', reference], ok=False)
    assert reference.read_bytes() == before
    checks.append('profile cannot overwrite input/output snapshots')
    stop = work / 'already.stop'
    stop.touch()
    output = work / 'pre-stopped.txt'
    result = subprocess.run([str(binary), '-i', str(reference), '-P', '1e10', '-o', str(output), '--stop-file', str(stop)], capture_output=True)
    assert result.returncode == 130 and output.read_bytes() == reference.read_bytes()
    checks.append('stop before calibration does not falsely mark COMPLETE')
    # A complete final batch whose last prime equals PMAX must not create an
    # inverted pipeline range during supplier-profile transitions.
    output, _ = run('prime-endpoint', ['--n', 1000, '--family', 'factorial', '-P', 104729,
                                     '--prime-mode', 'pipeline', '--prime-threads', 2, '--batch-primes', 16])
    assert snapshot(output)['sieved_to'] == '104729'
    checks.append('prime endpoint and partial tail')
    # Stop during the first calibration trial, without touching production data.
    stop = work / 'during-calibration.stop'
    output = work / 'calibration-stopped.txt'
    logfile = work / 'calibration-stopped.log'
    with logfile.open('wb') as stream:
        process = subprocess.Popen([str(binary), '--n', '26951', '--family', 'factorial', '-P', '1e12',
                                    '-o', str(output), '--stop-file', str(stop)], stdout=stream, stderr=stream)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if b'CUDA device=' in logfile.read_bytes():
                stop.touch()
                break
            assert process.poll() is None
            time.sleep(.005)
        assert process.wait(timeout=30) == 130
    assert int(snapshot(output)['sieved_to']) == 26951
    checks.append('interrupted calibration never commits pilot work')
report = {'ok': True, 'program': 'GFPDIVSV1.1 CUDA', 'binary': binary.name,
          'sha256': digest(binary), 'checks': checks}
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
