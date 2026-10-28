# Catena Siderum Siever (CSSV 1.3)

CPU/CUDA sieve for the CSPS family. The `--sign both` path shares the modular
product calculation while retaining independent plus/minus survivor masks.
This is a sieve: survivors are not declared prime.

```sh
./CSSV_sm_89 --start 1 --end 100000 --terms 2 --step 1 --sign both \
  --pmax 1e10 --out csp_t2_x1_100000.txt
./CSSV_sm_89 -i csp_t2_x1_100000.txt.state -P 2e10 -o continued.txt
python scripts/cssv_to_tasks.py --help
```

The default output is a checked CSSV v2 survivor list `(x, sign)` plus a
resumable `.state` file. Preserve both for archival/continuation. Ctrl+C or
`--stop-file` safely commits a completed prime-batch boundary; default periodic
checkpoint interval is60seconds. A stopped run records its actual frontier,
not its requested target. Exact integral scientific notation is accepted.

The supported prime bound is `2^62-1`. `--backend gpu` requires a matching
CUDA target. `--backend cpu` is useful for bounded independent checks. Unlike
GFNSV, work generally depends on the active start values, term count and the
bit lengths of the modular exponents; width/size independence is not promised.

Place the supplied libprimesieve library in the packaged `bin/lib` location
alongside CSSV. See `--prime-generator` for explicit backend selection. The
BSD-licensed runtime and its copyright notice accompany release archives.

Build using the scripts in `scripts/`; each native sm_86/89/100/120 binary
is architecture-specific. Actual physical-GPU test coverage is recorded
separately from compilation and CPU-mode tests.

## 中文

CSSV是真正的CPU/CUDA筛子，支持同时筛正负号、进度、科学计数筛深、
按实际完成深度保存与续筛。输出列表可以由配套脚本拆成CSPS任务。
脚本本身不上传任务或自动发放积分。不同候选必须使用独立PRP断点。
