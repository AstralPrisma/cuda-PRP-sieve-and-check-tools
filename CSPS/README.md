# Catena Siderum Prime Search (CSPS 1.0)

CUDA Fermat PRP checking for

`product((x+j*d)^(x+j*d+1), j=0..t-1) +/- 1`, with step `d=1` or `d=2`.

For example, `Catena(x=6,terms=4,step=1)+1` represents
`6^7 * 7^8 * 8^9 * 9^10 + 1`. The explicit numeric construction is exact;
GPU modular arithmetic uses integer NTT/CRT and checked Montgomery reduction.
A passing Fermat test is **not** a primality proof.

```sh
./CSPS_sm_89 --start 6 --terms 4 --step 1 --sign +
./CSPS_sm_89 --start 8670 --terms 2 --step 1 --sign + --checkpoint progress.ckpt
```

Run `--help` for resource bounds, checkpoint and diagnostic options. Prefix
tests ending in `PARTIAL` are not complete PRP results. Keep separate work
directories/checkpoints for different candidates. Checkpoint compatibility
with other programs such as GFPPS is not assumed.

Release archives include Windows and Linux executables for sm_86, sm_89,
sm_100 and sm_120. Native compilation/CPU startup tests do not establish
physical-GPU validation on every target; consult release provenance.

Build scripts use CUDA C++17 and Boost headers (`BOOST_ROOT`). Linux release
builds target Ubuntu22.04/glibc2.35 compatibility and link C++ runtimes statically.

## 中文

CSPS 检验连续或隔项的幂乘积加减1。Windows/Linux均提供四种SM版本。
结果PRP只表示通过费马测试，仍需独立素性证明；中断和部分结果不能当作合数。
配套筛子为[CSSV](../CSSV/)，候选可由其脚本转换为逐任务文件。
