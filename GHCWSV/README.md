# GHCWSV 1.1

Generalized Hyper-Cullen / Woodall Siever, by A.P., Sept 2026.

固定 b，扫描 n，筛选
`b^n*n^b+1`、`b^n*n^b-1`，或使用 `--sign both` 同时筛选两者。一个文件固定一个 b，包含单个符号或两种符号的剩余候选。

```sh
./GHCWSV -b 7 -n 2 -N 100000 --sign +1 -P 1e8 -o hc7_plus.txt
./GHCWSV -i hc7_plus.txt -P 1e9 -o hc7_plus.txt
./GHCWSV -b 7 -n 2 -N 100000 --sign both -P 1e8 -o hc7_both.txt
./GHCWSV -i hc7_both.txt -P 1e9 -o hc7_both.txt
```

Windows 程序为 `GHCWSV.exe`。n 上下界均包含，b,n 范围2～4294967295，
一次新建最多一千万个 n。交换 b,n 是同一个数，跨项目应自行避免重复范围。

## 双符号模式

`--sign both`（也可写 `--sign +/-1`）在同一轮素数生成中处理正负两项，共用模幂/模积结果；每个符号有独立的存活位和因子槽。筛掉 `+1` 不会连带删除同一个 n 的 `-1`，反之亦然。奇偶性和代数因子预排除也分别应用。

`survivors` / `candidates` 统计完整表达式数量，另外显示 `plus`、`minus`；`n_values` 是不同 n 的数量。例如同一个 n 的两种符号都存活时计为两个候选、一个 n。双符号最多两千万个候选。

双符号结果使用 v2 文件，每行两列 `n sign`，相同 n 按 `-1`、`+1` 排序：

```text
ABC 7^$a*$a^7$b // GHCWSV v2 sieved_to=100000000 count=...
...
1000 -1
1000 +1
...
#SHA256 <64 lowercase hex digits>
```

这里只是格式示意，不能作为真实已筛选结果使用。续筛自动从文件识别模式，无需再写 `--sign`。即使后来仅剩一种符号，文件仍保留原来的 both 模式和完成边界。

旧版 v1 单符号文件可直接续筛，输出仍为 v1。不能把已有单符号文件加 `--sign both` 直接扩大成双符号：另一符号没有该筛选深度的证据，程序会拒绝这种模式变更。GHCWSV 1.0 不能读取新 v2 文件。

## 筛选与保存

- `--algorithm auto`：若整批候选的 n^b 都小于筛因子，就预计算整数系数；
  否则，小 b 使用直接路径，b>64 的奇数底数按 gcd(b,p-1) 分组，互素组走
  逆指数变换，其余组完整计算 n^b mod p。所有 p 都会处理，不遗漏不互素组。
- `--algorithm transform`：强制尝试逆指数变换；不满足条件仍完整回退。
- `--algorithm direct`：不使用变换，作为 GPU 对照路径。
- 先排除奇偶性和 gcd(b,n) 可直接证明为合数的项。这不是完整代数因子检测。
- 每个新发现的因子都在 CPU 独立验证；不把候选等于因子的质数误删。
- Ctrl+C 等待当前已提交 GPU 批次结束，只保存确实完成的素数边界。
- 默认每秒报告进度、剩余候选、吞吐和 ETA；`--progress-seconds N` 可改。
- 默认每60秒自动保存，`--checkpoint-seconds N` 可改；结束时也保存。
- 输出使用原子替换，并保留上一份 `.previous`。恢复只需候选文件，不需因子文件。
- `-O factors.txt` 可选保存因子；不能与候选输入/输出共用路径。
- `--prime-threads N` 限1～16，默认8；`--batch-primes N` 上限默认8192。
  实际批量随不同 n 的数量缩小，n 按256个一组并行，每次核函数至多计算200万个 `(p,n)` 对；both 模式从每个模积检查两种符号。
- Linux 优先加载程序旁 `lib/libprimesieve.so.12`；仅接受匹配的12.x迭代器。
  若无兼容库，auto 使用内建分段筛。Windows 可提供12.x的 primesieve.dll。
- `--cpu-reference` 是小规模参考测试模式，不调用 CUDA；不是日常性能模式。
- `--selftest` 在 GPU 上验证252组模运算与变换，覆盖接近2^62的筛因子。

文件格式示例（末尾实际 SHA256 不可省略）：

```text
ABC 7^$a*$a^7+1 // GHCWSV v1 sieved_to=100000000 count=...
...
#SHA256 <64 lowercase hex digits>
```

勿手工修改筛选边界或删除行。截断、损坏、重复/乱序候选及参数不匹配会被拒绝。
空候选文件也是有效的完成状态；文件名中的深度不参与恢复，实际读取头部边界。

## 转换为单候选文件

```sh
python scripts/ghcw_to_cands.py hc7_plus.txt --out-dir cands7 --nmin 2 --nmax 5000
python scripts/ghcw_to_cands.py hc7_both.txt --out-dir network_cands --network
python scripts/ghcw_to_cands.py hc7_both.txt --out-dir minus_only --sign=-1
```

每个 TXT 含一行 `b^n*n^b±1`，供 GHCWPS 使用。转换器不上传服务器。
加 `--network` 可导出 Prime Seeker 的两行任务格式，不自动联网或上传。
转换器支持 v1/v2，默认导出文件中全部符号；`--sign=+1/-1` 可只导出一侧。文件名的 S/R 后缀与表达式逐项对应，不会覆盖同一个 n 的另一符号。

## 构建与验证

附带 Windows 和 Ubuntu22.04+ x86-64 原生程序，各平台分别提供 sm_86、sm_89、sm_100、sm_120 四个文件。
1.1 的八个原生构建均检查目标架构、UTF-8、help/version及CPU参考路径。Windows和WSL的 sm_89 构建在同一台 RTX4060 Laptop 上完成双符号GPU对照和中断续筛测试；其他架构仅交叉编译，未实机验证。构建使用 CUDA13.3；驱动仍须支持目标 GPU。

安装 CUDA Toolkit、Boost 头文件、CMake 和 C++编译器后：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;100;120"
cmake --build build --config Release --parallel 2
```

需要 Ubuntu22.04 兼容性时，请在22.04或相应容器内构建；仅静态链接
libstdc++ 并不能消除较新 glibc 的依赖。1.1验证见 VALIDATION_both_20260913.md；VALIDATION.md保留之前版本的验证范围。

素数流水线和 Montgomery 模运算继承 GNCWSV / GSRSV；其 mtsieve 来源保留
Mark Rodenkirch 等原作者信息。许可证 GPL-2.0-or-later。

横幅规则：筛法在 main 开头打印；PRP 检查器仅 help/usage 打印。Windows 控制台使用 UTF-8，非 ASCII 横线以显式 UTF-8 字节表示。
