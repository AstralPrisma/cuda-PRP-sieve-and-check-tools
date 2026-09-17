# GHCWPS 1.0

Generalized Hyper-Cullen / Woodall Prime Seeker, by A.P., Sept 2026.

用于 `b^n*n^b±1` 的 Fermat PRP
检验，复用 GFPPS 的整数 NTT / Montgomery 算术；PRP 不是素性证明。

```sh
./GHCWPS --check '7^118330*118330^7+1' --checkpoint hc_7_118330.ckpt
./GHCWPS --check '7^118330*118330^7+1' --checkpoint hc_7_118330.ckpt --resume-checkpoint
```

Windows 使用 `GHCWPS.exe`，表达式用双引号。`b,n` 均为十进制整数，范围
2～4294967295；实际数的位数另受 NTT 容量限制，构数前检查大小。

- Ctrl+C 保存安全边界；默认每100000 bit保存并打印进度。
- `--checkpoint-every-bits`、`--progress-every-bits` 分别调整保存/输出间隔。
- `--max-bits N` 仅推进 N bit，用于有限测试；结果 PARTIAL 不是 PRP。
- 断点含参数、模数哈希、完整状态 SHA256；不同项目或表达式的断点不能混用。
- 已存在断点必须明确 `--resume-checkpoint`，否则拒绝覆盖。
- 默认底数2；`--witness 2..255` 可调整。
- `--verify-cpp-int` 独立核对小数的完整/部分余数，限8192 bit。
- `--cpu-reference` 只用于小规模测试，不调用 CUDA，不是优化过的 CPU PRP 工具。
- 程序会直接排除显然为偶数和由 gcd(b,n) 导致可代数分解的候选。

与筛法配合时，使用 GHCWSV 目录中的 `ghcw_to_cands.py` 生成独立表达式文件。
Prime Seeker 网络项目使用独立 GHCWPS 任务格式，见统一客户端文档。

算术代码来自 GFPPS / GPRPS，保留 GPL-2.0-or-later 许可证。CUDA Toolkit 和
Boost 头文件用于编译；运行需要兼容的 NVIDIA 驱动。

附带 Windows 和 Ubuntu22.04+ x86-64 原生程序，各平台分别提供 sm_86、sm_89、sm_100、sm_120 四个文件。
当前横幅重建的16个文件均检查了目标架构、UTF-8、help/version和CPU参考路径，未运行GPU测试。
前一版算术路径曾在 RTX4060 Laptop 上验证；不把该证据当成本轮二进制实机测试。构建使用 CUDA13.3；可以通过 CMake 自行重建：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;100;120"
cmake --build build --config Release --parallel 2
```

Windows 请在已配置 MSVC/CUDA 的终端运行，按需要向 CMake 指定 Boost 路径。
Linux 若需要兼容 Ubuntu22.04，应在对应系统/容器内构建。测试记录见 VALIDATION.md。

横幅规则：筛法在 main 开头打印；PRP 检查器仅 help/usage 打印。Windows 控制台使用 UTF-8，非 ASCII 横线以显式 UTF-8 字节表示。
