# 自有 Linux 云服务器

不要求 AutoDL，但 GPU 实验默认配置仍面向至少 20GiB 显存的 NVIDIA GPU。实际 GPU、驱动、内存和隔离能力必须先检查；CPU 服务器可用于离线机制测试和具备已验证隔离后端时的判题。

先在已经安装 Conda 的 shell 中设置以下变量，每次开启实验终端都要设置。数据路径可改为容量充足的挂载盘，不要求 root 用户：

```bash
export TP_DATA="$HOME/trainingprogram-data"
export CONDA_EXE="$(conda info --base)/bin/conda"
export TP_TOOLS_PYTHON="$(conda info --base)/envs/tp-tools/bin/python"
bash scripts/bootstrap.sh --agent-only
conda activate tp-agent
tp check --stage agent
tp selftest
```

若 Conda 配置了其他环境目录，在创建 tp-tools 后将 `TP_TOOLS_PYTHON` 改为该环境中 Python 的真实路径。离线自检使用模拟响应，不产生真实模型成绩。

GPU 安装仍固定 CUDA 12.8，要求驱动至少 570.26；脚本不修改宿主机驱动。编译默认架构 89 适用于原定 RTX 4090。其他 GPU 可在 GPU 可见时设置 `export TP_CUDA_ARCHITECTURES=native`，然后运行 `bash scripts/bootstrap.sh --all`。不能把此设置视为所有 GPU 已经验证兼容。

准备完成后按 README 的冒烟、正式评测和 Agent 步骤执行。若 Docker 不可用，可验证用户级 Bubblewrap 后端，或把结果移到具备 Docker 的机器判题，参见 [隔离执行说明](SANDBOX.md)。服务器配置尚未确认时不要直接开始完整模型下载或正式实验。

当前云服务器的源码位于 `/home/sunyue/trainingprogram`，模型和结果位于 `/backup/trainingprogram-data`，环境和缓存位于 `/backup/trainingprogram-setup`；不使用 `/backup/sunyue`。

在激活项目 Python 3.11 环境并设置 TP_DATA 后，可运行 `bash scripts/setup_bubblewrap.sh` 创建独立判题环境并执行四项真实隔离测试。两个框架示例环境由 `bash scripts/setup_framework_examples.sh` 创建。环境安装成功不代表真实 Agent 成功，仍须保留实际工具调用和独立验收记录。
