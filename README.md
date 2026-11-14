# Coding Agent 与 Qwen3.5 量化实验

面向三天课程作业：**LangGraph coding agent + 自行量化 + 可复现评测**。目标机器为 AutoDL RTX 4090 24GB、90GiB 内存、25 CPU配额、Ubuntu 20.04、50GB数据盘。

本次覆盖 Qwen3.5-0.8B／2B／4B 的 BF16 文本 GGUF 与 Q4_K_M。Qwen3.8、9B、27B及更大规模留待后续，**不代表原作业全部完成**。仓库不包含实验成绩、模型权重或密钥。

## 0. 服务器拉取与只读检查

```bash
cd /root
git clone https://github.com/sy-victoria/trainingprogram.git
cd trainingprogram
bash scripts/diagnose.sh
```

若仓库已拉取，先确认无本地修改再 `git pull --ff-only`。真实实验启动后记录提交并固定它；不要在同一运行ID下升级代码或依赖。

完整依赖版本见 `requirements/agent.lock`；官方模型revision、llama.cpp和EvalPlus提交见 `configs/versions.json`。本地验证范围见 [VALIDATION.md](docs/VALIDATION.md)，GPU和容器结果以实际服务器运行记录为准。

## 1. 安装独立环境

```bash
export TP_DATA=/root/autodl-tmp/trainingprogram-data
bash scripts/bootstrap.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tp-agent
tp doctor
python -m pytest -q
```

`bootstrap.sh` 创建 `tp-agent`、`tp-tools` 两个 Conda 环境，保留旧base环境。工具环境安装CUDA 12.8编译工具链，以4090的sm_89构建固定提交的llama.cpp；转换使用CPU版PyTorch，不为推理安装另一份GPU版PyTorch。原镜像的nvcc 11.3不参与编译。

脚本需要访问GitHub、PyPI、Conda/NVIDIA源和Hugging Face。连接失败先保留日志，修复网络后重跑；不要无依据更换模型revision或依赖版本。脚本不修改宿主驱动。

每次新终端都执行 `conda activate tp-agent`，并保持相同的 `TP_DATA`。`TP_TOOLS_PYTHON` 默认 `/root/miniconda3/envs/tp-tools/bin/python`；只有更改工具环境安装位置时才需要设置它。

### 判题隔离检查

```bash
docker info
bash scripts/build_sandbox.sh
tp doctor
```

**AutoDL容器不一定支持Docker。**如果失败，可以先完成生成，再把结果移至有Docker的机器判题；详见 [SANDBOX.md](docs/SANDBOX.md)。没有隔离环境时，不会偷偷在宿主机执行生成代码，也不能宣称Agent测试闭环已完成。

## 2. 首先只跑0.8B冒烟实验

```bash
bash scripts/run_pair.sh 0.8B smoke
```

脚本依次下载固定revision的权重、转换BF16、量化Q4_K_M、启动原版服务、生成两题、测速度、停止服务，再执行量化版。每题原始输出立即落盘；中断后用相同命令恢复。

进度日志位于终端和 `$TP_DATA/services/8080.log`。运行时不要同时启动另一个8080服务。若0.8B转换或生成失败，先反馈完整错误日志，**不要继续下载更大的模型**。

有Docker时判题：

```bash
tp eval judge --run qwen35-0.8B-BF16-smoke
tp eval judge --run qwen35-0.8B-Q4_K_M-smoke
tp summary
```

先检查两组都能正常生成和判题。冒烟题不要求模型全部答对，验收重点是数据链路、格式和错误分类正确。

## 3. 运行正式六组实验

```bash
bash scripts/run_pair.sh 0.8B subset
tp eval judge --run qwen35-0.8B-BF16-subset
tp eval judge --run qwen35-0.8B-Q4_K_M-subset

bash scripts/run_pair.sh 2B subset
tp eval judge --run qwen35-2B-BF16-subset
tp eval judge --run qwen35-2B-Q4_K_M-subset

bash scripts/run_pair.sh 4B subset
tp eval judge --run qwen35-4B-BF16-subset
tp eval judge --run qwen35-4B-Q4_K_M-subset
tp summary
```

30道固定题在六组中完全相同；`full`模式为全部164题。详细协议见 [EXPERIMENT.md](docs/EXPERIMENT.md)。更换mode、模型、配置或代码时必须使用新运行ID。请求过程中崩溃的题标记interrupted，不重复采样；希望重测需新建完整运行。

### 50GB数据盘管理

每个规模处理完成并核对结果后，可以显式清理可再下载/生成的文件，保留manifest与结果。例如准备4B前清理小模型：

```bash
tp model cleanup --size 0.8B --target source
tp model cleanup --size 0.8B --target BF16
tp model cleanup --size 0.8B --target Q4_K_M
tp model cleanup --size 2B --target source
tp model cleanup --size 2B --target BF16
tp model cleanup --size 2B --target Q4_K_M
df -h /root/autodl-tmp
```

命令只删除指定模型路径，不自动清除原权重。4B开始前建议至少26GiB可用空间。`tp model prepare` 可以重新下载原权重并校验已有产物。

### 分步操作与自定义运行ID

```bash
tp model prepare --size 0.8B
tp model serve --size 0.8B --variant BF16
```

保持服务终端运行，在第二个终端：

```bash
conda activate tp-agent
tp eval generate --run my-bf16-smoke --size 0.8B --variant BF16 --mode smoke
tp eval speed --run my-bf16-smoke
tp eval judge --run my-bf16-smoke
```

服务别名包含GGUF哈希，客户端会核对别名，避免误测另一个模型。切换精度先在服务终端Ctrl+C退出，再启动另一版本。

## 4. Coding Agent演示

完成4B实验后，启动4B量化服务并复制打印的别名（形如 `tp-0123456789abcdef`）：

```bash
tp model serve --size 4B --variant Q4_K_M
```

第二终端：

```bash
conda activate tp-agent
tp agent run --run demo-logic --task logic --model 替换为实际服务别名
tp agent status --run demo-logic
```

看到 `awaiting_approval` 时，选择其一：

```bash
tp agent resume --run demo-logic --decision approve
# 或拒绝并反馈
tp agent resume --run demo-logic --decision reject --feedback '先读取完整函数再修改'
# 或补充指令；原补丁不执行
tp agent resume --run demo-logic --decision feedback --feedback '需要同时处理平方数'
```

每次补丁都独立审批，直到finished或budget_exhausted。默认最多10次模型调用，包括错误请求。模型自己说“完成”不代表通过验收：

```bash
tp agent verify --run demo-logic
```

另外两个任务：`--task boundary` 和 `--task feature`，使用新的运行ID。验收测试不会发送给Agent，验收时恢复原始公开测试并加入隐藏测试。

### 演示checkpoint恢复

在等待审批时退出命令/关闭终端，然后新开终端执行 `tp agent status` 与 `tp agent resume --decision approve`。在工具调用期间中断后，用不带decision的 `tp agent resume --run demo-logic` 恢复。

SQLite保存图状态，工作目录和操作日志分别持久化。写补丁前检查原文件哈希，操作日志支持崩溃后识别已完成写入；外部修改会导致旧补丁被拒绝。HTTP请求的服务端推理状态本身无法恢复，已预留的调用计入预算。

框架对比示例及独立环境安装见 [FRAMEWORKS.md](docs/FRAMEWORKS.md)。这些是机制与体验对比，不是三框架严格等条件性能排名。

## 5. 汇总和回传

```bash
tp summary
tp export --output /root/autodl-tmp/trainingprogram-results.zip
```

将这个ZIP返回即可。导出包含逐题原始输出、配置、环境版本、判题结果、速度记录、Agent日志与工作目录、汇总CSV和报告骨架；排除权重、缓存、SQLite数据库、环境文件和常见密钥。SQLite仍留在服务器供恢复；导出包用于分析和外部判题，不用于恢复Agent会话。

`summary/REPORT.md`中未完成项显示“待完成”，不会虚构成绩。pass@1只在全部选定题完成有效判题后生成。同配置BF16和Q4_K_M配对时输出差值及逐题回归/改善列表。

## 本地开发

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements/agent.lock
pip install --no-deps -e .
python -m pytest -q
```

核心机制测试不下载模型、不调用外部模型API、不执行生成代码。GPU、量化、框架示例和真实容器判题另行验收，不与模拟测试混为一谈。
