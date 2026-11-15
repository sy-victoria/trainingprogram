# Coding Agent 与 Qwen3.5 量化实验

面向三天课程作业：**LangGraph coding agent + 自行量化 + 可复现评测**。目标机器为 AutoDL RTX 4090 24GB、90GiB 内存、25 CPU配额、Ubuntu 20.04、50GB数据盘。

本次覆盖 Qwen3.5-0.8B／2B／4B 的 BF16 文本 GGUF 与 Q4_K_M。Qwen3.8、9B、27B及更大规模留待后续，**不代表原作业全部完成**。仓库不包含实验成绩、模型权重或密钥。

## 没有算力时：准备与离线自检

无卡模式只需要轻量Python环境，不下载权重、不安装CUDA工具链、不编译llama.cpp：

```bash
cd /root/trainingprogram
git -c http.version=HTTP/1.1 pull --ff-only && bash scripts/bootstrap.sh --agent-only
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tp-agent
tp check --stage agent
tp selftest
python -m pytest -q
```

`tp selftest` 演示读取文件、审批暂停、重新连接SQLite后的恢复、幂等补丁和评测续跑。它使用确定性模拟响应，不调用真实模型、不执行生成代码，数据仅写入 `$TP_DATA/offline/`，不计入成绩或普通导出包。2GiB/0.5CPU的无卡模式可以做这些检查；安装速度可能较慢。

## 算力恢复后：统一实验入口

先完成后文的GPU环境安装和0.8B冒烟检查，再用一个命令运行全部六组正式实验：

```bash
bash scripts/bootstrap.sh --all
conda activate tp-agent
tp check --stage gpu --size 4B
tp experiment run --id formal-v1 --mode subset --judge defer --cleanup-completed
```

- 默认顺序为0.8B、2B、4B，每个规模运行BF16与Q4_K_M。`--sizes 0.8B`可以只跑一个规模。
- `--judge defer`明确只生成/测速；如果Docker可用且希望一起判题，改用`--judge required`，会在GPU工作开始前检查隔离环境。
- `--cleanup-completed`是显式清理授权：一个规模两种精度完成后清理它的原权重和GGUF，保留manifest及结果；最后一个规模保留给Agent演示。不传此参数则保留所有文件，空间不足会停止。
- 中断后重跑完全相同命令；已完成的组跳过，不重新下载已清理的小模型。配置或代码变化需用新ID。
- 各组ID为`formal-v1-0.8B-BF16-subset`等，阶段状态保存在`experiments/formal-v1/status.json`。测速也逐请求保存；中断的测速请求标记缺失，不偷偷重测。

有Docker后统一判题，无需再次启动模型：

```bash
tp experiment judge --id formal-v1
tp summary
tp export --output /root/autodl-tmp/formal-v1-results.zip
```

统一入口的阶段结束不等于每题都成功；真实成绩以summary中的有效判题数、错误类型和完整性为准。完整验收清单见 [ACCEPTANCE.md](docs/ACCEPTANCE.md)。

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

安装脚本默认显式使用官方 `https://pypi.org/simple`，避免AutoDL预设镜像尚未同步锁定版本。不修改全局pip配置；确需使用其他已同步的源时，可设置 `TP_PIP_INDEX_URL`。CPU版PyTorch仍使用其官方CPU轮子源。

若此前出现 `No matching distribution found for langgraph==1.2.22`，随后出现 `tp: command not found`，说明依赖安装中断，并非环境没有创建。执行 `git pull --ff-only` 后重跑 `bash scripts/bootstrap.sh` 即可复用已有Conda环境。等脚本输出 `Environment ready` 再继续；无需删除环境或降低锁定版本。

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

也可以在4B服务运行时交互完成三个任务：

```bash
tp agent suite --run agent-v1 --model 替换为实际服务别名 --judge defer
```

每个补丁都显示后询问`a`批准、`r`拒绝、`f`补充指令或`q`暂停。再次执行同一命令可继续；不会自动批准。若Docker可用，可从开始时选择`--judge required`自动做每个任务的独立验收。延后判题时执行：

```bash
tp agent verify --run agent-v1-logic
tp agent verify --run agent-v1-boundary
tp agent verify --run agent-v1-feature
tp summary
```

报告会单独汇总Agent状态、模型调用数、审批记录数量和独立验收结果。

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
