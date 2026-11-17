# 完整版作业实施计划

状态：2026-11-16 重新规划，未完成的新范围不能用先导实验代替。

## 目标与范围

完成完整 HumanEval+、完整 MBPP+、SWE-bench Verified 全部实例，三套完整 Agent（LangGraph、Qwen-Agent、Deep Agents），多量化方法、多随机种子。现有 30 题评测与三个自建修复任务仅作为先导证据。SWE-bench Verified 与 SWE-bench 原始完整 test split 分开命名，不把 Verified 称为原始全集。

用户已明确最大做到 27B。正式模型范围为 Qwen3.5 的 0.8B、2B、4B、9B、27B，以及 Qwen3.8-27B，共六个模型。35B 及以上和超大 MoE 不再属于此次验收范围。下载前逐个核对模型卡、架构、原始精度、权重许可、revision 与后端支持。完整基准、多方法、多种子和三套完整 Agent 不因模型上限调整而缩减。机器可读实验清单见 configs/full-study.json，当前标记 development_not_frozen，不能当成正式结果。

## A. 原始模型与量化评测

1. 使用 EvalPlus 官方完整 HumanEval+（164）及 MBPP+（当前版本预期 378），锁定数据 revision、哈希与题目清单；数量不符时停止并核对版本。
2. 精度/方法：原精度、GGUF Q8_0、GGUF Q4_K_M、GPTQ W4A16、AWQ W4A16。Q8/Q4 为同类量化的不同配置，GPTQ/AWQ 为不同方法。各方法需要在目标架构实际量化并验证，不以下载他人量化文件代替全部量化过程。
3. 每个后端单独配置原精度基线：llama.cpp 原精度/Q8/Q4；经兼容性验证并锁定的另一后端原精度/GPTQ/AWQ，共预计六个运行配置。若 GPTQ 与 AWQ 不能共享后端，增加相应基线；禁止把跨后端差异全部归因于量化。
4. 量化校准数据与测试题、测试答案严格分离，锁定校准数据哈希、种子、长度、样本数和参数。量化失败或架构不支持记为待解决项，不静默换方法或精度。超大模型若官方原精度不是 BF16，按模型事实保留原精度，不伪称 BF16 原版。
5. 每个配置运行一遍 greedy 基线，再运行 temperature=0.2、top_p=0.95、seeds=[2026,2027,2028] 三次采样。每题每种子一个候选，不择优；随机采样的三次 pass@1 均值、标准差及配对 bootstrap 区间单独报告，不把三个种子的最好答案称为 pass@1。temperature=0 的换种子重复不作为随机性实验。
6. 统一文本任务模板与 thinking 设置，输出预算初定 4096 token；在非测试开发集完成兼容性确认后冻结。不得根据正式测试成绩调整提示或重新挑样。保存终止原因、格式错误、截断、逐题耗时与资源；不可恢复的请求不偷偷重采样。
7. 完整报告 base/plus 判题结果、失败类型、量化差值、逐题回归/改善、磁盘、RSS、显存、TTFT、吞吐与总耗时。性能测试固定输入和输出预算，按长度分层，至少 10 次重复；记录其他 GPU 负载。共享卡上的结果不冒充独占基准。

按 6 模型、6 配置、542 题和 4 次运行计算，模型评测预计至少 78048 个候选；后端拆分或数据版本变化后重新计算，不省略模型来压缩总量。

## B. 三套完整 Coding/SWE Agent

复现参照选 mini-SWE-agent 的 issue→观察→动作→反馈→patch 基本循环，冻结源 revision，逐项记录复现内容和偏差；不声称复现其公开排行榜分数。

三套均实现任务初始化、仓库检索、读取、补丁、受限命令/测试、最终 diff、持久化、暂停恢复、审批/拒绝/反馈、预算、超时、独立日志。Qwen-Agent 不再只是只读示例；Deep Agents 不只展示工具调用。持久化可由应用层实现，但报告区分框架原生与额外适配代码。

统一 ACI 接口和隔离执行器，三套分别使用各自原生调度；不能只更换名称而复用同一个 LangGraph 调度器。工具返回截断、工作目录、命令权限、模型、上下文长度、token/调用/时间预算一致。工具序列化不可完全一致时记录差异。

主对照禁用额外委派和未对齐的工具。若 Deep Agents 内置能力无法关闭，则清楚披露并另列原生能力实验，不声称严格控制变量。三套都运行恢复、重复补丁、文件变更、审批拒绝、越界、超时和预算的相同机制测试。

正式自主实验使用事先固定的权限策略，不由助手提示修复内容；拒绝危险动作可记录，但不提供解题反馈。代理审批辅助实验另列，不混入自主 resolved rate。现有三个任务保留作为回归与审批演示。

## C. SWE-bench

默认主基准为 SWE-bench Verified 完整 500 题，不以 Lite 或抽样代替；锁定官方 dataset 与 harness revision。原始 SWE-bench 全量 test split 是另一数据集，若纳入则独立实验和分母，不能混称。

主矩阵：3 框架 × 2 模型（Qwen3.5-9B、Qwen3.8-27B 的原精度）× 3 seeds × 500 instances = 9000 次 issue 修复。模型固定后不因低分切换。初定每实例 100 次模型调用、累计 100000 输入+输出 token、30 分钟 Agent 时间上限，独立判题另计；先在非正式开发集校准可运行性再冻结。预算耗尽保留 patch 并按规则评测，不追加预算救成绩。

使用官方容器化环境，先 gold patch 验证环境，再做预测补丁评估；Agent 无权读取 gold patch、隐藏补丁或最终测试答案。每次还原仓库和环境。按官方 FAIL_TO_PASS/PASS_TO_PASS 判断 resolved，不能只看模型自称成功、patch 能应用或自写测试通过。

环境失败与模型失败分开，列出全部预定实例的状态，不删除难题缩小分母；仅环境失败可按固定重试规则复跑，不重新生成已有有效 patch。报告每种子 resolved rate、置信区间、步数、token、耗时、patch/测试失败类型及轨迹。

服务器无法安装 Docker，用户已指定使用本地机器。2026-11-17 本地 Docker Desktop 已实测镜像拉取、Linux容器与资源限制可用。正式方案为服务器 GPU 推理、本地 Agent/仓库工具/官方判题，经 SSH 隧道连接；见 LOCAL_SWEBENCH_WORKER.md。仍须完成官方实例 gold patch 验证，不能以基础容器成功宣称 SWE-bench 判题完成。

## D. 资源与实施顺序

源码继续放 /home/sunyue/trainingprogram，大型产物放 /backup/trainingprogram-data，不创建 /backup/sunyue。6GiB 为选卡最低门槛，模型实际所需显存、KV cache 与余量仍必须满足；多卡推理需一次锁定完整卡组，不侵占其他任务显存。

27B BF16 仅权重约 54GB（十进制估算），需多卡或更大显存，并预留上下文和运行时空间。外部 API 可用于补充对照，但不能代替自量化与原精度权重对照。购买/租赁新资源须另获授权。当前 /backup 约剩 497GiB，SWE-bench 镜像按批次缓存并保留可重建元数据，不能假设现有空间足够。

实施里程碑：

1. 冻结旧实验、导出证据；建立模型/基准/框架/精度/种子的完整实验清单与状态表。
2. 完成架构兼容性与资源盘点；解决 SWE-bench worker；验证三个量化路线，锁定版本。
3. 实现 MBPP+、多种子、跨后端原精度基线、校准与统计；测试隔离与续跑。
4. 实现三套完整 Agent，统一工具与预算；用同一开发集完成机制验收，冻结正式配置。
5. 逐规模运行完整 HumanEval+/MBPP+；量化或资源失败保留日志，修复后按新配置运行。
6. 运行完整 SWE-bench Verified 三框架矩阵；失败不删样本；同时生成可审计结果。
7. 完成统计、消融/失败分析、框架工程差异、复现偏差、源码、结果包、报告和演示。

先导计时后再估算工期：9000 个 SWE 轨迹若全部达到 30 分钟上限，对应 4500 轨迹小时，尚未计入环境构建/判题；不是准确工期。根据实测均值、P95、卡组与 worker 并发计算完成时间，不能承诺旧版几小时内完成。

## 完成标准

所有正式配置均有锁定 manifest；完整题集无漏项；每种子独立结果齐全；三套完整 Agent 的 checkpoint/审批/工具机制和真实仓库评测完成；每个量化方法有校准/转换证据与匹配原精度基线；原始轨迹、统计脚本、汇总与报告可复算；结果包脱敏且排除权重和缓存。资源不足、不支持方法、未完成大模型均显式标注，不能在有这些欠项时称为全范围完成。

## 官方依据

- https://github.com/evalplus/evalplus
- https://www.swebench.com/SWE-bench/guides/evaluation/
- https://www.swebench.com/SWE-bench/guides/docker_setup/
- https://huggingface.co/Qwen/models
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://github.com/QwenLM/Qwen-Agent
- https://github.com/langchain-ai/deepagents
- https://github.com/SWE-agent/mini-swe-agent
