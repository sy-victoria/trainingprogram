# 0.8B smoke 实测核对（2026-11-16）

运行：`cloud-smoke-v2`，服务器 `sunyue@10.82.1.226`，GPU 1（RTX 4090）。
原始数据在 `/backup/trainingprogram-data/runs/cloud-smoke-v2-*`；本地副本位于仓库外的 `trainingprogram-smoke-results`，不提交模型或真实运行产物。

## 已验证

- llama.cpp 固定提交 `32bdc3a89e6528d3e419024c833c5dabf32c7541` 完成 CUDA 编译。
- Qwen3.5-0.8B 原权重 SHA256 为 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`，分段下载后合并验证通过。
- BF16 转换、Q4_K_M 量化及两组服务启动成功。
- 两组均生成 HumanEval/102、HumanEval/134，并完成独立测速。
- 逐字段比对两组 config.json：model_revision、llama_cpp、evalplus、dataset_sha256、task_ids、generation、server、harness_hash、hardware 完全一致。
- 固定生成设置为 temperature=0、seed=2026、max_tokens=2048、thinking=false；上下文 8192，线程 8，并行槽位 1。

## 观察及解释边界

|配置|正常结束|截断|代码提取错误|
|---|---:|---:|---:|
|BF16|2/2|0/2|0/2|
|Q4_K_M|0/2|2/2|2/2（syntax_error）|

Q4_K_M 的原始回答含很长的 Python 注释，达到输出预算后以 length 结束。不能据此区分量化误差、模型对该设置的敏感性或推理实现兼容性，也不能将两题外推为完整性能结论。BF16 可解析不代表答对。

当前尚无隔离判题成绩，不能填写 pass@1。原始记录保留，不通过增大 token 预算或重采样覆盖失败。任何诊断变更应使用独立运行 ID；正式实验仍需六组共同设置、30 题固定题集和隔离判题。

## 尚待完成

- 构建并实测隔离判题环境，得到 smoke 正确性结果。
- 独立诊断量化版长输出，保留与固定基准设置的区别。
- 六组正式生成、判题、资源与速度汇总。
- 三个真实 Agent 修复任务、人工审批与恢复演示、独立验收。
- 两个其他框架示例的真实调用比较。
- 导出完整结果包和基于实测的中文报告。
