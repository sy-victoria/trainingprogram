# 完整交付与后续实验验收清单

本项目代码覆盖已约定的三天版本：一个完整LangGraph coding agent，两个框架的最小机制对比，Qwen3.5的0.8B/2B/4B量化实验。Qwen3.8和更大规模未纳入此版本，不能把扩展工作当作已经完成。

## 不需要GPU的验收

- [ ] `bash scripts/bootstrap.sh --agent-only`成功，原base环境保留。
- [ ] `tp check --stage agent`返回ready=true。
- [ ] `tp selftest`所有checks=true，且real_model_called=false。
- [ ] `python -m pytest -q`机制测试通过；无Docker时隔离测试显示跳过。
- [ ] 独立环境中的两个框架示例能够导入和构造Agent。

本地模拟只验证实现机制，不能填写模型成功率、量化精度或GPU速度。

## 算力到位后的顺序

1. **GPU环境**：有卡模式下运行bootstrap --all；check --stage gpu检查真实GPU、容器配额、工具文件及llama.cpp提交。
2. **冒烟**：0.8B两种精度各生成2题；确认服务启动、代码提取、日志与落盘。模型答错不是安装失败。
3. **六组正式生成**：固定30题，不改变模板和采样参数；运行过程中不要同时启动另一个GPU任务。
4. **测速**：每组五个提示、三轮测量；保留错误和截断记录，不用缺失值补0。
5. **隔离判题**：有Docker时本机判题，否则导出到Docker主机。所有题有有效判题后才形成pass@1。
6. **Agent**：三个任务分别运行，记录人类批准/拒绝；演示跨进程恢复；独立测试验收。
7. **框架对比**：相同文件任务下运行Qwen-Agent和Deep Agents示例，记录工具调用、输出和失败原因。
8. **回传**：summary及export；反馈ZIP和异常日志，随后补写实际分析和结论。

## 验收证据

| 能力 | 主要证据 |
|---|---|
| 依赖和工具链 | diagnostics中的环境、pip freeze和Conda explicit清单 |
| 模型来源与量化 | artifact.json中的revision、dtype、命令和文件哈希 |
| 固定题集 | 每个run的config.json和problems.json |
| 生成与续跑 | samples逐题记录，started请求不会重复采样 |
| 测速续跑 | speed目录的逐请求记录和speed.json |
| 隔离判题 | judge/config.json中的镜像ID、judgments逐题状态 |
| 实验进度 | experiments中的配置、当前组及错误状态 |
| Agent恢复和审批 | 服务器checkpoint.sqlite、approvals、journal、tools和status |
| Agent独立成功 | acceptance.json，而不是模型的finished声明 |
| 最终分析 | summary/results.csv、results.json、REPORT.md |

正常导出不包含模型权重、SQLite、离线模拟数据。实验用的公开题目会随结果导出以支持外部判题；没有发送隐藏测试给模型。

## 已知边界

- 无卡模式只能完成轻量准备与机制检查；模型转换/推理入口会在下载前阻止无GPU或配额过低的运行。
- 模型服务进程可以重新启动，但HTTP请求中的推理状态不能checkpoint恢复；中断请求不会暗中再采样。
- 微小子集成绩不具有完整榜单的统计意义，报告应保留分母与逐题差异。
- Windows开发环境与GitHub Docker CI不能替代AutoDL GPU、CUDA编译和实际权重兼容性验证。
