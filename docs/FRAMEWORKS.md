# 框架对比记录

主实现是 LangGraph。两个对比示例只读取同一个 `tasks/logic/workspace/solution.py` 并建议修复，不执行修改。

| 维度 | LangGraph 主实现 | Qwen-Agent 示例 | Deep Agents 示例 |
|---|---|---|---|
| 抽象层次 | 显式状态图与条件边 | Assistant 与工具注册 | 较高层 Agent，内置中间件和规划能力 |
| 本项目工具协议 | JSON action，经验证后调度 | BaseTool / function calling | Python 工具 / LangChain 工具调用 |
| 持久化 | SQLite checkpoint，应用另存工作目录 | 本示例没有实现恢复 | 基于其底层 LangGraph 能力，本示例没有配置持久化 |
| 人工审批 | 独立 interrupt 节点 | 本示例没有实现 | 本示例没有配置 |
| 工程取舍 | 状态、预算和恢复行为清晰，需要更多应用代码 | 快速连接 Qwen 与工具；需自己验证应用级恢复 | 起步功能丰富，默认中间件也影响行为 |

Deep Agents 并非与 LangGraph 完全平行的独立底层框架，不能把两者的表现简单归因为“框架优劣”。

## 运行小示例

保持 4B 服务运行。分别创建独立环境，避免与主项目锁定依赖冲突：

```bash
conda create -y -n tp-qwen python=3.11 pip
conda run -n tp-qwen pip install -r requirements/qwen-example.txt
conda run -n tp-qwen python examples/qwen_agent_demo.py --model 实际服务别名

conda create -y -n tp-deep python=3.11 pip
conda run -n tp-deep pip install -r requirements/deepagents-example.txt
conda run -n tp-deep python examples/deepagents_demo.py --model 实际服务别名
```

示例依赖文件保存了独立环境的完整解析版本，已验证导入与Agent对象构造。主项目不依赖它们。2026-11-16 已在服务器上连接相同的 4B BF16 完成两个示例的真实工具调用：均成功读取文件并指出平方根边界错误，正常退出且无超时。原始输出保存在结果包 `diagnostics/framework-comparison/frameworks-4b-v1/`，见[实验记录](EXPERIMENT_RESULTS.md)。这不代表完整修复流程的成功率。

Deep Agents 默认还提供虚拟文件系统等内置工具。本示例不配置宿主机文件系统后端或 shell，提示禁止委派；这不等于关闭它所有内置能力。因此只比较机制和开发体验，不声称这是严格同工具集合的性能对照。

实验表记录：版本、提示词、输出、是否实际调用读取工具、工具错误、总耗时、实现代码量和主观调试体验。Deep Agents使用read_repository_file这个名称避免覆盖框架内置read_file。没有运行的数据填“未验证”。

参考：[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)、[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)、[Deep Agents](https://github.com/langchain-ai/deepagents)。
