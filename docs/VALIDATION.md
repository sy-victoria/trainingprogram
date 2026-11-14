# 验证记录

## 本地已完成

- Windows、Python 3.11.9，安装并检查 `requirements/agent.lock`，`pip check` 无冲突。
- 项目editable安装成功，`python -m trainingprogram.cli --help` 正常。当前Windows应用控制策略阻止新生成的 `tp.exe`，因此本机通过Python模块入口验证；Linux `tp`入口由CI和服务器检查。
- `python -m pytest -q`：30 passed，3 skipped。跳过的是需要真实Docker引擎的三个集成测试。
- 覆盖审批与拒绝、重建图后的SQLite恢复、请求中断预算、过期补丁拒绝、操作日志幂等、路径越界拦截、测试文件保护、错误工具反馈。
- 覆盖真实本地HTTP SSE模拟服务、逐题保存与续跑、配置改变拒绝、代码提取、判题基础设施错误重试、pass@1分母、配对差值、导出排除与脱敏、权重dtype头读取和定向清理。
- Python源码编译检查、逐文件Bash语法检查通过。
- Qwen-Agent和Deep Agents分别在独立环境完成依赖安装、帮助入口与Agent对象构造检查；这些检查没有调用真实模型。已补齐Qwen-Agent导入需要但未声明的numpy、soundfile、tqdm、python-dateutil，两个示例环境的完整解析版本均已保存。

## CI配置

GitHub Actions包含Linux/Windows单元测试，以及Linux真实Docker测试：官方判题函数的正确/错误代码、无限循环超时、禁网与只读挂载。CI配置存在不代表它已经通过；以仓库Actions状态为准。

## 仍需服务器验证

- Conda CUDA 12.8工具链安装、固定llama.cpp提交在4090上编译。
- 官方模型下载、BF16转换、Q4_K_M量化与GPU推理。已核对上游转换参数和服务参数，但未在本机下载权重或运行GPU实验。
- AutoDL能否启动受限Docker容器。不可用时按SANDBOX.md转移判题。
- 真实模型的Agent成功率、工具格式遵循、审批恢复演示，以及六组评测成绩和资源占用。
- 两个对比框架的真实模型工具调用效果。

软件机制测试、基础设施验证、模型能力评测是三类不同证据。仓库不填入任何模拟生成的“模型成绩”。
