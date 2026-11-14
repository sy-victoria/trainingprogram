# 隔离判题与 AutoDL 兼容性

代码生成不需要 Docker；执行生成代码需要 Docker。项目不会在 Docker 不可用时回退到宿主机 `exec` 或普通 subprocess 判题。

容器采用：禁网、只读根目录、非 root 用户、移除 capabilities、禁止提权、2 CPU、2GiB 内存、128 PID、临时目录限额、输出截断和墙钟超时。Agent 的工作目录只读挂载，模型不能改变测试命令。正式验收重新复制原始测试，并加入 Agent 未见过的测试。

## AutoDL 内可以使用 Docker

```bash
docker info
bash scripts/build_sandbox.sh
tp doctor
tp eval judge --run qwen35-0.8B-BF16-smoke
```

镜像使用固定 Python 标签、EvalPlus 提交和核心执行依赖。每次判题记录实际镜像 ID；同一运行更换镜像时拒绝混合结果。需要跨机器完全一致时用 `docker save` / `docker load` 传递同一个镜像。

## AutoDL 不允许 Docker

不用在租用容器里强行启动特权 Docker。先完成六组生成，随后：

1. 执行 `tp export --output /root/autodl-tmp/generation.zip`。
2. 将结果包复制到支持 Linux Docker 的电脑或服务器，克隆同一仓库提交，安装 `requirements/agent.lock` 并 `pip install --no-deps -e .`。
3. 将可信结果包解压到新目录，将 `TP_DATA` 设置为该目录的绝对路径。
4. 构建本项目判题镜像，逐个执行 `tp eval judge --run 原运行ID`，无需下载模型。
5. 执行 `tp summary` 和 `tp export` 生成包含判题结果的新包。

对 Agent：没有 Docker 时仍能演示读文件、审批、写补丁、checkpoint 恢复；`run_tests` 会返回隔离环境不可用。模型不得声称测试通过。导出工作目录后，在 Docker 主机执行 `tp agent verify --run 原任务ID` 获取独立验收结果。此时报告应注明：测试反馈闭环未在 AutoDL 原机完成。

不要把 EvalPlus 自带的 Python reliability_guard 当作完整操作系统沙箱；本项目同时使用 Docker 隔离。项目不是用于运行蓄意恶意代码的多租户服务。
