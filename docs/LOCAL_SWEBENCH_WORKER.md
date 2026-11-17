# 本地 SWE-bench worker 与远程 GPU 推理

用户已指定服务器不能安装 Docker，改用本地 Windows Docker Desktop 的 Linux 引擎。服务器仅负责模型权重、量化和推理；本地运行 Agent 控制器、仓库实例容器及官方判题。Agent 通过 SSH 本地端口转发访问服务器环回模型服务，不公开服务端口。单纯下载最终补丁回本地判题不足以支持完整 Agent：Agent 迭代期间的读取、命令和测试也必须在本地对应实例中进行。

2026-11-17 实测：Docker Engine 29.8.2、linux/amd64；12 CPU、Docker 可用内存 16513753088 bytes；主机约 32GiB 内存；D盘约剩 504GiB。Docker Hub hello-world 镜像拉取成功。ubuntu:24.04 实际运行成功，cgroup 校验得到 memory.max=1073741824、pids.max=64、cpu.max=200000 100000。上述仅证明基础容器功能，不代表 SWE-bench 实例已经通过。

## 后续验收

已锁定官方源码 `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`（SWE-bench 5.0.2），控制器成功构建并运行 CLI。对应数据为 `SWE-bench/SWE-bench_Verified` revision `78f471bf655a3137b2e8a75af1501690ec009ec3`，500 个实例，导出 JSON SHA256 为 `c5de877fa0a088059038ffef37ba9c24f2329b3e114c25ecc7e1ef559c6dd93d`。旧 `princeton-nlp` 数据缺少新版 harness 所需字段，首次预检失败已保留，不能混用两个版本。

`scripts/cache_swebench.py` 会验证数量、唯一 ID 和新版必需字段；完整含 gold 的数据只挂载给可信 harness，不能挂入 Agent 实例。构建入口为 `scripts/build_swebench_worker.ps1`，依赖记录在 `requirements/swebench-worker-resolved.txt`。若 Windows 禁止执行 ps1，直接按脚本执行 git archive、Copy-Item、docker build，不修改系统执行策略。

单实例 `django__django-10914` 已通过正反对照：`gold-preflight-fixed` 标准补丁 resolved=1；`negative-preflight-v1` 能应用但只增加无关文本文件的补丁 resolved=0。两次均 completed=1、error=0、infra_failure=0、unstopped_containers=0。实例镜像和控制器标识见 `configs/swebench-lock.json`。这只验证环境，不计作模型或 Agent 成绩。初次运行因控制器未包含 Docker CLI 导致清理失败，已补齐官方 CLI 后重跑，旧日志保留。

1. 锁定官方 SWE-bench harness revision、Verified 数据 revision 和实例镜像 digest。
2. 为 harness 准备独立 Linux 环境；Agent 与受测代码进程分离，Docker socket 仅由可信控制器使用，不挂入受测仓库容器。
3. 运行单个官方 gold patch 验证 FAIL_TO_PASS/PASS_TO_PASS，再验证已知错误补丁确实失败。
4. 通过 SSH 隧道验证服务器模型别名与权重身份，跑一个完整 Agent issue→工具→patch→判题轨迹。
5. 完成三个框架相同协议的机制验收后，启动 500 题正式任务。

初期并发 1。按实例需求测量内存，资源不足记基础设施错误并解决，不判作模型失败。按批次缓存镜像，清理仅限本项目管理的缓存与容器，不能执行全局 docker prune。Docker Desktop 的磁盘镜像可能位于 C 盘，D 盘剩余空间不等于 Docker 可用存储；扩大缓存前核实其物理位置和限额。

本地机器需要保持开机、Docker运行、SSH可达。断开后保留任务状态与原始日志，恢复不得静默重新采样。正式分数中不加入人工或代理解题反馈；审批演示另列。
