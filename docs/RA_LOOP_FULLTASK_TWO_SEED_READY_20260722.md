# RA-LOOP Spatial 全任务双 seed 长训准备 — 20260722

> 状态：训练前准备，尚未启动。

## 实验目的

四任务 lambda 消融没有 checkpoint 超过 warm-start baseline，说明单独调恢复奖励权重
不足。当前最直接的剩余假设是训练覆盖不足：此前只有 4/10 tasks、每任务 4 demos。
本轮保持唯一显示恢复行为的 mode-stratified lambda=0.5，不再改变目标函数，只扩大
task/demo/init 覆盖。

## 配置

| run | GPU | train seed | perturb base seed |
|---|---:|---:|---:|
| seed10000 | 7 | 10000 | 20260720 |
| seed20000 | 6 | 20000 | 20270720 |

共同配置：

- 10/10 LIBERO Spatial tasks；
- 每任务 50 demos，完整 500-demo 池参与 shuffle；
- 100 updates，约 10 updates/task，与旧四任务 35/4≈8.75 updates/task 匹配；
- 每 update 为 K=8 的 4 anchor/perturbed pairs，预计每 run 800 rollouts；
- fixed-L2 0.1 rad、mode-stratified、lambda recovery 0.5；
- 同一 pilot step-5 warm-start、LR 1e-5、horizon 220，每 10 step 保存；
- W&B 与周期评测关闭，输出和日志完全分离。

两个 seed 改变 DataLoader shuffle 与 perturbation seed，用于区分稳定覆盖收益和单次随机
幸运。按既有 35-step 六小时实测，预计每 run 17--22 小时，两组并行不增加墙钟。

## “全任务”与“全数据 epoch”

本轮是第一轮 full-task coverage，不宣称遍历全部 500 demos。100 updates 从完整池中
无放回 shuffle 采样约 20%，预计每任务约 10 个不同 demo。完整 500-update epoch 单卡
预计 80--90 小时；只有本轮独立评测出现净提升后才值得执行。

## 安全门禁

- 默认只打印命令；显式 `--run` 才创建 CUDA；
- 启动前同时检查 GPU 6/7、60 GB 可用磁盘和所有输出目录新鲜度；
- 每个训练器再次检查卡状态、20 GB 磁盘、模型、header、dataset 和 warm-start；
- 两组各自保存 exit marker，tmux 断开不终止训练。
