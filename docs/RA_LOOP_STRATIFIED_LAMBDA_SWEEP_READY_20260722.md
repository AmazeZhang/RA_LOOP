# RA-LOOP mode-stratified 恢复权重消融 — 20260722

> 状态：训练前准备；尚未启动。

## 动机

mode-stratified 的 lambda=0.5 训练恢复了 anchor 24/24，但最佳 checkpoint 只是把
stove 的一条 fixed-L2 失败换成 original 的一条失败，总分仍为训练前 baseline 的
45/48。相同步数继续延长没有改善依据，因此先隔离 perturbed group 的奖励权重。

## 下午长时实验

| 实验 | GPU | lambda recovery | 预计时长 |
|---|---:|---:|---:|
| stratified lambda0 | 7 | 0.0 | 约 6 小时 |
| stratified lambda025 | 6 | 0.25 | 约 6 小时 |

两组并行，其余条件与已完成的 lambda=0.5 实验一致：同一 step-5 warm-start、4 个
spatial tasks、35 steps、K=8、horizon=220、fixed-L2=0.1 rad、学习率 1e-5、固定
扰动 seed、每 5 step 保存一次。现有 lambda=0.5 结果作为第三组，不重复训练。

## 判定方式

训练完成后，对每组 checkpoints 使用同一套 paired anchor/fixed-L2 独立评测。若降低
lambda 后保留 stove 增益并消除 original 回归，说明主要问题是恢复项相对梯度过强；
若三档权重都只是交换失败，则下一步应增加 demos/init 覆盖或加入 reference-policy
guard，而不是继续增加相同步数。

## 安全措施

- 两组使用独立输出目录、日志和 tmux 窗口；
- 启动器先同时检查 GPU 7 和 GPU 6，任一不满足阈值便在创建会话前拒绝；
- 单组训练器还会再次检查目标 GPU、磁盘空间和输出目录新鲜度；
- 默认只打印计划，必须显式 `--run` 才会创建 CUDA 训练。
