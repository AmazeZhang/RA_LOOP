# RA-LOOP Spatial 全任务 lambda 2×2 消融 — 20260722

> 状态：训练已完成。四组于 2026-07-23 16:11--16:43 CST 正常结束，均为
> 100/100、`Finished training`、exit 0；GPU 4--7 已释放。结果见
> `docs/RA_LOOP_FULLTASK_LAMBDA_ABLATION_RESULT_20260723.md`。

tmux 会话为 `ra_loop_fulltask_lambda_ablation`；窗口分别为
`lambda05_seed10000_gpu7`、`lambda05_seed20000_gpu6`、`lambda0_seed10000_gpu5`、
`lambda0_seed20000_gpu4`。启动前可用磁盘 212 GB，GPU 4--7 均为空闲状态。

## 设计

第一轮 full-task coverage 使用四个独立单卡 run：lambda recovery 取 0 与 0.5，训练
seed 取 10000 与 20000，形成 2×2 paired ablation。相同 seed 的两档 lambda 使用
相同 DataLoader shuffle seed 与 perturb base seed，便于把差异归因于恢复奖励权重。

| lambda | train seed | perturb seed | GPU |
|---:|---:|---:|---:|
| 0.5 | 10000 | 20260720 | 7 |
| 0.5 | 20000 | 20270720 | 6 |
| 0 | 10000 | 20260720 | 5 |
| 0 | 20000 | 20270720 | 4 |

共同配置为 10/10 Spatial tasks、50 demos/task、100 updates、K=8、horizon=220、
fixed-L2 0.1、mode-stratified、同一 pilot step-5 warm-start、LR 1e-5，每 10 step
保存。预计每 run 17--22 小时，四组并行墙钟不叠加。

本轮每个配置只有两个 seed，定位是 full-task gate，而不是最终统计结论。若某档 lambda
在相同 seed 下均取得净提升，再对获胜档补第 3 个 seed；若两档都没有提升，则优先加入
reference-policy/KL guard，不继续扩大相同训练。

按现有 checkpoint 实测，每 run 预计约 11--12 GB，四组约 45--48 GB。启动器要求
至少 80 GB 可用磁盘，并在创建任何 tmux/output 前统一检查 GPU 4--7；单训练器还会
再次执行卡、磁盘、模型、数据与输出新鲜度门禁。
