# RA-LOOP mode-stratified 四任务长训结果 — 20260722

> 启动：2026-07-21 22:32 CST
> 完成：2026-07-22 04:28 CST
> 状态：35/35 updates、`Finished training`、exit 0；GPU 已释放。

## 完成状态

- 训练循环：5小时55分03秒
- 35 steps × K8 = 280 stochastic rollouts
- 总成功：193/280 = 68.93%
- anchor：95/140 = 67.86%
- fixed-L2：98/140 = 70.00%
- `advantage_mode_stratified=1.0`：35/35 steps
- 28/35 steps 有非零 advantage；7 个组内结果完全一致的 step 按设计零更新
- 无 traceback/OOM，`RA_LOOP_EXIT=0`
- GPU 7 完成后：18 MiB、0%、41°C

## 分任务在线采样

| task | sampled steps | total | anchor | fixed-L2 |
|---|---:|---:|---:|---:|
| between plate/ramekin | 9 | 58/72 = 80.56% | 29/36 | 29/36 |
| original next-to-plate | 10 | 55/80 = 68.75% | 27/40 | 28/40 |
| on stove | 7 | 45/56 = 80.36% | 21/28 | 24/28 |
| top drawer | 9 | 35/72 = 48.61% | 18/36 | 17/36 |

旧 mixed-objective run 在线为 188/280（anchor 101、perturbed 87），本次为 193/280
（anchor 95、perturbed 98）。这个方向与预期相符，但两次运行受动态跳过和 task/init
采样差异影响，不能把差值解释成真实性能变化，也不能据此选择 checkpoint。

## 优化稳定性

| metric | mean | min | max | first 5 | last 5 |
|---|---:|---:|---:|---:|---:|
| success | 0.6893 | 0.2500 | 1.0000 | 0.7500 | 0.6500 |
| anchor | 0.6786 | 0.2500 | 1.0000 | 0.8000 | 0.6000 |
| perturbed | 0.7000 | 0.2500 | 1.0000 | 0.7000 | 0.7000 |
| pg clipfrac | 0.0535 | 0.0000 | 0.1608 | 0.0744 | 0.0474 |
| pg ratio | 1.0004 | 0.9801 | 1.0339 | 1.0060 | 1.0079 |
| model grad norm（裁剪前） | 2.3310 | 0.0000 | 7.4036 | 3.3603 | 2.9077 |
| header grad norm（裁剪前） | 47.0375 | 0.0000 | 88.7969 | 56.4125 | 51.5406 |

所有 step 的 mean anchor advantage 都在 `0--5.56e-17`，mean perturbed advantage
均为 0，符合 mode-stratified RLOO 的组内零和不变量。七个零梯度 step 是两个模式各自
reward 完全一致时的正确行为；旧 objective 在这种情况下仍制造跨模式更新，正是本次
修复的目标。PPO ratio、clipfrac 和 gradient 未显示数值发散。

## Checkpoint 验收

产生 step 5/10/15/20/25/30 六份 checkpoint。每份均通过 CPU tensor 级读取：

- `adapter_model.safetensors`：484,458,600 bytes，879 tensors
- `openvla_headers.pt`：638,046,816 bytes，含 action/scale headers
- `adapter_config.json`：LoRA rank 32
- 无 `.incomplete`

保存语义与旧 run 相同：step 5/10/.../30 对应完成第 6/11/.../31 次 update；最终四次
update 未保存，不存在 step 35。因此独立评测只使用六份已落盘 checkpoint。

## 下一步

使用与训练前 baseline 和旧 mixed-objective run 完全相同的四任务 paired evaluator，
分别评测六份 checkpoint 的 6 anchor + 6 fixed-L2 0.1，共 288 episodes。主要判据：

1. anchor 是否恢复到 baseline 的 24/24；
2. fixed-L2 是否超过 baseline 21/24 和旧训练最佳 21/24；
3. drawer 的两个固定失败和 stove 的一个固定失败是否被修复；
4. paired gain/loss 是否证明修复不是由新的 anchor regression 换来的。

在该独立评测完成前，不声称 mode-stratified 方法提升性能。

日志：`logs/ra_loop_spatial_stratified_multitask_20260721/run1.log`。
