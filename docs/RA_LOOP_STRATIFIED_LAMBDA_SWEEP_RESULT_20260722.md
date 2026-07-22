# RA-LOOP mode-stratified 恢复权重消融训练结果 — 20260722

> 启动：2026-07-22 12:46 CST
> 完成：lambda=0 于 18:44，lambda=0.25 于 18:47
> 状态：两组 35/35、`Finished training`、`[RA_LOOP_EXIT] 0`，GPU 6/7 已释放。

## 训练验收

| 配置 | GPU | 训练循环 | 在线 anchor | 在线 perturbed | 在线总计 |
|---|---:|---:|---:|---:|---:|
| lambda=0 | 7 | 5:57:44 | 100/140 | 96/140 | 196/280 |
| lambda=0.25 | 6 | 6:00:48 | 98/140 | 89/140 | 187/280 |

两组均完整记录 35 个 metrics，35/35 使用 `mode_stratified`，其中各 32 步产生非零
advantage。lambda effective 分别全程为 0 和 0.25。

在线数字不能用于选择权重：策略行为会影响 dynamic skip/task 采样，两组实际任务步数
并不完全相同，例如 stove 分别为 8 和 6 steps。因此 196/280 与 187/280 不是 paired
比较，也不是独立泛化成绩。

## Checkpoint 验收

每组均保存 step 5/10/15/20/25/30 六份 checkpoint。十二份 checkpoint 均满足：

- `adapter_model.safetensors` 可读，包含 879 个 LoRA tensors；
- `openvla_headers.pt` 可用 weights-only/mmap 读取，包含 34 个 tensors；
- `adapter_config.json` 存在；
- 没有 `.incomplete` 或临时 checkpoint 文件。

## 下一步

对两组十二份 checkpoint 使用与 baseline、旧 mixed objective、lambda=0.5 完全相同的
四任务 paired evaluator：每 checkpoint 为 24 anchor + 24 fixed-L2，共 48 episodes。
只有该独立评测才能判断降低恢复奖励权重是否消除 original perturbation regression，
并决定全量训练使用 lambda=0、0.25，还是返回覆盖/regularization 设计。
