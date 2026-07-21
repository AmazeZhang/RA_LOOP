# RA-LOOP Spatial fixed-L2 学习信号验证 — 20260720

> 启动：2026-07-20 23:43 CST  
> 完成：2026-07-20 23:55:40 CST（UTC+08:00）  
> 结论：GPU learning-signal gate 通过；本次不保存 checkpoint，不能作为性能提升证据。

## 配置

- 物理 GPU 7，单 GPU、单 in-process 环境
- 单个 LIBERO Spatial 任务，4 demos，固定 dataloader 顺序
- 1 training step，K=8（4 anchor + 4 paired Robot-init）
- horizon=220，RIPT scale factor=5
- Robot-init `fixed_l2=0.1 rad`，seed=20260720
- recovery-only，`lambda_recovery=0.5`，无 action consistency
- 1 PPO epoch，8 个 PPO batch
- W&B、periodic eval、checkpoint 保存均关闭

## 结果

- 8 条 rollout：`[1, 0, 1, 0, 0, 1, 1, 0]`
- 总成功率：4/8 = 50%
- anchor：3/4 = 75%
- perturbed：1/4 = 25%
- `mean_R_success=0.5`
- `mean_R_recovery=0.125`
- `mean_R_total=0.5625`
- `lambda_r_effective=0.5`
- `non_zero_adv_ratio=1.0`
- `pg_loss_stats=0.00264756`
- `gradient_norm_model_stats=1.96533`
- `gradient_norm_header_stats=93.5`（裁剪前统计；scale head 固定）
- `pg_clipfrac_stats=0.42708`
- rollout：622.39 秒
- PPO：60.23 秒
- 单训练步总计：683.00 秒（11分23秒）

日志出现 `[info] Finished training`，tmux 捕获到 `[RA_LOOP_EXIT] 0`。结束后 GPU 7
回落到 18 MiB、0%、41°C。输出目录只有空目录层级，没有 checkpoint 或其他文件，
符合关闭保存的配置。

## 判定

exactly-once rollout/reward/PPO 链路在真实 h220/K8 下通过；anchor 与 fixed-L2
Robot-init 产生明显成功率差异，reward、leave-one-out advantage、loss 和模型梯度
均为非零，因此 RA-LOOP 已具备进入短程、保存 checkpoint 的 pilot 训练条件。

`pg_clipfrac=0.427` 较高，模型裁剪前 gradient norm 1.97 也超过 1.0 阈值。短程
pilot 不应直接扩大为长训练；应先使用保守学习率/步数、定期保存，并在少量步骤后
同时检查原始 anchor 与 fixed-L2 Robot-init，避免过强更新和标准能力退化。

## 证据

- 日志：`logs/ra_loop_spatial_learning_probe_20260720/run1.log`
- 启动器：`train/ra_loop_spatial_learning_probe.sh`
- tmux：`ra_loop_spatial_learning_probe:run1`
