# RA-LOOP Spatial overnight pilot — 20260721

> 状态：训练前检查完成，等待/准备启动。  
> 目标：在用户休息期间运行一个有边界、可留中间 checkpoint 的保守短程 pilot。

> **启动更新：2026-07-21 00:11 CST** — 经用户确认，pilot 已在物理 GPU 7 和
> tmux `ra_loop_spatial_overnight_pilot:run1` 启动。模型、scale header、4-demo
> dataset 和单环境初始化完成，已进入 training loop 的 step 1/21、episode 1/8。
> 模型报告 allocated 15.75 GB、reserved 17.69 GB，无立即错误。按先前单步实测，
> 预计约 04:10 完成，保守范围 03:45--05:15。当前尚未产生第一个 step-5
> checkpoint。

> **完成更新：2026-07-21 03:41:27 CST** — 21/21 steps 完整完成，训练循环
> 3小时29分25秒，`[info] Finished training`、`[RA_LOOP_EXIT] 0`。在线采样总成功
> 112/168，anchor 58/84、perturbed 54/84，21/21 步均有非零 advantage。step
> 5/10/15/20 四份 checkpoint 已保存并通过 CPU tensor 级可读性验证。GPU 7 已
> 释放。训练期指标不能替代独立评测；下一步评测四个 checkpoint 后选型。详细
> 结果和 SHA-256 见 `docs/RA_LOOP_OVERNIGHT_PILOT_RESULT_20260721.md`。

## 配置

- 单任务：`pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- 单 GPU、单 in-process 环境
- 21 training steps
- 每步 K=8：4 anchor + 4 paired Robot-init rollout
- horizon=220，`fixed_l2=0.1 rad`，seed=20260720
- recovery-only，`lambda_recovery=0.5`
- RIPT scale factor=5
- model/action-header LR：`1e-5`（learning probe 的 1/5）
- scale head 固定
- 1 PPO epoch，max step batch size=1
- checkpoint interval=5；预期保存 step 5/10/15/20
- W&B 和 periodic evaluation 关闭
- 输出必须全新；启动器在 GPU 路径拒绝复用已有输出
- 启动器要求至少 20 GB 可用空间；当前约 252 GB

按已完成 probe 的 683 秒/step 粗略估计，21 steps 约 4 小时，实际可能随成功
episode 提前结束比例在 3.5--5 小时内变化。四份 LoRA+headers checkpoint 预计约
4 GB，磁盘空间充足。

## 训练前验证

- shell syntax：passed
- CPU-only Hydra compose：21 steps、K8/h220、fixed-L2 0.1、LR 1e-5、save 5
  均正确
- `create_env=false` 四 factory 实例化：passed
- official-LIBERO 隔离环境项目回归：30/30 passed
- `--print-command`：passed
- 新输出目录和新日志目录：启动前均不存在
- GPU 7：18 MiB、0%、41°C
- tmux `ra_loop_spatial_overnight_pilot`：启动前不存在

## 保存与恢复限制

上游保存逻辑只保存 LoRA adapter 和 `openvla_headers.pt`，不保存 optimizer、
scheduler 或 dataloader state。若任务中断，最近的 step checkpoint 可用于评测或
作为新 run 的初始化，但不能精确恢复为原训练进程的下一步。不会自动删除任何
中间 checkpoint。

## 路径

- launcher：`train/ra_loop_spatial_overnight_pilot.sh`
- output root：`outputs/ra_loop_spatial_overnight_pilot`
- tmux：`ra_loop_spatial_overnight_pilot:run1`
- log：`logs/ra_loop_spatial_overnight_pilot_20260721/run1.log`
