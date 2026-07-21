# RA-LOOP 下午多任务训练就绪 — 20260721

> 当前边界：训练已于 2026-07-21 18:58:18 CST 完成；结果见
> `docs/RA_LOOP_AFTERNOON_TRAINING_RESULT_20260721.md`。

> **正式启动：2026-07-21 12:56:59 CST** — 经用户确认，训练已在 GPU 7 和
> shared tmux `ra_loop_spatial_afternoon_multitask:run1` 启动。step-5 LoRA 与
> action/scale headers 加载成功，四任务各 4 demos（共 16 examples）加载成功，
> 四个环境创建完成，已进入 step 1/35 的第 1/8 rollout。检查时 GPU allocated
> 约 18.4 GiB、46°C，无立即错误。训练不依赖当前对话持续连接。

## 为什么进入多任务训练

三任务短评测在 14分59秒内完成：step 0 与 step 5 在 unseen tasks 上均为 33/36，
失败逐对一致。step 5 无泛化退化，但单任务 pilot 没有带来可见跨任务提升。因此下午
不继续单任务堆 steps，而是将原任务与三个有独立评测基线的 unseen tasks 混合训练。

## 固定训练配置

- warm-start：已独立评测的 pilot step 5 LoRA + `openvla_headers.pt`
- Spatial 四任务：原 black-bowl-next-to-plate，加 between、top-drawer、on-stove
- 35 local training steps，K=8（4 anchor + 4 paired perturb），共 280 rollout
- 单 GPU、单 in-process env、horizon=220
- fixed-L2 0.1 rad，`lambda_recovery=0.5`，无 consistency reward
- model/action-header LR `1e-5`，scale head 固定，1 PPO epoch
- 四任务 dataset batch `shuffle=true`；项目 seed 固定
- checkpoint interval=5；实际保存 global-step 5/10/15/20/25/30
- W&B disabled，periodic eval disabled

实际 35 steps 训练循环耗时 6小时00分29秒。上游 global-step 从 0 开始且没有终点
自动保存，因此产生 6 份 checkpoint；最后可评测目录为 step 30。

## 文件与启动边界

- profile wrapper：`train/ra_loop_spatial_afternoon_multitask.sh`
- shared-tmux launcher：`train/launch_ra_loop_spatial_afternoon_tmux.sh`
- 预定 GPU：7
- tmux：`ra_loop_spatial_afternoon_multitask:run1`
- output：`outputs/ra_loop_spatial_afternoon_multitask`
- log：`logs/ra_loop_spatial_afternoon_multitask_20260721/run1.log`

launcher 无参数/`--plan` 只打印命令，不初始化 CUDA。正式启动必须再次获得用户确认，
然后运行：

```bash
bash train/launch_ra_loop_spatial_afternoon_tmux.sh --run 7
```

launcher 会拒绝已有 tmux、output/log root；底层 trainer 还会检查 GPU 显存/利用率/
温度、至少 20 GB 磁盘以及全部 warm-start 文件。中断对话不会停止 tmux 训练。

## 恢复限制

上游 checkpoint 只保存 LoRA 与 headers，不保存 optimizer、scheduler 或 dataloader
state。中断后可从最近 checkpoint 开新 run，但不能精确恢复同一训练进程。因此每
5 steps 保存，并保持所有已有结果只读。
