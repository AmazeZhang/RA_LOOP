# RA-LOOP Spatial 21-step pilot 结果 — 20260721

> 启动：2026-07-21 00:11 CST  
> 完成：2026-07-21 03:41:27 CST（UTC+08:00）  
> 结论：21-step 保存型训练完整通过；产生四份可读 checkpoint，下一步必须独立评测后选型。

## 完成状态

- 21/21 training steps
- 每步 8/8 rollout 和 8/8 PPO batch
- `[info] Finished training`
- `[RA_LOOP_EXIT] 0`
- 总训练循环：3小时29分25秒
- 无 traceback 或训练错误
- GPU 7 完成后：18 MiB、0%、41°C

## 在线训练指标

21 步共 168 条 rollout：

- 总成功：112/168 = 66.67%
- anchor：58/84 = 69.05%
- fixed-L2 Robot-init：54/84 = 64.29%
- 全部 21 步 `non_zero_adv_ratio=1.0`
- 平均 `mean_R_total=0.82738`
- 平均 `pg_clipfrac=0.07544`
- 平均裁剪前 model gradient norm=2.97184

前 5 步与后 5 步在线窗口：

| 窗口 | 总成功 | anchor | perturbed | pg clipfrac |
|---|---:|---:|---:|---:|
| first 5 | 55.0% | 60.0% | 50.0% | 0.1182 |
| last 5 | 67.5% | 70.0% | 65.0% | 0.0549 |

最后两步均为 7/8 成功：倒数第二步 anchor 3/4、perturbed 4/4；最后一步
anchor 4/4、perturbed 3/4。

这些数字来自不断更新的策略和训练初始化，且 rollout tracking 曾跳过一个已全成功
样本，因此不能当作独立、同分布的性能评测，也不能据此声称提升。它们只说明训练
链路持续获得信号且没有明显崩溃。

稳定性方面，平均 clipfrac 比单步 probe 的 0.427 明显降低到 0.075，符合降低 LR
后的预期；但出现一次 `pg_ratio=3.0886` 离群点，裁剪前 model gradient norm 最大
8.948。PPO clipping 和 gradient clipping 仍生效，训练未发散，但独立评测前不应
继续放大训练长度。

## Checkpoint 完整性

保存根目录：

`outputs/ra_loop_spatial_overnight_pilot/libero_spatial/LIBERO_SPATIAL/openvla/RA-LOOP_spatial_robot_init_overnight_pilot/one_task_21step_k8_h220_fixed_l2_0p1_recovery_lr1e5/run_000`

四份 checkpoint 均包含：

- `adapter_model.safetensors`：484,458,600 bytes，CPU 打开后 879 tensors
- `openvla_headers.pt`：638,046,816 bytes，CPU `weights_only=True` 加载通过
- headers：action 16 tensors、scale 18 tensors
- `adapter_config.json`：LoRA rank 32，可解析

SHA-256：

| step | adapter | headers |
|---:|---|---|
| 5 | `bbcc2d58b5466772de2e0ea919119d86d8cbf624754377a48efb96cc07040511` | `6f5652ad49c8710fef00d045158875083db9e4edf4cb9652f8ec54e2bfcffa68` |
| 10 | `ba32746c29b79183582432c58d6ef2ef7a80242e3f61af1497b8040381a889d8` | `875db12cbd10854ee7047bcf3979a587a5359446da58c255268bfc9de95a5437` |
| 15 | `9ce5cb27e746f415dfd4ea110cc13e1830f5e98cf2a21dcbdc879952e633732d` | `65b3828a14a50822ad6579253a63a496bdb60161026bbdd989edc05fded36feb` |
| 20 | `d51c0640dcf4b2d7815f531238c332310ae6cd25ca85a93d6f26dfc21e5ee096` | `188a3376302099d4d4c32b58432224bc2cbad1ba69d60fda85d16a76743d0f59` |

## 下一步

不继续训练。先为 step 5/10/15/20 建立只读加载/评测配置，在相同初始状态和种子下
分别评测：

1. anchor，检查标准任务能力是否退化；
2. 正确施加的 fixed-L2 0.1 Robot-init，检查 recovery 是否提升。

候选选择必须基于独立评测，而不是训练期最后几步成功率。之后才决定是否扩展到
Spatial 10 任务。

## 证据

- 日志：`logs/ra_loop_spatial_overnight_pilot_20260721/run1.log`
- 启动器：`train/ra_loop_spatial_overnight_pilot.sh`
- 计划/启动记录：`docs/RA_LOOP_OVERNIGHT_PILOT_20260721.md`
