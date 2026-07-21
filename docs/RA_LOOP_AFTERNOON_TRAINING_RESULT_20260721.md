# RA-LOOP 下午四任务训练结果 — 20260721

> 启动：2026-07-21 12:56:59 CST  
> 完成：2026-07-21 18:58:18 CST  
> 状态：35/35 optimizer updates，`[info] Finished training`，`[RA_LOOP_EXIT] 0`。

## 完成状态

- 训练循环：6小时00分29秒
- 35 steps × K8 = 280 stochastic rollout
- anchor：101/140 = 72.14%
- fixed-L2：87/140 = 62.14%
- 总成功：188/280 = 67.14%
- 35/35 steps 均有非零 advantage（每步 ratio 最低 0.75）
- GPU 7 完成后：18 MiB、0%、41°C
- 无 traceback 或异常退出

## 分任务在线采样

| task | sampled steps | rollout success |
|---|---:|---:|
| between plate/ramekin | 7 | 45/56 = 80.36% |
| original next-to-plate | 10 | 51/80 = 63.75% |
| on stove | 10 | 63/80 = 78.75% |
| top drawer | 8 | 29/64 = 45.31% |

`shuffle=true` 确实覆盖四任务；top-drawer 是明显最难任务，提供了未饱和学习信号。
这些是训练期随机采样，不可直接与 deterministic independent evaluation 对比。

## 优化稳定性

| metric | mean | min | max | first 5 | last 5 |
|---|---:|---:|---:|---:|---:|
| success | 0.6714 | 0.1250 | 1.0000 | 0.7000 | 0.7000 |
| anchor | 0.7214 | 0.2500 | 1.0000 | 0.7500 | 0.8500 |
| perturbed | 0.6214 | 0.0000 | 1.0000 | 0.6500 | 0.5500 |
| pg clipfrac | 0.0638 | 0.0208 | 0.1390 | 0.0893 | 0.0478 |
| pg ratio | 1.0024 | 0.9797 | 1.0389 | 1.0102 | 1.0035 |
| model grad norm (pre-clip) | 3.1891 | 0.7732 | 21.1694 | 8.2700 | 1.7927 |

PPO ratio 接近 1、clipfrac 后期下降，gradient clipping 始终启用，未见数值发散。
在线 perturbed first/last 窗口没有显示提升，但窗口任务构成与随机动作不同，不能据此
判定独立鲁棒性变化。

## Checkpoint 完整性与保存语义

产生六份目录：step 5/10/15/20/25/30。每份均通过 CPU tensor 级读取：

- `adapter_model.safetensors`：484,458,600 bytes，879 tensors
- `openvla_headers.pt`：638,046,816 bytes，action 16 + scale 18 tensors
- `adapter_config.json`：LoRA rank 32

没有 step 35。上游循环使用从 0 开始的 `global_step`，并在完成本次更新后按
`global_step % save_interval == 0` 保存。因此目录 step 5/10/.../30 分别对应完成
第 6/11/.../31 次 optimizer update；最后 4 次更新只存在于退出前内存中，未保存。
这不是 crash，但独立评测只能使用 step 5--30 六份 checkpoint。

## 结论与下一步

本次运行完成了第一个真正的多任务 RA-LOOP recovery-only 训练，证明四任务混合数据、
paired Robot-init 和 PPO 可以连续稳定运行 6 小时。它还不能证明性能提升。下一步必须
对保存 checkpoint 做与训练前完全相同的 deterministic paired evaluation，重点检查：

1. top-drawer fixed-L2 是否超过训练前 4/6；
2. stove fixed-L2 是否超过 5/6；
3. 四任务 anchor 是否保持；
4. checkpoint 曲线是否存在早期提升、后期退化。

日志：`logs/ra_loop_spatial_afternoon_multitask_20260721/run1.log`。

