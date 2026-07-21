# RA-LOOP 训练前就绪记录 — 20260720

> 完成时间：2026-07-20 23:33 CST（UTC+08:00）  
> 当前边界：已推进到 GPU 学习信号验证启动前；本步骤未启动训练、未查询或使用 GPU。

> **启动更新：2026-07-20 23:44 CST** — 经用户确认，h220/K8 learning-signal
> probe 已在空闲物理 GPU 7 启动，共享 tmux 为
> `ra_loop_spatial_learning_probe:run1`。启动前 GPU 为 18 MiB、0%、41°C；模型、
> scale header、dataset 和单环境初始化均已通过，已进入 8 条 rollout，第 1 条
> anchor 成功并开始第 2 条。首次运行时 GPU 约 18.1/24.6 GB、47°C。日志为
> `logs/ra_loop_spatial_learning_probe_20260720/run1.log`。当前仍在运行，尚无最终
> PPO/梯度结论；按既有 K8/h220 实测预计总耗时约 4--6 分钟。

> **完成更新：2026-07-20 23:55:40 CST** — probe 已完成并通过：8 条 rollout 为
> `[1,0,1,0,0,1,1,0]`，总成功 4/8；anchor 3/4、fixed-L2 perturb 1/4；
> `non_zero_adv_ratio=1.0`，PPO loss 与模型梯度非零，`[info] Finished training`、
> `[RA_LOOP_EXIT] 0`。实际耗时为 rollout 622.39 秒、PPO 60.23 秒、训练循环
> 683.00 秒（11分23秒）。GPU 7 已释放到 18 MiB/0%/41°C。没有 checkpoint
> 文件。详细结果见 `docs/RA_LOOP_SPATIAL_LEARNING_PROBE_20260720.md`。下一步是
> 准备会保存 checkpoint 的保守短程 RA-LOOP pilot，而不是直接开始长训练。

## 当前目标

最终方法为 OpenVLA-OFT + RIPT-VLA 的 robustness-aware RL（RA-LOOP）。首个方法
聚焦 LIBERO Spatial Robot-init recovery-only，`lambda_consistency=0`。主要目标是
在正确施加 Robot-init 物理扰动的口径上提升鲁棒成功率，同时控制标准 Spatial
性能下降。

## 本步骤完成内容

1. 在 `ra_loop/robustness.py` 增加 `fixed_l2` Robot-init 采样模式：先采样随机
   7D 方向，再归一化到指定总 L2 半径；仅修改经具名验证的 Panda state[1:8]，
   最后应用真实关节限位。
2. 保留 `gaussian_std` 默认模式，使历史 connectivity smoke 的语义不被静默
   改写。新 probe 显式使用 `fixed_l2`。
3. rollout metadata 新增 `perturb_sampling_mode` 和裁剪后的
   `joint_noise_l2`，区分请求强度与实际施加偏移。
4. `RobotInitRecoveryRolloutGenerator` 接入可校验的
   `robot_init_sampling_mode`；未知模式 fail-closed。
5. 新增安全启动器 `train/ra_loop_spatial_learning_probe.sh`。无参数只执行
   CPU-only Hydra/factory preflight；`--print-command` 只打印；只有显式
   `--run <GPU_ID>` 才可能进入 GPU 路径，并在启动前检查显存、利用率和温度。

## 已锁定的下一次运行配置

- 单任务：`pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- 单 GPU、单环境、in-process runner
- 1 training step
- K=8：4 个 anchor + 4 个 paired Robot-init rollout
- horizon=220
- `fixed_l2=0.1 rad`，对齐 LIBERO-Plus 最轻生成档
- perturb seed：20260720
- recovery-only：`lambda_recovery=0.5`，无 action consistency
- RIPT scale factor：5.0
- 4 demos，固定 dataloader 顺序
- W&B 关闭、periodic eval 关闭、checkpoint 保存关闭

这仍是“学习信号 probe”，不是长训练。判据为：8 条 rollout 完整结束、PPO 正常
退出；若出现混合成功/失败，应同时出现非零 leave-one-out advantage 和梯度，并
记录 anchor/perturbed success、实际 joint L2 和增广奖励。通过后才配置短程
RA-LOOP pilot 训练和 checkpoint 保存策略。

## 验证证据

- fixed-L2 核心单测：16/16 passed。
- official-LIBERO 隔离环境全项目回归：30/30 passed。
- `py_compile`：`ra_loop/robustness.py`、`ra_loop/ript_recovery.py` passed。
- CPU-only Hydra compose：K8/h220/fixed-L2 0.1、四个项目 factory target、关闭项
  均正确。
- `create_env=false` factory 实例化：passed；未加载 dataset/env/model。
- shell 语法与 `--print-command`：passed。
- `outputs/ra_loop_spatial_learning_probe` 不存在，证明 preflight 未进入训练输出。

CPU factory import 期间 TensorFlow 打印 `CUDA_ERROR_NO_DEVICE`，这是
`CUDA_VISIBLE_DEVICES=''` 下的设备探测结果，不是 GPU 运行或失败。

## 当前风险与非阻塞事项

现有 LIBERO-Plus bounded baseline evaluator 的 `reset → set_init_state` 顺序会按
静态代码路径覆盖 PandaN reset qpos。因此旧 38.57% 可继续作为原调用口径记录，
但不能证明真实 PandaN 偏移被保留。该问题不阻塞当前 RA probe，因为 RA 在送入
rollout 前直接修改 flattened state[1:8]；正式方法对比前仍需校正评测口径。

## 下一动作（尚未执行）

经用户确认并选择空闲 GPU 后，在共享 tmux 中运行：

```bash
bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh --run <GPU_ID>
```

启动前 launcher 会再次拒绝显存超过 1 GiB、利用率超过 10% 或温度超过 75°C 的
GPU。运行无 timeout，并由 tmux 保留进度。
