# RA-LOOP mode-stratified 四任务长训准备 — 20260721

## 目的

对下午已完成的旧 objective 35-step run 做单变量受控重复。除 advantage 计算从
upstream mixed RLOO 改为 mode-stratified RLOO 外，warm-start、四任务、rollout
配置、学习率、训练步数和 checkpoint 间隔保持一致。

这是一轮约 6 小时的中等规模方法验证，不是论文完整 benchmark 训练。训练期指标只
用于稳定性检查；性能结论必须来自训练后同 seeds 的独立 paired evaluation。

## 冻结配置

- warm-start：pilot step 5 LoRA + headers
- tasks：原任务、between、top-drawer、stove
- 35 optimizer updates，K=8，预计 280 online rollouts
- 每步 4 anchor + 4 fixed-L2 0.1 Robot-init
- horizon 220，lambda recovery 0.5
- model/header LR 1e-5，单 GPU
- `advantage_mode=mode_stratified`
- DataLoader `shuffle=true`
- 每 5 local steps 保存，预期 step 5/10/15/20/25/30
- W&B 与 periodic evaluation disabled
- 预计运行 6--8 小时

## 隔离与安全

- 输出：`outputs/ra_loop_spatial_stratified_multitask/`
- 日志：`logs/ra_loop_spatial_stratified_multitask_20260721/run1.log`
- shared tmux：`ra_loop_stratified_multitask:run1`
- launcher 拒绝复用已有 session、输出或日志目录
- trainer 启动前检查 checkpoint 文件、磁盘至少 20 GB，以及目标 GPU 的显存、利用率
  和温度
- 中断对话或 detach tmux 不会停止训练

## 启动前 gates

1. mode-stratified 数学 helper/history replay passed；
2. 全项目 40/40 CPU tests passed；
3. 一任务一步真实 GPU smoke：8/8 rollout、PPO/backward/optimizer、exit 0；
4. smoke 中 `advantage_mode_stratified=1`、non-zero advantage 和非零梯度；
5. 长训配置 CPU-only Hydra composition/factory preflight；
6. shell syntax、独立路径、warm-start 文件、磁盘和 GPU 最终检查。

完成后不按在线成功率选型。必须对保存的 checkpoint 使用与 baseline 和旧 objective
完全相同的四任务 paired evaluator，再比较 anchor retention、fixed-L2 recovery 和逐
pair gain/loss。
