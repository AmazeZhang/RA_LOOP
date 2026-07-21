# RA-LOOP mode-stratified connectivity smoke — 20260721

> 启动：2026-07-21 22:09 CST
> 完成：2026-07-21 22:21 CST
> 状态：8/8 rollout、1/1 PPO update、`Finished training`、exit 0；GPU 已释放。

## 目的

只验证 mode-stratified advantage 已进入真实 OpenVLA-OFT PPO/gradient 路径。该运行
只有一任务、一个 optimizer step，不能作为性能提升证据。

## 配置与隔离

- task：`pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- warm-start：已独立评测的 pilot step 5 LoRA + headers
- K=8：4 anchor + 4 fixed-L2 0.1 Robot-init
- horizon=220，lambda recovery=0.5，LR/header LR=1e-5
- `advantage_mode=mode_stratified`
- `n_steps=1`，save interval 9999，不保存 checkpoint
- W&B/periodic evaluation disabled
- 独立输出：`outputs/ra_loop_stratified_connectivity_smoke/`
- shared tmux：`ra_loop_stratified_connectivity_smoke:run1`
- 日志：`logs/ra_loop_stratified_connectivity_smoke_20260721/run1.log`

## 启动验收

- CPU helper/history replay、全项目 40/40 tests passed
- CPU-only Hydra composition/factory instantiation passed，`create_env=false`
- GPU 7 启动前：18/24564 MiB、0%、41°C
- 启动后检查：18437/24564 MiB、49°C
- pilot LoRA、action header、scale header 均已确认加载
- 独立 `run_000` 创建成功，进入第 2/8 rollout

## 结果

- rollout：4/8 success
- anchor：2/4 success
- perturbed：2/4 success
- rollout 时间：604.50 秒
- PPO 时间：57.96 秒
- 单步训练循环：662.82 秒（11分02秒）
- `advantage_mode_stratified=1.0`
- `mean_anchor_advantage=5.55e-17`
- `mean_perturbed_advantage=0.0`
- `non_zero_adv_ratio=1.0`
- model gradient norm（裁剪前）：2.6982
- header gradient norm（裁剪前）：104.5
- `pg_ratio=0.9591`，`pg_clipfrac=0.2340`
- `Finished training`，`RA_LOOP_EXIT=0`
- GPU 7 完成后：18 MiB、0%、41°C
- 没有 traceback/OOM，也没有生成 checkpoint（save interval 9999）

两个 mode 的 advantage 均值为零是分层 RLOO 的预期不变量；同时
`non_zero_adv_ratio=1.0` 和非零梯度证明每条 rollout 仍提供了学习信号。header norm
是 gradient clipping 前的总范数，配置的 1.0 clipping 仍由上游 PPO 执行。

## 判断

mode-stratified objective 已通过真实模型、simulator、rollout、上游 PPO、backward 和
optimizer step 的端到端 connectivity gate。该单步结果不证明性能提升。下一步可以
准备小规模受控 A/B，以相同 warm-start 和 seeds 比较 `upstream`、`lambda=0` 与
`mode_stratified`，然后用现有 paired evaluator 判断 anchor 保持和扰动恢复。

完整日志可通过以下 tmux 查看：

```bash
tmux attach -t ra_loop_stratified_connectivity_smoke
```

观察后保持任务运行并退出 tmux：`Ctrl-b`，再按 `d`。
