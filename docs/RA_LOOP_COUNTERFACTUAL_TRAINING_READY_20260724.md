# RA-LOOP counterfactual-constrained 训练前状态 — 2026-07-24

> 状态：代码、配置、CPU 测试和 Hydra factory preflight 已通过；**尚未启动 GPU
> smoke 或训练**。

## 1. 旧目标最终结果

2026-07-24 03:31 CST，13 个候选、1560 个 paired episodes 全部完成，13/13
candidate exit 0。

| 候选 | Anchor | fixed-L2 | 总计 | CRR | RG |
|---|---:|---:|---:|---:|---:|
| warm-start baseline | 59/60 | 55/60 | 114/120 | 91.53% | 8.47% |
| lambda=0.5, seed10000, step60 | 60/60 | 56/60 | 116/120 | 93.33% | 6.67% |
| lambda=0, seed10000, step60 | 60/60 | 56/60 | 116/120 | 93.33% | 6.67% |
| lambda=0.5, seed20000 最佳 | 59/60 | 54/60 | 113/120 | 89.83% | 10.17% |
| lambda=0, seed20000 最佳 | 59/60 | 53/60 | 112/120 | 88.14% | 11.86% |

两个 seed10000 最佳候选在 120 条二元结果上完全一致，只比 baseline 改变两条：
一条 anchor failure 变 success，一条 perturbed failure 变 success。seed20000 没有
候选超过 baseline。因此 fixed recovery bonus 没有独立贡献证据，不再补相同目标的
lambda/邻近 checkpoint。

## 2. 已接入的新目标

训练器模式：

```text
advantage_mode = counterfactual_constrained
legacy lambda_recovery = 0

A_total = A_CRA + mu_task * A_anchor
```

- `A_CRA` 只使用配对 anchor 成功的 perturbed rollout；
- anchor 失败的 pair 不被误记为 recovery failure；
- `mu_task` 由 nominal success constraint 的 per-task primal-dual 更新产生；
- 旧 `lambda_recovery` 明确关闭，新恢复信号直接进入 advantage，不是 reward bonus；
- 每个自定义 group 的 advantage 和为零，再逆编码为 surrogate reward，使上游
  PPO 精确恢复目标 advantage。

## 3. Warm-start calibration 安全约束

不能使用 deterministic 独立评测的 `59/60` 约束 stochastic training rollout。
训练前在相同 task、相同 stochastic policy、相同 rollout protocol 下校准：

- smoke：每任务 1 个 calibration batch；
- 4-task gate：每任务 3 个 calibration batches，即 12 anchor samples/task；
- 多任务使用全局 calibration barrier：所有任务 reference 完成前，任何任务均不更新；
- calibration/全零 advantage 时阻止 AdamW `optimizer.step()`，但正常清理 gradient，
  防止 weight decay 偷偷改变 warm-start；
- dual/reference/EMA 状态只有在上游 PPO 正常返回后才提交。

当前状态只在进程内保存，因此 profile 强制 fresh output、禁用 resume。短 gate 可接受；
进入长时/可恢复训练前应实现 constraint state checkpoint。

## 4. 下一次运行：2-step GPU smoke

目标不是验证效果，只验证真实模型上的两个状态：

1. step 1：校准完成，`parameter_update_applied=0`；
2. step 2：若出现非零 CRA/anchor advantage，则
   `parameter_update_applied=1`，并输出 CRR eligibility 与 nominal dual metrics。

选用历史上 fixed-L2 成败更混合的 top-drawer task。配置：

```text
one task
2 steps
K=8: 4 anchor + 4 paired fixed-L2
horizon=220
fixed-L2=0.1 rad
calibration batches=1
lambda_recovery=0
warm-start=pilot step5
```

共享 tmux 启动器已准备，安全默认仅打印计划：

```bash
bash train/launch_ra_loop_counterfactual_smoke_tmux.sh --plan
```

用户确认后才执行：

```bash
bash train/launch_ra_loop_counterfactual_smoke_tmux.sh --run <GPU_ID:2-7>
```

预计约 20--30 分钟。输出和日志目录当前均不存在；启动器拒绝复用同名 tmux、日志或
训练输出。

### Smoke 验收

- tmux/进程 exit 0；
- 两个 optimize loops 完成；
- 首批全局校准完成且 `parameter_update_applied=0`；
- 第二批 pair metadata、CRA、nominal metrics 全部出现；
- 若第二批有非零目标，参数 step 恰好启用；
- 若第二批全成或全败导致零 advantage，安全通过但学习通路证据不足，应换固定 seed
  做一次新的 bounded smoke，不能把零梯度解释为实现失败。

## 5. Smoke 通过后的 4-task gate

已准备 `train/ra_loop_counterfactual_gate.sh`：

```text
4 tasks
50 total steps
3 calibration batches/task
global calibration barrier
至少约 35 个 active-update 机会
checkpoint interval=10
K=8, h=220, fixed-L2=0.1
lambda_recovery=0
```

预计单卡约 8--9 小时。它只验证 CRA + NPC 核心，不包含 Recovery-Frontier
Curriculum；RFC 必须在 A4/A5 已有正结果后再加入，避免无法归因。

## 6. 验证与资源

- 全部 CPU tests：76 passed；
- smoke Hydra compose/factory instantiate：passed；
- gate Hydra compose/factory instantiate：passed；
- shell syntax 与 `git diff --check`：passed；
- CPU preflight 没有创建 environment、model 或 CUDA context；
- 2026-07-24 preflight 后磁盘可用约 175 GB；
- GPU 2--7 均为空闲，约 18 MiB、0% utilization、40--42°C；
- 本阶段未启动 GPU，smoke/gate 输出目录均不存在。

## 7. 当前停止点

下一条有状态命令就是启动 2-step GPU smoke。在用户确认前不执行。
