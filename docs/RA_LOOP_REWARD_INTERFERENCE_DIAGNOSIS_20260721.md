# RA-LOOP reward interference 诊断 — 20260721

## 结论

下午训练的 anchor regression 存在一个明确的训练目标机制：当前 recovery reward
让成功 anchor 得 1.0、成功 perturbed episode 得 1.5，随后上游 RIPT-OFT 将同一
K=8 batch 的四条 anchor 和四条 perturbed reward 一起计算 RLOO advantage。这会
使表现已经正确的 anchor 在高成功 batch 中得到负 advantage。

该机制与独立评测观察到的 anchor 退化方向一致，是下一轮必须先修正的问题；但现有
日志不能单独证明它是每个失败轨迹的唯一原因。

## 公式核对

上游 `rl_optimizer_openvla_oft.py` 对 K=8 使用：

```text
baseline_i = (sum_j reward_j - reward_i) / 7
advantage_i = reward_i - baseline_i
            = (8 * reward_i - sum_j reward_j) / 7
```

当前 reward 为：

```text
anchor success       = 1.0
perturbed success    = 1.5
failure              = 0.0
```

因此当四条 anchor、四条 perturbed 全部成功时，总 reward=10：

```text
successful anchor advantage    = (8 * 1.0 - 10) / 7 = -2/7
successful perturbed advantage = (8 * 1.5 - 10) / 7 = +2/7
```

也就是说，8/8 成功并不是零更新，而是继续降低成功 anchor 动作的概率、提高成功
perturbed 动作的概率。这不是期望的 anchor-preserving recovery objective。

## 35-step 日志复算

从 `run1.log` 的 35 个逐步 metric 复原每步四条 anchor 和四条 perturbed 的 success
count，再严格按上游公式复算：

- 10/35 steps 中，成功 anchor 的 advantage 小于 0；
- 共 39 条成功 anchor rollout 收到负 advantage；
- 两个 8/8 全成功 batch（step 10、30）仍产生上述 `-2/7` / `+2/7` 分化；
- 140 条 anchor 的 advantage 均值为 `-0.120408`；
- 140 条 perturbed 的 advantage 均值为 `+0.120408`；
- 若同一批 success pattern 使用 `lambda_recovery=0`，两者分别为 `+0.057143` 和
  `-0.057143`，说明 recovery bonus 直接反转了两种模式的总更新方向。

最后一项不是说 `lambda=0` 就是最终方法；它用于隔离并确认模式偏置来自 bonus 与
跨模式 RLOO baseline 的耦合，而不是 success count 本身。

## 独立评测轨迹摘要

step 5 相对 warm-start 的三个成功翻转均使用完全相同的 init/noise/seed：

| 变化 | pair | baseline | afternoon step 5 |
|---|---|---:|---:|
| loss | 原任务 `anchor/init 4` | success, 12 action chunks | fail, 28 chunks |
| loss | 原任务 `fixed_l2/init 1` | success, 12 chunks | fail, 28 chunks |
| gain | stove `fixed_l2/init 2` | fail, 28 chunks | success, 24 chunks |

两条 loss 都从快速完成变成跑满 horizon，gain 则从跑满 horizon 变成完成。这确认翻转
是实际 rollout 行为差异，而不是 summary 聚合错误。现有 evaluator 没有保存视频、
逐动作或对象状态，因此不能继续判断是抓取、运输还是放置阶段失败。

## 下一轮最小修正

首选不是直接增加 KL 或继续延长训练，而是先消除已确认的 objective interference：

1. anchor 与 perturbed 各自在组内计算 leave-one-out advantage（每组 K=4）；
2. perturbed 组仍可保留 1.5 的成功 reward，以强调 recovery；
3. 一个模式组 4/4 成功时，该组 advantage 应严格为 0；
4. perturbed 的成功与否不得改变 anchor advantage，反之亦然；
5. 保留现有 valid-mask、exactly-once rollout 和上游 PPO/clipping 行为；
6. 先做纯 CPU 单元测试和历史 35-step reward replay，未通过前不启动 GPU。

这个方案称为 **mode-stratified RLOO**。它比立即加入 reference-policy KL 更小、更能
直接对应已确认的问题。修正后再做一个短小的 A/B：原 objective、`lambda=0`、
stratified recovery；只有 anchor 不退化且 fixed-L2 有改善时才扩大训练。

## CPU helper 验证

已在 `ra_loop/robustness.py` 实现纯 NumPy
`compute_mode_stratified_rloo_advantages`：

- 按 `(rollout_group_id, is_perturbed)` 分层计算 leave-one-out；
- invalid padding 不参与 baseline 且输出零 advantage；
- reward、mode、group、valid mask 严格校验；
- 任一有效 stratum 少于两条时 fail closed，不回退到跨模式 baseline。

新增八个 CPU case，覆盖全成功归零、模式互不干扰、多 rollout group、padding 和异常
输入。目标文件 24/24 tests、official-LIBERO 隔离环境全项目 38/38 tests 通过。

将历史 35-step 的实际 anchor/perturbed success count 与 reward 原样 replay 到新
helper 后：

- 101 条成功 anchor 中，负 advantage 从旧目标的 39 条降为 0；
- 57 条成功 anchor 得到正 advantage；
- 44 条成功 anchor 在 anchor 组全成功时得到零 advantage；
- 每一步、每个模式的 advantage 和均为零（浮点误差量级小于 `3e-15`）。

## 当前停点

mode-stratified helper 已接入 `RobotInitRecoveryOptimizer`，但没有复制或修改上游 PPO。
接入方式是把目标 advantage 乘以 `(K-1)/K`，可逆编码成组内和为零的 surrogate
reward；上游原有 K-way RLOO 再将其精确还原为目标 advantage。只有父优化器看到该
surrogate，训练日志中的 success/recovery/total reward 仍由真实 reward 重新计算。

安全与兼容状态：

- `advantage_mode=mode_stratified` 已显式写入 Hydra launcher；
- 保留 `advantage_mode=upstream`，用于复现旧目标和后续 A/B；
- rollout generator 与 reward method 均在 `finally` 恢复；
- 父优化器真实复算的 advantage 与 helper 在 `1e-15` 精度内一致；
- rollout exactly-once、episode 顺序、valid mask、每组 K 和异常恢复均 fail closed；
- 全项目 40/40 tests 通过；
- 四任务下午配置 CPU-only Hydra composition/factory instantiation passed，输出明确包含
  `"advantage_mode": "mode_stratified"`；
- preflight 的 `create_env=false`，没有加载模型、启动 simulator/GPU 或写 checkpoint。

下一步安全边界是一任务、一 optimizer step 的 GPU connectivity smoke，检查真实上游
PPO 日志出现 `advantage_mode_stratified=1`、两个 mode advantage 均值接近零、梯度非零，
并在退出后确认 GPU 释放。它只能验证接线，不能作为性能提升证据。
