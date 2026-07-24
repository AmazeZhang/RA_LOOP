# RA-LOOP ICLR 方法重定向 — 2026-07-23

> 工作标题：**RA-LOOP: Constrained Counterfactual Recovery for Robust VLA
> Post-Training**
>
> 当前结论：已实现的 fixed-L2 扰动、恢复奖励和 mode-stratified RLOO 是可靠的
> **诊断基线**，但不是最终创新。正在运行的 13-candidate 独立评测完成前不改动训练
> 进程；评测结果用于结束这一代目标，而不是继续无期限调 `lambda_recovery`。

## 1. 研究问题

我们不再问：

> 对扰动 rollout 多加一个成功奖励，是否能提高鲁棒性？

而改为：

> 在不损伤 nominal task competence 的条件下，如何利用同一环境状态的成对干预，
> 识别并优化 VLA 的**可恢复性缺口**？

机器人初态发生变化后，正确动作往往应该补偿性改变。因此：

- 不要求 anchor 与 perturbed rollout 动作不变；
- 要求任务结果对合理干预保持稳定；
- 训练信号应区分“模型本来就不会”与“模型原本会，但被干预击穿”；
- nominal retention 是约束，不是另一个手工 reward 权重。

## 2. 为什么当前方法不足

当前训练使用

```text
R_total = I(success) + lambda_recovery * I(perturbed and success)
```

并在 anchor、perturbed 两组内分别计算 RLOO advantage。它解决了混组 advantage
互相污染的问题，但仍有四个根本局限：

1. `lambda_recovery` 只是静态重加权，没有利用某条 perturbed rollout 与其 anchor
   来自同一个 base state 的反事实关系。
2. anchor 失败与“anchor 成功但 perturbation 失败”被混在一起；前者是基础能力不足，
   后者才是可恢复性缺口。
3. 固定 `lambda` 不能直接表达“提升 robustness，但 nominal 最多下降 ε”的目标。
4. 固定 0.1 rad 只测一个点，可能把太容易、可学习和不可能恢复的干预混为一谈。

因此，即使 `lambda=0` 最后最好，也只说明 paired rollout/训练分布可能有价值，不能证明
恢复奖励或新算法成立。

## 3. 核心定义：成对反事实可恢复性

对同一个任务和 base simulator state `x`，构造：

- anchor：`tau_a ~ pi(theta, x)`；
- intervention：`do(q_robot <- q_robot + delta)`；
- perturbed：`tau_p ~ pi(theta, x, delta)`。

记二元成功为 `S_a` 和 `S_p`。定义：

```text
Counterfactual Recovery Rate (CRR)
    = P(S_p = 1 | S_a = 1)

Recoverability Gap (RG)
    = 1 - CRR
    = E[S_a * (1 - S_p)] / E[S_a]
```

`S_a=1, S_p=0` 是最关键样本：同一个 base state 上策略原本具备完成任务的能力，但受到
干预后丢失。`S_a=0` 的 pair 不能证明是 robustness failure，应进入基础能力目标或降低
其 recovery 权重。

这里的“不变性”是 outcome invariance；policy/action 应允许 corrective adaptation，
而不是 action invariance。

## 4. 方法：Constrained Counterfactual Recovery LOOP

### 4.1 Counterfactual Recovery Advantage（CRA）

每次采样一个 base state，产生少量 anchor rollouts 和多个不同方向/强度的 perturbed
rollouts。先估计该 state 的 anchor competence：

```text
c_hat(x) = mean_j S_a,j
```

再只对具备足够 anchor competence 的 state 激活 recovery objective：

```text
w(x) = stop_gradient(gate_or_soft_weight(c_hat(x)))
A_rec,i = w(x) * [S_p,i - LOO_mean(S_p,-i)]
```

第一版使用 soft weight，避免一次随机 anchor 失败造成硬切换；最终需比较 hard gate、
soft gate 与不使用 pair condition。与当前 mode-stratified RLOO 的关键区别是：
perturbed advantage 由同一个 base state 的 anchor competence 条件化，而不是只按
mode 分组。

### 4.2 Nominal Performance Constraint（NPC）

目标不是再引入一个固定 reward 权重，而是：

```text
maximize    J_recovery(theta)
subject to  J_anchor(theta) >= J_anchor,ref - epsilon
```

使用 primal-dual 更新：

```text
max_theta min_mu>=0

L(theta, mu)
  = J_recovery(theta)
  + mu * [J_anchor(theta) - (J_anchor,ref - epsilon)]

violation = (J_anchor,ref - epsilon) - J_anchor(theta)
mu <- projection(mu + eta_mu * violation)
```

其中 `J_anchor,ref` 来自 warm-start 在固定 nominal calibration set 上的表现。参考值
必须和训练时 anchor 使用相同的 task、stochastic action 与 rollout protocol；不能把
deterministic 独立评测成功率直接作为 stochastic training batch 的约束目标。多任务
训练为每个任务独立维护 EMA 与 `mu`，避免总体均值掩盖单任务退化。`mu` 根据约束违反
程度自适应，而不是人工扫描 `lambda_recovery`。

为降低二元成功率的小批量噪声，工程上分两阶段：

1. v1：训练前用同 sampler 校准 warm-start，并使用 per-task anchor-success EMA
   约束验证算法方向；
2. v2：在固定 clean observations 上缓存 warm-start action distribution/log-prob，
   加行为 KL trust region，避免训练时额外常驻一份 7B reference model。

多任务校准必须使用全局 barrier：所有任务 reference 完成前不允许任何 PPO 参数
更新，否则先完成校准的任务会改变模型，使后完成任务测到的已不再是 warm-start。
校准/全零 advantage 还必须跳过 AdamW `step()`，避免 decoupled weight decay 在零
policy-gradient 下改变参数。

成功率约束是论文语义，cached KL 是低方差的执行护栏，两者分别报告。

### 4.3 Recovery-Frontier Curriculum（RFC）

采样多个 Robot-init 方向和半径，不单独把“发现失败”宣称为创新。优先级来自成对缺口：

```text
priority(x, delta)
  = anchor_competence
    * recovery_failure_probability
    * learnability
```

- anchor 经常失败：属于基础能力不足，不作为 recovery frontier；
- perturbed 总成功：过于简单，降低采样率；
- perturbed 长期全失败且无改善：可能超出可恢复域，降低采样率；
- anchor 稳定成功、perturbed 成败混合或正在改善：位于 competence frontier，优先。

候选半径先用 `{0.025, 0.05, 0.10, 0.15}` rad，并保留 held-out 半径和方向只用于
泛化评测。课程机制是辅助组件；论文核心仍是 paired counterfactual objective 与
nominal constraint。

## 5. 与最近工作的边界

| 邻近方向 | 已有核心 | RA-LOOP 必须证明的不同点 |
|---|---|---|
| RIPT-VLA | sparse success、dynamic rollout、LOO optimization | 利用同一 base state 的干预 pair，优化条件可恢复性而非普通 success |
| RobustVLA | Jacobian/smoothness、语义保持输入上的 action consistency、最坏扰动选择 | 不约束 proprio perturbation 下动作相同；优化 outcome-level counterfactual gap |
| RoboMD | 学习 adversary，寻找并重采样造成失败的环境配置 | 不把 failure discovery 当主创新；区分“本来不会”和“被干预击穿” |
| PLR / UED / CVaR RL | learning-potential replay、regret environment design、尾部风险优化 | paired VLA intervention estimator + nominal competence constraint |
| Contractive recovery policies | imitation policy 的 out-of-sample recovery/contractivity | sparse online VLA post-training，不预设动作轨迹应收缩到同一条参考轨迹 |

相关材料：

- [RIPT-VLA](https://arxiv.org/abs/2505.17016)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626)
- [Robustness-Aware Reinforcement Post-Training for VLA Models](https://arxiv.org/abs/2511.01331)
- [RobustVLA repository](https://github.com/gakakulicc/RobustVLA)
- [RoboMD](https://www.alphaxiv.org/abs/2412.02818v4)
- [Prioritized Level Replay](https://arxiv.org/abs/2010.03934)
- [Contractive Dynamical Imitation Policies](https://openreview.net/forum?id=lILEtkWOXD)
- [Adversarial Counterfactual Error](https://openreview.net/forum?id=eUEMjwh5wK)

禁止使用的过度声明：

- “首次使用扰动训练 VLA”；
- “首次做 action consistency / adversarial perturbation / failure replay”；
- “固定 reward bonus 本身是新算法”；
- 只在单个固定扰动强度上提升，就声称获得一般鲁棒性。

## 6. ICLR 论文级贡献

目标贡献应当收敛为三项：

1. **问题与度量**：提出 paired intervention 下的 CRR/RG，隔离 task incompetence 与
   intervention-induced failure。
2. **算法**：Counterfactual Recovery Advantage + nominal constrained optimization，
   直接优化 recovery gap 并控制 clean regression。
3. **学习机制与证据**：Recovery-frontier curriculum，并在未见强度、方向、任务和
   policy backbone 上验证更高的 robustness/sample efficiency。

理论/分析部分至少包含：

- CRR/RG estimator 的定义、有限样本偏差/方差；
- paired estimator 相比 unpaired success difference 的方差分析；
- primal-dual constraint 的目标与实际 violation 曲线；
- 哪些 pair 被判断为基础能力失败、recoverable failure 或 impossible intervention。

## 7. 实验矩阵

### 7.1 必须回答的问题

1. 是否提高 perturbed success？
2. nominal success 是否保持在预设 `epsilon` 内？
3. 是否泛化到 held-out perturbation direction/magnitude？
4. paired conditioning 是否优于相同 episode budget 的 unpaired/group-stratified RLOO？
5. adaptive constraint 是否优于固定 `lambda`？
6. frontier curriculum 是否提高 sample efficiency，而不是只增加计算？

### 7.2 Ablation

```text
A0  warm-start, no post-training
A1  vanilla RIPT / task-success RLOO
A2  current paired + mode-stratified lambda=0
A3  current paired + fixed recovery bonus
A4  A2 + Counterfactual Recovery Advantage
A5  A4 + Nominal Performance Constraint
A6  A5 + Recovery-Frontier Curriculum
```

所有组使用相同 simulator episode budget、训练 tasks、评测 seeds 和 checkpoint selection
规则。核心比较是 A4/A5，不允许只凭 A6 的额外采样策略赢得结果。

### 7.3 规模

探索阶段：

- LIBERO-Spatial 4-task gate；
- 3 seeds；
- anchor、seen perturbation、held-out strength/direction；
- paired confidence interval 和逐任务 win/loss，而非只报总成功数。

主结果最低要求：

- 至少 LIBERO-Spatial + 另一个 LIBERO suite；
- 完整 nominal 与 LIBERO-Plus Robot-init；
- 至少两个 policy initialization/backbone，若算力不允许，第二个至少做关键配置；
- 3 seeds；
- success、CRR、RG、nominal drop、sample efficiency、constraint violation；
- 与 RIPT、当前 fixed-lambda、可兼容的 RobustVLA/RoboMD 思路比较。

只有单 backbone、单 suite、单扰动维度时，更适合作为有力的 method pilot，而不是
把 ICLR 主会结论写满。真实机器人不是方法成立的必要条件，但会显著增强论文；缺少
真实实验时必须加强跨 suite/backbone 和 held-out intervention 证据。

## 8. 分阶段 gate

### Gate 0：结束旧目标

等待 2026-07-23 已启动的 13-candidate 独立评测完成：

- 若 fixed-lambda 无净增益：作为“静态 bonus 不足”的直接动机；
- 若偶有增益：仍需检查 seed/checkpoint 稳定性，不把它升级为最终方法；
- 不再启动同结构的长时 `lambda` sweep。

### Gate 1：CPU 与历史数据验证

- 复用当前已经严格校验的 identical base states、`pair_id` 和 interleaved
  anchor/perturbed 数据结构，不另造一套 rollout 通路；
- 实现 CRR/RG、pair identity、conditional mask 和 dual update；
- truth-table、invalid episode、全成/全败、单 pair、mixed pair 单元测试；
- 用已完成 rollout 的 success pattern 离线 replay advantage；
- 证明 `S_a=0` 不会被误计为 recoverability evidence；
- 不使用 GPU。

### Gate 2：4-task 小训练

- A2/A3/A4/A5，先不加 curriculum；
- 20--35 updates，3 seeds 或先 2 seeds 筛选；
- 必须出现逐 task 的 paired recovery 改善，且 nominal 不明显退化；
- 失败则优先检查 estimator/constraint，不扩大训练。

### Gate 3：10-task 与多强度

- 加 RFC；
- 比较固定 0.1 与 multi-radius/held-out radius；
- 只有 perturbed success 提升至少约 5 percentage points、nominal drop 不超过
  预注册 `epsilon`（建议 1--2 points），才进入全规模。

### Gate 4：论文实验

- 扩 suite/backbone/seeds；
- 预先冻结 checkpoint selection 和 statistical protocol；
- 报告所有失败任务与 compute/episode cost。

## 9. 最近两步

1. 当前只读观察 `ra_loop_fulltask_candidate_eval`，完成后写旧目标的最终结论。
2. 下一次代码改动从 Gate 1 开始：先写算法接口和 CPU 测试，不直接占用 GPU，也不
   在尚未验证目标语义时启动长训练。

## 10. 当前判断

这一重定向比“扰动 + 不变性 + reward bonus”更接近 ICLR 所需的研究命题，因为它：

- 用 pair 定义了可证伪的新学习对象；
- 明确处理 clean/robust trade-off；
- 与已有 consistency、adversarial sampling、failure replay 工作划清边界；
- 能通过 A2--A6 消融判断提升究竟来自哪里。

但它目前仍是**研究假设，不是已证明贡献**。是否成为论文主线，取决于 Gate 2 是否
出现稳定的 counterfactual recovery 增益；不得用在线训练 reward 代替独立证据。
