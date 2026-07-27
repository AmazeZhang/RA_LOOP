# 任务书：CI-GRPO —— 对比式指令强化学习修复 VLA 语言失聪

> 三段式规范（用户 spec-driven 偏好）：**第一段 EARS 验收标准 → 第二段 技术设计 → 第三段 可勾选任务清单**
> 执行者：Claude Code（在 7×4090 集群）；本文档负责"讲清楚做什么、怎么验收"，不含实现代码。
> 配套：`00_BACKGROUND.md`（动机+根因）、`01_RELATED_WORK.md`（文献+核实等级）。

---

# 第一段 · EARS 验收标准

采用 EARS（Easy Approach to Requirements Syntax）。每条为可独立验证的行为契约。

## 1.1 环境与基座（Ubiquitous / 恒定要求）

- **R1** 系统**应当**在 7×RTX 4090 上跑通 SmolVLA-450M 的全参数 GRPO 训练，单卡显存占用 ≤ 24 GB。
- **R2** 系统**应当**提供一个 rollout harness，能在 LIBERO（MuJoCo）中执行策略、读回末态并由 BDDL 目标谓词判定成功/失败，判定结果与官方 evaluator 在 100 条抽样上一致率 = 100%。
- **R3** 当选用 OpenVLA-OFT-7B 做规模确认时，系统**应当**以 LoRA 方式在单张 80 GB 卡上完成训练（RIPT-VLA 授权确认后）。

## 1.2 对比组构造（Event-driven）

- **R4** 当给定一个初始状态 s₀ 时，系统**应当**生成 K 条（建议 K=3–4，太大稀释每条指令梯度）**互相冲突**的指令 {ℓ_k}，每条绑定一个**在同一 s₀ 下可由仿真器独立判定**的目标谓词 g_k，且满足：(a) 所有 g_k 在 s₀ 下**可达且未满足**；(b) **目标两两互斥** `∀i≠j: g_i(traj)∧g_j(traj)=False`（硬校验，防"苹果放碗里"与"放左边"在碗位于左边时同时判真而污染奖励）。
- **R5** 当两条指令 ℓ_i≠ℓ_j 属于同一对比组时，其初始渲染像素**应当**完全一致（pixel-identical s₀），差异**仅**存在于语言指令。
- **R5b** 系统**应当**保证组内任意两条指令的最优轨迹**在早期就分叉**：以最优/参考轨迹前 N 步的 DTW 距离 ≥ 阈值为筛选条件，剔除"前若干步动作雷同（都在接近同一物体）"的指令组，否则共享前缀上组内优势恒为 0、梯度只砸末段，长 horizon 稀疏奖励下训不动。
- **R6** 若某个 s₀ 无法构造出满足 R4/R5/R5b 的 ≥3 条指令，系统**应当**跳过该 s₀ 并记录原因，而**不应当**用近似/污染样本填充。

## 1.3 奖励与优化（Event-driven）

- **R7** 当一条 rollout 在指令 ℓ_k 下结束时，系统**应当**赋 r_k=1 当且仅当末态满足 g_k，否则 r_k=0。
- **R8** 系统**应当**用 GRPO 组内相对优势 Â_k=(r_k−mean_group(r))/(std_group(r)+ε) 更新策略，且只对 action token 计算 loss（observation masking）。CI-GRPO 定位为 **post-training**：须从一个 clean SR≈40–60% 的 BC/SFT base（臂1）冷启动，**不得** from scratch（否则组内全 0、std=0 无梯度）。
- **R9** 默认**不启用**任何 reward shaping；仅当训练熵 entropy<0.1 持续 ≥200 步时，**才可**触发防御性 shaping（该逻辑作为监控触发项，不进 method section）。

## 1.4 评测指标（Ubiquitous，本项目命根子）

- **R10** 系统**应当**报告主指标 **LSG，并拆成 LSG-hit 与 LSG-miss 两项**：LSG-hit=succ(正确指令)（越高越好），LSG-miss=succ(错配指令，同场景)（越低越好），LSG=LSG-hit−LSG-miss。**只有 hit 维持高 且 miss 显著降才算达标**，防止"整体变差也显得差值变大"。可类比 ATE（instruction 为 treatment）。
- **R11** 系统**应当**报告**指令交换重定向率**：同 s₀ 下 ℓ_i→ℓ_j 后行为切换到 g_j 的比例。
- **R11b** 系统**应当**报告 **language dropout 探针**：推理时把语言 token 置零/换 [MASK]，测 SR 下降幅度（失聪≈不降，真听懂应崩到≈1/K）；与错配指令指标配对呈现。此图作为 Figure 1 候选。
- **R12** 系统**应当**报告标准成功率，且 CI-GRPO 相对 SFT baseline 的标准成功率下降**不得** > 5 个百分点（sanity 约束）。
- **R13** 系统**应当**在**留出**集报告 R10–R11，且留出须覆盖三类：(i) LIBERO-Plus 语言轴；(ii) CALVIN 长程指令；(iii) **paraphrase 与训练未见的指令组合**（如训练"put apple on the left"、测试"place the fruit to the west side"）。若仅训练分布内指令组合上赢、paraphrase 塌，则判定为"关键词聋"而非语义理解，须如实降低 claim。

## 1.5 对照矩阵（立论，Complex / State-driven）

- **R14** 系统**应当**产出**核心四臂**完整结果：臂1 SFT baseline、臂2a 任务成功率 RL（单指令组标准 GRPO，RLVLA 式）、臂3 CAST 式反事实 SFT（仿真适配）、臂4 CI-GRPO。并**应当**产出**强化臂**：臂2b paraphrase 增强 RL（同 goal 多说法，语言侧堆数据但无冲突组）、臂5 LangForce 式损失侧 PMI（适配 LIBERO 公平复现）。臂6（CI-GRPO+LangForce 叠加）为 stretch，视时间。
- **R15** 立论达标当且仅当：**臂4 在留出集 LSG 与重定向率上显著优于臂2a、臂2b、臂3、臂5**（配对显著性检验 p<0.05，≥3 seed）。臂6 用于探索正交叠加（叠加涨=同超两 SOTA；不涨=攻击同一 bottleneck 的 interesting finding，两种结论都写）。
- **R16** 若 R15 不成立，系统**应当**如实记录并输出诊断（哪一臂、哪个指标、可能机理），**不得**通过挑 seed / 挑子集包装结论。特别地，若臂2a/2b 已把 LSG 修到接近臂4，须先自查其组构造是否混入隐性对比，确认无误则核心卖点崩，转向分析型论文。

## 1.6 可复现与证据链（Ubiquitous）

- **R17** 每个报告数字**应当**附可验证证据链：训练/评测命令、config hash、日志路径、seed 列表。
- **R18** 所有引用的外部数字**应当**在 `01_RELATED_WORK.md` 标注 Tier，且正文引用前当场 fetch 核对；核不到的留 "—"。

---

# 第二段 · 技术设计

## 2.1 总体数据流

```
                    ┌─────────────────────────────────────────┐
                    │  对比组采样器 (Contrastive Group Sampler) │
                    │  输入: LIBERO-Goal 任务池                  │
                    │  输出: {s₀, [(ℓ_1,g_1),...,(ℓ_K,g_K)]}    │
                    └───────────────┬─────────────────────────┘
                                    │  同一 s₀ (pixel-identical)
                    ┌───────────────▼─────────────────────────┐
                    │  Rollout Harness (SmolVLA @ LIBERO)       │
                    │  对每条 ℓ_k 从同一 s₀ 起 rollout          │
                    │  → 末态 → BDDL 谓词判定 → r_k∈{0,1}       │
                    └───────────────┬─────────────────────────┘
                                    │  组内 {r_1..r_K}
                    ┌───────────────▼─────────────────────────┐
                    │  GRPO 优势: Â_k=(r_k−mean)/(std+ε)        │
                    │  observation masking (仅 action token)    │
                    │  PPO-clip 更新 SmolVLA policy             │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │  评测器: LSG / 重定向率 / 成功率           │
                    │  留出集: LIBERO-Plus 语言轴 + CALVIN      │
                    └───────────────────────────────────────── ┘
```

## 2.2 对比组构造（最关键、最先验证的模块）

**为什么最关键**：整个方法的因果信号来自"图像相同、正确动作因指令而异"。若构造不出像素一致的 s₀ + 互斥目标，方法退化为普通多任务 RL。

**首选实现路径（LIBERO-Goal）**：LIBERO-Goal 的设计是"同一场景布局、不同目标谓词"，天然接近对比组需求。设计步骤：
1. 固定一个场景布局与物体初始位姿 → 得到 s₀。
2. 从该场景可达的目标谓词集合中选 K 个互斥目标 g_1..g_K（例如"把 A 放到 B 上" vs "把 A 放到 C 上"）。
3. 为每个 g_k 取其自然语言指令 ℓ_k。
4. 校验：K 条指令共享 s₀ 渲染（R5），且各 g_k 在 s₀ 下都是"可完成且互斥"的（R4）。

**前置验证（必须最先做，进任务清单 P0）**：确认 LIBERO-Goal 是否允许"锁定同一初始 state（含物体位姿、机器人 qpos）+ 切换 BDDL 目标谓词"。参考 RA-LOOP 已有的 `set_state_from_flattened` / BDDL 加载经验（见 repo `ra_loop/` 与 eval 脚本）。若原生不支持，需写 s₀ 冻结 + 目标谓词替换的适配层。

**回退方案**：若 LIBERO-Goal 不满足像素一致约束，回退到"程序化生成互斥指令对"（如颜色/方位反事实），但须保证同一 s₀。

## 2.3 奖励与 GRPO

- **奖励**：纯稀疏 0/1，来自 BDDL 谓词（R7）。不引入学习型奖励模型，避免额外噪声与不可复现。
- **GRPO 组**：一个 group = 一个 s₀ 的 K 条指令 rollout（可对每条指令再采 M 条 rollout，组大小 = K×M）。组内相对优势天然编码"对比"：忽略语言 → 组内 r 同分布 → 优势≈0 → 无学习信号朝"分叉"方向。
- **observation masking**：只对 action token 计 policy loss（沿用 SimpleVLA-RL / Agent RL 常规做法）。
- **框架**：veRL + GRPO；组构造借鉴 SimpleVLA-RL `ray_trainer.py` group-by-uid，奖励接口借鉴 `RobRewardManager`。
- **稳定性**：默认关闭 shaping（R9）；KL 正则对齐 SFT 初始策略防坍塌；若 GRPO 不稳，按 RLVLA（2505.19789）结论回退 PPO（method section 里说明选择依据）。

## 2.4 对照臂实现要点

| 臂 | 实现 | 关键区别 |
|----|------|----------|
| 1 SFT baseline | 直接用 SmolVLA 在 LIBERO 上 BC，clean SR≈40–60% 作为 RL base | 不做 RL，测 LSG≈0；同时是臂2/4 的冷启动 |
| 2a 任务成功率 RL | 标准 GRPO，**单指令组**（每 group 同一指令多 rollout） | 无冲突指令，等价 RLVLA 设置 |
| 2b paraphrase 增强 RL | 同 goal 换 N 种说法做 RL | 语言侧堆数据，但**无冲突组结构**、不改 H(ℓ|v)；赢它才证明"结构才是关键" |
| 3 CAST 式反事实 SFT | 用大模型/脚本对同 s₀ 不同指令合成动作标签，SFT | 数据侧解法，标签有噪声（对齐 CAST 60–70%）|
| 4 CI-GRPO（ours） | 本设计 2.2–2.3 | 冲突指令组 + 组内对比奖励，label-free |
| 5 LangForce 式损失侧 PMI | 双分支 p(a|v)/π(a|v,ℓ) + 条件 PMI 惩罚，适配到 LIBERO | 损失侧解法，用于同协议公平对比 + 验证正交 |
| 6 CI-GRPO + LangForce（stretch） | 臂4 训练目标叠加臂5 的 PMI 项 | 探索正交叠加，成本低、故事完整度高 |

**公平性控制**：所有臂用同一基座、同一 s₀ 采样池、同等训练步数/样本预算、同一评测器与 seed 集合。臂3 的合成标签预算、臂2b 的 paraphrase 增广量、臂4 的 rollout 样本预算按"等效交互次数"对齐，避免"谁数据多谁赢"的质疑。臂5 因在 LIBERO 上复现（非原文 SimplerEnv/RoboCasa），须注明是"适配复现"，不与 LangForce 原文 +11.3% 数字并列。

## 2.5 评测器设计

- **LSG（拆分）**：对每个留出 s₀，分别记 LSG-hit=succ(正确指令)、LSG-miss=succ(错配指令，同场景其它 g)，主指标 LSG=hit−miss；达标须 hit 高且 miss 低（R10）。
- **重定向率**：s₀ 固定，指令 ℓ_i→ℓ_j，统计末态命中 g_j 的比例。
- **language dropout 探针**：语言 token 置零/[MASK]，测 SR 降幅（R11b），与错配指令配对呈现。
- **留出划分（三类）**：训练指令集与评测指令集不相交；(i) LIBERO-Plus 语言轴、(ii) CALVIN、(iii) paraphrase + 训练未见指令组合，均为训练未见（R13）。
- **统计**：≥3 seed，配对检验（臂4 vs 臂2a/2b/3/5），报告均值±std 与 p 值；seed std 应 ≤ 均值 1/3。

## 2.6 算力与排期映射（60 天）

| 阶段 | 天数 | 内容 | 卡 |
|------|------|------|----|
| P0 前置验证 | 1–7 | 对比组可构造性（含 DTW 早期分叉、目标互斥校验）、SmolVLA+LIBERO harness、奖励回读一致性 | 1×4090 |
| P1 臂1+臂2a/2b | 8–20 | SFT baseline + 单指令 RL + paraphrase 增强 RL，确立"问题存在 + 普通RL/语言增强都修不好" | 2–3×4090 |
| P2 臂4 CI-GRPO | 21–38 | 主方法训练+调参+LSG(hit/miss) 曲线 | 4–6×4090 |
| P3 臂3 + 臂5 | 39–48 | CAST 式反事实 SFT + LangForce 式损失侧 PMI（LIBERO 适配复现） | 2–3×4090 |
| P4 留出泛化+规模确认 | 49–56 | LIBERO-Plus/CALVIN/paraphrase + dropout 探针 + OpenVLA-OFT LoRA（短租 80GB） | 7×4090 +租卡 |
| P5 消融+叠加臂6+写作 | 57–60 | K/组大小/KL 消融 + 臂6 叠加实验 + 论文表格 | 混合 |

## 2.7 风险与预案

- **风险1 对比组构造不成立** → P0 一周内 go/no-go；回退程序化反事实指令。
- **风险2 CI-GRPO 训练坍塌 / 组内无方差** → KL 正则 + 防御性 shaping（监控触发）+ 回退 PPO；监控组内 r 方差分布（全 0=任务太难回查 base，全 1=谓词太松收紧）。
- **风险3 臂4 未显著胜臂3/臂5** → 诚实写入；转而分析"RL vs SFT vs 损失侧在语言 grounding 上的权衡"作为次要贡献；不包装。
- **风险4 关键词聋（学到 token→动作映射而非语义）** → paraphrase 留出（R13-iii）专门探测；若 paraphrase 塌，退一步 claim "治好关键词级失聪，语义级需额外语言增强"，方法仍有价值。
- **风险5 RIPT-VLA 授权** → 邮件确认；未获授权则 OpenVLA-OFT 规模确认改用 SimpleVLA-RL + 自加 LoRA，或砍掉 7B 只留 SmolVLA。
- **风险6 scaling claim 过度** → 仅 SmolVLA-450M + OpenVLA-OFT-7B 两点，**不画 scaling 曲线**，claim 降级为"compatible with modern VLA architectures"。

---

# 第三段 · 可勾选任务清单

## P0 前置验证（go/no-go，第 1–7 天）

- [ ] **T0.1** 确认 LIBERO-Goal 支持"锁定像素一致 s₀ + 切换 BDDL 目标谓词"，产出可行性结论（参考 RA-LOOP `set_state_from_flattened` 经验）。
- [ ] **T0.2** 跑通 SmolVLA-450M + LIBERO rollout harness，单卡显存 ≤24GB（验收 R1）。
- [ ] **T0.3** 校验奖励回读：自研 BDDL 判定 vs 官方 evaluator 在 100 条抽样 100% 一致（验收 R2）。
- [ ] **T0.4** 构造 1 个对比组样例（K=3，像素一致，目标互斥），人工核对渲染与目标（验收 R4/R5）。
- [ ] **T0.4b** 实现并验证两条硬约束：目标两两互斥检查 `g_i∧g_j=False`（R4b）、前 N 步 DTW 早期分叉筛选（R5b）；在样例组上确认过滤生效。
- [ ] **T0.5** novelty-check 精查 "contrastive instruction / language grounding RL for VLA"，**重点核对 LangForce(2601.15197) 是否已覆盖我们的方法侧**（当前结论：诊断撞车、方法正交），产出撞车报告。
- [ ] **T0.6** 邮件确认 RIPT-VLA 授权状态；记录结论与预案分支。
- [ ] **T0.7** P0 决策会：go/no-go + 回退方案确定。

## P1 问题确立（第 8–20 天）

- [ ] **T1.1** 臂1 SFT baseline 训练（clean SR≈40–60% 作为 RL base）+ 评测，报告 LSG(hit/miss)（预期 hit 高、miss 高即 LSG≈0，验收 R14）。
- [ ] **T1.2** 实现评测器：LSG-hit/miss、重定向率、language dropout 探针、标准成功率（验收 R10–R12、R11b）。
- [ ] **T1.3** 臂2a 单指令 RL + 臂2b paraphrase 增强 RL，训练 + 评测。
- [ ] **T1.4** 对比臂1 vs 臂2a/2b 的 LSG，验证"普通 RL 与语言侧增强都修不好语言失聪"假设。

## P2 主方法（第 21–38 天）

- [ ] **T2.1** 实现对比组采样器（批量生成 {s₀, [(ℓ_k,g_k)]}），含 R6 跳过 + R4b/R5b 校验逻辑。
- [ ] **T2.2** 实现 CI-GRPO 组内相对优势 + observation masking（验收 R8）；监控组内 r 方差。
- [ ] **T2.3** 臂4 主训练（post-training，从臂1 冷启动）；记录 LSG(hit/miss)/重定向率随步数曲线。
- [ ] **T2.4** 调参：组大小 K×M（K=3–4）、KL 系数、学习率；默认关 shaping（验收 R9）。
- [ ] **T2.5** ≥3 seed 复跑，产出均值±std（std ≤ mean/3）。

## P3 最强基线（第 39–48 天）

- [ ] **T3.1** 实现 CAST 式反事实标签合成（同 s₀ 不同指令 → 动作标签），记录标签准确率。
- [ ] **T3.2** 臂3 反事实 SFT 训练 + 评测。
- [ ] **T3.3** 臂5 LangForce 式损失侧 PMI 适配到 LIBERO（双分支 + PMI 惩罚）训练 + 评测。
- [ ] **T3.4** 校验各臂预算等效（交互次数/数据量对齐，验收公平性 2.4）。

## P4 泛化与规模确认（第 49–56 天）

- [ ] **T4.1** 各臂在 LIBERO-Plus 语言轴上评测 LSG/重定向率（验收 R13-i）。
- [ ] **T4.2** 各臂在 CALVIN 长程指令上评测（验收 R13-ii）。
- [ ] **T4.3** **paraphrase + 训练未见指令组合** 留出评测 + language dropout 探针（验收 R13-iii、R11b；防关键词聋）。
- [ ] **T4.4** OpenVLA-OFT-7B LoRA 复现臂4（短租 80GB），确认方法随规模同号（验收 R3；scaling 只做 2 点、不画曲线）。
- [ ] **T4.5** 配对显著性检验：臂4 vs 臂2a/2b/3/5（验收 R15/R16）。

## P5 消融、叠加、证据链与写作（第 57–60 天）

- [ ] **T5.1** 消融：K 值、组大小、KL、有无 observation masking、奖励来源（稀疏 vs shaping）。
- [ ] **T5.2** 臂6 叠加实验（CI-GRPO + LangForce PMI），记录是否正交涨点。
- [ ] **T5.3** 整理证据链：命令、config hash、日志路径、seed（验收 R17）。
- [ ] **T5.4** 复核所有外部引用数字，fetch 核对 Tier，核不到留 "—"（验收 R18）。
- [ ] **T5.5** 出论文主表（各臂 × {LSG-hit/miss, 重定向率, dropout, SR} × 训练/留出）+ 消融表 + 帕累托图（LSG vs SR）。
- [ ] **T5.6** 若立论未达标，写诚实诊断段落（验收 R16）。

---

## 交接说明（给 Claude Code）

1. 本任务书为契约，实现前先读 `00_BACKGROUND.md` 理解根因（MI 论证）与 `01_RELATED_WORK.md` 的核实纪律。
2. **P0 是硬门**：T0.1/T0.4 决定方法是否成立，未通过不进 P1。
3. 复用 RA-LOOP 现有 harness/BDDL/eval 脚本经验（`ra_loop/`、`eval/`、`train/`），新代码全放 `ci_grpo/` 相关包，不改上游。
4. 任何"新增 trick/超参"前，须能一句话回答审稿人 "why this value"，否则不加。
5. 不确定处写诊断脚本实测，禁止连续猜想；一次只改一处。
