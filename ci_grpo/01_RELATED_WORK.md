# 文献综述：VLA 语言 Grounding 与 RL 后训练

> 配套文档：`00_BACKGROUND.md`（动机）、`02_TASKBOOK.md`（任务书）
> **核实等级约定**：
> - **Tier A** = 已逐一 fetch arXiv 原文/摘要核实，ID、结论、数字均对齐；可直接引用。
> - **Tier B** = 检索到但未完整核实原文，引用前须再 fetch 核对 metric 与 Judge 协议。
> - **⚠️ 未证实** = 2026 年 arXiv ID，检索层可能 confabulate，**引用前必须 fetch 原页确认存在**，否则留 "—"。
>
> 铁律（用户偏好）：读论文须读原文，不凭标题推测；跨 Judge 协议的数字对比无效；核不到的数字一律留 "—"，不 hedging。

---

## A. 语言失聪 / 鲁棒性评测（本项目立论根基，Tier A）

| arXiv | 名称 | 关键结论（本项目用法） |
|-------|------|------------------------|
| 2510.13626 | LIBERO-Plus（CVPR 2026） | 七维扰动；**语言维度替换后成功率几乎不降**（语言失聪直接证据）；视角扰动 95%→<30%（视觉捷径证据）。用作现象证据 + 留出泛化的语言轴。 |
| 2510.03827 | LIBERO-PRO | 受控"损坏指令"协议；破坏指令后名义 90% 可掉到 0%，暴露分数虚高。用作评测协议参考 + LSG 设计灵感。 |
| 2510.00037 | RobustVLA | 多模态扰动下**动作模态最脆弱**，语言 grounding 分布外失效。支撑"grounding 是共现幻觉"论点。 |
| 2510.01711 | RS-CL（ICML 2026） | 鲁棒性相关，视觉扰动的对比学习视角。作为相关工作对照。 |

补充鲁棒性/评测（Tier A，引用可选）：BYOVLA 2410.01971、RoboArena 2506.18123、SureSim 2510.04354、Data Scaling Laws for Robot Learning 2410.18647。

---

## B. VLA RL 后训练（对手 + 可复用基线，Tier A）

| arXiv | 名称 | 关键结论 / 算力 | 本项目定位 |
|-------|------|-----------------|-----------|
| 2505.19789 | RLVLA | **PPO > GRPO > DPO**；1×A100-80G，42h | "RL 已有用"威胁；即我们的**臂 2**（任务成功率 RL），要证其修不好 LSG |
| 2505.18719 | VLA-RL | ~48 GPU-h，成功率导向 RL | 相关工作；无冲突指令组 |
| 2505.17016 | RIPT-VLA | OpenVLA-OFT + **LoRA**，LOOP 算法，3–4 GPU；**仓库无 license** | 可复用 LoRA rollout 路径（须先邮件确认授权） |
| 2506.08440 | TGRPO | 轨迹级 GRPO for VLA | 方法对照；无语言对比 |
| 2509.09674 | SimpleVLA-RL | veRL/GRPO，**group-by-uid 现成**，8×A800-80G，**无 LoRA** | 主要代码参考（group 构造、reward manager、LIBERO utils） |
| 2501.16664 | iRe-VLA | 迭代 RL + BC | 相关工作 |
| 2502.05450 | ConRFT | 一致性 RFT | 相关工作 |
| 2411.19309 | GRAPE | 偏好对齐 RL for VLA | 成功率/偏好导向，无语言对比 |
| 2510.00406 | VLA-RFT | RFT for VLA | 相关工作 |
| 2506.17811 | RoboMonkey | test-time scaling | 相关工作（推理侧） |
| 2510.05681 | MG-Select | test-time 选择 | 相关工作（推理侧） |
| 2505.15660 | AGNOSTOS | 跨任务泛化评测 | 泛化评测对照 |

**可复用代码定位（SimpleVLA-RL，github PRIME-RL/SimpleVLA-RL，MIT）**：
- group 构造：`ray_trainer.py` 的 group-by-uid
- 奖励：`RobRewardManager` / `main_ppo.py`
- LIBERO 接入：`libero_utils.py`
- 注意：无 LoRA，全参需 8×80GB → 本项目仅借其组/奖励接口，主力用 SmolVLA。

**RIPT-VLA（github Ariostgx/ript-vla）**：LoRA for OpenVLA-OFT，LOOP 在 `ript/algos/rl_optimizers/rl_optimizer.py`，3–4 GPU；**无 license 文件，动手前必须邮件确认授权**。

---

## C. Novelty-check 相邻工作（差分论证核心，Tier A）

| arXiv | 名称 | 关键事实（已核实） | 与本项目差分 |
|-------|------|--------------------|--------------|
| 2508.13446 | **CAST** | base=3B PaliGemma；Gemini 2.5 Pro relabel 合成动作标签（准确率 60–70%）；GNM+BridgeData v2；真机导航+操作；nav +27% / manip 2×（35/60 vs 19/60）；**Lemma IV.1 互信息形式化**；**纯 SFT，无 swap 指标** | **最强 incremental 对手**。CAST=数据侧 SFT+噪声标签+真机+无因果语言指标；本项目=奖励侧 RL+无标签+仿真+LSG/重定向率。作为**臂 3** 适配到仿真复现 |
| 2307.00117 | GRIF | 语言-目标对齐表征 | 表征对齐，非 RL 对比奖励 |
| 2211.11736 | DIAL | 指令增强数据 | 数据增强，无对比 RL |
| 2005.07648 | LangLfP | play data 语言条件 | 早期语言条件，无对比 |
| 2206.08522 | VLMbench | 语言引导操作基准 | 评测基准对照 |

⚠️ **未证实（2026 ID，引用前必须 fetch 核实存在性）**：VLA Grounder 2607.04517。当前一律按"待核实"处理，不进正文数字。

---

## D. VLA 基座模型（选型与背景，Tier A）

| arXiv | 名称 | 备注 |
|-------|------|------|
| 2506.01844 | **SmolVLA-450M** | Apache-2.0，单卡全参可训，**本项目主力基座** |
| 2502.19645 | **OpenVLA-OFT** | 7B，LoRA 单张 80GB，**规模确认基座** |
| 2406.09246 | OpenVLA | 7B 开源 VLA 起点 |
| 2410.24164 | π0 | flow-matching 动作专家 |
| 2504.16054 | π0.5 | π0 后续，开放世界泛化 |
| 2410.07864 | RDT-1B | diffusion VLA |
| 2405.12213 | Octo | 通用机器人策略 |
| 2503.14734 | GR00T N1 | NVIDIA 人形基座 |
| 2411.19650 | CogACT | 认知-动作分解 |
| 2501.15830 | SpatialVLA | 空间表征增强 |
| 2412.10345 | TraceVLA | 视觉轨迹提示 |
| 2307.15818 | RT-2 | VLA 奠基工作 |

---

## E. 基准与数据集（Tier A）

| arXiv | 名称 | 用途 |
|-------|------|------|
| 2306.03310 | LIBERO | **主基准**（LIBERO-Goal/Object/Spatial/10），MuJoCo 单卡 |
| 2112.03227 | CALVIN | 长程语言指令，**留出泛化** |
| 2405.05941 | SimplerEnv | 真-仿一致性评测 |
| 2410.00425 | ManiSkill3 | 备选仿真 |
| 2406.02523 | RoboCasa | 大规模家庭场景 |
| 2310.08864 | Open X-Embodiment | 跨具身数据 |
| 2403.12945 | DROID | 真机大规模数据 |

---

## F. 引用纪律小结

1. 正文任何数字，引用前当场 fetch 对应 arXiv 摘要/表格核对；跨 Judge/协议的数字不并列比较。
2. 2026 年 ID（26xx）默认存疑，逐一 fetch 核实存在性后方可引用，否则留 "—"。
3. CAST 的 60–70% 标签准确率、nav+27%/manip 2× 已核实，可用于差分论证。
4. RLVLA 的 PPO>GRPO>DPO 已核实，用于臂 2 的方法选择说明（注：本项目主线用 GRPO 因其组内相对优势与对比组同构；若 pilot 显示 GRPO 不稳，按 RLVLA 结论回退 PPO）。
