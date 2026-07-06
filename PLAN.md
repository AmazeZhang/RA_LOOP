# SmolRoBot → RoRIPT-VLA：可行性优先的实验方案

> **重要变更**：原SmolVLA+CFG+3DGS+RL方案因技术栈未验证风险过高，改为 **OpenVLA-OFT + RIPT-VLA + Robustness-Aware Reward** 主线。
>
> 硬件：7×RTX4090 (24GB, 无NVLink)
> 时间：3个月
> 目标：AAAI-27 / ICLR-27 / CoRL-26 主会 (或 workshop 保底)
> 日期：2026-07-06

---

## 0. 关于成功概率的坦白

在写具体方案之前，先把风险全告诉你，避免"隐瞒核心组件失效"的情况：

**领域宏观风险**（不因方案变动而消失）：
1. VLA是2024-2025年最热的方向之一。3个月内做出能中AAAI/ICLR的工作，即使对老手也是挑战。**保守估计：主会成功率 20-35%，workshop 60-75%，arXiv preprint 95%+**。
2. 你之前主要做memory/agent，切入具身智能的学习曲线（MuJoCo/机器人坐标系/RL训练稳定性）约需2-3周。
3. 4090无NVLink对多卡通信是硬约束。任何要求NVLink或H100的方法（π0-RL, SimpleVLA-RL, RLinf-VLA的完整功能）都用不了。

**方案级风险**（本方案特有）：
1. RIPT-VLA的LOOP算法在LIBERO-Plus上还没有公开成绩，我们是首个跑通者——好处是novelty，坏处是没有先例参考。
2. "Robustness reward" 的设计成不成功要到Week 5-6才知道，如果失败，需要及时切backup (纯SFT+data augmentation路线)。
3. OpenVLA-7B QLoRA + RIPT-VLA rollout在24GB上是bs=1紧运行，训练时间会拉长。

**如果你想更保守**：可以把目标降为 "复现OpenVLA-OFT在LIBERO-Plus的robustness特性 + 一个reward shaping的ablation study"，投CoRL workshop或ICLR Robot Learning workshop，成功率可以拉到70%+。

---

## 1. 实验原理

### 1.1 问题定义

**VLA的鲁棒性缺口**：当前VLA (Vision-Language-Action) 模型在标准LIBERO benchmark上做到95%+，但在扰动版benchmark (LIBERO-Plus, LIBERO-PRO) 上表现暴跌：

- OpenVLA-OFT: 原版LIBERO 97% → LIBERO-Plus 67.9%
- π0-FAST: 原版98% → Plus 75%左右
- RoVLA (2025-10 SOTA): 82.0%
  - 其中 Robot-init 32%, Layout 74% ← 明显缺口

**根本原因**：SFT训练数据的初始条件分布过窄，模型学到的策略对**未见过的初始机器人位姿 (Robot-init)** 和 **未见过的物体布局 (Layout)** 极其敏感。

### 1.2 核心假设

**假设H1**：如果在RL阶段显式引入"扰动一致性奖励"（同任务在不同扰动下action应保持相似），模型的鲁棒性会显著提升，且不需要额外的SFT数据增广。

**假设H2**：这一提升在Robot-init和Layout两个维度上最显著，因为这两维度是SFT无法通过重放demo数据修复的（demo本身就是特定init拍的）。

**假设H3**：LOOP (Leave-One-Out) 无critic算法能在24GB VRAM下稳定训练OpenVLA-OFT-7B (QLoRA)，训练成本可以接受（<1000卡时）。

### 1.3 方法核心：Robustness-Aware LOOP (RA-LOOP)

**基础算法**：RIPT-VLA的LOOP (arXiv 2505.17629)
- 对每个初始状态采样K条trajectory
- 每条的advantage = 该条reward - K条平均reward (leave-one-out baseline)
- 无需critic网络，显存节省50%

**我们的改动**：在rollout时对**同一任务采样不同的扰动初始条件**，reward除了task success外，加入：
```
R_total = R_success + λ_c · R_consistency + λ_r · R_recovery

R_consistency = -||π(o_perturbed) - π̄||²   # action空间一致性
R_recovery   =  I[从扰动init仍到达goal]     # 恢复奖励
```

其中 `π̄` 是同任务K条trajectory的平均action分布。

**技术直觉**：把robustness从evaluation-only指标变成training signal，让模型在训练时就"知道"扰动的存在。

### 1.4 与前人工作的区别

| 工作 | 方法 | 需要RL | 需要额外SFT数据 | Plus总分 |
|------|------|--------|----------------|---------|
| OpenVLA-OFT (base) | 纯SFT | ✗ | — | 67.9% |
| RoVLA | SFT + geometry augment | ✗ | 需要 (3DGS渲染) | 82.0% |
| AVA-VLA | Adversarial SFT | ✗ | 需要 | 74.7% |
| RIPT-VLA (原版) | RL但只做原版LIBERO | ✓ | ✗ | 未报告 |
| **本工作 (RA-LOOP)** | **RL + robustness rewards** | ✓ | ✗ | **目标 78-82%** |

**卖点定位**：不是打SOTA，而是**"零额外数据 + RL的robustness路线首次系统评估"**。审稿人可能追问的问题都能回答：
- "为什么不用RoVLA的3DGS数据？" → "证明纯RL能匹配数据增广，工程成本更低"
- "为什么不打SOTA？" → "SOTA需要大规模合成数据，我们证明RL是正交且互补的方向"

---

## 2. 实验步骤（12周）

### Phase 1: 环境与Baseline (Week 1-2)

#### Step 1.1 [W1] 环境搭建
```bash
# 主环境
conda create -n roript python=3.10
conda activate roript
git clone https://github.com/Ariostgx/ript-vla ~/code/ript-vla
cd ~/code/ript-vla
pip install -e .
pip install flash-attn --no-build-isolation

# LIBERO仿真
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ~/code/LIBERO
cd ~/code/LIBERO && pip install -e .

# LIBERO-Plus评估套件
git clone https://github.com/sylvestf/LIBERO-plus ~/code/LIBERO-plus
```

#### Step 1.2 [W1] 权重下载
- HF: `moojink/openvla-7b-oft-finetuned-libero-object` (或 libero-spatial/goal/long, 4个都下)
- HF: `Sylvest/LIBERO-plus` → assets.zip (扰动配置)

#### Step 1.3 [W2] Baseline复现
- 目标：在7×4090上跑通OpenVLA-OFT在LIBERO原版4个suite的评估，成绩落在官方报告±3%内
- 目标：跑通LIBERO-Plus评估，得到67-70%的baseline number
- 目标：跑通RIPT-VLA官方的LOOP训练脚本（LIBERO-90 subset），能看到reward上升曲线

**Step 1.3 Gate**：如果任何一个复现偏差>5%，暂停投入，先debug再继续。

### Phase 2: 原型算法 (Week 3-4)

#### Step 2.1 [W3] Robustness rollout wrapper
- 在 `ript_vla/envs/libero_wrapper.py` 中扩展，使每个rollout batch内的K条trajectory采样不同的robot-init/layout扰动
- 复用LIBERO-Plus的扰动生成代码 (`assets.zip` 里的perturbation config)

#### Step 2.2 [W4] Reward函数实装
- 实装 `R_consistency` 和 `R_recovery`
- 加入LOOP的advantage计算
- **关键超参**：λ_c ∈ {0.1, 0.5, 1.0}, λ_r ∈ {0.5, 1.0, 2.0}，先跑网格搜索的最小方格 (2×2=4组)

**Step 2.4 Gate (Month 1 end)**：
- 成功标准：至少一组超参在LIBERO-Plus Robot-init子集上比OpenVLA-OFT baseline高 ≥3pt
- 失败标准：所有超参组合都比baseline低 或 训练崩溃 → 触发plan B

### Phase 3: 主实验 (Week 5-8)

#### Step 3.1 [W5-6] Full training
- 用Month 1找到的最优超参
- 在LIBERO-90全集上训练 (原版90个任务 + LIBERO-Plus的扰动injection)
- 训练量：~500-800 GPU-hours (7×4090 ≈ 3-5天)

#### Step 3.2 [W7] 全维度评估
- LIBERO-Plus 7维度 + Total
- LIBERO 原版 4个suite (防止rob-training伤害原版性能)
- **额外**：LIBERO-PRO作为generalization测试（不训练但评估）

#### Step 3.3 [W8] Ablation matrix
| 配置 | 说明 |
|------|------|
| A0 | OpenVLA-OFT baseline (无RL) |
| A1 | LOOP w/o robustness rewards (纯task success) |
| A2 | LOOP + R_consistency only |
| A3 | LOOP + R_recovery only |
| A4 | Full RA-LOOP |
| A5 | Full + 更长训练 (2x steps) |

### Phase 4: 写作 (Week 9-12)

- W9: 主表 + 3张figure (维度雷达图 / 训练曲线 / consistency可视化)
- W10: Introduction + Method章节
- W11: Related work + Discussion + limitations
- W12: 反例分析 + arXiv投稿

---

## 3. 可行性分析（分组件）

### 3.1 硬件可行性

| 组件 | 显存需求 | 4090兼容 | 备注 |
|------|---------|----------|------|
| OpenVLA-OFT-7B推理 | ~14GB (bf16) | ✓ | 单卡装得下 |
| OpenVLA-OFT-7B QLoRA训练 | ~18-22GB (bs=1, 4bit) | ✓ (紧) | 需gradient checkpointing |
| LIBERO环境rollout | ~2GB/env | ✓ | 每卡跑10-15env并行 |
| RIPT-VLA LOOP (K=8) | 显存×1.2倍vs SFT | ✓ | critic-free省一半 |

**多卡策略**：7卡分工
- Card 0-4: Actor rollout (每卡10 env × 5 = 50 envs)
- Card 5-6: Actor training (DDP, QLoRA)

**关键风险**：PCIe Gen4带宽 ≈ 64GB/s, FSDP-style参数同步会瓶颈。但QLoRA只训练adapter (<300MB)，同步开销可以接受。

### 3.2 软件可行性

| 依赖 | 状态 | 风险 |
|------|------|------|
| RIPT-VLA 官方repo | 开源，活跃 | 低 |
| OpenVLA-OFT 权重 | HF公开 | 低 |
| LIBERO仿真 | 广泛使用 | 低 |
| LIBERO-Plus 评估 | 开源3个月 | 中 (issue较多) |
| MuJoCo 3.3.2 | 稳定 | 低 |

### 3.3 数据可行性

- 训练数据：LIBERO-90 (~130GB, 已存在HF `HuggingFaceVLA/libero`)
- 评估数据：LIBERO-Plus (75.5GB) + LIBERO-PRO (~30GB)
- **不需要**任何数据合成/3DGS渲染

### 3.4 时间可行性

给定7×4090，估算：
- Baseline评估：~200 GPU-hours (Week 1-2)
- 原型探索：~300 GPU-hours (Week 3-4)
- 主训练：~800 GPU-hours (Week 5-6)
- Ablation：~1000 GPU-hours (Week 7-8)
- **总计**：~2300 GPU-hours = 7×24×14天 = 2352小时刚好覆盖

**时间紧张，无返工余量**。所以Step 1.3和Step 2.4两个gate必须严格执行。

---

## 4. 每步的成功预期与失败选择

### Step 1.3 Baseline复现 (Week 2 end)

| | 情况 |
|--|--|
| **✅ 成功标准** | OpenVLA-OFT: LIBERO原版4 suites 平均 ≥94%，LIBERO-Plus Total 65-72% |
| **⚠️ 部分成功** | 一个suite偏差5-8% → 排查config，最多花3天，可以带偏差继续 |
| **❌ 失败** | 多个suite偏差>10% 或 完全跑不通 |
| **失败选择** | Plan B1: 换成SmolVLA (450M) + 官方`HuggingFaceVLA/smolvla_libero` 起步。虽然放弃了RL，但SmolVLA的LIBERO评估pipeline在zuoxingdong的repo里被反复验证。<br>Plan B2: 如果连SmolVLA也不行，说明是环境问题，退回arXiv 2410.05243的最小复现package。 |

### Step 2.4 原型算法 (Week 4 end) ⭐ 最关键节点

| | 情况 |
|--|--|
| **✅ 成功标准** | 至少一组 (λ_c, λ_r) 在Plus-Robot-init上比baseline高 ≥3pt，且原版LIBERO不掉>2pt |
| **⚠️ 部分成功** | 训练收敛但只在某一维度有效 → 继续深挖那一维度，story变窄但仍可发 |
| **❌ 失败核心组件** | **诚实告诉你的情况**：<br>a) LOOP+robustness rewards导致training不收敛（reward hacking或梯度爆炸）<br>b) 收敛但成绩不比baseline好<br>c) Robot-init好了但原版LIBERO掉超过5pt |
| **失败选择** | Plan C1: **切纯SFT + adversarial augmentation路线**。用OpenVLA-OFT + LIBERO-Plus的扰动数据混入SFT，走"data-side robustness"。Novelty降级但保命。<br>Plan C2: **切方向C** (memory-VLA on MIKASA-Robo)，方案已经在前文文档中备好。<br>Plan C3 (核选项): **写negative result paper**，坦诚"we found RL-based robustness doesn't work OOTB, and here's why"，投workshop有一定几率。 |

### Step 3.1-3.2 主训练与评估 (Week 6-7)

| | 情况 |
|--|--|
| **✅ 成功标准** | Plus Total 76-82%，Robot-init ≥40%，Layout ≥78% |
| **⚠️ 部分成功** | Total 72-75%，只在1-2维度显著提升 → 论文focus这1-2维，其它维度honest report |
| **❌ 失败** | Total ≤ baseline (67.9%) |
| **失败选择** | Plan D: 承认"我们的方法在完整规模上无法scale"，回到Week 4的最优超参组做深入分析，转向understanding paper（分析robustness与RL的关系），投ICLR/NeurIPS workshop。 |

### Step 3.3 Ablation (Week 8)

| | 情况 |
|--|--|
| **✅ 成功标准** | 6组ablation之间有清晰trend，R_consistency和R_recovery各自贡献可分离 |
| **⚠️ 部分成功** | Trend不清晰 → 只在主表report Full config，Ablation降级为appendix |
| **❌ 失败** | Ablation结果自相矛盾 (如A4<A3) → **诚实报告**，讨论interference between rewards |

---

## 5. 特别声明：可能导致方向无效的核心风险

按照要求诚实列出所有可能让整个方向"无效"的情况，不做隐瞒：

### 风险R1: LOOP在OpenVLA-OFT上根本训不起来
**触发条件**：QLoRA + LOOP + K=8 rollout 超出24GB
**发现时点**：Step 2.1 (Week 3)
**证据**：RIPT-VLA官方在24GB卡验证过LOOP，但只在OpenVLA-7B原版上，OFT版参数量略多且没有官方RL复现
**应对**：降K到4或6, 或换Octo-Small (93M)，但Octo原版LIBERO成绩较低（60-70%），novelty空间受限

### 风险R2: LIBERO-Plus评估的determinism问题
**触发条件**：LIBERO-Plus刚发布3个月，评估脚本可能有随机性未固定，导致数字波动
**发现时点**：Step 1.3 baseline复现
**证据**：github.com/sylvestf/LIBERO-plus 的issue #1-#10已有multiple report seed不稳定
**应对**：每个config跑3-5个seed取平均，工作量增加~30%

### 风险R3: RA-LOOP的核心假设H1不成立
**触发条件**：consistency reward在训练时看起来收敛，但对eval时的robustness无提升
**发现时点**：Step 2.4 (Week 4)
**根本原因**：如果模型学到"consistency"只是在训练分布上的consistency，不能generalize到eval分布
**应对**：这是**方向级失败**，除转Plan C之外无解

### 风险R4: 3个月对新手不够
**触发条件**：Week 1-2 环境搭建就花掉了3周（MuJoCo版本冲突、CUDA driver、HF访问慢等）
**发现时点**：Week 3 开始时
**应对**：立即启动plan D-workshop路线，目标从AAAI/ICLR降为CoRL workshop / ICLR Robot Learning workshop

### 风险R5: 学术圈已经在做同样的事
**触发条件**：2026-08 到 2026-10 有其他团队arxiv了"RL for VLA robustness"论文，且做得比我们好
**发现时点**：Week 7-8 (投稿前scan最新论文时)
**应对**：追加distinguish章节，emphasize我们的unique angle (LOOP+robustness reward的组合)；如果差异化不足，转workshop

---

## 6. Week 1 立即执行清单

给你4条命令，Week 1可以直接跑：

```bash
# 1. clone仓库
cd ~ && mkdir -p code && cd code
git clone https://github.com/Ariostgx/ript-vla
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
git clone https://github.com/sylvestf/LIBERO-plus

# 2. 创建conda环境
conda create -n roript python=3.10 -y
conda activate roript

# 3. 安装
cd ~/code/ript-vla && pip install -e . && cd ~/code/LIBERO && pip install -e .
pip install mujoco==3.3.2 flash-attn --no-build-isolation

# 4. 下载权重 (需要HF账号并接受OpenVLA license)
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-object --local-dir ~/models/openvla-oft-object
huggingface-cli download Sylvest/LIBERO-plus --include "assets.zip" --local-dir ~/data/libero-plus
unzip ~/data/libero-plus/assets.zip -d ~/data/libero-plus/
```

跑通后Week 2的评估脚本，我拆到`week1_baseline.md`里另写。

---

## 7. 三个月后的最坏情况

假设一切失败（虽然概率不高），你至少拥有：

1. 一套本地跑通的VLA + RL环境（4090上可用），可以复用到任何后续项目
2. 对LIBERO/LIBERO-Plus benchmark的深入理解
3. 一份"我们尝试了X，发现Y失效"的技术报告，可以作为group meeting分享
4. RIPT-VLA / OpenVLA-OFT 的deep expertise，行业内不多

即使不发论文，这些也是硬技能积累。

---

## 8. 现在需要你确认

1. 接受这个更保守但成功率更高的方案？(取代之前SmolVLA的方案)
2. 你的HuggingFace账号是否已经accept过 openvla/openvla-7b 的license？(不然下不了权重)
3. 是否可以现在（Week 1）就开始跑环境搭建？

回一句"go"我就把Week 1的详细执行任务书拆出来。
