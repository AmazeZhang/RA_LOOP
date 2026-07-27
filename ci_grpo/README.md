# CI-GRPO 项目文档集

对比式指令强化学习修复 VLA 语言失聪（Contrastive-Instruction GRPO）。
目标 ICLR 2027，约 60 天窗口，7×RTX 4090（可短租单张 80GB）。

## 文档导航

- [`00_BACKGROUND.md`](./00_BACKGROUND.md) — 背景与动机：VLA 语言失聪现象（三条 Tier A 硬证据）、互信息根因分析、方法概览、四臂对照与算力现实。
- [`01_RELATED_WORK.md`](./01_RELATED_WORK.md) — 文献综述：语言失聪/鲁棒性评测、VLA RL 后训练、novelty-check 相邻工作、基座与基准，每篇附 arXiv ID 与核实等级（Tier A/B/⚠️）。
- [`02_TASKBOOK.md`](./02_TASKBOOK.md) — 三段式任务书：EARS 验收标准 → 技术设计 → 可勾选任务清单（P0–P5，60 天排期），交 Claude Code 执行。
- [`03_EXPECTATIONS_AND_EXPLORATION.md`](./03_EXPECTATIONS_AND_EXPLORATION.md) — 发表级实验预期（三层可量化门槛 + 一句话故事模板）、失败信号（P0/立论/稳定性三档早停判据 + 止损时间盒）、可钻研方向（主线深挖 / 后续论文种子 / 高风险开放问题）。

## 一句话方法

同一初始视觉状态 s₀ × K 条冲突指令构成 GRPO 组，只有 rollout 达成"当前指令对应目标"才给奖励；策略若忽略语言，组内期望奖励 ≤1/K，被组内相对优势惩罚，从而被迫把动作因果地条件在语言上。用 LSG（语言敏感度差距）与指令交换重定向率量化"到底听没听懂"。

## 状态

文档阶段（spec）。P0 前置验证（对比组可构造性 + harness 打通）为动手硬门，未通过不进 P1。
