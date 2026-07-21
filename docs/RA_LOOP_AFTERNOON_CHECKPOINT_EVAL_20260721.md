# RA-LOOP 下午训练 checkpoint 独立评测 — 20260721

> 启动：2026-07-21 20:18 CST  
> 状态：六份 checkpoint 已并行进入首个 episode，预计 45--60 分钟。

## 设计

- checkpoint：下午训练 step 5/10/15/20/25/30
- GPU：1/2/3/4/5/6，一 checkpoint 一卡
- 每 checkpoint：四任务 ×（6 anchor + 6 fixed-L2 0.1）= 48 episodes
- 总计：288 episodes
- deterministic action mean，horizon=220，单环境
- 原任务使用 init 0--5 / seeds 20260720--20260725
- 三个 unseen tasks 使用 init 0--5 / seeds 20260730--20260735
- 与已有训练前 step 0 / pilot step 5 baseline 精确匹配

## 安全与状态

- CPU 24 plans、Python/shell syntax、项目回归 30/30 passed
- 启动前 GPU 1--6 均为 18 MiB、0%、41--43°C
- 六份 LoRA 与 headers 均已确认加载并进入首个 episode
- 启动后约 16.7 GiB、45--52°C
- shared tmux：`ra_loop_afternoon_checkpoint_eval`
- 日志：`logs/ra_loop_afternoon_checkpoint_eval_20260721/console_step{5,10,15,20,25,30}.log`
- 输出：`logs/ra_loop_afternoon_checkpoint_eval_20260721/step<step>/task<0..3>/`

每个 task 只有完整 12 条后才原子生成结果；launcher 拒绝复用已有 step 输出。完成后
验收 288/288，再按 anchor 保持、fixed-L2 改善和 checkpoint 曲线选型。

