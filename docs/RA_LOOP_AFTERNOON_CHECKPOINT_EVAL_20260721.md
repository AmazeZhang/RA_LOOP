# RA-LOOP 下午训练 checkpoint 独立评测 — 20260721

> 启动：2026-07-21 20:18 CST  
> 完成：2026-07-21 21:04 CST（约 47 分钟）
> 状态：六份 checkpoint 全部 exit 0，288/288 episodes 完整；GPU 已释放。

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

每个 task 只有完整 12 条后才原子生成结果；launcher 拒绝复用已有 step 输出。

## 结论

训练前 warm-start step 5 基线为 anchor 24/24、fixed-L2 21/24、总计 45/48。
六个下午训练 checkpoint 均未超过该基线：

| checkpoint | anchor | fixed-L2 | 总计 | 相对基线逐样本变化 |
|---|---:|---:|---:|---:|
| baseline | 24/24 | 21/24 | 45/48 | -- |
| step 5 | 23/24 | 21/24 | 44/48 | +1 / -2 |
| step 10 | 23/24 | 20/24 | 43/48 | +0 / -2 |
| step 15 | 23/24 | 21/24 | 44/48 | +0 / -1 |
| step 20 | 23/24 | 20/24 | 43/48 | +0 / -2 |
| step 25 | 23/24 | 20/24 | 43/48 | +0 / -2 |
| step 30 | 23/24 | 20/24 | 43/48 | +0 / -2 |

step 5 修复了 stove 的 `fixed_l2/init 2`，这是本轮唯一新恢复成功；但同时使原任务
`anchor/init 4` 和 `fixed_l2/init 1` 失败。step 15 只退化原任务
`anchor/init 4`，是训练后最保守候选，但仍不能替代 45/48 的 warm-start 基线。
step 20--30 没有继续改善，说明延长当前配置不是合理的下一步。

本结果只有每 checkpoint 48 条 deterministic episodes，适合筛选和定位干扰，尚不
足以作统计显著性结论。下一步应保留 warm-start baseline，针对跨任务干扰加入 anchor
回归保护，并围绕 drawer/stove 的固定失败 pair 做方法消融，而不是直接继续加训练步数。
