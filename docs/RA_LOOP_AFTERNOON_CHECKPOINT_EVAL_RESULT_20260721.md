# RA-LOOP 下午训练 checkpoint 独立评测结果 — 20260721

## 验收

- 时间：2026-07-21 20:18--21:04 CST，约 47 分钟
- checkpoint：下午训练 step 5/10/15/20/25/30
- 规模：6 checkpoints × 4 tasks × 12 episodes = 288/288
- 每任务：6 anchor + 6 fixed-L2 0.1
- deterministic action mean，horizon 220；init/seed 与训练前评测精确匹配
- 24 个 task 输出全部生成 `summary.json`，无 `.incomplete`
- 六个进程均 `[AFTERNOON_EVAL_EXIT] 0`，GPU 1--6 已释放

## 汇总

这里的 baseline 是下午训练实际使用的 warm-start pilot step 5。它在同一组 48 个
`(task, mode, init, seed)` 上的结果为 45/48。

| 模型 | anchor | fixed-L2 | 总计 | gain | loss | 净变化 |
|---|---:|---:|---:|---:|---:|---:|
| warm-start baseline | 24/24 | 21/24 | 45/48 | -- | -- | -- |
| afternoon step 5 | 23/24 | 21/24 | 44/48 | 1 | 2 | -1 |
| afternoon step 10 | 23/24 | 20/24 | 43/48 | 0 | 2 | -2 |
| afternoon step 15 | 23/24 | 21/24 | 44/48 | 0 | 1 | -1 |
| afternoon step 20 | 23/24 | 20/24 | 43/48 | 0 | 2 | -2 |
| afternoon step 25 | 23/24 | 20/24 | 43/48 | 0 | 2 | -2 |
| afternoon step 30 | 23/24 | 20/24 | 43/48 | 0 | 2 | -2 |

任务顺序为：原始 black-bowl-to-plate、between-plate-and-ramekin、top-drawer、
stove。每格为 `anchor/6, fixed-L2/6`：

| checkpoint | 原任务 | between | drawer | stove |
|---|---:|---:|---:|---:|
| baseline | 6, 6 | 6, 6 | 6, 4 | 6, 5 |
| step 5 | 5, 5 | 6, 6 | 6, 4 | 6, 6 |
| step 10 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 15 | 5, 6 | 6, 6 | 6, 4 | 6, 5 |
| step 20 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 25 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 30 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |

## Paired 变化

- step 5 gain：stove `fixed_l2/init 2` 从失败变成功。
- step 5 loss：原任务 `anchor/init 4`、`fixed_l2/init 1` 从成功变失败。
- step 15 loss：原任务 `anchor/init 4`。
- step 10 loss：原任务 `anchor/init 4`、`fixed_l2/init 1`。
- step 20/25/30 loss：原任务 `anchor/init 4`、`fixed_l2/init 2`。
- drawer 原有的 `fixed_l2/init 0,4` 在所有 checkpoint 都未修复。

## 判断与下一步

1. 这轮训练有局部恢复信号：step 5 修复了一个 stove 扰动 pair。
2. 该信号不足以抵消跨任务干扰；所有训练后 checkpoint 总分都低于 warm-start。
3. 不用下午 checkpoint 替换当前 baseline。若后续必须保留一个训练后诊断候选，选
   step 15（44/48、只有一个 regression）；若专门研究恢复机制，保留 step 5 与
   baseline 做 paired 分析。
4. 当前曲线在 step 20--30 没有改善，不建议原配置继续加步数。
5. 下一轮方法实验应加入 anchor regression guard / reference-policy constraint，重点
   针对 drawer 的两个固定失败 pair 和 stove 的一个固定失败 pair，再使用相同 paired
   evaluator 验证。

48 条 deterministic episodes 的样本量适合 checkpoint 筛选，但不能支持强统计结论；
方法定型后仍需扩大 init/seed，并最终回到完整 benchmark 评测。
