# RA-LOOP checkpoint paired-eval 结果 — 20260721

> 完成区间：2026-07-21 09:41:13--09:55:34 CST（UTC+08:00）  
> 状态：五个进程全部 exit 0；100/100 条 episode 完整，无 `.incomplete`。

## 结果

| checkpoint | anchor | fixed-L2 0.1 | 总计 | anchor 平均 action chunks | 扰动平均 action chunks |
|---:|---:|---:|---:|---:|---:|
| step 0 | 9/10 | 10/10 | 19/20 | 13.7 | 12.7 |
| step 5 | 10/10 | 10/10 | 20/20 | 12.0 | 12.0 |
| step 10 | 10/10 | 10/10 | 20/20 | 13.6 | 12.1 |
| step 15 | 10/10 | 10/10 | 20/20 | 13.6 | 13.2 |
| step 20 | 10/10 | 10/10 | 20/20 | 13.8 | 12.7 |

step 0 唯一失败是 anchor `init_index=4`。对应 fixed-L2 样本成功；四个训练后
checkpoint 在该 anchor 上也都成功。

## 完整性检查

- 每个 checkpoint 都有 20 行、20 个唯一 `(mode, init_index)`。
- anchor/fixed-L2 各 10 条，init indices 均为 0--9。
- fixed-L2 的实际 joint-noise L2 全部为 `0.1`。
- perturb seeds 完整覆盖 20260720--20260729。
- 五份 summary 均为 `result_rows=20`，五个 console 均为 `[PAIR_EVAL_EXIT] 0`。
- step 5/10/15/20 summary 已记录各自 adapter/header SHA-256。

## 解释边界

这组结果支持：21-step pilot 至少没有破坏该 Spatial 单任务，在 step 5 时已把本组
样本从 19/20 提到 20/20。step 5 的 episode action-chunk 数也最短且最稳定，可作为
当前保守的临时候选。

但本评测已明显饱和，不能据此声称 step 5 优于 step 10/15/20，也不能证明统计显著
提升。step 0 的 perturbed 反而为 10/10，说明这 10 个扰动样本并非逐例都比 anchor
更难。评测只有一个任务、10 对初态，不能与论文的完整 LIBERO suite 数字直接比较。

## 下一步

在继续更长训练前扩大独立评测：优先增加任务和初态覆盖，并保留 paired anchor /
fixed-L2 设计。暂以 step 5 为低训练量候选，同时保留 step 20；用更强评测打破饱和后
再决定 checkpoint 和后续训练长度。

原始结果位于：
`logs/ra_loop_checkpoint_pair_eval_20260721/step{0,5,10,15,20}/`。
