# RA-LOOP 三任务短评测 — 20260721

> 运行：2026-07-21 12:35:23--12:50:22 CST（14分59秒）  
> 状态：6/6 进程 exit 0，72/72 episodes 完整，无 `.incomplete`。

## 设计

比较 base step 0 与单任务 pilot step 5，在三个未参与 pilot 训练的 Spatial 任务上
分别运行 6 anchor + 6 fixed-L2 0.1。两 checkpoint 使用相同 init indices 0--5、
perturb seeds 20260730--20260735 和 deterministic action mean。

## 结果

| unseen task | step 0 anchor | step 0 pert | step 5 anchor | step 5 pert |
|---|---:|---:|---:|---:|
| between plate/ramekin | 6/6 | 6/6 | 6/6 | 6/6 |
| top drawer | 6/6 | 4/6 | 6/6 | 4/6 |
| on stove | 6/6 | 5/6 | 6/6 | 5/6 |
| **合计** | **18/18** | **15/18** | **18/18** | **15/18** |

两 checkpoint 均为 33/36。失败也逐对完全相同：top-drawer fixed-L2 init 0/4，
on-stove fixed-L2 init 2。step 5 平均 action chunks 略低，但差异不足以作为性能证据。

## 结论

step 5 没有破坏这三个 unseen task，但也没有显示出跨任务 robustness 提升。结合上一轮
已训练任务上 step 0=19/20、step 5=20/20，step 5 可作为低训练量 warm-start；不能
声称它优于 base 或其他 checkpoint。新训练需要真正混合多个任务，而不是继续只在已
饱和的原任务上追加 steps。

原始结果：`logs/ra_loop_disambiguation_eval_20260721/task{0,1,2}_step{0,5}/`。

