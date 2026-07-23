# RA-LOOP Spatial 全任务 lambda 2×2 训练结果 — 20260723

> 启动：2026-07-22 23:39 CST
> 完成：2026-07-23 16:11--16:43 CST
> 状态：四组 100/100、`Finished training`、`[RA_LOOP_EXIT] 0`，GPU 4--7 已释放。

## 训练验收

| lambda | seed | 训练循环 | 在线 anchor | 在线 perturbed | 在线总计 | 非零更新步 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 10000 | 17:02:40 | 298/400 | 259/400 | 557/800 | 89/100 |
| 0.5 | 20000 | 16:57:13 | 290/400 | 282/400 | 572/800 | 89/100 |
| 0 | 10000 | 16:45:41 | 304/400 | 281/400 | 585/800 | 88/100 |
| 0 | 20000 | 16:30:31 | 302/400 | 291/400 | 593/800 | 84/100 |

四组均完整记录 100 个 metrics、100/100 使用 mode-stratified，lambda effective 全程
与配置一致。每个任务被采样 7--14 steps，合计 100，符合 10-task shuffle 预期。

相同 seed 下 lambda=0 的在线成功数均高于 lambda=0.5，但在线 rollout 由当前策略行为、
dynamic skip 和训练路径共同决定，不能作为 paired 泛化比较，也不能直接选 checkpoint。

## Checkpoint 与空间验收

上游在 update 前按 interval 保存，因此每组保存 step 10/20/30/40/50/60/70/80/90
九份，不存在 step 100；这与此前 35-step run 不保存 step35 的语义一致。

36 份 checkpoint 均通过张量级读取：LoRA 879 tensors、headers 34 tensors、
`adapter_config.json` 齐全，无 `.incomplete` 或临时文件。每组约 9.5 GB，总计约 38 GB；
验收后磁盘仍可用 175 GB。

## 下一步评测

不直接对 36 份 checkpoint 全部做 10-task 评测。先用完全固定的 10-task paired set
评测 warm-start baseline，并筛选每个 run 的 step 30/60/90：共 13 个候选，每个候选
为 10 tasks ×（6 anchor + 6 fixed-L2）=120 episodes。六卡约三轮。若中间 checkpoint
出现净提升，再补评邻近 step；若所有候选均不超过 baseline，则停止相同目标的长训，
转向 reference-policy/KL guard。
