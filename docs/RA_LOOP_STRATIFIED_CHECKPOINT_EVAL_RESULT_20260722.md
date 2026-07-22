# RA-LOOP mode-stratified checkpoint 独立评测结果 — 20260722

> 启动：2026-07-22 09:13 CST
> 完成：2026-07-22 10:00 CST
> 状态：六进程 exit 0，288/288 episodes 完整，GPU 1--6 已释放。

## 汇总

训练前 warm-start baseline 为 anchor 24/24、fixed-L2 21/24、总计 45/48。旧
mixed-objective 训练最佳为 44/48；本次 mode-stratified 的 step 10/20 达到 baseline
的 45/48，但未超过 baseline。

| 模型 | anchor | fixed-L2 | 总计 | 相对 baseline gain/loss |
|---|---:|---:|---:|---:|
| warm-start baseline | 24/24 | 21/24 | 45/48 | -- |
| stratified step 5 | 24/24 | 20/24 | 44/48 | +0 / -1 |
| stratified step 10 | 24/24 | 21/24 | 45/48 | +1 / -1 |
| stratified step 15 | 23/24 | 20/24 | 43/48 | +0 / -2 |
| stratified step 20 | 24/24 | 21/24 | 45/48 | +1 / -1 |
| stratified step 25 | 23/24 | 20/24 | 43/48 | +0 / -2 |
| stratified step 30 | 24/24 | 20/24 | 44/48 | +0 / -1 |

每格为 `anchor/6, fixed-L2/6`：

| checkpoint | original | between | drawer | stove |
|---|---:|---:|---:|---:|
| baseline | 6, 6 | 6, 6 | 6, 4 | 6, 5 |
| step 5 | 6, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 10 | 6, 5 | 6, 6 | 6, 4 | 6, 6 |
| step 15 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 20 | 6, 5 | 6, 6 | 6, 4 | 6, 6 |
| step 25 | 5, 5 | 6, 6 | 6, 4 | 6, 5 |
| step 30 | 6, 5 | 6, 6 | 6, 4 | 6, 5 |

## Paired 结论

- step 10/20 修复了 stove `fixed_l2/init 2`；
- 同时所有 checkpoint 都使 original `fixed_l2/init 1` 从成功变失败；
- step 15/25 还使 original `anchor/init 4` 失败；
- drawer `fixed_l2/init 0,4` 在所有 checkpoint 中仍未修复。

与旧 mixed-objective 对比，新目标在 step 10/20 分别净改善 2 条，并把 anchor 从旧
训练的 23/24 恢复到 24/24。这支持 reward-interference 诊断：mode stratification
确实缓解了 anchor regression。但恢复一个 stove 扰动的同时稳定损失一个 original
扰动，说明 lambda recovery 对 perturbed group 的相对梯度权重仍可能过强或泛化不足。

## 判断与下一步

1. mode-stratified 修复是必要且有效的，但单独使用尚未超过 warm-start baseline；
2. step 10/20 是本轮最优诊断候选，不替换 baseline；
3. step 30 下降到 44/48，不支持在相同配置上盲目延长步数；
4. 下一项最有信息量的长时实验是保持 stratified，分别将 lambda recovery 从 0.5
   降到 0 和 0.25，检验 perturbed gradient scaling 是否导致 original perturbation
   regression；
5. 完成权重消融后再决定增加 demos/init 覆盖或加入 reference-policy guard。

当前每 checkpoint 只有 48 条 deterministic episodes，足以进行 paired 筛选，但仍不
是论文级统计结论。
