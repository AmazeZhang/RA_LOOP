# RA-LOOP stratified lambda checkpoint 独立评测结果 — 20260722

> 启动：2026-07-22 20:49 CST
> 完成：2026-07-22 22:21 CST
> 状态：12/12 exit 0、48 summaries、576/576 rows、无 incomplete，GPU 2--7 已释放。

## 汇总

训练前 warm-start baseline 为 anchor 24/24、fixed-L2 21/24、总计 45/48。

| lambda | checkpoint | anchor | fixed-L2 | 总计 |
|---:|---:|---:|---:|---:|
| 0 | step 5 | 24/24 | 21/24 | **45/48** |
| 0 | step 10 | 24/24 | 20/24 | 44/48 |
| 0 | step 15 | 23/24 | 21/24 | 44/48 |
| 0 | step 20 | 23/24 | 21/24 | 44/48 |
| 0 | step 25 | 24/24 | 20/24 | 44/48 |
| 0 | step 30 | 24/24 | 21/24 | **45/48** |
| 0.25 | step 5 | 24/24 | 20/24 | 44/48 |
| 0.25 | step 10 | 24/24 | 20/24 | 44/48 |
| 0.25 | step 15 | 23/24 | 21/24 | 44/48 |
| 0.25 | step 20 | 23/24 | 21/24 | 44/48 |
| 0.25 | step 25 | 23/24 | 20/24 | 43/48 |
| 0.25 | step 30 | 23/24 | 21/24 | 44/48 |

既有 lambda=0.5 最佳 step 10/20 也是 45/48，但它把 baseline 的 stove
`fixed_l2/init 2` 修复成成功，同时使 original `fixed_l2/init 1` 变成失败，一换一。

## Paired 观察

- lambda=0 step 5/30 与 baseline 的 48 条结果完全一致：失败仍为 drawer
  `fixed_l2/init 0,4` 与 stove `fixed_l2/init 2`；
- lambda=0 step 20 单独修复 drawer `fixed_l2/init 4`，但新增 original
  `anchor/init 4` 和 `fixed_l2/init 2` 两个失败，因此总分下降；
- lambda=0.25 step 5/10 新增 original `fixed_l2/init 1`，其余 baseline 失败不变；
- lambda=0.25 step 15/20/30 新增 original `anchor/init 4`，其余 baseline 失败不变；
- 两档权重都没有稳定修复 drawer；十二份 checkpoint 没有一份超过 baseline。

## 结论

mode stratification 能避免旧 mixed objective 的系统性 anchor regression，但单独调节
lambda recovery 不能产生净泛化提升。较低 lambda 不是缺失的关键超参数：lambda=0
最多退化为 baseline，lambda=0.25 更差；lambda=0.5 仅交换失败样本。

因此停止继续扫 lambda，也不选择训练后 checkpoint 替换 warm-start baseline。下一步
应验证覆盖不足假设：扩展到 10 个 Spatial tasks、更多 demos 与更多训练 init；若放大
覆盖后仍有 nominal regression，再加入 reference-policy/KL guard。当前 4 tasks ×
4 demos、每 checkpoint 48 条独立 episodes 的证据支持机制诊断，但不足以证明最终方法
有效。
