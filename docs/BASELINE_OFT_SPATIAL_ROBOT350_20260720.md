# OpenVLA-OFT libero_spatial / LIBERO-Plus Robot-init 350 baseline

## 1. 结论

2026-07-20 完成 suite-specific OpenVLA-OFT spatial checkpoint 在 LIBERO-Plus
`libero_spatial / Robot Initial States` 全部 350 个任务上的评测：

```text
Coverage: 350 / 350 (missing=0, duplicate=0)
Success:  135 / 350
Rate:     38.571429%（报告为 38.6%）
```

七片均完成 50/50，结果与 manifest 的任务名称、类别、难度逐条一致；checkpoint、
suite、seed、manifest SHA 和 render backend 元数据校验通过。

这是正式复现官方 **suite-specific OpenVLA-OFT** 的第一个 suite 中间结果。官方
Robot 31.9% 是四个 suite 使用各自 checkpoint 后的合并指标，不能直接用这个
spatial 38.6% 单独判断是否复现 31.9%；还需完成 object、goal 和 libero_10。

## 2. 与同任务 OFT_m 的配对比较

此前四套件联合训练的 OFT_m checkpoint 在相同 350 个 spatial Robot-init
任务上为 `83/350 = 23.7143%`。suite-specific spatial checkpoint 为
`135/350 = 38.5714%`：

```text
Absolute change: +14.8571 percentage points
Net successes:   +52 tasks
```

| OFT_m | OFT spatial | Tasks |
|---:|---:|---:|
| Fail | Fail | 194 |
| Fail | Success | 73 |
| Success | Fail | 21 |
| Success | Success | 62 |

它在 73 个原失败任务上转为成功，同时有 21 个原成功任务转为失败，净增加 52。
该配对结果直接说明 union/OFT_m 与 suite-specific OFT 是不同基线，不能混用。

## 3. 按难度等级

| Difficulty | Success / Tasks | Rate |
|---:|---:|---:|
| 1 | 81 / 90 | 90.0000% |
| 2 | 40 / 69 | 57.9710% |
| 3 | 11 / 75 | 14.6667% |
| 4 | 3 / 46 | 6.5217% |
| 5 | 0 / 70 | 0.0000% |

难度 2 到 3 仍有明显断层，difficulty 5 为 0/70。后续 robustness 方法需要重点
追踪 d3--d5，而不能只看总体成功率。

## 4. 按基础任务

| Base task（省略共同动作后缀） | Success / Tasks | Rate |
|---|---:|---:|
| between plate and ramekin | 14 / 32 | 43.8% |
| from table center | 20 / 35 | 57.1% |
| in top drawer of wooden cabinet | 20 / 40 | 50.0% |
| next to cookie box | 7 / 27 | 25.9% |
| next to plate | 10 / 37 | 27.0% |
| next to ramekin | 8 / 38 | 21.1% |
| on cookie box | 16 / 24 | 66.7% |
| on ramekin | 5 / 32 | 15.6% |
| on stove | 19 / 43 | 44.2% |
| on wooden cabinet | 16 / 42 | 38.1% |

## 5. 评测协议

| 项目 | 值 |
|---|---|
| Checkpoint | `/home/imc/models/ra-loop/openvla-oft-spatial` |
| HF repository | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| Suite/category | `libero_spatial / Robot Initial States` |
| Tasks / episodes | 350 / 350（每任务 1 episode） |
| Seed | 7 |
| Shards | 7 × 50 个确定性、不重叠 manifest |
| Physical GPUs | 1--7；GPU 0 未使用 |
| Render backend | OSMesa；未使用 EGL |
| Python | `/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python` |
| Python isolation | `PYTHONNOUSERSITE=1` |
| Launcher | `eval/launch_oft_spatial_robot350_tmux.sh --execute` |
| Tmux session | `ra_oft_spatial_robot350` |

会话于 2026-07-20 09:19:25 CST 启动，最后一个任务于 10:27:39 CST 完成，
七片 wall time 约 58--67 分钟。模型只做推理；没有训练或修改 checkpoint。

## 6. 分片结果与文件校验

| Shard | GPU | Success / Tasks | Rate | results.jsonl SHA-256 |
|---:|---:|---:|---:|---|
| 0 | 1 | 19 / 50 | 38.0% | `a8619d3c9c0344eb6332c08aac10fae5e1dba780d897ee41be68b346f0da153d` |
| 1 | 2 | 24 / 50 | 48.0% | `44863a542d7a6d7208cb3da015f28f47a6fadafe96520675257de2fa78ac42fc` |
| 2 | 3 | 15 / 50 | 30.0% | `e3680c4a9374838cf6941bead5ecc918a14dcb546ac69ae03ccdee3928663119` |
| 3 | 4 | 19 / 50 | 38.0% | `1d5aac673713edda85d793b3daee78a3af2d987585eea17c27985ff39804a14d` |
| 4 | 5 | 19 / 50 | 38.0% | `57a9970161e7e79eb672f33c3ab7f2a33f11c90715c43b19c7c0fee917667cc8` |
| 5 | 6 | 21 / 50 | 42.0% | `76203eb3acedc4553c1a78fb232f036f9c190e0735bd613f9ccad5a1968e677c` |
| 6 | 7 | 18 / 50 | 36.0% | `856e213c4c8c70b1740a2116f0420ef88b176605c2a11e0702883fcb0496c176` |

按 shard 0 到 6 的原始字节依次拼接后：

```text
SHA-256: 1ce4c90800dd8e8171db2e8c2807fb0906ee39788ec9512afb2ad5c870f52ac2
```

结果根目录：

```text
/home/imc/yzy/RA_LOOP/logs/openvla_oft_spatial_robot_full350_7gpu_20260720
```

验收时七个 evaluator 均已完成、无残留 evaluator 进程，GPU 1--7 回到 18 MiB
基础占用和 0% 利用率。tmux 会话保留为七个空 shell，便于人工检查。

## 7. 下一步

按逐步流程，下一步不是立即训练，而是下载并静态校验 suite-specific
`libero_object` checkpoint；完成后再单独准备 object 的 CPU-only dry-run 和正式
评测。object、goal、libero_10 全部完成后，才汇总四 suite OpenVLA-OFT Robot
指标并与官方 31.9% 对比。
