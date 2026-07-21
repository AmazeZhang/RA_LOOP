# OpenVLA-OFT libero_object / LIBERO-Plus Robot-init 398 baseline

## 1. 结论

2026-07-20 完成 suite-specific OpenVLA-OFT object checkpoint 在 LIBERO-Plus
`libero_object / Robot Initial States` 全部 398 个任务上的评测：

```text
Coverage: 398 / 398 (missing=0, duplicate=0)
Success:  137 / 398
Rate:     34.422111%（报告为 34.4%）
```

六片完成 57/57，末片完成 56/56。结果与 manifest 的任务名称、类别、难度逐条
一致；checkpoint、suite、seed、manifest SHA 和 render backend 元数据校验通过。

这是 suite-specific OpenVLA-OFT 四套件复现中的第二个结果。官方 Robot 31.9%
是四个 suite 的合并指标，不能用 object 34.4% 单独判断最终复现是否通过。

## 2. 与同任务 OFT_m 的配对比较

四套件联合训练的 OFT_m 在相同 398 个任务上为 `83/398 = 20.8543%`；
suite-specific object checkpoint 为 `137/398 = 34.4221%`：

```text
Absolute change: +13.5678 percentage points
Net successes:   +54 tasks
```

| OFT_m | OFT object | Tasks |
|---:|---:|---:|
| Fail | Fail | 245 |
| Fail | Success | 70 |
| Success | Fail | 16 |
| Success | Success | 67 |

## 3. 按难度等级

| Difficulty | Success / Tasks | Rate |
|---:|---:|---:|
| 1 | 41 / 50 | 82.0000% |
| 2 | 51 / 70 | 72.8571% |
| 3 | 33 / 90 | 36.6667% |
| 4 | 9 / 87 | 10.3448% |
| 5 | 3 / 101 | 2.9703% |

## 4. 按基础任务

| Object | Success / Tasks | Rate |
|---|---:|---:|
| alphabet soup | 16 / 42 | 38.1% |
| BBQ sauce | 9 / 38 | 23.7% |
| butter | 25 / 42 | 59.5% |
| chocolate pudding | 10 / 34 | 29.4% |
| cream cheese | 10 / 39 | 25.6% |
| ketchup | 14 / 38 | 36.8% |
| milk | 15 / 40 | 37.5% |
| orange juice | 14 / 47 | 29.8% |
| salad dressing | 13 / 40 | 32.5% |
| tomato sauce | 11 / 38 | 28.9% |

## 5. 评测协议

| 项目 | 值 |
|---|---|
| Checkpoint | `/home/imc/models/ra-loop/openvla-oft-object` |
| HF repository | `moojink/openvla-7b-oft-finetuned-libero-object` |
| Suite/category | `libero_object / Robot Initial States` |
| Tasks / episodes | 398 / 398（每任务 1 episode） |
| Seed | 7 |
| Shards | `57 × 6 + 56` 个确定性、不重叠 manifest |
| Physical GPUs | 1--7；GPU 0 未使用 |
| Render backend | OSMesa；未使用 EGL |
| Python | `/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python` |
| Python isolation | `PYTHONNOUSERSITE=1` |
| Launcher | `eval/launch_oft_object_robot398_tmux.sh --execute` |
| Tmux session | `ra_oft_object_robot398` |

会话于 2026-07-20 13:47:01 CST 启动，最后一个任务于 14:47:13 CST 完成；
各片耗时约 54--60 分钟。模型只做推理，没有训练或修改 checkpoint。

## 6. 分片结果与文件校验

| Shard | GPU | Success / Tasks | Rate | results.jsonl SHA-256 |
|---:|---:|---:|---:|---|
| 0 | 1 | 15 / 57 | 26.3% | `b1884fc0ff2fc675c708394e623c2402ed708de399da1179848e51d79ff0b8f2` |
| 1 | 2 | 15 / 57 | 26.3% | `9756907a10ec228f443c9f084a407c162b6730442a729fadd72263cc94dbc9ce` |
| 2 | 3 | 22 / 57 | 38.6% | `3bb4dd8ba42e62a12f0b918f759731d2f52443f4b3143324e890fd38e81a43dd` |
| 3 | 4 | 19 / 57 | 33.3% | `bb99574e9605fadff47e74d9a8c3df7ce1423972cad178c8f10272da9929ce75` |
| 4 | 5 | 25 / 57 | 43.9% | `57c84f56bd3032b127eac474e0f5af9b9c81004624b2ca4a2f5c16383efea2e4` |
| 5 | 6 | 18 / 57 | 31.6% | `3ef411a66a768b0950c55dc75d82639a982f83e4c353a380b7f8564ef7cd37ec` |
| 6 | 7 | 23 / 56 | 41.1% | `e6ef2f58f1cd2319cb834477f55837c43f8b73b263ac2173a08d1b0892e40e6c` |

按 shard 0 到 6 的原始字节依次拼接后：

```text
SHA-256: 4d76dc015167a54baec66ed9dd64e09395896304560ee462587df5c4660d131c
```

结果根目录：

```text
/home/imc/yzy/RA_LOOP/logs/openvla_oft_object_robot_full398_7gpu_20260720
```

验收时无 evaluator 进程，GPU 1--7 回到 18 MiB 基础占用和 0% 利用率。tmux
会话保留为七个空 shell，便于人工检查。

## 7. 下一步

下载并 CPU-only 静态校验 suite-specific `libero_goal` checkpoint。goal 完成后，
再使用已有 `libero_10` 专用 checkpoint 重跑 393 条。四套件全部完成后汇总并与
官方 OpenVLA-OFT Robot 31.9% 比较。
