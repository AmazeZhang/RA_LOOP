# OpenVLA-OFT_m / LIBERO-Plus Robot-init 1550 baseline

> 20260720 当前工作状态及 spatial checkpoint 下载进度见
> `docs/STATUS_20260720.md`。

## 1. 结论与模型身份

2026-07-20 完成四套件联合 checkpoint 在 LIBERO-Plus
`Robot Initial States` 分类全部 1550 个任务上的严格汇总：

```text
Coverage: 1550 / 1550 (missing=0, duplicate=0)
Success:  393 / 1550
Rate:     25.354839%（报告为 25.4%）
```

使用的 Hugging Face 仓库是：

```text
moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

该联合模型在 LIBERO-Plus 论文与官方 README 中记作
**OpenVLA-OFT_m（mix-SFT）**。官方 Robot 指标是 **21.7%**，因此本次结果应与
21.7% 比较，差值为 **+3.65 个百分点**。此前将其与 31.9% 比较属于模型身份
误判；31.9% 对应每个 suite 使用独立 checkpoint 的 OpenVLA-OFT。

本次结果证明联合模型的完整评测工程链路可用，但与官方 OFT_m 仍有可见数值
差异，不能表述为“精确复现”。可能影响因素包括 GPU/软件小版本、官方运行时
模型快照以及仿真数值差异；在完成同设备或多 seed 对照前不进一步归因。

## 2. 官方证据

- LIBERO-Plus 官方表格同时报告 `OpenVLA-OFT: Robot 31.9` 与
  `OpenVLA-OFT_m: Robot 21.7`，并注明 OFT_m 是四套件 mix-SFT：
  <https://github.com/sylvestf/LIBERO-plus>
- CVPR 2026 论文对 OFT_m 作相同定义：
  <https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf>
- OpenVLA-OFT 官方评测文档说明论文主要结果为每个 suite 使用一个独立模型，
  联合模型是额外实验：
  <https://github.com/moojink/openvla-oft/blob/main/LIBERO.md>
- 作者对 checkpoint 使用方式的说明：
  <https://github.com/openvla/openvla/issues/218>

## 3. 分套件结果

| Suite | Success / Tasks | Rate |
|---|---:|---:|
| `libero_spatial` | 83 / 350 | 23.7143% |
| `libero_object` | 83 / 398 | 20.8543% |
| `libero_goal` | 86 / 409 | 21.0269% |
| `libero_10` | 141 / 393 | 35.8779% |
| **Total** | **393 / 1550** | **25.3548%** |

## 4. 按难度等级

| Difficulty | Success / Tasks | Rate |
|---:|---:|---:|
| 1 | 171 / 248 | 68.9516% |
| 2 | 137 / 292 | 46.9178% |
| 3 | 63 / 374 | 16.8449% |
| 4 | 13 / 285 | 4.5614% |
| 5 | 9 / 351 | 2.5641% |

难度 2 到 3 出现明显断层；difficulty 4/5 的成功率低于 5%，后续方法应重点
报告高难度子集变化，而不能只报告总体平均值。

## 5. 评测协议与完整性

| 项目 | 值 |
|---|---|
| Checkpoint | `/home/imc/models/ra-loop/openvla-oft-union` |
| Category | `Robot Initial States` |
| Suites | spatial / object / goal / libero_10 |
| Tasks / episodes | 1550 / 1550（每任务 1 episode） |
| Seed | 7 |
| Physical GPUs | 1--7；GPU 0 未使用 |
| Render backend | OSMesa；未使用 EGL |
| Python | `/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python` |
| Python isolation | `PYTHONNOUSERSITE=1` |

所有分片的 checkpoint、suite、seed、manifest SHA 和 render backend 元数据均与
预期一致。逐条核对官方分类后，覆盖 1550/1550，无缺失、无重复、无任务串片。
按 suite（spatial、object、goal、libero_10）和各 suite shard 序拼接原始
`results.jsonl` 的 SHA-256 为：

```text
c79686b04f55ce380bcbe9481ade98a54a76c425aa60001c37b88915e9e419dc
```

结果目录：

```text
/home/imc/yzy/RA_LOOP/logs/openvla_oft_union_robot_remaining1157_7gpu_20260719
/home/imc/yzy/RA_LOOP/logs/openvla_oft_union_libero10_robot_full393_7gpu_20260718
```

评测结束后无 evaluator 进程，GPU 1--7 空闲。模型只用于推理，没有训练或修改
checkpoint。

## 6. 下一步：正式复现 OpenVLA-OFT 31.9%

需要按 suite 使用以下四个独立 checkpoint：

```text
moojink/openvla-7b-oft-finetuned-libero-spatial
moojink/openvla-7b-oft-finetuned-libero-object
moojink/openvla-7b-oft-finetuned-libero-goal
moojink/openvla-7b-oft-finetuned-libero-10
```

本地已有 `libero_10` checkpoint。spatial 已于 2026-07-20 完成下载和 CPU-only
静态校验，随后完成 Robot-init 350 条正式评测，结果为 135/350（38.5714%），
见 `docs/BASELINE_OFT_SPATIAL_ROBOT350_20260720.md`。object 已于同日完成下载和
CPU-only 静态校验，随后完成 Robot-init 398 条正式评测，结果为
137/398（34.4221%），见 `docs/BASELINE_OFT_OBJECT_ROBOT398_20260720.md`；
仍需下载和评测 goal。
为控制磁盘和文件风险，继续按 suite 逐个准备、评测并停顿汇报。当前未启动
spatial GPU 评测。
