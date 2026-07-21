# LIBERO-Plus Spatial Robot-init 关节偏移标定 — 20260720

> CPU-only、只读标定；未导入 LIBERO/robosuite/MuJoCo，未创建模拟器，未使用 GPU，未修改 Plus 数据。

## 标定对象与机制

- suite：`libero_spatial`
- task：`pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- 分类表中该任务 Robot Initial States 变体：37 个
- difficulty 数量：{1: 7, 2: 11, 3: 14, 4: 3, 5: 2}
- canonical Panda qpos：`[0.0000, -0.1610, 0.0000, -2.4446, 0.0000, 2.2268, 0.7854]` rad
- Plus 的 `initstate_N` 会选择 `PandaN`；500 个 qpos 是以 canonical qpos 为中心、固定随机方向生成。
  编号 1--100、101--200、…、401--500 的 L2 半径分别为 0.1、0.2、…、0.5 rad，
  而不是从任务的 50 条 `.pruned_init` 中按 N 索引。
- 分类 JSON 的 `difficulty_level` 是另一套标签，不等同于物理 L2 半径；下面分别报告，不能互换。

## 选中变体的实测分布

按生成半径：

| 生成 L2 半径 | 数量 | difficulty 计数 | 实测 L2 中位数 | RMS/关节中位数 | 最大单关节绝对偏移中位数 |
|---:|---:|---|---:|---:|---:|
| 0.1 | 3 | {1: 2, 2: 1} | 0.1000 | 0.0378 | 0.0686 |
| 0.2 | 9 | {1: 3, 3: 5, 4: 1} | 0.2000 | 0.0756 | 0.1368 |
| 0.3 | 8 | {1: 1, 2: 3, 3: 2, 4: 2} | 0.3000 | 0.1134 | 0.2044 |
| 0.4 | 9 | {1: 1, 2: 3, 3: 3, 5: 2} | 0.4000 | 0.1512 | 0.2651 |
| 0.5 | 8 | {2: 4, 3: 4} | 0.5000 | 0.1890 | 0.3274 |

按分类 difficulty（可见它与半径不单调对应）：

| difficulty | 数量 | joint-delta L2 中位数 | RMS/关节中位数 | 最大单关节绝对偏移中位数 |
|---:|---:|---:|---:|---:|
| 1 | 7 | 0.2000 | 0.0756 | 0.1312 |
| 2 | 11 | 0.4000 | 0.1512 | 0.3012 |
| 3 | 14 | 0.3500 | 0.1323 | 0.2167 |
| 4 | 3 | 0.3000 | 0.1134 | 0.1899 |
| 5 | 2 | 0.4000 | 0.1512 | 0.2566 |

全部 37 个变体的 L2 分位数：`{"min": 0.09999999990089671, "p25": 0.20000000299630696, "p50": 0.30000000392648113, "p75": 0.40000000173237676, "p90": 0.49999999941112094, "max": 0.5000000041799734}`。
全部变体 RMS/关节分位数：`{"min": 0.037796447263465205, "p25": 0.07559289573434302, "p50": 0.11338934338683855, "p75": 0.15118578985846776, "p90": 0.18898223628203825, "max": 0.18898223808449505}`。
逐关节 mean |delta|：`[0.1055, 0.0974, 0.1095, 0.1062, 0.1045, 0.0865, 0.1124]` rad。
逐关节 p90 |delta|：`[0.2251, 0.2244, 0.2362, 0.2102, 0.2209, 0.1724, 0.2077]` rad。

## 与原任务 init-state 的关系

原 `.pruned_init` 为 `(50, 92)`，其中 state[1:8] 是 Panda 7 关节。
其相对 canonical qpos 的 L2 分位数为：`{"min": 0.021530678339453314, "p25": 0.040869915369578416, "p50": 0.050570765252297714, "p75": 0.061538766471742556, "p90": 0.06751728002286178, "max": 0.09042094094945191}`。
原任务关节标准差：`[0.0200, 0.0206, 0.0199, 0.0179, 0.0201, 0.0201, 0.0210]` rad。
这 50 条状态本身已带有小幅初始化散布；Plus Robot-init 是在机器人模型默认 qpos 上施加独立、明显更大的分档偏移。

## 对首个 RA learning-signal probe 的结论

- 当前 RA `strength` 实际是每关节独立 Gaussian 的标准差，不是总 L2；connectivity smoke 的 0.001 rad 对应期望 L2 约 0.00255 rad，只适合验证链路。
- 若保持现实现，首个 h220/K8 probe 可用 `strength=0.0392` rad/关节，使 7 维 Gaussian 的期望 L2 为 0.1 rad，近似对齐 Plus 最轻生成档；仍保持 recovery-only 与 `lambda_consistency=0`。
- 更严格的复现方式是在下一步先增加 `fixed_l2` 采样模式，再直接配置 0.1 rad；这会与 Plus 的归一化随机方向完全同单位，也更容易解释。
- 若首轮 perturbed 成功率与 anchor 几乎无差异，再单独确认后试 0.2 rad L2；第一次 probe 不同时扫描多档。

## 实际 baseline evaluator 的覆盖语义风险

静态调用链显示：本项目 bounded evaluator 使用 upstream `run_task`，每个变体只跑 episode 0；
upstream 先 `env.reset()`，随后把基础 `.pruned_init[0]` 传给 Plus `set_init_state()`；
Plus 的该方法直接调用 `sim.set_state_from_flattened()`。这会在 PandaN reset 之后覆盖 7 个 robot qpos。
因此现有 38.57% 结果按代码路径并未保留 PandaN 的默认关节偏移；它是分类表覆盖完整的结果，
但不能作为‘真实 Robot-init 物理偏移’已被正确施加的证据。该问题不会影响 RA adapter：RA 是在送入 rollout 前直接修改 flattened state[1:8]。
在改动 evaluator 或重跑任何 GPU baseline 前，应单独做一个 CPU/live 双状态读回 gate，并由用户确认评测口径。

## 可复跑命令

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python \
  scripts/calibrate_libero_plus_robot_init.py
```
