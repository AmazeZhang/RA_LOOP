# RA-LOOP: Robustness-Aware LOOP for VLA Post-Training

> **状态**: 实验阶段；旧 fixed-recovery-bonus 作为诊断基线，ICLR 方法已重定向
> **硬件**: 7×RTX4090 (24GB, 无NVLink, PCIe Gen4)
> **主库**: RIPT-VLA + OpenVLA-OFT + LIBERO-Plus
> **验证过的软件栈**: Python 3.10.14, PyTorch 2.2.0, CUDA 12.1, flash-attn 2.7.4.post1
> **最后一次方法/文献核对时间**: 2026-07-23

## 关键版本清单（联网核对后）

**注意**: 所有版本已经在 2026-07-06 从各个仓库的官方 pyproject.toml / INSTALL.md 中直接读取。

### Python 与 PyTorch (RIPT-VLA + OpenVLA-OFT 交集)

```
python==3.10.14   # RIPT-VLA 官方指定
torch==2.2.0      # OpenVLA-OFT 官方 pyproject.toml 硬约束
torchvision==0.17.0
torchaudio==2.2.0
CUDA >= 12.1     # 4090 兼容
```

### transformers - 关键陷阱

**必须使用 moojink 的 fork**，不能用官方 transformers：
```
pip install git+https://github.com/moojink/transformers-openvla-oft.git
```

原因: OpenVLA-OFT 依赖 bidirectional attention 做 parallel decoding，官方 transformers 不支持。

fork 是从 4.40.1 分叉的。如果 pip 装了官方 transformers，OFT 的 prismatic 模块会报错。

### Flash Attention

```
flash-attn==2.7.4.post1   # RIPT-VLA 官方指定（最新）
# 或 flash-attn==2.5.5    # OpenVLA-OFT 原版指定（较老但稳定）
```

RA-LOOP 用 2.7.4.post1（RIPT-VLA 官方版本）。

### 其它关键包

```
peft==0.11.1
tokenizers==0.19.1
timm==0.9.10
accelerate==1.6.0     # RIPT-VLA 官方指定
diffusers==0.30.3
sentencepiece==0.1.99
tensorflow==2.15.0
tensorflow_datasets==4.9.3
mujoco==3.3.2         # LIBERO 需要
robosuite==1.4.1      # RIPT-VLA 版本，比 LIBERO 官方 1.4.0 新
bddl==1.0.1
gym==0.25.2
```

### 已知的 LIBERO / LIBERO-Plus 关键差异

| 项目 | 官方 LIBERO | LIBERO-Plus |
|-----|------------|-------------|
| num_trials_per_task | 50 | **1** (共10030任务) |
| 任务总数 | 130 | 10030 |
| assets zip | 内置 | 需要单独下载 |

## Baseline 参考成绩（LIBERO-Plus 官方 README, 2026-07 核实）

| Model | Camera | Robot | Language | Light | BG | Noise | Layout | **Total** |
|-------|--------|-------|----------|-------|-----|-------|--------|-----------|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | **69.6** |
| OpenVLA-OFT_m | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | **67.9** |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | **68.4** |
| OpenVLA-OFT+ (SOTA) | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | **79.6** |

**关键观察**:
- RIPT-VLA (纯 LOOP) 与 baseline OpenVLA-OFT 几乎持平 (68.4 vs 69.6)
- 说明**纯任务成功 reward 的 RL 对 robustness 无提升**
- 这是我们的机会：加 robustness-aware reward
- 目标: 达到 76-80% Total, 主要拉动 Robot / Camera 两个维度

## 目录结构

```
RA-LOOP/
├── PLAN.md              # 3个月总方案
├── README.md            # 本文件
├── setup/               # 环境搭建脚本
│   ├── step01_env.sh    # Conda + PyTorch
│   ├── step02_repos.sh  # Clone 三个仓库
│   ├── step03_deps.sh   # 依赖安装
│   ├── step04_data.sh   # 权重与数据下载
│   └── step05_verify.py # 环境验证
├── eval/                # 评估脚本
│   ├── run_libero.sh
│   ├── run_libero_plus.sh
│   └── parse_results.py
├── train/               # 训练脚本
│   ├── vanilla_loop.sh  # 复现 RIPT-VLA baseline
│   └── ra_loop.sh       # 我们的方法
├── code/                # 修改后的代码
│   └── ra_optimizer.py  # RA-LOOP 实现
├── config/              # Hydra 配置
│   └── train_ra_loop.yaml
├── docs/                # 决策日志、gate 判断
└── logs/                # 训练日志
```

## 快速导航

- 当前 ICLR 方法重定向 → 看 `docs/RA_LOOP_ICLR_METHOD_REDIRECT_20260723.md`
- Counterfactual 训练前状态 → 看 `docs/RA_LOOP_COUNTERFACTUAL_TRAINING_READY_20260724.md`
- 当前全任务训练结果 → 看 `docs/RA_LOOP_FULLTASK_LAMBDA_ABLATION_RESULT_20260723.md`
- 正在运行的独立评测 → 看 `docs/RA_LOOP_FULLTASK_CANDIDATE_EVAL_READY_20260723.md`
- 想立刻开始 → 看 `setup/step01_env.sh`
- 想了解算法 → 看 `code/ra_optimizer.py` (Week 3-4 写)
- 想了解 gate 判断 → 看 `PLAN.md` §4

> `PLAN.md` 保留项目启动时的原始假设，包含已经过时的 action-consistency /
> fixed-reward 设计；后续方法决策以 2026-07-23 的 ICLR 重定向文档为准。
