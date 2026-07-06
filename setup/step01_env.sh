#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  Step 01 — Conda 环境 + PyTorch 安装
# ============================================================================
#
# 目的:
#   建立两个 conda 环境:
#     * ript_vla              — 用于 QueST 分支 (备胎，Plan B 用)
#     * ript_vla_openvla_oft  — 主环境, 跑 OpenVLA-OFT + RIPT-VLA + RA-LOOP
#
# 版本来源:
#   * python==3.10.14        — RIPT-VLA INSTALL.md 官方
#   * torch==2.2.0           — OpenVLA-OFT pyproject.toml 硬约束
#   * torchvision==0.17.0    — 匹配 torch 2.2.0
#   * torchaudio==2.2.0      — 匹配 torch 2.2.0
#   * CUDA 12.1              — 4090 官方支持, torch 2.2.0 兼容
#
# 硬件前置条件:
#   * NVIDIA driver >= 530 (支持 CUDA 12.1)
#   * 通过 `nvidia-smi` 应能看到 7 张 4090
#   * 空闲显存 > 22GB / 卡
#
# 运行方式:
#   bash setup/step01_env.sh
#
# 预期完成时间: 15-30 分钟 (取决于网络)
# ============================================================================

set -euo pipefail  # 任何步骤失败立即退出, 避免污染环境

# ---------------------------------------------------------------------------
# 0. Sanity check — 硬件与 driver
# ---------------------------------------------------------------------------
echo "[STEP 01] Checking hardware..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  ERROR: nvidia-smi not found. Are you on a machine with NVIDIA GPUs?"
  exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | xargs)
echo "  Detected $GPU_COUNT GPUs"
if [[ "$GPU_COUNT" -lt 7 ]]; then
  echo "  WARN: expected 7 GPUs (7x RTX4090), got $GPU_COUNT"
  echo "  Continuing anyway — 4090 baseline works with fewer cards."
fi

# 打印 driver 与 CUDA 版本, 记录到日志
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee logs/hardware_probe.log

# ---------------------------------------------------------------------------
# 1. Conda 安装检查
# ---------------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
  echo "  ERROR: conda not found. Please install miniconda3 first."
  echo "  Suggested: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  exit 1
fi

# 加载 conda 到当前 shell (避免 `conda activate` 报错)
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. 创建主环境: ript_vla_openvla_oft (RA-LOOP 主环境)
# ---------------------------------------------------------------------------
ENV_MAIN="ript_vla_openvla_oft"
if conda env list | grep -q "^$ENV_MAIN "; then
  echo "  [MAIN ENV] $ENV_MAIN already exists, skipping creation"
else
  echo "  [MAIN ENV] Creating $ENV_MAIN with python 3.10..."
  # 注意: RIPT-VLA INSTALL.md 说这个环境用 python 3.10 (不带 .14 后缀)
  # 我们用 3.10.14 以对齐 QueST 分支, 保证两个环境的 python 补丁号一致
  conda create -n "$ENV_MAIN" python=3.10.14 -y
fi

conda activate "$ENV_MAIN"

# 打印 python 位置, 验证环境激活成功
echo "  [MAIN ENV] which python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# 3. 安装 PyTorch 2.2.0 + CUDA 12.1
# ---------------------------------------------------------------------------
echo "  [MAIN ENV] Installing PyTorch 2.2.0 + CUDA 12.1..."

# 严格用 --index-url, 保证从 PyTorch 官方源拉取 cu121 wheel
# 用 pip 而不是 conda 的原因:
#   1. conda 的 pytorch channel 已弃用 (2024)
#   2. 官方 wheel 自带 CUDA runtime, 免除 nvcc 冲突
pip install --upgrade pip
pip install \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 验证 CUDA 是否可用 — 这一步失败必须停下来 debug
python - <<'PY'
import torch
print(f"[VERIFY] torch version : {torch.__version__}")
print(f"[VERIFY] CUDA available: {torch.cuda.is_available()}")
print(f"[VERIFY] CUDA version  : {torch.version.cuda}")
print(f"[VERIFY] GPU count     : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"[VERIFY]   GPU {i}: {torch.cuda.get_device_name(i)}")
assert torch.cuda.is_available(), "CUDA not available, cannot proceed"
assert torch.cuda.device_count() >= 1, "No GPUs found"
print("[VERIFY] Main env OK.")
PY

# ---------------------------------------------------------------------------
# 4. 创建备胎环境: ript_vla (QueST + RIPT, 显存需求更低)
# ---------------------------------------------------------------------------
# 只在需要 Plan B 时启用. Week 4 gate 之前保持备用状态.
# 如果磁盘紧张, 可以先跳过, 需要时再执行 setup/step01b_backup_env.sh
ENV_BACKUP="ript_vla"
if [[ "${SKIP_BACKUP_ENV:-0}" == "1" ]]; then
  echo "  [BACKUP ENV] SKIP_BACKUP_ENV=1, skipping ript_vla env"
else
  if conda env list | grep -q "^$ENV_BACKUP "; then
    echo "  [BACKUP ENV] $ENV_BACKUP already exists, skipping"
  else
    echo "  [BACKUP ENV] Creating $ENV_BACKUP (Plan B) ..."
    conda create -n "$ENV_BACKUP" python=3.10.14 -y
    conda activate "$ENV_BACKUP"
    pip install \
      torch==2.2.0 torchvision==0.17.0 \
      --index-url https://download.pytorch.org/whl/cu121
  fi
fi

conda deactivate

echo ""
echo "============================================================================"
echo "[STEP 01] DONE."
echo "  Main env  : $ENV_MAIN"
echo "  Backup env: $ENV_BACKUP  (only for Plan B, safe to ignore for now)"
echo ""
echo "  Next: bash setup/step02_repos.sh"
echo "============================================================================"

# ============================================================================
# 成功预期:
#   * 主环境激活后 `torch.cuda.is_available()` 返回 True
#   * `torch.cuda.device_count()` == 7 (或至少 >= 1)
#   * 打印的 GPU name 全部是 "NVIDIA GeForce RTX 4090"
#
# 失败模式与应对:
#   A. `torch.cuda.is_available()` 返回 False
#      —— driver 版本过低. 需要升级到 >=530.
#      —— 或 conda 环境激活失败, 检查 which python 是否指向环境内.
#
#   B. GPU name 显示 "unknown"
#      —— PCIe 通信问题. 重启机器或联系集群管理员.
#
#   C. pip install torch 卡在网络
#      —— 使用国内镜像:
#         pip install torch==2.2.0 ... -i https://pypi.tuna.tsinghua.edu.cn/simple
#         注意: PyTorch cu121 wheel 只能从 download.pytorch.org 拿, 清华源
#         没有 cu121 wheel, 必须走官方源. 若网络受限, 用代理或先下 wheel
#         再本地 pip install <wheel文件>.
#
#   D. 显存不足 (`out of memory` during verify)
#      —— 其他进程占用. 用 `nvidia-smi` 查看谁占了, 清理后重跑此步.
# ============================================================================
