#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  Step 03 — Python 依赖安装 (主环境)
# ============================================================================
#
# 目的:
#   在 ript_vla_openvla_oft 环境中依次安装:
#     1. openvla-oft 及其依赖 (通过 pip install -e .)
#     2. 加速库: ninja, flash-attn, accelerate
#     3. RIPT-VLA 及其依赖
#     4. LIBERO-plus 及其额外依赖
#
# 版本关键陷阱 (2026-07 联网核对):
#   * transformers 必须用 moojink 的 fork (bidirectional attn for parallel decoding)
#     git+https://github.com/moojink/transformers-openvla-oft.git
#   * flash-attn==2.7.4.post1 (RIPT-VLA 官方指定, 比 OpenVLA-OFT README 的 2.5.5 新)
#   * accelerate==1.6.0 (RIPT-VLA 官方指定)
#   * mujoco==3.3.2 (LIBERO 需要, 与 robosuite 兼容)
#
# 运行方式:
#   bash setup/step03_deps.sh
#
# 预期完成时间: 30-60 分钟 (flash-attn 编译最慢)
# ============================================================================

set -euo pipefail

CODE_DIR="${CODE_DIR:-$HOME/code}"
ENV_MAIN="ript_vla_openvla_oft"

# ---------------------------------------------------------------------------
# 0. 激活主环境
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_MAIN"

echo "[STEP 03] Installing dependencies in $ENV_MAIN"
echo "  python: $(which python)"

# 确保 pip 是最新的 (避免 wheel 兼容性问题)
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 1. openvla-oft (pip install -e .)
# ---------------------------------------------------------------------------
# 这一步会自动装:
#   * transformers @ git+https://github.com/moojink/transformers-openvla-oft.git
#   * peft==0.11.1, tokenizers==0.19.1, timm==0.9.10
#   * tensorflow==2.15.0 (dataset 加载用)
#   * diffusers==0.30.3
#   * torch/torchvision 已经装过, pip 会跳过
#
# 陷阱:
#   * 如果 pip 之前装过官方 transformers, 会有版本冲突.
#     解决方案: 先 pip uninstall transformers, 再 pip install -e .
#   * tensorflow 2.15 在 macOS ARM 上会失败, 但我们在 Linux + CUDA 上跑 OK
echo "[STEP 03.1] Installing openvla-oft..."
cd "$CODE_DIR/openvla-oft"

# 预防性卸载: 官方 transformers 会与 fork 冲突
pip uninstall transformers -y 2>/dev/null || true

# 主安装. -e 参数使代码修改立即生效, 不用重装.
pip install -e .

# ---------------------------------------------------------------------------
# 2. 加速库: ninja + flash-attn
# ---------------------------------------------------------------------------
echo "[STEP 03.2] Installing acceleration libraries..."

# ninja 用于并行编译, 加速 flash-attn build
pip install packaging ninja

# 验证 ninja 装好
if ! ninja --version 2>/dev/null; then
  echo "  ERROR: ninja install failed"
  exit 1
fi

# flash-attn 编译很慢 (10-30 分钟), 且需要能访问 CUDA 编译器
# 用 --no-build-isolation 保证 flash-attn 用当前环境的 torch 编译
#
# 版本选择: RIPT-VLA INSTALL.md 指定 2.7.4.post1
# 兼容性: 2.7.4 支持 torch 2.2.0 + CUDA 12.1 (官方 wheel)
#
# 如果编译失败, 可能是:
#   1. nvcc 版本 < 11.8   → 装 CUDA toolkit
#   2. 内存不足           → 编译时 OOM. 用 MAX_JOBS=2 pip install ...
#   3. gcc 版本 > 11      → gcc 12+ 与旧 cuda 冲突. 降级到 gcc 11.
#
# 保险起见, 先 clear pip cache
pip cache remove flash_attn 2>/dev/null || true

# 加大 build job 数, 加速编译 (7×4090 机器一般 CPU 也强)
MAX_JOBS=${MAX_JOBS:-8} pip install "flash-attn==2.7.4.post1" --no-build-isolation

# accelerate 版本必须严格锁定
pip install accelerate==1.6.0

# ---------------------------------------------------------------------------
# 3. RIPT-VLA (pip install -e .)
# ---------------------------------------------------------------------------
# 这一步会装:
#   * hydra-core==1.3.2, wandb==0.18.3
#   * diffusers==0.28.0 → 会与前面 0.30.3 冲突!
#     解决方案: 让 openvla-oft 的 0.30.3 优先 (后装 ript-vla 会降级 diffusers,
#     但主要用 diffusion 的是 OpenVLA-OFT 的 action head, 需要 0.30.3)
#   * robosuite==1.4.1 (LIBERO 官方是 1.4.0, RIPT 用 1.4.1)
#
# 陷阱: diffusers 版本冲突
#   * RIPT-VLA requirements 写死 diffusers==0.28.0
#   * OpenVLA-OFT 需要 diffusers==0.30.3 (用了新 API)
#   * 谁后装谁生效. 我们让 openvla-oft 优先 (装两次)
echo "[STEP 03.3] Installing ript-vla..."
cd "$CODE_DIR/ript-vla"

pip install -e .

# 修复 diffusers 版本 (被 ript-vla 降级了, 必须回到 0.30.3)
pip install diffusers==0.30.3

# 修复 transformers 版本 (可能被 ript-vla 覆盖)
# ript-vla requirements.txt 里 `transformers` 无版本号, 可能装到最新版, 冲突
pip install --no-deps --force-reinstall \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"

# ---------------------------------------------------------------------------
# 4. LIBERO-plus (替代官方 LIBERO)
# ---------------------------------------------------------------------------
# LIBERO-plus 兼容官方 LIBERO API, 可以直接 pip install -e . 覆盖.
# 系统依赖 (apt) 需要 root, 如果没有 root 请联系管理员.
echo "[STEP 03.4] Installing LIBERO-plus..."

# 检查 apt 依赖. 如果没有 root, 提示用户联系管理员.
NEED_APT_PKGS=(libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev)
MISSING_PKGS=()
for pkg in "${NEED_APT_PKGS[@]}"; do
  if ! dpkg -s "$pkg" &>/dev/null; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo "  Missing apt packages: ${MISSING_PKGS[*]}"
  if command -v sudo &>/dev/null; then
    echo "  Trying to install with sudo..."
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PKGS[@]}"
  else
    echo "  ERROR: sudo not available, and packages missing:"
    echo "    ${MISSING_PKGS[*]}"
    echo "  Ask your admin to run:"
    echo "    apt-get install -y ${MISSING_PKGS[*]}"
    echo "  Then re-run this script."
    exit 1
  fi
fi

cd "$CODE_DIR/LIBERO-plus"
pip install -e .

# LIBERO-plus 的额外依赖 (wand, scikit-image, 见 extra_requirements.txt)
pip install -r extra_requirements.txt

# 关键: LIBERO 需要 mujoco 3.3.2 (RIPT-VLA 官方指定)
# 之前 pip install -e . 可能装了默认版本 (2.x), 强制升级到 3.3.2
pip install "mujoco==3.3.2"

# ---------------------------------------------------------------------------
# 5. openvla-oft 的 LIBERO 特定依赖
# ---------------------------------------------------------------------------
# openvla-oft 有一个 experiments/robot/libero/libero_requirements.txt
# 里面装 robosuite 等. 但 LIBERO-plus 已经装过了, 只是版本可能不同.
echo "[STEP 03.5] Installing openvla-oft LIBERO-specific deps..."
cd "$CODE_DIR/openvla-oft"

if [[ -f experiments/robot/libero/libero_requirements.txt ]]; then
  pip install -r experiments/robot/libero/libero_requirements.txt
fi

# ---------------------------------------------------------------------------
# 6. 最终版本 pin (记录到日志, 便于 debug)
# ---------------------------------------------------------------------------
PROJ_DIR="${PROJ_DIR:-$HOME/Desktop/essay/RA-LOOP}"
mkdir -p "$PROJ_DIR/logs"

echo "[STEP 03.6] Dumping final environment to logs/pip_freeze.txt..."
pip freeze > "$PROJ_DIR/logs/pip_freeze.txt"

# 关键包版本快速核对
echo ""
echo "  Key package versions:"
for pkg in torch transformers peft flash_attn accelerate diffusers mujoco robosuite tokenizers timm; do
  ver=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
  echo "    $pkg == ${ver:-NOT INSTALLED}"
done | tee "$PROJ_DIR/logs/key_versions.txt"

echo ""
echo "============================================================================"
echo "[STEP 03] DONE."
echo "  Full pip freeze: $PROJ_DIR/logs/pip_freeze.txt"
echo "  Key versions   : $PROJ_DIR/logs/key_versions.txt"
echo ""
echo "  Next: bash setup/step04_data.sh"
echo "============================================================================"

# ============================================================================
# 成功预期:
#   * key_versions.txt 显示:
#       torch        == 2.2.0
#       transformers == 4.40.1  (or 4.40.1.dev0, from moojink fork)
#       peft         == 0.11.1
#       flash_attn   == 2.7.4.post1
#       accelerate   == 1.6.0
#       diffusers    == 0.30.3
#       mujoco       == 3.3.2
#       robosuite    == 1.4.1  (或 1.4.0, 都可)
#       tokenizers   == 0.19.1
#       timm         == 0.9.10
#
# 失败模式与应对:
#   A. flash-attn 编译失败, error like "nvcc: command not found"
#      —— CUDA toolkit 未装. 参考 https://developer.nvidia.com/cuda-12-1-0-download-archive
#         或用 conda: conda install cuda-toolkit=12.1 -c nvidia
#      —— 若集群禁止装 CUDA, 用预编译 wheel:
#         URL 见 https://github.com/Dao-AILab/flash-attention/releases
#         找对应 torch 2.2 + cu121 + py310 的 whl 文件, 直接 pip install <wheel>.
#
#   B. flash-attn 编译成功但 import 失败, error like "undefined symbol"
#      —— torch/CUDA/flash-attn 版本三者不一致. 检查:
#         python -c "import torch; print(torch.version.cuda, torch.__version__)"
#         期望: 12.1, 2.2.0
#
#   C. transformers 冲突, 报错 "cannot import LlamaForCausalLM"
#      —— 官方 transformers 覆盖了 fork. 重装:
#         pip uninstall transformers -y
#         pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
#
#   D. tensorflow 装不上或 import 失败
#      —— tf 2.15 需要 GLIBC 2.28+. 老系统需要升级 GLIBC 或用 tf-nightly.
#      —— 如果 tf 只用在 dataset loading 阶段, 且我们不做 SFT 只做 RL,
#         可以先忽略 tf import error, 到 dataset 阶段再修.
#
#   E. mujoco 3.3.2 与 robosuite 1.4.1 不兼容
#      —— LIBERO-plus 声称兼容 mujoco 3.3.2, 但如果实际报错:
#         mujoco==3.2.5 是 fallback (与 robosuite 1.4.x 都兼容)
#         如果还不行, 降到 mujoco==2.3.7 (官方 LIBERO 原版指定)
#
#   F. LIBERO-plus 的 apt 依赖没有 sudo 权限
#      —— 联系管理员. 或用 conda-forge:
#         conda install -c conda-forge libmagickwand
#         (可能不完全等价但可尝试)
# ============================================================================
