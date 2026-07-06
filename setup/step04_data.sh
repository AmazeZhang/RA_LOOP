#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  Step 04 — 权重与数据下载
# ============================================================================
#
# 目的:
#   下载以下资源:
#     1. OpenVLA-OFT 官方 4 个 SFT 权重 (LIBERO 四个 suite)
#     2. RIPT-VLA 官方的 Scale Header 权重
#     3. LIBERO-plus 的 assets.zip (perturbation config + textures)
#     4. LIBERO-plus 训练数据集 (可选, 只有做 mix-SFT 才需要)
#
# 下载体积 (2026-07 核实):
#   * OpenVLA-OFT 单个 checkpoint  ~15 GB   (总计 60GB, 4 个 suite)
#   * OpenVLA-OFT 联合 checkpoint  ~15 GB
#   * RIPT-VLA Scale Header         ~300 MB / suite (共 4 个 = 1.2 GB)
#   * LIBERO-plus assets.zip       ~15 GB
#   * LIBERO-plus RLDS (可选)       75 GB
#
# 磁盘预算:
#   * 最小 (只跑 evaluation): 30 GB   (1 suite + assets)
#   * 推荐 (全 4 个 suite):   90 GB
#   * 完整 (加 RLDS):        165 GB
#
# 运行方式:
#   bash setup/step04_data.sh
#     默认下 LIBERO-Long 一个 suite (最短任务序列, 最难, 最有 novelty 空间)
#     加 --all-suites  下四个 suite
#     加 --with-rlds   下训练数据 (mix-SFT 用)
#
# 前置:
#   * 已 accept 过 openvla/openvla-7b 的 license (HF 上手动点)
#   * huggingface-cli 已 login (`huggingface-cli login`)
# ============================================================================

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/data/ra-loop}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/ra-loop}"
mkdir -p "$DATA_DIR" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# 0. 参数解析
# ---------------------------------------------------------------------------
DOWNLOAD_ALL_SUITES=0
DOWNLOAD_RLDS=0
for arg in "$@"; do
  case "$arg" in
    --all-suites) DOWNLOAD_ALL_SUITES=1 ;;
    --with-rlds)  DOWNLOAD_RLDS=1 ;;
    *) echo "  Unknown arg: $arg" ; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. huggingface-cli 检查
# ---------------------------------------------------------------------------
if ! command -v huggingface-cli &>/dev/null; then
  echo "  Installing huggingface_hub[cli]..."
  pip install "huggingface_hub[cli]"
fi

# 检查是否已登录 (whoami 不报错就是已登录)
if ! huggingface-cli whoami &>/dev/null; then
  echo "  ERROR: not logged in to HuggingFace."
  echo "  Run: huggingface-cli login"
  echo "  Then paste your access token (get one at https://huggingface.co/settings/tokens)"
  echo "  Also make sure you have accepted the license at:"
  echo "    https://huggingface.co/openvla/openvla-7b"
  exit 1
fi

echo "  HF login OK, user: $(huggingface-cli whoami)"

# ---------------------------------------------------------------------------
# 2. OpenVLA-OFT 权重下载
# ---------------------------------------------------------------------------
# 4 个独立 SFT checkpoint. Week 1 只需要 libero-10 (Long) 一个做 baseline.
# Week 2 展开做全 4 suite 时再下其它.
declare -A SUITES=(
  ["long"]="moojink/openvla-7b-oft-finetuned-libero-10"
  ["spatial"]="moojink/openvla-7b-oft-finetuned-libero-spatial"
  ["object"]="moojink/openvla-7b-oft-finetuned-libero-object"
  ["goal"]="moojink/openvla-7b-oft-finetuned-libero-goal"
)

# Week 1 只下 long, 其他 suite 需要时再下 (省时间, 省磁盘)
DEFAULT_SUITES=(long)

if [[ "$DOWNLOAD_ALL_SUITES" == "1" ]]; then
  DEFAULT_SUITES=(long spatial object goal)
fi

for suite in "${DEFAULT_SUITES[@]}"; do
  repo="${SUITES[$suite]}"
  local_dir="$MODEL_DIR/openvla-oft-$suite"
  if [[ -d "$local_dir" && -f "$local_dir/config.json" ]]; then
    echo "  [OFT-$suite] already downloaded, skipping"
    continue
  fi
  echo "  [OFT-$suite] Downloading $repo ..."
  # 用 --local-dir-use-symlinks False 保证真实文件复制到 local_dir
  # (默认是 symlink 到 HF cache, 磁盘监控会不准, 且不利于迁移)
  huggingface-cli download "$repo" \
    --local-dir "$local_dir" \
    --local-dir-use-symlinks False
done

# ---------------------------------------------------------------------------
# 3. RIPT-VLA Scale Headers + LoRA Adaptors
# ---------------------------------------------------------------------------
# RIPT-VLA 提供两类 checkpoint:
#   * Scale Header (~300MB): SFT 后的 head 权重, RIPT 训练用它作为初始化
#   * LoRA Adaptor (~1GB):   RIPT 训练完成后的 LoRA 权重, 评估用
#
# 我们的 RA-LOOP 是"从 SFT 起手继续 RL", 所以需要 Scale Header 作为初始化.
# LoRA Adaptor 用来跑 baseline (对照组: 纯 LOOP 的效果)
echo "  [RIPT-VLA] Downloading scale headers and LoRA adaptors..."

RIPT_REPO="tanshh97/RIPT_VLA"
RIPT_LOCAL="$MODEL_DIR/ript-vla"

for suite in "${DEFAULT_SUITES[@]}"; do
  # 命名对齐: RIPT-VLA 用 LIBERO_LONG (大写), 我们的变量是 long
  suite_upper=$(echo "$suite" | tr '[:lower:]' '[:upper:]')

  # Scale Header (SFTed) - 单文件
  # 位置: openvla_oft/scale_header/LIBERO_LONG_scale_header.pth
  huggingface-cli download "$RIPT_REPO" \
    "openvla_oft/scale_header/LIBERO_${suite_upper}_scale_header.pth" \
    --local-dir "$RIPT_LOCAL" \
    --local-dir-use-symlinks False

  # LoRA Adaptor (完整目录, 用 include 匹配)
  huggingface-cli download "$RIPT_REPO" \
    --include "openvla_oft/ript_adaptors/LIBERO_${suite_upper}_lora/*" \
    --local-dir "$RIPT_LOCAL" \
    --local-dir-use-symlinks False
done

# ---------------------------------------------------------------------------
# 4. LIBERO-plus assets.zip
# ---------------------------------------------------------------------------
# 官方要求解压到 $CODE_DIR/LIBERO-plus/libero/libero/
# 因为 pip install -e . 装的是 editable, 代码在 CODE_DIR 里
LIBERO_ASSETS_DIR="${CODE_DIR:-$HOME/code}/LIBERO-plus/libero/libero"

if [[ ! -d "$LIBERO_ASSETS_DIR/assets/articulated_objects" ]]; then
  echo "  [LIBERO-plus] Downloading assets.zip..."
  huggingface-cli download "Sylvest/LIBERO-plus" \
    --repo-type dataset \
    --include "assets.zip" \
    --local-dir "$DATA_DIR/libero-plus-assets" \
    --local-dir-use-symlinks False

  echo "  [LIBERO-plus] Extracting assets.zip to $LIBERO_ASSETS_DIR ..."
  # -o 覆盖已有文件, -q 静默
  unzip -oq "$DATA_DIR/libero-plus-assets/assets.zip" -d "$LIBERO_ASSETS_DIR"
else
  echo "  [LIBERO-plus] assets already extracted, skipping"
fi

# 验证 assets 目录结构
EXPECTED_DIRS=(
  articulated_objects new_objects scenes stable_hope_objects
  stable_scanned_objects textures turbosquid_objects
)
for d in "${EXPECTED_DIRS[@]}"; do
  if [[ ! -d "$LIBERO_ASSETS_DIR/assets/$d" ]]; then
    echo "  WARN: expected $LIBERO_ASSETS_DIR/assets/$d not found"
  fi
done

# ---------------------------------------------------------------------------
# 5. (可选) LIBERO-plus RLDS 训练数据 (75GB)
# ---------------------------------------------------------------------------
if [[ "$DOWNLOAD_RLDS" == "1" ]]; then
  echo "  [LIBERO-plus] Downloading RLDS training data (75GB, will take hours)..."
  huggingface-cli download "Sylvest/libero_plus_rlds" \
    --repo-type dataset \
    --local-dir "$DATA_DIR/libero_plus_rlds" \
    --local-dir-use-symlinks False
fi

# ---------------------------------------------------------------------------
# 6. LIBERO 原版 hdf5 数据集 (评估必需, RL 训练 rollout 用)
# ---------------------------------------------------------------------------
# LIBERO 官方数据集在 HF 上, 用 modified_libero_rlds
# 但 RIPT-VLA 训练用的是原版 hdf5, 需要从 LIBERO 官方 GDrive 或 HF 单独下.
#
# 简化路径: 用 openvla/modified_libero_rlds (10GB), 与 OpenVLA-OFT 训练一致
if [[ ! -d "$DATA_DIR/modified_libero_rlds" ]]; then
  echo "  [LIBERO] Downloading modified_libero_rlds (~10GB)..."
  huggingface-cli download "openvla/modified_libero_rlds" \
    --repo-type dataset \
    --local-dir "$DATA_DIR/modified_libero_rlds" \
    --local-dir-use-symlinks False
fi

# ---------------------------------------------------------------------------
# 7. 记录数据路径到 paths.yaml (供 RIPT-VLA config 读取)
# ---------------------------------------------------------------------------
PROJ_DIR="${PROJ_DIR:-$HOME/Desktop/essay/RA-LOOP}"

cat > "$PROJ_DIR/config/paths.yaml" <<EOF
# ============================================================================
# 数据/模型路径 (由 setup/step04_data.sh 自动生成)
# Generated at $(date -Iseconds)
# ============================================================================
paths:
  # RIPT-VLA 训练输出 (checkpoint, log)
  output_prefix: $PROJ_DIR/logs/ript_output

  # LIBERO 数据集 (hdf5 / rlds)
  data_prefix: $DATA_DIR/modified_libero_rlds

  # W&B 项目名 (可以改)
  wandb_project: ra-loop-2026

# OpenVLA-OFT 官方权重
openvla_oft:
  long:    $MODEL_DIR/openvla-oft-long
  spatial: $MODEL_DIR/openvla-oft-spatial
  object:  $MODEL_DIR/openvla-oft-object
  goal:    $MODEL_DIR/openvla-oft-goal

# RIPT-VLA scale header + LoRA
ript_vla:
  scale_headers:
    long:    $RIPT_LOCAL/openvla_oft/scale_header/LIBERO_LONG_scale_header.pth
    spatial: $RIPT_LOCAL/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
    object:  $RIPT_LOCAL/openvla_oft/scale_header/LIBERO_OBJECT_scale_header.pth
    goal:    $RIPT_LOCAL/openvla_oft/scale_header/LIBERO_GOAL_scale_header.pth
  lora_adaptors:
    long:    $RIPT_LOCAL/openvla_oft/ript_adaptors/LIBERO_LONG_lora
    spatial: $RIPT_LOCAL/openvla_oft/ript_adaptors/LIBERO_SPATIAL_lora
    object:  $RIPT_LOCAL/openvla_oft/ript_adaptors/LIBERO_OBJECT_lora
    goal:    $RIPT_LOCAL/openvla_oft/ript_adaptors/LIBERO_GOAL_lora

# LIBERO-plus 相关
libero_plus:
  assets: $LIBERO_ASSETS_DIR/assets
  rlds:   $DATA_DIR/libero_plus_rlds  # 只有 --with-rlds 时才存在
EOF

echo ""
echo "============================================================================"
echo "[STEP 04] DONE."
echo "  Paths config    : $PROJ_DIR/config/paths.yaml"
echo "  Models          : $MODEL_DIR/"
echo "  Data            : $DATA_DIR/"
echo "  LIBERO assets   : $LIBERO_ASSETS_DIR/assets/"
echo ""
echo "  Total disk used: $(du -sh "$MODEL_DIR" "$DATA_DIR" 2>/dev/null | awk '{s+=$1}END{print s}' || echo '?')"
echo ""
echo "  Next: python setup/step05_verify.py"
echo "============================================================================"

# ============================================================================
# 成功预期:
#   * $MODEL_DIR/openvla-oft-long/config.json 存在
#   * $LIBERO_ASSETS_DIR/assets/textures/ 有大量图片
#   * $PROJ_DIR/config/paths.yaml 包含所有路径
#
# 失败模式与应对:
#   A. HuggingFace 429 rate limit
#      —— 用国内镜像:
#         export HF_ENDPOINT=https://hf-mirror.com
#         然后重跑此脚本
#
#   B. openvla/openvla-7b license 没 accept
#      —— HF 网页上找 openvla/openvla-7b, 点 "Access repository"
#      —— 然后 accept license, 15-30分钟后 token 才能下这些 fine-tune 版本
#
#   C. 磁盘空间不足
#      —— 只下 long suite, 别加 --all-suites
#      —— assets.zip 15GB 是必需, 不能省
#
#   D. RIPT-VLA scale header 下载失败, 报 404
#      —— tanshh97/RIPT_VLA 的目录名可能有 SPATIAL / LONG 大小写差异.
#      —— 用浏览器打开 https://huggingface.co/tanshh97/RIPT_VLA/tree/main/openvla_oft/scale_header
#         看实际文件名. 大写 LIBERO_LONG 是官方命名, 应该 OK.
#
#   E. LIBERO-plus 数据 include filter 匹配不到文件
#      —— HF 上文件名可能是 LIBERO_10 (对应 long). 修改脚本或手动下:
#         huggingface-cli download tanshh97/RIPT_VLA --local-dir <local>
#         下全部, 磁盘富余的话 ~10GB.
# ============================================================================
