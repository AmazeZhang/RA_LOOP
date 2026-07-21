#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  eval/run_libero_orig.sh — 原版 LIBERO 4-suite 基线复现
# ============================================================================
#
# 目的:
#   跑 OpenVLA-OFT 官方 SFT 权重在**原版 LIBERO** (未扰动) 上的评估.
#   目的是验证:
#     * 环境搭对了
#     * 权重下对了
#     * 我们的 GPU (4090) 与官方 (A100) 结果差异可控 (<5%)
#
# 官方基线成绩 (arXiv 2502.19645, in A100):
#   * LIBERO-Spatial: ~97.6%
#   * LIBERO-Object:  ~98.4%
#   * LIBERO-Goal:    ~97.9%
#   * LIBERO-Long:    ~94.5%   (10 episodes/task × 10 tasks = 100 trials/seed, 3 seeds)
#   * Average:        ~97.1%
#
# 我们的 gate 判据 (在 4090 上):
#   * 每个 suite 与官方偏差 <5%       → PASS
#   * 单个 suite 偏差 5-8%           → 排查 (可能 seed / 图像插值)
#   * 单个 suite 偏差 >8%            → 停下, 必须 debug
#
# 时间预算:
#   * 单 suite × 50 trials/task × 10 tasks = 500 rollouts
#   * 4090 上 ~1.5-2 小时/suite
#   * 全 4 suite = 6-8 小时
#
# 运行方式:
#   # 只跑一个 suite (Week 1 冲刺, 快)
#   bash eval/run_libero_orig.sh long
#
#   # 全 4 suite (Week 2 完整复现)
#   bash eval/run_libero_orig.sh all
#
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. 环境激活 + 路径
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ript_vla_openvla_oft

CODE_DIR="${CODE_DIR:-$HOME/code}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/ra-loop}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_DIR="${PROJ_DIR:-$PROJECT_ROOT}"
OFFICIAL_LIBERO_DIR="${OFFICIAL_LIBERO_DIR:-$CODE_DIR/LIBERO-official}"

# LIBERO-plus intentionally replaces the `libero` package for robustness
# evaluation.  Prepend the official checkout here so original LIBERO results
# are never silently evaluated on the expanded Plus task set.
if [[ ! -d "$OFFICIAL_LIBERO_DIR/libero/libero" ]]; then
  echo "  ERROR: official LIBERO not found at $OFFICIAL_LIBERO_DIR"
  exit 1
fi
export PYTHONPATH="$OFFICIAL_LIBERO_DIR:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PROJECT_ROOT/.libero-official}"

cd "$CODE_DIR/openvla-oft"

# ---------------------------------------------------------------------------
# 1. suite 映射
# ---------------------------------------------------------------------------
# 输入的 short name 映射到 (task_suite_name, unnorm_key, ckpt suffix)
declare -A TASK_SUITE=(
  [spatial]="libero_spatial"
  [object]="libero_object"
  [goal]="libero_goal"
  [long]="libero_10"        # 官方 task suite name 是 libero_10, 但我们习惯叫 long
)
declare -A UNNORM_KEY=(
  [spatial]="libero_spatial_no_noops"
  [object]="libero_object_no_noops"
  [goal]="libero_goal_no_noops"
  [long]="libero_10_no_noops"
)

# ---------------------------------------------------------------------------
# 2. 参数解析
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <suite>"
  echo "  suite: spatial | object | goal | long | all"
  exit 1
fi

if [[ "$1" == "all" ]]; then
  SUITES=(spatial object goal long)
else
  SUITES=("$1")
fi

# ---------------------------------------------------------------------------
# 3. 每个 suite 跑评估
# ---------------------------------------------------------------------------
LOG_DIR="$PROJ_DIR/logs/baseline_orig_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

for suite in "${SUITES[@]}"; do
  CKPT="$MODEL_DIR/openvla-oft-$suite"
  TASK="${TASK_SUITE[$suite]}"
  UN_KEY="${UNNORM_KEY[$suite]}"

  if [[ ! -d "$CKPT" ]]; then
    echo "  ERROR: checkpoint $CKPT not found. Run setup/step04_data.sh first."
    exit 1
  fi

  LOG_FILE="$LOG_DIR/eval_${suite}.log"
  echo ""
  echo "============================================================================"
  echo "  Evaluating suite: $suite"
  echo "  Checkpoint      : $CKPT"
  echo "  Task suite name : $TASK"
  echo "  Log file        : $LOG_FILE"
  echo "============================================================================"

  # 关键参数 (from OpenVLA-OFT LIBERO.md):
  #   --center_crop True       ← 必须, SFT 用了 random crop augmentation
  #   --num_trials_per_task 50 ← 官方默认, 10 tasks × 50 = 500 rollouts
  #   --seed 7                 ← 官方默认
  #
  # 我们的改动:
  #   --local_log_dir  → 我们的 LOG_DIR
  #   --use_wandb True → 开 WandB 便于对比
  #
  # 注意: OpenVLA-OFT 的 run_libero_eval.py 假定 CWD = openvla-oft 根目录,
  # 因为它有 `sys.path.append("../..")`.  所以我们 `cd` 到 openvla-oft.

  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK" \
    --unnorm_key "$UN_KEY" \
    --center_crop True \
    --num_trials_per_task 50 \
    --seed 7 \
    --local_log_dir "$LOG_DIR" \
    --use_wandb False \
    2>&1 | tee "$LOG_FILE"

  # 从 log 中提取最终 success rate
  # OpenVLA-OFT 的 eval script 会在末尾打印 "Average success rate: X.XX"
  RESULT=$(grep -oE "Average success rate: [0-9.]+" "$LOG_FILE" | tail -1 | awk '{print $NF}')
  echo "  Suite $suite result: $RESULT"
  echo "$suite,$RESULT" >> "$LOG_DIR/summary.csv"
done

# ---------------------------------------------------------------------------
# 4. 汇总输出与 gate 判断
# ---------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "  Summary (this run vs official A100 numbers):"
echo "============================================================================"

declare -A OFFICIAL=(
  [spatial]=0.976
  [object]=0.984
  [goal]=0.979
  [long]=0.945
)

# 打印表格 + 简单 gate 判断
python - <<PY
import csv
official = {"spatial": 0.976, "object": 0.984, "goal": 0.979, "long": 0.945}
with open("$LOG_DIR/summary.csv") as f:
    rows = list(csv.reader(f))

print(f"  {'Suite':<10}{'Ours':>10}{'Official':>12}{'Delta':>10}   Gate")
print(f"  {'-'*10}{'-'*10}{'-'*12}{'-'*10}   ----")
any_fail = False
for suite, ours in rows:
    try:
        ours_f = float(ours)
    except ValueError:
        print(f"  {suite:<10}{'PARSE_FAIL':>10}")
        any_fail = True
        continue
    off = official.get(suite, 0.0)
    delta = ours_f - off
    delta_pct = delta * 100
    if abs(delta_pct) < 5:
        gate = "PASS"
    elif abs(delta_pct) < 8:
        gate = "WARN"
        any_fail = True
    else:
        gate = "FAIL"
        any_fail = True
    print(f"  {suite:<10}{ours_f*100:>9.1f}%{off*100:>11.1f}%{delta_pct:>+9.1f}%   {gate}")

if any_fail:
    print("\n  Some suites failed gate. Do NOT proceed to Week 3 until fixed.")
    print("  Debug hints:")
    print("    * Confirm --center_crop True was set")
    print("    * Confirm transformers is moojink fork")
    print("    * Try --seed 7 and rerun (should be deterministic)")
    print("    * Check that --unnorm_key matches suite (e.g. libero_10_no_noops for long)")
PY
