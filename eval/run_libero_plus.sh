#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  eval/run_libero_plus.sh — LIBERO-Plus 7 维度鲁棒性评估
# ============================================================================
#
# 目的:
#   跑 OpenVLA-OFT 官方 SFT 权重在 LIBERO-Plus (7 维度扰动) 上的评估.
#   这个成绩就是我们的 baseline, 也是要超过的目标.
#
# 官方基线成绩 (LIBERO-Plus README, 2026-07 核实):
#   OpenVLA-OFT (LIBERO-Long/10 权重): 69.6% Total
#     - Camera:  56.4%
#     - Robot:   31.9%     ← 最大缺口, 我们主攻方向
#     - Language: 79.5%
#     - Light:   88.7%
#     - Background: 93.3%
#     - Noise:   75.8%
#     - Layout:  74.2%     ← 第二缺口, 次要目标
#
# 关键差异 (vs 原版 LIBERO 评估):
#   * num_trials_per_task = 1 (共 10030 任务, 不是 50 trials × 10 tasks)
#   * 每个 task 有 category label (7 维度之一), 结果需要按 category 聚合
#   * task_suite_name 可能是 libero_plus 或类似, 需要看 LIBERO-plus repo
#
# 时间预算:
#   * 10030 rollouts, 每个 ~5-20 秒 (取决于 max steps)
#   * 4090 上单卡 ~15-25 小时/次评估
#   * 用 7 张卡并行 (每卡负责 ~1430 任务) → 2-4 小时
#
# 运行方式:
#   bash eval/run_libero_plus.sh <ckpt_path> [num_gpus]
#     ckpt_path: OpenVLA-OFT 权重路径, 默认 $MODEL_DIR/openvla-oft-long
#     num_gpus:  并行 GPU 数, 默认 7
#
# ============================================================================

set -euo pipefail

CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ript_vla_openvla_oft

CODE_DIR="${CODE_DIR:-$HOME/code}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/ra-loop}"
PROJ_DIR="${PROJ_DIR:-$HOME/Desktop/essay/RA-LOOP}"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
CKPT="${1:-$MODEL_DIR/openvla-oft-long}"
NUM_GPUS="${2:-7}"

if [[ ! -d "$CKPT" ]]; then
  echo "  ERROR: checkpoint $CKPT not found"
  exit 1
fi

# ---------------------------------------------------------------------------
# LIBERO-plus 评估策略:
#
# LIBERO-plus 兼容原版 LIBERO API, 但 task suite 定义不同.
# 关键: 修改 num_trials_per_task = 1 (作者原话:
#   "The only required modification is adjusting num_trials_per_task from 50 to 1")
#
# LIBERO-plus 提供 task_classification.json, 记录每个 task 属于哪个 perturbation category.
# 位置: $CODE_DIR/LIBERO-plus/libero/libero/benchmark/task_classification.json
#
# 评估流程:
#   1. 用 LIBERO-plus 的 benchmark 列表, 遍历所有 category 的 task suite
#   2. 每个 task 只跑 1 次
#   3. 按 category 聚合 success rate
#
# LIBERO-plus 具体的 benchmark 名字, 我们从 task_classification.json 里读.
# 每个 suite (原版 spatial/object/goal/long) 都有对应的 plus 版本, 命名如
#   libero_spatial_plus, libero_object_plus, ... (待核实, W1 跑通时确认)
# ---------------------------------------------------------------------------

LOG_DIR="$PROJ_DIR/logs/baseline_plus_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cd "$CODE_DIR/openvla-oft"

# 并行策略: 用 CUDA_VISIBLE_DEVICES 分发, 每卡负责一部分 task
# 这里先写单卡版本 (Week 1 D3, 让流程跑通), 多卡版本在 eval/run_libero_plus_parallel.sh 里
#
# 单卡实现: 遍历 4 个 suite (spatial/object/goal/long), 每个都跑 LIBERO-plus 的完整任务集
#
# 注意: 我们只用 libero-10 权重 (long suite 训练的), 因为它是 LIBERO-Plus README 的
# 参考对象 (Table 中 OpenVLA-OFT 一行). 单 suite 权重比 mix-sft 权重更 focused.

echo "============================================================================"
echo "  LIBERO-Plus evaluation"
echo "  Checkpoint : $CKPT"
echo "  Log dir    : $LOG_DIR"
echo "============================================================================"

# LIBERO-plus 的四个 suite 名字 (官方 task_classification.json 里的 key)
# 从核实到的 JSON 结构确认: libero_spatial, libero_object, libero_goal, libero_10
PLUS_SUITES=(libero_spatial libero_object libero_goal libero_10)

# 单卡串行版本 (Week 1 冲刺, 慢)
for TASK in "${PLUS_SUITES[@]}"; do
  LOG_FILE="$LOG_DIR/plus_${TASK}.log"
  echo ""
  echo "  [$(date +%H:%M:%S)] Evaluating $TASK on LIBERO-plus..."

  # ⭐ 关键: num_trials_per_task=1
  # ⭐ initial_states_path: LIBERO-plus 用它读扰动 init 状态
  #    默认路径 = LIBERO-plus assets 里的 JSON. 具体路径 openvla-oft 应该自动找,
  #    如果找不到, 我们 Week 1 debug 时再指定.
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK" \
    --unnorm_key "${TASK}_no_noops" \
    --center_crop True \
    --num_trials_per_task 1 \
    --seed 7 \
    --local_log_dir "$LOG_DIR" \
    --use_wandb False \
    2>&1 | tee "$LOG_FILE"
done

# ---------------------------------------------------------------------------
# 汇总: 按 LIBERO-Plus 的 7 category 分类
# ---------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "  Aggregating results by 7 perturbation categories..."
echo "============================================================================"

python "$PROJ_DIR/eval/parse_libero_plus_results.py" \
  --log_dir "$LOG_DIR" \
  --task_classification "$CODE_DIR/LIBERO-plus/libero/libero/benchmark/task_classification.json" \
  --output "$LOG_DIR/summary_by_category.csv"

cat "$LOG_DIR/summary_by_category.csv"

# ============================================================================
# 成功预期 (baseline):
#   Total: 65-72%  (官方 A100 报 69.6%, 4090 允许 ±3%)
#   Camera : 50-60%
#   Robot  : 28-35%  ← 我们的机会缺口
#   Layout : 68-78%  ← 第二机会缺口
#
# 失败模式:
#   A. task_suite_name 报 unknown suite
#      —— 说明 LIBERO-plus 覆盖不完全. 检查 python -c "from libero.libero.benchmark import get_benchmark_dict; print(get_benchmark_dict().keys())"
#         如果没看到 libero_spatial 等, LIBERO-plus 的 pip install -e . 没生效.
#
#   B. 每个 task 只跑 1 次导致方差大 (成绩波动 ±5%)
#      —— 这是 LIBERO-Plus 设计特性, 靠 10030 任务的大数定律平均.
#      —— 如果单次成绩明显低于官方, 用 --seed 7 复现, 或用多 seed 平均.
#
#   C. 显存不足 (OOM during rollout)
#      —— 关闭 quantization, 用 bf16.
#      —— 或降 batch size (rollout 本来就是 bs=1).
#
#   D. 某些 category 结果异常 (比如 Language 30% 但官方 79%)
#      —— 说明 LIBERO-plus 的 language 扰动数据没找到.
#      —— 检查 $LIBERO-plus/libero/libero/assets/ 里是不是有对应 JSON.
# ============================================================================
