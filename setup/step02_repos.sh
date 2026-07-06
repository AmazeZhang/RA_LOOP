#!/usr/bin/env bash
# ============================================================================
# RA-LOOP  Step 02 — Clone 三个核心仓库
# ============================================================================
#
# 目的:
#   拉取 RA-LOOP 依赖的三个 upstream 仓库:
#     1. openvla-oft         — VLA 模型代码 (moojink/openvla-oft)
#     2. ript-vla            — RIPT / LOOP 算法代码 (Ariostgx/ript-vla)
#     3. LIBERO-plus         — 鲁棒性 benchmark (sylvestf/LIBERO-plus)
#
# 注意:
#   * 我们**不 clone** 官方 LIBERO (Lifelong-Robot-Learning/LIBERO)
#     因为 LIBERO-plus 是它的完全替代品 (作者原话:
#     "You can simply replace the original libero with a pip install -e .")
#   * clone 到 ~/code/ 而不是项目目录, 避免 IDE 卡顿
#
# 运行方式:
#   bash setup/step02_repos.sh
#
# 预期完成时间: 5-10 分钟 (取决于网络)
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. 建立代码目录
# ---------------------------------------------------------------------------
CODE_DIR="${CODE_DIR:-$HOME/code}"
mkdir -p "$CODE_DIR"
cd "$CODE_DIR"

echo "[STEP 02] Cloning repositories into $CODE_DIR"

# ---------------------------------------------------------------------------
# 1. Clone openvla-oft (Moo Jin Kim / Stanford)
# ---------------------------------------------------------------------------
# 这个仓库是 OpenVLA 的官方 OFT (Optimized Fine-Tuning) 版本.
# 包含:
#   * prismatic/       — 模型代码 (从 OpenVLA fork)
#   * experiments/robot/libero/  — LIBERO 评估脚本
#   * vla-scripts/finetune.py    — SFT 训练脚本
#
# 我们的 RA-LOOP 会从这个仓库复用:
#   * get_vla() / get_processor() 载入 checkpoint
#   * run_libero_eval.py 的评估 framework
if [[ ! -d openvla-oft ]]; then
  git clone https://github.com/moojink/openvla-oft.git
else
  echo "  openvla-oft already cloned, pulling latest..."
  (cd openvla-oft && git pull --ff-only) || echo "  (pull skipped: might be diverged)"
fi

# 记录 commit hash, 保证可复现
(cd openvla-oft && git rev-parse HEAD > /tmp/openvla-oft-commit.txt)
echo "  openvla-oft commit: $(cat /tmp/openvla-oft-commit.txt)"

# ---------------------------------------------------------------------------
# 2. Clone ript-vla (Shuhan Tan / UT Austin)
# ---------------------------------------------------------------------------
# 官方 RIPT / LOOP 实现. 我们的 RA-LOOP 直接在这个 codebase 上改.
# 关键文件:
#   * ript/algos/rl_optimizers/rl_optimizer.py    ← 我们要改的核心
#   * ript/algos/rl_optimizers/rollout_generator.py ← 也要改 (加扰动 rollout)
#   * ript/env_runner/libero_runner.py            ← Env wrapper
#   * scripts/openvla_oft/stage_3_ript/*.sh       ← 参考训练脚本
if [[ ! -d ript-vla ]]; then
  git clone https://github.com/Ariostgx/ript-vla.git
else
  echo "  ript-vla already cloned, pulling latest..."
  (cd ript-vla && git pull --ff-only) || echo "  (pull skipped: might be diverged)"
fi

(cd ript-vla && git rev-parse HEAD > /tmp/ript-vla-commit.txt)
echo "  ript-vla commit: $(cat /tmp/ript-vla-commit.txt)"

# ---------------------------------------------------------------------------
# 3. Clone LIBERO-plus (sylvestf / Fudan)
# ---------------------------------------------------------------------------
# 鲁棒性 benchmark. 用 pip install -e . 覆盖官方 LIBERO.
# 关键点:
#   * 与官方 LIBERO API 完全兼容 (作者刻意保持)
#   * 只需要额外下载 assets.zip (~几GB)
#   * benchmark/task_classification.json 包含 7 维度 x 10030 任务的分类
if [[ ! -d LIBERO-plus ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git
else
  echo "  LIBERO-plus already cloned, pulling latest..."
  (cd LIBERO-plus && git pull --ff-only) || echo "  (pull skipped: might be diverged)"
fi

(cd LIBERO-plus && git rev-parse HEAD > /tmp/LIBERO-plus-commit.txt)
echo "  LIBERO-plus commit: $(cat /tmp/LIBERO-plus-commit.txt)"

# ---------------------------------------------------------------------------
# 4. 记录三个 commit 到项目日志 (可复现关键)
# ---------------------------------------------------------------------------
PROJ_DIR="${PROJ_DIR:-$HOME/Desktop/essay/RA-LOOP}"
mkdir -p "$PROJ_DIR/logs"
{
  echo "# Repo commits pinned at $(date -Iseconds)"
  echo "openvla-oft  $(cat /tmp/openvla-oft-commit.txt)"
  echo "ript-vla     $(cat /tmp/ript-vla-commit.txt)"
  echo "LIBERO-plus  $(cat /tmp/LIBERO-plus-commit.txt)"
} > "$PROJ_DIR/logs/repo_commits.txt"

cat "$PROJ_DIR/logs/repo_commits.txt"

echo ""
echo "============================================================================"
echo "[STEP 02] DONE."
echo "  Repos cloned to: $CODE_DIR/"
echo "  Commit hashes  : $PROJ_DIR/logs/repo_commits.txt"
echo ""
echo "  Next: bash setup/step03_deps.sh"
echo "============================================================================"

# ============================================================================
# 成功预期:
#   * 三个目录都存在 openvla-oft/, ript-vla/, LIBERO-plus/
#   * 每个目录都有 .git/
#   * repo_commits.txt 记录了 SHA
#
# 失败模式与应对:
#   A. git clone 卡住或超时
#      —— 换 HTTPS 为 SSH: git clone git@github.com:moojink/openvla-oft.git
#      —— 或用代理: git config --global http.proxy http://proxy:port
#      —— 或走国内镜像: sed 's|github.com|hub.fastgit.xyz|' (不稳定, 慎用)
#
#   B. LIBERO-plus 与官方 LIBERO API 不完全兼容
#      —— 这是**中风险**. 作者声称 100% 兼容, 但少数任务名有变化.
#      —— 应对: 先跑 step05_verify.py, 看能不能 load task suite.
#      —— 如果不行, Plan A' = clone 官方 LIBERO, 只用 LIBERO-plus 的 assets 和评估脚本.
#
#   C. commit hash 与本 README 记录不同
#      —— 说明 upstream 有更新. 通常没问题, 但如果 Week 2 复现失败,
#         可以 `git checkout <old-commit>` 回滚到本 README 记录的版本.
# ============================================================================
