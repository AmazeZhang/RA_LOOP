#!/bin/bash
# =====================================================================
# step06_patch_ript.sh
# ---------------------------------------------------------------------
# 把 RIPT-VLA 官方 train_ript_openvla_oft.py 里创建 RLOptimizer /
# RolloutGenerator 的两行, 替换成调用 code.ra_optimizer.build_rl_optimizer_from_cfg
# 与 RAPerturbedRolloutGenerator, 让 RA-LOOP 训练可以走 Hydra 组合出来的 cfg。
#
# 该 patch 是可逆的:
#   - 首次执行前, 会把原文件备份到 train_ript_openvla_oft.py.bak
#   - 若发现 .bak 已存在, 视为已 patch 过, 幂等退出
#
# 用法:
#   bash setup/step06_patch_ript.sh
# =====================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${PROJECT_ROOT}/../repos/ript-vla"
TARGET="${REPO}/train_ript_openvla_oft.py"

if [[ ! -f "${TARGET}" ]]; then
    echo "[FATAL] 找不到 ${TARGET}"
    echo "  请先执行 setup/step02_repos.sh 克隆 RIPT-VLA"
    exit 1
fi

if [[ -f "${TARGET}.bak" ]]; then
    echo "[INFO] ${TARGET}.bak 已存在, patch 已施加, 幂等退出"
    exit 0
fi

cp "${TARGET}" "${TARGET}.bak"
echo "[OK] 备份原文件到 ${TARGET}.bak"

# ---- 用 python 做精确替换（避免 sed 在 macOS/Linux 语法差异）----
python <<'PY'
import re
import pathlib
import os

target = pathlib.Path(os.environ.get('TARGET_FILE',
    pathlib.Path(__file__).parent.parent.parent + '/repos/ript-vla/train_ript_openvla_oft.py'
))

# 找不到 TARGET_FILE 环境变量时按传统路径推断
if 'TARGET_FILE' not in os.environ:
    # 由脚本执行时传入
    pass

PY

# 用一个真正 portable 的 python 单行替换
TARGET_FILE="${TARGET}" python - <<'PY'
import os, re, pathlib

target = pathlib.Path(os.environ['TARGET_FILE'])
src = target.read_text()

patched = src

# --- 1) 替换 RLOptimizer 实例化 ---
# 官方大致形如:
#   rl_optimizer = RLOptimizerOpenVLAOFT(
#       rollout_generator=rollout_generator,
#       reward_function=reward_function,
#       ...
#   )
# 我们把整块换成一行 factory 调用。
rl_pattern = re.compile(
    r"rl_optimizer\s*=\s*RLOptimizerOpenVLAOFT\s*\((?:[^()]|\([^()]*\))*\)",
    re.MULTILINE,
)
replacement_rl = (
    "from code.ra_optimizer import build_rl_optimizer_from_cfg  # RA-LOOP patch\n"
    "    rl_optimizer = build_rl_optimizer_from_cfg(cfg, rollout_generator, reward_function)"
)
new_src, n_rl = rl_pattern.subn(replacement_rl, patched, count=1)
if n_rl == 0:
    raise SystemExit(
        "[FATAL] 未在 train_ript_openvla_oft.py 中找到 RLOptimizerOpenVLAOFT(...) 构造,\n"
        "        可能上游文件版本已变, 请人工核对后再 patch"
    )
patched = new_src

# --- 2) 替换 RolloutGenerator 实例化 ---
rg_pattern = re.compile(
    r"rollout_generator\s*=\s*RolloutGenerator\s*\((?:[^()]|\([^()]*\))*\)",
    re.MULTILINE,
)
replacement_rg = (
    "from code.ra_optimizer import RAPerturbedRolloutGenerator  # RA-LOOP patch\n"
    "    rollout_generator = RAPerturbedRolloutGenerator(\n"
    "        rloo_batch_size=cfg.algo.rloo_batch_size,\n"
    "        demo_batch_size=cfg.train_dataloader.batch_size,\n"
    "        enable_dynamic_sampling=cfg.algo.enable_dynamic_sampling,\n"
    "        task_names_to_use=task_names_to_use,\n"
    "        env_runner=env_runner,\n"
    "        create_env=True,\n"
    "        perturbation_cfg=dict(cfg.algo.perturbation),\n"
    "    )"
)
new_src2, n_rg = rg_pattern.subn(replacement_rg, patched, count=1)
if n_rg == 0:
    print("[WARN] 未找到 RolloutGenerator(...) 构造, 跳过 rollout patch")
    print("       若你手动改过 rollout_generator 创建位置, 请自行确认已使用 RAPerturbedRolloutGenerator")
else:
    patched = new_src2

target.write_text(patched)
print(f"[OK] RLOptimizer 替换次数: {n_rl}")
print(f"[OK] RolloutGenerator 替换次数: {n_rg}")
PY

echo "[OK] patch 完成, 请用 diff 检查:"
echo "  diff ${TARGET}.bak ${TARGET}"
