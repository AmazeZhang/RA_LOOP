#!/usr/bin/env python3
"""CPU-only calibration of LIBERO-Plus Robot Initial States variants.

This script parses source files and metadata.  It does not import LIBERO,
robosuite, or MuJoCo, and it never creates an environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_TASK = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"


def _safe_numeric_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_numeric_eval(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_numeric_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _safe_numeric_eval(node.left)
        right = _safe_numeric_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
        and node.attr == "pi"
    ):
        return math.pi
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "array"
        and len(node.args) == 1
    ):
        return np.asarray(_safe_numeric_eval(node.args[0]), dtype=np.float64)
    raise ValueError(f"unsupported numeric AST: {ast.dump(node, include_attributes=False)}")


def parse_qpos_classes(source_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    canonical = None
    variants: dict[int, np.ndarray] = {}
    class_pattern = re.compile(r"OnTheGroundPanda(\d+)$")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "OnTheGroundPanda":
            variant_id = None
        else:
            match = class_pattern.fullmatch(node.name)
            if not match:
                continue
            variant_id = int(match.group(1))
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name != "init_qpos":
                continue
            returns = [item for item in member.body if isinstance(item, ast.Return)]
            if len(returns) != 1 or returns[0].value is None:
                raise ValueError(f"unexpected init_qpos body in {node.name}")
            qpos = np.asarray(_safe_numeric_eval(returns[0].value), dtype=np.float64)
            if qpos.shape != (7,):
                raise ValueError(f"{node.name} qpos has shape {qpos.shape}, expected (7,)")
            if variant_id is None:
                canonical = qpos
            else:
                variants[variant_id] = qpos
    if canonical is None:
        raise ValueError("canonical OnTheGroundPanda.init_qpos was not found")
    return canonical, variants


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def fmt(values: list[float] | np.ndarray, digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    plus_root = args.plus_root.resolve()
    classification_path = plus_root / "libero/libero/benchmark/task_classification.json"
    robot_source_path = plus_root / "libero/libero/envs/robots/on_the_ground_panda.py"
    init_path = plus_root / f"libero/libero/init_files/{args.suite}/{args.task}.pruned_init"

    classification = json.loads(classification_path.read_text(encoding="utf-8"))[args.suite]
    selected = [
        row
        for row in classification
        if row["category"] == "Robot Initial States"
        and row["name"].startswith(args.task + "_view_")
    ]
    if not selected:
        raise ValueError(f"no Robot Initial States entries found for {args.suite}/{args.task}")

    selected_ids: list[int] = []
    for row in selected:
        match = re.search(r"_initstate_(\d+)$", row["name"])
        if not match:
            raise ValueError(f"cannot parse initstate id: {row['name']}")
        selected_ids.append(int(match.group(1)))

    canonical, variants = parse_qpos_classes(robot_source_path)
    missing = sorted(set(selected_ids) - variants.keys())
    if missing:
        raise ValueError(f"qpos classes missing for initstate ids: {missing}")
    selected_qpos = np.stack([variants[item] for item in selected_ids])
    deltas = selected_qpos - canonical
    l2 = np.linalg.norm(deltas, axis=1)
    rms = np.sqrt(np.mean(np.square(deltas), axis=1))
    max_abs = np.max(np.abs(deltas), axis=1)

    base_states = torch.load(init_path, map_location="cpu", weights_only=False)
    base_states = np.asarray(base_states, dtype=np.float64)
    if base_states.ndim != 2 or base_states.shape[1] < 8:
        raise ValueError(f"unexpected base init-state shape {base_states.shape}")
    demo_qpos = base_states[:, 1:8]
    demo_deltas = demo_qpos - canonical
    demo_l2 = np.linalg.norm(demo_deltas, axis=1)

    by_level: dict[int, list[int]] = defaultdict(list)
    for pos, row in enumerate(selected):
        by_level[int(row["difficulty_level"])].append(pos)

    level_rows = []
    for level in sorted(by_level):
        positions = np.asarray(by_level[level], dtype=np.int64)
        level_rows.append(
            {
                "difficulty": level,
                "count": int(len(positions)),
                "ids": [selected_ids[pos] for pos in positions],
                "l2": quantiles(l2[positions]),
                "rms": quantiles(rms[positions]),
                "max_abs": quantiles(max_abs[positions]),
            }
        )

    radius_by_id = {item: 0.1 * math.ceil(item / 100) for item in selected_ids}
    radius_rows = []
    for radius in sorted(set(radius_by_id.values())):
        positions = np.asarray(
            [pos for pos, item in enumerate(selected_ids) if radius_by_id[item] == radius],
            dtype=np.int64,
        )
        radius_rows.append(
            {
                "radius": radius,
                "count": int(len(positions)),
                "difficulty_counts": dict(
                    sorted(Counter(int(selected[pos]["difficulty_level"]) for pos in positions).items())
                ),
                "l2": quantiles(l2[positions]),
                "rms": quantiles(rms[positions]),
                "max_abs": quantiles(max_abs[positions]),
            }
        )

    expected_chi7 = math.sqrt(2.0) * math.gamma(4.0) / math.gamma(3.5)
    sigma_for_expected_l2_01 = 0.1 / expected_chi7

    result = {
        "suite": args.suite,
        "task": args.task,
        "selected_count": len(selected),
        "difficulty_counts": dict(sorted(Counter(int(row["difficulty_level"]) for row in selected).items())),
        "selected_ids": selected_ids,
        "canonical_qpos": canonical.tolist(),
        "selected": {
            "l2": quantiles(l2),
            "rms_per_joint": quantiles(rms),
            "max_abs_joint": quantiles(max_abs),
            "mean_abs_per_joint": np.mean(np.abs(deltas), axis=0).tolist(),
            "p90_abs_per_joint": np.quantile(np.abs(deltas), 0.90, axis=0).tolist(),
        },
        "by_difficulty": level_rows,
        "by_generation_radius": radius_rows,
        "gaussian_sigma_for_expected_l2_0.1": sigma_for_expected_l2_01,
        "base_init_states": {
            "shape": list(base_states.shape),
            "qpos_mean": np.mean(demo_qpos, axis=0).tolist(),
            "qpos_std": np.std(demo_qpos, axis=0).tolist(),
            "l2_from_canonical": quantiles(demo_l2),
        },
    }

    lines = [
        "# LIBERO-Plus Spatial Robot-init 关节偏移标定 — 20260720",
        "",
        "> CPU-only、只读标定；未导入 LIBERO/robosuite/MuJoCo，未创建模拟器，未使用 GPU，未修改 Plus 数据。",
        "",
        "## 标定对象与机制",
        "",
        f"- suite：`{args.suite}`",
        f"- task：`{args.task}`",
        f"- 分类表中该任务 Robot Initial States 变体：{len(selected)} 个",
        f"- difficulty 数量：{dict(sorted(Counter(int(row['difficulty_level']) for row in selected).items()))}",
        f"- canonical Panda qpos：`{fmt(canonical)}` rad",
        "- Plus 的 `initstate_N` 会选择 `PandaN`；500 个 qpos 是以 canonical qpos 为中心、固定随机方向生成。",
        "  编号 1--100、101--200、…、401--500 的 L2 半径分别为 0.1、0.2、…、0.5 rad，",
        "  而不是从任务的 50 条 `.pruned_init` 中按 N 索引。",
        "- 分类 JSON 的 `difficulty_level` 是另一套标签，不等同于物理 L2 半径；下面分别报告，不能互换。",
        "",
        "## 选中变体的实测分布",
        "",
        "按生成半径：",
        "",
        "| 生成 L2 半径 | 数量 | difficulty 计数 | 实测 L2 中位数 | RMS/关节中位数 | 最大单关节绝对偏移中位数 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in radius_rows:
        lines.append(
            f"| {row['radius']:.1f} | {row['count']} | {row['difficulty_counts']} | "
            f"{row['l2']['p50']:.4f} | {row['rms']['p50']:.4f} | {row['max_abs']['p50']:.4f} |"
        )
    lines.extend(
        [
            "",
            "按分类 difficulty（可见它与半径不单调对应）：",
            "",
            "| difficulty | 数量 | joint-delta L2 中位数 | RMS/关节中位数 | 最大单关节绝对偏移中位数 |",
        "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in level_rows:
        lines.append(
            f"| {row['difficulty']} | {row['count']} | {row['l2']['p50']:.4f} | "
            f"{row['rms']['p50']:.4f} | {row['max_abs']['p50']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"全部 37 个变体的 L2 分位数：`{json.dumps(quantiles(l2), ensure_ascii=False)}`。",
            f"全部变体 RMS/关节分位数：`{json.dumps(quantiles(rms), ensure_ascii=False)}`。",
            f"逐关节 mean |delta|：`{fmt(np.mean(np.abs(deltas), axis=0))}` rad。",
            f"逐关节 p90 |delta|：`{fmt(np.quantile(np.abs(deltas), 0.90, axis=0))}` rad。",
            "",
            "## 与原任务 init-state 的关系",
            "",
            f"原 `.pruned_init` 为 `{tuple(base_states.shape)}`，其中 state[1:8] 是 Panda 7 关节。",
            f"其相对 canonical qpos 的 L2 分位数为：`{json.dumps(quantiles(demo_l2), ensure_ascii=False)}`。",
            f"原任务关节标准差：`{fmt(np.std(demo_qpos, axis=0))}` rad。",
            "这 50 条状态本身已带有小幅初始化散布；Plus Robot-init 是在机器人模型默认 qpos 上施加独立、明显更大的分档偏移。",
            "",
            "## 对首个 RA learning-signal probe 的结论",
            "",
            "- 当前 RA `strength` 实际是每关节独立 Gaussian 的标准差，不是总 L2；connectivity smoke 的 0.001 rad 对应期望 L2 约 0.00255 rad，只适合验证链路。",
            f"- 若保持现实现，首个 h220/K8 probe 可用 `strength={sigma_for_expected_l2_01:.4f}` rad/关节，使 7 维 Gaussian 的期望 L2 为 0.1 rad，近似对齐 Plus 最轻生成档；仍保持 recovery-only 与 `lambda_consistency=0`。",
            "- 更严格的复现方式是在下一步先增加 `fixed_l2` 采样模式，再直接配置 0.1 rad；这会与 Plus 的归一化随机方向完全同单位，也更容易解释。",
            "- 若首轮 perturbed 成功率与 anchor 几乎无差异，再单独确认后试 0.2 rad L2；第一次 probe 不同时扫描多档。",
            "",
            "## 实际 baseline evaluator 的覆盖语义风险",
            "",
            "静态调用链显示：本项目 bounded evaluator 使用 upstream `run_task`，每个变体只跑 episode 0；",
            "upstream 先 `env.reset()`，随后把基础 `.pruned_init[0]` 传给 Plus `set_init_state()`；",
            "Plus 的该方法直接调用 `sim.set_state_from_flattened()`。这会在 PandaN reset 之后覆盖 7 个 robot qpos。",
            "因此现有 38.57% 结果按代码路径并未保留 PandaN 的默认关节偏移；它是分类表覆盖完整的结果，",
            "但不能作为‘真实 Robot-init 物理偏移’已被正确施加的证据。该问题不会影响 RA adapter：RA 是在送入 rollout 前直接修改 flattened state[1:8]。",
            "在改动 evaluator 或重跑任何 GPU baseline 前，应单独做一个 CPU/live 双状态读回 gate，并由用户确认评测口径。",
            "",
            "## 可复跑命令",
            "",
            "```bash",
            "PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \\",
            "/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python \\",
            "  scripts/calibrate_libero_plus_robot_init.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus-root", type=Path, default=Path("/home/imc/code/LIBERO-plus"))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--report", type=Path, default=Path("docs/LIBERO_PLUS_ROBOT_INIT_CALIBRATION_20260720.md"))
    parser.add_argument("--json", type=Path, default=Path("docs/LIBERO_PLUS_ROBOT_INIT_CALIBRATION_20260720.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, result = build_report(args)
    args.report.write_text(report, encoding="utf-8")
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report, end="")
    print(f"[saved] {args.report}")
    print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
