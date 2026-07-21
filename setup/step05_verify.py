#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# RA-LOOP  Step 05 — 环境验证脚本
# ============================================================================
#
# 目的:
#   验证 Step 01-04 是否装对了. 具体检查:
#     1. Python / PyTorch / CUDA 版本
#     2. 关键包 import 无报错
#     3. LIBERO / LIBERO-plus 可以 load
#     4. OpenVLA-OFT 权重可以载入并跑一次前向
#     5. RIPT-VLA 的关键模块可以 import
#
# 运行方式:
#   conda activate ript_vla_openvla_oft
#   python setup/step05_verify.py
#
# 预期完成时间: 3-5 分钟 (首次载入模型慢)
# ============================================================================

from __future__ import annotations

import importlib
import os
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Avoid inheriting a stale ~/.libero/config.yaml from another checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))

# ---------------------------------------------------------------------------
# ANSI 颜色输出, 便于快速识别失败点
# ---------------------------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def section(name: str) -> None:
    print(f"\n{BOLD}=== {name} ==={RESET}")


# 收集所有失败, 最后统一报告
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        ok(f"{name}: {detail}" if detail else name)
    else:
        fail(f"{name}: {detail}" if detail else name)
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. Python & PyTorch 版本
# ---------------------------------------------------------------------------
section("1. Python & PyTorch")

py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
check(
    "Python 3.10.x",
    sys.version_info[:2] == (3, 10),
    f"got {py_ver}",
)

try:
    import torch

    check("torch 2.2.0", torch.__version__.startswith("2.2.0"), f"got {torch.__version__}")
    check("CUDA available", torch.cuda.is_available(), f"cuda={torch.version.cuda}")
    check("GPU count >= 1", torch.cuda.device_count() >= 1, f"got {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        # 检查是不是 4090
        gpu_name = torch.cuda.get_device_name(0)
        check("GPU is RTX 4090", "4090" in gpu_name, f"got '{gpu_name}'")

        # 检查每卡显存 >= 22GB (4090 是 24GB, 留 2GB 缓冲)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        check("VRAM >= 22 GB/GPU", vram_gb >= 22, f"got {vram_gb:.1f} GB")
except Exception as e:
    fail(f"torch import failed: {e}")
    FAILURES.append("torch")

# ---------------------------------------------------------------------------
# 2. 关键包 import
# ---------------------------------------------------------------------------
section("2. Key packages")

# (import_name, expected_version_prefix, is_critical)
PACKAGES = [
    ("transformers", "4.40", True),
    ("peft", "0.11.1", True),
    ("tokenizers", "0.19.1", True),
    ("timm", "0.9.10", True),
    ("accelerate", "1.6", True),
    ("diffusers", "0.30.3", True),
    ("flash_attn", "2.7", True),
    ("robosuite", "1.4", True),
    ("mujoco", "3.3", True),
    ("hydra", "1.3", False),
    ("wandb", "0.18", False),
    ("google.protobuf", "4.25", True),
    ("bddl", None, True),
    ("libero", None, True),
]

for pkg_name, expected_prefix, critical in PACKAGES:
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "unknown")
        if expected_prefix and not str(ver).startswith(expected_prefix):
            (fail if critical else warn)(f"{pkg_name}: expected {expected_prefix}.x, got {ver}")
            if critical:
                FAILURES.append(pkg_name)
        else:
            ok(f"{pkg_name} == {ver}")
    except Exception as e:
        (fail if critical else warn)(f"{pkg_name}: import failed — {e}")
        if critical:
            FAILURES.append(pkg_name)

# ---------------------------------------------------------------------------
# 3. transformers 是不是 moojink fork
# ---------------------------------------------------------------------------
section("3. transformers fork (moojink)")

try:
    import transformers

    # moojink fork 在 transformers/models/llama/modeling_llama.py 里
    # 加了 bidirectional attention 相关的参数.
    # 检测标志: LlamaForCausalLM.forward 签名里应该有 attention_mask 但更重要的是
    # 检查 transformers 的 __file__ 是不是从 git+ 装的.
    tf_file = transformers.__file__
    # moojink 装的会显示 site-packages/transformers/... 但 pip show 会显示
    # source 是 git+https://github.com/moojink/...
    check(
        "transformers is installed",
        tf_file is not None,
        f"path: {tf_file}",
    )

    # 用 pip show 拿 source URL 确认是 moojink fork
    result = subprocess.run(
        ["pip", "show", "transformers"],
        capture_output=True,
        text=True,
        check=False,
    )
    source_ok = "moojink" in result.stdout or "moojink" in result.stderr
    # 如果不是 moojink fork, 用 warn 提示 (可能能跑, 但 parallel decoding 会失败)
    if source_ok:
        ok("transformers is moojink fork (verified via pip show)")
    else:
        # 尝试另一个检测: 看是否有 attn_bias 参数支持
        warn(
            "transformers may NOT be moojink fork. "
            "OpenVLA-OFT parallel decoding might fail. "
            "Fix: pip install 'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git'"
        )
except Exception as e:
    fail(f"transformers check failed: {e}")
    FAILURES.append("transformers-fork")

# ---------------------------------------------------------------------------
# 4. flash-attn 能否实际跑
# ---------------------------------------------------------------------------
section("4. flash-attn runtime")

try:
    from flash_attn import flash_attn_func

    if torch.cuda.is_available():
        # 试跑一次前向, 保证 CUDA kernel 能加载
        # (import 成功不等于 kernel 能跑, torch/CUDA/flash-attn 版本不匹配时会崩)
        q = torch.randn(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16)
        out = flash_attn_func(q, k, v)
        assert out.shape == q.shape
        ok(f"flash_attn_func works (bf16, GPU): output shape {tuple(out.shape)}")
    else:
        warn("no CUDA, cannot test flash_attn runtime")
except Exception as e:
    fail(f"flash_attn runtime failed: {e}")
    FAILURES.append("flash_attn-runtime")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 5. LIBERO / LIBERO-plus 载入
# ---------------------------------------------------------------------------
section("5. LIBERO env")

try:
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()
    ok(f"libero.benchmark loaded, available: {list(bm.keys())[:5]}...")

    # LIBERO-plus 关键: 应该能找到 7 维度的 task_classification.json
    from libero.libero import get_libero_path

    task_cls_file = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    if task_cls_file.exists():
        ok(f"LIBERO-plus task_classification.json found: {task_cls_file}")
    else:
        # 可能路径不同, 打个 warn
        warn(f"task_classification.json not at {task_cls_file}")
        warn("This is only needed for LIBERO-plus 7-dimension eval, not blocking.")
except Exception as e:
    fail(f"LIBERO load failed: {e}")
    FAILURES.append("libero-load")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 6. RIPT-VLA 模块可以 import
# ---------------------------------------------------------------------------
section("6. RIPT-VLA modules")

try:
    # ript-vla 的核心模块, RA-LOOP 会改的
    from ript.algos.rl_optimizers.rl_optimizer import RLOptimizer  # noqa: F401

    ok("RLOptimizer (LOOP) importable")
except Exception as e:
    fail(f"RIPT-VLA RLOptimizer import failed: {e}")
    FAILURES.append("ript-vla-rloptim")

try:
    from ript.env_runner.libero_runner import LiberoRunner_rl  # noqa: F401

    ok("LiberoRunner_rl importable")
except Exception as e:
    fail(f"RIPT-VLA LiberoRunner_rl import failed: {e}")
    FAILURES.append("ript-vla-runner")

try:
    from ript.algos.rl_optimizers.rollout_generator import RolloutGenerator  # noqa: F401

    ok("RolloutGenerator importable")
except Exception as e:
    fail(f"RIPT-VLA RolloutGenerator import failed: {e}")
    FAILURES.append("ript-vla-rollout")

# ---------------------------------------------------------------------------
# 7. OpenVLA-OFT 权重存在
# ---------------------------------------------------------------------------
section("7. OpenVLA-OFT weights on disk")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", f"{os.environ['HOME']}/models/ra-loop"))
for suite in ["long", "spatial", "object", "goal"]:
    ckpt_dir = MODEL_DIR / f"openvla-oft-{suite}"
    ckpt = ckpt_dir / "config.json"
    index = ckpt_dir / "model.safetensors.index.json"
    if ckpt.exists() and index.exists():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            missing_shards = sorted(shard for shard in shards if not (ckpt_dir / shard).is_file())
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            missing_shards = [f"invalid index: {exc}"]
        if missing_shards:
            warn(f"openvla-oft-{suite}: incomplete checkpoint; missing {', '.join(missing_shards)}")
            continue
        # 显示模型大小
        size_gb = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / (1024**3)
        ok(f"openvla-oft-{suite}: {size_gb:.1f} GB")
    else:
        warn(f"openvla-oft-{suite}: not downloaded (run step04_data.sh)")

# ---------------------------------------------------------------------------
# 8. (可选) 载入 OpenVLA-OFT 并跑一次前向
# ---------------------------------------------------------------------------
section("8. OpenVLA-OFT forward pass (slow, ~2min)")

RUN_FORWARD = os.environ.get("SKIP_FORWARD", "0") != "1"
long_ckpt = MODEL_DIR / "openvla-oft-long"

if not RUN_FORWARD:
    warn("SKIP_FORWARD=1, skipping forward pass test")
elif not long_ckpt.exists():
    warn("openvla-oft-long not downloaded, skipping forward pass test")
elif not torch.cuda.is_available():
    warn("no CUDA, skipping forward pass test")
else:
    try:
        import pickle

        # 需要把 openvla-oft 代码目录加到 sys.path (它不是 pip install 时暴露的顶层包)
        CODE_DIR = Path(os.environ.get("CODE_DIR", f"{os.environ['HOME']}/code"))
        sys.path.insert(0, str(CODE_DIR / "openvla-oft"))

        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

        # 用 official README 的示例 config, 只改成 long suite
        cfg = GenerateConfig(
            pretrained_checkpoint=str(long_ckpt),
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key="libero_10_no_noops",  # long suite 对应的 unnorm key
        )

        t0 = time.time()
        vla = get_vla(cfg)
        ok(f"VLA loaded in {time.time() - t0:.1f}s")

        processor = get_processor(cfg)
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

        # 用官方的 sample observation
        sample_pkl = (
            CODE_DIR
            / "openvla-oft"
            / "experiments"
            / "robot"
            / "libero"
            / "sample_libero_spatial_observation.pkl"
        )
        with open(sample_pkl, "rb") as f:
            observation = pickle.load(f)

        t0 = time.time()
        actions = get_vla_action(
            cfg,
            vla,
            processor,
            observation,
            observation["task_description"],
            action_head,
            proprio_projector,
        )
        latency = time.time() - t0
        ok(f"Forward pass: got action chunk of len {len(actions)}, latency {latency:.2f}s")

        # 4090 上 latency 应该 < 0.5s (bf16)
        if latency > 2.0:
            warn(f"Latency {latency:.2f}s > 2.0s, might indicate FP32 fallback or slow disk")

    except Exception as e:
        fail(f"Forward pass failed: {e}")
        FAILURES.append("openvla-oft-forward")
        traceback.print_exc()

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print(f"\n{BOLD}{'='*76}{RESET}")
if not FAILURES:
    print(f"{BOLD}{GREEN}[STEP 05] ALL CHECKS PASSED{RESET}")
    print("  Next: bash eval/run_libero.sh   (Week 1 D3 baseline reproduction)")
else:
    print(f"{BOLD}{RED}[STEP 05] {len(FAILURES)} CHECK(S) FAILED:{RESET}")
    for name in FAILURES:
        print(f"  {RED}- {name}{RESET}")
    print(f"\n  {YELLOW}Consult setup/step0X_*.sh comments (`失败模式与应对`) for fixes.{RESET}")
    sys.exit(1)

# ============================================================================
# 成功预期:
#   * 所有 [OK], 最后打印 "ALL CHECKS PASSED"
#   * OpenVLA-OFT forward pass latency < 0.5s
#
# 失败模式与应对:
#   见每个 check 的 comment. 主要几类:
#     * transformers 不是 moojink fork          → 见 §3 的 fix 命令
#     * flash_attn 装了但 runtime 挂            → §4, 通常是 CUDA 版本不匹配
#     * LIBERO 找不到 task_classification.json  → §5, LIBERO-plus 覆盖失败
#     * OpenVLA-OFT 载入慢或 OOM               → §8, 检查 bf16/quantization 设置
# ============================================================================
