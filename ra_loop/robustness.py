"""CPU-only primitives for the first recovery-only RA-LOOP pilot.

This module intentionally has no RIPT, LIBERO, MuJoCo, or torch dependency.
Simulator integration belongs in a later adapter and must pass a validated
named-joint layout into these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PANDA_ARM_JOINT_NAMES = tuple(f"robot0_joint{i}" for i in range(1, 8))
SUPPORTED_PERTURBATIONS = frozenset({"none", "robot_init"})
SUPPORTED_SAMPLING_MODES = frozenset({"gaussian_std", "fixed_l2"})


@dataclass(frozen=True)
class RolloutPlanEntry:
    """One member of an anchor/Robot-init rollout pair."""

    rollout_index: int
    pair_id: int
    perturb_type: str
    strength: float
    seed: int
    is_perturbed: bool
    sampling_mode: str = "gaussian_std"


@dataclass(frozen=True)
class NamedJointLayout:
    """Validated scalar qpos addresses and limits for named joints."""

    joint_names: tuple[str, ...]
    qpos_indices: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class PerturbationResult:
    state: np.ndarray
    noise: np.ndarray
    applied: bool


@dataclass(frozen=True)
class RecoveryRewardResult:
    total: np.ndarray
    recovery: np.ndarray
    stats: dict[str, float]


def compute_mode_stratified_rloo_advantages(
    rewards: Sequence[float],
    perturbed_mask: Sequence[bool],
    *,
    rollout_group_ids: Sequence[int] | None = None,
    valid_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Compute leave-one-out advantages independently by rollout mode.

    Each ``(rollout_group_id, is_perturbed)`` stratum receives its own RLOO
    baseline. This prevents rewards from perturbed episodes from changing an
    anchor episode's advantage (and vice versa). Invalid padding receives zero
    advantage and is excluded from every baseline.

    At least two valid members are required in every represented stratum. The
    function fails closed instead of silently falling back to a cross-mode or
    self-referential baseline.
    """

    reward_array = np.asarray(rewards)
    if reward_array.ndim != 1 or reward_array.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional sequence")
    if np.issubdtype(reward_array.dtype, np.bool_) or not np.issubdtype(
        reward_array.dtype, np.number
    ):
        raise ValueError("rewards must be numeric and not boolean")
    reward_array = reward_array.astype(np.float64, copy=False)
    if not np.isfinite(reward_array).all():
        raise ValueError("rewards must be finite")

    count = reward_array.size

    def strict_bool_mask(values: Sequence[bool], name: str) -> np.ndarray:
        raw = np.asarray(values)
        if raw.shape != (count,) or not np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(f"{name} must contain one boolean per reward")
        return raw.astype(bool, copy=False)

    perturbed = strict_bool_mask(perturbed_mask, "perturbed_mask")
    valid = (
        np.ones(count, dtype=bool)
        if valid_mask is None
        else strict_bool_mask(valid_mask, "valid_mask")
    )
    if not valid.any():
        raise ValueError("valid_mask must select at least one reward")

    if rollout_group_ids is None:
        group_ids = np.zeros(count, dtype=np.int64)
    else:
        raw_groups = np.asarray(rollout_group_ids)
        if (
            raw_groups.shape != (count,)
            or np.issubdtype(raw_groups.dtype, np.bool_)
            or not np.issubdtype(raw_groups.dtype, np.integer)
        ):
            raise ValueError(
                "rollout_group_ids must contain one integer per reward"
            )
        group_ids = raw_groups.astype(np.int64, copy=False)
        if (group_ids < 0).any():
            raise ValueError("rollout_group_ids must be non-negative")

    advantages = np.zeros(count, dtype=np.float64)
    for group_id in np.unique(group_ids[valid]):
        group = valid & (group_ids == group_id)
        for is_perturbed in (False, True):
            stratum = group & (perturbed == is_perturbed)
            stratum_count = int(stratum.sum())
            if stratum_count < 2:
                mode = "perturbed" if is_perturbed else "anchor"
                raise ValueError(
                    f"rollout group {group_id} requires at least two valid "
                    f"{mode} rewards"
                )
            stratum_rewards = reward_array[stratum]
            baseline = (
                stratum_rewards.sum() - stratum_rewards
            ) / (stratum_count - 1)
            advantages[stratum] = stratum_rewards - baseline

    return advantages


def build_robot_init_rollout_plan(
    *,
    num_pairs: int,
    strength: float,
    base_seed: int,
    sampling_mode: str = "gaussian_std",
) -> tuple[RolloutPlanEntry, ...]:
    """Build interleaved real-anchor/Robot-init pairs.

    A positive strength is required so an entry labelled as perturbed cannot be
    a silent no-op. Each pair receives one anchor and one independently seeded
    Robot-init member.
    """

    if isinstance(num_pairs, bool) or not isinstance(num_pairs, int) or num_pairs < 1:
        raise ValueError("num_pairs must be a positive integer")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("Robot-init strength must be finite and positive")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"unsupported Robot-init sampling mode: {sampling_mode}")

    plan: list[RolloutPlanEntry] = []
    for pair_id in range(num_pairs):
        seed = base_seed + pair_id
        plan.append(
            RolloutPlanEntry(
                rollout_index=2 * pair_id,
                pair_id=pair_id,
                perturb_type="none",
                strength=0.0,
                seed=seed,
                is_perturbed=False,
                sampling_mode=sampling_mode,
            )
        )
        plan.append(
            RolloutPlanEntry(
                rollout_index=2 * pair_id + 1,
                pair_id=pair_id,
                perturb_type="robot_init",
                strength=float(strength),
                seed=seed,
                is_perturbed=True,
                sampling_mode=sampling_mode,
            )
        )
    return tuple(plan)


def _scalar_qpos_address(raw_address: Any, joint_name: str) -> int:
    if isinstance(raw_address, (tuple, list, np.ndarray)):
        if len(raw_address) != 2:
            raise ValueError(f"unexpected qpos address for {joint_name}: {raw_address}")
        start, end = (int(raw_address[0]), int(raw_address[1]))
        if end - start != 1:
            raise ValueError(f"joint {joint_name} is not a scalar qpos joint")
        return start
    if isinstance(raw_address, (int, np.integer)):
        return int(raw_address)
    raise ValueError(f"missing scalar qpos address for joint {joint_name}")


def resolve_named_joint_layout(
    model: Any,
    joint_names: Sequence[str] = PANDA_ARM_JOINT_NAMES,
    *,
    state_qpos_offset: int = 1,
) -> NamedJointLayout:
    """Resolve scalar qpos addresses and limits using simulator joint names.

    Supports the legacy MuJoCo-py/robosuite lookup methods used by LIBERO and
    the newer ``model.joint(name)`` accessor. Missing, duplicate, unlimited, or
    invalid joints fail closed. LIBERO stores ``MjSimState.flatten()``, whose
    first scalar is simulation time, so qpos addresses use an explicit default
    offset of one when mapped into the flattened initialization state.
    """

    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be non-empty and unique")
    if (
        isinstance(state_qpos_offset, bool)
        or not isinstance(state_qpos_offset, int)
        or state_qpos_offset < 0
    ):
        raise ValueError("state_qpos_offset must be a non-negative integer")

    indices: list[int] = []
    lower: list[float] = []
    upper: list[float] = []

    for name in names:
        joint_obj = None
        if hasattr(model, "get_joint_qpos_addr"):
            raw_address = model.get_joint_qpos_addr(name)
        elif hasattr(model, "joint"):
            joint_obj = model.joint(name)
            raw_address = joint_obj.qposadr
        else:
            raise ValueError("simulator model has no named joint qpos lookup")
        qpos_index = _scalar_qpos_address(raw_address, name)

        if joint_obj is not None and hasattr(joint_obj, "id"):
            joint_id = int(joint_obj.id)
        elif hasattr(model, "joint_name2id"):
            joint_id = int(model.joint_name2id(name))
        else:
            raise ValueError("simulator model has no named joint id lookup")

        if not hasattr(model, "jnt_limited") or not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"joint {name} must have a finite position limit")
        if not hasattr(model, "jnt_range"):
            raise ValueError("simulator model has no joint range table")
        lo, hi = (float(x) for x in model.jnt_range[joint_id])
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"invalid position limits for joint {name}: {(lo, hi)}")

        indices.append(qpos_index + state_qpos_offset)
        lower.append(lo)
        upper.append(hi)

    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("resolved joint qpos addresses must be unique and non-negative")

    return NamedJointLayout(
        joint_names=names,
        qpos_indices=tuple(indices),
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
    )


def apply_robot_init_perturbation(
    init_state: np.ndarray,
    *,
    layout: NamedJointLayout,
    strength: float,
    seed: int,
    sampling_mode: str = "gaussian_std",
) -> PerturbationResult:
    """Add deterministic noise only to validated arm joint qpos.

    ``gaussian_std`` treats strength as each joint's standard deviation.
    ``fixed_l2`` treats it as the total seven-joint L2 radius, matching the
    LIBERO-Plus Robot-init generator.
    """

    state = np.asarray(init_state)
    if state.ndim != 1 or not np.issubdtype(state.dtype, np.floating):
        raise ValueError("init_state must be a one-dimensional floating array")
    if not np.isfinite(state).all():
        raise ValueError("init_state must contain only finite values")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"unsupported Robot-init sampling mode: {sampling_mode}")

    indices = np.asarray(layout.qpos_indices, dtype=np.int64)
    if len(indices) == 0 or len(indices) != len(layout.joint_names):
        raise ValueError("joint layout is empty or inconsistent")
    if indices.min() < 0 or indices.max() >= state.size or len(set(indices)) != len(indices):
        raise ValueError("joint qpos address is outside init_state or duplicated")
    lower = np.asarray(layout.lower, dtype=np.float64)
    upper = np.asarray(layout.upper, dtype=np.float64)
    if lower.shape != indices.shape or upper.shape != indices.shape:
        raise ValueError("joint limit shape does not match joint addresses")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError("joint limits must be finite and ordered")

    rng = np.random.default_rng(seed)
    sampled_noise = rng.normal(0.0, 1.0, size=len(indices))
    if sampling_mode == "gaussian_std":
        sampled_noise *= strength
    else:
        direction_norm = float(np.linalg.norm(sampled_noise))
        if not np.isfinite(direction_norm) or direction_norm == 0.0:
            raise RuntimeError("failed to sample a finite nonzero Robot-init direction")
        sampled_noise *= strength / direction_norm
    original = state[indices].astype(np.float64, copy=True)
    perturbed = np.clip(original + sampled_noise, lower, upper)

    output = state.copy()
    output[indices] = perturbed.astype(state.dtype, copy=False)
    applied_noise = output[indices].astype(np.float64) - original
    return PerturbationResult(
        state=output,
        noise=applied_noise,
        applied=bool(np.any(applied_noise != 0.0)),
    )


def materialize_rollout_plan(
    init_state: np.ndarray,
    *,
    plan: Sequence[RolloutPlanEntry],
    layout: NamedJointLayout,
) -> tuple[tuple[np.ndarray, dict[str, Any]], ...]:
    """Materialize rollout states and truthful perturbation metadata."""

    base = np.asarray(init_state)
    materialized: list[tuple[np.ndarray, dict[str, Any]]] = []
    for entry in plan:
        if entry.perturb_type not in SUPPORTED_PERTURBATIONS:
            raise ValueError(f"unsupported perturbation type: {entry.perturb_type}")
        if entry.perturb_type == "none":
            if entry.is_perturbed or entry.strength != 0:
                raise ValueError("anchor entry has inconsistent perturbation metadata")
            state = base.copy()
            applied = False
            noise = np.zeros(len(layout.qpos_indices), dtype=np.float64)
        else:
            result = apply_robot_init_perturbation(
                base,
                layout=layout,
                strength=entry.strength,
                seed=entry.seed,
                sampling_mode=entry.sampling_mode,
            )
            if not entry.is_perturbed or not result.applied:
                raise ValueError("Robot-init entry did not apply a real perturbation")
            state, applied, noise = result.state, result.applied, result.noise

        metadata = {
            "rollout_index": entry.rollout_index,
            "pair_id": entry.pair_id,
            "perturb_type": entry.perturb_type,
            "perturb_strength": entry.strength,
            "perturb_seed": entry.seed,
            "perturb_sampling_mode": entry.sampling_mode,
            "is_perturbed": applied,
            "perturb_applied": applied,
            "joint_noise": noise.copy(),
            "joint_noise_l2": float(np.linalg.norm(noise)),
        }
        materialized.append((state, metadata))
    return tuple(materialized)


def compute_recovery_rewards(
    episodes: Sequence[Mapping[str, Any]],
    base_scores: Sequence[float],
    *,
    lambda_recovery: float,
    valid_mask: Sequence[bool] | None = None,
) -> RecoveryRewardResult:
    """Add recovery bonus only for successful, actually applied Robot-init."""

    count = len(episodes)
    base = np.asarray(base_scores, dtype=np.float64)
    if base.shape != (count,) or not np.isfinite(base).all():
        raise ValueError("base_scores must be one finite value per episode")
    if not np.isfinite(lambda_recovery) or lambda_recovery < 0:
        raise ValueError("lambda_recovery must be finite and non-negative")
    valid = np.ones(count, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (count,):
        raise ValueError("valid_mask must contain one value per episode")

    recovery = np.zeros(count, dtype=np.float64)
    perturbed = np.zeros(count, dtype=bool)
    successes = np.zeros(count, dtype=bool)
    for index, episode in enumerate(episodes):
        perturb_type = str(episode.get("perturb_type", "none"))
        if perturb_type not in SUPPORTED_PERTURBATIONS:
            raise ValueError(f"unsupported perturbation type: {perturb_type}")
        labelled = bool(episode.get("is_perturbed", False))
        applied = bool(episode.get("perturb_applied", False))
        if labelled != applied or (perturb_type == "none" and applied):
            raise ValueError(f"episode {index} has inconsistent perturbation metadata")
        if perturb_type == "robot_init" and not applied:
            raise ValueError(f"episode {index} labels Robot-init without applying it")

        successes[index] = bool(episode.get("success", False))
        perturbed[index] = applied
        recovery[index] = float(applied and successes[index])

    total = base + float(lambda_recovery) * recovery
    anchor_valid = valid & ~perturbed
    perturbed_valid = valid & perturbed

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else 0.0

    stats = {
        "mean_R_success": masked_mean(base, valid),
        "mean_R_recovery": masked_mean(recovery, valid),
        "anchor_success_rate": masked_mean(successes, anchor_valid),
        "perturbed_success_rate": masked_mean(successes, perturbed_valid),
        "valid_anchor_count": float(anchor_valid.sum()),
        "valid_perturbed_count": float(perturbed_valid.sum()),
        "lambda_r_effective": float(lambda_recovery),
    }
    return RecoveryRewardResult(total=total, recovery=recovery, stats=stats)
