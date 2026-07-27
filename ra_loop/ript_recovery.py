"""Thin RIPT adapters for recovery-only Robot-init rollouts."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ript.algos.reward_functions.libero import BaseRewardFunction
from ript.algos.rl_optimizers.rl_optimizer_openvla_oft import (
    RLOptimizerOpenVLAOFT,
)
from ript.algos.rl_optimizers.rollout_generator import RolloutGenerator

from ra_loop.robustness import (
    NamedJointLayout,
    build_robot_init_rollout_plan,
    compute_counterfactual_recovery_advantages,
    compute_mode_stratified_rloo_advantages,
    compute_recovery_rewards,
    compute_soft_counterfactual_recovery_advantages,
    materialize_rollout_plan,
    resolve_named_joint_layout,
    update_nominal_performance_constraint,
)

COUNTERFACTUAL_ADVANTAGE_MODES = {
    "counterfactual_constrained",
    "counterfactual_soft_constrained",
}


class _OptimizerStepGate:
    """Delegate optimizer APIs while conditionally suppressing parameter steps."""

    def __init__(self, optimizer: Any, should_step: Callable[[], bool]) -> None:
        self._optimizer = optimizer
        self._should_step = should_step

    def step(self, *args: Any, **kwargs: Any):
        if self._should_step():
            return self._optimizer.step(*args, **kwargs)
        return None

    def zero_grad(self, *args: Any, **kwargs: Any):
        return self._optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._optimizer, name)


def resolve_in_process_joint_layout(created_env: Any) -> NamedJointLayout:
    """Resolve joints from the one live environment used by the safe runner.

    Subprocess workers are deliberately rejected: their MuJoCo model is not
    directly inspectable, so silently guessing addresses would violate the
    Robot-init safety contract.
    """

    if not isinstance(created_env, tuple) or len(created_env) != 3:
        raise ValueError("created_env must be the RIPT (env, task_id, env_num) tuple")
    vector_env, _, env_num = created_env
    if env_num != 1:
        raise ValueError("Robot-init adapter requires exactly one in-process environment")
    workers = getattr(vector_env, "workers", None)
    if workers is None or len(workers) != 1 or not hasattr(workers[0], "env"):
        raise ValueError("Robot-init adapter requires a directly inspectable worker")
    live_env = workers[0].env
    sim = getattr(live_env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        raise ValueError("live environment does not expose sim.model")
    return resolve_named_joint_layout(model, state_qpos_offset=1)


class RobotInitRecoveryRolloutGenerator(RolloutGenerator):
    """Inject anchor/Robot-init pairs through the upstream rollout hook.

    The parent `generate_rollouts` is called exactly once. During that call only,
    the environment runner method is wrapped to replace the parent's K identical
    states and tag yielded episodes. The original method is restored in `finally`
    even if rollout generation fails.
    """

    def __init__(
        self,
        *args: Any,
        robot_init_strength: float,
        perturb_seed: int,
        robot_init_sampling_mode: str = "gaussian_std",
        joint_layout_resolver: Callable[[Any], NamedJointLayout] = (
            resolve_in_process_joint_layout
        ),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.rloo_batch_size < 2 or self.rloo_batch_size % 2 != 0:
            raise ValueError("Robot-init recovery requires an even RLOO K >= 2")
        if not np.isfinite(robot_init_strength) or robot_init_strength <= 0:
            raise ValueError("robot_init_strength must be finite and positive")
        if isinstance(perturb_seed, bool) or not isinstance(perturb_seed, int) or perturb_seed < 0:
            raise ValueError("perturb_seed must be a non-negative integer")
        if robot_init_sampling_mode not in {"gaussian_std", "fixed_l2"}:
            raise ValueError("unsupported robot_init_sampling_mode")
        if not callable(joint_layout_resolver):
            raise ValueError("joint_layout_resolver must be callable")
        if self.use_val_init and self.mix_val_init_in_rloo:
            raise ValueError("paired Robot-init does not support mixed validation initializations")

        self.robot_init_strength = float(robot_init_strength)
        self.perturb_seed = perturb_seed
        self.robot_init_sampling_mode = robot_init_sampling_mode
        self.joint_layout_resolver = joint_layout_resolver
        self._recovery_step = 0
        self._adapter_active = False

    def _validate_runner_contract(self) -> None:
        if getattr(self.env_runner, "num_parallel_envs", None) != 1:
            raise ValueError("Robot-init adapter requires num_parallel_envs=1")
        if getattr(self.env_runner, "rollouts_per_env", None) != self.rloo_batch_size:
            raise ValueError("env runner rollouts_per_env must equal RLOO K")

    def generate_rollouts(self, model, batch, data_iterator, dataloader):
        if self._adapter_active:
            raise RuntimeError("Robot-init rollout adapter is not re-entrant")
        self._validate_runner_contract()

        runner = self.env_runner
        had_instance_override = "run_policy_in_env" in vars(runner)
        previous_instance_value = vars(runner).get("run_policy_in_env")
        original_run_policy = runner.run_policy_in_env
        call_index = 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        def wrapped_run_policy(
            env_name,
            policy,
            all_init_states=None,
            render=False,
            created_env=None,
            random_init=False,
        ):
            nonlocal call_index
            if random_init:
                raise ValueError("Robot-init recovery requires an explicit base state")
            if created_env is None:
                raise ValueError("Robot-init recovery requires a pre-created environment")

            states = np.asarray(all_init_states)
            expected_shape = (self.rloo_batch_size,)
            if states.ndim != 2 or states.shape[0] != expected_shape[0]:
                raise ValueError(
                    f"expected {self.rloo_batch_size} explicit initialization states"
                )
            if not np.issubdtype(states.dtype, np.floating) or not np.isfinite(states).all():
                raise ValueError("initialization states must be finite floating arrays")
            if not np.all(states == states[0]):
                raise ValueError("paired Robot-init requires K identical base states")

            layout = self.joint_layout_resolver(created_env)
            call_seed = (
                self.perturb_seed
                + self._recovery_step * 1_000_003
                + rank * 10_007
                + call_index * (self.rloo_batch_size // 2)
            )
            call_index += 1
            plan = build_robot_init_rollout_plan(
                num_pairs=self.rloo_batch_size // 2,
                strength=self.robot_init_strength,
                base_seed=call_seed,
                sampling_mode=self.robot_init_sampling_mode,
            )
            materialized = materialize_rollout_plan(
                states[0], plan=plan, layout=layout
            )
            injected_states = np.stack([item[0] for item in materialized])
            metadata = [item[1] for item in materialized]

            underlying = original_run_policy(
                env_name,
                policy,
                injected_states,
                render=render,
                created_env=created_env,
                random_init=False,
            )

            def tagged_results() -> Generator[tuple[Any, Any, dict[str, Any]], None, None]:
                collected = []
                for index in range(self.rloo_batch_size):
                    try:
                        collected.append(next(underlying))
                    except StopIteration as error:
                        raise RuntimeError(
                            f"environment yielded {index} episodes, expected "
                            f"{self.rloo_batch_size}"
                        ) from error
                try:
                    next(underlying)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError("environment yielded more than RLOO K episodes")

                for index, (success, total_reward, episode) in enumerate(collected):
                    if not isinstance(episode, dict):
                        raise TypeError("rollout episode must be a dictionary")
                    tagged_episode = dict(episode)
                    for key, value in metadata[index].items():
                        if key in tagged_episode:
                            raise ValueError(f"rollout episode already contains reserved key {key}")
                        tagged_episode[key] = (
                            value.tolist() if isinstance(value, np.ndarray) else value
                        )
                    yield success, total_reward, tagged_episode

            return tagged_results()

        self._adapter_active = True
        runner.run_policy_in_env = wrapped_run_policy
        try:
            result = super().generate_rollouts(
                model, batch, data_iterator, dataloader
            )
        finally:
            if had_instance_override:
                runner.run_policy_in_env = previous_instance_value
            else:
                delattr(runner, "run_policy_in_env")
            self._adapter_active = False

        self._recovery_step += 1
        return result


class RobotInitRecoveryReward(BaseRewardFunction):
    """Per-episode success plus truthful Robot-init recovery bonus."""

    def __init__(self, lambda_recovery: float) -> None:
        super().__init__()
        if not np.isfinite(lambda_recovery) or lambda_recovery < 0:
            raise ValueError("lambda_recovery must be finite and non-negative")
        self.lambda_recovery = float(lambda_recovery)

    def compute_reward(
        self,
        rollout_idx: int,
        rollout_episode: dict[str, Any],
        ground_truth_batch: dict[str, Any],
    ) -> float:
        del rollout_idx, ground_truth_batch
        base_success = float(bool(rollout_episode.get("success", False)))
        result = compute_recovery_rewards(
            [rollout_episode],
            [base_success],
            lambda_recovery=self.lambda_recovery,
        )
        return float(result.total[0])


class RobotInitRecoveryOptimizer(RLOptimizerOpenVLAOFT):
    """Run upstream PPO once with validated RA-LOOP advantages."""

    def __init__(
        self,
        *args: Any,
        enable_rollout_stats_tracking: bool = False,
        advantage_mode: str = "mode_stratified",
        nominal_allowed_drop: float = 0.02,
        nominal_ema_decay: float = 0.9,
        nominal_dual_learning_rate: float = 0.1,
        nominal_initial_multiplier: float = 1.0,
        nominal_max_multiplier: float = 10.0,
        nominal_calibration_batches: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.reward_function, RobotInitRecoveryReward):
            raise TypeError(
                "RobotInitRecoveryOptimizer requires RobotInitRecoveryReward"
            )
        if advantage_mode not in {
            "counterfactual_constrained",
            "counterfactual_soft_constrained",
            "mode_stratified",
            "upstream",
        }:
            raise ValueError("unsupported recovery advantage_mode")
        if (
            advantage_mode
            in COUNTERFACTUAL_ADVANTAGE_MODES | {"mode_stratified"}
            and self.rloo_over_all_rollouts
        ):
            raise ValueError(
                "custom RA-LOOP advantages require per-rollout-group upstream RLOO"
            )
        if (
            advantage_mode in COUNTERFACTUAL_ADVANTAGE_MODES
            and self.reward_function.lambda_recovery != 0.0
        ):
            raise ValueError(
                "counterfactual-constrained mode requires lambda_recovery=0"
            )
        if (
            isinstance(nominal_calibration_batches, bool)
            or not isinstance(nominal_calibration_batches, int)
            or nominal_calibration_batches < 1
        ):
            raise ValueError("nominal_calibration_batches must be a positive integer")

        # Validate all scalar constraint settings through the CPU primitive.
        update_nominal_performance_constraint(
            observed_anchor_success_rate=1.0,
            reference_anchor_success_rate=1.0,
            allowed_drop=nominal_allowed_drop,
            previous_multiplier=nominal_initial_multiplier,
            previous_anchor_success_ema=1.0,
            ema_decay=nominal_ema_decay,
            dual_learning_rate=nominal_dual_learning_rate,
            max_multiplier=nominal_max_multiplier,
        )
        self.advantage_mode = advantage_mode
        self.nominal_allowed_drop = float(nominal_allowed_drop)
        self.nominal_ema_decay = float(nominal_ema_decay)
        self.nominal_dual_learning_rate = float(nominal_dual_learning_rate)
        self.nominal_initial_multiplier = float(nominal_initial_multiplier)
        self.nominal_max_multiplier = float(nominal_max_multiplier)
        self.nominal_calibration_batches = nominal_calibration_batches
        self._nominal_calibration_success_sums: dict[int, float] = {}
        self._nominal_calibration_batch_counts: dict[int, int] = {}
        self._nominal_reference_by_task: dict[int, float] = {}
        self._nominal_multiplier_by_task: dict[int, float] = {}
        self._nominal_ema_by_task: dict[int, float] = {}
        self.rollout_generator.enable_rollout_stats_tracking = bool(
            enable_rollout_stats_tracking
        )

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def optimize(
        self,
        model,
        batch,
        optimizers,
        data_iterator=None,
        dataloader=None,
    ):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError(
                "recovery metric correction is currently gated to one GPU"
            )
        custom_advantage = self.advantage_mode in {
            "counterfactual_constrained",
            "counterfactual_soft_constrained",
            "mode_stratified",
        }
        if custom_advantage and getattr(self, "rloo_over_all_rollouts", False):
            raise ValueError(
                "custom RA-LOOP advantages require per-rollout-group upstream RLOO"
            )

        generator = self.rollout_generator
        had_instance_override = "generate_rollouts" in vars(generator)
        previous_instance_value = vars(generator).get("generate_rollouts")
        original_generate = generator.generate_rollouts
        captured: list[tuple[Any, Any, Any, Any]] = []
        stratified_advantages: np.ndarray | None = None
        surrogate_rewards: np.ndarray | None = None
        counterfactual_result = None
        pending_nominal_state = None
        constraint_updates: dict[int, Any] = {}
        calibrating_task_ids: set[int] = set()
        allow_parameter_update = (
            self.advantage_mode not in COUNTERFACTUAL_ADVANTAGE_MODES
        )

        def capture_once(*args, **kwargs):
            nonlocal allow_parameter_update
            nonlocal calibrating_task_ids
            nonlocal constraint_updates
            nonlocal counterfactual_result
            nonlocal pending_nominal_state
            nonlocal stratified_advantages
            nonlocal surrogate_rewards
            if captured:
                raise RuntimeError("upstream PPO requested rollouts more than once")
            result = original_generate(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("rollout generator returned an invalid result")
            captured.append(result)
            if custom_advantage:
                episodes, task_ids, valid_mask, _ = result
                rloo_batch_size = int(generator.rloo_batch_size)
                if rloo_batch_size < 4 or rloo_batch_size % 2 != 0:
                    raise RuntimeError(
                        "custom RA-LOOP advantages require an even batch size >= 4"
                    )
                if len(episodes) == 0 or len(episodes) % rloo_batch_size != 0:
                    raise RuntimeError(
                        "rollout count must be a positive multiple of RLOO batch size"
                    )
                valid = self._as_numpy(valid_mask).astype(bool, copy=False)
                ids = self._as_numpy(task_ids).astype(np.int64, copy=False)
                if valid.shape != (len(episodes),) or ids.shape != (len(episodes),):
                    raise RuntimeError("rollout task_ids/valid_mask shape mismatch")
                successes = np.asarray(
                    [bool(episode.get("success", False)) for episode in episodes],
                    dtype=bool,
                )
                base_scores = successes.astype(np.float64)
                perturbed = np.asarray(
                    [bool(episode.get("is_perturbed", False)) for episode in episodes],
                    dtype=bool,
                )
                rollout_group_ids = (
                    np.arange(len(episodes), dtype=np.int64) // rloo_batch_size
                )

                if self.advantage_mode == "mode_stratified":
                    reward_result = compute_recovery_rewards(
                        episodes,
                        base_scores,
                        lambda_recovery=self.reward_function.lambda_recovery,
                        valid_mask=valid,
                    )
                    stratified_advantages = (
                        compute_mode_stratified_rloo_advantages(
                            reward_result.total,
                            perturbed,
                            rollout_group_ids=rollout_group_ids,
                            valid_mask=valid,
                        )
                    )
                else:
                    pair_ids = []
                    for episode in episodes:
                        pair_id = episode.get("pair_id")
                        if (
                            isinstance(pair_id, bool)
                            or not isinstance(pair_id, (int, np.integer))
                            or pair_id < 0
                        ):
                            raise RuntimeError(
                                "counterfactual mode requires non-negative pair_id"
                            )
                        pair_ids.append(int(pair_id))

                    anchor_advantages = compute_mode_stratified_rloo_advantages(
                        base_scores,
                        perturbed,
                        rollout_group_ids=rollout_group_ids,
                        valid_mask=valid,
                    )
                    if self.advantage_mode == "counterfactual_soft_constrained":
                        counterfactual_result = (
                            compute_soft_counterfactual_recovery_advantages(
                                successes,
                                perturbed,
                                pair_ids,
                                rollout_group_ids=rollout_group_ids,
                                valid_mask=valid,
                            )
                        )
                    else:
                        counterfactual_result = (
                            compute_counterfactual_recovery_advantages(
                                successes,
                                perturbed,
                                pair_ids,
                                rollout_group_ids=rollout_group_ids,
                                valid_mask=valid,
                            )
                        )
                    stratified_advantages = (
                        counterfactual_result.advantages.copy()
                    )

                    calibration_sums = dict(
                        self._nominal_calibration_success_sums
                    )
                    calibration_counts = dict(
                        self._nominal_calibration_batch_counts
                    )
                    references = dict(self._nominal_reference_by_task)
                    multipliers = dict(self._nominal_multiplier_by_task)
                    anchor_emas = dict(self._nominal_ema_by_task)
                    expected_task_count = len(generator.task_names_to_use)
                    if expected_task_count < 1:
                        raise RuntimeError(
                            "counterfactual mode requires at least one task"
                        )
                    global_calibration_ready = (
                        len(references) == expected_task_count
                    )

                    for group_id in np.unique(rollout_group_ids):
                        group = rollout_group_ids == group_id
                        group_task_ids = np.unique(ids[group])
                        if group_task_ids.size != 1:
                            raise RuntimeError(
                                "one rollout group must contain exactly one task id"
                            )
                        task_id = int(group_task_ids[0])
                        anchor_valid = group & valid & ~perturbed
                        if not anchor_valid.any():
                            raise RuntimeError(
                                "counterfactual group contains no valid anchors"
                            )
                        observed_anchor_rate = float(
                            successes[anchor_valid].mean()
                        )

                        if task_id not in references:
                            calibrating_task_ids.add(task_id)
                            calibration_sums[task_id] = (
                                calibration_sums.get(task_id, 0.0)
                                + observed_anchor_rate
                            )
                            calibration_counts[task_id] = (
                                calibration_counts.get(task_id, 0) + 1
                            )
                            stratified_advantages[group] = 0.0
                            if (
                                calibration_counts[task_id]
                                == self.nominal_calibration_batches
                            ):
                                reference = (
                                    calibration_sums[task_id]
                                    / calibration_counts[task_id]
                                )
                                references[task_id] = reference
                                multipliers[task_id] = (
                                    self.nominal_initial_multiplier
                                )
                                anchor_emas[task_id] = reference
                            continue

                        if not global_calibration_ready:
                            stratified_advantages[group] = 0.0
                            continue

                        update = update_nominal_performance_constraint(
                            observed_anchor_success_rate=observed_anchor_rate,
                            reference_anchor_success_rate=references[task_id],
                            allowed_drop=self.nominal_allowed_drop,
                            previous_multiplier=multipliers[task_id],
                            previous_anchor_success_ema=anchor_emas[task_id],
                            ema_decay=self.nominal_ema_decay,
                            dual_learning_rate=self.nominal_dual_learning_rate,
                            max_multiplier=self.nominal_max_multiplier,
                        )
                        multipliers[task_id] = update.multiplier
                        anchor_emas[task_id] = update.anchor_success_ema
                        constraint_updates[task_id] = update
                        stratified_advantages[anchor_valid] = (
                            update.multiplier
                            * anchor_advantages[anchor_valid]
                        )

                    if (
                        not global_calibration_ready
                        or calibrating_task_ids
                        or len(references) < expected_task_count
                    ):
                        # All task references must describe the same untouched
                        # warm-start policy. Do not let an early-calibrated task
                        # update parameters while another task is calibrating.
                        stratified_advantages[:] = 0.0

                    pending_nominal_state = (
                        calibration_sums,
                        calibration_counts,
                        references,
                        multipliers,
                        anchor_emas,
                    )
                    allow_parameter_update = bool(
                        np.any(stratified_advantages != 0.0)
                    )

                # Upstream maps rewards q to K/(K-1) * (q - mean(q)). Each
                # custom group sums to zero, as do its invalid zero pads, so
                # this inverse makes upstream recover the exact target.
                surrogate_rewards = (
                    (rloo_batch_size - 1)
                    / rloo_batch_size
                    * stratified_advantages
                )
            return result

        reward_function = self.reward_function
        had_reward_override = "compute_reward" in vars(reward_function)
        previous_reward_value = vars(reward_function).get("compute_reward")

        def reward_for_upstream(rollout_idx, rollout_episode, ground_truth_batch):
            if surrogate_rewards is None or not captured:
                raise RuntimeError("upstream requested reward before generating rollouts")
            if (
                isinstance(rollout_idx, bool)
                or not isinstance(rollout_idx, int)
                or rollout_idx < 0
                or rollout_idx >= len(surrogate_rewards)
            ):
                raise RuntimeError("upstream requested an invalid rollout reward index")
            if rollout_episode is not captured[0][0][rollout_idx]:
                raise RuntimeError("upstream reward episode order changed")
            return float(surrogate_rewards[rollout_idx])

        generator.generate_rollouts = capture_once
        if custom_advantage:
            reward_function.compute_reward = reward_for_upstream
        parent_optimizers = optimizers
        if self.advantage_mode in COUNTERFACTUAL_ADVANTAGE_MODES:
            parent_optimizers = [
                _OptimizerStepGate(
                    optimizer,
                    lambda: allow_parameter_update,
                )
                for optimizer in optimizers
            ]
        try:
            metrics = super().optimize(
                model,
                batch,
                parent_optimizers,
                data_iterator=data_iterator,
                dataloader=dataloader,
            )
        finally:
            if had_instance_override:
                generator.generate_rollouts = previous_instance_value
            else:
                delattr(generator, "generate_rollouts")
            if custom_advantage:
                if had_reward_override:
                    reward_function.compute_reward = previous_reward_value
                else:
                    delattr(reward_function, "compute_reward")

        if pending_nominal_state is not None:
            (
                self._nominal_calibration_success_sums,
                self._nominal_calibration_batch_counts,
                self._nominal_reference_by_task,
                self._nominal_multiplier_by_task,
                self._nominal_ema_by_task,
            ) = pending_nominal_state

        if len(captured) != 1:
            raise RuntimeError("upstream PPO did not request exactly one rollout batch")
        episodes, task_ids, valid_mask, _ = captured[0]
        valid = self._as_numpy(valid_mask).astype(bool, copy=False)
        ids = self._as_numpy(task_ids).astype(np.int64, copy=False)
        if valid.shape != (len(episodes),) or ids.shape != (len(episodes),):
            raise RuntimeError("rollout task_ids/valid_mask shape mismatch")

        base_scores = [float(bool(episode.get("success", False))) for episode in episodes]
        reward_result = compute_recovery_rewards(
            episodes,
            base_scores,
            lambda_recovery=self.reward_function.lambda_recovery,
            valid_mask=valid,
        )
        valid_count = int(valid.sum())
        if valid_count == 0:
            raise RuntimeError("rollout batch contains no valid episodes")

        metrics.update(reward_result.stats)
        metrics["mean_R_total"] = float(reward_result.total[valid].mean())
        # Preserve upstream names while fixing their semantics after reward shaping.
        metrics["mean_scores"] = reward_result.stats["mean_R_success"]
        metrics["mean_rlhf_reward"] = metrics["mean_R_total"]
        metrics["advantage_mode_stratified"] = float(
            self.advantage_mode == "mode_stratified"
        )
        metrics["advantage_mode_counterfactual_constrained"] = float(
            self.advantage_mode == "counterfactual_constrained"
        )
        metrics["advantage_mode_counterfactual_soft_constrained"] = float(
            self.advantage_mode == "counterfactual_soft_constrained"
        )
        if self.advantage_mode in COUNTERFACTUAL_ADVANTAGE_MODES:
            metrics["parameter_update_applied"] = float(allow_parameter_update)
        if stratified_advantages is not None:
            perturbed = np.asarray(
                [bool(episode.get("is_perturbed", False)) for episode in episodes],
                dtype=bool,
            )
            anchor_valid = valid & ~perturbed
            perturbed_valid = valid & perturbed
            metrics["mean_anchor_advantage"] = float(
                stratified_advantages[anchor_valid].mean()
            )
            metrics["mean_perturbed_advantage"] = float(
                stratified_advantages[perturbed_valid].mean()
            )
        if counterfactual_result is not None:
            if self.advantage_mode == "counterfactual_soft_constrained":
                metrics["cra_soft_weighted_pairs"] = float(
                    counterfactual_result.weighted_pairs
                )
                metrics["cra_anchor_failure_pairs"] = float(
                    counterfactual_result.anchor_failure_pairs
                )
                metrics["cra_groups_with_nonzero_advantage"] = float(
                    counterfactual_result.groups_with_nonzero_advantage
                )
                metrics["cra_groups_all_anchor_failure"] = float(
                    counterfactual_result.groups_all_anchor_failure
                )
                metrics["cra_groups_uniform_perturbed_outcome"] = float(
                    counterfactual_result.groups_uniform_perturbed_outcome
                )
                metrics["cra_mean_anchor_competence"] = float(
                    counterfactual_result.mean_anchor_competence
                )
            else:
                metrics["cra_eligible_pairs"] = float(
                    counterfactual_result.eligible_pairs
                )
                metrics["cra_excluded_anchor_failure_pairs"] = float(
                    counterfactual_result.excluded_anchor_failure_pairs
                )
                metrics["cra_groups_with_update"] = float(
                    counterfactual_result.groups_with_update
                )
                metrics["cra_groups_without_baseline"] = float(
                    counterfactual_result.groups_without_baseline
                )
            metrics["cra_dropped_invalid_pairs"] = float(
                counterfactual_result.dropped_invalid_pairs
            )
            metrics["nominal_calibrating_tasks"] = float(
                len(calibrating_task_ids)
            )
            metrics["nominal_global_calibration_complete"] = float(
                len(self._nominal_reference_by_task)
                == len(generator.task_names_to_use)
            )

        successes = np.asarray(base_scores, dtype=np.float64)
        for task_id in np.unique(ids[valid]):
            task_mask = valid & (ids == task_id)
            try:
                task_name = generator.task_names_to_use[int(task_id)]
            except (IndexError, TypeError) as error:
                raise RuntimeError(f"invalid rollout task id {task_id}") from error
            metrics[f"rl_train_succeess_rate/{task_name}"] = float(
                successes[task_mask].mean()
            )
            if int(task_id) in self._nominal_reference_by_task:
                metrics[f"nominal_reference/{task_name}"] = (
                    self._nominal_reference_by_task[int(task_id)]
                )
                metrics[f"nominal_multiplier/{task_name}"] = (
                    self._nominal_multiplier_by_task[int(task_id)]
                )
                metrics[f"nominal_anchor_ema/{task_name}"] = (
                    self._nominal_ema_by_task[int(task_id)]
                )
            if int(task_id) in constraint_updates:
                metrics[f"nominal_violation/{task_name}"] = (
                    constraint_updates[int(task_id)].violation
                )
            metrics[f"nominal_calibration_batches/{task_name}"] = float(
                self._nominal_calibration_batch_counts.get(int(task_id), 0)
            )

        return metrics
