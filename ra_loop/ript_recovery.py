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
    compute_recovery_rewards,
    materialize_rollout_plan,
    resolve_named_joint_layout,
)


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
    """Run upstream PPO once and correct augmented-reward metric semantics."""

    def __init__(
        self,
        *args: Any,
        enable_rollout_stats_tracking: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.reward_function, RobotInitRecoveryReward):
            raise TypeError(
                "RobotInitRecoveryOptimizer requires RobotInitRecoveryReward"
            )
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

        generator = self.rollout_generator
        had_instance_override = "generate_rollouts" in vars(generator)
        previous_instance_value = vars(generator).get("generate_rollouts")
        original_generate = generator.generate_rollouts
        captured: list[tuple[Any, Any, Any, Any]] = []

        def capture_once(*args, **kwargs):
            if captured:
                raise RuntimeError("upstream PPO requested rollouts more than once")
            result = original_generate(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("rollout generator returned an invalid result")
            captured.append(result)
            return result

        generator.generate_rollouts = capture_once
        try:
            metrics = super().optimize(
                model,
                batch,
                optimizers,
                data_iterator=data_iterator,
                dataloader=dataloader,
            )
        finally:
            if had_instance_override:
                generator.generate_rollouts = previous_instance_value
            else:
                delattr(generator, "generate_rollouts")

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

        return metrics
