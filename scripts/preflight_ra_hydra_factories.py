#!/usr/bin/env python3
"""Compose and instantiate bounded RA-LOOP factories without env/model creation."""

from __future__ import annotations

import json
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ra_loop.ript_compat import InProcessOpenVLAOFTLiberoRunner
from ra_loop.ript_recovery import (
    RobotInitRecoveryOptimizer,
    RobotInitRecoveryReward,
    RobotInitRecoveryRolloutGenerator,
)
from ript.algos.rl_optimizers.file_counter import cleanup_counter


RIPT_CONFIG_DIR = "/home/imc/code/ript-vla/config"
CONFIG_NAME = "train_rl_openvla_oft_all_task_spatial.yaml"


class NoEnvironmentRunner:
    """Constructor-only stand-in; no simulator method may be called."""

    def __init__(self, rollouts_per_env: int) -> None:
        self.num_parallel_envs = 1
        self.rollouts_per_env = rollouts_per_env

    def __getattr__(self, name):
        raise AssertionError(f"preflight unexpectedly accessed env runner attribute {name}")


def main() -> None:
    with initialize_config_dir(config_dir=RIPT_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=sys.argv[1:])
    OmegaConf.resolve(cfg)

    expected_targets = {
        "env_runner": "ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner",
        "rollout": "ra_loop.ript_recovery.RobotInitRecoveryRolloutGenerator",
        "reward": "ra_loop.ript_recovery.RobotInitRecoveryReward",
        "optimizer": "ra_loop.ript_recovery.RobotInitRecoveryOptimizer",
    }
    actual_targets = {
        "env_runner": cfg.algo.env_runner._target_,
        "rollout": cfg.algo.rollout_generator_factory._target_,
        "reward": cfg.reward_function._target_,
        "optimizer": cfg.algo.rl_optimizer_factory._target_,
    }
    if actual_targets != expected_targets:
        raise AssertionError(f"unexpected Hydra targets: {actual_targets}")
    if cfg.algo.rloo_batch_size < 2 or cfg.algo.rloo_batch_size % 2 != 0:
        raise AssertionError("bounded recovery preflight requires an even K >= 2")
    if cfg.algo.rollouts_per_env != cfg.algo.rloo_batch_size:
        raise AssertionError("bounded recovery preflight requires rollouts_per_env == K")
    if cfg.algo.num_parallel_envs != 1:
        raise AssertionError("bounded recovery preflight requires one environment")
    if cfg.rollout.enabled or cfg.training.n_steps < 1:
        raise AssertionError("periodic eval must be disabled and n_steps must be positive")
    if cfg.logging.mode != "disabled":
        raise AssertionError("external logging must be disabled")
    if len(cfg.task.task_names_to_use) > 1 and not cfg.train_dataloader.shuffle:
        raise AssertionError("multi-task training requires shuffled batches")

    reward = instantiate(cfg.reward_function)
    rollout_factory = instantiate(cfg.algo.rollout_generator_factory)
    runner = NoEnvironmentRunner(int(cfg.algo.rollouts_per_env))
    rollout = rollout_factory(
        env_runner=runner,
        task_names_to_use=list(cfg.task.task_names_to_use),
        demo_batch_size=1,
        create_env=False,
    )
    try:
        optimizer_factory = instantiate(
            cfg.algo.rl_optimizer_factory,
            enable_rollout_stats_tracking=True,
        )
        optimizer = optimizer_factory(
            rollout_generator=rollout,
            reward_function=reward,
        )

        if not isinstance(reward, RobotInitRecoveryReward):
            raise AssertionError(f"wrong reward type: {type(reward)}")
        if not isinstance(rollout, RobotInitRecoveryRolloutGenerator):
            raise AssertionError(f"wrong rollout type: {type(rollout)}")
        if not isinstance(optimizer, RobotInitRecoveryOptimizer):
            raise AssertionError(f"wrong optimizer type: {type(optimizer)}")
        if cfg.algo.env_runner._target_ != (
            f"{InProcessOpenVLAOFTLiberoRunner.__module__}."
            f"{InProcessOpenVLAOFTLiberoRunner.__name__}"
        ):
            raise AssertionError("wrong in-process environment runner target")
        if not rollout.enable_rollout_stats_tracking:
            raise AssertionError("misrouted rollout stats flag was not forwarded")
        if optimizer.advantage_mode not in {
            "counterfactual_constrained",
            "mode_stratified",
        }:
            raise AssertionError("custom RA-LOOP advantage was not configured")
        if optimizer.advantage_mode == "counterfactual_constrained":
            if reward.lambda_recovery != 0.0:
                raise AssertionError(
                    "counterfactual mode requires zero legacy recovery bonus"
                )
            if optimizer.nominal_calibration_batches < 1:
                raise AssertionError("nominal calibration must be enabled")

        print(
            json.dumps(
                {
                    "config_name": CONFIG_NAME,
                    "targets": actual_targets,
                    "task_names": list(cfg.task.task_names_to_use),
                    "train_dataloader_shuffle": bool(cfg.train_dataloader.shuffle),
                    "n_steps": int(cfg.training.n_steps),
                    "rloo_k": int(cfg.algo.rloo_batch_size),
                    "rollouts_per_env": int(cfg.algo.rollouts_per_env),
                    "num_parallel_envs": int(cfg.algo.num_parallel_envs),
                    "max_episode_length": int(cfg.algo.max_episode_length),
                    "robot_init_strength": float(rollout.robot_init_strength),
                    "robot_init_sampling_mode": str(rollout.robot_init_sampling_mode),
                    "perturb_seed": int(rollout.perturb_seed),
                    "lambda_recovery": float(reward.lambda_recovery),
                    "advantage_mode": str(optimizer.advantage_mode),
                    "nominal_allowed_drop": float(
                        optimizer.nominal_allowed_drop
                    ),
                    "nominal_ema_decay": float(optimizer.nominal_ema_decay),
                    "nominal_dual_learning_rate": float(
                        optimizer.nominal_dual_learning_rate
                    ),
                    "nominal_initial_multiplier": float(
                        optimizer.nominal_initial_multiplier
                    ),
                    "nominal_max_multiplier": float(
                        optimizer.nominal_max_multiplier
                    ),
                    "nominal_calibration_batches": int(
                        optimizer.nominal_calibration_batches
                    ),
                    "rollout_stats_tracking": bool(
                        rollout.enable_rollout_stats_tracking
                    ),
                    "create_env": False,
                    "periodic_eval": bool(cfg.rollout.enabled),
                    "wandb_mode": str(cfg.logging.mode),
                    "factory_instantiation": "passed",
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        cleanup_counter(rollout.counter_filename)


if __name__ == "__main__":
    main()
