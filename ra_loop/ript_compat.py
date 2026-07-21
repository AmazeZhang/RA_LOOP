"""Narrow compatibility shims for the pinned RIPT-VLA upstream commit."""

from __future__ import annotations

from typing import Any

from libero.libero.envs import DummyVectorEnv
from ript.algos.rl_optimizers.rl_optimizer_openvla_oft import (
    RLOptimizerOpenVLAOFT,
)
from ript.env_runner.openvla_oft_libero_runner import (
    OpenVLAOFTLiberoRunner,
    get_libero_env,
)


class RLOptimizerOpenVLAOFTCompat(RLOptimizerOpenVLAOFT):
    """Route a train-entry argument that upstream passes to the wrong factory.

    ``train_ript_openvla_oft.py`` at commit 440990e passes
    ``enable_rollout_stats_tracking`` while instantiating the RL optimizer,
    although the option belongs to ``RolloutGenerator``. Accept it here and
    apply it to the supplied generator without changing upstream source.
    """

    def __init__(
        self,
        *args: Any,
        enable_rollout_stats_tracking: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_generator.enable_rollout_stats_tracking = bool(
            enable_rollout_stats_tracking
        )


class InProcessOpenVLAOFTLiberoRunner(OpenVLAOFTLiberoRunner):
    """Use LIBERO's sequential vector wrapper for a one-environment smoke.

    A spawned environment worker inherits torchrun rendezvous variables and
    reinitializes distributed state while importing OpenVLA. Keeping this
    bounded smoke in-process avoids the duplicate TCPStore without changing
    upstream runner code or global distributed settings.
    """

    def create_env(self, env_name: str):
        if self.num_parallel_envs != 1:
            raise ValueError(
                "InProcessOpenVLAOFTLiberoRunner requires num_parallel_envs=1"
            )
        task_id = self.env_names.index(env_name)
        task = self.benchmark.get_task(task_id)

        def env_factory():
            env, _ = get_libero_env(task, "openvla", resolution=256)
            return env

        env = DummyVectorEnv([env_factory])
        return env, task_id, 1
