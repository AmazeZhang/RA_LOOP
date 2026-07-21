import multiprocessing

import ra_loop.ript_compat as compat
from ra_loop.ript_compat import (
    InProcessOpenVLAOFTLiberoRunner,
    RLOptimizerOpenVLAOFTCompat,
)
from libero.libero.envs.venv import DummyEnvWorker, DummyVectorEnv
from hydra.utils import instantiate
from omegaconf import OmegaConf


class DummyRolloutGenerator:
    enable_rollout_stats_tracking = False


class DummyRewardFunction:
    pass


class DummyBenchmark:
    def get_task(self, task_id):
        return f"task-{task_id}"


class DummyEnvironment:
    def close(self):
        pass


def test_misrouted_rollout_stats_flag_is_applied_to_generator() -> None:
    generator = DummyRolloutGenerator()
    optimizer = RLOptimizerOpenVLAOFTCompat(
        rollout_generator=generator,
        reward_function=DummyRewardFunction(),
        enable_rollout_stats_tracking=True,
    )

    assert optimizer.rollout_generator is generator
    assert generator.enable_rollout_stats_tracking is True


def test_hydra_partial_matches_upstream_two_stage_instantiation() -> None:
    config = OmegaConf.create(
        {
            "_target_": "ra_loop.ript_compat.RLOptimizerOpenVLAOFTCompat",
            "_partial_": True,
            "gradient_accumulation_steps": 1,
        }
    )
    factory = instantiate(config, enable_rollout_stats_tracking=True)
    generator = DummyRolloutGenerator()
    optimizer = factory(
        rollout_generator=generator,
        reward_function=DummyRewardFunction(),
    )

    assert optimizer.rollout_generator is generator
    assert generator.enable_rollout_stats_tracking is True


def test_single_env_runner_uses_no_child_process(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "get_libero_env",
        lambda task, model_family, resolution: (DummyEnvironment(), "description"),
    )
    runner = InProcessOpenVLAOFTLiberoRunner.__new__(
        InProcessOpenVLAOFTLiberoRunner
    )
    runner.num_parallel_envs = 1
    runner.env_names = ["selected-task"]
    runner.benchmark = DummyBenchmark()
    children_before = {child.pid for child in multiprocessing.active_children()}

    env, task_id, env_num = runner.create_env("selected-task")

    assert isinstance(env, DummyVectorEnv)
    assert env.worker_class is DummyEnvWorker
    assert task_id == 0
    assert env_num == 1
    assert {child.pid for child in multiprocessing.active_children()} == children_before
    env.close()
