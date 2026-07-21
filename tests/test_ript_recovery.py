from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ra_loop.ript_recovery as adapter_module
from ra_loop.ript_recovery import (
    RobotInitRecoveryOptimizer,
    RobotInitRecoveryReward,
    RobotInitRecoveryRolloutGenerator,
)
from ra_loop.robustness import NamedJointLayout


def fake_layout(_created_env) -> NamedJointLayout:
    return NamedJointLayout(
        joint_names=tuple(f"joint{i}" for i in range(7)),
        qpos_indices=(1, 2, 3, 4, 5, 6, 7),
        lower=np.full(7, -1.0),
        upper=np.full(7, 1.0),
    )


class FakeRunner:
    num_parallel_envs = 1
    rollouts_per_env = 4

    def __init__(self):
        self.calls = 0
        self.received_states = None

    def run_policy_in_env(
        self,
        env_name,
        policy,
        all_init_states=None,
        render=False,
        created_env=None,
        random_init=False,
    ):
        self.calls += 1
        self.received_states = np.asarray(all_init_states).copy()
        for index in range(len(all_init_states)):
            yield bool(index % 2), float(index % 2), {"episode_index": index}


def make_adapter(runner: FakeRunner) -> RobotInitRecoveryRolloutGenerator:
    adapter = RobotInitRecoveryRolloutGenerator.__new__(
        RobotInitRecoveryRolloutGenerator
    )
    adapter.rloo_batch_size = 4
    adapter.robot_init_strength = 0.02
    adapter.robot_init_sampling_mode = "fixed_l2"
    adapter.perturb_seed = 11
    adapter.joint_layout_resolver = fake_layout
    adapter._recovery_step = 0
    adapter._adapter_active = False
    adapter.env_runner = runner
    return adapter


def install_fake_parent(monkeypatch, parent_calls, base_states, *, fail=False):
    def fake_parent(self, model, batch, data_iterator, dataloader):
        parent_calls.append(1)
        results = self.env_runner.run_policy_in_env(
            "task",
            model,
            base_states,
            created_env=(object(), 0, 1),
            random_init=False,
        )
        episodes = []
        for _ in range(self.rloo_batch_size):
            success, _, episode = next(results)
            episode["success"] = success
            episodes.append(episode)
        if fail:
            raise RuntimeError("parent failure")
        return episodes, "task_ids", "valid_mask", 1

    monkeypatch.setattr(
        adapter_module.RolloutGenerator, "generate_rollouts", fake_parent
    )


def test_adapter_calls_parent_and_environment_once_and_tags_pairs(monkeypatch) -> None:
    runner = FakeRunner()
    adapter = make_adapter(runner)
    base = np.tile(np.linspace(-0.2, 0.2, 12), (4, 1))
    parent_calls = []
    install_fake_parent(monkeypatch, parent_calls, base)
    original_bound_method = runner.run_policy_in_env

    episodes, _, _, _ = adapter.generate_rollouts(
        SimpleNamespace(), {}, None, None
    )

    assert len(parent_calls) == 1
    assert runner.calls == 1
    assert runner.run_policy_in_env == original_bound_method
    assert adapter._recovery_step == 1
    assert [episode["pair_id"] for episode in episodes] == [0, 0, 1, 1]
    assert [episode["perturb_type"] for episode in episodes] == [
        "none", "robot_init", "none", "robot_init"
    ]
    assert [episode["is_perturbed"] for episode in episodes] == [
        False, True, False, True
    ]
    np.testing.assert_array_equal(runner.received_states[0], base[0])
    np.testing.assert_array_equal(runner.received_states[2], base[0])
    assert not np.array_equal(runner.received_states[1, 1:8], base[0, 1:8])
    assert not np.array_equal(runner.received_states[3, 1:8], base[0, 1:8])
    assert episodes[1]["perturb_sampling_mode"] == "fixed_l2"
    assert episodes[1]["joint_noise_l2"] == pytest.approx(0.02)
    np.testing.assert_array_equal(
        runner.received_states[:, [0, 8, 9, 10, 11]],
        base[:, [0, 8, 9, 10, 11]],
    )


def test_adapter_restores_runner_method_when_parent_raises(monkeypatch) -> None:
    runner = FakeRunner()
    adapter = make_adapter(runner)
    base = np.zeros((4, 12), dtype=np.float64)
    parent_calls = []
    install_fake_parent(monkeypatch, parent_calls, base, fail=True)
    original_bound_method = runner.run_policy_in_env

    with pytest.raises(RuntimeError, match="parent failure"):
        adapter.generate_rollouts(SimpleNamespace(), {}, None, None)

    assert len(parent_calls) == 1
    assert runner.calls == 1
    assert runner.run_policy_in_env == original_bound_method
    assert adapter._adapter_active is False
    assert adapter._recovery_step == 0


def test_adapter_rejects_nonidentical_pair_bases_and_restores(monkeypatch) -> None:
    runner = FakeRunner()
    adapter = make_adapter(runner)
    base = np.zeros((4, 12), dtype=np.float64)
    base[3, 4] = 0.1
    parent_calls = []
    install_fake_parent(monkeypatch, parent_calls, base)
    original_bound_method = runner.run_policy_in_env

    with pytest.raises(ValueError, match="identical base states"):
        adapter.generate_rollouts(SimpleNamespace(), {}, None, None)

    assert len(parent_calls) == 1
    assert runner.calls == 0
    assert runner.run_policy_in_env == original_bound_method


@pytest.mark.parametrize(
    ("parallel_envs", "rollouts", "message"),
    [(2, 4, "num_parallel_envs"), (1, 8, "rollouts_per_env")],
)
def test_adapter_rejects_unsafe_runner_contract(
    parallel_envs, rollouts, message
) -> None:
    runner = FakeRunner()
    runner.num_parallel_envs = parallel_envs
    runner.rollouts_per_env = rollouts
    adapter = make_adapter(runner)

    with pytest.raises(ValueError, match=message):
        adapter.generate_rollouts(SimpleNamespace(), {}, None, None)


def test_adapter_detects_too_few_environment_episodes(monkeypatch) -> None:
    runner = FakeRunner()

    def short_rollout(*args, **kwargs):
        yield False, 0.0, {}

    runner.run_policy_in_env = short_rollout
    adapter = make_adapter(runner)
    base = np.zeros((4, 12), dtype=np.float64)
    parent_calls = []
    install_fake_parent(monkeypatch, parent_calls, base)

    with pytest.raises(RuntimeError, match="yielded 1 episodes"):
        adapter.generate_rollouts(SimpleNamespace(), {}, None, None)

    assert runner.run_policy_in_env is short_rollout


def test_adapter_detects_too_many_environment_episodes(monkeypatch) -> None:
    runner = FakeRunner()

    def long_rollout(*args, **kwargs):
        for _ in range(5):
            yield False, 0.0, {}

    runner.run_policy_in_env = long_rollout
    adapter = make_adapter(runner)
    base = np.zeros((4, 12), dtype=np.float64)
    parent_calls = []
    install_fake_parent(monkeypatch, parent_calls, base)

    with pytest.raises(RuntimeError, match="more than RLOO K"):
        adapter.generate_rollouts(SimpleNamespace(), {}, None, None)

    assert runner.run_policy_in_env is long_rollout


class FakeRolloutGenerator:
    def __init__(self, episodes, task_ids, valid_mask):
        self.episodes = episodes
        self.task_ids = torch.tensor(task_ids)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        self.calls = 0
        self.task_names_to_use = ["spatial-task"]
        self.enable_rollout_stats_tracking = False

    def generate_rollouts(self, model, batch, data_iterator, dataloader):
        self.calls += 1
        return self.episodes, self.task_ids, self.valid_mask, 1


def recovery_episode(*, perturbed: bool, success: bool):
    return {
        "perturb_type": "robot_init" if perturbed else "none",
        "is_perturbed": perturbed,
        "perturb_applied": perturbed,
        "success": success,
    }


def make_optimizer(generator, lambda_recovery=0.5):
    optimizer = RobotInitRecoveryOptimizer.__new__(RobotInitRecoveryOptimizer)
    optimizer.rollout_generator = generator
    optimizer.reward_function = RobotInitRecoveryReward(lambda_recovery)
    return optimizer


def install_fake_parent_optimizer(monkeypatch, parent_calls, *, call_twice=False, fail=False):
    def fake_optimize(
        self, model, batch, optimizers, data_iterator=None, dataloader=None
    ):
        parent_calls.append(1)
        result = self.rollout_generator.generate_rollouts(
            model, batch, data_iterator, dataloader
        )
        if call_twice:
            self.rollout_generator.generate_rollouts(
                model, batch, data_iterator, dataloader
            )
        if fail:
            raise RuntimeError("PPO failure")
        episodes = result[0]
        augmented = [
            self.reward_function.compute_reward(i, episode, batch)
            for i, episode in enumerate(episodes)
        ]
        # Deliberately mimic the misleading upstream shaped-reward metrics.
        return {
            "mean_scores": float(np.mean(augmented)),
            "mean_rlhf_reward": float(np.mean(augmented)),
            "rl_train_succeess_rate/spatial-task": float(np.mean(augmented)),
        }

    monkeypatch.setattr(
        adapter_module.RLOptimizerOpenVLAOFT, "optimize", fake_optimize
    )


def test_recovery_reward_requires_actual_successful_perturbation() -> None:
    reward = RobotInitRecoveryReward(lambda_recovery=0.5)

    assert reward.compute_reward(0, recovery_episode(perturbed=False, success=True), {}) == 1.0
    assert reward.compute_reward(1, recovery_episode(perturbed=True, success=True), {}) == 1.5
    assert reward.compute_reward(2, recovery_episode(perturbed=True, success=False), {}) == 0.0


def test_optimizer_calls_upstream_once_excludes_padding_and_corrects_metrics(monkeypatch) -> None:
    episodes = [
        recovery_episode(perturbed=False, success=True),
        recovery_episode(perturbed=True, success=True),
        recovery_episode(perturbed=False, success=False),
        # Padded duplicate must not affect any reported mean/rate.
        recovery_episode(perturbed=True, success=True),
    ]
    generator = FakeRolloutGenerator(episodes, [0, 0, 0, 0], [True, True, True, False])
    optimizer = make_optimizer(generator)
    original_bound_method = generator.generate_rollouts
    parent_calls = []
    install_fake_parent_optimizer(monkeypatch, parent_calls)

    metrics = optimizer.optimize(SimpleNamespace(), {}, [])

    assert len(parent_calls) == 1
    assert generator.calls == 1
    assert generator.generate_rollouts == original_bound_method
    assert metrics["mean_scores"] == pytest.approx(2 / 3)
    assert metrics["mean_R_success"] == pytest.approx(2 / 3)
    assert metrics["mean_R_recovery"] == pytest.approx(1 / 3)
    assert metrics["mean_R_total"] == pytest.approx(5 / 6)
    assert metrics["mean_rlhf_reward"] == pytest.approx(5 / 6)
    assert metrics["anchor_success_rate"] == 0.5
    assert metrics["perturbed_success_rate"] == 1.0
    assert metrics["valid_anchor_count"] == 2.0
    assert metrics["valid_perturbed_count"] == 1.0
    assert metrics["rl_train_succeess_rate/spatial-task"] == pytest.approx(2 / 3)


def test_optimizer_restores_generator_after_parent_failure(monkeypatch) -> None:
    generator = FakeRolloutGenerator(
        [recovery_episode(perturbed=False, success=False)], [0], [True]
    )
    optimizer = make_optimizer(generator)
    original_bound_method = generator.generate_rollouts
    parent_calls = []
    install_fake_parent_optimizer(monkeypatch, parent_calls, fail=True)

    with pytest.raises(RuntimeError, match="PPO failure"):
        optimizer.optimize(SimpleNamespace(), {}, [])

    assert generator.calls == 1
    assert generator.generate_rollouts == original_bound_method


def test_optimizer_rejects_second_rollout_request_and_restores(monkeypatch) -> None:
    generator = FakeRolloutGenerator(
        [recovery_episode(perturbed=False, success=False)], [0], [True]
    )
    optimizer = make_optimizer(generator)
    original_bound_method = generator.generate_rollouts
    parent_calls = []
    install_fake_parent_optimizer(monkeypatch, parent_calls, call_twice=True)

    with pytest.raises(RuntimeError, match="more than once"):
        optimizer.optimize(SimpleNamespace(), {}, [])

    assert generator.calls == 1
    assert generator.generate_rollouts == original_bound_method
