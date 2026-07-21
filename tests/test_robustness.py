import numpy as np
import pytest

from ra_loop.robustness import (
    PANDA_ARM_JOINT_NAMES,
    NamedJointLayout,
    RolloutPlanEntry,
    apply_robot_init_perturbation,
    build_robot_init_rollout_plan,
    compute_recovery_rewards,
    compute_mode_stratified_rloo_advantages,
    materialize_rollout_plan,
    resolve_named_joint_layout,
)


class FakeMujocoModel:
    def __init__(self):
        self.addresses = {name: index for index, name in enumerate(PANDA_ARM_JOINT_NAMES)}
        self.ids = {name: index for index, name in enumerate(PANDA_ARM_JOINT_NAMES)}
        self.jnt_limited = np.ones(7, dtype=bool)
        self.jnt_range = np.tile(np.array([-1.0, 1.0]), (7, 1))

    def get_joint_qpos_addr(self, name):
        return self.addresses[name]

    def joint_name2id(self, name):
        return self.ids[name]


@pytest.fixture
def layout() -> NamedJointLayout:
    return resolve_named_joint_layout(FakeMujocoModel())


def test_rollout_plan_contains_truthful_anchor_pairs() -> None:
    plan = build_robot_init_rollout_plan(num_pairs=4, strength=0.02, base_seed=17)

    assert len(plan) == 8
    assert [item.pair_id for item in plan] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [item.perturb_type for item in plan] == ["none", "robot_init"] * 4
    assert [item.is_perturbed for item in plan] == [False, True] * 4
    assert [item.seed for item in plan[1::2]] == [17, 18, 19, 20]


@pytest.mark.parametrize("strength", [0.0, -0.1, np.nan])
def test_rollout_plan_rejects_noop_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        build_robot_init_rollout_plan(num_pairs=1, strength=strength, base_seed=0)


def test_named_joint_resolution_does_not_assume_leading_slice(layout) -> None:
    assert layout.qpos_indices == (1, 2, 3, 4, 5, 6, 7)
    np.testing.assert_array_equal(layout.lower, -np.ones(7))
    np.testing.assert_array_equal(layout.upper, np.ones(7))


def test_named_joint_resolution_has_explicit_flat_state_offset() -> None:
    model = FakeMujocoModel()
    qpos_layout = resolve_named_joint_layout(model, state_qpos_offset=0)
    flat_state_layout = resolve_named_joint_layout(model, state_qpos_offset=1)

    assert qpos_layout.qpos_indices == (0, 1, 2, 3, 4, 5, 6)
    assert flat_state_layout.qpos_indices == (1, 2, 3, 4, 5, 6, 7)


def test_named_joint_resolution_rejects_unlimited_joint() -> None:
    model = FakeMujocoModel()
    model.jnt_limited[3] = False
    with pytest.raises(ValueError, match="finite position limit"):
        resolve_named_joint_layout(model)


def test_perturbation_is_deterministic_limited_and_changes_only_named_joints(layout) -> None:
    state = np.zeros(12, dtype=np.float64)
    state[0] = 0.25
    state[8:] = [0.034, -0.034, 9.0, 10.0]

    first = apply_robot_init_perturbation(state, layout=layout, strength=10.0, seed=5)
    second = apply_robot_init_perturbation(state, layout=layout, strength=10.0, seed=5)

    np.testing.assert_array_equal(first.state, second.state)
    assert first.applied is True
    assert np.all(first.state[1:8] >= -1.0)
    assert np.all(first.state[1:8] <= 1.0)
    np.testing.assert_array_equal(first.state[[0, 8, 9, 10, 11]], state[[0, 8, 9, 10, 11]])
    np.testing.assert_array_equal(state, np.array([0.25, 0, 0, 0, 0, 0, 0, 0, 0.034, -0.034, 9, 10]))


def test_fixed_l2_perturbation_matches_requested_plus_radius(layout) -> None:
    state = np.zeros(12, dtype=np.float64)
    first = apply_robot_init_perturbation(
        state,
        layout=layout,
        strength=0.1,
        seed=20260720,
        sampling_mode="fixed_l2",
    )
    second = apply_robot_init_perturbation(
        state,
        layout=layout,
        strength=0.1,
        seed=20260720,
        sampling_mode="fixed_l2",
    )

    np.testing.assert_array_equal(first.state, second.state)
    assert np.linalg.norm(first.noise) == pytest.approx(0.1, abs=1e-12)
    np.testing.assert_array_equal(
        first.state[[0, 8, 9, 10, 11]], state[[0, 8, 9, 10, 11]]
    )


def test_perturbation_rejects_unknown_sampling_mode(layout) -> None:
    with pytest.raises(ValueError, match="unsupported Robot-init sampling mode"):
        apply_robot_init_perturbation(
            np.zeros(12),
            layout=layout,
            strength=0.1,
            seed=1,
            sampling_mode="per_joint_l2",
        )


def test_materialized_plan_has_real_anchor_and_truthful_metadata(layout) -> None:
    state = np.linspace(-0.2, 0.2, 12)
    plan = build_robot_init_rollout_plan(num_pairs=2, strength=0.02, base_seed=7)
    rollouts = materialize_rollout_plan(state, plan=plan, layout=layout)

    np.testing.assert_array_equal(rollouts[0][0], state)
    assert rollouts[0][1]["perturb_applied"] is False
    assert rollouts[1][1]["perturb_applied"] is True
    assert not np.array_equal(rollouts[1][0][1:8], state[1:8])
    np.testing.assert_array_equal(rollouts[1][0][[0, 8, 9, 10, 11]], state[[0, 8, 9, 10, 11]])


def test_materialized_fixed_l2_plan_records_requested_and_applied_radius(layout) -> None:
    state = np.zeros(12, dtype=np.float64)
    plan = build_robot_init_rollout_plan(
        num_pairs=1,
        strength=0.1,
        base_seed=9,
        sampling_mode="fixed_l2",
    )
    rollouts = materialize_rollout_plan(state, plan=plan, layout=layout)
    metadata = rollouts[1][1]

    assert metadata["perturb_sampling_mode"] == "fixed_l2"
    assert metadata["perturb_strength"] == 0.1
    assert metadata["joint_noise_l2"] == pytest.approx(0.1, abs=1e-12)


def test_materialization_rejects_unsupported_perturbation(layout) -> None:
    plan = [RolloutPlanEntry(0, 0, "camera", 0.1, 1, True)]
    with pytest.raises(ValueError, match="unsupported"):
        materialize_rollout_plan(np.zeros(12), plan=plan, layout=layout)


def test_stratified_rloo_all_success_is_zero_in_both_modes() -> None:
    advantages = compute_mode_stratified_rloo_advantages(
        [1.0, 1.5] * 4,
        [False, True] * 4,
    )

    np.testing.assert_array_equal(advantages, np.zeros(8))


def test_stratified_rloo_modes_do_not_change_each_others_advantages() -> None:
    modes = [False, True] * 4
    first = compute_mode_stratified_rloo_advantages(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        modes,
    )
    second = compute_mode_stratified_rloo_advantages(
        [1.0, 1.5, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0],
        modes,
    )

    np.testing.assert_allclose(first[::2], [2 / 3, -2 / 3, 2 / 3, -2 / 3])
    np.testing.assert_array_equal(first[::2], second[::2])
    assert first[1::2].sum() == pytest.approx(0.0)
    assert second[1::2].sum() == pytest.approx(0.0)


def test_stratified_rloo_separates_rollout_groups_and_excludes_padding() -> None:
    rewards = [
        1.0, 1.5, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0,
        0.0, 1.5, 0.0, 1.5, 0.0, 1.5, 0.0, 1.5,
        99.0, 99.0,
    ]
    modes = [False, True] * 9
    groups = [0] * 8 + [1] * 8 + [1] * 2
    valid = [True] * 16 + [False, False]

    advantages = compute_mode_stratified_rloo_advantages(
        rewards,
        modes,
        rollout_group_ids=groups,
        valid_mask=valid,
    )

    for group_id in (0, 1):
        group = np.asarray(groups) == group_id
        for mode in (False, True):
            stratum = group & (np.asarray(modes) == mode) & np.asarray(valid)
            assert advantages[stratum].sum() == pytest.approx(0.0)
    np.testing.assert_array_equal(advantages[-2:], [0.0, 0.0])
    # Group 1's all-failed anchors and all-successful perturbations are both
    # already mode-optimal and therefore receive zero update.
    np.testing.assert_array_equal(advantages[8:16], np.zeros(8))


@pytest.mark.parametrize(
    ("rewards", "modes", "kwargs", "message"),
    [
        ([1.0, np.nan, 0.0, 0.0], [False, True, False, True], {}, "finite"),
        ([1.0, 1.5, 0.0, 0.0], [0, 1, 0, 1], {}, "boolean"),
        (
            [1.0, 1.5, 0.0, 0.0],
            [False, True, False, True],
            {"rollout_group_ids": [False, False, False, False]},
            "integer",
        ),
        (
            [1.0, 1.5, 0.0, 0.0],
            [False, True, False, True],
            {"valid_mask": [False, False, False, False]},
            "at least one",
        ),
        (
            [1.0, 1.5, 0.0, 0.0],
            [False, True, True, True],
            {},
            "at least two valid anchor",
        ),
    ],
)
def test_stratified_rloo_rejects_unsafe_inputs(
    rewards, modes, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_mode_stratified_rloo_advantages(rewards, modes, **kwargs)


def test_recovery_reward_and_valid_mask_statistics() -> None:
    episodes = [
        {"perturb_type": "none", "is_perturbed": False, "perturb_applied": False, "success": True},
        {"perturb_type": "robot_init", "is_perturbed": True, "perturb_applied": True, "success": True},
        {"perturb_type": "none", "is_perturbed": False, "perturb_applied": False, "success": False},
        {"perturb_type": "robot_init", "is_perturbed": True, "perturb_applied": True, "success": False},
    ]
    result = compute_recovery_rewards(
        episodes, [1.0, 1.0, 0.0, 0.0], lambda_recovery=0.5,
        valid_mask=[True, True, False, True],
    )

    np.testing.assert_array_equal(result.total, [1.0, 1.5, 0.0, 0.0])
    np.testing.assert_array_equal(result.recovery, [0.0, 1.0, 0.0, 0.0])
    assert result.stats["anchor_success_rate"] == 1.0
    assert result.stats["perturbed_success_rate"] == 0.5
    assert result.stats["valid_anchor_count"] == 1.0
    assert result.stats["valid_perturbed_count"] == 2.0


def test_zero_recovery_weight_is_exact_vanilla() -> None:
    episodes = [
        {"perturb_type": "robot_init", "is_perturbed": True, "perturb_applied": True, "success": True},
        {"perturb_type": "none", "is_perturbed": False, "perturb_applied": False, "success": False},
    ]
    base = np.array([0.25, -0.5], dtype=np.float64)
    result = compute_recovery_rewards(episodes, base, lambda_recovery=0.0)
    np.testing.assert_array_equal(result.total, base)


def test_recovery_reward_rejects_false_perturbation_label() -> None:
    episode = {
        "perturb_type": "robot_init",
        "is_perturbed": True,
        "perturb_applied": False,
        "success": True,
    }
    with pytest.raises(ValueError, match="inconsistent"):
        compute_recovery_rewards([episode], [1.0], lambda_recovery=0.5)
