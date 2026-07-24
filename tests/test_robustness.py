import numpy as np
import pytest

from ra_loop.robustness import (
    PANDA_ARM_JOINT_NAMES,
    CounterfactualRecoveryAdvantageResult,
    CounterfactualRecoveryMetrics,
    NamedJointLayout,
    RolloutPlanEntry,
    apply_robot_init_perturbation,
    build_robot_init_rollout_plan,
    compute_counterfactual_recovery_advantages,
    compute_counterfactual_recovery_metrics,
    compute_recovery_rewards,
    compute_mode_stratified_rloo_advantages,
    materialize_rollout_plan,
    resolve_named_joint_layout,
)


def test_counterfactual_advantage_uses_only_successful_anchor_pairs() -> None:
    result = compute_counterfactual_recovery_advantages(
        # Pair outcomes: SS, SF, FS, FF.
        [True, True, True, False, False, True, False, False],
        [False, True] * 4,
        [0, 0, 1, 1, 2, 2, 3, 3],
    )

    np.testing.assert_array_equal(
        result.advantages,
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        result.eligible_mask,
        [False, True, False, True, False, False, False, False],
    )
    assert result.eligible_pairs == 2
    assert result.excluded_anchor_failure_pairs == 2
    assert result.groups_with_update == 1
    assert result.groups_without_baseline == 0


def test_counterfactual_advantage_is_unchanged_by_excluded_perturbation() -> None:
    common = dict(
        perturbed_mask=[False, True] * 3,
        pair_ids=[0, 0, 1, 1, 2, 2],
    )
    first = compute_counterfactual_recovery_advantages(
        [True, True, True, False, False, False],
        **common,
    )
    second = compute_counterfactual_recovery_advantages(
        [True, True, True, False, False, True],
        **common,
    )

    np.testing.assert_array_equal(first.advantages, second.advantages)
    assert first.advantages[5] == 0.0
    assert second.advantages[5] == 0.0


def test_counterfactual_advantage_respects_rollout_groups() -> None:
    result = compute_counterfactual_recovery_advantages(
        [
            True, True, True, False,
            True, False, True, True,
        ],
        [False, True] * 4,
        [0, 0, 1, 1, 0, 0, 1, 1],
        rollout_group_ids=[0, 0, 0, 0, 1, 1, 1, 1],
    )

    np.testing.assert_array_equal(
        result.advantages,
        [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
    )
    assert result.groups_with_update == 2


def test_counterfactual_advantage_audits_invalid_and_small_groups() -> None:
    result = compute_counterfactual_recovery_advantages(
        [True, True, True, False, True, True],
        [False, True] * 3,
        [0, 0, 1, 1, 2, 2],
        valid_mask=[True, True, True, False, True, False],
    )

    np.testing.assert_array_equal(result.advantages, np.zeros(6))
    assert result.eligible_pairs == 1
    assert result.dropped_invalid_pairs == 2
    assert result.groups_without_baseline == 1


@pytest.mark.parametrize(
    ("successes", "modes", "pair_ids", "kwargs", "message"),
    [
        ([1, 0], [False, True], [0, 0], {}, "boolean"),
        ([True, False], [0, 1], [0, 0], {}, "boolean"),
        ([True, False], [False, True], [False, False], {}, "integer"),
        (
            [True, False, True],
            [False, True, False],
            [0, 0, 0],
            {},
            "exactly one anchor",
        ),
        (
            [True, False],
            [False, True],
            [0, 0],
            {"valid_mask": [False, False]},
            "at least one",
        ),
    ],
)
def test_counterfactual_advantage_rejects_unsafe_inputs(
    successes, modes, pair_ids, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_counterfactual_recovery_advantages(
            successes,
            modes,
            pair_ids,
            **kwargs,
        )


def test_counterfactual_metrics_cover_all_pair_outcomes() -> None:
    result = compute_counterfactual_recovery_metrics(
        [True, True, True, False, False, True, False, False],
        [False, True] * 4,
        [0, 0, 1, 1, 2, 2, 3, 3],
    )

    assert result == CounterfactualRecoveryMetrics(
        both_success=1,
        anchor_only_success=1,
        perturbed_only_success=1,
        both_failure=1,
        complete_pairs=4,
        dropped_incomplete_pairs=0,
        anchor_success_rate=0.5,
        perturbed_success_rate=0.5,
        counterfactual_recovery_rate=0.5,
        recoverability_gap=0.5,
    )


def test_counterfactual_metrics_condition_only_on_anchor_success() -> None:
    result = compute_counterfactual_recovery_metrics(
        [True, True, True, False, False, True, False, False],
        [False, True] * 4,
        [0, 0, 1, 1, 2, 2, 3, 3],
    )

    # The perturbed-only success and both-failure pairs do not enter the CRR
    # denominator: they show base-policy uncertainty, not recoverability.
    assert result.counterfactual_recovery_rate == 1 / 2
    assert result.recoverability_gap == 1 / 2


def test_counterfactual_metrics_report_incomplete_valid_pair() -> None:
    result = compute_counterfactual_recovery_metrics(
        [True, False, True, True],
        [False, True, False, True],
        [0, 0, 1, 1],
        valid_mask=[True, True, True, False],
    )

    assert result.complete_pairs == 1
    assert result.dropped_incomplete_pairs == 1
    assert result.anchor_only_success == 1


def test_counterfactual_metrics_disambiguate_reused_pair_ids_by_group() -> None:
    result = compute_counterfactual_recovery_metrics(
        [True, True, True, False],
        [False, True, False, True],
        [0, 0, 0, 0],
        rollout_group_ids=[0, 0, 1, 1],
    )

    assert result.complete_pairs == 2
    assert result.both_success == 1
    assert result.anchor_only_success == 1


def test_counterfactual_metrics_mark_crr_undefined_without_anchor_success() -> None:
    result = compute_counterfactual_recovery_metrics(
        [False, True, False, False],
        [False, True, False, True],
        [0, 0, 1, 1],
    )

    assert np.isnan(result.counterfactual_recovery_rate)
    assert np.isnan(result.recoverability_gap)


@pytest.mark.parametrize(
    ("successes", "modes", "pair_ids", "kwargs", "message"),
    [
        ([1, 0], [False, True], [0, 0], {}, "boolean"),
        ([True, False], [0, 1], [0, 0], {}, "boolean"),
        ([True, False], [False, True], [False, False], {}, "integer"),
        (
            [True, False, True],
            [False, True, False],
            [0, 0, 0],
            {},
            "exactly one valid anchor",
        ),
        (
            [True, False],
            [False, True],
            [0, 0],
            {"valid_mask": [False, False]},
            "at least one",
        ),
    ],
)
def test_counterfactual_metrics_reject_unsafe_inputs(
    successes, modes, pair_ids, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_counterfactual_recovery_metrics(
            successes,
            modes,
            pair_ids,
            **kwargs,
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
