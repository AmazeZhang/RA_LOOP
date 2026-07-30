import numpy as np

from ci_grpo.p0_openvla_k3_rollout import (
    TASKS,
    calibrate_dtw_threshold,
    diagnostic_metrics,
    dtw_distance,
)


def test_dtw_zero_for_identical_trajectories():
    trajectory = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    result = dtw_distance(trajectory, trajectory)

    assert result["total"] == 0.0
    assert result["mean_per_path_step"] == 0.0


def test_dtw_positive_for_different_trajectories():
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    second = np.asarray([[0.0, 1.0], [1.0, 1.0]])

    result = dtw_distance(first, second)

    assert result["total"] > 0.0
    assert result["mean_per_path_step"] > 0.0


def test_calibrated_threshold_uses_effect_floor_when_repeat_is_exact():
    result = calibrate_dtw_threshold([0.08, 0.12, 0.16], [0.0, 0.0, 0.0])

    assert result["threshold"] == 0.01
    assert result["all_between_instruction_pairs_pass"] is True


def test_calibrated_threshold_rejects_pair_inside_repeat_noise_margin():
    result = calibrate_dtw_threshold([0.02, 0.12], [0.01])

    assert result["threshold"] == 0.05
    assert result["all_between_instruction_pairs_pass"] is False


def test_language_sensitivity_summary():
    rows = []
    for task in TASKS:
        rows.append(
            {
                "instruction_task": task,
                "terminal_goal_truth": {
                    scored: scored == task for scored in TASKS
                },
            }
        )

    result = diagnostic_metrics(rows)

    assert result["lsg_hit"] == 1.0
    assert result["lsg_miss"] == 0.0
    assert result["lsg"] == 1.0
    assert result["instruction_swap_redirection_rate"] == 1.0
    assert result["baseline_language_deafness_premise_supported"] is False
