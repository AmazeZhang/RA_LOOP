import numpy as np

from ci_grpo.p3_scripted_reachability_audit import (
    libero_env_action_to_training_action,
    position_action,
    summarize_oracle_rows,
    transport_waypoints,
)


def test_libero_action_conversion_matches_openvla_gripper_convention():
    closed = libero_env_action_to_training_action(
        np.array([0.1, 0, 0, 0, 0, 0, 1.0])
    )
    opened = libero_env_action_to_training_action(
        np.array([0.1, 0, 0, 0, 0, 0, -1.0])
    )
    np.testing.assert_allclose(closed, [0.1, 0, 0, 0, 0, 0, 0.0])
    np.testing.assert_allclose(opened, [0.1, 0, 0, 0, 0, 0, 1.0])


def test_position_action_clips_and_preserves_rotation():
    action = position_action(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, -0.025, 0.0]),
        gripper=1.0,
    )
    np.testing.assert_allclose(action, [1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_transport_waypoints_preserve_grasp_offset():
    points = transport_waypoints(
        [0.0, 0.0, 0.9],
        [0.2, 0.1, 1.0],
        [0.0, 0.0, 0.1],
    )
    np.testing.assert_allclose(points[0], [0.0, 0.0, 1.25])
    np.testing.assert_allclose(points[1], [0.2, 0.1, 1.25])
    np.testing.assert_allclose(points[2], [0.2, 0.1, 1.1])


def test_reachability_gate_requires_success_in_every_pair():
    rows = []
    for pair_index in range(3):
        for offset in (1, 4, 7):
            rows.append(
                {
                    "origin_task": f"old-{pair_index}",
                    "revised_task": f"new-{pair_index}",
                    "eligible": True,
                    "revised_goal_success": offset != 7,
                }
            )
    metrics = summarize_oracle_rows(rows)
    assert metrics["n_revised_goal_success"] == 6
    assert metrics["reachability_pass"]
