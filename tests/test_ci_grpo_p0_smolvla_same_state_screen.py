from ci_grpo.p0_smolvla_same_state_screen import group_metrics


def test_group_metrics_detect_language_deafness_with_high_hit():
    rows = []
    for goal in (1, 2):
        for instruction in (1, 2):
            rows.append(
                {
                    "scored_goal_task_id": goal,
                    "instruction_task_id": instruction,
                    "success": True,
                }
            )

    result = group_metrics(rows, (1, 2))

    assert result["hit"] == 1.0
    assert result["miss"] == 1.0
    assert result["lsg"] == 0.0
    assert result["language_deaf_candidate"] is True


def test_group_metrics_reject_sensitive_policy():
    rows = []
    for goal in (1, 2):
        for instruction in (1, 2):
            rows.append(
                {
                    "scored_goal_task_id": goal,
                    "instruction_task_id": instruction,
                    "success": goal == instruction,
                }
            )

    result = group_metrics(rows, (1, 2))

    assert result["hit"] == 1.0
    assert result["miss"] == 0.0
    assert result["lsg"] == 1.0
    assert result["language_deaf_candidate"] is False
