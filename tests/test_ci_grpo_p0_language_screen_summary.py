from ci_grpo.p0_language_screen_summary import rates_for_tasks


def test_rates_identify_high_hit_high_miss_candidate():
    tasks = ("goal_a", "goal_b")
    rows = [
        {
            "instruction_task": "goal_a",
            "terminal_goal_truth": {"goal_a": True, "goal_b": False},
        },
        {
            "instruction_task": "goal_b",
            "terminal_goal_truth": {"goal_a": True, "goal_b": True},
        },
    ]

    result = rates_for_tasks(rows, tasks)

    assert result["hit"] == 1.0
    assert result["miss"] == 0.5
    assert result["lsg"] == 0.5
    assert result["language_deaf_candidate"] is False


def test_rates_reject_low_hit_even_when_lsg_is_small():
    tasks = ("goal_a", "goal_b")
    rows = [
        {
            "instruction_task": "goal_a",
            "terminal_goal_truth": {"goal_a": False, "goal_b": False},
        },
        {
            "instruction_task": "goal_b",
            "terminal_goal_truth": {"goal_a": False, "goal_b": False},
        },
    ]

    result = rates_for_tasks(rows, tasks)

    assert result["hit"] == 0.0
    assert result["lsg"] == 0.0
    assert result["language_deaf_candidate"] is False


def test_rates_accept_high_hit_near_zero_lsg_candidate():
    tasks = ("goal_a", "goal_b")
    rows = [
        {
            "instruction_task": "goal_a",
            "terminal_goal_truth": {"goal_a": True, "goal_b": True},
        },
        {
            "instruction_task": "goal_b",
            "terminal_goal_truth": {"goal_a": True, "goal_b": True},
        },
    ]

    result = rates_for_tasks(rows, tasks)

    assert result["hit"] == 1.0
    assert result["miss"] == 1.0
    assert result["lsg"] == 0.0
    assert result["language_deaf_candidate"] is True
