from ci_grpo.p2_semantic_chunk_interrupt_screen import (
    penultimate_chunk_index,
    summarize_interrupt_rows,
)


def test_penultimate_chunk_index():
    assert penultimate_chunk_index(10) == 8


def _row(case, method, success):
    origin, revised, offset = case
    return {
        "origin_task": origin,
        "revised_task": revised,
        "offset": offset,
        "method": method,
        "baseline_success": True,
        "checkpoint_all_goals_false": True,
        "exact_restore": True,
        "checkpoint_state_sha256": f"{origin}-{offset}",
        "terminal_goal_truth": {
            origin: False,
            revised: success,
        },
        "response_latency_actions": 4 if method == "stale" else 0,
        "switch_action_jerk_l2": 0.2 if method == "stale" else 0.4,
    }


def test_flush_candidate_requires_two_additional_successes():
    cases = [
        (f"old-{index}", f"new-{index}", index)
        for index in range(9)
    ]
    rows = []
    for index, case in enumerate(cases):
        rows.append(_row(case, "stale", index < 4))
        rows.append(_row(case, "flush", index < 7))
    metrics = summarize_interrupt_rows(rows)
    assert metrics["stale_revised_goal_success_rate"] == 4 / 9
    assert metrics["flush_revised_goal_success_rate"] == 7 / 9
    assert metrics["semantic_interrupt_candidate"]


def test_equal_success_is_no_go():
    cases = [("plate", "stove", index) for index in range(1, 7)]
    rows = [
        _row(case, method, True)
        for case in cases
        for method in ("stale", "flush")
    ]
    metrics = summarize_interrupt_rows(rows)
    assert metrics["flush_success_improvement"] == 0.0
    assert metrics["decision"] == "no_go"
