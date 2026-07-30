from ci_grpo.p1_temporal_regrounding_screen import (
    checkpoint_indices,
    summarize_temporal_rows,
)


def test_checkpoint_indices_are_distinct_and_internal():
    assert checkpoint_indices(90) == [29, 59]


def _row(origin, instruction, checkpoint, instruction_success, terminal):
    return {
        "origin_task": origin,
        "instruction_task": instruction,
        "checkpoint_label": checkpoint,
        "is_control": origin == instruction,
        "checkpoint_all_goals_false": True,
        "exact_restore": True,
        "success_for_instruction_goal": instruction_success,
        "terminal_goal_truth": terminal,
    }


def test_temporal_inertia_candidate_requires_controls_and_multiple_cells():
    tasks = ("plate", "stove", "cabinet")
    rows = []
    for origin, checkpoint in (("plate", "early"), ("stove", "late")):
        rows.append(
            _row(
                origin,
                origin,
                checkpoint,
                True,
                {task: task == origin for task in tasks},
            )
        )
        for instruction in tasks:
            if instruction != origin:
                rows.append(
                    _row(
                        origin,
                        instruction,
                        checkpoint,
                        False,
                        {task: task == origin for task in tasks},
                    )
                )
    metrics = summarize_temporal_rows(rows)
    assert metrics["switch_success_rate"] == 0.0
    assert metrics["original_goal_inertia_rate"] == 1.0
    assert metrics["n_distinct_inertia_checkpoints"] == 2
    assert metrics["temporal_inertia_candidate"]
    assert metrics["decision"] == "provisional_candidate"


def test_successful_switching_is_no_go():
    tasks = ("plate", "stove", "cabinet")
    rows = []
    for origin, checkpoint in (("plate", "early"), ("stove", "late")):
        rows.append(
            _row(
                origin,
                origin,
                checkpoint,
                True,
                {task: task == origin for task in tasks},
            )
        )
        for instruction in tasks:
            if instruction != origin:
                rows.append(
                    _row(
                        origin,
                        instruction,
                        checkpoint,
                        True,
                        {task: task == instruction for task in tasks},
                    )
                )
    metrics = summarize_temporal_rows(rows)
    assert metrics["switch_success_rate"] == 1.0
    assert metrics["original_goal_inertia_rate"] == 0.0
    assert not metrics["temporal_inertia_candidate"]
    assert metrics["decision"] == "no_go"


def test_failed_control_makes_screen_inconclusive():
    rows = [
        _row("plate", "plate", "late", False, {"plate": False, "stove": False}),
        _row("plate", "stove", "late", False, {"plate": False, "stove": False}),
    ]
    metrics = summarize_temporal_rows(rows)
    assert metrics["decision"] == "inconclusive_control_failure"
