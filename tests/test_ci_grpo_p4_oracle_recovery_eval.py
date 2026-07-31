import json
from pathlib import Path

from ci_grpo.p4_oracle_recovery_eval import summarize_recovery_eval


def test_recovery_eval_gate_passes_at_preregistered_thresholds():
    retention = [
        {"original_goal_success": True},
        {"original_goal_success": True},
        {"original_goal_success": False},
    ]
    recovery = []
    for pair in range(3):
        for offset in (1, 4, 7):
            recovery.append(
                {
                    "origin_task": f"old-{pair}",
                    "revised_task": f"new-{pair}",
                    "exact_checkpoint_replay": True,
                    "revised_goal_success": offset != 7,
                }
            )
    metrics = summarize_recovery_eval(retention, recovery)
    assert metrics["n_revised_goal_successes"] == 6
    assert metrics["recovery_sft_gate_pass"]


def test_recovery_eval_gate_rejects_missing_directed_pair():
    retention = [{"original_goal_success": True}] * 3
    recovery = []
    for pair in range(3):
        for _ in range(3):
            recovery.append(
                {
                    "origin_task": f"old-{pair}",
                    "revised_task": f"new-{pair}",
                    "exact_checkpoint_replay": True,
                    "revised_goal_success": pair != 2,
                }
            )
    metrics = summarize_recovery_eval(retention, recovery)
    assert metrics["n_revised_goal_successes"] == 6
    assert not metrics["all_directed_pairs_have_success"]
    assert not metrics["recovery_sft_gate_pass"]


def test_p2_p3_state_provenance_mismatch_is_machine_readable():
    path = (
        Path(__file__).resolve().parents[1]
        / "ci_grpo/artifacts/p4_p2_p3_state_provenance/result.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["n_states"] == 9
    assert result["n_byte_exact_matches"] == 0
    assert result["all_states_differ"]
    assert len(result["rows"]) == 9
    assert all(not row["byte_exact_match"] for row in result["rows"])
