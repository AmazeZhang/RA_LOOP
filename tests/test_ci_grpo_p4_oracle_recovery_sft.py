import numpy as np

from ci_grpo.p4_oracle_recovery_sft import (
    action_chunk,
    balanced_microbatch_schedule,
    normalize_action_chunk,
)


def test_action_chunk_repeats_terminal_action():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk = action_chunk(actions, 1)
    assert chunk.shape == (8, 7)
    np.testing.assert_array_equal(chunk[0], actions[1])
    np.testing.assert_array_equal(chunk[1:], np.repeat(actions[2:3], 7, axis=0))


def test_normalize_action_chunk_preserves_unmasked_gripper():
    stats = {
        "q01": [-1, -1, -1, -1, -1, -1, 0],
        "q99": [1, 1, 1, 1, 1, 1, 1],
        "mask": [True, True, True, True, True, True, False],
    }
    chunk = np.zeros((8, 7), dtype=np.float32)
    chunk[:, -1] = 1
    normalized = normalize_action_chunk(chunk, stats)
    np.testing.assert_allclose(normalized[:, :6], 0)
    np.testing.assert_allclose(normalized[:, -1], 1)


def test_schedule_is_balanced_within_each_optimizer_step():
    schedule = balanced_microbatch_schedule(11, 5, optimizer_steps=3)
    assert len(schedule) == 24
    for start in range(0, len(schedule), 8):
        splits = [split for split, _ in schedule[start : start + 8]]
        assert splits.count("recovery") == 4
        assert splits.count("retention") == 4
