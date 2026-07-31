# Late-state oracle recovery SFT — result

Date: 2026-07-31

Preregistration:
`docs/ORACLE_RECOVERY_SFT_PREREG_20260731.md`

Primary artifacts:

- data audit: `ci_grpo/artifacts/p4_oracle_recovery_data/result.json`;
- training: `ci_grpo/artifacts/p4_oracle_recovery_sft/result.json`;
- evaluations:
  `ci_grpo/artifacts/p4_oracle_recovery_eval_step25_run2/result.json`,
  `ci_grpo/artifacts/p4_oracle_recovery_eval_step50/result.json`,
  `ci_grpo/artifacts/p4_oracle_recovery_eval_step100/result.json`, and
  `ci_grpo/artifacts/p4_oracle_recovery_eval_step200/result.json`.

## Decision

**The fixed small-data recovery SFT fails. Do not proceed to GRPO from this
checkpoint family.**

The generated dataset contained nine successful scripted recovery trajectories
and three successful frozen-policy retention trajectories, totaling 1,243
transitions. Training completed the fixed 200 optimizer steps. The final sampled
microbatch L1 was 0.0798893, down from the smoke sample's initial 0.155104, but
this was not a held-out metric.

All four preregistered checkpoints retained the original tasks and failed every
late instruction revision:

| Checkpoint | Exact P2 replays | Revised goal | Original goal after switch | No tested goal | Retention |
|---:|---:|---:|---:|---:|---:|
| 25 | 9/9 | 0/9 | 8/9 | 1/9 | 3/3 |
| 50 | 9/9 | 0/9 | 9/9 | 0/9 | 3/3 |
| 100 | 9/9 | 0/9 | 9/9 | 0/9 | 3/3 |
| 200 | 9/9 | 0/9 | 9/9 | 0/9 | 3/3 |

No checkpoint reached even one revised goal, so none approached the required
6/9 threshold or succeeded on any directed task pair. The supported conclusion
is narrow: this rank-32 LoRA plus action-head, 1:1, 200-step intervention
preserved the three tested initial tasks but did not repair the byte-exact P2
late-state failure.

## Integrity correction: P2 and P3 state provenance

The first step-25 evaluation attempted to replay prefixes stored as float32 in
the training NPZ files. That quantization changed MuJoCo state hashes, so the
run was stopped and its incomplete output is invalid. `step25_run2` and all
later evaluations instead used float64 processed-action prefixes independently
reconstructed from the frozen policy. Every prefix was checked against the
original corrected P2 SHA before evaluation.

This correction exposed a deeper provenance issue. The scripted recovery
demonstrations were generated from P3-reconstructed switch states (for example,
the first P3 hash begins `b44081...`), while the corrected temporal evaluation
uses P2 states (the corresponding first hash begins `4d5a61...`). Episode
lengths also differ. The two sets follow the same tasks, offsets, checkpoint,
and mechanical rule, but they are not the same nine byte-exact states.

Consequences:

- the four final SFT evaluations on P2 are valid and exact;
- the claim that the P3 oracle succeeded from the exact P2 states is withdrawn;
- the SFT failure cannot distinguish failed imitation from state-provenance
  mismatch or closed-loop compounding;
- no general claim that supervised recovery cannot work is supported.

## Result-to-claim review

A fresh same-family reviewer returned:

- `claim_supported: no`;
- `confidence: high`;
- `route: pivot`.

It confirmed the 0/9 recovery and 3/3 retention numbers and independently
flagged the P2/P3 hash mismatch. Because the deterministic evidence checker was
unavailable, the semantic verdict remains provisional. A subsequent
same-family experiment-integrity audit returned `WARN`: primary counts and
normalization passed, while state provenance and narrow scope require the
qualifiers above. Full repository tests passed in the isolated official-LIBERO
environment: 108 passed.

## Next decision

Do not increase SFT steps, change the learning rate, or start sparse-reward
GRPO on this intervention. If one final diagnostic is authorized, it should be
bounded and causal:

1. evaluate predicted-versus-oracle action error on stored recovery
   observations; and
2. evaluate closed-loop checkpoints on the exact P3 demonstration start states.

Those two checks distinguish “the policy did not imitate” from “imitation
worked offline but failed under state mismatch/closed-loop drift.” If recovery
is still absent on the exact demonstration states, retire this SFT formulation
and pivot to a materially different temporal-control representation rather than
another budget sweep.
