# Research findings

## 2026-07-31 — small oracle-recovery SFT

Status: **negative; pivot**

Tested a preregistered rank-32 LoRA plus continuous action-head SFT using nine
scripted late-recovery trajectories mixed 1:1 with three frozen-policy
retention trajectories. Training completed 200 optimizer steps.

What held:

- original initial-task retention was 3/3 at steps 25, 50, 100, and 200;
- all 9/9 corrected P2 switch prefixes replayed byte-exactly at every valid
  evaluation.

What failed:

- revised-goal success was 0/9 at every checkpoint;
- steps 50, 100, and 200 returned to the old goal in 9/9 branches;
- a lower sampled training loss did not translate into closed-loop recovery.

Integrity/provenance lesson:

- P3 recovery demonstrations and corrected P2 evaluation states share the same
  task/offset construction but have different state hashes and episode lengths;
- therefore P3 reachability does not establish exact P2-state reachability, and
  the SFT failure is confounded by state provenance;
- the initial float32-prefix step-25 evaluation was stopped because it failed
  byte-exact replay and is not evidence.

Do not repeat:

- longer training or learning-rate sweeps on the same dataset without first
  checking offline oracle-action imitation;
- sparse-reward GRPO from a policy with 0/9 positive recovery;
- claims that P2 and P3 used the same exact states without matching hashes.

Next bounded diagnostic, if pursued: offline predicted-versus-oracle action
error plus closed-loop evaluation on exact demonstration start states. The
same-family result-to-claim review judged `claim_supported: no`, confidence
high, route `pivot`. The subsequent same-family integrity audit returned
`WARN` because of state provenance and scope; primary result fidelity and score
normalization passed. Deterministic evidence checking was unavailable.
