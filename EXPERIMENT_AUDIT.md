# Experiment Audit Report

**Date:** 2026-07-31  
**Auditor:** fresh same-family Codex reviewer (provisional)  
**Project:** RA_LOOP late-state oracle recovery SFT

## Overall Verdict: WARN

## Integrity Status: warn

Reason: `disclosed_state_provenance_mismatch_and_minor_claim_fidelity_error`.

The final negative SFT result is supported: all four valid checkpoints used
byte-exact P2 replays, achieved 0/9 revised-goal success, and retained 3/3
initial tasks. The warning is about causal interpretation, not phantom results.
Recovery demonstrations came from P3 states whose hashes match 0/9 corrected P2
states, so the original “same exact states” premise was not achieved.

## Checks

### A. Ground Truth Provenance: WARN

P2, P3, and P4 terminal success use official LIBERO BDDL predicates through
`env.check_success()`. P3 controller target coordinates are taken from
previously successful P0 terminal states, and recovery labels are scripted
controller actions; both are disclosed proxies. The P2/P3 state mismatch
confounds the SFT failure's cause.

### B. Score Normalization: PASS

Success metrics are raw counts over valid rollouts. Training actions use the
checkpoint's immutable q01/q99 statistics. No metric is divided by the model's
own prediction statistics.

### C. Result Existence and Fidelity: WARN

All primary files, keys, counts, and checkpoint hashes exist and match.
Invalid float32-prefix output is excluded. One old report transcribed flush jerk
as 0.2828 although 0.282934585 rounds to 0.2829; this report now corrects it.

### D. Dead Code Detection: PASS

The P2, P3, and P4 summary functions are called and their outputs appear in
result files. Conversion, chunking, normalization, and scheduling helpers are
used and unit-tested.

### E. Scope Assessment: WARN

Evidence covers one init state, one seed, three cyclic task revisions, and
three offsets. The four checkpoints are correlated snapshots of one run. This
supports only the fixed-intervention failure claim.

### F. Evaluation Type: PASS

- P2/P3/P4 outcome evaluation: simulation-only with official-predicate GT.
- P3 recovery supervision: scripted synthetic proxy.
- Retention supervision: frozen-policy behavior distillation proxy.
- Human evaluation: none.

## Action Items

- Correct the jerk transcription (done).
- Record all nine P2/P3 hashes machine-readably (done).
- Annotate the preregistered state-identity assumption (done).
- Do not claim exact P2 reachability or general SFT impossibility.
- Before causal attribution, test offline imitation and exact-demo-state
  closed-loop recovery.

## Claim Impact

- 0/9 P2 recovery and 3/3 retention at all four checkpoints: supported.
- Fixed 200-step intervention failed its gate: supported.
- P3 oracle succeeded on its own nine states: supported.
- P3 oracle succeeded on the exact P2 states: unsupported and withdrawn.
- Supervised recovery cannot work: unsupported.

