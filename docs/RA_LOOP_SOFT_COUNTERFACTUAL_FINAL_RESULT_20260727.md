# RA-LOOP soft counterfactual final result — 2026-07-27

## Status

The final pre-registered soft-CRA trial completed successfully at the systems
level but did not pass the independent-effect Gate. The current RA-LOOP
counterfactual sparse-success RL line should stop rather than receive more
checkpoint, seed, training-length, or perturbation-strength searches.

## Training

Both four-task runs completed 51/51 steps and saved the preselected step-50
checkpoint.

| Seed | Runtime | Post-calibration steps | Nonzero soft-CRA steps | Parameter updates |
|---:|---:|---:|---:|---:|
| 10000 | 8:45:20 | 36 | 29 (80.6%) | 31 |
| 20000 | 8:45:38 | 36 | 26 (72.2%) | 32 |

For comparison, hard-pair CRA produced a nonzero CRA update on only 17/35
active steps (48.6%). Soft state-competence conditioning therefore fixed the
identified signal-sparsity mechanism, with no invalid pairs or runtime errors.
Online rollout rates are not used as independent effect evidence.

## Independent evaluation

All 40 candidate jobs completed, totaling 1,360 new episodes. Existing
warm-start results were reused from the exact same frozen task/init/seed
protocols.

### Fixed-L2 0.1 rad

Protocol: ten tasks, init indices `[6, 26)`, perturbation seed `20260801`.

| Model | Anchor | Perturbed | Perturbed delta |
|---|---:|---:|---:|
| warm-start | 195/200 (97.5%) | 189/200 (94.5%) | — |
| soft CRA+NPC, seed 10000 | 197/200 (98.5%) | 193/200 (96.5%) | +2.0 points |
| soft CRA+NPC, seed 20000 | 196/200 (98.0%) | 193/200 (96.5%) | +2.0 points |

Seed 10000 had 7 paired perturbation wins and 3 losses; seed 20000 had 8 wins
and 4 losses. Both paired-bootstrap confidence intervals crossed zero, and the
gain was below the pre-registered roughly 5-point Gate.

### Fixed-L2 0.2 rad

Protocol: ten tasks, init indices `[36, 50)`, perturbation seed `20260821`.
The strength was selected using a disjoint warm-start-only calibration set.

| Model | Anchor | Perturbed | Perturbed delta |
|---|---:|---:|---:|
| warm-start | 139/140 (99.3%) | 106/140 (75.7%) | — |
| soft CRA+NPC, seed 10000 | 136/140 (97.1%) | 108/140 (77.1%) | +1.4 points |
| soft CRA+NPC, seed 20000 | 140/140 (100.0%) | 102/140 (72.9%) | -2.9 points |

Seed 10000 had 6 perturbation wins and 4 losses but an anchor drop of 2.1
points. Seed 20000 had 5 perturbation wins and 9 losses. The mean perturbation
delta across the two training seeds was -0.7 points, and both confidence
intervals crossed zero.

## Decision

Soft CRA demonstrated that the engineering implementation can densify the
intended recovery signal while preserving the nominal constraint machinery.
It did not convert that mechanism improvement into a stable independent
recovery gain across seeds and intervention strengths.

The final Gate is therefore **not passed**. The evidence no longer supports
spending the remaining experiment budget on this sparse-success online-RL
objective. A future project should treat these results as a diagnostic baseline
and move to a materially different source of supervision, such as
failure-focused SFT or targeted data augmentation.
