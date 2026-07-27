# Counterfactual held-out evaluation protocol — 2026-07-25

## Purpose

Decide whether the completed 4-task CRA/NPC pilot has a reproducible independent
recovery benefit.  This is an exploratory gate, not a claim that the method has
already succeeded.

## Frozen comparison

| Candidate | Checkpoint |
|---|---:|
| warm-start baseline | pilot step 5 |
| A4 CRA-only | counterfactual_cra_only step 40 |
| A5 CRA+NPC | counterfactual_npc step 40 |

All candidates use exactly the same task, init-state, perturbation, and action
protocol.  No alternate checkpoint is selected after seeing these results.

## Episodes

- LIBERO-Spatial's ten tasks: the four training tasks reported as **seen**, the
  other six as **unseen**.
- Twenty paired benchmark init states per task: indices `[6, 26)`; these are
  disjoint from the prior narrow evaluation's indices `0..5`.
- One anchor and one fixed-L2 `0.1 rad` Robot-init perturbation per pair.
- Perturbation seed `20260801`, distinct from the training seed `20260720` and
  prior narrow-evaluation seeds.
- Three models × ten tasks × twenty pairs × two modes = **1,200 episodes**.

## Outcomes and decision rule

Primary outcome: paired-model difference in fixed-L2 success rate relative to
warm-start, pooled over the 200 held-out pairs.  Secondary outcomes: anchor
success, CRR/RG, and the same quantities split by seen/unseen task.

For each candidate versus baseline, report per-init paired wins, losses, and
ties; a paired bootstrap confidence interval for the fixed-L2 difference; and
all ten task-level results.  Do not substitute online training success for these
outcomes.

Advance to multi-seed training only if a candidate shows roughly >=5 percentage
points fixed-L2 improvement, <=2 points anchor decline, and a positive paired
gain pattern rather than an equal exchange of failures.  Otherwise, redesign
the sparse hard-pair CRA signal before running longer training.
