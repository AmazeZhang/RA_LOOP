# CI-GRPO P0 experiment log — 2026-07-27

## Scope and decision discipline

This P0 validates feasibility only. It does not start CI-GRPO training and does
not claim that contrastive instruction rewards improve language grounding.
Every gate must be backed by a reproducible command and a saved artifact.

## P0 gates

| Task | Gate | Status |
|---|---|---|
| T0.1 | Same pixel/state `s0` can be paired with different BDDL goals | **pass** |
| T0.2 | SmolVLA-450M + LIBERO harness uses at most 24 GiB VRAM | **pass** |
| T0.3 | Custom goal reward agrees with official evaluator on 100 samples | **pass** |
| T0.4 | One K=3 reachable, initially-unsatisfied contrastive group exists | **pass** |
| T0.4b | Pairwise exclusion and early-trajectory branching checks work | **pass** |
| T0.5 | Method-side novelty survives focused literature audit | **pass, claim narrowed** |
| T0.6 | RIPT-VLA license is resolved or an independent fallback is selected | pending |
| T0.7 | Recorded go/no-go decision | **no-go for current premise/backbone/group** |

## Experiment P0-E01: same-state/different-goal probe

Candidate group:

1. `put_the_bowl_on_the_plate`
2. `put_the_bowl_on_the_stove`
3. `put_the_bowl_on_top_of_the_cabinet`

These tasks share the LIBERO-Goal kitchen scene and manipulate the same bowl
toward three spatially separated targets. The probe checks:

- equality of scene-defining BDDL fields;
- exact equality of restored MuJoCo state;
- exact equality of `agentview` and wrist-camera pixels;
- all three goals are false at the shared initial state;
- all three official goal predicates can be swapped and evaluated in one live
  environment without rebuilding its scene.

Command:

```bash
LIBERO_CONFIG_PATH="$PWD/.libero-official" \
NUMBA_CACHE_DIR=/tmp/ci_grpo_numba_cache \
conda run -n ript_vla_openvla_oft \
python ci_grpo/p0_state_goal_probe.py
```

Artifacts:

- `ci_grpo/artifacts/p0_same_state_k3/result.json`
- six camera renders under the same directory

Result: **pass**, using one live environment plus dynamically selected official
goal predicates.

- All three BDDL files have the same scene signature.
- The restored 79-dimensional MuJoCo states are exactly equal (`max_abs_delta=0`).
- In one live environment, changing the active goal leaves both camera arrays
  byte-identical (`different_values=0`, `max_abs_delta=0`).
- All three official goal predicates are false at the shared initial state.
- The three parsed official goal predicates can be evaluated in the same live
  environment without rebuilding it.

Diagnostic finding: constructing three separate environments and restoring the
same MuJoCo data state does **not** produce byte-identical pixels. Their physics
states are exact, but renderer/model construction retains differences outside
the flattened data state. CI-GRPO groups must therefore use one physical
environment instance (or one canonical XML/model) and switch only the
instruction and scoring predicate. Separate task environments are disallowed
for the same-pixel claim.

Machine-readable result:
`ci_grpo/artifacts/p0_same_state_k3/result.json`

Implementation corrections made before the successful run:

1. Official LIBERO requires its repository root on `sys.path` for the
   `libero.libero` namespace.
2. `NUMBA_CACHE_DIR` must point to writable `/tmp`; otherwise the read-only
   robosuite installation fails before simulator creation.

## Experiment P0-E02: OpenVLA K=3 rollout and terminal cross-scoring

Purpose: execute all three candidate instructions from the exact same state in
one live environment, then score every terminal state with every official goal
predicate. The probe also stores the first 32 normalized predicted actions and
reports pairwise DTW distances.

Dry-run and four focused unit tests passed. The first GPU launch exited before
model loading because the interpreter admitted an incompatible user-site
`peft/transformers` pair. No rollout was produced. The executable now removes
the user-site path before importing ML dependencies; launches must additionally
set `PYTHONNOUSERSITE=1`.

Result: **pass** for T0.4 and the empirical exclusion half of T0.4b.

- All three deterministic OpenVLA-Goal rollouts succeeded from the same
  `put_the_bowl_on_the_plate` init state at index 0.
- Cross-scoring each terminal state with all three official predicates produced
  a strict one-hot truth vector: own goal true, both alternatives false.
- First-32 predicted-action DTW mean/path-step:
  - plate vs stove: `0.08958`
  - plate vs cabinet: `0.17855`
  - stove vs cabinet: `0.17437`
- Peak GPU memory: `15.58 GiB` allocated, `15.82 GiB` reserved on RTX 4090 D.

The nonzero DTW result proves behavioral branching but does not yet define a
scientifically calibrated R5b threshold. T0.4b remains partial until
between-instruction distances are compared against within-instruction
seed/rollout variation.

Machine-readable result:
`ci_grpo/artifacts/p0_openvla_k3_rollout/result.json`

Operational findings:

1. The local Goal checkpoint initially lacked shard
   `model-00002-of-00004.safetensors`.
2. A seven-day stalled downloader (PID 1136845) held the lock with 4.09/4.95GB
   already present. After confirming zero CPU and no file progress, it was
   terminated with approval and the existing incomplete file was resumed.
3. The completed shard size is 4,947,392,496 bytes and SHA-256 is
   `4cd51d327e1891f24f75d033017ed3824cc5fb9009e38498efd2dca6f89ee01d`.

## Experiment P0-E03: 100-state reward consistency

The CI-GRPO scorer explicitly conjuncts the official predicate primitives,
while the reference path calls LIBERO's official `_check_success`. It samples
25 no-op-evolved states from each of four sources: the shared initial state and
the three successful terminal states. This yields 100 physical states and 300
state-goal comparisons, with positive and negative cases for every goal.

Result: **pass**.

- Physical states sampled: 100.
- State-goal comparisons: 300.
- Agreements: 300/300 (100%).
- Positive counts: plate 25, stove 25, cabinet 24.
- Disagreements: 0.

Machine-readable result:
`ci_grpo/artifacts/p0_reward_consistency_100/result.json`

The first invocation through `conda run` received SIGTERM before writing an
artifact. Re-running the identical command with the environment's absolute
Python binary completed normally; this is recorded as launcher behavior, not
an experimental disagreement.

## Experiment P0-E04: official SmolVLA-LIBERO harness and VRAM

Purpose: verify that the official LIBERO-specific SmolVLA checkpoint, LeRobot
environment adapter, and one complete rollout fit on one RTX 4090. This is an
infrastructure gate only; a single episode is not an efficacy estimate.

Environment:

- LeRobot checkout: `/home/imc/wangdi/lerobot_vla/lerobot` (0.4.4)
- Python environment: `/home/imc/anaconda3/envs/lerobot_vla`
- checkpoint: `/home/imc/models/ra-loop/smolvla-libero`
- official environment dependency: `hf-libero==0.1.4`
- GPU: physical GPU 7; GPU 0 was not used

Full-rollout command:

```bash
CUDA_VISIBLE_DEVICES=7 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
LIBERO_CONFIG_PATH="$PWD/.libero-official" \
NUMBA_CACHE_DIR=/tmp/ci_grpo_numba_cache \
MPLCONFIGDIR=/tmp/ci_grpo_mpl_cache \
/home/imc/anaconda3/envs/lerobot_vla/bin/lerobot-eval \
  --policy.path=/home/imc/models/ra-loop/smolvla-libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[3]' \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$PWD/ci_grpo/artifacts/p0_smolvla_libero_harness_run4"
```

Result: **pass** for T0.2.

- The official evaluator completed all 300 steps and exited normally.
- Evaluation time: 254.48 seconds.
- A separate 30-step probe observed 1,658 MiB process memory with
  `nvidia-smi` (1,681 MiB total device memory including the display/driver
  baseline), far below the 24 GiB gate.
- Both sampled episodes had zero task reward. This does not invalidate the
  harness/VRAM gate, but it means this one task cannot yet serve as a
  high-success SFT baseline.

Artifacts:

- `ci_grpo/artifacts/p0_smolvla_libero_harness_run4/eval_info.json`
- `ci_grpo/artifacts/p0_smolvla_libero_harness_run4/videos/libero_goal_3/eval_episode_0.mp4`
- `ci_grpo/artifacts/p0_smolvla_vram_probe_run5/eval_info.json`

Operational corrections:

1. The local cache originally contained only `lerobot/smolvla_base`, whose
   SO100 camera/state/action schema is incompatible with LIBERO. The dedicated
   `HuggingFaceVLA/smolvla_libero` checkpoint was selected instead.
2. `hf-libero` initially failed to build its EGL probes because pip's isolated
   build could not execute CMake. Building `hf-egl-probe==1.0.2` and
   `egl-probe==1.0.2` with `--no-build-isolation` resolved it.
3. The environment exposed an empty `cffi` namespace. Installing
   `cffi==1.17.1` restored `cffi.FFI` and allowed Numba/robosuite initialization.

## Experiment P0-E05: multi-state language sensitivity and DTW calibration

This experiment is pre-registered before execution:

- init-state indices: 0, 1, 2, 3, 4;
- three same-scene conflicting instructions per state (15 rollouts total);
- first 32 normalized predicted actions for DTW;
- `LSG-hit` is the diagonal goal hit rate;
- `LSG-miss` is the off-diagonal goal hit rate;
- `LSG = LSG-hit - LSG-miss`;
- instruction-swap redirection is the rate at which the two non-source
  instructions reach their own requested goal;
- the R5b DTW threshold is
  `max(0.01, 5 × maximum same-instruction repeat DTW)`;
- the existing independently launched init-index-0 result is the repeat-noise
  reference;
- per the existing F1 stop rule, `LSG > 15pp` together with redirection above
  35% rejects the proposed baseline-language-deafness premise on this group.

Execution command:

```bash
PYTHONNOUSERSITE=1 \
/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python \
  ci_grpo/p0_openvla_k3_rollout.py \
  --execute \
  --gpu-id 7 \
  --init-index 0 \
  --num-init-states 5 \
  --output-dir ci_grpo/artifacts/p0_openvla_k3_multistate
```

Result: completed normally in 10 minutes 10 seconds. **The group-construction
gate passes, but the baseline-language-deafness premise fails decisively.**

- 15/15 correct-instruction rollouts reached their requested goal.
- All 30 off-diagonal terminal goal checks were false.
- `LSG-hit = 100%`, `LSG-miss = 0%`, and `LSG = +100pp`.
- Instruction-swap redirection rate was 100%.
- Every terminal truth vector was strictly one-hot, confirming empirical
  pairwise goal exclusion across all five initial states.
- Same-instruction repeat DTW was exactly 0 for all three independently
  relaunched index-0 rollouts.
- The calibrated R5b threshold was therefore `max(0.01, 5×0) = 0.01`.
- All 15 same-state between-instruction pairs passed. DTW mean/path-step ranged
  from `0.08958` to `0.24499` (median `0.15439`).
- Peak GPU memory was 15.58 GiB allocated and 15.82 GiB reserved on physical
  GPU 7. Physical GPU 0 was not used.

Machine-readable result:
`ci_grpo/artifacts/p0_openvla_k3_multistate/result.json`

Interpretation: OpenVLA-OFT-Goal is already perfectly language-sensitive on
this carefully controlled K=3 group. It does not exhibit the required
high-hit/high-miss “language deafness” scissors pattern. The result exceeds the
pre-existing F1 stop threshold (`LSG > 15pp`) by a wide margin and also has
perfect redirection, so it is not a borderline statistical decision.

## Experiment P0-E06: focused novelty audit

Audit date: 2026-07-27. Primary paper pages and current revisions were checked,
including work published after the original CI-GRPO specification.

Result: **pass with a materially narrowed claim**.

| Work | Closest overlap | Remaining distinction from CI-GRPO |
|---|---|---|
| [LangForce (arXiv:2601.15197v7)](https://arxiv.org/abs/2601.15197) | Diagnoses information collapse and maximizes language/action dependence | Dual-branch PMI/LLR objective; no same-pixel conflicting-goal reward groups |
| [CAST (arXiv:2508.13446v2)](https://arxiv.org/abs/2508.13446) | Counterfactual labels increase semantic diversity for similar observations | Synthetic language/action supervision rather than label-free simulator reward and on-policy action updates |
| [IGAR/ICBench (arXiv:2603.06001v2)](https://arxiv.org/abs/2603.06001) | Same-scene contradictory instructions and OpenVLA-OFT language-blindness diagnosis on LIBERO | Train-free attention recalibration for impossible/OOD contradictions, not contrastive RL over reachable mutually exclusive goals |
| [VLA Grounder (arXiv:2607.04517)](https://arxiv.org/abs/2607.04517) | Uses GRPO groups and sparse rollout reward to improve VLA conditioning | Optimizes an upstream command-rewriting policy over same-intent candidate commands while freezing the VLA; CI-GRPO groups different valid goals at identical pixels and updates the VLA action policy |

The novelty claim must therefore **not** be “first GRPO for VLA language
grounding,” “first counterfactual language training,” or “first same-scene
contradiction benchmark.” The defensible method claim is the conjunction:

> group-relative on-policy action-model optimization where each group fixes the
> exact visual/physical state but varies reachable, initially false, mutually
> exclusive instruction-goal pairs, using simulator truth to make language
> causally necessary for reward.

This distinction survives the focused audit. VLA Grounder becomes a required
concurrent baseline or discussion point because it is the closest GRPO-based
method, despite optimizing a different policy and grouping variable.

## P0 decision

Decision: **do not start CI-GRPO training on the current OpenVLA-OFT-Goal
backbone and this bowl/three-target contrastive group.**

The infrastructure and data-construction hypotheses pass:

- identical pixels/state with swappable goals;
- valid, reachable, mutually exclusive K=3 group;
- calibrated early action branching;
- exact reward/evaluator agreement;
- both OpenVLA and SmolVLA harnesses fit one RTX 4090.

The scientific treatment premise does not pass: the selected OpenVLA baseline
already maps each instruction to the correct distinct behavior with
`LSG=+100pp` and 100% redirection. CI-GRPO has no measurable language-grounding
headroom on this sample, so training here could only preserve performance or
create regressions; it cannot support the intended “repair language deafness”
claim.

This is a scoped no-go, not evidence that all VLAs are language-sensitive.
Before any new training, the project must choose a different falsifiable
problem setting: for example, search a broader set for high-hit/high-miss
groups, target OOD/impossible contradictions as in ICBench, or pivot from
“repair” to a controlled benchmark/diagnostic contribution. That choice is
deferred to the post-P0 discussion rather than made by spending more GPU.

## Experiment P0-E07: bounded all-Goal language-deafness screen

Start date: 2026-07-30.

Purpose: perform one final bounded search before abandoning the training claim.
All ten official LIBERO-Goal instructions share the exact static scene
signature, so the screen fixes three initial states from one official state
bank and executes every instruction from each state. This yields 30 rollouts
and a 10×10 terminal-goal cross-score matrix per state.

Pre-registered decision rules:

- a useful baseline candidate must have correct-instruction success
  (`LSG-hit`) at least 50%;
- it must simultaneously have `LSG <= 15pp`, meaning a sufficiently high
  mismatched-instruction goal rate to support the language-deafness premise;
- compatible goal pairs that can be simultaneously true are diagnostics only
  and cannot be promoted to CI-GRPO mutually exclusive groups;
- if the broad screen remains high-LSG, or low-LSG appears only because hit is
  below 50% or goals are compatible, the original repair/training direction
  stops;
- no training or hyperparameter search is authorized by this screen.

Command:

```bash
PYTHONNOUSERSITE=1 \
/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python -u \
  ci_grpo/p0_openvla_k3_rollout.py \
  --execute \
  --all-goal-tasks \
  --source-task put_the_bowl_on_the_plate \
  --init-index 0 \
  --num-init-states 3 \
  --gpu-id 7 \
  --output-dir ci_grpo/artifacts/p0_openvla_goal10_state3_screen
```

Operational constraint: physical GPU 0 is excluded. The run uses physical GPU
7 and is logged in `ci_grpo_goal10_screen.log`.

OpenVLA result: **no candidate; stop condition triggered for this backbone.**

- All 30/30 correct-instruction rollouts succeeded.
- All 270/270 off-diagonal terminal goal checks were false.
- Every terminal goal vector was strictly one-hot.
- Across all ten tasks: `hit=100%`, `miss=0%`, `LSG=+100pp`.
- Per target, every one of the ten tasks independently had
  `hit=100%`, `miss=0%`, `LSG=+100pp`.
- Known mutually exclusive bowl K=3 and wine-bottle K=2 groups both had
  `hit=100%`, `miss=0%`, `LSG=+100pp`.
- No raw target candidate and no valid exclusive-group candidate passed the
  high-hit/low-LSG screen.
- Peak GPU memory was 15.59 GiB allocated and 15.82 GiB reserved.

Artifacts:

- `ci_grpo/artifacts/p0_openvla_goal10_state3_screen/result.json`
- `ci_grpo/artifacts/p0_openvla_goal10_state3_screen/summary.json`

DTW caveat: the three init-index-0 bowl trajectories differed from the earlier
launch by `0.0599–0.1288` mean/path-step when three other tasks preceded them.
The resulting five-times-noise threshold (`0.6439`) rejected 45 of 135
between-instruction pairs. This indicates sequence/order-dependent rollout
variation and makes this run unsuitable for strengthening the R5b claim.
Terminal goal truth is unaffected and supplies the decision signal here.

SmolVLA follow-up initially used task IDs `[3,4,5,7,8]` under an incorrect
task-ID mapping. The run itself is valid as a general five-task capability
diagnostic, but it is **not** an exclusive-group screen and cannot support an
exclusive-group conclusion. It obtained 3/5 successes (tasks 4, 5, and 7) in
752.26 seconds. The task mapping was checked before the same-state experiment:

- bowl destination K=3: task IDs `[1,4,8]` (stove, cabinet, plate);
- wine destination K=2: task IDs `[2,9]` (cabinet, rack).

The incorrectly labelled diagnostic command was:

```bash
CUDA_VISIBLE_DEVICES=7 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
LIBERO_CONFIG_PATH="$PWD/.libero-official" \
NUMBA_CACHE_DIR=/tmp/ci_grpo_numba_cache \
MPLCONFIGDIR=/tmp/ci_grpo_mpl_cache \
/home/imc/anaconda3/envs/lerobot_vla/bin/lerobot-eval \
  --policy.path=/home/imc/models/ra-loop/smolvla-libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[3,4,5,7,8]' \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$PWD/ci_grpo/artifacts/p0_smolvla_exclusive5_hit_screen"
```

Artifact:
`ci_grpo/artifacts/p0_smolvla_exclusive5_hit_screen/eval_info.json`

### SmolVLA same-state screen with corrected groups

The corrected screen uses one live environment from task 8, restores the same
official init state for every rollout, swaps only the instruction and active
scoring goal, and resets the policy/environment RNG to seed 1000 for every
matrix cell. Diagonal cells are executed first. A group stops as soon as it is
mathematically impossible to satisfy both `hit >= 50%` and `LSG <= 15pp`.

Two earlier launch directories are not evidence: the first launch was stopped
after detecting that RNG state was not reset per cell; the second was stopped
after detecting the incorrect task-ID grouping. Both remain only as incomplete
operational diagnostics. The corrected run is:

```bash
cd /home/imc/wangdi/lerobot_vla/lerobot
CUDA_VISIBLE_DEVICES=7 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
LIBERO_CONFIG_PATH=/home/imc/yzy/RA_LOOP/.libero-official \
NUMBA_CACHE_DIR=/tmp/ci_grpo_numba_cache \
MPLCONFIGDIR=/tmp/ci_grpo_mpl_cache \
/home/imc/anaconda3/envs/lerobot_vla/bin/python -u \
  /home/imc/yzy/RA_LOOP/ci_grpo/p0_smolvla_same_state_screen.py \
  --execute \
  --gpu-id 7 \
  --source-task-id 8 \
  --init-index 0 \
  --seed 1000 \
  --output-dir \
  /home/imc/yzy/RA_LOOP/ci_grpo/artifacts/p0_smolvla_same_state_screen_run3
```

Result: **no candidate; stop condition triggered for SmolVLA.**

- Bowl K=3 diagonal hit was 2/3: cabinet and plate succeeded; stove failed.
- Its first three mismatches all failed. Even if all three unexecuted
  mismatches had succeeded, `miss` could be at most 50%, so the minimum
  possible `LSG` was 16.67pp. The group therefore stopped early because the
  15pp candidate threshold had become mathematically unreachable.
- Wine K=2 diagonal hit was 1/2: cabinet succeeded and rack failed.
- Both wine mismatches failed, giving `hit=50%`, `miss=0%`, and `LSG=+50pp`.
- `valid_exclusive_group_candidates=[]` and
  `continue_training_direction=false`.
- Peak GPU memory was 2.10 GiB allocated and 2.13 GiB reserved on physical
  GPU 7. Physical GPU 0 was not used.

Machine-readable result:
`ci_grpo/artifacts/p0_smolvla_same_state_screen_run3/result.json`

## Final bounded-screen decision

The original CI-GRPO treatment claim is now a hard **no-go** for the tested
in-distribution LIBERO-Goal setting. OpenVLA was perfectly instruction
sensitive over all 10 tasks and 3 shared states. SmolVLA had lower task ability,
but its successful behavior did not show the required high-hit/high-miss
language-deafness scissors pattern in either valid mutually exclusive group.

No CI-GRPO training or additional group search should be launched under this
premise. Any subsequent work is a research pivot, not another tuning round:
retain the same-state harness as a benchmark/diagnostic, study genuinely
OOD/impossible contradictions, or formulate a different failure mode with a
new pre-registered treatment hypothesis.
