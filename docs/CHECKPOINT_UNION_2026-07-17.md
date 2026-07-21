# OpenVLA-OFT Union Checkpoint Verification

> **Identity note (2026-07-20):** LIBERO-Plus names this four-suite combined
> checkpoint **OpenVLA-OFT_m (mix-SFT)**. Its reported Robot score is 21.7%; the
> 31.9% OpenVLA-OFT result uses four suite-specific checkpoints. See
> `docs/BASELINE_OFTM_ROBOT1550_2026-07-20.md`.

Verification date: 2026-07-17 (Asia/Shanghai)

Repository:
`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

Local directory:
`/home/imc/models/ra-loop/openvla-oft-union`

## Static verification

- Hugging Face download completed: 25/25 files.
- No `.incomplete` model files remained after download.
- Total directory size: approximately 15 GB.
- `model.safetensors.index.json` references four shards and all four exist.
- All model shards and the LoRA adapter can be opened with `safetensors.safe_open` on CPU.
- Required normalization keys are present:
  - `libero_spatial_no_noops`
  - `libero_object_no_noops`
  - `libero_goal_no_noops`
  - `libero_10_no_noops`
- Required action head and proprio projector checkpoints are present.

## SHA-256

```text
8b30a7951e68703e1731957190d9f1d6e1eaa82f05b53909608eb0510875b11d  model-00001-of-00004.safetensors
af4773166950ddda1da6b3c5367796a52e7d5e7216f97041d9ad0721a25e53fe  model-00002-of-00004.safetensors
5a62791dd46ec4a85b353920aa07eff3582ceb5e9b6b853ad0f38de36fcab75d  model-00003-of-00004.safetensors
a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60  model-00004-of-00004.safetensors
f9647226ca9a1ee64ff8ed1ec380c89110afc6fc2bf6d0ae463cc4023165b4dc  action_head--300000_checkpoint.pt
1792c6e19381d3c7c814e2d49836a855d3a38b0693919416f52b3c5af39505c8  proprio_projector--300000_checkpoint.pt
2e895d13475d45fa79b9d8a71c952b526fa1ba23dfe15c73b8c0518351d660e7  lora_adapter/adapter_model.safetensors
f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa  modeling_prismatic.py
68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68  configuration_prismatic.py
```

## Known model-logic difference

The checkpoint's `modeling_prismatic.py` differs from the current local
OpenVLA-OFT tree by exactly two lines that initialize the diffusion scheduler.
The bounded evaluator fixes `use_diffusion=False` and uses the L1 regression
head, so this branch is unreachable. The evaluator accepts only this exact
difference and rejects any other mismatch.

The bounded evaluator disables OpenVLA's automatic checkpoint rewrite after
the read-only preflight passes. No checkpoint file is modified during model
loading.
