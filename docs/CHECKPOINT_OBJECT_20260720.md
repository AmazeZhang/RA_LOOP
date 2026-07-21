# OpenVLA-OFT libero_object Checkpoint Verification — 20260720

Verification completed: **2026-07-20 13:42:19 CST (UTC+08:00)**

Repository:
`moojink/openvla-7b-oft-finetuned-libero-object`

Local directory:
`/home/imc/models/ra-loop/openvla-oft-object`

## Result

CPU-only static verification passed. The checkpoint was not loaded for inference,
no evaluator or training process was started, and no GPU was used.

- Hugging Face download completed: 25/25 files.
- Exactly 25 repository files are present when `.cache` is excluded.
- No `.incomplete` files remain.
- Total directory size: approximately 15 GB.
- `model.safetensors.index.json` maps 982 tensors to four shards; all four exist.
- The four shards contain `760 + 109 + 112 + 1 = 982` readable tensor entries.
- The LoRA adapter opens successfully with 879 tensor entries.
- Action head and proprio projector both parse successfully with PyTorch on CPU.
- The only normalization key is the expected `libero_object_no_noops`.
- Remaining filesystem space after verification: approximately 296 GB.

## Component sizes

```text
model-00001-of-00004.safetensors       4,925,122,448 bytes
model-00002-of-00004.safetensors       4,947,392,496 bytes
model-00003-of-00004.safetensors       4,947,417,456 bytes
model-00004-of-00004.safetensors         262,668,432 bytes
lora_adapter/adapter_model.safetensors   484,458,600 bytes
```

## SHA-256

```text
fedf457ad486f5e4ce28579a85e7b3ae1448d01884a674e9baaf7fc47125401c  model-00001-of-00004.safetensors
cb9e4eb3991dfdb7d111e332d7d565543cabc3bc63bfc8ebba83695ad05dcaf3  model-00002-of-00004.safetensors
f28ce17430e501109e4f512373bea70bc77d4cfe66c17a1a120a04a5a5623f8d  model-00003-of-00004.safetensors
a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60  model-00004-of-00004.safetensors
7e8c81b0960747337141c666f39f607dca0ba7581933fb94b232d96e5dc6d7d8  lora_adapter/adapter_model.safetensors
8465509d85b063f3c90359de0c1876c34ce3f162cc9baafbe9e75e170146e4c7  action_head--150000_checkpoint.pt
4ae5121c2740a57f5747061e938246f9e3994bbdbfb97676ffef795221e6f8d9  proprio_projector--150000_checkpoint.pt
f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa  modeling_prismatic.py
68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68  configuration_prismatic.py
```

## Next gate

The bounded CPU preflight completed successfully at 2026-07-20 13:45:18 CST.
`eval/launch_oft_object_robot398_tmux.sh` verified 398/398 official object
Robot-init tasks in seven non-overlapping manifests (`57 x 6 + 56`), the
suite-specific checkpoint path, OSMesa, seed 7, and physical GPU mapping 1--7
while CUDA was hidden. It created no output directory or tmux session and left
no evaluator process. GPU evaluation still requires explicit user confirmation.
