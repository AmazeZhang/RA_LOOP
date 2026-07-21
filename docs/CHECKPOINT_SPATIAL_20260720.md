# OpenVLA-OFT libero_spatial Checkpoint Verification — 20260720

Verification completed: **2026-07-20 09:13:55 CST (UTC+08:00)**

Repository:
`moojink/openvla-7b-oft-finetuned-libero-spatial`

Local directory:
`/home/imc/models/ra-loop/openvla-oft-spatial`

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
- The only normalization key is the expected `libero_spatial_no_noops`.
- Remaining filesystem space after download: approximately 191 GB.

## Component sizes

```text
model-00001-of-00004.safetensors       4,925,122,448 bytes
model-00002-of-00004.safetensors       4,947,392,496 bytes
model-00003-of-00004.safetensors       4,947,417,456 bytes
model-00004-of-00004.safetensors         262,668,432 bytes
lora_adapter/adapter_model.safetensors   484,458,600 bytes
action_head--150000_checkpoint.pt        302,242,674 bytes
proprio_projector--150000_checkpoint.pt   67,275,256 bytes
```

## SHA-256

```text
2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251  model-00001-of-00004.safetensors
a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3  model-00002-of-00004.safetensors
a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a  model-00003-of-00004.safetensors
a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60  model-00004-of-00004.safetensors
4bd2e808805f9b67af090c37e70f239b3b4da7a6473ec621ead6702b16a302ec  lora_adapter/adapter_model.safetensors
809858636cf0a65009dd567d2f4e116249442790f02b8fe31f24500ea6118908  action_head--150000_checkpoint.pt
438d28e81e125166d0771762424aaf017de3a8daaebde06a5cb71157e62b3bf3  proprio_projector--150000_checkpoint.pt
f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa  modeling_prismatic.py
68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68  configuration_prismatic.py
```

These hashes match the immutable Hugging Face cache object identifiers recorded
during download for the corresponding files.

## Next gate

The bounded CPU preflight completed successfully at 2026-07-20 09:17:40 CST.
`eval/launch_oft_spatial_robot350_tmux.sh` verified 350/350 official spatial
Robot-init tasks in seven non-overlapping 50-task manifests, the suite-specific
checkpoint path, OSMesa, seed 7, and physical GPU mapping 1--7 while CUDA was
hidden. It created no output directory or tmux session and left no evaluator
process. GPU evaluation still requires explicit user confirmation.
