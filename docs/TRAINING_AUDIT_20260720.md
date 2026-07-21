# RIPT-VLA / RA-LOOP Training Readiness Audit — 20260720

Audit completed: **2026-07-20**  
Scope: CPU-only inspection and import/config tests. No GPU training was started,
no checkpoint was modified, and the clean upstream RIPT-VLA tree was not patched.

## Outcome

The evaluation environment is healthy, but the current training launchers are
**not ready to execute safely**. Running `train/vanilla_loop.sh` or
`train/ra_loop.sh` now would fail before meaningful training or would use the
wrong experimental initialization. The blockers are concrete and repairable.

## Verified healthy components

- RIPT-VLA upstream commit: `440990e8864e12e4578b490ff6359e4f2c49ae3e`.
- The upstream RIPT-VLA worktree is clean.
- Python 3.10.20 and the dedicated `ript_vla_openvla_oft` environment import:
  PyTorch 2.2.0+cu121, Hydra 1.3.2, OmegaConf 2.3.1, W&B 0.18.3,
  PEFT 0.11.1, draccus 0.3.1, h5py 3.11.0, LIBERO, and RIPT-VLA.
- CPU isolation test reported `torch.cuda.is_available() == False`.
- `code/ra_optimizer.py` and the upstream training entry point are syntactically
  valid Python.
- The suite-specific spatial checkpoint and its 350-task pre-training robustness
  baseline are already available for the future RA-LOOP pilot.

## Blocking issues

### 1. Required LIBERO HDF5 demonstrations are absent

RIPT-VLA's `build_dataset()` opens one original LIBERO `.hdf5` demonstration file
per task below `${data_prefix}/libero/...`. No `.hdf5` files were found in the
configured data locations, and `/home/imc/data/ra-loop/libero-datasets` does not
exist.

The existing `setup/step04_data.sh` incorrectly proposes
`openvla/modified_libero_rlds` as the RIPT data prefix. RLDS is not a drop-in
replacement for the HDF5 loader used by this RIPT commit. The official LIBERO
HDF5 dataset must be downloaded and validated instead.

### 2. RIPT Laplace scale headers are absent

OpenVLA-OFT RIPT constructs a Laplace scale head and requires the suite's SFT
scale-header checkpoint from `tanshh97/RIPT_VLA`. The local scale-header directory
is empty apart from a zero-byte interrupted cache object. A clean spatial smoke
needs `LIBERO_SPATIAL_scale_header.pth`.

When training from the SFT OpenVLA-OFT checkpoint, official RIPT instructions say
`lora_adaptor_ckpt=null`. Loading the released RIPT adaptor would continue from an
already post-trained model and invalidate the vanilla-from-SFT control.

### 3. Path configuration is missing or points to another machine

- Workspace `config/paths.yaml` is absent, so `train_ra_loop` Hydra composition
  fails with `MissingConfigException: Could not load 'paths'`.
- Upstream `/home/imc/code/ript-vla/config/paths.yaml` points to `/storage/...`,
  which does not exist on this machine.
- `train/vanilla_loop.sh` parses a workspace paths file but launches the upstream
  Hydra config without overriding `data_prefix` and `output_prefix`.

### 4. Current vanilla launcher does not define the desired control

`train/vanilla_loop.sh` is hard-coded to LIBERO-Long and loads a released RIPT
LoRA adaptor. Our first controlled pilot should use LIBERO-Spatial, start from the
verified SFT spatial checkpoint, load only the SFT scale header, and set the RIPT
adaptor to `null`.

### 5. RA module cannot currently be imported

The Hydra target is `code.ra_optimizer...`, but `code/` has no `__init__.py` and
collides with Python's standard-library `code` module. The verified failure is:

```text
ModuleNotFoundError: No module named 'code.ra_optimizer'; 'code' is not a package
```

The RA implementation should move to a uniquely named package before use.

### 6. The proposed upstream patch does not match this RIPT commit

`setup/step06_patch_ript.sh` searches for direct constructors named
`RLOptimizerOpenVLAOFT(...)` and `RolloutGenerator(...)`. Both patterns are absent
from the current entry point, which uses Hydra factories. Running the patch would
create an external `.bak` and then fail. It was deliberately not executed.

The safer integration is to provide Hydra `_target_` factories/configuration from
the workspace and leave the clean upstream repository unchanged.

### 7. Current RA perturbation semantics are not yet valid

- `camera` and `light` perturbations are documented but are no-ops in the current
  rollout implementation; they can still be marked perturbed and receive recovery
  bonus.
- Group 0 can receive a robot-init perturbation but is labelled as the unperturbed
  anchor (`is_perturbed=False`).
- Consistency is calculated within each perturbation group, not across different
  perturbations, so it does not yet implement the stated cross-perturbation
  hypothesis.
- The assumed raw LIBERO state-vector layout for robot/layout noise has not been
  validated against the actual HDF5 and simulator state schema.

These issues must be fixed and unit-tested before any RA-LOOP GPU run.

## Safe next sequence

1. Download only the official LIBERO HDF5 subset required for a spatial smoke and
   verify task files and state tensors.
2. Download and CPU-validate `LIBERO_SPATIAL_scale_header.pth`.
3. Create a workspace `config/paths.yaml` with real local paths and disabled W&B.
4. Build a new bounded spatial vanilla launcher with:
   one GPU, one task, `n_steps=1`, small K/batch, isolated timestamped output,
   SFT checkpoint + SFT scale header, and `lora_adaptor_ckpt=null`.
5. Hydra-compose and dataset-open dry-run with CUDA hidden; create no training
   output during this validation.
6. Only after explicit confirmation, run the single-GPU vanilla smoke.
7. Refactor and unit-test RA perturbation/reward semantics before the first
   RA-LOOP smoke.

The full four-suite Robot-init baseline can continue independently; it is no
longer a prerequisite for starting this bounded training-system validation.
