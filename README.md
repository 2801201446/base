# nnUNet MG-SSM Baseline

This project is an nnU-Net v2.1.2 baseline with the low-frequency module from
[FMCNet](https://github.com/anaanaa/FMCNet) inserted after every **encoder**
stage.

## Architecture contract

- Backbone: standard nnU-Net `PlainConvUNet`.
- Encoder: native strided-convolution downsampling, followed by one MG-SSM at
  every stage (including the bottleneck).
- Decoder: unchanged nnU-Net convolutional decoder with the original skip
  connections and deep supervision.
- Frequency operations: none. There is no DWT/IDWT, frequency split, HFR, or
  high-frequency branch. MG-SSM receives the complete spatial feature map.
- Dimensionality: 3-D nnU-Net configurations only, matching FMCNet's `Conv3d`
  implementation.

MG-SSM retains FMCNet's stage-dependent token rule: spatial tokens are used
while the number of voxels is greater than the number of channels; deeper
stages switch to channel tokens when `D * H * W <= C`.

## Environment

Use the existing compatible environment, or create a separate Python 3.10
environment with PyTorch/CUDA first. FMCNet's module targets:

```bash
conda activate lkmunet
cd /home/dministrator/projects/nnUNet_MGSSM_baseline
pip install -e .
```

The baseline pins `mamba-ssm==1.2.0.post1`, the version already available in
the `lkmunet` environment. Installing this project editable makes its
`nnunetv2` package the active nnU-Net checkout in that environment.

Set the standard nnU-Net paths (adjust them if a different dataset root is
desired):

```bash
export nnUNet_raw=/home/dministrator/projects/LKM-UNet_fresh_20260804/data/nnUNet_raw
export nnUNet_preprocessed=/home/dministrator/projects/LKM-UNet_fresh_20260804/data/nnUNet_preprocessed
export nnUNet_results=/home/dministrator/projects/nnUNet_MGSSM_baseline/data/nnUNet_results
```

## Train

Planning and preprocessing remain standard nnU-Net operations:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerMGSSMBaseline
```

Train folds `0` through `4` in the same way as a normal nnU-Net experiment.
Inference and validation automatically reconstruct this architecture through
the saved trainer name.

## Verification

Run the small CUDA smoke test:

```bash
cd /home/dministrator/projects/nnUNet_MGSSM_baseline
PYTHONPATH="$PWD" python tests/test_mgssm_baseline.py
```
