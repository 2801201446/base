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

The project uses its own Conda environment. It does not import nnU-Net from the
LKM-UNet checkout:

```bash
conda activate nnunet_mgssm
cd /home/dministrator/projects/nnUNet_MGSSM_baseline
pip install -e .
```

The dedicated environment contains Python 3.10, PyTorch 2.0.1+cu118,
`mamba-ssm==1.2.0.post1`, and `einops==0.7.0`. The editable installation must
resolve `nnunetv2` to this project directory.

Set the standard nnU-Net paths (adjust them if a different dataset root is
desired):

```bash
source /home/dministrator/projects/nnUNet_MGSSM_baseline/scripts/set_nnunet_paths.sh
```

## Train

The existing PARSE2022 raw and preprocessed data are available as Dataset502:

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerMGSSMBaseline
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
