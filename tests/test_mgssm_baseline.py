"""Small GPU smoke test and structural contract for the baseline network."""

import torch
from torch import nn

from nnunetv2.nets.nnunet_mgssm import nnUNetMGSSM


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("mamba-ssm smoke test requires CUDA")

    model = nnUNetMGSSM(
        input_size=(8, 16, 16),
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * 3,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
        n_conv_per_stage=(2, 2, 2),
        num_classes=3,
        n_conv_per_stage_decoder=(2, 2),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    ).cuda()

    assert len(model.encoder.mgssm_layers) == len(model.encoder.stages) == 3
    assert model.encoder.channel_token_stages == [False, False, True]
    assert not hasattr(model.decoder, "mgssm_layers")

    model.eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 1, 8, 16, 16, device="cuda"))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 3, 8, 16, 16),
        (1, 3, 4, 8, 8),
    ]

    model.train()
    train_outputs = model(torch.randn(1, 1, 8, 16, 16, device="cuda"))
    sum(output.square().mean() for output in train_outputs).backward()
    parameters_without_grad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert parameters_without_grad == [], parameters_without_grad

    print("MG-SSM baseline smoke test passed")
    print("channel-token stages:", model.encoder.channel_token_stages)
    print("output shapes:", [tuple(output.shape) for output in outputs])
    print("all trainable parameters received gradients")


if __name__ == "__main__":
    main()
