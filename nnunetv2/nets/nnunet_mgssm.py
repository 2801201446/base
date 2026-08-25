"""Standard nnU-Net with FMCNet MG-SSM after every encoder stage."""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.nets.mgssm import MGSSMLayer
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class MGSSMPlainConvEncoder(PlainConvEncoder):
    """nnU-Net PlainConvEncoder followed by one MG-SSM per stage."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
    ) -> None:
        if convert_conv_op_to_dim(conv_op) != 3:
            raise NotImplementedError(
                "FMCNet MG-SSM is a 3-D module; use a 3d_* nnU-Net configuration."
            )
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=return_skips,
            nonlin_first=nonlin_first,
            pool="conv",
        )

        feature_shapes = []
        current_shape = [int(i) for i in input_size]
        for stride in self.strides:
            current_shape = [
                size // int(step) for size, step in zip(current_shape, stride)
            ]
            feature_shapes.append(tuple(current_shape))

        self.mgssm_layers = nn.ModuleList()
        self.channel_token_stages = []
        for channels, shape in zip(self.output_channels, feature_shapes):
            voxel_count = int(np.prod(shape))
            channel_tokens = voxel_count <= channels
            d_model = voxel_count if channel_tokens else channels
            self.mgssm_layers.append(
                MGSSMLayer(
                    channels=channels,
                    d_model=d_model,
                    channel_tokens=channel_tokens,
                )
            )
            self.channel_token_stages.append(channel_tokens)
        self.feature_shapes = feature_shapes

    def forward(self, x):
        skips = []
        for stage, mgssm in zip(self.stages, self.mgssm_layers):
            x = mgssm(stage(x))
            skips.append(x)
        return skips if self.return_skips else skips[-1]


class nnUNetMGSSM(nn.Module):
    """PlainConvUNet backbone with MG-SSM only in the encoder."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MGSSMPlainConvEncoder(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        if len(input_size) != convert_conv_op_to_dim(self.encoder.conv_op):
            raise ValueError("input_size must contain spatial dimensions only")
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)


def get_nnunet_mgssm_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
) -> nnUNetMGSSM:
    """Build the baseline from an ordinary nnU-Net v2 plan."""

    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    if dim != 3:
        raise NotImplementedError(
            "nnUNet_MGSSM_baseline supports 3-D configurations only."
        )
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = nnUNetMGSSM(
        input_size=tuple(configuration_manager.patch_size),
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[
            min(
                configuration_manager.UNet_base_num_features * 2**stage,
                configuration_manager.unet_max_num_features,
            )
            for stage in range(num_stages)
        ],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(conv_op),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.apply(InitWeights_He(1e-2))
    return model

