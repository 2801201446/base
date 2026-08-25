"""Multi-granularity state-space module adapted from FMCNet's low-frequency branch.

This file intentionally contains no wavelet transform, frequency decomposition,
or high-frequency refinement code. The input is the complete spatial feature map.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # mamba-ssm also supports the PyTorch convolution fallback
    causal_conv1d_fn = None


class CrossGatedMamba(nn.Module):
    """Mamba block whose SSM path and gate come from different token streams.

    FMCNet calls this block ``Mamba3``. The first stream is one granularity,
    while ``gate_states`` is the fused multi-granularity context. The original
    fused fast path cannot accept the separate gate stream, so this baseline
    explicitly uses the selective-scan path that implements the intended math.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner, bias=bias)
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _project_tokens(self, states: Tensor, projection: nn.Linear) -> Tensor:
        projected = rearrange(
            projection.weight @ rearrange(states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=states.shape[1],
        )
        if projection.bias is not None:
            projected = projected + rearrange(
                projection.bias.to(dtype=projected.dtype), "d -> d 1"
            )
        return projected

    def forward(self, hidden_states: Tensor, gate_states: Tensor) -> Tensor:
        if hidden_states.shape != gate_states.shape:
            raise ValueError(
                "hidden_states and gate_states must have identical BLD shapes, "
                f"got {tuple(hidden_states.shape)} and {tuple(gate_states.shape)}"
            )

        sequence_length = hidden_states.shape[1]
        x = self._project_tokens(hidden_states, self.in_proj)
        z = self._project_tokens(gate_states, self.gate_proj)

        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :sequence_length])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, b_state, c_state = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=sequence_length)
        b_state = rearrange(
            b_state, "(b l) n -> b n l", l=sequence_length
        ).contiguous()
        c_state = rearrange(
            c_state, "(b l) n -> b n l", l=sequence_length
        ).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(self.A_log.float()),
            b_state,
            c_state,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return self.out_proj(rearrange(y, "b d l -> b l d"))


class MGSSMLayer(nn.Module):
    """FMCNet MG-SSM applied directly to a full 3-D feature map."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.channel_tokens = channel_tokens
        self.norm = nn.LayerNorm(d_model)

        if channel_tokens:
            self.channel_mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.granularity_mamba = None
            self.granularity_convs = nn.ModuleList()
            self.fuse_context = None
            self.fuse_output = None
        else:
            self.channel_mamba = None
            self.granularity_mamba = CrossGatedMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.granularity_convs = nn.ModuleList(
                [
                    nn.Conv3d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                    )
                    for d in (1, 2, 3)
                ]
            )
            self.fuse_context = nn.Conv3d(
                3 * channels, channels, kernel_size=1, bias=False
            )
            self.fuse_output = nn.Conv3d(
                4 * channels, channels, kernel_size=1, bias=False
            )

    def _spatial_tokens(self, x: Tensor) -> Tensor:
        batch, channels = x.shape[:2]
        if channels != self.d_model:
            raise ValueError(
                f"Expected {self.d_model} feature channels, got {channels}"
            )
        return self.norm(x.reshape(batch, channels, -1).transpose(1, 2))

    def _forward_channel_tokens(self, x: Tensor) -> Tensor:
        batch, channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        flattened = x.flatten(2)
        if flattened.shape[-1] != self.d_model:
            raise ValueError(
                "The runtime feature-map shape differs from the nnU-Net plan: "
                f"expected {self.d_model} voxels, got {flattened.shape[-1]}"
            )
        assert self.channel_mamba is not None
        return self.channel_mamba(self.norm(flattened)).reshape(
            batch, channels, *spatial_shape
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"MG-SSM requires BCDHW input, got shape {tuple(x.shape)}")
        if self.channel_tokens:
            return self._forward_channel_tokens(x)

        batch, channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        granularities = [conv(x) for conv in self.granularity_convs]
        assert self.fuse_context is not None
        assert self.fuse_output is not None
        shared_context = self._spatial_tokens(
            self.fuse_context(torch.cat(granularities, dim=1))
        )
        assert self.granularity_mamba is not None
        outputs = [
            self.granularity_mamba(self._spatial_tokens(feature), shared_context)
            .transpose(1, 2)
            .reshape(batch, channels, *spatial_shape)
            for feature in granularities
        ]
        return self.fuse_output(torch.cat([*outputs, x], dim=1))
