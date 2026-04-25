# Liu et al. - 2020 - A ConvNet for the 2020s
# Modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

__all__ = ['ConvNeXt']


class LayerNorm(nn.Module):
    '''LayerNorm supporting both channels_last and channels_first formats.

    This implementation allows for flexible normalization across different
    tensor layouts commonly used in convolutional and transformer-based
    architectures.

    Attributes:
        weight (nn.Parameter): Learnable scaling weights.
        bias (nn.Parameter): Learnable bias offsets.
        eps (float): Small constant for numerical stability.
        data_format (str): Data format of the input ('channels_last' or
            'channels_first').
        normalized_shape (tuple): Shape of the normalization dimensions.
    '''

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = 'channels_last',
    ) -> None:
        '''Initializes the LayerNorm module.

        Args:
            normalized_shape: Number of expected features.
            eps: Epsilon value. Defaults to 1e-6.
            data_format: Ordering of input dimensions. Defaults to
                'channels_last'.
        '''
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Applies layer normalization to the input tensor.

        Args:
            x: Input tensor of shape (N, L, C) or (N, C, L).

        Returns:
            Normalized tensor with the same shape as the input.
        '''
        if self.data_format == 'channels_last':
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]

            return x


class Block(nn.Module):
    '''ConvNeXt Block following the inverted bottleneck strategy.

    The block consists of a depthwise convolution followed by layer
    normalization and two pointwise convolutions implemented as linear
    layers for speed.

    Attributes:
        dwconv (nn.Conv1d): Depthwise convolution layer.
        norm (LayerNorm): Normalization layer.
        pwconv1 (nn.Linear): First pointwise projection layer.
        act (nn.GELU): Activation function.
        pwconv2 (nn.Linear): Second pointwise projection layer.
        gamma (nn.Parameter): Learnable scale for residual scaling.
        drop_path (nn.Module): Stochastic depth layer.
    '''

    def __init__(
        self,
        dim: int,
        ratio_rb: int = 4,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        '''Initializes the ConvNeXt block.

        Args:
            dim: Number of input channels.
            ratio_rb: Scale ratio for the reversed bottleneck. Defaults to 4.
            drop_path: Stochastic depth rate. Defaults to 0.0.
            layer_scale_init_value: Initial value for layer scale.
                Defaults to 1e-6.
        '''
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, ratio_rb * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(ratio_rb * dim, dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the block.

        Args:
            x: Input tensor of shape (N, C, L).

        Returns:
            Residual output tensor of shape (N, C, L).
        '''
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = input + self.drop_path(x)

        return x


class ConvNeXt(nn.Module):
    '''ConvNeXt architecture for 1D sequence modeling.

    A modern ConvNet implementation inspired by 'A ConvNet for the 2020s',
    adapted for 1D signal/feature processing.

    Attributes:
        depths (list): Number of blocks at each stage.
        dims (list): Feature dimension at each stage.
        num_stage (int): Total number of stages.
        downsample_layers (nn.ModuleList): Stem and downsampling layers.
        stages (nn.ModuleList): Sequence of residual block stages.
        norm (nn.LayerNorm): Final layer normalization.
    '''

    def __init__(
        self,
        in_chans: int,
        depths: list[int] = [2, 2, 2],
        dims: list[int] = [96, 192, 384],
        ratio_rb: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        '''Initializes the ConvNeXt model.

        Args:
            in_chans: Number of input channels.
            depths: Block depths for each stage. Defaults to [2, 2, 2].
            dims: Dimensions for each stage. Defaults to [96, 192, 384].
            ratio_rb: Reversed bottleneck ratio. Defaults to 3.
            drop_path_rate: Base stochastic depth rate. Defaults to 0.0.
            layer_scale_init_value: Layer scale starting value.
                Defaults to 1e-6.
        '''
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.num_stage = len(dims)

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'),
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        ratio_rb=ratio_rb,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        '''Initializes the weights of linear and convolutional layers.

        Args:
            m: The module to initialize.
        '''
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Processes input through the ConvNeXt stages.

        Args:
            x: Input tensor of shape (batch_size, num_chan, len_seq).

        Returns:
            Processed feature tensor of shape (batch_size, num_chan, len_seq).
        '''
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.transpose(1, 2)
        x = self.norm(x)

        return x

    @property
    def dim_out(self) -> int:
        '''Returns the number of channels in the final feature map.'''
        return self.dims[-1]

    @property
    def ratio_ds(self) -> int:
        '''Calculates the total temporal downsampling factor.'''
        return 2 ** len(self.depths)