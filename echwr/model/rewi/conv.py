import torch
import torch.nn as nn

__all__ = ['BLConv']


class PatchEmbed(nn.Module):
    '''Patch embedding layer for downsampling 1D sequences.

    This layer acts as a transition block, projecting input channels to a new
    dimension and reducing sequence length via strided convolution.

    Args:
        in_chan: Number of input channels.
        out_chan: Number of output channels.
        kernel: Kernel size for the convolution. Defaults to 2.
        stride: Stride for downsampling. Defaults to 2.

    Attributes:
        conv (nn.Conv1d): The projection and downsampling convolution.
        norm (nn.InstanceNorm1d): Instance normalization layer.
    '''

    def __init__(
        self, in_chan: int, out_chan: int, kernel: int = 2, stride: int = 2
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_chan, out_chan, kernel, stride)
        self.norm = nn.InstanceNorm1d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Applies convolution and normalization.

        Args:
            x: Input tensor of shape (batch_size, in_chan, len_seq).

        Returns:
            Output tensor of shape (batch_size, out_chan, len_seq_out).

            The output length is calculated as:
            L_out = floor((L_in - kernel) / stride + 1)
        '''
        x = self.conv(x)
        x = self.norm(x)

        return x


class MSConv(nn.Module):
    '''Multi-scale depth-wise separable convolutional block with
    re-parameterization.

    **Training Behavior:**
    The input passes through three parallel depth-wise branches with kernel
    sizes 1, 3, and 5. Their outputs are summed.

    **Inference Behavior (after `fuse()`):**
    The weights of the 1x1 and 3x3 branches are mathematically merged into the
    5x5 branch. The smaller branches are deleted to reduce memory access costs
    and latency.

    Args:
        dim: Input/Output feature dimension.
        r_drop: Dropout probability. Defaults to 0.2.

    Attributes:
        fused (bool): State flag indicating if layers have been merged.
        dwconv1 (nn.Conv1d): 1x1 Depth-wise convolution (removed after fuse).
        dwconv3 (nn.Conv1d): 3x3 Depth-wise convolution (removed after fuse).
        dwconv5 (nn.Conv1d): 5x5 Depth-wise convolution (acts as main layer).
        pwconv (nn.Conv1d): Point-wise convolution for channel mixing.
        norm (nn.InstanceNorm1d): Normalization layer applied after convolution.
        act (nn.GELU): Activation function.
        drop (nn.Dropout): Dropout layer for regularization.
    '''

    def __init__(self, dim: int, r_drop: float = 0.2) -> None:
        super().__init__()

        self.fused = False
        self.dwconv1 = nn.Conv1d(dim, dim * 2, 1, padding='same', groups=dim)
        self.dwconv3 = nn.Conv1d(dim, dim * 2, 3, padding='same', groups=dim)
        self.dwconv5 = nn.Conv1d(dim, dim * 2, 5, padding='same', groups=dim)
        self.pwconv = nn.Conv1d(dim * 2, dim, 1)
        self.norm = nn.InstanceNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(r_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs the multi-scale convolution block pass.

        Args:
            x: Input tensor of shape (batch_size, dim, len_seq).

        Returns:
            Output tensor of shape (batch_size, dim, len_seq).
        '''
        if self.fused:
            x = self.dwconv5(x)
        else:
            x = self.dwconv1(x) + self.dwconv3(x) + self.dwconv5(x)

        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

    def fuse(self) -> None:
        '''Merges parallel convolutional branches into a single layer.

        This performs structural re-parameterization:
        1. Centers the 1x1 kernel onto the center of the 5x5 kernel.
        2. Centers the 3x3 kernel onto the center of the 5x5 kernel.
        3. Adds the biases.
        4. Deletes the 1x1 and 3x3 layers to free memory.
        '''
        with torch.no_grad():
            self.dwconv5.weight.data[
                :, :, 2
            ] += self.dwconv1.weight.data.squeeze(-1)
            self.dwconv5.weight.data[:, :, 1:4] += self.dwconv3.weight.data
            self.dwconv5.bias.data += (
                self.dwconv1.bias.data + self.dwconv3.bias.data
            )

        del self.dwconv1
        del self.dwconv3
        self.fused = True


class BLConv(nn.Module):
    '''Convolutional baseline encoder using multi-stage stacking.

    This architecture builds a stack of `PatchEmbed` (downsampling) and
    `MSConv` (processing) blocks.

    Args:
        in_chan: Number of input channels.
        depths: A list defining the number of `MSConv` blocks in each stage.
            Defaults to `[3, 3, 3]`.
        dims: A list defining the feature dimension for each stage.
            Must have the same length as `depths`.
            Defaults to `[128, 256, 512]`.

    Attributes:
        depths (list[int]): The stored configuration for block depths.
        dims (list[int]): The stored configuration for feature dimensions.
        layers (nn.ModuleList): The sequential list of all network layers
            (PatchEmbed and MSConv).
    '''

    def __init__(
        self,
        in_chan: int,
        depths: list[int] = [3, 3, 3],
        dims: list[int] = [128, 256, 512],
    ) -> None:
        super().__init__()

        self.depths = depths
        self.dims = [in_chan] + dims
        self.layers = nn.ModuleList([])

        for i in range(len(depths)):
            self.layers.append(PatchEmbed(self.dims[i], self.dims[i + 1]))
            self.layers.extend(
                [MSConv(self.dims[i + 1]) for _ in range(depths[i])]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through all stages.

        Args:
            x: Input tensor of shape (batch_size, in_chan, len_seq).

        Returns:
            Output tensor of shape (batch_size, len_seq_out, dim_out).

            Note: The output is transposed to (Batch, Sequence, Channel)
            format to be compatible with standard RNN/Linear layers.
        '''
        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)

        return x

    def fuse(self) -> None:
        '''Trigger structural re-parameterization for all child layers.

        This calls `.fuse()` on every submodule that supports it (specifically
        `MSConv`).
        '''
        for m in self.layers:
            if hasattr(m, 'fuse'):
                m.fuse()

    @property
    def dim_out(self) -> int:
        '''The final output feature dimension size.'''
        return self.dims[-1]

    @property
    def ratio_ds(self) -> int:
        '''The total downsampling ratio of the encoder.

        Calculated as: 2 ** num_stages
        (Assuming standard stride-2 downsampling).
        '''
        return 2 ** len(self.depths)
