import torch
import torch.nn as nn

from .bilstm import BiLSTM
from .conv import BLConv
from .others.swinv2 import SwinTransformerV2
from .others.convnext import ConvNeXt

__all__ = ['BaseModel']


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Module:
    '''Factory function to build a CTC encoder module.

    Args:
        in_chan: Number of input channels.
        arch: Encoder architecture key. Supported: 'blconv_b', 'blconv_s',
            'convnext', 'swinv2', 'vit', 'blconv_s_d2', 'blconv_s_d3', and
            'blconv_b_d1'.
        len_seq: Expected length of the input sequence. Defaults to 0.

    Returns:
        An initialized encoder module (nn.Module).

    Raises:
        ValueError: If `arch` is not a supported architecture string.
    '''
    match arch:
        case 'blconv_b':
            return BLConv(in_chan)
        case 'blconv_s':
            return BLConv(in_chan, [1, 1, 1], [64, 128, 256])
        case 'convnext':
            return ConvNeXt(in_chan)
        case 'swinv2':
            return SwinTransformerV2(in_chan, len_seq)
        case 'blconv_s_d2':
            return BLConv(in_chan, [2, 2, 2], [64, 128, 256])
        case 'blconv_s_d3':
            return BLConv(in_chan, [3, 3, 3], [64, 128, 256])
        case 'blconv_b_d1':
            return BLConv(in_chan, [1, 1, 1])
        case _:
            raise ValueError(
                f'Unknown encoder architecture: "{arch}". '
                'Supported: ["blconv_b", "blconv_s"]'
            )


def build_decoder(
    dim_in: int, num_cls: int, arch: str, len_seq: int = 0
) -> nn.Module:
    '''Factory function to build a CTC decoder module.

    Args:
        dim_in: Input feature dimension.
        num_cls: Number of output classes (vocabulary size).
        arch: Decoder architecture key. Must be one of:
            * 'bilstm_b': Base BiLSTM (standard hidden size/layers).
            * 'bilstm_s': Small BiLSTM (reduced hidden size/layers).
        len_seq: Expected length of the input sequence.
            Defaults to 0 (unused by current LSTM implementations).

    Returns:
        An initialized decoder module (nn.Module).

    Raises:
        ValueError: If `arch` is not a supported architecture string.
    '''
    match arch:
        case 'bilstm_b':
            return BiLSTM(dim_in, num_cls)
        case 'bilstm_s':
            return BiLSTM(dim_in, num_cls, 64, 2)
        case _:
            raise ValueError(
                f'Unknown decoder architecture: "{arch}". '
                'Supported: ["bilstm_b", "bilstm_s"]'
            )


class BaseModel(nn.Module):
    '''End-to-end handwriting recognition model using a CTC loss framework.

    This model functions as a wrapper connecting a feature extraction encoder
    and a sequence modeling decoder. It handles the downsampling calculations
    required for CTC alignment.

    Args:
        arch_en: Architecture key for the encoder (passed to `build_encoder`).
        arch_de: Architecture key for the decoder (passed to `build_decoder`).
        in_chan: Number of input channels.
        num_cls: Number of output classes (vocabulary size + blank token).
        len_seq: Expected length of input sequence (used for initialization
            calculations in some architectures). Defaults to 0.

    Attributes:
        encoder (nn.Module): The backbone feature extractor.
        decoder (nn.Module): The sequence modeling head.
        ratio_ds (int): The total downsampling factor of the encoder.
    '''

    def __init__(
        self,
        arch_en: str,
        arch_de: str,
        in_chan: int,
        num_cls: int,
        len_seq: int = 0,
    ) -> None:
        super().__init__()

        self.encoder = build_encoder(in_chan, arch_en, len_seq)
        self.decoder = build_decoder(
            self.encoder.dim_out,
            num_cls,
            arch_de,
            len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs the full forward pass: Encoder -> Decoder.

        Args:
            x: Input tensor of shape (batch_size, in_chan, len_seq).

        Returns:
            Logits tensor of shape (batch_size, len_seq_out, num_cls),
            where `len_seq_out` is the downsampled sequence length.
        '''
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def infer(self) -> None:
        '''Optimizes the model architecture for inference deployment.

        This method triggers structural re-parameterization in the encoder
        (if supported).

        Note:
            This operation is **irreversible**. The model structure is
            permanently altered, so training cannot be resumed after calling
            this method.
        '''
        # blconv: fuse parameters of layers
        if hasattr(self.encoder, 'fuse'):
            self.encoder.fuse()

    @property
    def ratio_ds(self) -> int:
        '''The total downsampling factor of the encoder.'''
        return self.encoder.ratio_ds
