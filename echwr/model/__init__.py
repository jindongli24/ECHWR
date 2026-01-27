import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolings import get_pooling
from .rewi import BaseModel
from .transformers import get_encoder_txt

__all__ = ['ECHWR']


class ECHWR(nn.Module):
    '''Error-based Contrastive-enhanced Handwriting Recognition.

    This architecture combines a standard Encoder-Decoder HWR backbone with a
    multi-modal contrastive learning branch. It projects both the time-series
    sequence (via pooling) and the text label (via a Transformer) into a shared
    embedding space.

    Args:
        arch_en: Architecture of the handwriting recognition encoder.
        arch_de: Architecture of the handwriting recognition decoder.
        arch_pool: Architecture of the time-series pooling layer.
        arch_txt: Architecture of the text encoder.
        in_chan: Number of input channels.
        num_cls: Number of classes (vocabulary size).
        dim_embed: The embedding dimension for contrastive learning. Defaults
            to 512.
        heads_pool: Number of heads of the attention pooling module. Defaults
            to 8.
        len_context: Maximum context length of text. Defaults to 64.
        len_seq: Expected length of the input sequence. Defaults to 0.
        num_layer: Number of Transformer layers for text encoder. Defaults
            to 3.
        heads_txt: Number of heads of the text transformer. Defaults to 8.
        dim_txt: Dimension of the text embedding. Defaults to 512.

    Attributes:
        rewi (BaseModel): The backbone HWR model (Encoder + Decoder).
        attnpool (nn.Module): Pooling layer to convert 1D sequence features
            into a global time-series vector.
        encoder_txt (nn.Module): Transformer encoder for text strings.
        logit_scale (nn.Parameter): Learnable temperature parameter for
            contrastive scaling.
        ratio_ds (int): Downsampling ratio of the time-series encoder.
        scale (torch.Tensor): The computed temperature scaling factor.
    '''

    def __init__(
        self,
        arch_en: str,
        arch_de: str,
        arch_pool: str,
        arch_txt: str,
        in_chan: int,
        num_cls: int,
        dim_embed: int = 512,
        heads_pool: int = 8,
        len_context: int = 64,
        len_seq: int = 0,
        num_layer: int = 3,
        heads_txt: int = 8,
        dim_txt: int = 512,
    ) -> None:
        super().__init__()

        self.rewi = BaseModel(arch_en, arch_de, in_chan, num_cls, len_seq)
        self.attnpool = get_pooling(
            arch_pool, self.rewi.encoder.dim_out, heads_pool, dim_embed
        )

        self.encoder_txt = get_encoder_txt(
            arch_txt,
            num_cls,
            len_context,
            num_layer,
            heads_txt,
            dim_txt,
            dim_embed,
        )
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def forward(
        self,
        seq: torch.Tensor = None,
        txts: torch.Tensor = None,
        tasks: list[str] = ['hwr'],
    ) -> dict[str, torch.Tensor]:
        '''Performs the forward pass for HWR and/or Contrastive tasks.

        Args:
            seq: Time-series input sequence (size_batch, num_chan, len_seq).
            txts: Text tensor where the first batch is the original text and
                the rest is (num_txt, size_batch, len_context). Defaults to
                None.
            tasks: List of tasks to calculate. Options are 'hwr' for CTC loss,
                'bc' for batch-wise contrastive loss, and 'ec' for error-based
                contrastive loss. Defaults to ['hwr'].

        Returns:
            Dictionary containing outputs: 'out_hwr' for HWR logits,
            'embed_seq' for time-series embeddings, and 'embeds_txt' for text
            embeddings.
        '''
        outputs = {'out_hwr': None, 'embed_seq': None, 'embeds_txt': None}

        # time-series encoding
        if seq is not None:
            features_seq = self.rewi.encoder(seq)

            if 'hwr' in tasks:
                outputs['out_hwr'] = self.rewi.decoder(features_seq)

            if 'bc' in tasks or 'ec' in tasks:
                embed_seq = self.attnpool(features_seq)
                outputs['embed_seq'] = F.normalize(embed_seq, 2, 1)

        if txts is not None and ('bc' in tasks or 'ec' in tasks):

            N, B, L = txts.shape
            embeds_txt = self.encoder_txt(txts.reshape(-1, L)).reshape(
                N, B, -1
            )
            outputs['embeds_txt'] = F.normalize(embeds_txt, 2, 2)

        return outputs

    @property
    def ratio_ds(self) -> int:
        '''The downsampling ratio of the time-series encoder.'''
        return self.rewi.ratio_ds

    @property
    def scale(self) -> torch.Tensor:
        '''The temperature scaling factor for calculating similarity.'''
        return self.logit_scale.exp()
