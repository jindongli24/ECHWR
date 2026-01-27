import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AttentionPool1D', 'GatedAttentionPool1D', 'get_pooling']


class AttentionPool1D(nn.Module):
    '''Attention-based pooling module for 1D inputs with cosine positional
    embeddings.

    This module aggregates a sequence of features into a single global
    representation using multi-head attention. It computes sinusoidal
    positional embeddings on the fly, allowing it to handle variable-length
    inputs seamlessly.

    Args:
        dim_in: Input feature dimension.
        num_heads: Number of attention heads.
        dim_out: Output feature dimension. Must be divisible by `num_heads`.

    Attributes:
        num_heads (int): Number of attention heads.
        proj_in (nn.Linear): Initial projection layer for inputs.
        proj_q (nn.Linear): Linear projection for queries.
        proj_k (nn.Linear): Linear projection for keys.
        proj_v (nn.Linear): Linear projection for values.
        proj_c (nn.Linear): Final output projection layer.

    Raises:
        AssertionError: If `dim_out` is not divisible by `num_heads`.
    '''

    def __init__(self, dim_in: int, num_heads: int, dim_out: int) -> None:
        super().__init__()

        assert (
            dim_out % num_heads == 0
        ), 'dim_out must be divisible by num_heads'

        self.num_heads = num_heads
        self.dim_out = dim_out  # Store for initialization usage

        self.proj_in = nn.Linear(dim_in, dim_out)
        self.proj_q = nn.Linear(dim_out, dim_out)
        self.proj_k = nn.Linear(dim_out, dim_out)
        self.proj_v = nn.Linear(dim_out, dim_out)
        self.proj_c = nn.Linear(dim_out, dim_out)

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        '''Initialize weights with specific scaling.'''
        attn_std = self.dim_out**-0.5
        proj_std = self.dim_out**-0.5

        nn.init.normal_(self.proj_in.weight, std=proj_std)
        nn.init.constant_(self.proj_in.bias, 0.0)

        nn.init.normal_(self.proj_q.weight, std=attn_std)
        nn.init.constant_(self.proj_q.bias, 0.0)
        nn.init.normal_(self.proj_k.weight, std=attn_std)
        nn.init.constant_(self.proj_k.bias, 0.0)
        nn.init.normal_(self.proj_v.weight, std=attn_std)
        nn.init.constant_(self.proj_v.bias, 0.0)

        nn.init.normal_(self.proj_c.weight, std=proj_std)
        nn.init.constant_(self.proj_c.bias, 0.0)

    @staticmethod
    def get_sinusoidal_embedding(
        L: int, C: int, device: str = None
    ) -> torch.Tensor:
        '''Generates standard sinusoidal positional embeddings.

        Calculates embeddings using the formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/C))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/C))

        Args:
            L: Sequence length.
            C: Embedding dimension (must be even).
            device: The device to create the tensor on.

        Returns:
            Tensor of shape (L, C) containing the positional embeddings.
        '''
        position = torch.arange(
            L, dtype=torch.float32, device=device
        ).unsqueeze(1)
        term_div = torch.exp(
            torch.arange(0, C, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / C)
        )

        # sin and cos embeddings
        embed_pos = torch.zeros(L, C, device=device)
        embed_pos[:, 0::2] = torch.sin(position * term_div)  # even indices
        embed_pos[:, 1::2] = torch.cos(position * term_div)  # odd indices

        return embed_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs the attention pooling forward pass.

        The input is projected, permuted to (Seq, Batch, Dim), and augmented
        with a global mean token. Attention extracts the global context.

        Args:
            x: Input tensor of shape (batch_size, len_seq, dim_in).

        Returns:
            Pooled output tensor of shape (batch_size, dim_out).
        '''
        x = self.proj_in(x)
        x = x.permute(1, 0, 2)  # B, L, C -> L, B, C
        _, _, C = x.shape

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # L+1, B, C

        # sinusoidal positional embedding
        pe = self.get_sinusoidal_embedding(
            x.size(0), x.size(2), device=x.device
        )
        x = x + pe[:, None, :]

        x, _ = F.multi_head_attention_forward(
            query=x[:1],  # global token
            key=x,  # all tokens
            value=x,  # all tokens
            embed_dim_to_check=C,
            num_heads=self.num_heads,
            q_proj_weight=self.proj_q.weight,
            k_proj_weight=self.proj_k.weight,
            v_proj_weight=self.proj_v.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.proj_q.bias, self.proj_k.bias, self.proj_v.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.proj_c.weight,
            out_proj_bias=self.proj_c.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x.squeeze(0)


class GatedAttentionPool1D(nn.Module):
    '''Global pooling module combining multi-head attention with a gating
    mechanism.

    This module aggregates a sequence of features into a single global
    representation. It utilizes sinusoidal positional embeddings for
    variable-length inputs and a gating layer to weigh the contribution of
    each attention head dynamically.

    Args:
        dim_in: Input feature dimension.
        num_heads: Number of attention heads.
        dim_out: Output feature dimension. Must be divisible by `num_heads`.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        proj_in (nn.Linear): Initial projection layer for inputs.
        proj_q (nn.Linear): Linear projection for queries.
        proj_k (nn.Linear): Linear projection for keys.
        proj_v (nn.Linear): Linear projection for values.
        proj_gate (nn.Linear): Layer that learns gating coefficients for each head.
        proj_c (nn.Linear): Final output projection layer.

    Raises:
        AssertionError: If `dim_out` is not divisible by `num_heads`.
    '''

    def __init__(self, dim_in: int, num_heads: int, dim_out: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out  # Store for initialization

        assert (
            self.head_dim * num_heads == dim_out
        ), 'dim_out must be divisible by num_heads'

        self.proj_in = nn.Linear(dim_in, dim_out)
        self.proj_q = nn.Linear(dim_out, dim_out)
        self.proj_k = nn.Linear(dim_out, dim_out)
        self.proj_v = nn.Linear(dim_out, dim_out)
        self.proj_gate = nn.Linear(dim_out, num_heads)
        self.proj_c = nn.Linear(dim_out, dim_out)

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        '''Initialize weights with specific scaling.'''
        attn_std = self.dim_out**-0.5
        proj_std = self.dim_out**-0.5

        nn.init.normal_(self.proj_in.weight, std=proj_std)
        nn.init.constant_(self.proj_in.bias, 0.0)

        nn.init.normal_(self.proj_q.weight, std=attn_std)
        nn.init.constant_(self.proj_q.bias, 0.0)
        nn.init.normal_(self.proj_k.weight, std=attn_std)
        nn.init.constant_(self.proj_k.bias, 0.0)
        nn.init.normal_(self.proj_v.weight, std=attn_std)
        nn.init.constant_(self.proj_v.bias, 0.0)

        nn.init.normal_(self.proj_gate.weight, std=attn_std)
        nn.init.constant_(self.proj_gate.bias, 0.0)

        nn.init.normal_(self.proj_c.weight, std=proj_std)
        nn.init.constant_(self.proj_c.bias, 0.0)

    @staticmethod
    def get_sinusoidal_embedding(
        L: int, C: int, device: str = None
    ) -> torch.Tensor:
        '''Generates standard sinusoidal positional embeddings.

        Calculates embeddings using the formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/C))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/C))

        Args:
            L: Sequence length.
            C: Embedding dimension (must be even).
            device: The device to create the tensor on.

        Returns:
            Tensor of shape (L, C) containing the positional embeddings.
        '''
        position = torch.arange(
            L, dtype=torch.float32, device=device
        ).unsqueeze(1)
        term_div = torch.exp(
            torch.arange(0, C, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / C)
        )

        # sin and cos embeddings
        embed_pos = torch.zeros(L, C, device=device)
        embed_pos[:, 0::2] = torch.sin(position * term_div)  # even indices
        embed_pos[:, 1::2] = torch.cos(position * term_div)  # odd indices

        return embed_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs the attention pooling forward pass.

        The input is projected, permuted to (Seq, Batch, Dim), and augmented
        with a global mean token. Attention extracts the global context.

        Args:
            x: Input tensor of shape (batch_size, len_seq, dim_in).

        Returns:
            Pooled output tensor of shape (batch_size, dim_out).
        '''
        x = self.proj_in(x)
        x = x.permute(1, 0, 2)  # B, L, C -> L, B, C
        _, B, C = x.shape

        # append global token to sequence
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # L+1, B, C

        # add sinusoidal positional embedding
        pe = self.get_sinusoidal_embedding(
            x.size(0), x.size(2), device=x.device
        )
        x = x + pe[:, None, :]

        q = (
            self.proj_q(x[:1])
            .view(1, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )  # [size_batch, num_head, len_seq, dim_head]
        k = (
            self.proj_k(x)
            .view(-1, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )  # [size_batch, num_head, len_seq, dim_head]
        v = (
            self.proj_v(x)
            .view(-1, B, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )  # [size_batch, num_head, len_seq, dim_head]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )  # [size_batch, num_head, 1, dim_head]
        gate = (
            torch.sigmoid(self.proj_gate(x[:1])).permute(1, 2, 0).unsqueeze(-1)
        )  # [size_batch, num_head, 1, 1]
        attn_out = attn_out * gate
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(
            1, B, C
        )  # [1, size_batch, num_dim]
        output = self.proj_c(attn_out)

        return output.squeeze(0)


def get_pooling(
    arch_pool: str, dim_in: int, num_heads: int, dim_out: int
) -> nn.Module:
    '''Factory function to instantiate a time-series pooling layer.

    Args:
        arch_pool: The architecture type to instantiate.
            Must be one of:
            * 'attn': Standard Attention Pooling.
            * 'attn_gated': Gated Attention Pooling.
        dim_in: Input feature dimension.
        num_heads: Number of attention heads.
        dim_out: Output feature dimension.

    Returns:
        An initialized pooling module (nn.Module).

    Raises:
        ValueError: If `arch_pool` is not a supported architecture string.
    '''
    match arch_pool:
        case 'attn':
            return AttentionPool1D(dim_in, num_heads, dim_out)
        case 'attn_gated':
            return GatedAttentionPool1D(dim_in, num_heads, dim_out)
