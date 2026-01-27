from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Transformer',
    'TransformerWithRMSNorm',
    'GatedTransformer',
    'GatedTransformerWithRMSNorm',
    'GatedTransformerWithRegisters',
    'GatedTransformerWithRMSNormAndRegisters',
    'get_encoder_txt',
]


class QuickGELU(nn.Module):
    '''Applies the QuickGELU activation function.

    Approximation: x * sigmoid(1.702 * x)

    Args:
        None

    Attributes:
        None
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Applies the activation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with same shape as input.
        '''
        return x * torch.sigmoid(1.702 * x)


class GatedMultiheadAttention(nn.Module):
    '''Multi-head attention with a gating mechanism.

    As proposed in 'Gated Attention for Large Language Models:
    Non-linearity, Sparsity, and Attention-Sink-Free' (arXiv:2312.14913),
    this module uses separate Q, K, and V projections and adds a learnable
    gate to weigh the attention output based on the input context.

    Args:
        d_model: Total dimension of the model.
        n_head: Number of parallel attention heads.
        dropout: Dropout probability. Defaults to 0.0.

    Attributes:
        n_head (int): Number of heads.
        dim_head (int): Dimension per head.
        proj_q (nn.Linear): Projection for Query.
        proj_k (nn.Linear): Projection for Key.
        proj_v (nn.Linear): Projection for Value.
        proj_gate (nn.Linear): Gate projection layer.
        proj_out (nn.Linear): Output projection layer.
        dropout (float): Dropout probability.
    '''

    def __init__(
        self, d_model: int, n_head: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.n_head = n_head
        self.dim_head = d_model // n_head

        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        self.proj_gate = nn.Linear(d_model, n_head)
        self.proj_out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        '''Forward pass with gated attention.

        Args:
            query: Query tensor (len_query, batch, d_model).
            key: Key tensor (len_key, batch, d_model).
            value: Value tensor (len_value, batch, d_model).
            attn_mask: Mask to avoid attending to certain positions.
                Defaults to None.
            **kwargs: API compatibility arguments.

        Returns:
            Tuple containing the attention output and None for weights.
        '''
        # shapes: l=sequence length, n=batch size, e=embedding dimension
        L_q, N, E = query.shape
        L_k, _, _ = key.shape
        L_v, _, _ = value.shape

        # shape: (l, n, e) -> (l, n, h, d) -> (n, h, l, d)
        q = (
            self.proj_q(query)
            .reshape(L_q, N, self.n_head, self.dim_head)
            .permute(1, 2, 0, 3)
        )
        k = (
            self.proj_k(key)
            .reshape(L_k, N, self.n_head, self.dim_head)
            .permute(1, 2, 0, 3)
        )
        v = (
            self.proj_v(value)
            .reshape(L_v, N, self.n_head, self.dim_head)
            .permute(1, 2, 0, 3)
        )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # broadcast for batch and heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # gating mechanism:
        # the gate depends on the query (context)
        # (l_q, n, e) -> (l_q, n, h)
        gate = torch.sigmoid(self.proj_gate(query))

        # (l_q, n, h) -> (n, h, l_q, 1) to broadcast over head dimension
        gate = gate.permute(1, 2, 0).unsqueeze(-1)

        attn_out = attn_out * gate

        # recombine heads
        # (n, h, l_q, d) -> (l_q, n, e)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(L_q, N, E)
        output = self.proj_out(attn_out)

        return output, None


class ResidualAttentionBlock(nn.Module):
    '''Standard Transformer block with Pre-LayerNorm.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        attn_mask: Attention mask tensor. Defaults to None.

    Attributes:
        attn (nn.MultiheadAttention): The attention module.
        norm_1 (nn.LayerNorm): Layer norm before attention.
        mlp (nn.Sequential): Feed-forward network.
        norm_2 (nn.LayerNorm): Layer norm before MLP.
        attn_mask (torch.Tensor): Mask used during attention.
    '''

    def __init__(
        self, d_model: int, n_head: int, attn_mask: torch.Tensor = None
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ('fc_c', nn.Linear(d_model, d_model * 4)),
                    ('gelu', QuickGELU()),
                    ('proj_c', nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        '''Helper to manage mask device/type and call attention.

        Args:
            x: Input tensor (len_seq, size_batch, dim).

        Returns:
            Attention output (len_seq, size_batch, dim).
        '''
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass with residual connections.

        Args:
            x: Input tensor (len_seq, size_batch, dim).

        Returns:
            Output tensor (len_seq, size_batch, dim).
        '''
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x


class ResidualAttentionBlockWithRMSNorm(ResidualAttentionBlock):
    '''Transformer block using Pre-RMSNorm.

    As proposed in 'Root Mean Square Layer Normalization' (arXiv:1910.07467),
    RMSNorm simplifies normalization by removing the mean centering
    property, reducing computational overhead.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        attn_mask: Attention mask tensor. Defaults to None.

    Attributes:
        attn (nn.MultiheadAttention): The attention module.
        norm_1 (nn.RMSNorm): RMSNorm before attention.
        mlp (nn.Sequential): Feed-forward network.
        norm_2 (nn.RMSNorm): RMSNorm before MLP.
        attn_mask (torch.Tensor): Mask used during attention.
    '''

    def __init__(
        self, d_model: int, n_head: int, attn_mask: torch.Tensor = None
    ) -> None:
        super().__init__(d_model, n_head, attn_mask)

        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)


class GatedResidualAttentionBlock(ResidualAttentionBlock):
    '''Residual block using Gated Attention.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        attn_mask: Attention mask tensor. Defaults to None.

    Attributes:
        attn (GatedMultiheadAttention): The gated attention module.
        norm_1 (nn.LayerNorm): Layer norm before attention.
        mlp (nn.Sequential): Feed-forward network.
        norm_2 (nn.LayerNorm): Layer norm before MLP.
        attn_mask (torch.Tensor): Mask used during attention.
    '''

    def __init__(
        self, d_model: int, n_head: int, attn_mask: torch.Tensor = None
    ):
        super().__init__(d_model, n_head, attn_mask)

        self.attn = GatedMultiheadAttention(d_model, n_head, dropout=0.1)


class GatedResidualAttentionBlockWithRMSNorm(
    ResidualAttentionBlockWithRMSNorm
):
    '''Residual block using Gated Attention and RMSNorm.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        attn_mask: Attention mask tensor. Defaults to None.

    Attributes:
        attn (GatedMultiheadAttention): The gated attention module.
        norm_1 (nn.RMSNorm): RMSNorm before attention.
        mlp (nn.Sequential): Feed-forward network.
        norm_2 (nn.RMSNorm): RMSNorm before MLP.
        attn_mask (torch.Tensor): Mask used during attention.
    '''

    def __init__(
        self, d_model: int, n_head: int, attn_mask: torch.Tensor = None
    ):
        super().__init__(d_model, n_head, attn_mask)

        self.attn = GatedMultiheadAttention(d_model, n_head, dropout=0.1)


class Transformer(nn.Module):
    '''Transformer Encoder with learnable positional embeddings.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.LayerNorm): Final layer normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_layer = num_layer

        self.embedding_token = nn.Embedding(num_cls, dim)
        self.embedding_position = nn.Parameter(torch.empty(len_context, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(dim, heads, attn_mask)
                for _ in range(num_layer)
            ]
        )

        self.norm_final = nn.LayerNorm(dim)
        self.projection = nn.Parameter(torch.empty(dim, dim_out))

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        '''Initialize weights using specific scaling rules.'''
        nn.init.normal_(self.embedding_token.weight, std=0.02)
        nn.init.normal_(self.embedding_position, std=0.01)

        proj_std = (self.dim**-0.5) * ((2 * self.num_layer) ** -0.5)
        attn_std = self.dim**-0.5
        fc_std = (2 * self.dim) ** -0.5

        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)

            nn.init.normal_(block.mlp.fc_c.weight, std=fc_std)
            nn.init.normal_(block.mlp.proj_c.weight, std=proj_std)

        nn.init.normal_(self.projection, std=self.dim**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes text tokens into a vector representation.

        Args:
            x: Text token IDs (size_batch, len_context).

        Returns:
            Projected CLS token embedding (size_batch, dim_out).
        '''
        x = self.embedding_token(x)
        x = x + self.embedding_position
        x = x.permute(1, 0, 2)  # nld -> lnd

        batch_size = x.shape[1]
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        x = torch.cat([cls_tokens, x], dim=0)

        x = self.resblocks(x)

        x = x.permute(1, 0, 2)  # lnd -> nld
        x = self.norm_final(x)
        x = x[:, 0, :] @ self.projection

        return x


class TransformerWithRMSNorm(Transformer):
    '''Transformer variant using RMSNorm layers.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.RMSNorm): Final root mean square normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
    ) -> None:
        super().__init__(
            num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
        )

        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlockWithRMSNorm(dim, heads, attn_mask)
                for _ in range(num_layer)
            ]
        )
        self.norm_final = nn.RMSNorm(dim)


class GatedTransformer(Transformer):
    '''Transformer variant using Gated Multi-Head Attention.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.LayerNorm): Final layer normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
    ) -> None:
        super().__init__(
            num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
        )

        self.resblocks = nn.Sequential(
            *[
                GatedResidualAttentionBlock(dim, heads, attn_mask)
                for _ in range(num_layer)
            ]
        )

        proj_std = (self.dim**-0.5) * ((2 * self.num_layer) ** -0.5)
        attn_std = self.dim**-0.5
        fc_std = (2 * self.dim) ** -0.5

        for block in self.resblocks:
            nn.init.normal_(block.attn.proj_q.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_k.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_v.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_out.weight, std=proj_std)
            nn.init.normal_(block.attn.proj_gate.weight, std=attn_std)

            nn.init.normal_(block.mlp.fc_c.weight, std=fc_std)
            nn.init.normal_(block.mlp.proj_c.weight, std=proj_std)


class GatedTransformerWithRMSNorm(TransformerWithRMSNorm):
    '''Transformer variant using Gated Multi-Head Attention and RMSNorm.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.RMSNorm): Final root mean square normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
    ) -> None:
        super().__init__(
            num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
        )

        self.resblocks = nn.Sequential(
            *[
                GatedResidualAttentionBlockWithRMSNorm(dim, heads, attn_mask)
                for _ in range(num_layer)
            ]
        )

        proj_std = (self.dim**-0.5) * ((2 * self.num_layer) ** -0.5)
        attn_std = self.dim**-0.5
        fc_std = (2 * self.dim) ** -0.5

        for block in self.resblocks:
            nn.init.normal_(block.attn.proj_q.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_k.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_v.weight, std=attn_std)
            nn.init.normal_(block.attn.proj_out.weight, std=proj_std)
            nn.init.normal_(block.attn.proj_gate.weight, std=attn_std)

            nn.init.normal_(block.mlp.fc_c.weight, std=fc_std)
            nn.init.normal_(block.mlp.proj_c.weight, std=proj_std)


class GatedTransformerWithRegisters(GatedTransformer):
    '''Transformer using Gated Attention and register tokens.

    Proposed in 'Vision Transformers Need Registers' (arXiv:2309.16588),
    register tokens improve feature quality by acting as global sinks.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.
        num_registers: Number of register tokens to append. Defaults to 2.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        registers (nn.Parameter): Learnable register tokens.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.LayerNorm): Final layer normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
        num_registers: int = 2,
    ) -> None:
        super().__init__(
            num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
        )

        self.registers = nn.Parameter(torch.zeros(num_registers, 1, dim))
        nn.init.normal_(self.registers, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes text tokens into a vector representation with registers.

        Args:
            x: Text token IDs (size_batch, len_context).

        Returns:
            Projected CLS token embedding (size_batch, dim_out).
        '''
        x = self.embedding_token(x)
        x = x + self.embedding_position
        x = x.permute(1, 0, 2)  # nld -> lnd

        batch_size = x.shape[1]
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        registers = self.registers.expand(-1, batch_size, -1)

        # concatenate: [cls, registers, sequence]
        x = torch.cat([cls_tokens, registers, x], dim=0)

        x = self.resblocks(x)

        x = x.permute(1, 0, 2)  # lnd -> nld
        x = self.norm_final(x)

        # extract cls token only (index 0)
        x = x[:, 0, :] @ self.projection

        return x


class GatedTransformerWithRMSNormAndRegisters(GatedTransformerWithRMSNorm):
    '''Transformer with RMSNorm using Gated Attention and register tokens.

    Proposed in 'Vision Transformers Need Registers' (arXiv:2309.16588),
    register tokens improve feature quality by acting as global sinks.

    Args:
        num_cls: Size of the vocabulary.
        len_context: Maximum sequence length.
        num_layer: Number of transformer blocks. Defaults to 12.
        heads: Number of attention heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Dimension of the final CLS projection. Defaults to 512.
        attn_mask: Attention mask. Defaults to None.
        num_registers: Number of register tokens to append. Defaults to 2.

    Attributes:
        dim (int): Model dimension.
        num_layer (int): Number of layers.
        embedding_token (nn.Embedding): Token embedding layer.
        embedding_position (nn.Parameter): Learnable positional encoding.
        cls_token (nn.Parameter): Learnable classification token.
        registers (nn.Parameter): Learnable register tokens.
        resblocks (nn.Sequential): Stack of residual attention blocks.
        norm_final (nn.RMSNorm): Final root mean square normalization.
        projection (nn.Parameter): Projection matrix for the CLS token.
    '''

    def __init__(
        self,
        num_cls: int,
        len_context: int,
        num_layer: int = 12,
        heads: int = 8,
        dim: int = 512,
        dim_out: int = 512,
        attn_mask: torch.Tensor = None,
        num_registers: int = 2,
    ) -> None:
        super().__init__(
            num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
        )

        self.registers = nn.Parameter(torch.zeros(num_registers, 1, dim))
        nn.init.normal_(self.registers, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes text tokens into a vector representation with registers.

        Args:
            x: Text token IDs (size_batch, len_context).

        Returns:
            Projected CLS token embedding (size_batch, dim_out).
        '''
        x = self.embedding_token(x)
        x = x + self.embedding_position
        x = x.permute(1, 0, 2)  # nld -> lnd

        batch_size = x.shape[1]
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        registers = self.registers.expand(-1, batch_size, -1)

        # concatenate: [cls, registers, sequence]
        x = torch.cat([cls_tokens, registers, x], dim=0)

        x = self.resblocks(x)

        x = x.permute(1, 0, 2)  # lnd -> nld
        x = self.norm_final(x)

        # extract cls token only (index 0)
        x = x[:, 0, :] @ self.projection

        return x


def get_encoder_txt(
    arch_txt: str,
    num_cls: int,
    len_context: int,
    num_layer: int = 3,
    heads: int = 8,
    dim: int = 512,
    dim_out: int = 512,
    attn_mask: torch.Tensor = None,
) -> nn.Module:
    '''Factory function for text encoders.

    Args:
        arch_txt: Architecture key. One of: 'trans', 'trans_rms',
            'trans_gated', 'trans_rms_gated', 'trans_gated_reg',
            'trans_rms_gated_reg'.
        num_cls: Vocabulary size.
        len_context: Max context length.
        num_layer: Number of layers. Defaults to 3.
        heads: Number of heads. Defaults to 8.
        dim: Model dimension. Defaults to 512.
        dim_out: Output dimension. Defaults to 512.
        attn_mask: Optional attention mask. Defaults to None.

    Returns:
        Initialized Transformer module.
    '''
    match arch_txt:
        case 'trans':
            return Transformer(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
        case 'trans_rms':
            return TransformerWithRMSNorm(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
        case 'trans_gated':
            return GatedTransformer(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
        case 'trans_rms_gated':
            return GatedTransformerWithRMSNorm(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
        case 'trans_gated_reg':
            return GatedTransformerWithRegisters(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
        case 'trans_rms_gated_reg':
            return GatedTransformerWithRMSNormAndRegisters(
                num_cls, len_context, num_layer, heads, dim, dim_out, attn_mask
            )
