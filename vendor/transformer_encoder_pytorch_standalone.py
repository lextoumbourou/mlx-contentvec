"""
Standalone PyTorch implementation of TransformerEncoder_1 for reference testing.
This is extracted from the ContentVec codebase to allow direct comparison with MLX implementation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePad(nn.Module):
    """
    Padding layer that maintains the same output size as input for convolutions.
    """

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"activation function {activation} not supported")


class MultiheadAttention(nn.Module):
    """Simple multihead attention implementation for standalone use."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
    ):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn, attn_weights.mean(dim=1)
        else:
            return attn, None


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=need_weights,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class TransformerEncoder_1(nn.Module):
    """
    Transformer encoder with positional convolution.
    Simplified version without speaker conditioning for HuBERT-base compatibility.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        layer_norm_first: bool = True,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        encoder_layers: int = 12,
    ):
        super().__init__()

        self.dropout = dropout
        self.embedding_dim = embedding_dim

        # Positional convolution
        pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )

        # Initialize positional convolution weights
        dropout_init = 0
        std = math.sqrt((4 * (1.0 - dropout_init)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(pos_conv.weight, mean=0, std=std)
        nn.init.constant_(pos_conv.bias, 0)

        # Apply weight normalization to positional convolution
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), nn.GELU())

        # Build transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.layer_norm_first = layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x, padding_mask=None):
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            padding_mask: Padding mask of shape (batch, seq_len) where True indicates padding

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):
        """
        Extract features from the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            padding_mask: Padding mask where True indicates padding
        """
        if padding_mask is not None:
            # Zero out padded positions
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # Apply positional convolution
        # Input: (batch, seq_len, embed_dim)
        # Conv1d expects: (batch, embed_dim, seq_len)
        x_conv = self.pos_conv(x.transpose(1, 2))
        # Transpose back: (batch, seq_len, embed_dim)
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        # Convert B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)

        # Convert T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x
