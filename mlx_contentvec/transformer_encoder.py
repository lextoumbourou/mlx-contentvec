import math

import mlx.core as mx
import mlx.nn as nn

from .modules.cond_layer_norm import CondLayerNorm
from .modules.multihead_attention import MultiheadAttention
from .modules.weight_norm import WeightNorm


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    Initializes Linear and MultiheadAttention layers with normal distribution (mean=0, std=0.02).
    """
    if isinstance(module, nn.Linear):
        # Initialize with normal distribution
        module.weight = mx.random.normal(module.weight.shape, scale=0.02)
        if module.bias is not None:
            module.bias = mx.zeros_like(module.bias)
    elif isinstance(module, MultiheadAttention):
        # Initialize query, key, value projections
        module.q_proj.weight = mx.random.normal(module.q_proj.weight.shape, scale=0.02)
        module.k_proj.weight = mx.random.normal(module.k_proj.weight.shape, scale=0.02)
        module.v_proj.weight = mx.random.normal(module.v_proj.weight.shape, scale=0.02)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return nn.relu
    elif activation == "gelu":
        return nn.gelu
    elif activation == "tanh":
        return mx.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"activation function {activation} not supported")


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models.
    This is the regular version without speaker conditioning.
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

    def __call__(
        self,
        x: mx.array,
        self_attn_mask: mx.array | None = None,
        self_attn_padding_mask: mx.array | None = None,
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


class TransformerSentenceEncoderLayer_1(nn.Module):
    """
    Implements a Transformer Encoder Layer with speaker conditioning via CondLayerNorm.
    Used in ContentVec for speaker-conditioned encoding.
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
        dim_spk: int = 256,
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

        # layer norm associated with the self attention layer (conditional)
        self.self_attn_layer_norm = CondLayerNorm(self.embedding_dim, dim_spk=dim_spk)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN (conditional)
        self.final_layer_norm = CondLayerNorm(self.embedding_dim, dim_spk=dim_spk)

    def __call__(
        self,
        x: mx.array,
        spk_emb: mx.array,
        self_attn_mask: mx.array | None = None,
        self_attn_padding_mask: mx.array | None = None,
        need_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        Speaker embedding is used for conditional layer normalization.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x, spk_emb)
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
            x = self.final_layer_norm(x, spk_emb)
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

            x = self.self_attn_layer_norm(x, spk_emb)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x, spk_emb)

        return x, attn


class SamePad(nn.Module):
    """
    Padding layer that maintains the same output size as input for convolutions.
    Used with convolutions to ensure output length matches input length.
    For MLX Conv1d with input format (batch, length, channels).
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        # For "same" padding: padding = (kernel_size - 1) // 2
        # But we need to handle the case where the kernel is even
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.remove > 0:
            # Remove from the sequence length dimension (index 1 in MLX format: batch, length, channels)
            x = x[:, : -self.remove, :]
        return x


class TransformerEncoder_1(nn.Module):
    """
    Transformer encoder with positional convolution and optional speaker conditioning.
    Combines regular transformer layers with speaker-conditioned layers for ContentVec.
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
        encoder_layers_1: int = 0,
        layerdrop: float = 0.0,
        dim_spk: int = 256,
    ):
        super().__init__()

        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.encoder_layers_1 = encoder_layers_1

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
        pos_conv.weight = mx.random.normal(pos_conv.weight.shape, scale=std)
        if pos_conv.bias is not None:
            pos_conv.bias = mx.zeros_like(pos_conv.bias)

        # Apply weight normalization to positional convolution
        # Note: HuBERT normalizes along the kernel dimension (dim=2 in PyTorch format)
        # In MLX Conv1d format (out, kernel, in), kernel is at dim=1
        pos_conv_wn = WeightNorm(pos_conv, name="weight", dim=1)

        # Wrap in Sequential with SamePad and GELU
        self.pos_conv = nn.Sequential(pos_conv_wn, SamePad(conv_pos), nn.GELU())

        # Build transformer layers
        self.layers = []

        # Regular transformer layers
        for _ in range(encoder_layers):
            self.layers.append(
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
            )

        # Speaker-conditioned transformer layers
        for _ in range(encoder_layers_1):
            self.layers.append(
                TransformerSentenceEncoderLayer_1(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                    dim_spk=dim_spk,
                )
            )

        self.layer_norm_first = layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        if encoder_layers_1 > 0:
            self.cond_layer_norm = CondLayerNorm(self.embedding_dim, dim_spk=dim_spk)
        self.layerdrop = layerdrop
        self.num_layers = encoder_layers

        # Dropout applied after positional conv (stored as attribute to inherit eval mode)
        self.dropout_layer = nn.Dropout(dropout)

        # Apply BERT-style parameter initialization
        self._apply_init(init_bert_params)

    def _apply_init(self, fn):
        """Apply initialization function to all submodules"""
        # Apply to self
        fn(self)
        # Recursively apply to all children
        for child in self.children().values():
            if hasattr(child, "_apply_init"):
                child._apply_init(fn)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if hasattr(item, "_apply_init"):
                        item._apply_init(fn)
                    else:
                        fn(item)
            else:
                fn(child)

    def __call__(
        self,
        x: mx.array,
        spk_emb: mx.array | None = None,
        padding_mask: mx.array | None = None,
        layer: int | None = None,
        tap: bool = False,
    ):
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            spk_emb: Speaker embedding of shape (batch, dim_spk) for conditional layers
            padding_mask: Padding mask of shape (batch, seq_len) where True indicates padding
            layer: If specified, return features from this layer index
            tap: If True, return intermediate layer results even if layer is None
        """
        x, layer_results = self.extract_features(x, spk_emb, padding_mask, layer, tap)

        if self.layer_norm_first and layer is None:
            if self.encoder_layers_1 > 0:
                x = self.cond_layer_norm(x, spk_emb)
            else:
                x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x: mx.array,
        spk_emb: mx.array | None = None,
        padding_mask: mx.array | None = None,
        tgt_layer: int | None = None,
        tap: bool = False,
    ):
        """
        Extract features from the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            spk_emb: Speaker embedding for conditional layers
            padding_mask: Padding mask where True indicates padding
            tgt_layer: If specified, stop at this layer and return its output
            tap: If True, collect intermediate layer results
        """
        if padding_mask is not None:
            # Zero out padded positions
            x = mx.where(mx.expand_dims(padding_mask, -1), 0, x)

        # Apply positional convolution (includes WeightNorm, SamePad, and GELU)
        # Input: (batch, seq_len, embed_dim)
        # MLX Conv1d expects: (batch, seq_len, in_channels) - same format, no transpose needed!
        x_conv = self.pos_conv(x)
        x = x + x_conv

        # Convert B x T x C -> T x B x C
        x = mx.transpose(x, [1, 0, 2])

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout_layer(x)

        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            # Apply layer dropout only to regular (non-speaker-conditioned) layers during training
            # For inference (typical MLX use case), we process all layers
            # Note: PyTorch checks (not self.training or (dropout_probability > self.layerdrop)) and (i < self.num_layers)
            # We simplify to always process layers since MLX doesn't have a training flag

            if i < self.num_layers:
                # Regular transformer layer (no speaker conditioning)
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                # Collect intermediate results if requested
                if tgt_layer is not None or tap:
                    layer_results.append(mx.transpose(x, [1, 0, 2]))
            else:
                # Speaker-conditioned layer (encoder_layers_1)
                x, z = layer(x, spk_emb, self_attn_padding_mask=padding_mask, need_weights=False)

            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # Convert T x B x C -> B x T x C
        x = mx.transpose(x, [1, 0, 2])

        return x, layer_results
