import math

import mlx.core as mx
import mlx.nn as nn


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention for MLX.

    Adapted from fairseq's MultiheadAttention for use with MLX.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int | None = None,
        vdim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        # Projection layers
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = mx.zeros((1, 1, embed_dim))
            self.bias_v = mx.zeros((1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            gain = 1 / math.sqrt(2)
            self.k_proj.weight = self._xavier_uniform(self.k_proj.weight, gain)
            self.v_proj.weight = self._xavier_uniform(self.v_proj.weight, gain)
            self.q_proj.weight = self._xavier_uniform(self.q_proj.weight, gain)
        else:
            self.k_proj.weight = self._xavier_uniform(self.k_proj.weight)
            self.v_proj.weight = self._xavier_uniform(self.v_proj.weight)
            self.q_proj.weight = self._xavier_uniform(self.q_proj.weight)

        self.out_proj.weight = self._xavier_uniform(self.out_proj.weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias = mx.zeros_like(self.out_proj.bias)

        if self.bias_k is not None:
            self.bias_k = self._xavier_normal(self.bias_k)
        if self.bias_v is not None:
            self.bias_v = self._xavier_normal(self.bias_v)

    def _xavier_uniform(self, array: mx.array, gain: float = 1.0) -> mx.array:
        """Xavier uniform initialization."""
        fan_in, fan_out = array.shape
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        return mx.random.uniform(-a, a, array.shape)

    def _xavier_normal(self, array: mx.array, gain: float = 1.0) -> mx.array:
        """Xavier normal initialization."""
        fan_in, fan_out = array.shape
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return mx.random.normal(array.shape) * std

    def __call__(
        self,
        query: mx.array,
        key: mx.array | None = None,
        value: mx.array | None = None,
        key_padding_mask: mx.array | None = None,
        attn_mask: mx.array | None = None,
        need_weights: bool = True,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (tgt_len, batch_size, embed_dim)
            key: Key tensor of shape (src_len, batch_size, embed_dim)
            value: Value tensor of shape (src_len, batch_size, embed_dim)
            key_padding_mask: Mask for padding keys of shape (batch_size, src_len)
                where padding elements are indicated by 1s
            attn_mask: Attention mask of shape (tgt_len, src_len)
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: shape (tgt_len, batch_size, embed_dim)
            - attention_weights: shape (batch_size, tgt_len, src_len) or None
        """
        tgt_len, bsz, embed_dim = query.shape
        src_len = tgt_len

        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        if key is not None:
            src_len, key_bsz, _ = key.shape
            assert key_bsz == bsz
            assert value is not None
            assert src_len, bsz == value.shape[:2]

        # Compute Q, K, V projections
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling

        # Add bias to key and value if specified
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = mx.concatenate([k, mx.broadcast_to(self.bias_k, (1, bsz, self.embed_dim))], axis=0)
            v = mx.concatenate([v, mx.broadcast_to(self.bias_v, (1, bsz, self.embed_dim))], axis=0)
            if attn_mask is not None:
                attn_mask = mx.concatenate([attn_mask, mx.zeros((attn_mask.shape[0], 1))], axis=1)
            if key_padding_mask is not None:
                key_padding_mask = mx.concatenate(
                    [key_padding_mask, mx.zeros((key_padding_mask.shape[0], 1))],
                    axis=1,
                )

        # Reshape for multi-head attention
        # (tgt_len, bsz, embed_dim) -> (tgt_len, bsz * num_heads, head_dim) -> (bsz * num_heads, tgt_len, head_dim)
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        if k is not None:
            src_len = k.shape[0]
            k = k.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if v is not None:
            v = v.reshape(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Add zero attention if specified
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = mx.concatenate([k, mx.zeros((k.shape[0], 1, k.shape[2]))], axis=1)
            v = mx.concatenate([v, mx.zeros((v.shape[0], 1, v.shape[2]))], axis=1)
            if attn_mask is not None:
                attn_mask = mx.concatenate([attn_mask, mx.zeros((attn_mask.shape[0], 1))], axis=1)
            if key_padding_mask is not None:
                key_padding_mask = mx.concatenate(
                    [
                        key_padding_mask,
                        mx.zeros((key_padding_mask.shape[0], 1)),
                    ],
                    axis=1,
                )

        # Compute attention weights
        # (bsz * num_heads, tgt_len, head_dim) @ (bsz * num_heads, head_dim, src_len)
        # -> (bsz * num_heads, tgt_len, src_len)
        attn_weights = mx.matmul(q, k.transpose(0, 2, 1))

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        # Apply attention mask
        if attn_mask is not None:
            attn_mask = mx.expand_dims(attn_mask, 0)
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # Reshape mask for broadcasting
            # (bsz, src_len) -> (bsz, 1, 1, src_len) -> (bsz, num_heads, 1, src_len)
            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask_expanded = mx.expand_dims(mx.expand_dims(key_padding_mask, 1), 2)
            # Set attention weights to -inf where padding mask is 1
            attn_weights = mx.where(
                key_padding_mask_expanded.astype(mx.bool_), mx.full(attn_weights.shape, -1e9), attn_weights
            )
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        # Apply softmax
        attn_weights_float = mx.softmax(attn_weights.astype(mx.float32), axis=-1)
        attn_weights = attn_weights_float.astype(attn_weights.dtype)

        # Apply dropout
        if self.dropout > 0.0:
            # Note: MLX doesn't have a built-in dropout in the same way as PyTorch
            # For inference, dropout is typically disabled
            # For training, you would need to implement dropout here
            pass

        # Apply attention to values
        # (bsz * num_heads, tgt_len, src_len) @ (bsz * num_heads, src_len, head_dim)
        # -> (bsz * num_heads, tgt_len, head_dim)
        assert v is not None
        attn_output = mx.matmul(attn_weights, v)

        assert list(attn_output.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # Reshape back
        # (bsz * num_heads, tgt_len, head_dim) -> (tgt_len, bsz * num_heads, head_dim) -> (tgt_len, bsz, embed_dim)
        attn_output = attn_output.transpose(1, 0, 2).reshape(tgt_len, bsz, embed_dim)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Prepare attention weights for return
        attn_weights_out: mx.array | None = None
        if need_weights:
            attn_weights_out = attn_weights_float.reshape(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0, 2, 3)
            # Average attention weights over heads
            attn_weights_out = mx.mean(attn_weights_out, axis=0)

        return attn_output, attn_weights_out
