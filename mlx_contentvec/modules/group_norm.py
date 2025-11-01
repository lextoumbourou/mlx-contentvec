import mlx.core as mx
import mlx.nn as nn


class Fp32GroupNorm(nn.Module):
    """
    Group Normalization computed in FP32.
    In MLX, we'll just use regular computation.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor of shape (B, C, L)
        Returns:
            Normalized tensor of shape (B, C, L)
        """
        B, C, L = x.shape
        assert C == self.num_channels
        assert C % self.num_groups == 0

        # Reshape to (B, num_groups, C // num_groups, L)
        x = x.reshape(B, self.num_groups, C // self.num_groups, L)

        # Compute mean and variance over the last two dimensions (channels within group and length)
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.var(x, axis=(2, 3), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(B, C, L)

        # Apply affine transformation
        if self.affine:
            x = x * self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)

        return x


class GroupNormMasked(nn.Module):
    """
    Group Normalization with masking support for variable-length sequences.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        """
        Args:
            x: Input tensor of shape (B, C, L)
            mask: Binary mask of shape (B, L) where 1 indicates valid positions
        Returns:
            Normalized tensor of shape (B, C, L)
        """
        B, C, L = x.shape
        assert C == self.num_channels
        assert C % self.num_groups == 0

        # Reshape to (B, num_groups, C // num_groups, L)
        x = x.reshape(B, self.num_groups, C // self.num_groups, L)

        if mask is None:
            mask = mx.ones((B, L))
        else:
            # mask is (B, L), expand to match x shape
            mask = mask.reshape(B, 1, 1, L)

        # Apply mask
        x_masked = x * mask
        lengths = mx.sum(mask, axis=3, keepdims=True)

        # For numerical stability, ensure we're not dividing by zero
        lengths = mx.maximum(lengths, mx.array(1.0))

        # Compute mean considering mask
        # Note: assuming C // num_groups == 1 (which is the case in the original code)
        assert C // self.num_groups == 1, "Current implementation assumes C // num_groups == 1"

        mean_ = mx.mean(x_masked, axis=3, keepdims=True)
        mean = mean_ * L / lengths

        # Compute variance considering mask
        var_ = mx.var(x_masked, axis=3, keepdims=True)
        var = (var_ + mean_**2) * L / lengths - mean**2
        var = var + self.eps

        # Normalize
        x = (x - mean) / mx.sqrt(var)

        # Reshape back
        x = x.reshape(B, C, L)

        # Apply affine transformation
        if self.affine:
            x = x * self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)

        return x
