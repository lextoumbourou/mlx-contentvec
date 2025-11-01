import mlx.core as mx
import mlx.nn as nn

from .modules.group_norm import Fp32GroupNorm, GroupNormMasked


def lengths_to_padding_mask(lengths: mx.array) -> mx.array:
    """
    Convert lengths to a padding mask.

    Args:
        lengths: Array of shape (batch_size,) containing sequence lengths
    Returns:
        Padding mask of shape (batch_size, max_length) where True indicates padding
    """
    max_length = int(mx.max(lengths).item())

    # Create range array [0, 1, 2, ..., max_length-1]
    range_array = mx.arange(max_length).reshape(1, -1)  # (1, max_length)
    lengths_expanded = lengths.reshape(-1, 1)  # (batch_size, 1)

    # Create mask: True where position >= length (i.e., padding positions)
    mask = range_array >= lengths_expanded

    return mask


class ConvBlock(nn.Module):
    """A single convolutional block with optional normalization."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        stride: int,
        dropout: float = 0.0,
        is_layer_norm: bool = False,
        is_group_norm: bool = False,
        conv_bias: bool = False,
        mode: str = "default",
    ):
        super().__init__()
        self.mode = mode
        self.is_group_norm_masked = is_group_norm and mode == "group_norm_masked"

        # Convolutional layer
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, bias=conv_bias)

        # Initialize with Kaiming normal
        fan_in = n_in * kernel_size
        std = mx.sqrt(2.0 / fan_in)
        self.conv.weight = mx.random.normal(self.conv.weight.shape) * std

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Normalization
        self.norm = None
        if is_layer_norm:
            self.norm = nn.LayerNorm(n_out)
            self.use_transpose = True
        elif is_group_norm:
            if mode == "group_norm_masked":
                self.norm = GroupNormMasked(n_out, n_out, affine=True)
            else:
                self.norm = Fp32GroupNorm(n_out, n_out, affine=True)
            self.use_transpose = False
        else:
            self.use_transpose = False

        self.activation = nn.GELU()

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        # Input x is in PyTorch format (B, C, L), convert to MLX format (B, L, C)
        x = x.transpose(0, 2, 1)  # (B, C, L) -> (B, L, C)

        # Apply conv (MLX Conv1d expects (B, L, C))
        x = self.conv(x)

        # Convert back to (B, C, L) for normalization
        x = x.transpose(0, 2, 1)  # (B, L, C) -> (B, C, L)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.norm is not None:
            if self.use_transpose:
                # For LayerNorm: transpose to (B, L, C), apply norm, transpose back
                x = x.transpose(0, 2, 1)  # (B, C, L) -> (B, L, C)
                x = self.norm(x)
                x = x.transpose(0, 2, 1)  # (B, L, C) -> (B, C, L)
            elif self.is_group_norm_masked:
                # For GroupNormMasked: pass mask
                x = self.norm(x, mask)
            else:
                # For regular GroupNorm
                x = self.norm(x)

        x = self.activation(x)

        return x


class ConvFeatureExtractionModel(nn.Module):
    """
    Convolutional feature extraction model for audio processing.

    This model applies a series of 1D convolutions to extract features from raw audio.
    """

    def __init__(
        self,
        conv_layers: list[tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        """
        Args:
            conv_layers: List of tuples (dim, kernel_size, stride) defining each conv layer
            dropout: Dropout probability
            mode: One of "default", "group_norm_masked", or "layer_norm"
            conv_bias: Whether to use bias in convolutions
        """
        super().__init__()

        assert mode in {"default", "group_norm_masked", "layer_norm"}
        self.mode = mode

        in_d = 1
        self.conv_layers = []

        for i, (dim, k, stride) in enumerate(conv_layers):
            assert len(conv_layers[i]) == 3, f"invalid conv definition: {conv_layers[i]}"

            if i == 0:
                self.first_conv_params = (k, stride)  # Store for mask computation

            block = ConvBlock(
                n_in=in_d,
                n_out=dim,
                kernel_size=k,
                stride=stride,
                dropout=dropout,
                is_layer_norm=(mode == "layer_norm"),
                is_group_norm=((mode == "default" or mode == "group_norm_masked") and i == 0),
                conv_bias=conv_bias,
                mode=mode,
            )

            self.conv_layers.append(block)
            in_d = dim

    def __call__(self, x: mx.array, padding_mask: mx.array = None) -> mx.array:
        """
        Args:
            x: Input tensor of shape (batch_size, time)
            padding_mask: Optional padding mask of shape (batch_size, time)
                where True indicates padding positions
        Returns:
            Feature tensor of shape (batch_size, channels, time')
        """
        # Add channel dimension: (B, T) -> (B, 1, T)
        if len(x.shape) == 2:
            x = mx.expand_dims(x, 1)

        # Convert padding mask for first layer if using group_norm_masked
        numeric_mask = None
        if self.mode == "group_norm_masked" and padding_mask is not None:
            # padding_mask is True for padding positions
            # Convert to numeric: 1 for valid, 0 for padding
            numeric_mask = (~padding_mask).astype(mx.float32)

        for i, conv_layer in enumerate(self.conv_layers):
            if i == 0 and self.mode == "group_norm_masked":
                # Update mask for the first conv layer output
                if padding_mask is not None:
                    k, stride = self.first_conv_params
                    # Compute new lengths after convolution
                    lengths_org = mx.sum(~padding_mask, axis=1)
                    lengths = mx.floor(((lengths_org - k) / stride) + 1)
                    # Create new padding mask
                    new_padding_mask = lengths_to_padding_mask(lengths)
                    numeric_mask = (~new_padding_mask).astype(mx.float32)

                x = conv_layer(x, numeric_mask)
            else:
                x = conv_layer(x)

        return x
