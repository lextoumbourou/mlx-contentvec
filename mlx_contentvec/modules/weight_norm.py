import mlx.core as mx
from mlx.nn.layers.base import Module

"""
Weight norm implementation based on unmerged PR https://github.com/ml-explore/mlx/pull/1921
"""

class WeightNorm(Module):
    r"""Applies weight normalization [1] to a module's parameter.

    Weight normalization reparameterizes the weight vectors of a wrapped module
    in terms of a magnitude (scale) and a direction as:

    .. math::

        w = g \frac{v}{||v||}

    where :math:`g` is a scalar parameter and :math:`v` is a vector parameter.
    This reparameterization decouples the length of the weight vectors from their
    direction, which can improve the conditioning of the optimization problem and
    accelerate training.

    Args:
        module (Module): The module to wrap and apply weight normalization to.
        name (str): The name of the parameter to normalize. Default: ``"weight"``.
        dim (int or None): The dimension along which to normalize. If ``None``, the
            weight is normalized by its L2 norm over all dimensions. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> linear = nn.Linear(20, 30)
        >>> weight_norm_linear = nn.WeightNorm(linear)
        >>> x = mx.random.normal((8, 20))
        >>> output = weight_norm_linear(x)

    References:
        [1]: https://arxiv.org/abs/1602.07868
    """

    def __init__(self, module, name="weight", dim=0):
        super().__init__()
        self.module = module
        self.wn_name = name
        params = module.parameters()
        if name not in params:
            raise ValueError(f"Parameter '{name}' not found in module")
        mx.eval(params)
        weight = params[name]
        self.v = mx.array(weight)
        self.wn_module_type = type(module).__name__

        if dim is None:
            self.g = mx.linalg.norm(weight)
            self.wn_axes = []
        else:
            dim = dim if dim >= 0 else weight.ndim + dim
            if dim < 0 or dim >= weight.ndim:
                raise ValueError(
                    f"dim {dim} out of bounds for {weight.ndim} dimensions"
                )
            axes = [i for i in range(weight.ndim) if i != dim]
            if len(axes) > 2:
                reshape_dims = [weight.shape[dim], -1]
                weight_reshaped = mx.reshape(weight, reshape_dims)
                self.g = mx.linalg.norm(weight_reshaped, axis=1, keepdims=True)
                g_shape = [1] * weight.ndim
                g_shape[dim] = weight.shape[dim]
                self.g = mx.reshape(self.g, g_shape)
            elif "Conv" in self.wn_module_type and dim == 0:
                weight_flat = mx.reshape(weight, (weight.shape[0], -1))
                self.g = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
                g_shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
                self.g = mx.reshape(self.g, g_shape)
            else:
                self.g = mx.linalg.norm(weight, axis=tuple(axes), keepdims=True)
            self.wn_axes = axes
        self.wn_dim = dim

    def _extra_repr(self):
        return f"module={self.wn_module_type}, name={self.wn_name}, dim={self.wn_dim}"

    def __call__(self, *args, **kwargs):
        """Apply weight normalization to the wrapped module and then call it."""
        # Compute normalized weight: w = g * (v / ||v||)
        if self.wn_axes:
            # Normalize along specific axes
            # For more than 2 axes or Conv layers, reshape to 2D for norm computation
            if len(self.wn_axes) > 2 or ("Conv" in self.wn_module_type and self.wn_dim == 0):
                # Reshape to [dim_size, -1] for efficient norm computation
                v_shape = self.v.shape
                dim_size = v_shape[self.wn_dim]
                v_reshaped = mx.reshape(self.v, [dim_size, -1])
                v_norm = mx.linalg.norm(v_reshaped, axis=1, keepdims=True)
                v_norm = mx.maximum(v_norm, 1e-5)
                # Reshape norm back to match g shape
                g_shape = [1] * self.v.ndim
                g_shape[self.wn_dim] = dim_size
                v_norm = mx.reshape(v_norm, g_shape)
            else:
                v_norm = mx.linalg.norm(self.v, axis=tuple(self.wn_axes), keepdims=True)
                v_norm = mx.maximum(v_norm, 1e-5)
            normalized_weight = self.g * (self.v / v_norm)
        else:
            # Normalize over all dimensions
            v_norm = mx.linalg.norm(self.v)
            v_norm = mx.maximum(v_norm, 1e-5)  # Prevent division by zero
            normalized_weight = self.g * (self.v / v_norm)

        setattr(self.module, self.wn_name, normalized_weight)
        return self.module(*args, **kwargs)


class WeightNormConv1d:
    r"""Applies a 1D convolution with weight normalization over an input signal.

    This module is a convenience wrapper that combines Conv1d with weight normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: ``1``.
        padding (int): Zero-padding added to both sides of the input. Default: ``0``.
        dilation (int): Spacing between kernel elements. Default: ``1``.
        groups (int): Number of blocked connections from input to output channels. Default: ``1``.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Conv1d module with weight normalization applied.
    """

    def __new__(
        cls,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dim=0,
    ):
        from mlx.nn import Conv1d

        conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        return WeightNorm(conv, "weight", dim)


class WeightNormConv2d:
    r"""Applies a 2D convolution with weight normalization over an input signal.

    This module is a convenience wrapper that combines Conv2d with weight normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution. Default: ``1``.
        padding (int or tuple): Zero-padding added to both sides of the input. Default: ``0``.
        dilation (int or tuple): Spacing between kernel elements. Default: ``1``.
        groups (int): Number of blocked connections from input to output channels. Default: ``1``.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Conv2d module with weight normalization applied.
    """

    def __new__(
        cls,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dim=0,
    ):
        from mlx.nn import Conv2d

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        return WeightNorm(conv, "weight", dim)


class WeightNormLinear:
    r"""Applies a linear transformation with weight normalization to the incoming data.

    This module is a convenience wrapper that combines Linear with weight normalization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Linear module with weight normalization applied.
    """

    def __new__(cls, in_features, out_features, bias=True, dim=0):
        from mlx.nn import Linear

        linear = Linear(in_features, out_features, bias)
        return WeightNorm(linear, "weight", dim)
