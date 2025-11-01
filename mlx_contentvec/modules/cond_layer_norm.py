import mlx.core as mx
import mlx.nn as nn


class CondLayerNorm(nn.Module):
    """
    Conditional Layer Normalization that takes speaker embeddings as conditioning.
    """

    def __init__(self, dim_last: int, eps: float = 1e-5, dim_spk: int = 256, elementwise_affine: bool = True):
        super().__init__()
        self.dim_last = dim_last
        self.eps = eps
        self.dim_spk = dim_spk
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight_ln = nn.Linear(self.dim_spk, self.dim_last, bias=False)
            self.bias_ln = nn.Linear(self.dim_spk, self.dim_last, bias=False)

            # Initialize weights
            # weight_ln should output ones, bias_ln should output zeros
            self.weight_ln.weight = mx.ones_like(self.weight_ln.weight)
            self.bias_ln.weight = mx.zeros_like(self.bias_ln.weight)

    def __call__(self, input: mx.array, spk_emb: mx.array) -> mx.array:
        # Generate weight and bias from speaker embedding
        weight = self.weight_ln(spk_emb)
        bias = self.bias_ln(spk_emb)

        # Calculate mean and variance over the normalized dimensions
        mean = mx.mean(input, axis=tuple(range(1, len(input.shape))), keepdims=True)
        variance = mx.var(input, axis=tuple(range(1, len(input.shape))), keepdims=True)

        # Normalize
        normalized = (input - mean) / mx.sqrt(variance + self.eps)

        # Apply affine transformation
        return normalized * weight + bias
