"""
Test to verify TransformerEncoder_1 outputs match between PyTorch and MLX.

This test ensures that the MLX implementation produces the same outputs as the PyTorch
implementation when given the same weights and inputs.

Run with: pytest tests/test_transformer_encoder_pytorch.py -v

Note: Requires PyTorch to be installed (available in dev dependencies).
Install with: pip install -e ".[dev]"
"""

import sys
import types
from pathlib import Path

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import mlx.core as mx
import pytest

# Import MLX implementation
from mlx_contentvec.transformer_encoder import TransformerEncoder_1 as TransformerEncoder_1_MLX

# Import PyTorch implementation from vendor directory
if TORCH_AVAILABLE:
    vendor_dir = Path(__file__).parent.parent / "vendor"
    sys.path.insert(0, str(vendor_dir))
    try:
        from transformer_encoder_pytorch_standalone import TransformerEncoder_1 as TransformerEncoder_1_PT

        PYTORCH_MODEL_AVAILABLE = True
    except ImportError:
        PYTORCH_MODEL_AVAILABLE = False
else:
    PYTORCH_MODEL_AVAILABLE = False


# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not PYTORCH_MODEL_AVAILABLE,
    reason="PyTorch or PyTorch model implementation not available",
)


def create_fairseq_mock():
    """Create minimal mock for fairseq to allow unpickling weights."""
    fairseq = types.ModuleType("fairseq")
    fairseq.data = types.ModuleType("fairseq.data")
    fairseq.data.dictionary = types.ModuleType("fairseq.data.dictionary")

    class Dictionary:
        def __init__(self):
            pass

    fairseq.data.dictionary.Dictionary = Dictionary
    sys.modules["fairseq"] = fairseq
    sys.modules["fairseq.data"] = fairseq.data
    sys.modules["fairseq.data.dictionary"] = fairseq.data.dictionary


def load_pytorch_model():
    """Load PyTorch model with HuBERT base weights."""
    model = TransformerEncoder_1_PT(
        embedding_dim=768,
        ffn_embedding_dim=3072,
        num_attention_heads=12,
        dropout=0.0,  # Set to 0 for deterministic comparison
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_fn="gelu",
        layer_norm_first=True,
        conv_pos=128,
        conv_pos_groups=16,
        encoder_layers=12,
    )

    # Load weights
    weights_path = Path(__file__).parent.parent / "vendor" / "ref_weights" / "hubert_base.pt"
    if not weights_path.exists():
        pytest.skip(f"Weights file not found at {weights_path}")

    # Create fairseq mock to allow unpickling
    create_fairseq_mock()

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Load encoder weights
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            # Remove "encoder." prefix
            new_key = key[len("encoder.") :]
            model_state_dict[new_key] = value

    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    return model


def load_mlx_model():
    """Load MLX model with HuBERT base weights."""
    model = TransformerEncoder_1_MLX(
        embedding_dim=768,
        ffn_embedding_dim=3072,
        num_attention_heads=12,
        dropout=0.0,  # Set to 0 for deterministic comparison
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_fn="gelu",
        layer_norm_first=True,
        conv_pos=128,
        conv_pos_groups=16,
        encoder_layers=12,
        encoder_layers_1=0,  # No speaker-conditioned layers for HuBERT
    )

    # Load weights
    weights_path = Path(__file__).parent.parent / "vendor" / "ref_weights" / "hubert_base.pt"
    if not weights_path.exists():
        pytest.skip(f"Weights file not found at {weights_path}")

    # Create fairseq mock to allow unpickling
    create_fairseq_mock()

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Convert and load weights into MLX model
    for key, value in state_dict.items():
        if not key.startswith("encoder."):
            continue

        # Remove "encoder." prefix
        key = key[len("encoder.") :]
        # Convert to float32 to avoid float16 numerical issues
        value_np = value.cpu().float().numpy()

        # Navigate to the correct module and parameter
        parts = key.split(".")

        # Handle positional convolution with weight normalization
        if parts[0] == "pos_conv":
            # pos_conv.0.weight_g, pos_conv.0.weight_v, pos_conv.0.bias
            if parts[1] == "0":  # The conv layer with weight norm
                # In MLX, we have WeightNorm wrapping the conv
                # pos_conv is a Sequential containing: [WeightNorm(Conv1d), SamePad, GELU]
                weight_norm_layer = model.pos_conv.layers[0]  # This is WeightNorm
                if parts[2] == "weight_g":
                    # Weight_g shape in PyTorch: (1, 1, 128) - one scale per kernel position
                    # Since we transposed weight_v, kernel moved from dim 2 to dim 1
                    # So we need to transpose g from (1, 1, 128) to (1, 128, 1)
                    value_np = value_np.transpose(0, 2, 1)  # (1, 1, 128) -> (1, 128, 1)
                    weight_norm_layer.g = mx.array(value_np)
                elif parts[2] == "weight_v":
                    # Weight_v shape: (768, 48, 128) in PyTorch (out, in/groups, kernel)
                    # MLX Conv1d expects: (out, kernel, in/groups)
                    value_np = value_np.transpose(0, 2, 1)
                    # WeightNorm has direct attribute 'v'
                    weight_norm_layer.v = mx.array(value_np)
                elif parts[2] == "bias":
                    # The bias is on the wrapped Conv1d module
                    weight_norm_layer.module.bias = mx.array(value_np)

        # Handle layer norm
        elif parts[0] == "layer_norm":
            if parts[1] == "weight":
                model.layer_norm.weight = mx.array(value_np)
            elif parts[1] == "bias":
                model.layer_norm.bias = mx.array(value_np)

        # Handle transformer layers
        elif parts[0] == "layers":
            layer_idx = int(parts[1])
            layer = model.layers[layer_idx]

            # Self attention layer norm
            if parts[2] == "self_attn_layer_norm":
                if parts[3] == "weight":
                    layer.self_attn_layer_norm.weight = mx.array(value_np)
                elif parts[3] == "bias":
                    layer.self_attn_layer_norm.bias = mx.array(value_np)

            # Final layer norm
            elif parts[2] == "final_layer_norm":
                if parts[3] == "weight":
                    layer.final_layer_norm.weight = mx.array(value_np)
                elif parts[3] == "bias":
                    layer.final_layer_norm.bias = mx.array(value_np)

            # Self attention
            elif parts[2] == "self_attn":
                proj_name = parts[3]  # q_proj, k_proj, v_proj, out_proj
                param_name = parts[4]  # weight, bias

                proj = getattr(layer.self_attn, proj_name)
                if param_name == "weight":
                    # PyTorch: (out_features, in_features)
                    # MLX: (out_features, in_features) - same!
                    setattr(proj, "weight", mx.array(value_np))
                elif param_name == "bias":
                    setattr(proj, "bias", mx.array(value_np))

            # Feed-forward layers
            elif parts[2] in ["fc1", "fc2"]:
                fc = getattr(layer, parts[2])
                if parts[3] == "weight":
                    # PyTorch: (out_features, in_features)
                    # MLX: (out_features, in_features) - same!
                    fc.weight = mx.array(value_np)
                elif parts[3] == "bias":
                    fc.bias = mx.array(value_np)

    return model


def test_transformer_encoder_single_batch():
    """Test with single batch input."""
    pt_model = load_pytorch_model()
    mlx_model = load_mlx_model()

    # Create input - transformer expects (batch, seq_len, embed_dim)
    np.random.seed(42)
    batch_size, seq_len, embed_dim = 1, 100, 768
    input_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        input_pt = torch.from_numpy(input_np)
        output_pt = pt_model(input_pt, padding_mask=None)
    output_pt_np = output_pt.cpu().numpy()

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx, _ = mlx_model(input_mlx, spk_emb=None, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Check shapes match
    assert output_pt_np.shape == output_mlx_np.shape, (
        f"Shapes don't match: PyTorch {output_pt_np.shape} vs MLX {output_mlx_np.shape}"
    )

    # Check values are close
    # Expected precision: ~1e-6 for single-precision float operations
    atol, rtol = 1e-5, 1e-4
    max_diff = np.abs(output_pt_np - output_mlx_np).max()
    print(f"Max difference: {max_diff}")

    assert np.allclose(output_pt_np, output_mlx_np, atol=atol, rtol=rtol), (
        f"Outputs don't match within tolerance (atol={atol}, rtol={rtol}). Max diff: {max_diff}"
    )


def test_transformer_encoder_batch():
    """Test with batch input."""
    pt_model = load_pytorch_model()
    mlx_model = load_mlx_model()

    # Create input
    np.random.seed(123)
    batch_size, seq_len, embed_dim = 4, 200, 768
    input_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        input_pt = torch.from_numpy(input_np)
        output_pt = pt_model(input_pt, padding_mask=None)
    output_pt_np = output_pt.cpu().numpy()

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx, _ = mlx_model(input_mlx, spk_emb=None, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Check shapes match
    assert output_pt_np.shape == output_mlx_np.shape, (
        f"Shapes don't match: PyTorch {output_pt_np.shape} vs MLX {output_mlx_np.shape}"
    )

    # Check values are close
    # Expected precision: ~1e-6 for single-precision float operations
    atol, rtol = 1e-5, 1e-4
    max_diff = np.abs(output_pt_np - output_mlx_np).max()
    print(f"Max difference: {max_diff}")

    assert np.allclose(output_pt_np, output_mlx_np, atol=atol, rtol=rtol), (
        f"Outputs don't match within tolerance (atol={atol}, rtol={rtol}). Max diff: {max_diff}"
    )


def test_transformer_encoder_shape():
    """Test that output shape is correct."""
    mlx_model = load_mlx_model()

    # Create input
    batch_size, seq_len, embed_dim = 2, 50, 768
    input_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx, _ = mlx_model(input_mlx, spk_emb=None, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Expected shape: same as input
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_mlx_np.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output_mlx_np.shape}"


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Install dev dependencies with: pip install -e '.[dev]'")
        sys.exit(1)

    if not PYTORCH_MODEL_AVAILABLE:
        print("PyTorch model implementation not found. Make sure transformer_encoder_pytorch_standalone.py exists.")
        sys.exit(1)

    print("Running test_transformer_encoder_shape...")
    test_transformer_encoder_shape()
    print("PASSED\n")

    print("Running test_transformer_encoder_single_batch...")
    test_transformer_encoder_single_batch()
    print("PASSED\n")

    print("Running test_transformer_encoder_batch...")
    test_transformer_encoder_batch()
    print("PASSED\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
