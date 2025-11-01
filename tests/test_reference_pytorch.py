"""
Test to verify ConvFeatureExtractionModel outputs match between PyTorch and MLX.

This test ensures that the MLX implementation produces the same outputs as the PyTorch
implementation when given the same weights and inputs.

Run with: pytest tests/test_reference_pytorch.py -v

Note: Requires PyTorch to be installed (available in dev dependencies).
Install with: pip install -e ".[dev]"
"""

import sys
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
from mlx_contentvec.conv_feature_extraction import ConvFeatureExtractionModel as ConvFeatureExtractionModelMLX

# Import PyTorch implementation from vendor directory
if TORCH_AVAILABLE:
    vendor_dir = Path(__file__).parent.parent / "vendor"
    sys.path.insert(0, str(vendor_dir))
    try:
        from conv_feature_pytorch_standalone import ConvFeatureExtractionModel as ConvFeatureExtractionModelPT

        PYTORCH_MODEL_AVAILABLE = True
    except ImportError:
        PYTORCH_MODEL_AVAILABLE = False
else:
    PYTORCH_MODEL_AVAILABLE = False


# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not PYTORCH_MODEL_AVAILABLE, reason="PyTorch or PyTorch model implementation not available"
)


def load_pytorch_model():
    """Load PyTorch model with weights."""
    conv_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    model = ConvFeatureExtractionModelPT(conv_layers=conv_layers, dropout=0.0, mode="default", conv_bias=False)

    # Load weights
    weights_path = Path(__file__).parent.parent / "vendor" / "feature_extractor_weights.pt"
    if not weights_path.exists():
        pytest.skip(f"Weights file not found at {weights_path}")

    weights = torch.load(str(weights_path), map_location="cpu")
    state_dict = {key.replace("feature_extractor.", ""): value for key, value in weights.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_mlx_model():
    """Load MLX model with weights."""
    conv_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    model = ConvFeatureExtractionModelMLX(conv_layers=conv_layers, dropout=0.0, mode="default", conv_bias=False)

    # Load weights
    weights_path = Path(__file__).parent.parent / "vendor" / "feature_extractor_weights.pt"
    if not weights_path.exists():
        pytest.skip(f"Weights file not found at {weights_path}")

    weights_pt = torch.load(str(weights_path), map_location="cpu")

    # Convert and load weights into MLX model
    for key, value in weights_pt.items():
        key = key.replace("feature_extractor.", "")
        parts = key.split(".")

        if len(parts) >= 4:
            layer_idx = int(parts[1])
            sub_idx = int(parts[2])
            param_name = parts[3]

            value_np = value.cpu().numpy()
            layer = model.conv_layers[layer_idx]

            if sub_idx == 0:  # Conv layer
                if param_name == "weight":
                    # PyTorch: (out_channels, in_channels, kernel)
                    # MLX: (out_channels, kernel, in_channels)
                    value_np = value_np.transpose(0, 2, 1)
                    layer.conv.weight = mx.array(value_np)
                elif param_name == "bias":
                    layer.conv.bias = mx.array(value_np)
            elif sub_idx == 2:  # Norm layer
                if param_name == "weight":
                    layer.norm.weight = mx.array(value_np)
                elif param_name == "bias":
                    layer.norm.bias = mx.array(value_np)

    return model


def test_conv_feature_extraction_single_batch():
    """Test with single batch input."""
    pt_model = load_pytorch_model()
    mlx_model = load_mlx_model()

    # Create input
    np.random.seed(42)
    batch_size, seq_len = 1, 16000
    input_np = np.random.randn(batch_size, seq_len).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        input_pt = torch.from_numpy(input_np)
        output_pt = pt_model(input_pt, padding_mask=None)
    output_pt_np = output_pt.cpu().numpy()

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx = mlx_model(input_mlx, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Check shapes match
    assert output_pt_np.shape == output_mlx_np.shape, (
        f"Shapes don't match: PyTorch {output_pt_np.shape} vs MLX {output_mlx_np.shape}"
    )

    # Check values are close
    atol, rtol = 1e-4, 1e-3
    assert np.allclose(output_pt_np, output_mlx_np, atol=atol, rtol=rtol), (
        f"Outputs don't match within tolerance (atol={atol}, rtol={rtol})"
    )


def test_conv_feature_extraction_batch():
    """Test with batch input."""
    pt_model = load_pytorch_model()
    mlx_model = load_mlx_model()

    # Create input
    np.random.seed(123)
    batch_size, seq_len = 4, 32000
    input_np = np.random.randn(batch_size, seq_len).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        input_pt = torch.from_numpy(input_np)
        output_pt = pt_model(input_pt, padding_mask=None)
    output_pt_np = output_pt.cpu().numpy()

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx = mlx_model(input_mlx, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Check shapes match
    assert output_pt_np.shape == output_mlx_np.shape, (
        f"Shapes don't match: PyTorch {output_pt_np.shape} vs MLX {output_mlx_np.shape}"
    )

    # Check values are close
    atol, rtol = 1e-4, 1e-3
    assert np.allclose(output_pt_np, output_mlx_np, atol=atol, rtol=rtol), (
        f"Outputs don't match within tolerance (atol={atol}, rtol={rtol})"
    )


def test_conv_feature_extraction_shape():
    """Test that output shape is correct."""
    load_pytorch_model()
    mlx_model = load_mlx_model()

    # Create input
    batch_size, seq_len = 2, 16000
    input_np = np.random.randn(batch_size, seq_len).astype(np.float32)

    # MLX forward
    input_mlx = mx.array(input_np)
    output_mlx = mlx_model(input_mlx, padding_mask=None)
    output_mlx_np = np.array(output_mlx)

    # Expected shape calculation:
    # Input: 16000
    # Conv1 (kernel=10, stride=5): (16000-10)/5 + 1 = 3199
    # Conv2 (kernel=3, stride=2): (3199-3)/2 + 1 = 1599
    # Conv3 (kernel=3, stride=2): (1599-3)/2 + 1 = 799
    # Conv4 (kernel=3, stride=2): (799-3)/2 + 1 = 399
    # Conv5 (kernel=3, stride=2): (399-3)/2 + 1 = 199
    # Conv6 (kernel=2, stride=2): (199-2)/2 + 1 = 99
    # Conv7 (kernel=2, stride=2): (99-2)/2 + 1 = 49

    expected_shape = (batch_size, 512, 49)
    assert output_mlx_np.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {output_mlx_np.shape}"
    )


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed. Install dev dependencies with: pip install -e '.[dev]'")
        sys.exit(1)

    if not PYTORCH_MODEL_AVAILABLE:
        print("PyTorch model implementation not found. Make sure conv_feature_pytorch_standalone.py exists.")
        sys.exit(1)

    print("Running test_conv_feature_extraction_single_batch...")
    test_conv_feature_extraction_single_batch()
    print("PASSED\n")

    print("Running test_conv_feature_extraction_batch...")
    test_conv_feature_extraction_batch()
    print("PASSED\n")

    print("Running test_conv_feature_extraction_shape...")
    test_conv_feature_extraction_shape()
    print("PASSED\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
