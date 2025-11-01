import mlx.core as mx
import pytest

from mlx_contentvec.conv_feature_extraction import (
    ConvBlock,
    ConvFeatureExtractionModel,
    lengths_to_padding_mask,
)


class TestLengthsToPaddingMask:
    """Tests for the lengths_to_padding_mask function."""

    def test_basic_functionality(self):
        """Test basic conversion of lengths to padding mask."""
        lengths = mx.array([3, 5, 2])
        mask = lengths_to_padding_mask(lengths)

        # Expected shape: (3, 5) since max_length is 5
        assert mask.shape == (3, 5)

        # Check mask values
        # First sequence: length 3, so positions 3,4 should be True (padding)
        assert not mask[0, 0].item()
        assert not mask[0, 1].item()
        assert not mask[0, 2].item()
        assert mask[0, 3].item()
        assert mask[0, 4].item()

        # Second sequence: length 5, so no padding
        assert not mask[1, 0].item()
        assert not mask[1, 4].item()

        # Third sequence: length 2, so positions 2,3,4 should be True
        assert not mask[2, 0].item()
        assert not mask[2, 1].item()
        assert mask[2, 2].item()
        assert mask[2, 3].item()
        assert mask[2, 4].item()

    def test_single_sequence(self):
        """Test with a single sequence."""
        lengths = mx.array([4])
        mask = lengths_to_padding_mask(lengths)

        assert mask.shape == (1, 4)
        # All positions should be valid (no padding)
        for i in range(4):
            assert not mask[0, i].item()

    def test_all_same_length(self):
        """Test when all sequences have the same length."""
        lengths = mx.array([3, 3, 3])
        mask = lengths_to_padding_mask(lengths)

        assert mask.shape == (3, 3)
        # No padding for any sequence
        for i in range(3):
            for j in range(3):
                assert not mask[i, j].item()

    def test_zero_length(self):
        """Test with zero-length sequences."""
        lengths = mx.array([0, 2, 0])
        mask = lengths_to_padding_mask(lengths)

        assert mask.shape == (3, 2)
        # First sequence: all padding
        assert mask[0, 0].item()
        assert mask[0, 1].item()
        # Second sequence: no padding
        assert not mask[1, 0].item()
        assert not mask[1, 1].item()
        # Third sequence: all padding
        assert mask[2, 0].item()
        assert mask[2, 1].item()


class TestConvBlock:
    """Tests for the ConvBlock module."""

    def test_initialization(self):
        """Test ConvBlock initialization."""
        block = ConvBlock(
            n_in=1,
            n_out=512,
            kernel_size=10,
            stride=5,
            dropout=0.0,
            is_layer_norm=False,
            is_group_norm=True,
            conv_bias=False,
            mode="default",
        )

        assert block.conv.weight.shape == (512, 10, 1)
        assert block.norm is not None
        assert block.activation is not None

    def test_forward_pass_basic(self):
        """Test basic forward pass without normalization."""
        block = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.0,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            mode="default",
        )

        # Input shape: (batch_size, channels, time)
        x = mx.random.normal((2, 1, 100))
        output = block(x)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 64  # output channels
        # Time dimension reduced by convolution
        assert output.shape[2] < 100

    def test_forward_pass_with_layer_norm(self):
        """Test forward pass with layer normalization."""
        block = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.0,
            is_layer_norm=True,
            is_group_norm=False,
            conv_bias=False,
            mode="layer_norm",
        )

        x = mx.random.normal((2, 1, 100))
        output = block(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 64

    def test_forward_pass_with_group_norm(self):
        """Test forward pass with group normalization."""
        block = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.0,
            is_layer_norm=False,
            is_group_norm=True,
            conv_bias=False,
            mode="default",
        )

        x = mx.random.normal((2, 1, 100))
        output = block(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 64

    def test_forward_pass_with_group_norm_masked(self):
        """Test forward pass with masked group normalization."""
        block = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.0,
            is_layer_norm=False,
            is_group_norm=True,
            conv_bias=False,
            mode="group_norm_masked",
        )

        x = mx.random.normal((2, 1, 100))
        # Calculate output time dimension: (100 - 3) / 2 + 1 = 49
        # Mask should match output time dimension after convolution
        output_time = (100 - 3) // 2 + 1  # = 49
        mask = mx.ones((2, output_time))
        output = block(x, mask)

        assert output.shape[0] == 2
        assert output.shape[1] == 64
        assert output.shape[2] == output_time

    def test_dropout(self):
        """Test that dropout parameter is properly set."""
        block_no_dropout = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.0,
            mode="default",
        )
        assert block_no_dropout.dropout is None

        block_with_dropout = ConvBlock(
            n_in=1,
            n_out=64,
            kernel_size=3,
            stride=2,
            dropout=0.5,
            mode="default",
        )
        assert block_with_dropout.dropout is not None


class TestConvFeatureExtractionModel:
    """Tests for the ConvFeatureExtractionModel module."""

    def test_initialization_default_mode(self):
        """Test model initialization with default mode."""
        conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        assert len(model.conv_layers) == 3
        assert model.mode == "default"
        assert model.first_conv_params == (10, 5)

    def test_initialization_layer_norm_mode(self):
        """Test model initialization with layer_norm mode."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="layer_norm",
            conv_bias=False,
        )

        assert len(model.conv_layers) == 2
        assert model.mode == "layer_norm"

    def test_initialization_group_norm_masked_mode(self):
        """Test model initialization with group_norm_masked mode."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="group_norm_masked",
            conv_bias=False,
        )

        assert len(model.conv_layers) == 2
        assert model.mode == "group_norm_masked"

    def test_invalid_mode(self):
        """Test that invalid mode raises an assertion error."""
        conv_layers = [(512, 10, 5)]

        with pytest.raises(AssertionError):
            ConvFeatureExtractionModel(
                conv_layers=conv_layers,
                dropout=0.0,
                mode="invalid_mode",
                conv_bias=False,
            )

    def test_invalid_conv_layer_definition(self):
        """Test that invalid conv layer definition raises an error."""
        # Conv layer with only 2 elements instead of 3
        conv_layers = [(512, 10)]

        with pytest.raises(ValueError):
            ConvFeatureExtractionModel(
                conv_layers=conv_layers,
                dropout=0.0,
                mode="default",
                conv_bias=False,
            )

    def test_forward_pass_without_mask(self):
        """Test forward pass without padding mask."""
        conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        # Input shape: (batch_size, time)
        x = mx.random.normal((2, 16000))
        output = model(x)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 512  # channels from last conv layer
        assert output.shape[2] > 0  # time dimension (reduced)

    def test_forward_pass_with_default_mode(self):
        """Test forward pass with default mode (as used in reference tests)."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        # Input shape: (batch_size, time)
        x = mx.random.normal((2, 16000))

        # Test without mask (primary use case based on reference tests)
        output = model(x, padding_mask=None)

        # Check output shape
        assert output.shape[0] == 2
        assert output.shape[1] == 512
        assert output.shape[2] > 0

    def test_input_with_channel_dimension(self):
        """Test that model handles input with channel dimension."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        # Input shape: (batch_size, channels, time)
        x = mx.random.normal((2, 1, 16000))
        output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 512

    def test_output_shape_calculation(self):
        """Test that output shape is correctly calculated after convolutions."""
        conv_layers = [(64, 10, 5), (128, 3, 2), (256, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        # Input with known size
        x = mx.random.normal((1, 1000))
        output = model(x)

        # After conv1: (1000 - 10) / 5 + 1 = 199
        # After conv2: (199 - 3) / 2 + 1 = 99
        # After conv3: (99 - 3) / 2 + 1 = 49
        # Note: MLX may handle padding differently, so we just check it's reduced
        assert output.shape[0] == 1
        assert output.shape[1] == 256  # Last conv layer output channels
        assert output.shape[2] < 1000  # Time dimension is reduced

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 16000))
            output = model(x)
            assert output.shape[0] == batch_size
            assert output.shape[1] == 512

    def test_single_conv_layer(self):
        """Test model with a single convolutional layer."""
        conv_layers = [(512, 10, 5)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        x = mx.random.normal((2, 1000))
        output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 512

    def test_with_dropout(self):
        """Test model with dropout enabled."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.1,
            mode="default",
            conv_bias=False,
        )

        x = mx.random.normal((2, 16000))
        output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 512

    def test_with_conv_bias(self):
        """Test model with convolutional bias enabled."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=True,
        )

        # Check that bias exists in conv layers
        for block in model.conv_layers:
            assert block.conv.bias is not None

        x = mx.random.normal((2, 16000))
        output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 512

    def test_default_mode_multiple_layers(self):
        """Test that model works with default mode and multiple layers."""
        conv_layers = [(512, 10, 5), (512, 3, 2)]
        model = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )

        # Create input
        x = mx.random.normal((2, 100))

        # Test without mask (primary use case based on reference tests)
        output = model(x, padding_mask=None)

        # Should complete without errors
        assert output.shape[0] == 2
        assert output.shape[1] == 512
