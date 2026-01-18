"""
End-to-end integration tests for MLX ContentVec.

These tests verify the complete pipeline works correctly, including:
- Downloading weights from HuggingFace
- Loading the model
- Running inference
- Validating output shapes and values

Run with: pytest tests/test_end_to_end.py -v
"""

import numpy as np
import pytest

import mlx.core as mx

from mlx_contentvec import ContentvecModel


# Test constants
HUGGINGFACE_REPO = "lexandstuff/mlx-contentvec"
WEIGHTS_FILE = "contentvec_base.safetensors"
SAMPLE_RATE = 16000
EMBED_DIM = 768


@pytest.fixture(scope="module")
def weights_path():
    """Download and cache weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=WEIGHTS_FILE)


@pytest.fixture(scope="module")
def model(weights_path):
    """Load model with weights."""
    model = ContentvecModel(encoder_layers_1=0)
    model.load_weights(weights_path)
    model.eval()
    return model


class TestModelLoading:
    """Tests for model initialization and weight loading."""

    def test_model_initialization(self):
        """Test that model initializes with correct architecture."""
        model = ContentvecModel(encoder_layers_1=0)

        # Check key attributes exist
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "layer_norm")
        assert hasattr(model, "post_extract_proj")
        assert hasattr(model, "encoder")

    def test_weight_loading(self, weights_path):
        """Test that weights load without errors."""
        model = ContentvecModel(encoder_layers_1=0)
        model.load_weights(weights_path)

        # Verify some weights are non-zero
        params = model.parameters()
        assert "encoder" in params
        assert "layer_norm" in params

    def test_eval_mode(self, model):
        """Test that eval mode is properly set."""
        # Model should be in eval mode (training=False)
        # This affects dropout behavior
        assert not model.training


class TestInference:
    """Tests for model inference."""

    def test_basic_inference(self, model):
        """Test basic inference with synthetic input."""
        # Create 1 second of random audio
        audio = mx.random.normal((1, SAMPLE_RATE))

        result = model(audio)
        features = result["x"]
        mx.eval(features)

        # Check output exists and has correct embedding dimension
        assert features.shape[0] == 1  # batch size
        assert features.shape[2] == EMBED_DIM  # embedding dim

    def test_output_shape(self, model):
        """Test that output shape is correct for various input lengths."""
        test_cases = [
            (SAMPLE_RATE // 2, "0.5s"),  # 0.5 seconds
            (SAMPLE_RATE, "1s"),  # 1 second
            (SAMPLE_RATE * 2, "2s"),  # 2 seconds
            (SAMPLE_RATE * 5, "5s"),  # 5 seconds
        ]

        for num_samples, label in test_cases:
            audio = mx.random.normal((1, num_samples))
            result = model(audio)
            features = result["x"]
            mx.eval(features)

            assert features.shape[0] == 1, f"Batch size wrong for {label}"
            assert features.shape[2] == EMBED_DIM, f"Embed dim wrong for {label}"
            assert features.shape[1] > 0, f"No frames output for {label}"

    def test_batch_inference(self, model):
        """Test inference with batch of inputs."""
        batch_size = 4
        audio = mx.random.normal((batch_size, SAMPLE_RATE))

        result = model(audio)
        features = result["x"]
        mx.eval(features)

        assert features.shape[0] == batch_size
        assert features.shape[2] == EMBED_DIM

    def test_deterministic_output(self, model):
        """Test that same input produces same output (no dropout in eval)."""
        audio = mx.random.normal((1, SAMPLE_RATE))

        result1 = model(audio)
        mx.eval(result1["x"])

        result2 = model(audio)
        mx.eval(result2["x"])

        # Outputs should be identical in eval mode
        diff = mx.max(mx.abs(result1["x"] - result2["x"]))
        assert float(diff) < 1e-6, "Outputs differ between runs"


class TestFeatureExtraction:
    """Tests for feature extraction specifics."""

    def test_frame_rate(self, model):
        """Test that frame rate is approximately 50 Hz."""
        # 1 second of audio should produce ~50 frames
        # (hop size is 320 samples at 16kHz = 20ms per frame)
        audio = mx.random.normal((1, SAMPLE_RATE))

        result = model(audio)
        features = result["x"]
        mx.eval(features)

        num_frames = features.shape[1]
        # Allow some tolerance for edge effects
        assert 40 < num_frames < 60, f"Expected ~50 frames, got {num_frames}"

    def test_feature_statistics(self, model):
        """Test that output features have reasonable statistics."""
        audio = mx.random.normal((1, SAMPLE_RATE * 2))

        result = model(audio)
        features = result["x"]
        mx.eval(features)

        features_np = np.array(features)

        # Features should not be all zeros
        assert np.abs(features_np).max() > 0.1

        # Features should not contain NaN or Inf
        assert not np.any(np.isnan(features_np))
        assert not np.any(np.isinf(features_np))

        # Features should have reasonable range (layer norm output)
        assert np.abs(features_np).max() < 100


class TestWithRealAudio:
    """Tests with real audio file (if available)."""

    @pytest.fixture
    def test_audio_path(self):
        """Path to test audio file."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "assets" / "testing.mp3"
        if not path.exists():
            pytest.skip("Test audio file not found")
        return str(path)

    def test_real_audio_inference(self, model, test_audio_path):
        """Test inference with real audio file."""
        import librosa

        audio, sr = librosa.load(test_audio_path, sr=SAMPLE_RATE, mono=True)
        source = mx.array(audio).reshape(1, -1)

        result = model(source)
        features = result["x"]
        mx.eval(features)

        # Basic sanity checks
        assert features.shape[0] == 1
        assert features.shape[2] == EMBED_DIM
        assert features.shape[1] > 0

        # Check statistics
        features_np = np.array(features)
        assert not np.any(np.isnan(features_np))
        assert not np.any(np.isinf(features_np))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
