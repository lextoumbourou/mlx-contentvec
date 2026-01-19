# MLX ContentVec

MLX implementation of [ContentVec](https://arxiv.org/abs/2204.09224) / HuBERT for Apple Silicon.

This is the **feature extraction backbone** for RVC-MLX (coming soon), a native Apple Silicon implementation of [Retrieval-based Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

## What is ContentVec?

ContentVec extracts **speaker-agnostic semantic features** from audio. In the RVC pipeline, it captures the phonetic content of speech while discarding speaker identity, enabling voice conversion:

```
Input Audio (16kHz) → ContentVec → Semantic Features (768-dim) → RVC Decoder → Converted Voice
```

## Installation

```bash
pip install mlx-contentvec
```

For development:

```bash
git clone https://github.com/lexandstuff/mlx-contentvec.git
cd mlx-contentvec
pip install -e .
```

## Quick Start

```python
import mlx.core as mx
import librosa
from mlx_contentvec import ContentvecModel

# Load model (auto-downloads weights from HuggingFace)
model = ContentvecModel.from_pretrained()

# Load audio at 16kHz
audio, sr = librosa.load("input.wav", sr=16000, mono=True)
source = mx.array(audio).reshape(1, -1)

# Extract features
result = model(source)
features = result["x"]  # Shape: (1, num_frames, 768)

print(f"Audio: {len(audio)/16000:.2f}s -> Features: {features.shape}")
# Example: Audio: 3.00s -> Features: (1, 93, 768)
```

### Manual Weight Loading

If you prefer to manage weights yourself:

```python
from huggingface_hub import hf_hub_download
from mlx_contentvec import ContentvecModel

# Download weights
weights_path = hf_hub_download(
    repo_id="lexandstuff/mlx-contentvec",
    filename="contentvec_base.safetensors"
)

# Load model manually
model = ContentvecModel(encoder_layers_1=0)
model.load_weights(weights_path)
model.eval()
```

<details>
<summary>Converting weights from PyTorch (advanced)</summary>

If you need to convert from PyTorch yourself:

```bash
# Download original PyTorch weights
wget -O weights/hubert_base.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

# Convert (requires Python 3.9 + fairseq)
uv run --python 3.9 python scripts/convert_weights.py \
  --pytorch_ckpt weights/hubert_base.pt \
  --mlx_ckpt weights/contentvec_base.safetensors
```

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for details.
</details>

## API Reference

### ContentvecModel

```python
ContentvecModel(
    encoder_layers: int = 12,      # Number of transformer layers
    encoder_layers_1: int = 0,     # Speaker-conditioned layers (set to 0 for RVC)
    encoder_embed_dim: int = 768,  # Feature dimension
    ...
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `load_weights(path)` | Load weights from SafeTensors file |
| `eval()` | Set to inference mode (disables dropout) |
| `__call__(source, spk_emb=None)` | Extract features from audio |

**Input:**
- `source`: Audio waveform tensor, shape `(batch, samples)`, 16kHz sample rate

**Output:**
- Returns `{"x": features, "padding_mask": None}`
- `features` shape: `(batch, num_frames, 768)`
- Frame rate: ~50 frames/second (hop size = 320 samples at 16kHz)

## RVC Integration

In the RVC voice conversion pipeline, ContentVec provides semantic features that preserve speech content while enabling voice transformation:

```python
# 1. Extract content features with ContentVec
features = contentvec_model(audio)["x"]  # (1, T, 768)

# 2. Optional: Blend with voice index for timbre transfer
# features = faiss_index.search(features) * index_rate + features * (1 - index_rate)

# 3. Extract pitch (F0) with separate model (RMVPE, etc.)
f0 = pitch_extractor(audio)  # (1, T)

# 4. Generate converted audio with RVC synthesizer
output = rvc_synthesizer(features, f0, speaker_id)
```

The key insight is that ContentVec captures **what is being said** (phonetic content) while the RVC decoder adds **who is saying it** (speaker identity via F0 and speaker embedding).

## Validation

This implementation produces **numerically identical** outputs to the PyTorch reference:

| Metric | Value |
|--------|-------|
| Max absolute difference | 8e-6 |
| Cosine similarity | 1.000000 |

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for detailed validation methodology.

## Development

### Project Structure

```
mlx-contentvec/
├── mlx_contentvec/
│   ├── __init__.py
│   ├── contentvec.py              # Main model class
│   ├── conv_feature_extraction.py # 7-layer CNN feature extractor
│   ├── transformer_encoder.py     # 12-layer transformer with pos conv
│   └── modules/
│       ├── multihead_attention.py # Multi-head self-attention
│       ├── weight_norm.py         # Weight normalization for pos conv
│       ├── group_norm.py          # Group norm (incl. masked variant)
│       └── cond_layer_norm.py     # Conditional layer norm (speaker)
├── scripts/
│   └── convert_weights.py         # PyTorch → SafeTensors conversion
├── tests/
│   ├── test_conv_feature_extraction.py
│   ├── test_end_to_end.py
│   └── test_weight_norm.py
├── IMPLEMENTATION_NOTES.md        # Technical details & validation
└── README.md
```

### Setting Up for Development

Clone reference implementations for comparison:

```bash
mkdir -p vendor && cd vendor

# ContentVec reference
git clone https://github.com/auspicious3000/contentvec.git

# fairseq (required for loading PyTorch checkpoint)
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq && git checkout 0b21875
```

### Running Tests

```bash
uv run pytest
```

Test suite (48 tests):

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_conv_feature_extraction.py` | 24 | CNN feature extractor unit tests |
| `test_end_to_end.py` | 10 | Integration tests (HuggingFace → inference) |
| `test_weight_norm.py` | 16 | Weight normalization unit tests |

The end-to-end tests download weights from HuggingFace and verify:
- Model loading and initialization
- Inference with various input shapes
- Deterministic output in eval mode
- Feature statistics (no NaN/Inf)
- Real audio file processing

### Weight Conversion Details

The conversion from PyTorch to MLX requires:

1. **Tensor transposition**: Conv1d weights change from `(out, in, kernel)` to `(out, kernel, in)`
2. **Weight normalization**: The positional conv uses weight norm with `g` and `v` parameters
3. **Float32 precision**: Weights must be saved as float32 (not float16) for numerical accuracy

See `scripts/convert_weights.py` and `IMPLEMENTATION_NOTES.md` for details.

### Publishing to PyPI

1. Update the version in `pyproject.toml`
2. Update `CHANGELOG.md` with the new version
3. Build and upload:

```bash
# Build distribution packages
uv run python -m build

# Upload to PyPI
uv run twine upload dist/*
```

## License

MIT

## Acknowledgments

- [ContentVec](https://github.com/auspicious3000/contentvec) - Original implementation by Kaizhi Qian
- [fairseq](https://github.com/facebookresearch/fairseq) - HuBERT/wav2vec2 implementation
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Voice conversion pipeline
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
