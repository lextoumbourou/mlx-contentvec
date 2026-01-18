# MLX ContentVec

MLX implementation of [ContentVec](https://arxiv.org/abs/2204.09224) / HuBERT for Apple Silicon.

This is the **feature extraction backbone** for [RVC-MLX](https://github.com/example/rvc-mlx), a native Apple Silicon implementation of [Retrieval-based Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

## What is ContentVec?

ContentVec extracts **speaker-agnostic semantic features** from audio. In the RVC pipeline, it captures the phonetic content of speech while discarding speaker identity, enabling voice conversion:

```
Input Audio (16kHz) → ContentVec → Semantic Features (768-dim) → RVC Decoder → Converted Voice
```

## Installation

```bash
# With uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### 1. Download and Convert Weights

Download the HuBERT base weights used by RVC:

```bash
mkdir -p weights
wget -O weights/hubert_base.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

# Verify checksum
echo "b76f784c1958d4e535cd0f6151ca35e4  weights/hubert_base.pt" | md5sum -c
```

Convert to MLX SafeTensors format. This requires Python 3.9 and fairseq (due to compatibility issues with newer Python):

```bash
# Option A: Use the conversion script with vendored fairseq
# First, set up the vendor directory (see Development section)
uv run --python 3.9 python scripts/convert_weights.py \
  --pytorch_ckpt weights/hubert_base.pt \
  --mlx_ckpt weights/contentvec_base.safetensors

# Option B: Use a pre-converted weights file (if available)
# Check releases for pre-converted .safetensors files
```

### 2. Extract Features

```python
import mlx.core as mx
import librosa
from mlx_contentvec import ContentvecModel

# Load model (12 transformer layers, no speaker conditioning)
model = ContentvecModel(encoder_layers_1=0)
model.load_weights("weights/contentvec_base.safetensors")
model.eval()

# Load audio at 16kHz
audio, sr = librosa.load("input.wav", sr=16000, mono=True)
source = mx.array(audio).reshape(1, -1)

# Extract features
result = model(source)
features = result["x"]  # Shape: (1, num_frames, 768)

print(f"Audio: {len(audio)/16000:.2f}s -> Features: {features.shape}")
# Example: Audio: 3.00s -> Features: (1, 93, 768)
```

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
│   └── ...
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
pytest
```

### Weight Conversion Details

The conversion from PyTorch to MLX requires:

1. **Tensor transposition**: Conv1d weights change from `(out, in, kernel)` to `(out, kernel, in)`
2. **Weight normalization**: The positional conv uses weight norm with `g` and `v` parameters
3. **Float32 precision**: Weights must be saved as float32 (not float16) for numerical accuracy

See `scripts/convert_weights.py` and `IMPLEMENTATION_NOTES.md` for details.

## License

MIT

## Acknowledgments

- [ContentVec](https://github.com/auspicious3000/contentvec) - Original implementation by Kaizhi Qian
- [fairseq](https://github.com/facebookresearch/fairseq) - HuBERT/wav2vec2 implementation
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Voice conversion pipeline
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
