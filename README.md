# ContentVec MLX

This repository is a port of the [official PyTorch implementation](https://github.com/auspicious3000/contentvec) of [ContentVec](https://arxiv.org/abs/2204.09224) to the [MLX framework](https://github.com/ml-explore/mlx).

## Installation

Install the package with development dependencies:

```bash
uv pip install -e ".[dev]"
```

Or for basic usage without tests:

```bash
uv pip install -e .
```

## Testing

The project uses pytest for testing. To run all tests:

```bash
pytest
```

## Usage


### Extract Features

```python
import mlx.core as mx
from mlx_contentvec import ContentvecModel
import soundfile as sf

# Load model with safetensors
model = ContentvecModel()
model.load_weights("contentvec_base.safetensors")

# Load audio (16kHz mono)
audio, sr = sf.read("audio.wav")
audio = mx.array(audio).reshape(1, -1)

# Zero speaker embedding (or use your own)
spk_emb = mx.zeros((1, 256))

# Extract features from layer 12
features, _ = model.extract_features(
    source=audio,
    spk_emb=spk_emb,
    output_layer=12
)

print(f"Feature shape: {features.shape}")  # (1, time, 768)
```

## Development

For simplicity of reference comparison, I like to clone the reference repos to the vendor directory:

```bash
cd vendor
git clone git@github.com:auspicious3000/contentvec.git
git clone git@github.com:facebookresearch/fairseq.git --branch main --single-branch
cd fairseq && git reset --hard 0b21875e45f332bedbcc0617dcf9379d3c03855f
```

Also, download the reference weights that we're using to compare:

```bash
wget -O vendor/ref_weights/hubert_base.pt "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true"
# Check md5sum
md5sum vendor/ref_weights/hubert_base.pt
b76f784c1958d4e535cd0f6151ca35e4  vendor/ref_weights/hubert_base.pt
```

### Convert PyTorch Weights to Safetensors

The model weights need to be converted from PyTorch to safetensors format. Due to fairseq compatibility issues with Python 3.11+, use Python 3.9 for conversion:

```bash
# In a Python 3.9 environment
conda create -n contentvec-convert python=3.9
conda activate contentvec-convert
pip install torch fairseq safetensors numpy

# Convert weights
python scripts/convert_weights.py \
    --pytorch_ckpt vendor/ref_weights/hubert_base.pt \
    --mlx_ckpt output/contentvec_base.safetensors
```

### Project Structure

```
mlx_contentvec/
├── mlx_contentvec/           # Core implementation
│   ├── contentvec.py         # Main ContentVec model
│   ├── conv_feature_extraction.py
│   ├── transformer_encoder.py
│   └── modules/              # Custom layers
├── convert_weights.py        # PyTorch → safetensors
├── dump_contentvec_feature.py  # Feature extraction
├── test_feature_comparison.py  # Validation
└── vendor/                   # Reference implementations
```
