---
license: mit
library_name: mlx
tags:
  - mlx
  - audio
  - speech
  - feature-extraction
  - contentvec
  - hubert
  - voice-conversion
  - rvc
datasets:
  - librispeech_asr
language:
  - en
pipeline_tag: feature-extraction
---

# MLX ContentVec / HuBERT Base

MLX-converted weights for ContentVec/HuBERT base model, optimized for Apple Silicon.

This model extracts speaker-agnostic semantic features from audio, primarily used as the feature extraction backbone for [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

## Model Details

- **Architecture**: HuBERT Base (12 transformer layers)
- **Parameters**: ~90M
- **Input**: 16kHz mono audio
- **Output**: 768-dimensional features (~50 frames/second)
- **Framework**: [MLX](https://github.com/ml-explore/mlx)
- **Format**: SafeTensors (float32)

## Usage

```python
import mlx.core as mx
import librosa
from mlx_contentvec import ContentvecModel

# Load model
model = ContentvecModel(encoder_layers_1=0)
model.load_weights("contentvec_base.safetensors")
model.eval()

# Load audio at 16kHz
audio, sr = librosa.load("input.wav", sr=16000, mono=True)
source = mx.array(audio).reshape(1, -1)

# Extract features
result = model(source)
features = result["x"]  # Shape: (1, num_frames, 768)
```

## Installation

```bash
pip install git+https://github.com/example/mlx-contentvec.git
```

## Download Weights

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="lexandstuff/mlx-contentvec",
    filename="contentvec_base.safetensors"
)
```

## Validation

These weights produce **numerically identical** outputs to the original PyTorch implementation:

| Metric | Value |
|--------|-------|
| Max absolute difference | 7.3e-6 |
| Cosine similarity | 1.000000 |

## Source Weights

Converted from [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) (MD5: `b76f784c1958d4e535cd0f6151ca35e4`).

## Use Cases

- **Voice Conversion**: Feature extraction for RVC pipeline
- **Speaker Verification**: Content-based audio embeddings
- **Speech Analysis**: Semantic feature extraction

## Citation

```bibtex
@inproceedings{qian2022contentvec,
  title={ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers},
  author={Qian, Kaizhi and Zhang, Yang and Gao, Heting and Ni, Junrui and Lai, Cheng-I and Cox, David and Hasegawa-Johnson, Mark and Chang, Shiyu},
  booktitle={International Conference on Machine Learning},
  year={2022}
}

@article{hsu2021hubert,
  title={HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units},
  author={Hsu, Wei-Ning and others},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021}
}
```

## License

MIT
