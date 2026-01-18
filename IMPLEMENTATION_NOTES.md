# MLX ContentVec Specification

This document tracks the implementation status of the MLX port of ContentVec/HuBERT.

## Goal

Convert the PyTorch weights `vendor/ref_weights/hubert_base.pt` to MLX format and produce **identical outputs** for a given input audio file.

## Architecture Overview

```
Raw Audio (16kHz)
    │
    ▼
┌─────────────────────────────────────┐
│  ConvFeatureExtractionModel         │
│  (7 conv layers, downsampling)      │
│  Input:  (B, T)                     │
│  Output: (B, T', 512)               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Post-Extract Projection            │
│  Linear(512 → 768) + LayerNorm      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  TransformerEncoder                 │
│  - Positional Conv (weight norm)    │
│  - 12 Standard Transformer Layers   │
│  - 3 Speaker-Conditioned Layers     │
│  Output: (B, T', 768)               │
└─────────────────────────────────────┘
```

## Implementation Status

### Core Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| ConvFeatureExtractionModel | `conv_feature_extraction.py` | ✅ Complete | 7 conv layers with group norm |
| ContentvecModel | `contentvec.py` | ✅ Complete | Main model wrapper |
| TransformerEncoder_1 | `transformer_encoder.py` | ✅ Complete | 12 + 3 speaker-conditioned layers |
| MultiheadAttention | `modules/multihead_attention.py` | ✅ Complete | 12 heads, 768 dims |
| CondLayerNorm | `modules/cond_layer_norm.py` | ✅ Complete | Speaker embedding conditioning |
| WeightNorm | `modules/weight_norm.py` | ✅ Complete | For positional conv |
| GroupNorm | `modules/group_norm.py` | ✅ Complete | Includes masked variant |

### Scripts & Utilities

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/convert_weights.py` | ✅ Complete | PyTorch → SafeTensors conversion |
| `scripts/test_feature_comparison.py` | ⚠️ Unknown | Needs verification |
| `scripts/dump_contentvec_feature.py` | ⚠️ Unknown | Needs verification |
| `test_pytorch_features.py` | ✅ Ready | Reference feature extraction |

### Tests

| Test | Status | Notes |
|------|--------|-------|
| `test_conv_feature_extraction.py` | ✅ Complete | Comprehensive coverage |
| `test_reference_pytorch.py` | ⚠️ Partial | PyTorch vs MLX comparison |
| `test_transformer_encoder_pytorch.py` | ⚠️ Partial | Needs completion |
| `test_weight_norm.py` | ⚠️ Partial | Needs completion |

## Remaining Work

### High Priority

1. **End-to-End Validation** ✅ COMPLETE
   - [x] Run full model with `assets/testing.mp3`
   - [x] Compare MLX output vs PyTorch output
   - [x] Verify numerical tolerance (target: atol=1e-4, rtol=1e-3)

2. **Weight Conversion Verification** ✅ COMPLETE
   - [x] Run `scripts/convert_weights.py` on `vendor/ref_weights/hubert_base.pt`
   - [x] Verify all weights are converted correctly
   - [x] Check for any missing/extra keys

3. **Test Suite Completion**
   - [ ] Complete `test_transformer_encoder_pytorch.py`
   - [ ] Complete `test_weight_norm.py`
   - [ ] Add end-to-end integration test

### Medium Priority

4. **Documentation**
   - [ ] Usage examples in README
   - [ ] API documentation
   - [ ] Inference benchmarks (MLX vs PyTorch)

5. **Edge Cases**
   - [ ] Variable length audio handling
   - [ ] Padding mask correctness
   - [ ] Speaker embedding variations

### Low Priority

6. **Optional Features**
   - [ ] WAV2VEC alternative implementation
   - [ ] Training support (if needed)
   - [ ] Quantization/optimization

## Key Configuration

```python
# Default HuBERT Base / ContentVec configuration
encoder_embed_dim = 768
encoder_ffn_embed_dim = 3072
encoder_attention_heads = 12
encoder_layers = 12  # Regular transformer layers
speaker_conditioned_layers = 3  # Additional speaker-conditioned layers
conv_pos = 128  # Positional conv filters
conv_pos_groups = 16  # Positional conv groups
speaker_embed_dim = 256  # Speaker embedding dimension
```

## Weight Conversion Notes

### Tensor Format Differences

| Layer Type | PyTorch Shape | MLX Shape | Transform |
|------------|--------------|-----------|-----------|
| Conv1d weight | (out, in, kernel) | (out, kernel, in) | transpose(0, 2, 1) |
| Linear weight | (out, in) | (out, in) | none |
| LayerNorm | (dim,) | (dim,) | none |

### Key Mappings

```
PyTorch                          MLX
-------                          ---
feature_extractor.conv_layers    conv_feature_extractor.conv_layers
encoder.layers                   transformer_encoder.layers
encoder.layer_norm               transformer_encoder.layer_norm
encoder.pos_conv                 transformer_encoder.pos_conv
layer_norm_first                 (same, affects residual order)
```

## Test Assets

- `assets/testing.mp3` - Test audio file for validation

## Reference Implementation

- `vendor/fairseq.bak/` - Original fairseq implementation
- `vendor/ref_weights/hubert_base.pt` - Reference weights (MD5: b76f784c1958d4e535cd0f6151ca35e4)

## Validation Criteria

The MLX implementation is considered complete when:

1. **Numerical Match**: Output features match PyTorch within tolerance
   - Absolute tolerance: 1e-4
   - Relative tolerance: 1e-3

2. **Shape Match**: All intermediate and final tensor shapes match

3. **All Tests Pass**: Complete test suite executes without failures

4. **End-to-End**: Can process `assets/testing.mp3` and produce identical features

## Extracting PyTorch Reference Features

The reference checkpoint requires fairseq with specific patches for numpy compatibility. Use Python 3.9 with the vendored fairseq:

```bash
cd /tmp && uv run --python 3.9 --isolated \
  --with torch \
  --with 'numpy>=1.20,<2.0' \
  --with omegaconf \
  --with hydra-core \
  --with bitarray \
  --with regex \
  --with sacrebleu \
  --with cython \
  --with tqdm \
  --with librosa \
  --with soundfile \
  python << 'SCRIPT'
import sys
import numpy as np

# Patch numpy deprecated aliases (required for fairseq compatibility)
np.float = np.float64
np.int = np.int64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_

sys.path.insert(0, '/path/to/mlx-contentvec/vendor/fairseq.bak')

import torch
import librosa
from argparse import Namespace

# Load audio at 16kHz
audio, sr = librosa.load('assets/testing.mp3', sr=16000, mono=True)

# Load checkpoint with weights_only=False (required for fairseq)
model_path = 'vendor/ref_weights/hubert_base.pt'
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
model_cfg = ckpt['cfg']['model']
state_dict = ckpt['model']

# Create args namespace from config
args = Namespace(**model_cfg)
args.encoder_layers = 12  # Use standard 12 layers

# Build components
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel, TransformerEncoder
from fairseq.modules import LayerNorm

feature_extractor = ConvFeatureExtractionModel(
    conv_layers=eval(model_cfg['conv_feature_layers']),
    dropout=0.0,
    mode=model_cfg['extractor_mode'],
    conv_bias=model_cfg['conv_bias'],
)
encoder = TransformerEncoder(args)
layer_norm = LayerNorm(512)
post_extract_proj = torch.nn.Linear(512, model_cfg['encoder_embed_dim'])

# Load weights
fe_state = {k.replace('feature_extractor.', ''): v for k, v in state_dict.items() if k.startswith('feature_extractor.')}
feature_extractor.load_state_dict(fe_state)
enc_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
encoder.load_state_dict(enc_state)
layer_norm.load_state_dict({'weight': state_dict['layer_norm.weight'], 'bias': state_dict['layer_norm.bias']})
post_extract_proj.load_state_dict({'weight': state_dict['post_extract_proj.weight'], 'bias': state_dict['post_extract_proj.bias']})

feature_extractor.eval()
encoder.eval()
layer_norm.eval()
post_extract_proj.eval()

# Forward pass
source = torch.from_numpy(audio).float().unsqueeze(0)

with torch.no_grad():
    features = feature_extractor(source)     # (B, 512, T')
    features = features.transpose(1, 2)       # (B, T', 512)
    features = layer_norm(features)
    features = post_extract_proj(features)    # (B, T', 768)
    encoder_out = encoder(features, padding_mask=None)
    x = encoder_out[0]                        # (B, T', 768)

features_np = x.squeeze(0).cpu().numpy()
np.save('features_pytorch.npy', features_np)
SCRIPT
```

### Current Validation Status

**✅ VALIDATION PASSED (2025-01-19)**

End-to-end comparison with PyTorch reference (12 layers, no speaker conditioning):
- Max abs diff: 0.000008
- Mean abs diff: 0.000001
- Cosine similarity: 1.000000

**All Intermediate Outputs Match:**
- Conv output: ✅ Exact match
- LayerNorm output: ✅ Match (diff < 1e-5)
- Projection output: ✅ Match (diff < 1e-5)
- Pos Conv output: ✅ Match (diff < 1e-4)
- Encoder output: ✅ Match (diff < 1e-5)

**Bugs Fixed:**
1. **Float16 precision issue**: Weights were saved as float16 in safetensors, causing precision loss in WeightNorm computation. Fixed by explicitly casting to float32 during conversion.

2. **Dropout in eval mode**: `nn.Dropout(self.dropout)(x)` in `transformer_encoder.py` created a new Dropout instance each forward pass that didn't inherit the parent module's eval mode. Fixed by storing dropout as a class attribute (`self.dropout_layer`).

3. **Weight loading key format**: MLX's `update()` method expects nested dict structure, but safetensors uses flat keys like `encoder.layers.0.fc1.weight`. Fixed by implementing `_unflatten_weights()` to convert flat keys to nested structure.

**Usage Notes:**
- For inference without speaker conditioning, use `encoder_layers_1=0`:
  ```python
  model = ContentvecModel(encoder_layers_1=0)
  model.load_weights('contentvec_base.safetensors')
  model.eval()
  ```
- Speaker-conditioned layers (12-14) require a non-zero speaker embedding to produce non-zero output
