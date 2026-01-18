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

1. **End-to-End Validation**
   - [ ] Run full model with `assets/testing.mp3`
   - [ ] Compare MLX output vs PyTorch output
   - [ ] Verify numerical tolerance (target: atol=1e-4, rtol=1e-3)

2. **Weight Conversion Verification**
   - [ ] Run `scripts/convert_weights.py` on `vendor/ref_weights/hubert_base.pt`
   - [ ] Verify all weights are converted correctly
   - [ ] Check for any missing/extra keys

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
