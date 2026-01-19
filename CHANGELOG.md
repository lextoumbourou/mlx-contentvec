# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-01-20

### Added

- `ContentvecModel.from_pretrained()` class method for simplified model loading
  - Auto-downloads weights from HuggingFace Hub
  - Configures model for RVC-compatible inference (`encoder_layers_1=0`)
  - Returns model in eval mode, ready for inference

## [0.1.0] - 2025-01-19

### Added

- Initial release of MLX ContentVec
- `ContentvecModel` class for feature extraction from 16kHz audio
- 7-layer CNN feature extractor (`ConvFeatureExtractionModel`)
- 12-layer transformer encoder with positional convolution
- Support for speaker-conditioned layers (optional, set `encoder_layers_1=0` for RVC)
- Weight normalization implementation for positional conv
- Group normalization (including masked variant)
- Conditional layer normalization for speaker embeddings
- Weight conversion script (`scripts/convert_weights.py`) for PyTorch to SafeTensors
- Pre-converted weights available on [HuggingFace](https://huggingface.co/lexandstuff/mlx-contentvec)
- Comprehensive test suite (48 tests)
- End-to-end integration tests with HuggingFace weight download

### Validated

- Numerical accuracy vs PyTorch reference:
  - Max absolute difference: 8e-6
  - Cosine similarity: 1.000000

[0.1.1]: https://github.com/lexandstuff/mlx-contentvec/releases/tag/v0.1.1
[0.1.0]: https://github.com/lexandstuff/mlx-contentvec/releases/tag/v0.1.0
