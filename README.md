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
