# Development Guide for Agents

## Using uv

This project uses **uv** for package management.

### Install dependencies

```bash
# Install with dev dependencies (includes PyTorch for reference tests)
uv pip install -e ".[dev]"

# Install without dev dependencies
uv pip install -e .
```

### Important notes
- uv-managed environments don't have a `pip` binary - always use `uv pip`
- uv is much faster than pip and handles caching automatically

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run unit tests only
```bash
pytest tests/test_conv_feature_extraction.py -v
```

### Run reference tests (PyTorch comparison)
```bash
pytest tests/test_reference_pytorch.py -v
```

Reference tests require dev dependencies and will skip if PyTorch is not installed.

## Reference implementation

The reference implementation for ContentVec can be found in `vendor/contentvec`. Fairseq can be found in `vendor/fairseq`.

### Important: Standalone vs Reference Code

- **The core module (`mlx_contentvec`)** should NOT depend on fairseq or contentvec. Reimplement any code needed to keep this a standalone module.
- **Reference tests** CAN use fairseq/contentvec directly. No need to create standalone PyTorch versions for testing - just import from vendor directories.

## Coding Standards

- **Always use `pathlib`** for file path operations instead of string concatenation or `os.path`
