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
