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

## Development

For simplicity of reference comparison, I'm clone the reference repos to the vendor directory:

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
