#!/usr/bin/env python3
"""
Test script to compare MLX ContentVec features with PyTorch implementation.
"""

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from mlx_contentvec import ContentvecModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("test_feature_comparison")


def layer_norm_mlx(x: mx.array) -> mx.array:
    """Apply layer normalization (MLX version)."""
    mean = mx.mean(x)
    std = mx.sqrt(mx.mean((x - mean) ** 2))
    return (x - mean) / (std + 1e-5)


def load_pytorch_model(ckpt_path: str):
    """Load PyTorch ContentVec model."""
    import sys
    sys.path.insert(0, "vendor/contentvec")

    import fairseq
    from fairseq import checkpoint_utils

    logger.info(f"Loading PyTorch model from {ckpt_path}")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval().cuda()

    logger.info(f"PyTorch model loaded")
    logger.info(f"Task config: {task.cfg}")

    return model, task


def load_mlx_model(ckpt_path: str, dim_spk: int = 256):
    """Load MLX ContentVec model from safetensors."""
    logger.info(f"Loading MLX model from {ckpt_path}")

    model = ContentvecModel(
        conv_feature_layers="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        conv_bias=False,
        extractor_mode="default",
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=12,
        encoder_layers=12,
        encoder_layers_1=3,
        conv_pos=128,
        conv_pos_groups=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        encoder_layerdrop=0.0,
        activation_fn="gelu",
        layer_norm_first=False,
        feature_grad_mult=1.0,
        dim_spk=dim_spk,
    )

    # Load weights from safetensors
    model.load_weights(ckpt_path)
    logger.info(f"MLX model loaded")

    return model


def extract_pytorch_features(model, task, audio_path: str, layer: int):
    """Extract features using PyTorch model."""
    # Read audio
    wav, sr = sf.read(audio_path)
    assert sr == task.cfg.sample_rate, f"Expected {task.cfg.sample_rate} Hz, got {sr} Hz"

    if wav.ndim == 2:
        wav = wav.mean(-1)

    # Convert to tensor
    x = torch.from_numpy(wav).float().cuda()

    # Normalize if needed
    if task.cfg.normalize:
        x = F.layer_norm(x, x.shape)

    x = x.view(1, -1)

    # Create zero speaker embedding
    spk_emb = torch.zeros(1, 256).cuda()

    # Extract features
    with torch.no_grad():
        feat, _ = model.extract_features(
            source=x,
            spk_emb=spk_emb,
            padding_mask=None,
            mask=False,
            output_layer=layer,
        )

    return feat.squeeze(0).cpu().numpy()


def extract_mlx_features(model, audio_path: str, layer: int, normalize: bool = False):
    """Extract features using MLX model."""
    # Read audio
    wav, sr = sf.read(audio_path)

    if wav.ndim == 2:
        wav = wav.mean(-1)

    # Convert to MLX array
    x = mx.array(wav, dtype=mx.float32)

    # Normalize if needed
    if normalize:
        x = layer_norm_mlx(x)

    x = mx.expand_dims(x, 0)

    # Create zero speaker embedding
    spk_emb = mx.zeros((1, 256), dtype=mx.float32)

    # Extract features
    feat, _ = model.extract_features(
        source=x,
        spk_emb=spk_emb,
        padding_mask=None,
        mask=False,
        output_layer=layer,
    )

    return np.array(feat[0])


def compare_features(feat_pt: np.ndarray, feat_mlx: np.ndarray, tolerance: float = 1e-3):
    """Compare features from PyTorch and MLX models."""
    logger.info(f"PyTorch features shape: {feat_pt.shape}")
    logger.info(f"MLX features shape: {feat_mlx.shape}")

    # Check shapes match
    if feat_pt.shape != feat_mlx.shape:
        logger.error(f"Shape mismatch! PyTorch: {feat_pt.shape}, MLX: {feat_mlx.shape}")
        return False

    # Compute statistics
    abs_diff = np.abs(feat_pt - feat_mlx)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    # Compute relative error
    rel_error = abs_diff / (np.abs(feat_pt) + 1e-8)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)

    logger.info(f"Max relative error: {max_rel_error}")
    logger.info(f"Mean relative error: {mean_rel_error}")

    # Compute correlation
    feat_pt_flat = feat_pt.flatten()
    feat_mlx_flat = feat_mlx.flatten()
    corr = np.corrcoef(feat_pt_flat, feat_mlx_flat)[0, 1]

    logger.info(f"Correlation: {corr}")

    # Check if close enough
    if max_diff < tolerance:
        logger.info(f"✓ Features match within tolerance {tolerance}")
        return True
    else:
        logger.warning(f"✗ Features differ by more than tolerance {tolerance}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare ContentVec features between PyTorch and MLX"
    )
    parser.add_argument("--pytorch_ckpt", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--mlx_ckpt", type=str, required=True, help="Path to MLX checkpoint (.safetensors)")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio file")
    parser.add_argument("--layer", type=int, default=12, help="Layer to extract features from")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Tolerance for feature comparison")

    args = parser.parse_args()

    # Load models
    pt_model, pt_task = load_pytorch_model(args.pytorch_ckpt)
    mlx_model = load_mlx_model(args.mlx_ckpt)

    # Extract features
    logger.info(f"Extracting PyTorch features from layer {args.layer}")
    feat_pt = extract_pytorch_features(pt_model, pt_task, args.audio, args.layer)

    logger.info(f"Extracting MLX features from layer {args.layer}")
    feat_mlx = extract_mlx_features(mlx_model, args.audio, args.layer, normalize=pt_task.cfg.normalize)

    # Compare features
    logger.info("Comparing features...")
    match = compare_features(feat_pt, feat_mlx, args.tolerance)

    if match:
        logger.info("SUCCESS: Features match!")
        return 0
    else:
        logger.error("FAILURE: Features don't match")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
