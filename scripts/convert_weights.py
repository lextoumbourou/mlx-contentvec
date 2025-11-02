#!/usr/bin/env python3
"""
Convert PyTorch ContentVec weights to MLX format.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

# Add vendor paths for loading PyTorch model
sys.path.insert(0, "vendor/contentvec")
sys.path.insert(0, "vendor/fairseq")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("convert_weights")


class RestrictedUnpickler(pickle.Unpickler):
    """Custom unpickler that skips loading fairseq modules we don't need."""
    def find_class(self, module, name):
        # Skip fairseq configuration classes that cause issues
        if module.startswith('fairseq.') and not module.startswith('fairseq.data.dictionary'):
            if 'Config' in name or 'Task' in name:
                # Return a dummy class for configs/tasks
                return type(name, (), {})
        return super().find_class(module, name)


def load_pytorch_model(ckpt_path: str):
    """Load PyTorch ContentVec model."""
    logger.info(f"Loading PyTorch model from {ckpt_path}")

    # Load checkpoint with custom unpickler to avoid dataclass errors
    with open(ckpt_path, 'rb') as f:
        unpickler = RestrictedUnpickler(f)
        ckpt = unpickler.load()

    # Extract model state dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        cfg = ckpt.get('cfg', None)
    else:
        state_dict = ckpt
        cfg = None

    logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")

    # Infer config from state dict if not available
    encoder_layers = 12  # Default
    encoder_layers_1 = 3  # Default

    # Count encoder layers from state dict
    layer_keys = [k for k in state_dict.keys() if k.startswith('encoder.layers.')]
    if layer_keys:
        max_layer_idx = max([int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit()])
        total_layers = max_layer_idx + 1
        # Assume last 3 are conditional (encoder_layers_1)
        encoder_layers = total_layers - 3
        encoder_layers_1 = 3

    logger.info(f"Inferred config:")
    logger.info(f"  encoder_layers: {encoder_layers}")
    logger.info(f"  encoder_layers_1: {encoder_layers_1}")

    return state_dict, encoder_layers, encoder_layers_1


def convert_conv_feature_extractor(state_dict: dict, mlx_weights: dict):
    """Convert convolutional feature extractor weights."""
    logger.info("Converting feature extractor...")

    # Find all conv layer indices
    conv_indices = set()
    for key in state_dict.keys():
        if key.startswith('feature_extractor.conv_layers.'):
            parts = key.split('.')
            if len(parts) >= 3 and parts[2].isdigit():
                conv_indices.add(int(parts[2]))

    conv_indices = sorted(conv_indices)
    logger.info(f"Found {len(conv_indices)} conv layers")

    for i in conv_indices:
        prefix = f"feature_extractor.conv_layers.{i}"

        # Convert conv weights (PyTorch: (out, in, k) -> MLX: (out, k, in))
        if f"{prefix}.conv.weight" in state_dict:
            pt_conv_weight = state_dict[f"{prefix}.conv.weight"].numpy()
            mlx_conv_weight = np.transpose(pt_conv_weight, (0, 2, 1))
            mlx_weights[f"{prefix}.conv.weight"] = mlx_conv_weight

        if f"{prefix}.conv.bias" in state_dict:
            mlx_weights[f"{prefix}.conv.bias"] = state_dict[f"{prefix}.conv.bias"].numpy()

        # Convert normalization if present
        if f"{prefix}.layer_norm.weight" in state_dict:
            mlx_weights[f"{prefix}.norm.weight"] = state_dict[f"{prefix}.layer_norm.weight"].numpy()
            mlx_weights[f"{prefix}.norm.bias"] = state_dict[f"{prefix}.layer_norm.bias"].numpy()
        elif f"{prefix}.group_norm.weight" in state_dict:
            mlx_weights[f"{prefix}.norm.weight"] = state_dict[f"{prefix}.group_norm.weight"].numpy()
            mlx_weights[f"{prefix}.norm.bias"] = state_dict[f"{prefix}.group_norm.bias"].numpy()


def convert_transformer_layer(state_dict: dict, prefix: str, mlx_weights: dict, is_conditional: bool = False):
    """Convert a transformer layer."""
    # Self-attention
    for param in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        if f"{prefix}.self_attn.{param}.weight" in state_dict:
            mlx_weights[f"{prefix}.self_attn.{param}.weight"] = state_dict[f"{prefix}.self_attn.{param}.weight"].numpy()
        if f"{prefix}.self_attn.{param}.bias" in state_dict:
            mlx_weights[f"{prefix}.self_attn.{param}.bias"] = state_dict[f"{prefix}.self_attn.{param}.bias"].numpy()

    # Feed-forward
    for param in ['fc1', 'fc2']:
        if f"{prefix}.{param}.weight" in state_dict:
            mlx_weights[f"{prefix}.{param}.weight"] = state_dict[f"{prefix}.{param}.weight"].numpy()
        if f"{prefix}.{param}.bias" in state_dict:
            mlx_weights[f"{prefix}.{param}.bias"] = state_dict[f"{prefix}.{param}.bias"].numpy()

    # Layer norms (conditional or regular)
    if is_conditional:
        # CondLayerNorm - check for both patterns
        for norm_name in ['self_attn_layer_norm', 'final_layer_norm']:
            if f"{prefix}.{norm_name}.weight" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.weight"] = state_dict[f"{prefix}.{norm_name}.weight"].numpy()
            if f"{prefix}.{norm_name}.bias" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.bias"] = state_dict[f"{prefix}.{norm_name}.bias"].numpy()
            if f"{prefix}.{norm_name}.linear.weight" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.linear.weight"] = state_dict[f"{prefix}.{norm_name}.linear.weight"].numpy()
            if f"{prefix}.{norm_name}.linear.bias" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.linear.bias"] = state_dict[f"{prefix}.{norm_name}.linear.bias"].numpy()
    else:
        # Regular LayerNorm
        for norm_name in ['self_attn_layer_norm', 'final_layer_norm']:
            if f"{prefix}.{norm_name}.weight" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.weight"] = state_dict[f"{prefix}.{norm_name}.weight"].numpy()
            if f"{prefix}.{norm_name}.bias" in state_dict:
                mlx_weights[f"{prefix}.{norm_name}.bias"] = state_dict[f"{prefix}.{norm_name}.bias"].numpy()


def convert_transformer_encoder(state_dict: dict, mlx_weights: dict, encoder_layers: int, encoder_layers_1: int):
    """Convert transformer encoder weights."""
    logger.info("Converting transformer encoder...")

    # Positional convolution (with weight normalization)
    # Convert weight (PyTorch: (out, in, k) -> MLX: (out, k, in))
    if "encoder.pos_conv.0.weight_v" in state_dict:
        pt_weight = state_dict["encoder.pos_conv.0.weight_v"].numpy()
        mlx_weight = np.transpose(pt_weight, (0, 2, 1))
        mlx_weights["encoder.pos_conv.0.module.weight"] = mlx_weight
    elif "encoder.pos_conv.0.module.weight" in state_dict:
        pt_weight = state_dict["encoder.pos_conv.0.module.weight"].numpy()
        mlx_weight = np.transpose(pt_weight, (0, 2, 1))
        mlx_weights["encoder.pos_conv.0.module.weight"] = mlx_weight

    if "encoder.pos_conv.0.bias" in state_dict:
        mlx_weights["encoder.pos_conv.0.module.bias"] = state_dict["encoder.pos_conv.0.bias"].numpy()
    elif "encoder.pos_conv.0.module.bias" in state_dict:
        mlx_weights["encoder.pos_conv.0.module.bias"] = state_dict["encoder.pos_conv.0.module.bias"].numpy()

    # Convert transformer layers
    for i in range(encoder_layers):
        prefix = f"encoder.layers.{i}"
        convert_transformer_layer(state_dict, prefix, mlx_weights, is_conditional=False)

    # Convert conditional transformer layers
    for i in range(encoder_layers_1):
        prefix = f"encoder.layers.{encoder_layers + i}"
        convert_transformer_layer(state_dict, prefix, mlx_weights, is_conditional=True)

    # Final layer norm
    if "encoder.layer_norm.weight" in state_dict:
        mlx_weights["encoder.layer_norm.weight"] = state_dict["encoder.layer_norm.weight"].numpy()
    if "encoder.layer_norm.bias" in state_dict:
        mlx_weights["encoder.layer_norm.bias"] = state_dict["encoder.layer_norm.bias"].numpy()

    # Conditional layer norm if present
    if encoder_layers_1 > 0:
        if "encoder.cond_layer_norm.weight" in state_dict:
            mlx_weights["encoder.cond_layer_norm.weight"] = state_dict["encoder.cond_layer_norm.weight"].numpy()
        if "encoder.cond_layer_norm.bias" in state_dict:
            mlx_weights["encoder.cond_layer_norm.bias"] = state_dict["encoder.cond_layer_norm.bias"].numpy()
        if "encoder.cond_layer_norm.linear.weight" in state_dict:
            mlx_weights["encoder.cond_layer_norm.linear.weight"] = state_dict["encoder.cond_layer_norm.linear.weight"].numpy()
        if "encoder.cond_layer_norm.linear.bias" in state_dict:
            mlx_weights["encoder.cond_layer_norm.linear.bias"] = state_dict["encoder.cond_layer_norm.linear.bias"].numpy()


def convert_weights(pt_ckpt: str, mlx_ckpt: str):
    """Convert PyTorch weights to MLX format."""
    # Load PyTorch state dict
    state_dict, encoder_layers, encoder_layers_1 = load_pytorch_model(pt_ckpt)

    # Initialize MLX weights dictionary
    mlx_weights = {}

    # Convert layer norm
    logger.info("Converting layer norm...")
    if "layer_norm.weight" in state_dict:
        mlx_weights["layer_norm.weight"] = state_dict["layer_norm.weight"].numpy()
    if "layer_norm.bias" in state_dict:
        mlx_weights["layer_norm.bias"] = state_dict["layer_norm.bias"].numpy()

    # Convert post-extract projection if present
    if "post_extract_proj.weight" in state_dict:
        logger.info("Converting post-extract projection...")
        mlx_weights["post_extract_proj.weight"] = state_dict["post_extract_proj.weight"].numpy()
        mlx_weights["post_extract_proj.bias"] = state_dict["post_extract_proj.bias"].numpy()

    # Convert feature extractor
    convert_conv_feature_extractor(state_dict, mlx_weights)

    # Convert transformer encoder
    convert_transformer_encoder(
        state_dict,
        mlx_weights,
        encoder_layers,
        encoder_layers_1,
    )

    # Save to safetensors file
    logger.info(f"Saving MLX weights to {mlx_ckpt}")
    save_file(mlx_weights, mlx_ckpt)

    logger.info(f"Conversion complete! Saved {len(mlx_weights)} weight tensors")

    # Print summary
    total_params = sum(v.size for v in mlx_weights.values())
    logger.info(f"Total parameters: {total_params:,}")

    # Print first few keys for verification
    logger.info("Sample weight keys:")
    for i, key in enumerate(sorted(mlx_weights.keys())[:10]):
        logger.info(f"  {key}: {mlx_weights[key].shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch ContentVec weights to MLX")
    parser.add_argument(
        "--pytorch_ckpt",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--mlx_ckpt",
        type=str,
        required=True,
        help="Path to output MLX checkpoint (.safetensors file)",
    )

    args = parser.parse_args()

    convert_weights(args.pytorch_ckpt, args.mlx_ckpt)


if __name__ == "__main__":
    main()
