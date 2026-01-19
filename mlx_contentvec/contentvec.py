"""
ContentVec model implementation in MLX.

Based on the fairseq ContentVec implementation.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from .conv_feature_extraction import ConvFeatureExtractionModel, lengths_to_padding_mask
from .transformer_encoder import TransformerEncoder_1

logger = logging.getLogger(__name__)

# HuggingFace Hub defaults for easy weight loading
HF_REPO_ID = "lexandstuff/mlx-contentvec"
HF_WEIGHTS_FILE = "contentvec_base.safetensors"


class ContentvecModel(nn.Module):
    """
    ContentVec model for audio feature extraction.

    This model combines:
    - Convolutional feature extraction from raw audio
    - Transformer encoding with optional speaker conditioning
    - Feature projection and normalization
    """

    def __init__(
        self,
        # Feature extractor config
        conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        conv_bias: bool = False,
        extractor_mode: str = "default",
        # Encoder config
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        encoder_layers: int = 12,
        encoder_layers_1: int = 3,
        # Positional encoding config
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        encoder_layerdrop: float = 0.0,
        # Other config
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        feature_grad_mult: float = 1.0,
        dim_spk: int = 256,
    ):
        """
        Args:
            conv_feature_layers: String describing convolutional feature extraction layers
            conv_bias: Whether to include bias in conv encoder
            extractor_mode: Mode for feature extractor ("default", "group_norm_masked", "layer_norm")
            encoder_embed_dim: Encoder embedding dimension
            encoder_ffn_embed_dim: Encoder FFN embedding dimension
            encoder_attention_heads: Number of attention heads
            encoder_layers: Number of regular transformer layers
            encoder_layers_1: Number of speaker-conditioned transformer layers
            conv_pos: Number of filters for convolutional positional embeddings
            conv_pos_groups: Number of groups for convolutional positional embedding
            dropout: Dropout probability for the transformer
            attention_dropout: Dropout probability for attention weights
            activation_dropout: Dropout probability after activation in FFN
            dropout_input: Dropout to apply to the input (after feat extr)
            dropout_features: Dropout to apply to the features (after feat extr)
            encoder_layerdrop: Probability of dropping a transformer layer
            activation_fn: Activation function to use
            layer_norm_first: Apply layernorm first in the transformer
            feature_grad_mult: Multiply feature extractor var grads by this (for training)
            dim_spk: Speaker embedding dimension
        """
        super().__init__()

        # Parse conv feature layers from string
        feature_enc_layers = eval(conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        # Feature extractor
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=conv_bias,
        )

        # Post-extraction projection
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim
            else None
        )

        # Dropout layers
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed)

        # Transformer encoder
        self.encoder = TransformerEncoder_1(
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_ffn_embed_dim,
            num_attention_heads=encoder_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
            conv_pos=conv_pos,
            conv_pos_groups=conv_pos_groups,
            encoder_layers=encoder_layers,
            encoder_layers_1=encoder_layers_1,
            layerdrop=encoder_layerdrop,
            dim_spk=dim_spk,
        )

        self.feature_grad_mult = feature_grad_mult

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = HF_REPO_ID,
        filename: str = HF_WEIGHTS_FILE,
        weights_path: Optional[Union[str, Path]] = None,
    ) -> "ContentvecModel":
        """
        Load a pretrained ContentVec model.

        Downloads weights from HuggingFace Hub if not provided locally.

        Args:
            repo_id: HuggingFace repo ID (default: "lexandstuff/mlx-contentvec")
            filename: Weights filename in repo (default: "contentvec_base.safetensors")
            weights_path: Local path to weights file (overrides HF download)

        Returns:
            Initialized ContentvecModel with loaded weights

        Example:
            # Simple usage (auto-downloads from HuggingFace)
            model = ContentvecModel.from_pretrained()

            # Custom repo
            model = ContentvecModel.from_pretrained(repo_id="my-org/my-weights")

            # Local weights
            model = ContentvecModel.from_pretrained(weights_path="./weights.safetensors")
        """
        if weights_path is None:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading weights from {repo_id}/{filename}")
            weights_path = hf_hub_download(repo_id, filename)

        # Create model with RVC-compatible settings (no speaker conditioning layers)
        model = cls(encoder_layers_1=0)
        model.load_weights(weights_path)
        model.eval()

        return model

    def load_weights(self, weights_path: str):
        """
        Load model weights from a safetensors file.

        Args:
            weights_path: Path to the .safetensors weights file
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        logger.info(f"Loading weights from {weights_path}")

        # Load weights from safetensors (flat keys like "encoder.layers.0.fc1.weight")
        flat_weights = {}
        with safe_open(weights_path, framework="numpy") as f:
            for key in f.keys():
                flat_weights[key] = mx.array(f.get_tensor(key))

        # Convert flat keys to nested dict structure that MLX's update() expects
        weights = self._unflatten_weights(flat_weights)

        # Update model parameters
        self.update(weights)
        logger.info(f"Loaded {len(flat_weights)} weight tensors")

    def _unflatten_weights(self, flat_weights: dict) -> dict:
        """
        Convert flat weight keys like "encoder.layers.0.fc1.weight"
        into nested dict structure that MLX's update() expects.
        """
        result = {}
        for key, value in flat_weights.items():
            parts = key.split(".")
            current = result
            for i, part in enumerate(parts[:-1]):
                # Check if next part is a numeric index
                if part not in current:
                    # Look ahead to see if we need a list
                    next_part = parts[i + 1]
                    if next_part.isdigit():
                        current[part] = {}  # Will convert to list later
                    else:
                        current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Post-process: convert dict with numeric keys to lists
        return self._convert_numeric_dicts_to_lists(result)

    def _convert_numeric_dicts_to_lists(self, d):
        """Recursively convert dicts with all-numeric keys to lists."""
        if not isinstance(d, dict):
            return d

        # Check if all keys are numeric strings
        if d and all(k.isdigit() for k in d.keys()):
            # Convert to list, filling gaps with None
            max_idx = max(int(k) for k in d.keys())
            lst = [None] * (max_idx + 1)
            for k, v in d.items():
                lst[int(k)] = self._convert_numeric_dicts_to_lists(v)
            return lst
        else:
            return {k: self._convert_numeric_dicts_to_lists(v) for k, v in d.items()}

    def forward_features(
        self, source: mx.array, padding_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Extract convolutional features from raw audio.

        Args:
            source: Raw audio tensor of shape (batch, time)
            padding_mask: Optional padding mask where True indicates padding

        Returns:
            Features of shape (batch, channels, time')
        """
        # Note: In MLX, we don't have separate gradient contexts like torch.no_grad()
        # The feature_grad_mult would be used during training with gradient manipulation
        features = self.feature_extractor(source, padding_mask)
        return features

    def forward_padding_mask(
        self, features: mx.array, padding_mask: mx.array
    ) -> mx.array:
        """
        Update padding mask for batch inference after feature extraction.

        Args:
            features: Feature tensor from feature extractor
            padding_mask: Original padding mask

        Returns:
            Updated padding mask accounting for conv downsampling
        """
        # Compute new lengths after convolution
        # Original formula: (lengths_org - 400) // 320 + 1
        lengths_org = mx.sum(~padding_mask, axis=1).astype(mx.float32)
        lengths = mx.floor((lengths_org - 400) / 320) + 1
        lengths = lengths.astype(mx.int32)
        padding_mask = lengths_to_padding_mask(lengths)
        return padding_mask

    def extract_features(
        self,
        source: mx.array,
        spk_emb: Optional[mx.array] = None,
        padding_mask: Optional[mx.array] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        tap: bool = False,
    ) -> tuple[mx.array, Optional[mx.array]]:
        """
        Extract features from audio through the full model.

        Args:
            source: Raw audio tensor of shape (batch, time)
            spk_emb: Speaker embedding of shape (batch, dim_spk)
            padding_mask: Optional padding mask where True indicates padding
            mask: Whether to apply masking (for training)
            ret_conv: If True, return conv features instead of transformer output
            output_layer: If specified, return features from this layer (1-indexed)
            tap: If True, return intermediate layer results

        Returns:
            Tuple of (features, padding_mask)
            - features: Extracted features of shape (batch, time, embed_dim)
            - padding_mask: Updated padding mask or None
        """
        # Create default zero speaker embedding if not provided
        # Shape: (batch_size, dim_spk=256)
        if spk_emb is None:
            batch_size = source.shape[0]
            spk_emb = mx.zeros((batch_size, 256))

        # Extract convolutional features
        features = self.forward_features(source, padding_mask)

        # Update padding mask if provided
        if padding_mask is not None:
            logger.info("Batch generation mode!")
            padding_mask = self.forward_padding_mask(features, padding_mask)

        # Transpose from (B, C, L) to (B, L, C)
        features = features.transpose(0, 2, 1)

        # Apply layer normalization
        features = self.layer_norm(features)

        # Apply post-extraction projection if needed
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # Return conv features if requested
        if ret_conv:
            return features, padding_mask

        # Pass through transformer encoder
        x, layer_results = self.encoder(
            features,
            spk_emb,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
            tap=tap,
        )

        return x, padding_mask

    def __call__(
        self,
        source: mx.array,
        spk_emb: Optional[mx.array] = None,
        padding_mask: Optional[mx.array] = None,
        mask: bool = False,
        features_only: bool = True,
        output_layer: Optional[int] = None,
        tap: bool = False,
    ) -> dict[str, mx.array]:
        """
        Forward pass through the model.

        Args:
            source: Raw audio tensor of shape (batch, time)
            spk_emb: Speaker embedding of shape (batch, dim_spk)
            padding_mask: Optional padding mask where True indicates padding
            mask: Whether to apply masking (for training)
            features_only: If True, return only extracted features
            output_layer: If specified, return features from this layer (1-indexed)
            tap: If True, return intermediate layer results

        Returns:
            Dictionary containing extracted features and optionally padding mask
        """
        x, padding_mask = self.extract_features(
            source=source,
            spk_emb=spk_emb,
            padding_mask=padding_mask,
            mask=mask,
            ret_conv=False,
            output_layer=output_layer,
            tap=tap,
        )

        result = {"x": x, "padding_mask": padding_mask}
        return result
