#!/usr/bin/env python3
"""
Feature dumping script for ContentVec MLX implementation.

Based on fairseq's dump_hubert_feature.py but adapted for MLX.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
import tqdm
from npy_append_array import NpyAppendArray

from mlx_contentvec import ContentvecModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_contentvec_feature")


def layer_norm(x: mx.array) -> mx.array:
    """Apply layer normalization (equivalent to F.layer_norm in PyTorch)."""
    mean = mx.mean(x)
    std = mx.sqrt(mx.mean((x - mean) ** 2))
    return (x - mean) / (std + 1e-5)


class ContentvecFeatureReader:
    """Feature reader for ContentVec model."""

    def __init__(
        self,
        ckpt_path: str,
        layer: int,
        max_chunk: int = 1600000,
        sample_rate: int = 16000,
        normalize: bool = False,
        dim_spk: int = 256,
    ):
        """
        Initialize the feature reader.

        Args:
            ckpt_path: Path to model checkpoint (.safetensors file)
            layer: Which transformer layer to extract features from (1-indexed)
            max_chunk: Maximum chunk size for processing long audio
            sample_rate: Expected sample rate of audio files
            normalize: Whether to normalize audio input
            dim_spk: Speaker embedding dimension
        """
        self.layer = layer
        self.max_chunk = max_chunk
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.dim_spk = dim_spk

        logger.info(f"Loading model from {ckpt_path}")

        # Initialize model
        self.model = ContentvecModel(
            # These should match the checkpoint config
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
            dropout=0.0,  # No dropout for inference
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
        self.model.load_weights(ckpt_path)

        logger.info(f"Model loaded successfully")
        logger.info(f"sample_rate = {self.sample_rate}")
        logger.info(f"normalize = {self.normalize}")
        logger.info(f"max_chunk = {self.max_chunk}")
        logger.info(f"output_layer = {self.layer}")

    def read_audio(self, path: str, ref_len: int = None) -> np.ndarray:
        """
        Read audio file and validate format.

        Args:
            path: Path to audio file
            ref_len: Expected length (for validation)

        Returns:
            Audio waveform as numpy array
        """
        wav, sr = sf.read(path)

        if sr != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sr} for {path}")

        # Convert stereo to mono
        if wav.ndim == 2:
            wav = wav.mean(-1)

        assert wav.ndim == 1, f"Expected 1D waveform, got {wav.ndim}D"

        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logger.warning(f"ref {ref_len} != read {len(wav)} ({path})")

        return wav

    def get_feats(self, path: str, ref_len: int = None) -> mx.array:
        """
        Extract features from audio file.

        Args:
            path: Path to audio file
            ref_len: Expected length (for validation)

        Returns:
            Extracted features as MLX array
        """
        # Read audio
        x = self.read_audio(path, ref_len)

        # Convert to MLX array
        x = mx.array(x, dtype=mx.float32)

        # Apply normalization if needed
        if self.normalize:
            x = layer_norm(x)

        # Add batch dimension: (T,) -> (1, T)
        x = mx.expand_dims(x, 0)

        # Create zero speaker embedding (1, dim_spk)
        spk_emb = mx.zeros((1, self.dim_spk), dtype=mx.float32)

        # Process in chunks if needed
        feat = []
        for start in range(0, x.shape[1], self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]

            # Extract features
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                spk_emb=spk_emb,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )

            feat.append(feat_chunk)

        # Concatenate chunks along time dimension
        result = mx.concatenate(feat, axis=1)

        # Remove batch dimension: (1, T, D) -> (T, D)
        result = result[0]

        return result


def get_shard_range(tot: int, nshard: int, rank: int) -> tuple[int, int]:
    """Calculate the range of items to process for this shard."""
    assert rank < nshard and rank >= 0, f"invalid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv_path: str, nshard: int, rank: int):
    """
    Create an iterator over audio files from a TSV manifest.

    TSV format:
    - First line: root directory
    - Subsequent lines: <relative_path>\t<num_samples>
    """
    with open(tsv_path, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

    return iterate, len(lines)


def dump_feature(
    reader: ContentvecFeatureReader,
    generator,
    num: int,
    split: str,
    nshard: int,
    rank: int,
    feat_dir: str,
):
    """
    Dump features to file.

    Args:
        reader: Feature reader instance
        generator: Generator function that yields (path, nsample) tuples
        num: Total number of files to process
        split: Split name (e.g., "train", "valid")
        nshard: Total number of shards
        rank: Current shard rank
        feat_dir: Output directory for features
    """
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            # Convert MLX array to numpy for saving
            feat_np = np.array(feat)
            feat_f.append(feat_np)
            leng_f.write(f"{len(feat_np)}\n")

    logger.info("finished successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Dump ContentVec features from audio files"
    )
    parser.add_argument("tsv_dir", type=str, help="Directory containing TSV manifest files")
    parser.add_argument("split", type=str, help="Split name (e.g., train, valid)")
    parser.add_argument("ckpt_path", type=str, help="Path to model checkpoint (.safetensors file)")
    parser.add_argument("layer", type=int, help="Which layer to extract features from (1-indexed)")
    parser.add_argument("nshard", type=int, help="Total number of shards")
    parser.add_argument("rank", type=int, help="Current shard rank (0-indexed)")
    parser.add_argument("feat_dir", type=str, help="Output directory for features")
    parser.add_argument("--max_chunk", type=int, default=1600000, help="Maximum chunk size")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio input")
    parser.add_argument("--dim_spk", type=int, default=256, help="Speaker embedding dimension")

    args = parser.parse_args()

    logger.info(args)

    # Create feature reader
    reader = ContentvecFeatureReader(
        ckpt_path=args.ckpt_path,
        layer=args.layer,
        max_chunk=args.max_chunk,
        sample_rate=args.sample_rate,
        normalize=args.normalize,
        dim_spk=args.dim_spk,
    )

    # Get path iterator
    tsv_path = f"{args.tsv_dir}/{args.split}.tsv"
    generator, num = get_path_iterator(tsv_path, args.nshard, args.rank)

    # Dump features
    dump_feature(reader, generator, num, args.split, args.nshard, args.rank, args.feat_dir)


if __name__ == "__main__":
    main()
