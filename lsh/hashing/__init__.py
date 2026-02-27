"""
Locality-Sensitive Hashing module.

Implements PCA dimensionality reduction, random-projection hashing,
and bin partitioning. The mathematical formulation is preserved exactly
from the original ``hashing.py`` and ``distil.py``.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from lsh.config import HashingConfig
from lsh.logging_utils import get_logger


# ---------------------------------------------------------------------------
# Step A: Flatten and stack descriptors
# ---------------------------------------------------------------------------
def flatten_and_stack_descriptors(
    descriptor_folder: str,
    save_path: str,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Load per-frame ``.npy`` descriptor files, flatten each, and stack.

    Parameters
    ----------
    descriptor_folder : str
        Directory containing ``descriptor_frame_*.npy`` files.
    save_path : str
        Directory to save the combined tensor.
    device : torch.device
        Target device for tensor operations.

    Returns
    -------
    torch.Tensor
        Shape ``(n_frames, flattened_dim)``.
    """
    logger = get_logger()
    files = sorted([f for f in os.listdir(descriptor_folder) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {descriptor_folder}")

    flattened: list[torch.Tensor] = []
    for fname in files:
        arr = np.load(os.path.join(descriptor_folder, fname))
        vec = torch.FloatTensor(arr.flatten())
        flattened.append(vec)

    combined = torch.stack(flattened, dim=0).to(device)
    logger.info("Combined descriptor tensor shape: %s", tuple(combined.shape))

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "combined_tensor.npy"), combined.cpu().numpy())
    return combined


# ---------------------------------------------------------------------------
# Step B: PCA dimensionality reduction
# ---------------------------------------------------------------------------
def reduce_dimensionality(
    tensor: torch.Tensor,
    n_components: int,
    save_path: str,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Apply PCA to reduce dimensionality.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape ``(n_frames, feature_dim)``.
    n_components : int
        Number of principal components.
    save_path : str
        Directory to save reduced tensor.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Reduced tensor of shape ``(n_frames, n_components)``.
    """
    logger = get_logger()
    pca = PCA(n_components=n_components)
    reduced_np = pca.fit_transform(tensor.cpu().numpy())

    variance_retained = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA: %d components, variance retained: %.4f", n_components, variance_retained)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"reduced_tensor_{n_components}.npy"), reduced_np)

    return torch.FloatTensor(reduced_np).to(device)


# ---------------------------------------------------------------------------
# Step C: Hashing and partitioning (EXACT original algorithm)
# ---------------------------------------------------------------------------
def hashed_values(
    data: torch.Tensor,
    no_of_hash: int,
    feature_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate hashed values via random projection.

    Parameters
    ----------
    data : torch.Tensor
        Normalised input of shape ``(n, feature_size)``.
    no_of_hash : int
        Number of hash functions.
    feature_size : int
        Feature dimension.
    device : torch.device
        Device for computation.

    Returns
    -------
    torch.Tensor
        Projected values of shape ``(n, no_of_hash)``.
    """
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0, 1).to(device)
    data = data.to(device)
    return torch.matmul(data, Wl.T)


def partition(
    list_bin_width: list[float],
    bin_values: torch.Tensor,
    no_of_hash: int,
    random_seed: int = 42,
) -> dict[float, dict[int, int]]:
    """
    Partition hashed values into clusters / buckets.

    Algorithm is identical to the original implementation.

    Parameters
    ----------
    list_bin_width : list[float]
        Bin widths to apply.
    bin_values : torch.Tensor
        Hash-projected values.
    no_of_hash : int
        Number of hash functions.
    random_seed : int
        Seed (for bias generation).

    Returns
    -------
    dict[float, dict[int, int]]
        Mapping ``{bin_width: {frame_idx: cluster_id}}``.
    """
    summary: dict[float, dict[int, int]] = {}
    for bw in list_bin_width:
        bias = torch.tensor([random.uniform(-bw, bw) for _ in range(no_of_hash)])
        # Move bias to same device as bin_values
        bias = bias.to(bin_values.device)
        temp = torch.floor((1.0 / bw) * (bin_values + bias))
        cluster, _ = torch.max(temp, dim=1)

        mapping: dict[int, int] = {}
        for i in range(bin_values.shape[0]):
            mapping[i] = int(cluster[i].item())
        summary[bw] = mapping
    return summary


# ---------------------------------------------------------------------------
# Step D: Full hashing pipeline
# ---------------------------------------------------------------------------
def process_with_hashing(
    combined_tensor: torch.Tensor,
    hashing_cfg: HashingConfig,
    output_folder: str,
    device: torch.device = torch.device("cpu"),
) -> dict[float, dict[int, int]]:
    """
    Normalise → hash → partition → save.

    Parameters
    ----------
    combined_tensor : torch.Tensor
        Reduced (PCA) tensor.
    hashing_cfg : HashingConfig
        Hashing parameters.
    output_folder : str
        Directory for output files.
    device : torch.device
        Computation device.

    Returns
    -------
    dict[float, dict[int, int]]
        Hash cluster mapping.
    """
    logger = get_logger()
    bw = hashing_cfg.bin_width
    no_of_hash = hashing_cfg.n_hash

    # Normalise
    combined_tensor = torch.nn.functional.normalize(combined_tensor.to(device), p=2, dim=1)

    # Hash
    bin_vals = hashed_values(combined_tensor, no_of_hash, combined_tensor.shape[1], device=device)

    # Partition
    clusters = partition([bw], bin_vals, no_of_hash, random_seed=hashing_cfg.random_seed)

    # Stats
    unique = set()
    for _, mapping in clusters.items():
        unique.update(mapping.values())
    logger.info("Unique clusters: %d", len(unique))

    # Save hash buckets
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, f"hash_buckets_flattened_{bw}.txt")
    with open(out_file, "w") as fh:
        for width, mapping in clusters.items():
            for node, cid in mapping.items():
                fh.write(f"{node}: {cid}\n")

    logger.info("Hash buckets saved to %s", out_file)
    return clusters


# ---------------------------------------------------------------------------
# Step E: Organise into bins (replaces distil.py)
# ---------------------------------------------------------------------------
def organise_bins(
    hash_buckets_file: str,
    output_file: str,
) -> dict[int, list[int]]:
    """
    Read a hash-buckets file and organise frames into bins.

    Parameters
    ----------
    hash_buckets_file : str
        Path to ``hash_buckets_flattened_<bw>.txt``.
    output_file : str
        Path to write ``output_bins_<bw>.txt``.

    Returns
    -------
    dict[int, list[int]]
        Mapping ``{bin_index: [frame_numbers]}``.
    """
    logger = get_logger()
    bin_to_frames: dict[int, list[int]] = {}

    with open(hash_buckets_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            frame_str, bin_str = line.split(":")
            frame_num = int(frame_str.strip())
            bin_idx = int(bin_str.strip())
            bin_to_frames.setdefault(bin_idx, []).append(frame_num)

    total_frames = sum(len(v) for v in bin_to_frames.values())
    total_bins = len(bin_to_frames)

    with open(output_file, "w") as fh:
        fh.write(f"Total Frames: {total_frames}\n")
        fh.write(f"Total Bins: {total_bins}\n\n")
        for bin_idx in sorted(bin_to_frames.keys()):
            frames_str = ", ".join(map(str, bin_to_frames[bin_idx]))
            fh.write(f"Bin {bin_idx}: [{frames_str}]\n")

    logger.info("Bins: %d total, %d frames → %s", total_bins, total_frames, output_file)
    return bin_to_frames
