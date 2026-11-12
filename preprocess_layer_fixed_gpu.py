#!/usr/bin/env python3
"""
Fixed BucketMul preprocessing with packed FP16 format - BINARY OUTPUT VERSION.

Saves weights as raw binary files (.bin) for easy C++ loading.
No NPZ, no parsing overhead - just raw bytes!

Usage:
    python preprocess_layer_binary.py --layer 0 --out weights/
"""

import json
import numpy as np
import torch
from safetensors import safe_open
from dataclasses import dataclass
import os
import argparse

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


MODEL_DIR = "/home/gamman/Desktop/Masters/Mistral-7B-v0.3"
INDEX_FILE = f"{MODEL_DIR}/model.safetensors.index.json"


@dataclass
class BucketMulData:
    """BucketMul data with packed FP16 format (Swift-compatible)"""
    buckets: np.ndarray   # (1, in_dim*bsize, out_dim/bsize) FP16 - values+indices packed
    stats: np.ndarray     # (1, in_dim*bsize, 4) FP16 - bucket statistics  
    probes: np.ndarray    # (1, in_dim) FP16 - input importance scores
    in_dim: int
    out_dim: int
    bsize: int = 16


def pack_fp16_with_index(value: float, position_in_bucket: int) -> np.float16:
    """
    Pack a float value and 4-bit position into a single FP16.
    
    FP16 format: [sign:1 bit][exponent:5 bits][mantissa:10 bits]
    We store position (0-15) in the lower 4 mantissa bits.
    
    This causes ~0.1% precision loss but eliminates the need for 
    a separate int32 indices array, saving 3x memory.
    
    Args:
        value: The weight value (can be negative)
        position_in_bucket: Position 0-15 within the bucket
        
    Returns:
        Packed FP16 value with position encoded
    """
    # Convert to FP16
    fp16_val = np.float16(value)
    bits = np.uint16(fp16_val.view(np.uint16))  # Ensure it's a scalar uint16
    
    # Clear lower 4 bits and set to position
    # Mask: 0xFFF0 = 1111 1111 1111 0000
    bits = (bits & 0xFFF0) | (position_in_bucket & 0x000F)
    
    # Convert back to fp16 - use item() to ensure scalar
    return np.uint16(bits).view(np.float16).item()


def unpack_fp16_with_index(packed: np.float16) -> tuple:
    """
    Extract value and position from packed FP16.
    
    Args:
        packed: Packed FP16 value
        
    Returns:
        (value, position) tuple
    """
    # Convert to uint16 for bit manipulation
    bits = np.uint16(np.float16(packed).view(np.uint16))
    
    # Extract position from lower 4 bits
    position = int(bits & 0x000F)
    
    # Extract value by zeroing out lower 4 bits
    value_bits = np.uint16(bits & 0xFFF0)
    value = float(value_bits.view(np.float16))
    
    return value, position


def load_weight_to_fp16(name: str) -> np.ndarray:
    """Load a BF16 weight by name from the HF safetensors shards."""
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)["weight_map"]
    shard = index[name]
    path = os.path.join(MODEL_DIR, shard)

    print(f"Loading {name} from {path}")

    with safe_open(path, framework="pt") as f:
        t = f.get_tensor(name)  # BF16 tensor

    print(f"  dtype={t.dtype}, shape={tuple(t.shape)}")

    # BF16 -> FP32 -> FP16 -> numpy
    return t.to(torch.float32).to(torch.float16).cpu().numpy()


def bucketize_fp16_packed(W: np.ndarray, bsize: int = 16) -> BucketMulData:
    """
    BucketMul preprocessing with packed FP16 format (Swift-compatible).
    GPU-accelerated version using PyTorch with OPTIMIZED vectorized packing.
    
    Algorithm:
    1. Transpose weight matrix
    2. For each input dimension:
       - Divide row into fixed buckets of size bsize
       - Sort WITHIN each bucket by absolute value (largest first) - ON GPU
       - Store sorted values with their position in original bucket
    3. Transpose the bucket structure
    
    Key differences from original implementation:
    1. Stores values AND position in a SINGLE FP16 array (not separate arrays)
    2. Position in original bucket (0-15) encoded in lower 4 mantissa bits
    3. Sorting happens WITHIN buckets, not globally across row
    4. Stats array has 4 columns per Swift format (ALL FILLED!)
    5. Memory: Same as original weight + ~1.5% overhead
    6. GPU acceleration for sorting operations
    7. OPTIMIZED: Vectorized bit packing and stats computation
    8. Adds expert dimension for Swift compatibility
    
    Args:
        W: Weight matrix (out_dim, in_dim) in FP16
        bsize: Bucket size (16 for FP16, 8 for Q8)
        
    Returns:
        BucketMulData with packed format and correct shapes
    """
    assert W.ndim == 2
    out_dim, in_dim = W.shape
    assert out_dim % bsize == 0, f"{out_dim} not divisible by {bsize}"

    num_buckets = out_dim // bsize
    W_T = W.T.astype(np.float32, copy=False)
    
    print(f"  Bucketizing: in_dim={in_dim}, out_dim={out_dim}, bsize={bsize}")
    print(f"  Creating {in_dim * bsize} bucket rows × {num_buckets} buckets")
    print(f"  Using GPU acceleration on {DEVICE}")
    
    num_bucket_rows = in_dim * bsize
    
    # Compute probes (mean absolute value per input dimension)
    probes = np.mean(np.abs(W_T), axis=1).astype(np.float16)

    # Convert to PyTorch and move to GPU
    W_T_torch = torch.from_numpy(W_T).to(DEVICE)
    
    # Process all input dimensions at once on GPU
    print(f"    Processing all {in_dim} input dimensions on GPU...")
    
    # Reshape into buckets: (in_dim, out_dim) -> (in_dim, num_buckets, bsize)
    W_T_buckets = W_T_torch.reshape(in_dim, num_buckets, bsize)
    
    # Get absolute values for sorting
    abs_buckets = torch.abs(W_T_buckets)
    
    # Sort within each bucket (descending by absolute value)
    # sorted_indices shape: (in_dim, num_buckets, bsize)
    sorted_indices = torch.argsort(abs_buckets, dim=2, descending=True)
    
    # Gather sorted values
    # sorted_values shape: (in_dim, num_buckets, bsize)
    sorted_values = torch.gather(W_T_buckets, 2, sorted_indices)
    
    # Move back to CPU for packing
    sorted_values_cpu = sorted_values.cpu().numpy().astype(np.float32)
    sorted_indices_cpu = sorted_indices.cpu().numpy().astype(np.int32)
    
    print(f"    Packing values and positions into FP16 (vectorized)...")
    
    # Transpose dimensions: (in_dim, num_buckets, bsize) -> (in_dim, bsize, num_buckets)
    values_transposed = np.transpose(sorted_values_cpu, (0, 2, 1))
    positions_transposed = np.transpose(sorted_indices_cpu, (0, 2, 1))
    
    # Reshape to (in_dim * bsize, num_buckets)
    values_reshaped = values_transposed.reshape(num_bucket_rows, num_buckets)
    positions_reshaped = positions_transposed.reshape(num_bucket_rows, num_buckets)
    
    # OPTIMIZED PACKING - Vectorized bit operations
    # Convert values to FP16 and get bit representation
    values_fp16 = values_reshaped.astype(np.float16)
    values_bits = values_fp16.view(np.uint16)
    
    # Clear lower 4 bits and set to position (vectorized)
    values_bits_packed = (values_bits & 0xFFF0) | (positions_reshaped.astype(np.uint16) & 0x000F)
    
    # Create buckets array by reinterpreting bits
    buckets = values_bits_packed.copy()
    buckets.dtype = np.float16
    
    print(f"  Computing statistics (vectorized)...")
    
    # OPTIMIZED STATS - Vectorized computation
    # Extract values by clearing lower 4 bits (vectorized)
    buckets_bits = buckets.view(np.uint16)
    values_only_bits = buckets_bits & 0xFFF0
    values_only = values_only_bits.copy()
    values_only.dtype = np.float16
    
    # Compute mean absolute value per row (vectorized)
    abs_values = np.abs(values_reshaped.astype(np.float32))
    avg_per_row = np.mean(abs_values, axis=1).astype(np.float16)
    
    # Create stats array in Swift format: [avg, avg, avg, avg]
    # CRITICAL: Metal uses stats[3] for cutoff calculation!
    stats = np.zeros((num_bucket_rows, 4), dtype=np.float16)
    stats[:, 0] = avg_per_row
    stats[:, 1] = avg_per_row
    stats[:, 2] = avg_per_row  # Fill all columns!
    stats[:, 3] = avg_per_row  # Metal uses this!
    
    # Add expert dimension for Swift compatibility
    # Swift expects: [num_experts, rows, cols]
    # For Mistral: num_experts = 1, for Mixtral: num_experts = 8
    buckets = buckets[np.newaxis, :, :]  # (num_bucket_rows, num_buckets) -> (1, num_bucket_rows, num_buckets)
    stats = stats[np.newaxis, :, :]      # (num_bucket_rows, 4) -> (1, num_bucket_rows, 4)
    probes = probes[np.newaxis, :]       # (in_dim,) -> (1, in_dim)
    
    print(f"  ✓ Done!")
    print(f"    buckets: {buckets.shape} = {buckets.nbytes / (1024**2):.2f} MB")
    print(f"    stats:   {stats.shape} = {stats.nbytes / (1024**2):.2f} MB")
    print(f"    probes:  {probes.shape} = {probes.nbytes / (1024**2):.2f} MB")
    
    return BucketMulData(
        buckets=buckets,
        stats=stats,
        probes=probes,
        in_dim=in_dim,
        out_dim=out_dim,
        bsize=bsize,
    )


def save_bucketmul_binary(bm: BucketMulData, path: str):
    """Save BucketMul data as raw binary files for easy C++ loading"""
    
    # Save raw binary arrays (FP16 data)
    buckets_file = f"{path}_buckets.bin"
    stats_file = f"{path}_stats.bin"
    probes_file = f"{path}_probes.bin"
    meta_file = f"{path}_meta.txt"
    
    bm.buckets.tofile(buckets_file)
    bm.stats.tofile(stats_file)
    bm.probes.tofile(probes_file)
    
    # Save metadata as simple text file
    with open(meta_file, "w") as f:
        f.write(f"# BucketMul Weight Metadata\n")
        f.write(f"# All arrays are FP16 (2 bytes per element)\n\n")
        f.write(f"buckets_shape: {bm.buckets.shape[0]} {bm.buckets.shape[1]} {bm.buckets.shape[2]}\n")
        f.write(f"stats_shape: {bm.stats.shape[0]} {bm.stats.shape[1]} {bm.stats.shape[2]}\n")
        f.write(f"probes_shape: {bm.probes.shape[0]} {bm.probes.shape[1]}\n")
        f.write(f"in_dim: {bm.in_dim}\n")
        f.write(f"out_dim: {bm.out_dim}\n")
        f.write(f"bsize: {bm.bsize}\n")
        f.write(f"dtype: float16\n")
        f.write(f"\n# File sizes:\n")
        f.write(f"# buckets: {bm.buckets.nbytes} bytes ({bm.buckets.nbytes / (1024**2):.2f} MB)\n")
        f.write(f"# stats: {bm.stats.nbytes} bytes ({bm.stats.nbytes / (1024**2):.2f} MB)\n")
        f.write(f"# probes: {bm.probes.nbytes} bytes ({bm.probes.nbytes / (1024**2):.2f} MB)\n")
    
    # Calculate total size
    total_mb = (bm.buckets.nbytes + bm.stats.nbytes + bm.probes.nbytes) / (1024**2)
    
    print(f"\n  ✓ Saved binary files:")
    print(f"    {buckets_file} ({bm.buckets.nbytes / (1024**2):.2f} MB)")
    print(f"    {stats_file} ({bm.stats.nbytes / (1024**2):.2f} MB)")
    print(f"    {probes_file} ({bm.probes.nbytes / (1024**2):.2f} MB)")
    print(f"    {meta_file}")
    print(f"  Total size: {total_mb:.2f} MB")


def preprocess_layer(layer: int, out_dir: str = "."):
    """Preprocess all weights in a transformer layer"""
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    attn = {
        "q": f"model.layers.{layer}.self_attn.q_proj.weight",
        "k": f"model.layers.{layer}.self_attn.k_proj.weight",
        "v": f"model.layers.{layer}.self_attn.v_proj.weight",
        "o": f"model.layers.{layer}.self_attn.o_proj.weight",
    }

    ffn = {
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up":   f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }

    print(f"\n{'='*70}")
    print(f"PREPROCESSING LAYER {layer}")
    print(f"{'='*70}\n")

    names = {**attn, **ffn}

    for short, full in names.items():
        print(f"\n--- Processing {short}_proj ---")
        W = load_weight_to_fp16(full)
        
        # Save original FP16 for comparison/fallback
        fp16_path = os.path.join(out_dir, f"layer{layer}_{short}_proj_fp16.npy")
        np.save(fp16_path, W)
        print(f"  Saved FP16 {fp16_path}")
        print(f"  Size: {W.nbytes / (1024**2):.2f} MB")

        # Bucketize with packed format
        bm = bucketize_fp16_packed(W, bsize=16)

        # Save bucketized version as binary
        bin_path = os.path.join(out_dir, f"layer{layer}_{short}_proj_bucketmul_fp16")
        save_bucketmul_binary(bm, bin_path)
        
        # Show memory comparison
        original_mb = W.nbytes / (1024**2)
        bucketmul_mb = (bm.buckets.nbytes + bm.stats.nbytes + bm.probes.nbytes) / (1024**2)
        print(f"  Memory: {original_mb:.2f} MB → {bucketmul_mb:.2f} MB ({bucketmul_mb/original_mb:.2f}x)")
    
    print(f"\n{'='*70}")
    print(f"LAYER {layer} COMPLETE")
    print(f"{'='*70}\n")


def verify_packing():
    """Test that pack/unpack works correctly"""
    print("Testing pack/unpack functions...")
    
    test_cases = [
        (1.5, 0),
        (-0.5, 7),
        (0.0, 15),
        (3.14159, 8),
        (-2.71828, 3),
    ]
    
    for value, position in test_cases:
        packed = pack_fp16_with_index(value, position)
        unpacked_val, unpacked_pos = unpack_fp16_with_index(packed)
        
        # Check position is exact
        assert unpacked_pos == position, f"Position mismatch: {position} != {unpacked_pos}"
        
        # Check value is close (we lose ~4 bits of precision)
        rel_error = abs(unpacked_val - value) / (abs(value) + 1e-8)
        assert rel_error < 0.01, f"Value error too large: {value} vs {unpacked_val} (error={rel_error:.4f})"
        
        print(f"  ✓ ({value:8.4f}, {position:2d}) -> packed -> ({unpacked_val:8.4f}, {unpacked_pos:2d})")
    
    print("All tests passed!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Mistral layers to BucketMul packed FP16 format (BINARY OUTPUT)"
    )
    parser.add_argument("--layer", type=int, required=True,
                        help="Which layer index to preprocess (0-31)")
    parser.add_argument("--out", type=str, default=".",
                        help="Where to save the .bin and .txt files")
    parser.add_argument("--verify", action="store_true",
                        help="Run pack/unpack verification tests")
    args = parser.parse_args()
    
    if args.verify:
        verify_packing()
    
    preprocess_layer(args.layer, out_dir=args.out)
    
    print("\n" + "="*70)
    print("C++ LOADING INSTRUCTIONS:")
    print("="*70)
    print(f"\nTo load these weights in C++:")
    print(f"1. Read the metadata from *_meta.txt")
    print(f"2. Allocate buffers based on shapes")
    print(f"3. Read binary data directly:")
    print(f"   std::ifstream file(\"buckets.bin\", std::ios::binary);")
    print(f"   file.read((char*)data.data(), size);")
    print(f"4. Upload to GPU with CL_MEM_COPY_HOST_PTR")
    print("\nNo parsing, no dependencies - just raw bytes!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()