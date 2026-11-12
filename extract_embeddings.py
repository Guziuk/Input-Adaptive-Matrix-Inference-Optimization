#!/usr/bin/env python3
"""
Extract embeddings and LM head from Mistral safetensors.
"""

import torch
from safetensors import safe_open
import json
import os
import numpy as np

MODEL_DIR = "/home/gamman/Desktop/Masters/Mistral-7B-v0.3"
INDEX_FILE = f"{MODEL_DIR}/model.safetensors.index.json"
OUTPUT_DIR = "weights"

def load_weight(name: str) -> torch.Tensor:
    """Load a weight by name from safetensors"""
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)["weight_map"]
    
    shard = index[name]
    path = os.path.join(MODEL_DIR, shard)
    
    with safe_open(path, framework="pt") as f:
        t = f.get_tensor(name)
    
    return t

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Extracting embeddings and LM head...")

# 1. Token embeddings
print("\n1. Loading embed_tokens...")
embed = load_weight("model.embed_tokens.weight")
print(f"   Shape: {embed.shape}, dtype: {embed.dtype}")
# Convert to FP16 to save space
embed_fp16 = embed.to(torch.float16).cpu().numpy()
np.save(f"{OUTPUT_DIR}/embed_tokens.npy", embed_fp16)
print(f"   Saved: {embed_fp16.nbytes / (1024**2):.2f} MB")

# 2. Final LayerNorm
print("\n2. Loading final norm...")
norm_weight = load_weight("model.norm.weight")
print(f"   Shape: {norm_weight.shape}, dtype: {norm_weight.dtype}")
norm_fp16 = norm_weight.to(torch.float16).cpu().numpy()
np.save(f"{OUTPUT_DIR}/norm_weight.npy", norm_fp16)
print(f"   Saved: {norm_fp16.nbytes / (1024**2):.2f} MB")

# 3. LM head
print("\n3. Loading lm_head...")
lm_head = load_weight("lm_head.weight")
print(f"   Shape: {lm_head.shape}, dtype: {lm_head.dtype}")
lm_head_fp16 = lm_head.to(torch.float16).cpu().numpy()
np.save(f"{OUTPUT_DIR}/lm_head.npy", lm_head_fp16)
print(f"   Saved: {lm_head_fp16.nbytes / (1024**2):.2f} MB")

print("\n✓ Done! Saved to", OUTPUT_DIR)
print(f"\nTotal size: {(embed_fp16.nbytes + norm_fp16.nbytes + lm_head_fp16.nbytes) / (1024**2):.2f} MB")