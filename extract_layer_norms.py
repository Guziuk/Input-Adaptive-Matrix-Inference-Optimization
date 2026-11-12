#!/usr/bin/env python3
"""
Extract all layer norms from Mistral safetensors.
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

print("Extracting layer norms for all 32 layers...")

for layer_idx in range(32):
    print(f"\nLayer {layer_idx}:")
    
    # Input layernorm (pre-attention)
    input_norm_name = f"model.layers.{layer_idx}.input_layernorm.weight"
    input_norm = load_weight(input_norm_name)
    input_norm_fp16 = input_norm.to(torch.float16).cpu().numpy()
    np.save(f"{OUTPUT_DIR}/layer{layer_idx}_input_layernorm.npy", input_norm_fp16)
    print(f"  ✓ input_layernorm: {input_norm.shape}")
    
    # Post-attention layernorm (pre-FFN)
    post_norm_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    post_norm = load_weight(post_norm_name)
    post_norm_fp16 = post_norm.to(torch.float16).cpu().numpy()
    np.save(f"{OUTPUT_DIR}/layer{layer_idx}_post_attention_layernorm.npy", post_norm_fp16)
    print(f"  ✓ post_attention_layernorm: {post_norm.shape}")

print("\n✓ Done! All layer norms extracted.")