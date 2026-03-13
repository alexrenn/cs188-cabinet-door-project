#!/usr/bin/env python3
"""Analyze action statistics in training data."""
import numpy as np
import pyarrow.parquet as pq
import os
import glob

chunk_dir = "/tmp/cabinet_22dim_dataset/data/chunk-000"
files = sorted(glob.glob(os.path.join(chunk_dir, "*.parquet")))
print(f"Found {len(files)} parquet files")

all_actions = []
for f in files:
    table = pq.read_table(f)
    df = table.to_pandas()
    action_cols = [c for c in df.columns if c == "action" or c.startswith("action.")]
    for _, row in df.iterrows():
        parts = []
        for c in action_cols:
            v = row[c]
            if isinstance(v, np.ndarray):
                parts.extend(v.flatten().tolist())
            else:
                parts.append(float(v))
        all_actions.append(parts)

actions = np.array(all_actions)
print(f"Shape: {actions.shape}")
print(f"\nPer-dim stats:")
for i in range(actions.shape[1]):
    print(f"  dim {i:2d}: min={actions[:,i].min():+.4f}  max={actions[:,i].max():+.4f}  mean={actions[:,i].mean():+.4f}  std={actions[:,i].std():.4f}")
print(f"\nGlobal: min={actions.min():.4f}  max={actions.max():.4f}")

# How many actions exceed |1.0| (Tanh ceiling)?
over1 = np.abs(actions) > 1.0
print(f"\nActions with |value| > 1.0: {over1.sum()} / {actions.size} ({100*over1.mean():.2f}%)")
over05 = np.abs(actions) > 0.5
print(f"Actions with |value| > 0.5: {over05.sum()} / {actions.size} ({100*over05.mean():.2f}%)")
