"""Inspect one .npy file before training."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument("path", type=Path); args=p.parse_args()
x=np.load(args.path); print("file:", args.path); print("shape:", x.shape); print("dtype:", x.dtype); print("min:", x.min()); print("max:", x.max()); print("mean:", x.mean()); print("std:", x.std())
