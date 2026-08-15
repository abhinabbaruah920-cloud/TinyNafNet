"""Synthetic throughput benchmark for GPU sizing."""
from __future__ import annotations

import argparse
import time
import torch

from model import ModelConfig, build_model
from utils import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width-px", type=int, default=256)
    p.add_argument("--width-model", type=int, default=24)
    p.add_argument("--middle-blocks", type=int, default=2)
    p.add_argument("--scale", type=int, choices=(1,2,4), default=2)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()
    device = resolve_device("cuda")
    model = build_model(ModelConfig(width=args.width_model, middle_blocks=args.middle_blocks, scale=args.scale)).to(device).eval()
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    x = torch.randn(args.batch_size, 1, args.height, args.width_px, device=device)
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(args.warmup): model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters): model(x)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ips = args.batch_size * args.iters / elapsed
    print(f"Batch: {args.batch_size}")
    print(f"Input: {args.height}x{args.width_px}")
    print(f"Iterations: {args.iters}")
    print(f"Images/sec: {ips:.3f}")
    print(f"ms/image: {1000/ips:.3f}")


if __name__ == "__main__": main()
