# Tiny NAFNet — KLA AI Restoration Hackathon

This rebuild is designed around paired NumPy arrays in:

```text
datasets/train/GT/
datasets/train/NoisyLR/
```

It also supports PNG/JPEG/TIFF folders. Validation is optional; when `datasets/val/GT` and `datasets/val/NoisyLR` are absent, `train.py` creates a deterministic validation split from the training pairs using `--val-split`.

## Install

Create a Python 3.11+ virtual environment, activate it, install the PyTorch build appropriate for your NVIDIA GPU, then:

```powershell
python -m pip install -r requirements.txt
```

## Dataset

Paired files must have the same relative path stem. For example:

```text
train/GT/0001.npy
train/NoisyLR/0001.npy
```

`.npy` arrays can be HxW, 1xHxW, HxWx1, uint8/uint16, or float arrays in common normalized/range formats. For unusual numeric ranges use `--value-scale`.

## Train on the KLA layout

```powershell
python train.py --train-degraded datasets/train/NoisyLR --train-gt datasets/train/GT --epochs 200 --batch-size 16 --patch-size 256 --workers 4
```

With explicit validation:

```powershell
python train.py --train-degraded datasets/train/NoisyLR --train-gt datasets/train/GT --val-degraded datasets/val/NoisyLR --val-gt datasets/val/GT --val-split 0 --epochs 200 --batch-size 16
```

Resume:

```powershell
python train.py --train-degraded datasets/train/NoisyLR --train-gt datasets/train/GT --resume weights/last.pt
```

## Inference

```powershell
python infer.py --weights weights/best.pt --input test_images --output results --batch-size 16
```

The inference script recursively finds `.npy`, PNG, JPEG and TIFF files and preserves relative folders. For `.npy` input it writes `.npy` output.

## Export

```powershell
python export.py --weights weights/best.pt --onnx outputs/tiny_nafnet.onnx --torchscript outputs/tiny_nafnet.ts --dynamic
```

ONNX Runtime:

```powershell
python onnx_infer.py --model outputs/tiny_nafnet.onnx --input test_images --output outputs/onnx --batch-size 16
```

## Benchmark

```powershell
python benchmark.py --weights weights/best.pt --batch-size 32 --height 256 --width-px 256
```

## GUI

```powershell
python gui.py
```

## Architecture

The default configuration is single-channel Tiny NAFNet with width 24, encoder blocks `[1,1,1,1]`, two middle blocks, and decoder blocks `[1,1,1,1]`. The NAFBlock follows the official NAFNet pattern: LayerNorm2d, pointwise expansion, depthwise 3x3 convolution, SimpleGate, simplified channel attention and residual scaling.

## Notes for H100 evaluation

Use CUDA inference, FP16 autocast, large batch sizes, and benchmark several batches before selecting the evaluation batch size. For this hackathon, compare throughput against PSNR/SSIM instead of optimizing only one metric.
