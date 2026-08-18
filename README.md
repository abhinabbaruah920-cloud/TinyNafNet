# KLA AI-Based Restoration of Degraded Semiconductor Images

## TinyNAFNet — Testing & GUI Guide

This repository contains the final lightweight TinyNAFNet solution developed for the **KLA AI-Based Restoration of Degraded Semiconductor Images Hackathon**.

**This README is for testing and inference only. Training is not required.**

The supplied trained checkpoint is already available at:

```text
weights/best.pt
```

An optimized FP16 ONNX deployment model is also available at:

```text
outputs/tiny_nafnet_static16_fp16.onnx
```

---

## 1. What the Model Does

The supplied KLA data uses paired grayscale NumPy arrays:

```text
NoisyLR: 128 × 128
GT:      256 × 256
```

The final model performs **2× grayscale super-resolution and restoration**:

```text
128×128 degraded grayscale image
              ↓
        TinyNAFNet
              ↓
256×256 restored grayscale image
```

The degraded images can contain combinations of:

- Speckle noise
- Gaussian blur
- Low-resolution degradation

The model is designed to recover the clean high-resolution image while keeping the network small and GPU-efficient.

---

## 2. Final Model

```text
Architecture        TinyNAFNet
Input channels      1
Output channels     1
Scale               2×
Width               28
Encoder blocks      [1, 1, 1, 1]
Middle blocks       2
Decoder blocks      [1, 1, 1, 1]
Parameters          4,872,645 (~4.873M)
```

The network is inspired by NAFNet and uses lightweight restoration blocks with:

- Layer normalization
- Pointwise convolutions
- Depthwise convolution
- SimpleGate-style gating
- Lightweight channel attention
- Residual learning
- PixelShuffle upsampling

Heavy Transformer blocks and multi-head attention are intentionally avoided.

---

## 3. Quality Results

Current best checkpoint:

| Metric | Result |
|---|---:|
| Parameters | **4.873M** |
| PSNR | **26.9312 dB** |
| SSIM | **~0.71–0.73** |
| LPIPS | **0.3437** |

Metric direction:

```text
PSNR   → higher is better
SSIM   → higher is better
LPIPS  → lower is better
```

---

## 4. Project Structure

```text
project/
│
├── model.py
├── naf_blocks.py
├── utils.py
├── dataset.py
├── losses.py
├── metrics.py
├── trainer.py
├── train.py
├── infer.py
├── evaluate.py
├── export.py
├── onnx_infer.py
├── benchmark.py
├── run.py
├── config.py
├── npy_to_png.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── weights/
│   └── best.pt
│
├── outputs/
│   └── tiny_nafnet_static16_fp16.onnx
│
└── datasets/
    └── test/
        └── NoisyLR/
            ├── image001.npy
            ├── image002.npy
            └── ...
```

For normal testing, the tester mainly needs:

```text
run.py
```

---

## 5. Requirements

Recommended:

```text
Python 3.11+
PyTorch 2.x
NVIDIA GPU with CUDA support
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For the GUI, PySide6 must be installed. It is included in the requirements file, but it can also be installed directly:

```powershell
python -m pip install PySide6
```

### Check CUDA

Run:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected on a CUDA machine:

```text
CUDA: True
GPU: NVIDIA ...
```

---

# 6. Recommended Test Method — GUI

```powershell
python gui.py
```

The GUI is designed so that the tester does **not** need to know anything about the training pipeline.

### Step 1 — Prepare test images

Place degraded `.npy` files in:

```text
datasets/
└── test/
    └── NoisyLR/
        ├── 000001.npy
        ├── 000002.npy
        └── ...
```

The expected input shape is:

```text
128 × 128
```

### Step 2 — Start the GUI

From the project root:

```powershell
python gui.py
```

### Step 3 — Select the model

Choose:

```text
weights\best.pt
```

The checkpoint contains the model configuration, so the GUI automatically uses the correct:

```text
Width          28
Middle blocks  2
Scale          2×
Parameters     4.873M
```

### Step 4 — Select the input folder

Choose:

```text
datasets\test\NoisyLR
```

### Step 5 — Select the output folder

For example:

```text
results_gui
```

### Step 6 — Batch size

Use:

```text
16
```

This was the best-performing batch size measured on the development RTX 3050. The H100 may support a different optimal batch size.

### Step 7 — Device

Choose:

```text
GPU / CUDA
```

### Step 8 — Click Start

The GUI will:

1. Load `weights/best.pt`.
2. Detect the saved model configuration.
3. Read the `.npy` inputs.
4. Process images in batches.
5. Use FP16 CUDA inference when GPU is selected.
6. Save the restored `.npy` outputs.
7. Update progress and statistics.
8. Show a degraded/restored preview.

---

# 7. GUI Features

The GUI includes:

### Model selector

Select the trained checkpoint.

### Input folder

Folder containing degraded `.npy` files.

### Output folder

Folder where restored results are saved.

### Batch size

Controls the number of images processed per inference batch.

### GPU/CPU execution

GPU is recommended for performance. CPU mode is available for functional testing.

### Progress bar

Shows percentage of the folder processed.

### Current file

Displays the file currently being processed.

### Images processed

Displays the completed image count.

### Throughput

Reports:

```text
images/sec
```

### Forward time

Reports approximate neural-network execution time:

```text
ms/image
```

### Before / After preview

The GUI displays:

```text
Degraded Input       Restored Output
128×128               256×256
```

### Live log

Displays model loading, processing, and completion messages.

### Open Output Folder

Opens the folder containing the generated results.

---

# 8. Output Files

For an input:

```text
datasets/test/NoisyLR/000001.npy
```

the GUI produces:

```text
results_gui/000001.npy
```

Expected output shape:

```text
256 × 256
```

The saved output is clipped to the GT range:

```text
[0, 1]
```

---

# 9. Viewing `.npy` Files as PNG

NumPy arrays cannot normally be opened by standard image viewers.

Use the included conversion utility:

```powershell
python npy_to_png.py --input datasets\test\NoisyLR --output preview_noisy
```

For restored outputs:

```powershell
python npy_to_png.py --input results_gui --output preview_restored
```

For visualization-only contrast normalization:

```powershell
python npy_to_png.py --input datasets\test\NoisyLR --output preview_noisy --auto-range
```

`--auto-range` is only for viewing. **Do not use it before inference**, because it changes the original intensity distribution.

---

# 10. Command-Line PyTorch Test

The GUI is recommended, but the same model can be tested from the command line:

```powershell
python infer.py --weights weights\best.pt --input datasets\test\NoisyLR --output results --batch-size 16
```

The script reports:

```text
Images
Forward time
Forward images/sec
Forward ms/image
End-to-end time
End-to-end images/sec
End-to-end ms/image
```

---

# 11. ONNX FP16 Test

An optimized static-batch FP16 ONNX model is included:

```text
outputs/tiny_nafnet_static16_fp16.onnx
```

Run:

```powershell
python onnx_infer.py --model outputs\tiny_nafnet_static16_fp16.onnx --input datasets\test\NoisyLR --output results_onnx --batch-size 16
```

The model expects:

```text
Input:
[16, 1, 128, 128]

Output:
[16, 1, 256, 256]
```

The ONNX path uses CUDA execution and FP16 input.

---

# 12. Local Throughput Benchmarks

Development was performed on:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
4 GB VRAM
```

### PyTorch FP16

```text
Batch:              16
Forward throughput: ~278 images/sec
End-to-end:         ~212 images/sec
```

### Static FP16 ONNX

```text
Batch:              16
Forward throughput: ~296 images/sec
End-to-end:         ~178 images/sec
```

These are development-machine results. They should not be interpreted as expected H100 throughput.

---

# 13. Quality Evaluation

If ground truth is available, the model can be evaluated with:

```text
PSNR
SSIM
LPIPS
```

Example:

```powershell
python evaluate.py --weights weights\best.pt --degraded datasets\train\NoisyLR --gt datasets\train\GT --val-split 0.1 --batch-size 4
```

Example baseline:

```text
PSNR : 26.9312 dB
SSIM : 0.707151
LPIPS: 0.343738
```

LPIPS is lower-is-better.

---

# 14. Troubleshooting

## `ModuleNotFoundError`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## CUDA is False

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Make sure a CUDA-enabled PyTorch build is installed.

## No supported files found

The input folder should contain `.npy` files for the KLA test data.

## Size mismatch

The final model is designed for:

```text
128×128 input
256×256 output
```

and uses scale 2.

## ONNX invalid dimensions

The included static ONNX model was exported for batch 16. Use:

```text
--batch-size 16
```

If you need arbitrary batch sizes, use the dynamic ONNX model instead.

## `torch.compile` / Triton error

`torch.compile` is not required for testing. Use the normal GUI or `infer.py` path.

---

# 15. Quick Tester Checklist

A tester can run the project with only these steps:

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Check CUDA

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Put test `.npy` files in

```text
datasets/test/NoisyLR/
```

### 4. Start the GUI

```powershell
python gui.py
```

### 5. Select

```text
Model:       weights/best.pt
Input:       datasets/test/NoisyLR
Output:      results_gui
Batch size:  16
Device:      GPU/CUDA
```

### 6. Click `Start`

### 7. Check the output directory

```text
results_gui/
```

The outputs should be restored 256×256 `.npy` arrays.

---

# 16. Final Submission Artifacts

The two primary model artifacts are:

```text
weights/best.pt
outputs/tiny_nafnet_static16_fp16.onnx
```

`best.pt` is the canonical trained PyTorch model.

The FP16 ONNX model is the optimized deployment candidate for GPU inference.

---

# 17. Final Summary

```text
Model:              TinyNAFNet
Parameters:         4.873M
Input:              1 × 128 × 128
Output:             1 × 256 × 256
Scale:              2×
Precision:          FP16 inference
Input format:       .npy
Output format:      .npy

PSNR:               ~26.93 dB
SSIM:               ~0.71–0.73
LPIPS:              ~0.344

Local PyTorch:      ~278 images/sec
Local ONNX FP16:    ~296 images/sec
```

## Testing command

The simplest complete test is:

```powershell
python gui.py
```

Then select `weights/best.pt`, choose the test `.npy` folder, set batch size 16, select GPU, and click **Start**.

No model training is required for testing.

---

## References

- NAFNet — Simple Baselines for Image Restoration
- PyTorch — https://pytorch.org/
- ONNX Runtime — https://onnxruntime.ai/
- LPIPS — https://github.com/richzhang/PerceptualSimilarity
