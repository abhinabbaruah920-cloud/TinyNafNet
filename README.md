# AI-Based Restoration of Degraded Semiconductor Images

## TinyNAFNet — Testing & GUI Guide

This repository contains the final TinyNAFNet solution developed for the **AI-Based Restoration of Degraded Semiconductor Images Hackathon**.

The supplied trained checkpoint is already available at https://huggingface.co/Abhinab920/TinyNAfNet :
User is required to keep it in the folder as mentioned below:
```text
weights/best.pt
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
result
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
result/000001.npy
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
python npy_to_png.py --input results --output preview_restored
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



# Quality Evaluation

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


# Final Summary

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
