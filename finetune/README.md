# MiniFASNet Fine-tuning Suite

This directory contains tools to fine-tune MiniFASNet on hard spoofing cases and convert the resulting model to CoreML.

## Folders
- `data/`: Contains cropped training data (1_real, 0_spoof).
- `checkpoint/`: Contains the fine-tuned `.pth` and `.mlpackage` models.
- `inference/`: Dedicated scripts for running inference.

## Usage Guide

### 1. Data Preparation
```bash
python prepare_data.py
```

### 2. Fine-tuning
```bash
python train.py
```

### 3. CoreML Conversion
```bash
python convert_to_coreml.py
```

### 4. Inference
Use the scripts in the `inference/` folder to verify the results.
- **PyTorch:** `python inference/predict_finetuned.py --data_root <path>`
- **CoreML:** `python inference/predict_coreml.py --data_root <path>`