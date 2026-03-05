# Face Anti-Spoofing — MiniFASNet Test Suite

Pipeline: **MTCNN** (detect + 5-landmark crop) → **MiniFASNet V1SE / V2** (anti-spoof classification) → **Real / Spoof**

---

## Project Structure

```
test-spoof/
├── real/                    # Real face images for testing
├── spoof/                   # Spoof face images for testing
│
├── resources/
│   └── anti_spoof_models/   # PyTorch .pth model weights
│       ├── 2.7_80x80_MiniFASNetV2.pth
│       └── 4_0_0_80x80_MiniFASNetV1SE.pth
│
├── models/                  # CoreML .mlpackage model bundles
│   ├── 2.7_80x80_MiniFASNetV2.mlpackage
│   └── 4_0_0_80x80_MiniFASNetV1SE.mlpackage
│
├── src/
│   ├── model_lib/           # MiniFASNet model definitions
│   │   └── MiniFASNet.py    # MiniFASNetV1, V2, V1SE, V2SE
│   ├── data_io/             # Data transforms
│   ├── anti_spoof_predict.py
│   ├── generate_patches.py
│   ├── utility.py           # parse_model_name, get_kernel
│   └── default_config.py
│
├── single_baseline.py       # PyTorch inference (.pth models, Linux / Windows / Mac)
├── spoof_baseline.py        # CoreML inference (.mlpackage models, macOS / Apple Silicon)
│
├── output/                  # Annotated output images (auto-created)
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install facenet-pytorch opencv-python numpy torch coremltools
```

---

### 2. PyTorch baseline (`single_baseline.py`)

Uses MTCNN for face detection, crops via 5-point landmarks, and runs inference with the `.pth` MiniFASNet models.

**Works on:** Linux / Windows / macOS

```bash
# Single image
python single_baseline.py --data_root spoof/f8ebe9dc-0aa9-4a6a-99b0-a047c6d5544f.jpg
```

**Outputs:**
```bash
File: f8ebe9dc-0aa9-4a6a-99b0-a047c6d5544f.jpg - Label=Spoof | Score=0.9942.
```

```bash
# Full folder real
python single_baseline.py --data_root real/
python single_baseline.py --data_root spoof/
```

**Outputs:**

```bash
----- Summary -----
Total files processed: 335
  Spoof: 0
  Real: 334
  No face detected: 1
```

```bash
# Full folder spoof
python single_baseline.py --data_root spoof/
```

**Outputs:**

```bash
----- Summary -----
Total files processed: 15
  Spoof: 13
  Real: 0
  No face detected: 2
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Path to an image or folder of images |
| `--model_dir` | `./resources/anti_spoof_models` | Directory containing `.pth` model files |
| `--output_dir` | `./output` | Directory for annotated output images |
| `--margin` | `3.5` | Crop margin multiplier (landmark-based) |

**Example output:**
```
  Loading model: 2.7_80x80_MiniFASNetV2.pth
  Loading model: 4_0_0_80x80_MiniFASNetV1SE.pth
  Loaded 2 model(s) from ./resources/anti_spoof_models
File: sample_real.jpg   - Label=Real  | Score=0.8731.
File: sample_spoof.jpg  - Label=Spoof | Score=0.9124.

----- Summary -----
Total files processed: 2
  Spoof: 1
  Real:  1
  No face detected: 0
```

---

### 3. CoreML baseline (`spoof_baseline.py`)

Same detection + crop logic, but runs inference with `.mlpackage` CoreML models via `coremltools`.

**Works on:** macOS / Apple Silicon (CoreML runtime required)

```bash
# Single image
python3 spoof_baseline.py --data_root real/sample.jpg

# Full folder
python3 spoof_baseline.py --data_root real/
python3 spoof_baseline.py --data_root spoof/
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Path to an image or folder of images |
| `--model_dir` | `./models` | Directory containing `.mlpackage` model folders |
| `--output_dir` | `./output` | Directory for annotated output images |
| `--margin` | `3.5` | Crop margin multiplier (landmark-based) |

---

## Model Details

### MiniFASNetV2 — `2.7_80x80_MiniFASNetV2`
- Input: RGB 80×80, normalised to `[0, 1]`
- Output: 3-class logits → softmax → `[Spoof, Real, Other]`
- Scale factor: 2.7

### MiniFASNetV1SE — `4_0_0_80x80_MiniFASNetV1SE`
- Input: RGB 80×80, normalised to `[0, 1]`
- Output: 3-class logits → softmax → `[Spoof, Real, Other]`
- Scale factor: 4.0

### Prediction fusion
Both models run on the same cropped face. Their softmax outputs are **summed** and `argmax` selects the final class:
- Index `1` → **Real**
- Index `0` or `2` → **Spoof**

### Face detection
MTCNN (`facenet_pytorch`) is used for face detection. The 5 facial landmarks are used to derive a tight bounding box, which is then expanded with a configurable margin and cropped to a square region.

---

---

## Fine-tuning & CoreML

Tools for fine-tuning the MiniFASNet model on custom datasets and converting results to Apple's CoreML format.

### 1. Prepare Data
Crops faces from `real/` and `spoof/` folders with a 2.7 margin and resizes to 80x80.
```bash
cd finetune
python prepare_data.py
```

### 2. Training (Fine-tuning)
Fine-tunes the model for 20 epochs. Weights are saved to `finetune/checkpoint/finetuned.pth`.
```bash
cd finetune
python train.py
```

### 3. Inference (PyTorch .pth)
Run inference using the fine-tuned `.pth` model.
```bash
cd finetune/inference
python predict_finetuned.py --data_root ../../spoof/spoof15.png
```

### 4. Convert to CoreML
Converts the fine-tuned `.pth` model to Apple's `.mlpackage` format.
```bash
cd finetune
python convert_to_coreml.py
```

### 5. Inference (CoreML .mlpackage)
Run inference using the `.mlpackage` via `coremltools`.
**Note:** `.predict()` normally requires macOS.
```bash
cd finetune/inference
python predict_coreml.py --data_root ../../spoof/spoof15.png
```
