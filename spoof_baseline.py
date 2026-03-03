"""
spoof_baseline.py
Uses MTCNN (facenet_pytorch) for face detection + landmark-based crop.
Runs anti-spoof inference using CoreML (.mlpackage) models via coremltools.

Crop logic and output format are identical to single_baseline.py.

Usage:
    python3 spoof_baseline.py --data_root <image_or_folder>
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from facenet_pytorch import MTCNN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utility import parse_model_name


# ─────────────────── landmark-based crop (same as single_baseline) ───────────

def add_face_margin(x, y, w, h, margin=0.5):
    x_margin = int(w * margin / 2)
    y_margin = int(h * margin / 2)
    x1 = x - x_margin
    x2 = x + w + x_margin
    y1 = y - y_margin
    y2 = y + h + y_margin
    return x1, x2, y1, y2


def crop_from_5landmarks(img, pts, margin=2.0):
    """
    Crop a square patch from `img` using 5 facial landmarks.
    pts : ndarray (5, 2) — [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    x_list = pts[:, 0]
    y_list = pts[:, 1]
    x = round(float(min(x_list)))
    y = round(float(min(y_list)))
    w = round(float(max(x_list))) - x
    h = round(float(max(y_list))) - y
    side = max(w, h)
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(max_w, x2);  y2 = min(max_h, y2)
    return img[y1:y2, x1:x2], [x1, y1, x2 - x1, y2 - y1]


# ──────────────────────────── CoreML helpers ──────────────────────────────────

def parse_mlpackage_name(folder_name):
    """
    Parse h_input, w_input, model_type, scale from an .mlpackage folder name.
    Works the same as parse_model_name() but without requiring '.pth' suffix.
    e.g. '2.7_80x80_MiniFASNetV2' -> (80, 80, 'MiniFASNetV2', 2.7)
    """
    # strip .mlpackage if present
    base = folder_name.replace(".mlpackage", "")
    # reuse the existing utility (it splits on '.pth' which is absent here,
    # so split('.pth')[0] simply returns base — behaviour is identical)
    return parse_model_name(base)


def preprocess_for_coreml(img_bgr, h_input, w_input):
    """
    Resize, convert BGR→RGB, normalise to [0, 1] float32, return (1, 3, H, W).
    Matches the torchvision ToTensor() transform used in single_baseline.py.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w_input, h_input),
                             interpolation=cv2.INTER_LINEAR)
    arr = img_resized.astype(np.float32) / 255.0          # HWC [0,1]
    arr = np.transpose(arr, (2, 0, 1))                     # CHW
    arr = np.expand_dims(arr, axis=0)                      # 1CHW
    return arr, img_resized   # also return the uint8 resized crop (BGR→RGB)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────── Main predictor ──────────────────────────────────

class AntiSpoofPredictor:
    def __init__(self, model_dir, landmark_margin=3.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.landmark_margin = landmark_margin

        # MTCNN — keep_all=False returns best face with landmarks
        self.detector = MTCNN(
            keep_all=False,
            device=self.device,
            post_process=False,
        )

        self.models = []
        model_folders = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".mlpackage")
        )
        if not model_folders:
            print(f"[WARN] No .mlpackage folders found in {model_dir}")

        for folder_name in model_folders:
            model_path = os.path.join(model_dir, folder_name)
            print(f"  Loading model: {folder_name}")
            h_input, w_input, model_type, scale = parse_mlpackage_name(folder_name)
            ml_model = ct.models.MLModel(model_path)
            self.models.append({
                "model":    ml_model,
                "h_input":  h_input,
                "w_input":  w_input,
                "name":     folder_name,
            })

        print(f"  Loaded {len(self.models)} CoreML model(s) from {model_dir}")

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load {img_path}")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.detector.detect(img_rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        # Use landmarks of the best (first) detection to produce a square crop
        pts = landmarks[0]                                  # (5, 2)
        face_crop, bbox = crop_from_5landmarks(
            img_bgr, pts, margin=self.landmark_margin
        )

        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        prediction = np.zeros(3)

        for m in self.models:
            arr, img_resized_rgb = preprocess_for_coreml(
                face_crop, m["h_input"], m["w_input"]
            )

            # Optionally save the crop (saved as BGR for cv2.imwrite)
            crop_save_path = os.path.join(
                "./output/crops",
                f"{base_name}_{m['name'].replace('.mlpackage', '')}.jpg"
            )
            os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)
            cv2.imwrite(crop_save_path,
                        cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2BGR))

            # CoreML inference — input key is "input", output key is "logits"
            out = m["model"].predict({"input": arr})
            logits = np.array(out["logits"]).flatten()
            probs_arr = softmax(logits)
            prediction += probs_arr

        label_idx = int(np.argmax(prediction))
        label     = "Real" if label_idx == 1 else "Spoof"
        score     = float(prediction[label_idx]) / len(self.models)

        return [{
            "box_xywh":   bbox,
            "label":      label,
            "score":      score,
            "prediction": prediction,
        }]


# ──────────────────────────── Visualisation ──────────────────────────────────

def draw_and_save(img_path, results, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        return
    for r in results:
        x, y, w, h = r["box_xywh"]
        label, score = r["label"], r["score"]
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {score:.4f}"
        font_scale = 0.5 * img.shape[0] / 1024
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - baseline - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness)
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)


# ──────────────────────────────── CLI ────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Path to image or folder")
    parser.add_argument("--model_dir",  type=str,
                        default="./models",
                        help="Directory containing .mlpackage anti-spoof model folders")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--margin",     type=float, default=3.5,
                        help="Margin multiplier for landmark-based crop (default 3.5)")
    args = parser.parse_args()

    predictor = AntiSpoofPredictor(args.model_dir, landmark_margin=args.margin)

    if args.data_root:
        if os.path.isdir(args.data_root):
            files = sorted(
                os.path.join(args.data_root, f)
                for f in os.listdir(args.data_root)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            )
        else:
            files = [args.data_root]

        spoof_total   = 0
        real_total    = 0
        noface_total  = 0

        for f in files:
            res = predictor.predict(f)
            if not res:
                noface_total += 1
                print(f"File: {os.path.basename(f)} - No face detected.")
                continue
            for r in res:
                if r["label"] == "Spoof":
                    spoof_total += 1
                elif r["label"] == "Real":
                    real_total += 1
                print(f"File: {os.path.basename(f)} - Label={r['label']} | Score={r['score']:.4f}.")
            draw_and_save(f, res, output_dir=args.output_dir)

        print("\n----- Summary -----")
        print(f"Total files processed: {len(files)}")
        print(f"  Spoof: {spoof_total}")
        print(f"  Real:  {real_total}")
        print(f"  No face detected: {noface_total}")
    else:
        print("Please provide --data_root to run inference.")
