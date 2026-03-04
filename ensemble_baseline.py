"""
ensemble_baseline.py — Ensemble of two crop strategies for anti-spoofing.

Method 1 (Landmark crop): single_baseline.py with margin=4.5
  → MTCNN 5-landmark based wide crop

Method 2 (Full-width crop): single_baseline_crop_full.py
  → Square crop with side = image width, centered vertically

Fusion rule (strict AND):
  → "Real"  only if BOTH models predict "Real"
  → "Spoof" if EITHER model predicts "Spoof"

Usage:
    python ensemble_baseline.py --data_root <image_or_folder>
    python ensemble_baseline.py --data_root spoof/
    python ensemble_baseline.py --data_root real/
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.utility import get_kernel, parse_model_name
from src.data_io import transform as trans


MODEL_MAPPING = {
    "MiniFASNetV1":   MiniFASNetV1,
    "MiniFASNetV2":   MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


# ── crop helpers ──────────────────────────────────────────────────────────────

def add_face_margin(x, y, w, h, margin=0.5):
    xm = int(w * margin / 2)
    ym = int(h * margin / 2)
    return x - xm, x + w + xm, y - ym, y + h + ym


def crop_from_5landmarks(img, pts, margin=4.5):
    """Method 1 — landmark-based wide crop."""
    x_list = pts[:, 0];  y_list = pts[:, 1]
    x = round(float(min(x_list)));  y = round(float(min(y_list)))
    w = round(float(max(x_list))) - x
    h = round(float(max(y_list))) - y
    side = max(w, h)
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(max_w, x2);  y2 = min(max_h, y2)
    return img[y1:y2, x1:x2], [x1, y1, x2 - x1, y2 - y1]


def full_width_square_crop(img):
    """Method 2 — full-width square crop."""
    h, w = img.shape[:2]
    side = w
    cy   = h // 2
    y1   = max(0, cy - side // 2)
    y2   = y1 + side
    crop = img[y1:min(h, y2), 0:w].copy()
    pad  = side - crop.shape[0]
    if pad > 0:
        crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return crop, [0, y1, w, side]


# ── model helpers ─────────────────────────────────────────────────────────────

def load_model(model_path, device):
    model_name  = os.path.basename(model_path)
    h, w, mtype, _ = parse_model_name(model_name)
    kernel      = get_kernel(h, w)
    model       = MODEL_MAPPING[mtype](conv6_kernel=kernel, num_classes=3)
    sd          = torch.load(model_path, map_location=device)
    sd          = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, h, w


def run_minifas(models, face_crop, transform, device):
    """Run MiniFASNet ensemble on a crop, return (label, score)."""
    prediction = np.zeros(3)
    for m in models:
        img_r = cv2.resize(face_crop, (m["w"], m["h"]), interpolation=cv2.INTER_LINEAR)
        inp   = transform(img_r).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(m["model"](inp), dim=1).cpu().numpy()[0]
        prediction += probs
    idx   = int(np.argmax(prediction))
    label = "Real" if idx == 1 else "Spoof"
    score = float(prediction[idx]) / len(models)
    return label, score


# ── ensemble predictor ────────────────────────────────────────────────────────

class EnsemblePredictor:
    def __init__(self, model_dir, landmark_margin=4.5):
        self.device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.landmark_margin = landmark_margin
        self.transform       = trans.Compose([trans.ToTensor()])

        # MTCNN detector
        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)

        # MiniFAS models
        self.models = []
        model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))
        if not model_files:
            print(f"[WARN] No .pth files found in {model_dir}")
        for mf in model_files:
            path = os.path.join(model_dir, mf)
            print(f"  Loading model: {mf}")
            model, h, w = load_model(path, self.device)
            self.models.append({"model": model, "h": h, "w": w, "name": mf})
        print(f"  Loaded {len(self.models)} model(s) from {model_dir}")

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return []

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # face presence check
        boxes, _, landmarks = self.detector.detect(img_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return []

        # ── Method 1: landmark-based wide crop ────────────────────────────────
        pts        = landmarks[0]   # (5, 2)
        crop1, b1  = crop_from_5landmarks(img_bgr, pts, margin=self.landmark_margin)
        label1, s1 = run_minifas(self.models, crop1, self.transform, self.device)

        # ── Method 2: full-width square crop ──────────────────────────────────
        crop2, b2  = full_width_square_crop(img_bgr)
        label2, s2 = run_minifas(self.models, crop2, self.transform, self.device)

        # ── AND fusion ────────────────────────────────────────────────────────
        if label1 == "Real" and label2 == "Real":
            final_label = "Real"
            final_score = (s1 + s2) / 2
            final_box   = b1                  # landmark box as display bbox
        else:
            final_label = "Spoof"
            # pick the spoof's score (whichever triggered "Spoof")
            final_score = s2 if label2 == "Spoof" else s1
            final_box   = b1

        return [{
            "box_xywh":   final_box,
            "label":      final_label,
            "score":      final_score,
            "label1":     label1,  "score1": s1,
            "label2":     label2,  "score2": s2,
        }]


# ── draw + save ───────────────────────────────────────────────────────────────

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
        text = f"{label}: {score:.4f} [{r['label1']}:{r['score1']:.2f}|{r['label2']}:{r['score2']:.2f}]"
        fs = max(0.4, 0.45 * img.shape[0] / 1024)
        th = 2
        (tw, tth), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, fs, th)
        cv2.rectangle(img, (x, y - tth - bl - 4), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - bl - 2), cv2.FONT_HERSHEY_COMPLEX,
                    fs, (255, 255, 255), th)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble: landmark crop (margin=4.5) AND full-width crop"
    )
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--model_dir",  type=str, default="./resources/anti_spoof_models")
    parser.add_argument("--margin",     type=float, default=4.5,
                        help="Landmark crop margin (default 4.5)")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    predictor = EnsemblePredictor(args.model_dir, landmark_margin=args.margin)

    files = (
        sorted(os.path.join(args.data_root, f)
               for f in os.listdir(args.data_root)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if os.path.isdir(args.data_root) else [args.data_root]
    )

    spoof_total = real_total = noface_total = 0

    for f in files:
        res = predictor.predict(f)
        if not res:
            noface_total += 1
            print(f"File: {os.path.basename(f)} - No face detected.")
            continue
        for r in res:
            if r["label"] == "Spoof":
                spoof_total += 1
            else:
                real_total += 1
            print(
                f"File: {os.path.basename(f)} - Label={r['label']} | Score={r['score']:.4f}."
                f"  [lm={r['label1']}:{r['score1']:.3f} | fw={r['label2']}:{r['score2']:.3f}]"
            )
        draw_and_save(f, res, output_dir=args.output_dir)

    print("\n----- Summary -----")
    print(f"Total files processed: {len(files)}")
    print(f"  Spoof: {spoof_total}")
    print(f"  Real: {real_total}")
    print(f"  No face detected: {noface_total}")
