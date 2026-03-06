"""
ensemble_baseline_v2.py — Ensemble of two crop strategies with Strict AND fusion.

Method 1 (Landmark crop): margin=4.5
Method 2 (Full-width crop)

Fusion rule (Strict AND across ALL models and crops):
  → "Real"  ONLY if every model on every crop predicts "Real"
  → "Spoof" if ANY model on ANY crop predicts "Spoof"

Usage:
    python ensemble_baseline_v2.py --data_root <image_or_folder>
"""

import os
import sys
import argparse
import time
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

def run_minifas_strict(models, face_crop, transform, device):
    """
    Run models on a crop.
    Returns (label, score) where label is 'Real' only if ALL models predict 'Real'.
    """
    for m in models:
        img_r = cv2.resize(face_crop, (m["w"], m["h"]), interpolation=cv2.INTER_LINEAR)
        inp   = transform(img_r).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(m["model"](inp), dim=1).cpu().numpy()[0]
        
        idx = int(np.argmax(probs))
        if idx != 1: # Not Real
            return "Spoof", float(probs[idx])
            
    # If all Real, return average score
    total_score = 0
    for m in models:
        img_r = cv2.resize(face_crop, (m["w"], m["h"]), interpolation=cv2.INTER_LINEAR)
        inp   = transform(img_r).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(m["model"](inp), dim=1).cpu().numpy()[0]
        total_score += probs[1]
    
    return "Real", float(total_score) / len(models)

# ── ensemble predictor ────────────────────────────────────────────────────────

class EnsemblePredictorV2:
    def __init__(self, model_dir, landmark_margin=4.5):
        self.device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.landmark_margin = landmark_margin
        self.transform       = trans.Compose([trans.ToTensor()])

        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)

        self.models = []
        model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))
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
        boxes, _, landmarks = self.detector.detect(img_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return []

        pts = landmarks[0]
        
        # Method 1: landmark-based
        crop1, b1  = crop_from_5landmarks(img_bgr, pts, margin=self.landmark_margin)
        label1, s1 = run_minifas_strict(self.models, crop1, self.transform, self.device)

        # Method 2: full-width
        crop2, b2  = full_width_square_crop(img_bgr)
        label2, s2 = run_minifas_strict(self.models, crop2, self.transform, self.device)

        # Strict AND across ALL: Real only if both crops were Real
        if label1 == "Real" and label2 == "Real":
            final_label = "Real"
            final_score = (s1 + s2) / 2
        else:
            final_label = "Spoof"
            final_score = s2 if label2 == "Spoof" else s1

        return [{
            "box_xywh":   b1,
            "label":      final_label,
            "score":      final_score,
            "label1":     label1, "score1": s1,
            "label2":     label2, "score2": s2,
        }]

def draw_and_save(img_path, results, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None: return
    for r in results:
        x, y, w, h = r["box_xywh"]
        label, score = r["label"], r["score"]
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {score:.4f} [Crop1:{r['label1']} | Crop2:{r['label2']}]"
        fs = max(0.4, 0.45 * img.shape[0] / 1024)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, fs, color, 1)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--model_dir",  type=str, default="./resources/anti_spoof_models")
    parser.add_argument("--margin",     type=float, default=4.5)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    predictor = EnsemblePredictorV2(args.model_dir, landmark_margin=args.margin)
    files = (
        sorted(os.path.join(args.data_root, f)
               for f in os.listdir(args.data_root)
               if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if os.path.isdir(args.data_root) else [args.data_root]
    )

    spoof_total = real_total = noface_total = 0
    total_time = 0
    processed_count = 0

    for f in files:
        t0 = time.time()
        res = predictor.predict(f)
        dt = (time.time() - t0) * 1000 # ms
        
        if not res:
            noface_total += 1
            print(f"File: {os.path.basename(f)} - {dt:>5.1f}ms - No face detected.")
            continue
        
        processed_count += 1
        total_time += dt
        r = res[0]
        if r["label"] == "Spoof":
            spoof_total += 1
        else:
            real_total += 1
            
        print(f"File: {os.path.basename(f)} - {dt:>5.1f}ms - Label={r['label']} | Score={r['score']:.4f}")
        draw_and_save(f, res, args.output_dir)

    avg_time = total_time / processed_count if processed_count > 0 else 0
    print("\n----- Summary -----")
    print(f"Total files processed: {len(files)}")
    print(f"  Spoof: {spoof_total}")
    print(f"  Real: {real_total}")
    print(f"  No face detected: {noface_total}")
    print(f"  Average inference time: {avg_time:.2f} ms")
