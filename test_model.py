"""
test_model.py — Granular testing of individual model and crop combinations.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

# Add current directory to path
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def add_face_margin(x, y, w, h, margin=0.5):
    xm = int(w * margin / 2)
    ym = int(h * margin / 2)
    return x - xm, x + w + xm, y - ym, y + h + ym

def crop_from_5landmarks(img, pts, margin=4.0):
    x_list = pts[:, 0];  y_list = pts[:, 1]
    x = round(float(min(x_list)));  y = round(float(min(y_list)))
    w = round(float(max(x_list))) - x
    h = round(float(max(y_list))) - y
    side = max(w, h)
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(max_w, x2);  y2 = min(max_h, y2)
    return img[y1:y2, x1:x2]

def full_width_square_crop(img):
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
    return crop

def load_model(model_path, device):
    model_name = os.path.basename(model_path)
    h, w, mtype, _ = parse_model_name(model_name)
    kernel = get_kernel(h, w)
    model = MODEL_MAPPING[mtype](conv6_kernel=kernel, num_classes=3)
    sd = torch.load(model_path, map_location=device)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return {"model": model, "h": h, "w": w, "name": mtype}

def run_inference(model_dict, crop, transform, device):
    img_r = cv2.resize(crop, (model_dict["w"], model_dict["h"]), interpolation=cv2.INTER_LINEAR)
    inp = transform(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_dict["model"](inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    label = "Real" if idx == 1 else "Spoof"
    score = float(probs[idx])
    return label, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=False, device=device, post_process=False)
    transform = trans.Compose([trans.ToTensor()])

    # Path to specific models
    path_v2 = os.path.join(args.model_dir, "2.7_80x80_MiniFASNetV2.pth")
    path_v1se = os.path.join(args.model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")

    m_v2 = load_model(path_v2, device)
    m_v1se = load_model(path_v1se, device)

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        print(f"Error: Could not read image {args.img}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, _, landmarks = detector.detect(img_rgb, landmarks=True)

    if boxes is None or len(boxes) == 0:
        print("No face detected.")
        sys.exit(1)

    pts = landmarks[0]
    crop_40 = crop_from_5landmarks(img_bgr, pts, margin=4.0)
    crop_full = full_width_square_crop(img_bgr)

    print(f"\nProcessing image: {os.path.basename(args.img)}")
    print("-" * 50)

    # 1. V1 4.0
    l1, s1 = run_inference(m_v1se, crop_40, transform, device)
    print(f"V1 + Crop 4.0:   Label={l1:<6} | Score={s1:.4f}")

    # 2. V1 full
    l2, s2 = run_inference(m_v1se, crop_full, transform, device)
    print(f"V1 + Full Crop:  Label={l2:<6} | Score={s2:.4f}")

    # 3. V2 4.0
    l3, s3 = run_inference(m_v2, crop_40, transform, device)
    print(f"V2 + Crop 4.0:   Label={l3:<6} | Score={s3:.4f}")

    # 4. V2 full
    l4, s4 = run_inference(m_v2, crop_full, transform, device)
    print(f"V2 + Full Crop:  Label={l4:<6} | Score={s4:.4f}")
    print("-" * 50)
