"""
analysis_baseline.py — Compare 12 anti-spoofing options with Hard Vote logic and Pure Model Timing.
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

def crop_from_5landmarks(img, pts, margin=2.7):
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

def get_label_single_timed(model, crop, transform, device):
    t0 = time.time()
    img_r = cv2.resize(crop, (model["w"], model["h"]), interpolation=cv2.INTER_LINEAR)
    inp = transform(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model["model"](inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    dt = (time.time() - t0) * 1000
    label = "Real" if np.argmax(probs) == 1 else "Spoof"
    return label, dt

# ── Main Class ───────────────────────────────────────────────────────────────

class AnalysisPredictor:
    def __init__(self, resources_dir):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)
        self.transform = trans.Compose([trans.ToTensor()])

        path_v2 = os.path.join(resources_dir, "2.7_80x80_MiniFASNetV2.pth")
        path_v1se = os.path.join(resources_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")

        print(f"Loading models from {resources_dir}...")
        self.m_v2 = load_model(path_v2, self.device)
        self.m_v1se = load_model(path_v1se, self.device)

    def predict_detailed(self, img_path):
        """Returns (results_dict, times_dict, total_runtime) or (None, None, 0.0)."""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return None, None, 0.0

        t_start = time.time()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _, landmarks = self.detector.detect(img_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0: 
            return [], None, (time.time() - t_start) * 1000

        pts = landmarks[0]
        
        # Crops are NOT timed as part of per-option "model time"
        crop_27, _ = crop_from_5landmarks(img_bgr, pts, margin=2.7)
        crop_40, _ = crop_from_5landmarks(img_bgr, pts, margin=4.0)
        crop_full, _ = full_width_square_crop(img_bgr)

        results = {}
        times = {}

        # 1. MiniFASNetV2 crop 2.7
        l1, t1 = get_label_single_timed(self.m_v2, crop_27, self.transform, self.device)
        results["Opt1"], times["Opt1"] = l1, t1

        # 2. MiniFASNetV1SE crop 2.7
        l2, t2 = get_label_single_timed(self.m_v1se, crop_27, self.transform, self.device)
        results["Opt2"], times["Opt2"] = l2, t2

        # 3. V2 + V1SE crop 2.7 (Hard Vote)
        results["Opt3"] = "Real" if (l1 == "Real" and l2 == "Real") else "Spoof"
        times["Opt3"] = t1 + t2

        # 6. MiniFASNetV2 crop 4.0
        l6, t6 = get_label_single_timed(self.m_v2, crop_40, self.transform, self.device)
        results["Opt6"], times["Opt6"] = l6, t6

        # 7. MiniFASNetV1SE crop 4.0
        l7, t7 = get_label_single_timed(self.m_v1se, crop_40, self.transform, self.device)
        results["Opt7"], times["Opt7"] = l7, t7

        # 4. V2 + V1SE crop 4.0 (Hard Vote)
        results["Opt4"] = "Real" if (l6 == "Real" and l7 == "Real") else "Spoof"
        times["Opt4"] = t6 + t7

        # 8. MiniFASNetV2 full
        l8, t8 = get_label_single_timed(self.m_v2, crop_full, self.transform, self.device)
        results["Opt8"], times["Opt8"] = l8, t8

        # 9. MiniFASNetV1SE full
        l9, t9 = get_label_single_timed(self.m_v1se, crop_full, self.transform, self.device)
        results["Opt9"], times["Opt9"] = l9, t9

        # 10. V2 + V1SE full (Hard Vote)
        results["Opt10"] = "Real" if (l8 == "Real" and l9 == "Real") else "Spoof"
        times["Opt10"] = t8 + t9

        # Advanced Fusions
        # 5. V2+V1SE (4.0 and Full)
        results["Opt5"] = "Real" if (results["Opt4"] == "Real" and results["Opt10"] == "Real") else "Spoof"
        times["Opt5"] = times["Opt4"] + times["Opt10"]

        # 11. V2 (4.0 and full)
        results["Opt11"] = "Real" if (l6 == "Real" and l8 == "Real") else "Spoof"
        times["Opt11"] = t6 + t8

        # 12. V1SE (4.0 and full)
        results["Opt12"] = "Real" if (l7 == "Real" and l9 == "Real") else "Spoof"
        times["Opt12"] = t7 + t9

        total_runtime_all = (time.time() - t_start) * 1000
        return results, times, total_runtime_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models")
    args = parser.parse_args()

    predictor = AnalysisPredictor(args.model_dir)

    files = (
        sorted(os.path.join(args.data_root, f)
               for f in os.listdir(args.data_root)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if os.path.isdir(args.data_root) else [args.data_root]
    )

    opts_list = ["Opt1", "Opt2", "Opt3", "Opt4", "Opt5", "Opt6", "Opt7", "Opt8", "Opt9", "Opt10", "Opt11", "Opt12"]
    stats = {opt: {"Real": 0, "Spoof": 0, "total_model_time": 0} for opt in opts_list}
    noface = 0
    total_session_time = 0

    header = f"{'Filename':<30} | {'TotalRT':<7} | " + " | ".join([f"{o:<5}" for o in opts_list])
    print(header)
    print("-" * len(header))

    processed_count = 0
    for f in files:
        res, times, rt = predictor.predict_detailed(f)
        fname = os.path.basename(f)
        if res is None: continue
        if not res:
            noface += 1
            continue
        
        processed_count += 1
        total_session_time += rt
        
        row = f"{fname:<30} | {rt:>7.1f} |"
        for o in opts_list:
            row += f" {res[o]:<5} |"
            stats[o]["total_model_time"] += times[o]
            stats[o][res[o]] += 1
        print(row.rstrip("|"))

    print("\n" + "="*25 + " SUMMARY (PURE MODEL INF TIME) " + "="*25)
    print(f"{'Option':<45} | Real  | Spoof | Avg Model Time")
    print("-" * 90)
    labels_map = {
        "Opt1": "1. MiniFASNetV2 crop 2.7",
        "Opt2": "2. MiniFASNetV1SE crop 2.7",
        "Opt3": "3. V2 + V1SE crop 2.7 (Hard)",
        "Opt4": "4. V2 + V1SE crop 4.0 (Hard)",
        "Opt5": "5. V2+V1SE (4.0 & Full) (AND)",
        "Opt6": "6. MiniFASNetV2 crop 4.0",
        "Opt7": "7. MiniFASNetV1SE crop 4.0",
        "Opt8": "8. MiniFASNetV2 full",
        "Opt9": "9. MiniFASNetV1SE full",
        "Opt10": "10. V2 + V1SE full (Hard)",
        "Opt11": "11. V2 (4.0 & full) (AND)",
        "Opt12": "12. V1SE (4.0 & full) (AND)",
    }
    
    for opt in opts_list:
        r = stats[opt]["Real"]
        s = stats[opt]["Spoof"]
        avg = stats[opt]["total_model_time"] / processed_count if processed_count > 0 else 0
        print(f"{labels_map[opt]:<45} | {r:<5} | {s:<5} | {avg:>10.2f} ms")
    
    overall_avg = total_session_time / processed_count if processed_count > 0 else 0
    print(f"\nAverage Session Runtime (total per image): {overall_avg:.2f} ms")
    print(f"No face detected in {noface} files.")
    print("=" * 89)
