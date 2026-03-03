"""
mtcnn_baseline.py
Uses MTCNN (facenet_pytorch) for face detection + landmark-based crop.

Crop logic is derived directly from the 5 MTCNN landmarks, making it
independant of the original RetinaFace-tuned CropImage parameters.
Output format is identical to single_baseline.py / pytorch_baseline.py.

Usage:
    python3 mtcnn_baseline.py --data_root <image_or_folder>
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


# ─────────────────── landmark-based crop (user-provided) ─────────────────────

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
    side = max(w, h)   # square side = larger dimension
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(max_w, x2);  y2 = min(max_h, y2)
    return img[y1:y2, x1:x2], [x1, y1, x2 - x1, y2 - y1]


# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, device):
    model_name = os.path.basename(model_path)
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size, num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k[7:] if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, h_input, w_input


class AntiSpoofPredictor:
    def __init__(self, model_dir, landmark_margin=3.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.landmark_margin = landmark_margin

        # MTCNN — we need landmarks, so keep_all=False returns best face
        self.detector = MTCNN(
            keep_all=False,
            device=self.device,
            post_process=False,
        )

        self.test_transform = trans.Compose([trans.ToTensor()])

        self.models = []
        model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))
        if not model_files:
            print(f"[WARN] No .pth files found in {model_dir}")
        for model_name in model_files:
            model_path = os.path.join(model_dir, model_name)
            print(f"  Loading model: {model_name}")
            model, h_input, w_input = load_model(model_path, self.device)
            self.models.append({
                "model": model, "h_input": h_input,
                "w_input": w_input, "name": model_name,
            })
        print(f"  Loaded {len(self.models)} model(s) from {model_dir}")

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load {img_path}")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.detector.detect(img_rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            # print(None)
            return []

        # Use landmarks of the best (first) detection to produce a square crop
        pts         = landmarks[0]                         # (5, 2)
        face_crop, bbox = crop_from_5landmarks(
            img_bgr, pts, margin=self.landmark_margin
        )

        # print(bbox)   # print [x, y, w, h] like pytorch_baseline.py

        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        prediction = np.zeros(3)

        for m in self.models:
            # Resize the square crop directly to the model's input size
            img_resized = cv2.resize(
                face_crop, (m["w_input"], m["h_input"]),
                interpolation=cv2.INTER_LINEAR
            )

            # Optionally save the crop
            crop_save_path = os.path.join(
                "./output/crops",
                f"{base_name}_{m['name'].replace('.pth', '')}.jpg"
            )
            os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)
            cv2.imwrite(crop_save_path, img_resized)

            inp = self.test_transform(img_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = m["model"](inp)
                probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            prediction += probs

        label_idx = int(np.argmax(prediction))
        label     = "Real" if label_idx == 1 else "Spoof"
        score     = float(prediction[label_idx]) / 2

        return [{
            "box_xywh":   bbox,
            "label":      label,
            "score":      score,
            "prediction": prediction,
        }]


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
    # print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Path to image or folder")
    parser.add_argument("--model_dir",  type=str,
                        default="./resources/anti_spoof_models",
                        help="Directory containing .pth anti-spoof model files")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--margin",     type=float, default=3.5,
                        help="Margin multiplier for landmark-based crop (default 2.0)")
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

        # counters for summary
        spoof_total = 0
        real_total = 0
        noface_total = 0

        for f in files:
            res = predictor.predict(f)
            # print(f"\nFile: {os.path.basename(f)}")
            if not res:
                noface_total += 1
                print(f"File: {os.path.basename(f)} - No face detected.")
                continue
            for i, r in enumerate(res):
                if r['label'] == 'Spoof':
                    spoof_total += 1
                elif r['label'] == 'Real':
                    real_total += 1
                print(f"File: {os.path.basename(f)} - Label={r['label']} | Score={r['score']:.4f}.")
                # print(f"  Face {i}: Label={r['label']} | Score={r['score']:.4f}")
                # print(f"    Summed prediction [Spoof, Real, Other]: {r['prediction']}")
            draw_and_save(f, res, output_dir=args.output_dir)

        # summary output
        print("\n----- Summary -----")
        print(f"Total files processed: {len(files)}")
        print(f"  Spoof: {spoof_total}")
        print(f"  Real: {real_total}")
        print(f"  No face detected: {noface_total}")
    else:
        print("Please provide --data_root to run inference.")
