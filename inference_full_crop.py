import os
import sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F

# Add parent directory to sys.path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV2
from src.data_io import transform as trans
from facenet_pytorch import MTCNN

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

class FullCropPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)
        
        # Consistent with training script
        self.model = MultiFTNet(num_classes=3, img_channel=3, embedding_size=128, conv6_kernel=(5, 5)).to(self.device)
        self.model.model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5), num_classes=3, img_channel=3).to(self.device)
        
        print(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        self.transform = trans.Compose([trans.ToTensor()])

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # face presence check
        boxes, _, _ = self.detector.detect(img_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return []

        # 1. Apply full-width square crop
        crop, b = full_width_square_crop(img_bgr)
        
        # 2. Resize to model input size (80, 80)
        crop_resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        
        # 3. Preprocess
        input_tensor = self.transform(crop_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        score = probs[1] # Probability of Real
        label = "Real" if probs[1] > probs[0] else "Spoof"
        
        return [{
            'label': label,
            'score': score,
            'probs': probs,
            'box_xywh': b
        }]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with full-width square crop strategy.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to input image or folder.")
    parser.add_argument("--model", type=str, default="checkpoint/finetuned_full_crop.pth", help="Path to model checkpoint.")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
        
    predictor = FullCropPredictor(args.model)
    
    files = (
        sorted(os.path.join(args.data_root, f)
               for f in os.listdir(args.data_root)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if os.path.isdir(args.data_root) else [args.data_root]
    )

    spoof_total = real_total = noface_total = 0

    for f in files:
        res = predictor.predict(f)
        if res is None:
            print(f"File: {os.path.basename(f)} - Failed to load.")
            continue
        if not res:
            noface_total += 1
            print(f"File: {os.path.basename(f)} - No face detected.")
            continue
            
        for r in res:
            if r["label"] == "Spoof":
                spoof_total += 1
            else:
                real_total += 1
            print(f"File: {os.path.basename(f)} - Label={r['label']} | Real Score={r['score']:.4f}")

    print("\n----- Summary -----")
    print(f"Total files processed: {len(files)}")
    print(f"  Spoof: {spoof_total}")
    print(f"  Real: {real_total}")
    print(f"  No face detected: {noface_total}")
