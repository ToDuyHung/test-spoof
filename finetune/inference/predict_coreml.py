import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

# coremltools for inference
try:
    import coremltools as ct
except ImportError:
    print("Error: coremltools not installed. Run 'pip install coremltools'.")
    sys.exit(1)

# Add parent directory to sys.path to import src if needed 
# (although for CoreML we mostly need MTCNN and cropping)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def add_face_margin(x, y, w, h, margin=0.5):
    xm = int(w * margin / 2)
    ym = int(h * margin / 2)
    return x - xm, x + w + xm, y - ym, y + h + ym

def crop_from_landmarks(img_bgr, pts, margin=2.7):
    x_list, y_list = pts[:, 0], pts[:, 1]
    x = round(float(min(x_list)))
    y = round(float(min(y_list)))
    w = round(float(max(x_list))) - x
    h = round(float(max(y_list))) - y
    side = max(w, h)
    
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
    H, W = img_bgr.shape[:2]
    
    x1_pad = 0 if x1 >= 0 else -x1
    y1_pad = 0 if y1 >= 0 else -y1
    x2_pad = 0 if x2 <= W else x2 - W
    y2_pad = 0 if y2 <= H else y2 - H
    
    x1_real, y1_real = max(0, x1), max(0, y1)
    x2_real, y2_real = min(W, x2), min(H, y2)
    
    crop = img_bgr[y1_real:y2_real, x1_real:x2_real]
    if x1_pad > 0 or y1_pad > 0 or x2_pad > 0 or y2_pad > 0:
        crop = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return crop

class CoreMLPredictor:
    def __init__(self, model_path, margin=2.7):
        import torch # Needed for MTCNN device detection
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)
        self.margin = margin
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
            
        print(f"Loading CoreML model from {model_path}...")
        # Note: on Linux, model.predict might raise 'Core ML is only supported on macOS'
        self.model = ct.models.MLModel(model_path)

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, _, landmarks = self.detector.detect(img_rgb, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return []
            
        # Use landmarks of the best (first) detection, matching single_baseline.py
        pts = landmarks[0]
        crop = crop_from_landmarks(img_bgr, pts, margin=self.margin)
        crop = cv2.resize(crop, (80, 80))
        
        # Convert BGR crop to PIL RGB for CoreML (as defined in convert_to_coreml.py)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        try:
            # Inference
            # The input name was "image" in convert_to_coreml.py
            preds = self.model.predict({"image": pil_img})
            logits = preds["logits"] # Output name was "logits"
            
            # Post-process: Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            probs = probs.flatten()
            
            score = float(probs[1]) # Probability of Real
            label = "Real" if probs[1] > probs[0] else "Spoof"
            
            return [{
                'label': label,
                'score': score
            }]
        except Exception as e:
            print(f"CoreML Inference Error: {e}")
            if "macOS" in str(e):
                print("Tip: CoreML model.predict() usually requires running on a Mac.")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--model_path", type=str, default="../checkpoint/finetuned.mlpackage")
    parser.add_argument("--margin", type=float, default=2.7)
    args = parser.parse_args()
    
    if not os.path.isabs(args.model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(script_dir, args.model_path)

    predictor = CoreMLPredictor(args.model_path, margin=args.margin)
    
    if os.path.isdir(args.data_root):
        files = sorted([os.path.join(args.data_root, f) for f in os.listdir(args.data_root) 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    else:
        files = [args.data_root]
        
    spoof_total = 0
    real_total = 0
    noface_total = 0
    
    for f in files:
        res = predictor.predict(f)
        if res is None:
            print(f"File: {os.path.basename(f)} - Load failed.")
            continue
        if not res:
            noface_total += 1
            print(f"File: {os.path.basename(f)} - Prediction failed or no face.")
            continue
            
        for r in res:
            if r['label'] == 'Spoof':
                spoof_total += 1
            else:
                real_total += 1
            print(f"File: {os.path.basename(f)} - Label={r['label']} | Score={r['score']:.4f}.")
            
    print("\n----- Summary -----")
    print(f"Total files processed: {len(files)}")
    print(f"  Spoof: {spoof_total}")
    print(f"  Real: {real_total}")
    print(f"  No face detected: {noface_total}")
