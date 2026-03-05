import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

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
    
    # Square padding logic
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

def process_folder(input_dir, output_dir, detector, margin=2.7):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for f in tqdm(files, desc=f"Processing {os.path.basename(input_dir)}"):
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _, landmarks = detector.detect(img_rgb, landmarks=True)
        
        if boxes is not None and len(boxes) > 0:
            # Take the largest face if multiple detected
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            
            pts = landmarks[idx]
            crop = crop_from_landmarks(img, pts, margin=margin)
            
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (80, 80))
                out_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.jpg")
                cv2.imwrite(out_path, crop_resized)
                count += 1
                
    print(f"Saved {count} crops to {output_dir}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=False, device=device, post_process=False)
    
    # Process Real (assuming run from finetune/ folder)
    process_folder("../real", "data/1_real", detector, margin=2.7)
    
    # Process Spoof
    process_folder("../spoof", "data/0_spoof", detector, margin=2.7)
    
    # Optional: also process specific hard cases if they are in different folders
    # or just rely on the spoof/ folder containing them.
