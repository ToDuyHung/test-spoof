import os
import sys
import math
import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Internalized Utilities (from Silent-Face-Anti-Spoofing src)
# ---------------------------------------------------------------------------

class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y
        left_top_x = max(0, center_x - new_width/2)
        left_top_y = max(0, center_y - new_height/2)
        right_bottom_x = min(src_w-1, center_x + new_width/2)
        right_bottom_y = min(src_h-1, center_y + new_height/2)
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))
        src_h, src_w, _ = np.shape(org_img)
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, scale)
        img = org_img[y1:y2+1, x1:x2+1]
        return cv2.resize(img, (out_w, out_h))


def parse_model_name(model_name):
    # Matches: "2.7_80x80_MiniFASNetV2.mlpackage" or "org_192x192_MiniFASNetV1.mlpackage"
    info = model_name.split('.mlpackage')[0].split('_')
    h_w = info[-2].split('x')
    h_input, w_input = int(h_w[0]), int(h_w[1])
    scale = None if info[0] == "org" else float(info[0])
    return h_input, w_input, scale


# ---------------------------------------------------------------------------
# RetinaFace detector (CoreML Standalone)
# ---------------------------------------------------------------------------

class RetinaFaceDetector:
    def __init__(self, coreml_path):
        import coremltools as ct
        if not os.path.exists(coreml_path):
            raise FileNotFoundError(f"RetinaFace model not found at {coreml_path}")
        
        print(f"  [Detection] Loading CoreML model: {coreml_path}")
        self._ct_model = ct.models.MLModel(coreml_path)
        
        # Metadata parsing
        meta = self._ct_model.user_defined_metadata
        if "input_hw" in meta:
            h, w = map(int, meta["input_hw"].split(","))
            self._input_h, self._input_w = h, w
        else:
            self._input_h, self._input_w = 640, 640

    def get_bbox(self, img):
        height, width = img.shape[:2]
        # Fixed size resize as traced in convert.py
        det_img = cv2.resize(img, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        
        # Preprocessing: Mean-subtracted BGR (104, 117, 123)
        det_f = det_img.astype(np.float32)
        det_f[:, :, 0] -= 104.0
        det_f[:, :, 1] -= 117.0
        det_f[:, :, 2] -= 123.0
        blob = det_f.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        result = self._ct_model.predict({"data": blob})
        dets = result.get("detections") or list(result.values())[0]
        
        if hasattr(dets, 'numpy'):
            dets = dets.numpy()
        dets = np.array(dets) # (max_det, 5) [x1, y1, x2, y2, score]

        if dets.size == 0 or dets[0, 4] < 0.3: # Threshold
            return None
        
        x1n, y1n, x2n, y2n = dets[0, :4]
        x1 = int(round(x1n * width))
        y1 = int(round(y1n * height))
        x2 = int(round(x2n * width))
        y2 = int(round(y2n * height))
        
        return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


# ---------------------------------------------------------------------------
# Anti-spoof CoreML model wrapper
# ---------------------------------------------------------------------------

class AntiSpoofCoreMLModel:
    def __init__(self, mlpackage_path):
        import coremltools as ct
        self.mlmodel = ct.models.MLModel(mlpackage_path)
        meta = self.mlmodel.user_defined_metadata

        # CoreML metadata or filename parsing
        self.name = os.path.basename(mlpackage_path)
        try:
            self.scale   = float(meta["scale"]) if meta.get("scale") != "None" else None
            self.h_input = int(meta["h_input"])
            self.w_input = int(meta["w_input"])
        except:
            # Fallback to filename parsing if metadata is missing (older versions)
            h, w, s = parse_model_name(self.name)
            self.h_input, self.w_input, self.scale = h, w, s

        print(f"  [AntiSpoof] Loaded {self.name}  (scale={self.scale}, {self.h_input}x{self.w_input})")

    def predict(self, img_array):
        # Resize to model input
        if img_array.shape[0] != self.h_input or img_array.shape[1] != self.w_input:
            img_array = cv2.resize(img_array, (self.w_input, self.h_input))

        # CoreML model expects [0, 255] floats (matching original PyTorch ToTensor)
        inp = img_array.astype(np.float32)
        inp = inp.transpose(2, 0, 1)[np.newaxis]

        result = self.mlmodel.predict({"data": inp})
        logits = list(result.values())[0].squeeze()

        # Softmax
        e = np.exp(logits - np.max(logits))
        return e / (e.sum() + 1e-6)


# ---------------------------------------------------------------------------
# standalone CoreML predictor
# ---------------------------------------------------------------------------

class StandaloneCoreMLPredictor:
    def __init__(self, models_dir):
        # Load detection 
        retina_path = os.path.join(models_dir, "RetinaFace.mlpackage")
        if not os.path.exists(retina_path):
            raise FileNotFoundError(f"Missing face detector: {retina_path}")
        self.detector = RetinaFaceDetector(retina_path)

        # Load MiniFASNet models
        self.image_cropper = CropImage()
        self.models = []
        mlpackages = sorted([
            f for f in os.listdir(models_dir)
            if f.endswith('.mlpackage') and 'MiniFASNet' in f
        ])
        
        for pkg in mlpackages:
            self.models.append(AntiSpoofCoreMLModel(os.path.join(models_dir, pkg)))
        
        if not self.models:
            print(f"[WARN] No anti-spoof MiniFASNet models found in {models_dir}")
        else:
            print(f"  Loaded {len(self.models)} anti-spoof model(s)")

    def predict(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None

        # 1. Detect face
        bbox = self.detector.get_bbox(img_bgr)
        if bbox is None:
            return []

        # 2. Ensemble prediction
        prediction = np.zeros(3)
        for m in self.models:
            img_cropped = self.image_cropper.crop(
                img_bgr, bbox, scale=m.scale, 
                out_w=m.w_input, out_h=m.h_input, crop=(m.scale is not None)
            )
            prediction += m.predict(img_cropped)

        # 3. Final aggregation logic (average softmax)
        label_idx = int(np.argmax(prediction))
        label = "Real" if label_idx == 1 else "Spoof"
        score = float(prediction[label_idx]) / max(1, len(self.models))

        return [{
            "box_xywh": bbox,
            "label": label,
            "score": score,
            "prediction": prediction / max(1, len(self.models)),
        }]


def draw_and_save(img_path, results, output_dir="./output_coreml"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None: return

    for r in results:
        x, y, w, h = r["box_xywh"]
        label, score = r["label"], r["score"]
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {score:.4f}"
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Image or folder")
    parser.add_argument("--models_dir", type=str, default="./models", help="Dir with .mlpackage")
    parser.add_argument("--output_dir", type=str, default="./output_coreml")
    args = parser.parse_args()

    predictor = StandaloneCoreMLPredictor(args.models_dir)

    if os.path.isdir(args.data_root):
        files = sorted([os.path.join(args.data_root, f) for f in os.listdir(args.data_root)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    else:
        files = [args.data_root]

    for f in files:
        res = predictor.predict(f)
        print(f"\nFile: {os.path.basename(f)}")
        if not res:
            print("  No face detected.")
        else:
            for i, r in enumerate(res):
                print(f"  Result: {r['label']} ({r['score']:.4f})")
                print(f"  Box: {r['box_xywh']}")
            draw_and_save(f, res, args.output_dir)
