import os
import sys
import math
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.utility import get_kernel, parse_model_name
from src.generate_patches import CropImage
from src.data_io import transform as trans


DETECTION_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Silent-Face-Anti-Spoofing", "resources", "detection_model"
)


class RetinaFaceDetector:
    """Identical to Detection class in anti_spoof_predict.py."""
    def __init__(self):
        caffemodel = os.path.join(DETECTION_MODEL_DIR, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(DETECTION_MODEL_DIR, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        """Returns [x, y, w, h] as integers — same as official get_bbox."""
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))),
                             interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left   = out[max_conf_index, 3] * width
        top    = out[max_conf_index, 4] * height
        right  = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox


MODEL_MAPPING = {
    'MiniFASNetV1':   MiniFASNetV1,
    'MiniFASNetV2':   MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}


def load_model(model_path, device):
    """Load MiniFASNet model by parsing its filename for h/w/type."""
    model_name = os.path.basename(model_path)
    h_input, w_input, model_type, scale = parse_model_name(model_name)

    kernel_size = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size, num_classes=3)

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, h_input, w_input, scale


class AntiSpoofPredictor:
    def __init__(self, model_dir):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Official RetinaFace detector (same as anti_spoof_predict.py)
        self.detector = RetinaFaceDetector()

        # Official Cropper and Transform
        self.image_cropper = CropImage()
        self.test_transform = trans.Compose([trans.ToTensor()])

        # Load all models from model_dir (same as test.py)
        self.models = []
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            print(f"[WARN] No .pth files found in {model_dir}")
        for model_name in model_files:
            model_path = os.path.join(model_dir, model_name)
            print(f"  Loading model: {model_name}")
            model, h_input, w_input, scale = load_model(model_path, self.device)
            self.models.append({
                "model": model,
                "h_input": h_input,
                "w_input": w_input,
                "scale": scale,
                "name": model_name,
            })
        print(f"  Loaded {len(self.models)} model(s) from {model_dir}")

    def crop_face(self, img_bgr, bbox_xywh, h_input, w_input, scale, save_path=None):
        """Crop face patch using official CropImage logic.
        
        bbox_xywh: [x, y, w, h] as integers — from RetinaFace get_bbox (matches official format).
        """
        param = {
            "org_img": img_bgr,
            "bbox": bbox_xywh,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        img_cropped = self.image_cropper.crop(**param)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_cropped)

        return img_cropped

    def predict(self, img_path):
        """Run multi-model ensemble prediction. Matches test.py logic exactly."""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load {img_path}")
            return None

        # 1. Detect face with RetinaFace (same as official — returns [x, y, w, h])
        bbox = self.detector.get_bbox(img_bgr)

        if bbox is None:
            return []

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 2. Ensemble: sum softmax predictions from all models (same as test.py)
        prediction = np.zeros(3)

        for m in self.models:
            model_short = m['name'].replace('_MiniFASNetV2', '').replace('.pth', '')
            crop_save_path = os.path.join(
                "./output/crops",
                f"{base_name}_{model_short}.jpg"
            )
            img_cropped = self.crop_face(
                img_bgr, bbox,
                h_input=m['h_input'], w_input=m['w_input'], scale=m['scale'],
                save_path=crop_save_path
            )

            inp = self.test_transform(img_cropped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = m['model'](inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            prediction += probs

        # 3. Label via argmax — label==1 is Real, else Spoof (same as test.py)
        label_idx = int(np.argmax(prediction))
        label = "Real" if label_idx == 1 else "Spoof"
        score = float(prediction[label_idx]) / 2  # matches test.py exactly: prediction[0][label]/2

        # Return single result (RetinaFace returns only 1 face — highest confidence)
        return [{
            "box_xywh": bbox,
            "label": label,
            "score": score,
            "prediction": prediction,
        }]

    def export_coreml(self, output_path="MiniFASNetV2.mlpackage"):
        import coremltools as ct
        if not self.models:
            print("No models loaded.")
            return
        m = self.models[0]
        print(f"Exporting {m['name']} to {output_path}...")
        dummy_input = torch.rand(1, 3, m['h_input'], m['w_input']).to(self.device)
        traced_model = torch.jit.trace(m['model'], dummy_input)
        model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input_1", shape=dummy_input.shape,
                                 scale=1/255.0, color_layout="RGB")],
        )
        model.save(output_path)
        print("Export success!")


def draw_and_save(img_path, results, output_dir="./output"):
    """Draw bounding boxes + label/score on image and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        return

    for r in results:
        x, y, w, h = r["box_xywh"]
        x2, y2 = x + w, y + h
        label = r["label"]
        score = r["score"]

        color = (0, 255, 0) if label == "Real" else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x2, y2), color, 2)

        text = f"{label}: {score:.4f}"
        font_scale = 0.5 * img.shape[0] / 1024
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - baseline - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), thickness)

    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to image or folder")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models",
                        help="Directory containing .pth model files")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save annotated images")
    parser.add_argument("--export_coreml", action="store_true", help="Export to CoreML")
    args = parser.parse_args()

    predictor = AntiSpoofPredictor(args.model_dir)

    if args.export_coreml:
        predictor.export_coreml("MiniFASNetV2.mlpackage")
        sys.exit(0)

    if args.data_root:
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
                continue
            for i, r in enumerate(res):
                print(f"  Face {i}: Label={r['label']} | Score={r['score']:.4f}")
                print(f"    Summed prediction [Spoof, Real, Other]: {r['prediction']}")
            draw_and_save(f, res, output_dir=args.output_dir)
    else:
        print("Please provide --data_root to run inference, or --export_coreml")
