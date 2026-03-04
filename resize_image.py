"""
resize_image.py — Resize an image to H=1920, W=2560 and save.

Usage:
    python resize_image.py --image <path>
    python resize_image.py --image spoof/d0e08f853069be37e778.jpg
    python resize_image.py --image spoof/d0e08f853069be37e778.jpg --output resized.jpg
"""

import argparse
import cv2
from pathlib import Path

TARGET_H = 1920
TARGET_W = 2560


def main(image_path: str, output_path: str | None = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Original : {w}×{h}")

    resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    print(f"Resized  : {TARGET_W}×{TARGET_H}")

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_resized{p.suffix}")

    cv2.imwrite(output_path, resized)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True,  help="Input image path")
    parser.add_argument("--output", default=None,   help="Output path (default: <name>_resized.jpg)")
    args = parser.parse_args()
    main(args.image, args.output)
