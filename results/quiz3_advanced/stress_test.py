"""Controlled image-degradation test; these are not extra real photographs.

Run from the repository root:
    python results/quiz3_advanced/stress_test.py
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Week01" / "Code"))
from quiz3_perspective import detect_corners  # noqa: E402

INPUT = PROJECT / "Week01" / "Data" / "Quiz3"
OUTPUT = Path(__file__).resolve().with_name("stress.csv")


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))
    return cv2.resize(image, (round(width * scale), round(height * scale)))


rows = []
for path in sorted(INPUT.glob("document_0?.jpg")):
    original = read_image(path)
    conditions = [("brightness", str(factor),
                   cv2.convertScaleAbs(original, alpha=factor, beta=0))
                  for factor in (1.0, 0.7, 0.5, 0.3, 0.15)]
    conditions += [("gaussian_blur_kernel", str(kernel),
                    cv2.GaussianBlur(original, (kernel, kernel), 0))
                   for kernel in (3, 7, 15, 31)]
    for condition, value, image in conditions:
        try:
            _, method = detect_corners(image)
            status = "found"
        except ValueError:
            status, method = "not_found", ""
        rows.append({"image": path.name, "condition": condition,
                     "value": value, "status": status, "method": method})

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} controlled-test rows to {OUTPUT}")
