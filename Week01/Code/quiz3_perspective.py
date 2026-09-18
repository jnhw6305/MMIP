"""Quiz 3: automatically detect corners and rectify document*.jpg/png in batches.

Run from repository root: python Week01/Code/quiz3_perspective.py
Input: Week01/Data/Quiz3/document*.jpg/png (override with --input-dir)
Outputs: results/quiz3_auto/*_corners.jpg, *_corrected.jpg, summary.csv
The program never asks for mouse clicks or uses fixed corner coordinates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("無法讀取影像")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"無法儲存 {path.name}")
    data.tofile(path)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    top = points[np.argsort(points[:, 1])[:2]]
    bottom = points[np.argsort(points[:, 1])[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.float32([top[0], top[1], bottom[1], bottom[0]])


def valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
    if not np.isfinite(quad).all():
        return False
    if (quad[:, 0] < 0.005 * width).any() or (quad[:, 0] >= 0.995 * width).any():
        return False
    if (quad[:, 1] < 0.005 * height).any() or (quad[:, 1] >= 0.995 * height).any():
        return False
    polygon = np.round(quad).astype(np.int32)
    area = cv2.contourArea(polygon)
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    return bool(0.12 * width * height <= area <= 0.65 * width * height
                and cv2.isContourConvex(polygon)
                and sides.min() >= 0.12 * min(width, height))


def contour_quad(edges: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """First try the usual large, convex document-outline contour."""
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 0.12 * width * height:
            break
        perimeter = cv2.arcLength(contour, True)
        for factor in (0.015, 0.025, 0.03, 0.04, 0.06):
            polygon = cv2.approxPolyDP(contour, factor * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            try:
                quad = order_corners(polygon)
            except ValueError:
                continue
            if valid_quad(quad, width, height):
                return quad
    return None


def colored_book_quad(image: np.ndarray, edges: np.ndarray) -> np.ndarray | None:
    """For a two-tone book, combine its colored top and long lower edge."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 35, 30), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70,
                            minLineLength=max(70, round(0.20 * width)),
                            maxLineGap=round(0.03 * width))
    if lines is None:
        return None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 0.05 * width * height:
            break
        x, y, box_width, box_height = cv2.boundingRect(contour)
        margin = 0.01 * min(width, height)
        if (x <= margin or y <= margin or x + box_width >= width - margin
                or y + box_height >= height - margin):
            continue  # A colored background touching the photo border.
        hull = cv2.convexHull(contour).reshape(-1, 2)
        middle_x = x + box_width / 2
        left = hull[hull[:, 0] < middle_x]
        right = hull[hull[:, 0] >= middle_x]
        if not len(left) or not len(right):
            continue
        top_left = left[np.argmin(left[:, 1])]
        top_right = right[np.argmin(right[:, 1])]
        color_bottom = y + box_height
        target_y = color_bottom + 0.22 * height
        candidates = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            middle_y = (y1 + y2) / 2
            if (abs(angle) > 35 or middle_y < color_bottom + 0.08 * height
                    or middle_y > color_bottom + 0.45 * height
                    or x2 < x + 0.1 * box_width
                    or x1 > x + 0.9 * box_width):
                continue
            length = np.hypot(x2 - x1, y2 - y1)
            score = length - 1.1 * abs(middle_y - target_y)
            candidates.append((score, (x1, y1, x2, y2), angle))
        if not candidates:
            continue
        _, (x1, y1, x2, y2), main_angle = max(candidates,
                                              key=lambda item: item[0])
        endpoints = [(x1, y1), (x2, y2)]
        line_length = np.hypot(x2 - x1, y2 - y1)
        for _, (a, b, c, d), angle in candidates:
            if abs(angle - main_angle) > 8:
                continue
            distances = [abs((y2 - y1) * (px - x1) - (x2 - x1) * (py - y1))
                         / line_length for px, py in ((a, b), (c, d))]
            if max(distances) < 0.025 * height:
                endpoints.extend(((a, b), (c, d)))
        bottom_left = min(endpoints, key=lambda point: point[0])
        bottom_right = max(endpoints, key=lambda point: point[0])
        quad = np.float32([top_left, top_right,
                           bottom_right, bottom_left])
        if valid_quad(quad, width, height):
            return quad
    return None


def blue_cover_quad(image: np.ndarray) -> np.ndarray | None:
    """Find a blue/cyan cover when a complete outer edge is not visible."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for saturation in (18, 35, 70, 100):
        mask = cv2.inRange(hsv, (70, saturation, 20), (135, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((25, 25), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < 0.12 * width * height:
                break
            hull = cv2.convexHull(contour)
            polygon = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True),
                                       True)
            if len(polygon) != 4:
                continue
            quad = order_corners(polygon)
            if valid_quad(quad, width, height):
                return quad
    return None


def detect_corners(image: np.ndarray) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale)
    small_height, small_width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
    quad = contour_quad(edges, small_width, small_height)
    if quad is not None:
        return quad / scale, "四邊形輪廓"
    quad = colored_book_quad(small, edges)
    if quad is not None:
        return quad / scale, "彩色封面與底邊"
    softer_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 70)
    quad = contour_quad(softer_edges, small_width, small_height)
    if quad is not None:
        return quad / scale, "低門檻四邊形輪廓"
    quad = blue_cover_quad(small)
    if quad is not None:
        return quad / scale, "藍色封面區域"
    raise ValueError("無法找到可信的四角；可能是邊緣不清、物件被裁切或背景干擾")


def rectify(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if min(width, height) < 100:
        raise ValueError("校正後尺寸過小")
    scale = min(1.0, 1800 / max(width, height))
    width, height = round(width * scale), round(height * scale)
    target = np.float32([[0, 0], [width - 1, 0],
                         [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image, matrix, (width, height))


def marked_image(image: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    marked = image.copy()
    if quad is None:
        cv2.putText(marked, "CORNERS NOT FOUND", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return marked
    polygon = np.round(quad).astype(np.int32)
    thickness = max(2, round(min(image.shape[:2]) / 300))
    cv2.polylines(marked, [polygon], True, (0, 0, 255), thickness)
    for number, (x, y) in enumerate(polygon, 1):
        cv2.circle(marked, (int(x), int(y)), thickness * 2,
                   (0, 255, 255), -1)
        cv2.putText(marked, str(number), (int(x + 8), int(y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return marked


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=project / "Week01" / "Data" / "Quiz3")
    parser.add_argument("--pattern", default="document*")
    parser.add_argument("--output-dir", type=Path,
                        default=project / "results" / "quiz3_auto")
    args = parser.parse_args()
    files = sorted(path for path in args.input_dir.glob(args.pattern)
                   if path.is_file() and path.suffix.lower() in SUFFIXES
                   and not path.stem.lower().endswith(("_corners", "_corrected")))
    if not files:
        raise FileNotFoundError(f"找不到符合 {args.pattern} 的影像：{args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        row = {"image": path.name, "status": "", "method": "",
               "note": "", "corners_xy": ""}
        try:
            image = read_image(path)
            quad, method = detect_corners(image)
            save_image(args.output_dir / f"{path.stem}_corners.jpg",
                       marked_image(image, quad))
            save_image(args.output_dir / f"{path.stem}_corrected.jpg",
                       rectify(image, quad))
            row.update(status="成功", method=method,
                       note="請目視確認四角與校正效果",
                       corners_xy="; ".join(f"{x:.1f},{y:.1f}" for x, y in quad))
        except (ValueError, cv2.error) as error:
            row.update(status="失敗", note=str(error).splitlines()[0])
            try:
                image = read_image(path)
                save_image(args.output_dir / f"{path.stem}_corners.jpg",
                           marked_image(image, None))
            except (ValueError, cv2.error):
                pass
        rows.append(row)
        print(f"{path.name}: {row['status']} - {row['method'] or row['note']}")
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"結果資料夾：{args.output_dir}")
    print(f"成功 {sum(row['status'] == '成功' for row in rows)} / {len(rows)} 張")


if __name__ == "__main__":
    main()
