"""Quiz 4 進階：比較亮度、重疊範圍與 CLAHE 對影像拼接的影響。

從儲存庫根目錄執行：python Week01/Code/quiz4_stitching.py
每次結果會存入 results/quiz4_two_pairs/experiments_時間戳/，不覆蓋舊結果。
"""

from pathlib import Path
import argparse
import csv
from datetime import datetime

import cv2
import numpy as np


PAIRS = (
    ("success", "left.jpg", "right.jpg"),
    ("failure", "failure_view1.jpg", "failure_view2.jpg"),
)


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到照片：{path}")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"照片無法讀取：{path}")
    scale = min(1.0, 1600 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"無法儲存：{path}")
    encoded.tofile(path)


def comparison(left, right):
    """即使拼接失敗，也保留並排影像供作業報告比較。"""
    target_height = 800
    parts = []
    for image in (left, right):
        target_width = round(image.shape[1] * target_height / image.shape[0])
        parts.append(cv2.resize(image, (target_width, target_height)))
    return cv2.hconcat(parts)


def apply_clahe(image):
    """只增強亮度通道，避免直接對 B、G、R 各通道處理而改變色相。"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def experiment_cases(left, right):
    """固定原圖，僅改動右圖；右圖左側是兩張室內照片的重疊側。"""
    yield "original", "原圖", left, right
    for factor in (0.70, 0.40, 0.20):
        darker = np.clip(right.astype(np.float32) * factor, 0, 255)
        yield f"brightness_{int(factor * 100):02d}", f"右圖亮度 {factor:.0%}", left, darker.astype(np.uint8)
    for fraction in (0.15, 0.30, 0.45):
        cut = round(right.shape[1] * fraction)
        cropped = np.ascontiguousarray(right[:, cut:])
        yield f"overlap_crop_{int(fraction * 100):02d}", f"裁掉右圖左側 {fraction:.0%}", left, cropped


def sift_homography(left, right, prefix, results_dir):
    sift = cv2.SIFT_create()
    kp_left, desc_left = sift.detectAndCompute(
        cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None
    )
    kp_right, desc_right = sift.detectAndCompute(
        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None
    )
    stats = {
        "left_keypoints": len(kp_left),
        "right_keypoints": len(kp_right),
        "good_matches": 0,
        "inliers": 0,
    }
    if desc_left is None or desc_right is None:
        return None, stats, "找不到足夠 SIFT 特徵"

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_right, desc_left, k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    stats["good_matches"] = len(good)
    if good:
        visual = cv2.drawMatches(
            right, kp_right, left, kp_left, good[:60], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        save_image(results_dir / f"{prefix}_matches.jpg", visual)
    if len(good) < 10:
        return None, stats, "可靠匹配點少於 10 個"

    right_points = np.float32(
        [kp_right[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    left_points = np.float32(
        [kp_left[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        right_points, left_points, cv2.RANSAC, 5.0
    )
    if homography is None or mask is None:
        return None, stats, "無法計算透視變換"

    stats["inliers"] = int(mask.sum())
    inlier_ratio = stats["inliers"] / len(good)
    if stats["inliers"] < 12 or inlier_ratio < 0.25:
        return None, stats, "幾何驗證通過的匹配點太少"
    return homography, stats, "SIFT 幾何驗證通過"


def make_sift_panorama(left, right, homography):
    """用 SIFT 算出的變換拼接；異常畫布會被攔下。"""
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    left_corners = np.float32([
        [0, 0], [w_left, 0], [w_left, h_left], [0, h_left]
    ]).reshape(-1, 1, 2)
    right_corners = np.float32([
        [0, 0], [w_right, 0], [w_right, h_right], [0, h_right]
    ]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(right_corners, homography)
    if not np.isfinite(transformed).all():
        raise ValueError("透視變換產生無效座標")
    polygon = np.round(transformed.reshape(-1, 2)).astype(np.int32)
    if not cv2.isContourConvex(polygon):
        raise ValueError("變換後的照片四角交錯")

    corners = np.concatenate((left_corners, transformed))
    lower = np.floor(corners.min(axis=0).ravel()).astype(int)
    upper = np.ceil(corners.max(axis=0).ravel()).astype(int)
    canvas_width, canvas_height = (upper - lower).tolist()
    max_area = 4 * max(h_left * w_left, h_right * w_right)
    if (
        canvas_width <= 0 or canvas_height <= 0
        or canvas_width * canvas_height > max_area
    ):
        raise ValueError("拼接畫布過大或尺寸異常")

    translation = np.array([
        [1, 0, -lower[0]], [0, 1, -lower[1]], [0, 0, 1]
    ], dtype=np.float64)
    canvas_size = (canvas_width, canvas_height)
    left_warp = cv2.warpPerspective(left, translation, canvas_size)
    right_warp = cv2.warpPerspective(
        right, translation @ homography, canvas_size
    )
    left_mask = cv2.warpPerspective(
        np.full((h_left, w_left), 255, np.uint8), translation, canvas_size
    ) > 0
    right_mask = cv2.warpPerspective(
        np.full((h_right, w_right), 255, np.uint8),
        translation @ homography, canvas_size
    ) > 0

    panorama = np.zeros_like(left_warp)
    only_left = left_mask & ~right_mask
    only_right = right_mask & ~left_mask
    overlap = left_mask & right_mask
    panorama[only_left] = left_warp[only_left]
    panorama[only_right] = right_warp[only_right]
    panorama[overlap] = (
        (left_warp[overlap].astype(np.uint16)
         + right_warp[overlap].astype(np.uint16)) // 2
    ).astype(np.uint8)
    valid = left_mask | right_mask
    ys, xs = np.where(valid)
    return panorama[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def process_case(pair_name, case_name, condition, method, left, right, results_dir):
    label = pair_name if case_name == "original" and method == "raw" else f"{pair_name}_{case_name}_{method}"
    if method == "clahe":
        left, right = apply_clahe(left), apply_clahe(right)
    save_image(results_dir / f"{label}_comparison.jpg", comparison(left, right))

    cv2.setRNGSeed(0)
    homography, stats, sift_status = sift_homography(
        left, right, label, results_dir
    )
    sift_panorama_status = sift_status
    sift_ok = False
    if homography is not None:
        try:
            panorama = make_sift_panorama(left, right, homography)
            save_image(results_dir / f"{label}_sift_panorama.jpg", panorama)
            sift_panorama_status = "已產生 SIFT 拼接圖；仍須目視檢查接縫"
            sift_ok = True
        except ValueError as exc:
            sift_panorama_status = f"未產生 SIFT 拼接圖：{exc}"

    # OpenCV 內建 Stitcher 用來比較自製 SIFT 拼接的效果。
    # 它內部的特徵方法不一定是 SIFT，因此結果要分開標示。
    opencv_ok = False
    try:
        status, stitched = cv2.Stitcher_create(
            cv2.Stitcher_PANORAMA
        ).stitch([left, right])
        if status == cv2.Stitcher_OK and stitched is not None:
            save_image(results_dir / f"{label}_opencv_panorama.jpg", stitched)
            opencv_status = "成功"
            opencv_ok = True
        else:
            opencv_status = f"失敗，狀態碼 {status}"
    except cv2.error as exc:
        opencv_status = f"失敗：{str(exc).splitlines()[0]}"

    record = {
        "pair": pair_name,
        "case": case_name,
        "condition": condition,
        "preprocess": method,
        "left_keypoints": stats["left_keypoints"],
        "right_keypoints": stats["right_keypoints"],
        "good_matches": stats["good_matches"],
        "ransac_inliers": stats["inliers"],
        "inlier_ratio": round(stats["inliers"] / stats["good_matches"], 3) if stats["good_matches"] else 0.0,
        "sift_success": sift_ok,
        "opencv_success": opencv_ok,
        "sift_note": sift_panorama_status,
        "opencv_note": opencv_status,
        "file_prefix": label,
    }
    print(f"[{label}] 匹配 {record['good_matches']}、內點 {record['ransac_inliers']}、"
          f"SIFT {'成功' if sift_ok else '失敗'}、OpenCV {'成功' if opencv_ok else '失敗'}")
    return record


def write_reports(records, results_dir):
    csv_path = results_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Quiz 4 進階：拼接條件與 CLAHE 前處理比較", "",
        "原始成功組：left.jpg + right.jpg；原始失敗組：failure_view1.jpg + failure_view2.jpg。",
        "只改動成功組的右圖：亮度乘 0.70／0.40／0.20，或裁掉左側 15%／30%／45%。",
        "這是對既有照片的模擬實驗；裁切比例不是實際視野重疊率，拍攝角度也未在此實驗中改變。",
        "CLAHE 作用於 YCrCb 亮度通道；raw 與 clahe 使用相同的 SIFT、匹配與 RANSAC 流程。", "",
        "| 照片組 | 條件 | 前處理 | 左/右特徵點 | 可靠匹配 | RANSAC 內點 | 內點率 | SIFT 產圖 | OpenCV 產圖 |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['pair']} | {row['condition']} | {row['preprocess']} | "
            f"{row['left_keypoints']}/{row['right_keypoints']} | {row['good_matches']} | "
            f"{row['ransac_inliers']} | {row['inlier_ratio']:.1%} | "
            f"{'是' if row['sift_success'] else '否'} | {'是' if row['opencv_success'] else '否'} |"
        )
    def first_failed(case_prefix, method, success_field):
        for row in records:
            if (row["pair"] == "success" and row["case"].startswith(case_prefix)
                    and row["preprocess"] == method and not row[success_field]):
                return row["condition"]
        return "測試範圍內未失敗"

    lines.extend([
        "", "## 本次測試點的觀察", "",
        f"- 不加前處理時，SIFT 在「{first_failed('brightness_', 'raw', 'sift_success')}」首次無法產圖；"
        f"CLAHE 後的對應測試點是「{first_failed('brightness_', 'clahe', 'sift_success')}」。",
        f"- 不加前處理時，SIFT 在「{first_failed('overlap_crop_', 'raw', 'sift_success')}」首次無法產圖；"
        f"CLAHE 後的對應測試點是「{first_failed('overlap_crop_', 'clahe', 'sift_success')}」。",
        "- 這些是離散測試點，不是精確的亮度或重疊臨界值。不同前處理或執行環境可能改變結果。",
        "", "## 失敗原因與判讀", "",
    ])
    for row in records:
        if not row["sift_success"] or not row["opencv_success"]:
            lines.append(
                f"- `{row['file_prefix']}`：SIFT {row['sift_note']}；"
                f"OpenCV {row['opencv_note']}。"
            )
    lines.extend([
        "", "特徵點或匹配數量多，不保證估計出的透視變換合理。",
        "「產圖成功」只代表流程完成，仍須打開 panorama 圖目視檢查接縫、扭曲與重影。",
        "若本次測試全部成功，表示尚未測到失敗臨界值，不可宣稱已找到臨界條件。",
    ])
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path,
                        default=project_dir / "Week01" / "Data" / "Quiz4")
    parser.add_argument(
        "--results-dir", type=Path,
        default=project_dir / "results" / "quiz4_two_pairs"
    )
    args = parser.parse_args()
    results_dir = args.results_dir / datetime.now().strftime("experiments_%Y%m%d_%H%M%S_%f")
    results_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for pair_name, left_name, right_name in PAIRS:
        left = read_image(args.images_dir / left_name)
        right = read_image(args.images_dir / right_name)
        cases = experiment_cases(left, right) if pair_name == "success" else [
            ("original", "原圖（既有失敗組）", left, right)
        ]
        for case_name, condition, case_left, case_right in cases:
            for method in ("raw", "clahe"):
                records.append(process_case(
                    pair_name, case_name, condition, method,
                    case_left, case_right, results_dir
                ))
    write_reports(records, results_dir)
    print(f"結果資料夾：{results_dir}")
    print("請開啟 summary.md 與各 panorama 圖，目視檢查效果。")


if __name__ == "__main__":
    main()
