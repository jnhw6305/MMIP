"""第 60 頁：比較一組可拼接與一組難以拼接的照片。

在 MMIP作業 資料夾執行：python quiz4_two_pairs.py
輸出都放在 results/quiz4_two_pairs/。
"""

from pathlib import Path
import argparse

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
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
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


def process_pair(label, left_name, right_name, images_dir, results_dir):
    left = read_image(images_dir / left_name)
    right = read_image(images_dir / right_name)
    save_image(results_dir / f"{label}_comparison.jpg", comparison(left, right))

    homography, stats, sift_status = sift_homography(
        left, right, label, results_dir
    )
    sift_panorama_status = sift_status
    if homography is not None:
        try:
            panorama = make_sift_panorama(left, right, homography)
            save_image(results_dir / f"{label}_sift_panorama.jpg", panorama)
            sift_panorama_status = "已產生 SIFT 拼接圖；仍須目視檢查接縫"
        except ValueError as exc:
            sift_panorama_status = f"未產生 SIFT 拼接圖：{exc}"

    # OpenCV 內建 Stitcher 用來比較自製 SIFT 拼接的效果。
    # 它內部的特徵方法不一定是 SIFT，因此結果要分開標示。
    try:
        status, stitched = cv2.Stitcher_create(
            cv2.Stitcher_PANORAMA
        ).stitch([left, right])
        if status == cv2.Stitcher_OK and stitched is not None:
            save_image(results_dir / f"{label}_opencv_panorama.jpg", stitched)
            opencv_status = "成功"
        else:
            opencv_status = f"失敗，狀態碼 {status}"
    except cv2.error as exc:
        opencv_status = f"失敗：{str(exc).splitlines()[0]}"

    lines = [
        f"[{label}] {left_name} + {right_name}",
        f"左圖 SIFT 特徵點：{stats['left_keypoints']}",
        f"右圖 SIFT 特徵點：{stats['right_keypoints']}",
        f"可靠匹配點：{stats['good_matches']}",
        f"RANSAC 內點：{stats['inliers']}",
        f"SIFT 拼接：{sift_panorama_status}",
        f"OpenCV Stitcher：{opencv_status}",
        "",
    ]
    print("\n".join(lines))
    return lines


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=project_dir / "images")
    parser.add_argument(
        "--results-dir", type=Path,
        default=project_dir / "results" / "quiz4_two_pairs"
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    report = ["第 60 頁：兩組照片的影像拼接比較", ""]
    for pair in PAIRS:
        try:
            report.extend(process_pair(*pair, args.images_dir, args.results_dir))
        except (FileNotFoundError, RuntimeError, cv2.error) as exc:
            message = f"[{pair[0]}] 處理失敗：{exc}"
            print(message)
            report.extend([message, ""])
    report.append("註：失敗組仍會輸出並排比較圖與可取得的匹配圖。")
    (args.results_dir / "summary.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"結果資料夾：{args.results_dir}")


if __name__ == "__main__":
    main()