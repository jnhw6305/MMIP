"""Quiz 2: compare NumPy and OpenCV histogram equalization.

Run: python quiz2_histogram.py
The script writes exact PNG outputs, histograms, and summary.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取影像：{path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"無法儲存影像：{path}")
    encoded.tofile(path)


def equalize_numpy(image: np.ndarray) -> np.ndarray:
    histogram = np.bincount(image.ravel(), minlength=256)
    cdf = histogram.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0 or int(nonzero[0]) == image.size:
        return image.copy()
    cdf_min = int(nonzero[0])
    lut = np.clip(np.rint((cdf - cdf_min) * 255 /
                          (image.size - cdf_min)), 0, 255).astype(np.uint8)
    return lut[image]


def average_seconds(action, repeats: int) -> float:
    for _ in range(3):
        action()
    start = time.perf_counter()
    for _ in range(repeats):
        action()
    return (time.perf_counter() - start) / repeats


def image_stats(image: np.ndarray) -> tuple[int, int, float, float, float]:
    p5, p95 = np.percentile(image, [5, 95])
    return (int(image.min()), int(image.max()), float(image.mean()),
            float(image.std()), float(p95 - p5))


def image_panel(image: np.ndarray, title: str) -> np.ndarray:
    height, width = image.shape
    scale = min(420 / width, 280 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    panel = np.full((320, 420, 3), 255, dtype=np.uint8)
    y = (280 - resized.shape[0]) // 2
    x = (420 - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = cv2.cvtColor(
        resized, cv2.COLOR_GRAY2BGR
    )
    cv2.putText(panel, title, (16, 307), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (20, 20, 20), 2)
    return panel


def histogram_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = np.full((320, 420, 3), 255, dtype=np.uint8)
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    plot_height = 235
    plot_width = 350
    left = 48
    bottom = 275
    cv2.line(panel, (left, bottom), (left + plot_width, bottom),
             (30, 30, 30), 2)
    cv2.line(panel, (left, bottom), (left, bottom - plot_height),
             (30, 30, 30), 2)
    peak = max(float(hist.max()), 1.0)
    points = np.array([
        [left + round(index * plot_width / 255),
         bottom - round(value * plot_height / peak)]
        for index, value in enumerate(hist)
    ], dtype=np.int32)
    cv2.polylines(panel, [points], False, (180, 75, 25), 2)
    cv2.putText(panel, "0", (left - 8, bottom + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (20, 20, 20), 1)
    cv2.putText(panel, "255", (left + plot_width - 27, bottom + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(panel, title + " histogram", (58, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    return panel


def main() -> None:
    root = Path(__file__).resolve().parent
    submission = root / "Week01" / "Data" / "Quiz2"
    default_input = root / "images" / "dark.jpg"
    if not default_input.exists():
        default_input = submission / "dark.jpg"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path,
                        default=(submission / "advanced" if submission.is_dir()
                                 else root / "results" / "quiz2_advanced"))
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必須至少為 1")
    if not args.input.is_file():
        parser.error(f"找不到輸入影像：{args.input}")

    gray = read_gray(args.input)
    opencv_action = lambda: cv2.equalizeHist(gray)
    numpy_action = lambda: equalize_numpy(gray)
    opencv_time = average_seconds(opencv_action, args.repeat)
    numpy_time = average_seconds(numpy_action, args.repeat)
    equalized_cv = opencv_action()
    equalized_np = numpy_action()
    difference = cv2.absdiff(equalized_cv, equalized_np)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_png(output / "equalized_opencv.png", equalized_cv)
    save_png(output / "equalized_numpy.png", equalized_np)
    save_png(output / "difference.png", difference)

    versions = (("Before", gray), ("OpenCV", equalized_cv),
                ("NumPy", equalized_np))
    comparison = cv2.vconcat([
        cv2.hconcat([image_panel(image, title) for title, image in versions]),
        cv2.hconcat([histogram_panel(image, title)
                     for title, image in versions]),
    ])
    save_png(output / "comparison.png", comparison)

    before = image_stats(gray)
    after_cv = image_stats(equalized_cv)
    after_np = image_stats(equalized_np)
    mean_diff = float(difference.mean())
    max_diff = int(difference.max())
    differing = int(np.count_nonzero(difference))
    faster = "OpenCV" if opencv_time < numpy_time else "NumPy"
    result_note = (
        "本張影像的兩種輸出逐像素完全相同。"
        if max_diff == 0 else
        "細微差異可能來自查找表的捨入方式。"
    )
    report = f"""# Quiz 2 進階比較

輸入影像：`{args.input.name}`，大小 {gray.shape[1]} × {gray.shape[0]}。
兩種方法都執行 {args.repeat} 次，先預熱 3 次；計時不含讀檔、存檔與繪圖。
NumPy 方法由 256 格直方圖、累積分布函數（CDF）與查找表（LUT）實作。

| 方法 | 平均每次時間 |
| --- | ---: |
| OpenCV `equalizeHist` | {opencv_time * 1000:.4f} ms |
| NumPy 自行計算 | {numpy_time * 1000:.4f} ms |

| 影像 | 最小值 | 最大值 | 平均亮度 | 標準差 | P95 - P5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 處理前 | {before[0]} | {before[1]} | {before[2]:.2f} | {before[3]:.2f} | {before[4]:.2f} |
| OpenCV 後 | {after_cv[0]} | {after_cv[1]} | {after_cv[2]:.2f} | {after_cv[3]:.2f} | {after_cv[4]:.2f} |
| NumPy 後 | {after_np[0]} | {after_np[1]} | {after_np[2]:.2f} | {after_np[3]:.2f} | {after_np[4]:.2f} |

P95 - P5 是中央 90% 像素的亮度跨度，標準差反映整體亮度分散程度。
請參照 `comparison.png` 同時比較視覺效果與三張直方圖；
較寬的分布通常表示對比提高，但也可能放大雜訊。

| 兩種方法差異 | 數值 |
| --- | ---: |
| 平均絕對像素差 | {mean_diff:.4f} |
| 最大絕對像素差 | {max_diff} |
| 不同像素數 | {differing:,} / {gray.size:,} ({differing / gray.size:.2%}) |

本次測量由 **{faster}** 較快。差異以 `difference.png` 的實際像素值呈現；
{result_note}時間會隨電腦負載變動。
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"OpenCV 平均：{opencv_time * 1000:.4f} ms")
    print(f"NumPy  平均：{numpy_time * 1000:.4f} ms")
    print(f"處理前後標準差：{before[3]:.2f} -> {after_cv[3]:.2f}")
    print(f"平均像素差：{mean_diff:.4f}；最大像素差：{max_diff}")
    print(f"結果資料夾：{output}")


if __name__ == "__main__":
    main()
