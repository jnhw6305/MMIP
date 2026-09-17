"""Quiz 1: compare NumPy and OpenCV colour-to-grayscale conversion.

Run: python quiz1_grayscale.py
The script writes exact PNG outputs, a comparison figure, and summary.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def read_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"無法讀取影像：{path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"無法儲存影像：{path}")
    encoded.tofile(path)


def grayscale_numpy(image: np.ndarray) -> np.ndarray:
    # OpenCV reads colour images in BGR order, not RGB order.
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    return np.clip(np.rint(0.114 * b + 0.587 * g + 0.299 * r),
                   0, 255).astype(np.uint8)


def average_seconds(action, repeats: int) -> float:
    for _ in range(3):
        action()
    start = time.perf_counter()
    for _ in range(repeats):
        action()
    return (time.perf_counter() - start) / repeats


def comparison_panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    scale = min(420 / width, 420 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    panel = np.full((465, 420, 3), 255, dtype=np.uint8)
    y = (420 - resized.shape[0]) // 2
    x = (420 - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(panel, label, (12, 450), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (20, 20, 20), 2)
    return panel


def main() -> None:
    root = Path(__file__).resolve().parent
    submission = root / "Week01" / "Data" / "Quiz1"
    default_input = root / "images" / "color.jpg"
    if not default_input.exists():
        default_input = submission / "color.jpg"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path,
                        default=(submission / "advanced" if submission.is_dir()
                                 else root / "results" / "quiz1_advanced"))
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必須至少為 1")
    if not args.input.is_file():
        parser.error(f"找不到輸入影像：{args.input}")

    image = read_color(args.input)
    opencv_action = lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numpy_action = lambda: grayscale_numpy(image)
    opencv_time = average_seconds(opencv_action, args.repeat)
    numpy_time = average_seconds(numpy_action, args.repeat)
    gray_cv = opencv_action()
    gray_np = numpy_action()
    difference = cv2.absdiff(gray_cv, gray_np)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_png(output / "gray_opencv.png", gray_cv)
    save_png(output / "gray_numpy.png", gray_np)
    save_png(output / "difference.png", difference)

    display_difference = np.clip(
        difference.astype(np.uint16) * 40, 0, 255
    ).astype(np.uint8)
    comparison = cv2.hconcat([
        comparison_panel(image, "Original colour"),
        comparison_panel(gray_cv, "OpenCV grayscale"),
        comparison_panel(gray_np, "NumPy grayscale"),
        comparison_panel(display_difference, "Absolute difference x40"),
    ])
    save_png(output / "comparison.png", comparison)

    mean_diff = float(difference.mean())
    max_diff = int(difference.max())
    differing = int(np.count_nonzero(difference))
    total = difference.size
    faster = "OpenCV" if opencv_time < numpy_time else "NumPy"
    result_note = (
        "本張影像的兩種輸出逐像素完全相同。"
        if max_diff == 0 else
        "細微差異可能來自整數權重與捨入方式。"
    )
    report = f"""# Quiz 1 進階比較

輸入影像：`{args.input.name}`，大小 {image.shape[1]} × {image.shape[0]}。
兩種方法都執行 {args.repeat} 次，先預熱 3 次；計時不含讀檔、存檔與繪圖。

| 方法 | 平均每次時間 |
| --- | ---: |
| OpenCV `cvtColor` | {opencv_time * 1000:.4f} ms |
| NumPy 自行計算 | {numpy_time * 1000:.4f} ms |

NumPy 使用 BGR 權重：灰階 = 0.114 B + 0.587 G + 0.299 R，再四捨五入轉為 8-bit。

| 差異指標 | 數值 |
| --- | ---: |
| 平均絕對像素差 | {mean_diff:.4f} |
| 最大絕對像素差 | {max_diff} |
| 不同像素數 | {differing:,} / {total:,} ({differing / total:.2%}) |

本次測量由 **{faster}** 較快。兩張灰階結果的差異可由 `difference.png`
逐像素檢查；`comparison.png` 的差異面板放大 40 倍以便觀看，
不是實際像素差值。{result_note}
執行時間會隨電腦負載變動，應以多次測量而非單次結果判斷。
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"OpenCV 平均：{opencv_time * 1000:.4f} ms")
    print(f"NumPy  平均：{numpy_time * 1000:.4f} ms")
    print(f"平均像素差：{mean_diff:.4f}；最大像素差：{max_diff}")
    print(f"結果資料夾：{output}")


if __name__ == "__main__":
    main()
