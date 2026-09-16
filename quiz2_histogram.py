import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread("images/dark.jpg", cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError("找不到 images/dark.jpg")

def numpy_equalize(img):
    histogram = np.bincount(img.ravel(), minlength=256)
    cdf = histogram.cumsum()

    nonzero = cdf[cdf > 0]
    cdf_min = nonzero[0]
    total = img.size

    if total == cdf_min:
        return img.copy()

    lookup = np.round(
        (cdf - cdf_min) / (total - cdf_min) * 255
    )
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[img]

start = time.perf_counter()
for _ in range(100):
    equalized_cv = cv2.equalizeHist(gray)
opencv_time = (time.perf_counter() - start) / 100

start = time.perf_counter()
for _ in range(100):
    equalized_np = numpy_equalize(gray)
numpy_time = (time.perf_counter() - start) / 100

cv2.imwrite("equalized_opencv.jpg", equalized_cv)
cv2.imwrite("equalized_numpy.jpg", equalized_np)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].imshow(gray, cmap="gray")
axes[0, 0].set_title("Before")
axes[0, 1].hist(gray.ravel(), bins=256, range=(0, 256))
axes[0, 1].set_title("Histogram Before")

axes[1, 0].imshow(equalized_cv, cmap="gray")
axes[1, 0].set_title("After")
axes[1, 1].hist(equalized_cv.ravel(), bins=256, range=(0, 256))
axes[1, 1].set_title("Histogram After")

for ax in (axes[0, 0], axes[1, 0]):
    ax.axis("off")

plt.tight_layout()
plt.savefig("histogram_comparison.png", dpi=200)
plt.show()

difference = cv2.absdiff(equalized_cv, equalized_np)
print(f"OpenCV 平均時間：{opencv_time:.8f} 秒")
print(f"NumPy 平均時間：{numpy_time:.8f} 秒")
print(f"平均像素差：{difference.mean():.4f}")