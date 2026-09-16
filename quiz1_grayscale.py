import time
import cv2
import numpy as np

image = cv2.imread("images/color.jpg")
if image is None:
    raise FileNotFoundError("找不到 images/color.jpg")

# 方法一：OpenCV
start = time.perf_counter()
for _ in range(100):
    gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
opencv_time = (time.perf_counter() - start) / 100

# 方法二：NumPy，依 RGB 權重計算
start = time.perf_counter()
for _ in range(100):
    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    gray_np = np.clip(
        0.114 * blue + 0.587 * green + 0.299 * red,
        0, 255
    ).astype(np.uint8)
numpy_time = (time.perf_counter() - start) / 100

difference = cv2.absdiff(gray_cv, gray_np)

cv2.imwrite("results/gray_opencv.jpg", gray_cv)
cv2.imwrite("results/gray_numpy.jpg", gray_np)
cv2.imwrite("results/gray_difference.jpg", difference)

print(f"OpenCV 平均時間：{opencv_time:.8f} 秒")
print(f"NumPy 平均時間：{numpy_time:.8f} 秒")
print(f"平均像素差：{difference.mean():.4f}")
print(f"最大像素差：{difference.max()}")