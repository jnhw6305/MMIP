import cv2
import numpy as np

image = cv2.imread("images/document.jpg")
if image is None:
    raise FileNotFoundError("找不到 images/document.jpg")

# 縮小顯示視窗；點擊時再換算回原圖座標
height, width = image.shape[:2]
scale = min(700 / height, 900 / width, 1.0)
display = cv2.resize(image, None, fx=scale, fy=scale)
points = []

def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        original_x = x / scale
        original_y = y / scale
        points.append([original_x, original_y])
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            display, str(len(points)), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        cv2.imshow("Select four corners", display)

cv2.imshow("Select four corners", display)
cv2.setMouseCallback("Select four corners", on_mouse)

print("依序點選封面的：左上 → 右上 → 右下 → 左下")
print("點完四點後，回到照片視窗按任意鍵。")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("未選滿四個角，請重新執行")

source = np.float32(points)
output_width = 700
output_height = 1000

destination = np.float32([
    [0, 0],
    [output_width - 1, 0],
    [output_width - 1, output_height - 1],
    [0, output_height - 1]
])

matrix = cv2.getPerspectiveTransform(source, destination)
corrected = cv2.warpPerspective(
    image, matrix, (output_width, output_height)
)

cv2.imwrite("results/perspective_corrected.jpg", corrected)
print("完成：results/perspective_corrected.jpg")