# Quiz 4 進階：拼接條件與 CLAHE 前處理比較

原始成功組：left.jpg + right.jpg；原始失敗組：failure_view1.jpg + failure_view2.jpg。
只改動成功組的右圖：亮度乘 0.70／0.40／0.20，或裁掉左側 15%／30%／45%。
這是對既有照片的模擬實驗；裁切比例不是實際視野重疊率，拍攝角度也未在此實驗中改變。
CLAHE 作用於 YCrCb 亮度通道；raw 與 clahe 使用相同的 SIFT、匹配與 RANSAC 流程。

| 照片組 | 條件 | 前處理 | 左/右特徵點 | 可靠匹配 | RANSAC 內點 | 內點率 | SIFT 產圖 | OpenCV 產圖 |
|---|---|---|---:|---:|---:|---:|---|---|
| success | 原圖 | raw | 469/193 | 48 | 19 | 39.6% | 是 | 是 |
| success | 原圖 | clahe | 805/451 | 67 | 21 | 31.3% | 是 | 是 |
| success | 右圖亮度 70% | raw | 469/118 | 23 | 13 | 56.5% | 是 | 是 |
| success | 右圖亮度 70% | clahe | 805/299 | 44 | 16 | 36.4% | 是 | 是 |
| success | 右圖亮度 40% | raw | 469/33 | 9 | 0 | 0.0% | 否 | 否 |
| success | 右圖亮度 40% | clahe | 805/153 | 37 | 15 | 40.5% | 是 | 否 |
| success | 右圖亮度 20% | raw | 469/3 | 1 | 0 | 0.0% | 否 | 否 |
| success | 右圖亮度 20% | clahe | 805/43 | 9 | 0 | 0.0% | 否 | 否 |
| success | 裁掉右圖左側 15% | raw | 469/146 | 28 | 9 | 32.1% | 否 | 否 |
| success | 裁掉右圖左側 15% | clahe | 805/384 | 54 | 3 | 5.6% | 否 | 是 |
| success | 裁掉右圖左側 30% | raw | 469/134 | 29 | 9 | 31.0% | 否 | 否 |
| success | 裁掉右圖左側 30% | clahe | 805/352 | 43 | 12 | 27.9% | 否 | 否 |
| success | 裁掉右圖左側 45% | raw | 469/86 | 16 | 4 | 25.0% | 否 | 否 |
| success | 裁掉右圖左側 45% | clahe | 805/310 | 36 | 9 | 25.0% | 否 | 否 |
| failure | 原圖（既有失敗組） | raw | 2435/1524 | 97 | 28 | 28.9% | 否 | 否 |
| failure | 原圖（既有失敗組） | clahe | 5101/4813 | 108 | 35 | 32.4% | 否 | 否 |

## 本次測試點的觀察

- 不加前處理時，SIFT 在「右圖亮度 40%」首次無法產圖；CLAHE 後的對應測試點是「右圖亮度 20%」。
- 不加前處理時，SIFT 在「裁掉右圖左側 15%」首次無法產圖；CLAHE 後的對應測試點是「裁掉右圖左側 15%」。
- 這些是離散測試點，不是精確的亮度或重疊臨界值。不同前處理或執行環境可能改變結果。

## 失敗原因與判讀

- `success_brightness_40_raw`：SIFT 可靠匹配點少於 10 個；OpenCV 失敗，狀態碼 1。
- `success_brightness_40_clahe`：SIFT 已產生 SIFT 拼接圖；仍須目視檢查接縫；OpenCV 失敗，狀態碼 1。
- `success_brightness_20_raw`：SIFT 可靠匹配點少於 10 個；OpenCV 失敗，狀態碼 1。
- `success_brightness_20_clahe`：SIFT 可靠匹配點少於 10 個；OpenCV 失敗，狀態碼 1。
- `success_overlap_crop_15_raw`：SIFT 幾何驗證通過的匹配點太少；OpenCV 失敗，狀態碼 1。
- `success_overlap_crop_15_clahe`：SIFT 幾何驗證通過的匹配點太少；OpenCV 成功。
- `success_overlap_crop_30_raw`：SIFT 幾何驗證通過的匹配點太少；OpenCV 失敗，狀態碼 1。
- `success_overlap_crop_30_clahe`：SIFT 未產生 SIFT 拼接圖：變換後的照片四角交錯；OpenCV 失敗，狀態碼 1。
- `success_overlap_crop_45_raw`：SIFT 幾何驗證通過的匹配點太少；OpenCV 失敗，狀態碼 1。
- `success_overlap_crop_45_clahe`：SIFT 幾何驗證通過的匹配點太少；OpenCV 失敗，狀態碼 1。
- `failure`：SIFT 未產生 SIFT 拼接圖：變換後的照片四角交錯；OpenCV 失敗，狀態碼 1。
- `failure_original_clahe`：SIFT 未產生 SIFT 拼接圖：變換後的照片四角交錯；OpenCV 失敗，狀態碼 1。

特徵點或匹配數量多，不保證估計出的透視變換合理。
「產圖成功」只代表流程完成，仍須打開 panorama 圖目視檢查接縫、扭曲與重影。
若本次測試全部成功，表示尚未測到失敗臨界值，不可宣稱已找到臨界條件。

## 代表性影像與目視品質

- [亮度 40%、未前處理的輸入對照](evidence/success_brightness_40_raw_comparison.jpg)：右圖變暗，SIFT 未產生拼接圖。
- [亮度 40%、CLAHE 後的輸入對照](evidence/success_brightness_40_clahe_comparison.jpg)：右圖局部對比提升。
- [亮度 40%、CLAHE 後的 SIFT 拼接圖](evidence/success_brightness_40_clahe_sift_panorama.jpg)：流程產圖，但仍有大面積黑邊與可見接縫，不能視為高品質拼接。
- [裁掉右圖左側 15%、CLAHE 後的 OpenCV 拼接圖](evidence/success_overlap_crop_15_clahe_opencv_panorama.jpg)：流程產圖，但邊緣仍有黑色未覆蓋區域。

因此，CLAHE 在部分測試點改善了「能否產圖」的穩定性，未證明它總能改善拼接的視覺品質。
