# Week 01：基礎電腦視覺作業

資料分工：`Code/` 放四支主程式，`AI/` 記錄 AI 協助與人工驗證，`Data/` 放輸入影像與部分輸出；較完整的進階實驗報告在儲存庫根目錄的 `results/`。不重複存放同一份程式。

| 題目 | 基礎內容 | 進階內容與結果 |
| --- | --- | --- |
| Quiz 1 | 一張彩色影像轉灰階 | NumPy 與 OpenCV 比較；100 次平均計時、逐像素差異。[報告](Data/Quiz1/advanced/summary.md) |
| Quiz 2 | 暗圖等化與處理前後灰階直方圖 | NumPy CDF／LUT 與 OpenCV 比較；100 次平均計時與亮度統計。[報告](Data/Quiz2/advanced/summary.md) |
| Quiz 3 | 斜拍物件自動找四角、透視校正 | 四張實拍批次輸出角點圖與校正圖；另以模擬亮度／模糊找失敗條件。[報告](../results/quiz3_advanced/summary.md) |
| Quiz 4 | 兩張重疊影像的 SIFT 匹配與拼接 | 比較亮度、裁切及 CLAHE 前處理；記錄匹配點、內點與產圖狀態。[報告](../results/quiz4_advanced/summary.md) |

在 Anaconda Prompt 先切到儲存庫根目錄並啟用已安裝 NumPy、OpenCV 的環境：

```bat
cd /d D:\NYCU\MMIP-clone
conda activate mmip
python Week01\Code\quiz1_grayscale.py
python Week01\Code\quiz2_histogram.py
python Week01\Code\quiz3_perspective.py --output-dir results\quiz3_advanced
python Week01\Code\quiz4_stitching.py
```

Quiz 1／2 的預設輸出位於 `Week01/Data/Quiz1/advanced/`、`Week01/Data/Quiz2/advanced/`。Quiz 3 使用 `Week01/Data/Quiz3/` 中的 `document*.jpg`；會跳過既有的 `*_corners` 與 `*_corrected` 圖。Quiz 4 預設讀取 `Week01/Data/Quiz4/`，每次在 `results/quiz4_two_pairs/` 建立有時間戳記的新資料夾，不會自動提交到 GitHub。

注意：Quiz 3 四張照片未記錄實際拍攝角度，亮度／模糊測試為軟體模擬；Quiz 4 的裁切比例不是實際重疊率。程式產圖後仍須目視檢查，特別是 Quiz 3 的邊界裁切與 Quiz 4 的黑邊、接縫、重影。
