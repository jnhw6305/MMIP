# MMIP — Multi-Modality Image Processing

多模態影像資料處理課程作業。第一週包含彩色轉灰階、直方圖等化、透視校正與影像拼接四題；每題保留程式、輸入影像和可檢查的結果。

## 第一週導覽

- [Week01 作業總覽與執行方式](Week01/README.md)
- [程式碼與指令](Week01/Code/README.md)
- [AI 輔助紀錄與人工驗證](Week01/AI/README.md)
- [輸入與結果資料索引](Week01/Data/README.md)

| 題目 | 程式 | 主要結果 |
| --- | --- | --- |
| Quiz 1：彩色轉灰階 | [quiz1_grayscale.py](Week01/Code/quiz1_grayscale.py) | [NumPy／OpenCV 比較](Week01/Data/Quiz1/advanced/summary.md) |
| Quiz 2：直方圖等化 | [quiz2_histogram.py](Week01/Code/quiz2_histogram.py) | [直方圖與效能比較](Week01/Data/Quiz2/advanced/summary.md) |
| Quiz 3：透視校正 | [quiz3_perspective.py](Week01/Code/quiz3_perspective.py) | [多張實拍測試](results/quiz3_advanced/summary.md) |
| Quiz 4：影像拼接 | [quiz4_stitching.py](Week01/Code/quiz4_stitching.py) | [條件與前處理比較](results/quiz4_advanced/summary.md) |

執行環境需要 Python、NumPy 與 OpenCV。於儲存庫根目錄執行程式；詳細指令見 [Week01/Code/README.md](Week01/Code/README.md)。報告中的「成功產圖」不等於完美影像，請一併查看角點、接縫、黑邊及限制說明。
