# MMIP — Multi-Modality Image Processing

多模態影像資料處理課程作業彙整。本儲存庫依週次保存影像處理、機器學習與深度學習實驗，包含可重現的程式、資料說明、模型評估、圖表及結果分析。

## 第一週導覽

- [Week01 作業總覽與執行方式](Week01/README.md)
- [程式碼與指令](Week01/Code/README.md)
- [AI 輔助紀錄與人工驗證](Week01/AI/README.md)
- [輸入與結果資料索引](Week01/Data/README.md)

| 題目 | 程式 | 主要結果 |
| --- | --- | --- |
| Quiz 1：彩色轉灰階 | [quiz1_grayscale.py](Week01/Code/quiz1_grayscale.py) | [NumPy／OpenCV 比較](Week01/Data/Quiz1/advanced/summary.md) |
| Quiz 2：直方圖等化 | [quiz2_histogram.py](Week01/Code/quiz2_histogram.py) | [直方圖與效能比較](Week01/Data/Quiz2/advanced/summary.md) |
| Quiz 3：透視校正 | [quiz3_perspective.py](Week01/Code/quiz3_perspective.py) | [多張實拍測試](Week01/Data/Quiz3/advanced/summary.md) |
| Quiz 4：影像拼接 | [quiz4_stitching.py](Week01/Code/quiz4_stitching.py) | [條件與前處理比較](Week01/Data/Quiz4/advanced/summary.md) |

執行環境需要 Python、NumPy 與 OpenCV。於儲存庫根目錄執行程式；詳細指令見 [Week01/Code/README.md](Week01/Code/README.md)。報告中的「成功產圖」不等於完美影像，請一併查看角點、接縫、黑邊及限制說明。

## 第二週導覽

[Week02：信用卡違約分類與模型評估](Week02/README.md) 完成 Quiz 1–3 的基礎與進階要求：

- 使用 Logistic Regression 與 Random Forest，比較分類門檻、混淆矩陣、Accuracy、Precision、Recall 與 F1-score。
- 使用 PyTorch 建立 MLP 並訓練 60 epochs，再加入 Dropout 與 L2 regularization 建立改善版模型。
- 比較 Training／Validation Loss、ROC Curve 與 AUC，並保留原始資料、訓練歷程及驗證集逐筆預測。

第二週執行環境需要 Python、NumPy、Pandas、Scikit-Learn、PyTorch 與 Matplotlib；執行方式和完整實驗結果請見 [Week02 作業報告](Week02/README.md)。

## 第三週導覽

[Week03：CNN 影像分類與可解釋性分析](Week03/README.md) 使用 CIFAR-10 比較自行設計的 MultiScale-Pyramid CNN 與 ImageNet 預訓練 MobileNetV2：

- 將 32×32 RGB 圖片判斷為飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船或卡車。
- 比較 Testing Dataset 的 Top-1、Top-5、混淆矩陣、逐類 ROC、Macro-AUC、參數量與訓練時間。
- 完成超參數組合實驗及 Data Augmentation 前後對照，並誠實分析 augmentation 未提升本次結果的原因與限制。
- 視覺化兩個第一層 CNN Kernel、activation map，以及正確與錯誤案例的 Grad-CAM。

基準實驗中，MobileNetV2 的測試 Top-1／Top-5／Macro-AUC 為 **0.8909／0.9965／0.992878**；自行設計的 Plain CNN 以 299,754 個參數取得 **0.8427** Top-1。完整程式、正式輸出與分析請見 [MMIP_CNN完整作業.ipynb](Week03/MMIP_CNN完整作業.ipynb)。
