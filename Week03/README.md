# Week03：CNN 影像分類與可解釋性分析

## 使用 Google Colab 開啟完整作業

本週 **Quiz 1–Quiz 3 的所有基礎與進階問題**，皆已在 [MMIP_CNN完整作業.ipynb](MMIP_CNN完整作業.ipynb) 中依序回答。Notebook 包含問題定義、完整程式碼、正式訓練輸出、評估表格、圖像結果及文字分析；建議讀者直接透過 Google Colab 閱讀與執行。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnhw6305/MMIP/blob/main/Week03/MMIP_CNN%E5%AE%8C%E6%95%B4%E4%BD%9C%E6%A5%AD.ipynb)

操作步驟：

1. 點擊上方 **Open In Colab** 按鈕，在 Google Colab 開啟完整 Notebook。
2. 登入 Google 帳號；若要重新訓練，選擇「執行階段 → 變更執行階段類型 → T4 GPU」。
3. Notebook 已設定 `FULL_RUN=True`。如需重現正式結果，可選擇「執行階段 → 全部執行」；若只想查看作業，直接閱讀已保留的輸出即可。
4. 依 Notebook 順序閱讀 Quiz 1、Quiz 2 與 Quiz 3；各題的基礎要求、進階實驗、圖表與結論均放在對應章節內。

| Notebook 章節 | 已回答內容 |
| --- | --- |
| Quiz 1 | CIFAR-10 資料來源、問題定義、10 個真實事物類別、資料切分與理由 |
| Quiz 2 基礎 | 自行設計 CNN、MobileNetV2、Testing Dataset 預測、Top-1／Top-5、混淆矩陣、ROC、Macro-AUC 與參數量比較 |
| Quiz 2 進階 | Plain CNN 與 MobileNetV2 的超參數組合實驗及結果分析 |
| Quiz 3 基礎 | Data Augmentation 設計、前後對照、實際結果與限制分析 |
| Quiz 3 進階 | 兩個 CNN Kernel、activation map、Grad-CAM 正確／錯誤案例與可解釋性分析 |

本週以 CIFAR-10 建立影像分類系統，任務是輸入一張 32×32 RGB 圖片，判斷圖片中的主要物體最接近飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船或卡車中的哪一類。完整程式、訓練紀錄、圖表與分析均保留在 [MMIP_CNN完整作業.ipynb](MMIP_CNN完整作業.ipynb)。

## 資料與切分

CIFAR-10 共包含 60,000 張圖片，每類 6,000 張。官方 10,000 張測試集只用於最終評估；其餘 50,000 張資料以固定亂數種子 42 逐類切分為：

| 資料集 | 數量 | 用途 |
| --- | ---: | --- |
| Training | 45,000 | 模型參數學習 |
| Validation | 5,000 | 選模、early stopping 與超參數比較 |
| Testing | 10,000 | 最終效能評估 |

## 模型設計

### MultiScale-Pyramid CNN

自行設計的 Plain CNN 以平行 3×3 與 5×5 convolution 擷取不同尺度的局部特徵，串接後使用三個金字塔式卷積階段，讓空間尺寸逐層縮小、通道數由 32 增至 128。模型搭配 Batch Normalization、ReLU、Max Pooling、Dropout 與 Global Average Pooling，不使用預訓練權重或現成 backbone。

### MobileNetV2

對照模型使用 ImageNet 預訓練 MobileNetV2。輸入影像先放大至 96×96，第一階段凍結 backbone 訓練分類頭，正式模式再解凍最後 30 層，以較低 learning rate 執行 fine-tuning。

## 基礎與進階內容

- **Quiz 1**：說明資料來源、問題定義、10 個真實事物類別及固定分層切分策略。
- **Quiz 2 基礎**：訓練 Plain CNN 與 MobileNetV2，輸出 Testing Dataset 預測、Top-1、Top-5、參數量、混淆矩陣、逐類 ROC 與 Macro-AUC。
- **Quiz 2 進階**：比較 learning rate、dropout 與 filter 寬度等超參數組合，並以 validation Top-1 選模。
- **Quiz 3 基礎**：比較水平翻轉、平移、旋轉、縮放與對比調整等 Data Augmentation 的前後結果。
- **Quiz 3 進階**：視覺化兩個第一層 CNN Kernel 及 activation map，並以 Grad-CAM 解釋三個正確與三個錯誤案例。

## 主要結果

| 模型 | Top-1 | Top-5 | Macro-AUC | 參數量 |
| --- | ---: | ---: | ---: | ---: |
| MultiScale-Pyramid CNN | 0.8427 | 0.9931 | 0.986053 | 299,754 |
| MobileNetV2 | **0.8909** | **0.9965** | **0.992878** | 2,270,794 |

MobileNetV2 的 Top-1 比 Plain CNN 高 4.82 個百分點，但參數量約為 7.58 倍。Plain CNN 以較小模型取得 84.27% Top-1，呈現效能與模型規模之間的取捨。

超參數組合中，Plain CNN 的最佳 validation Top-1 為 0.8714，對應 `lr=0.001`、`dropout=0.3`、`filters=(64,128,256)`；MobileNetV2 最佳為 0.8734，對應 `lr=0.001`、`dropout=0.2`。這些設定屬於組合比較，不作單一超參數的因果結論。

## Data Augmentation 結果與限制

| 模型 | Baseline Top-1 | Augmentation Top-1 | 變化 |
| --- | ---: | ---: | ---: |
| Plain CNN | 0.8427 | 0.7116 | -0.1311 |
| MobileNetV2 | 0.8909 | 0.8026 | -0.0883 |

本次 augmentation 沒有提升測試表現。可能原因是同時使用 10% 平移、約 ±28.8° 旋轉、縮放及對比變化，對 32×32 小影像過強；此外，MobileNetV2 baseline 與 augmentation 版本的 fine-tuning 流程並不完全相同，因此不能把差異完全歸因於 augmentation。後續應降低擴增強度、固定每次模型初始化，並統一 fine-tuning 流程後重新比較。

## Kernel 與 Grad-CAM 觀察

選出的 Kernel 15 與 Kernel 2 主要對色彩對比、局部輪廓、紋理及前景背景交界產生反應，符合 CNN 第一層通常學習低階視覺特徵的現象。Grad-CAM 顯示部分正確案例會關注物體輪廓，但熱區也可能落在背景或高對比區域；錯誤案例則反映相近交通工具外觀及背景紋理可能干擾模型判斷。這些熱圖只能呈現單次預測的敏感區域，不代表因果關係。

## 執行方式

建議使用 Google Colab 的 T4 GPU 開啟 Notebook，依序執行所有儲存格。正式實驗已設定：

```python
FULL_RUN = True
EPOCHS = 30
HP_EPOCHS = 12
```

主要套件包含 TensorFlow／Keras、NumPy、Pandas、Scikit-Learn、Matplotlib 與 Seaborn。Notebook 執行時會建立 `cnn_homework_outputs/`，保存模型、CSV 與 PNG；GitHub 上的 Notebook 已嵌入本次正式執行的主要表格與圖像輸出。
