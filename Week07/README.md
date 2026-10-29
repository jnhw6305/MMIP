# MMIP HW7 — LLM、VLM 與 DPO 實驗

## 執行環境

本作業 A、B、C 皆使用 **Google Colab** 與 **NVIDIA Tesla T4 GPU** 完整執行。三份 notebook 保留執行記錄，各作業的逐題結果、摘要、圖表與報告收錄在 `outputs_A.zip`、`outputs_B.zip` 與 `outputs_C.zip`。

## 作業一：大型語言模型

模型：`Qwen/Qwen2.5-1.5B-Instruct`

- 20 題自訂 Instruction Benchmark。
- Default 與 Enhanced Prompt 配對比較。
- BF16/FP16 與 NF4 4-bit 量化比較。
- 量測正確率、推論延遲及 GPU 記憶體。
- 比較 BF16/FP16 + Default、BF16/FP16 + Enhanced、NF4 + Default、NF4 + Enhanced。

主要檔案：

- `HW7_1_LLM.ipynb`
- `benchmark.json`

## 作業 B：視覺多模態大模型

模型：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

實驗包含 20 道合成視覺 benchmark，涵蓋顏色、形狀、計數、空間關係、大小比較、OCR、圖表與混合場景。兩模型使用相同影像、問題、Prompt、解碼條件與評分器。

主要檔案：

- `HW7_B_VLM.ipynb`
- `vlm_benchmark.json`
- `vlm_benchmark_images.zip`

## 作業 C：DPO 微調 LLM 與 VLM

### 基礎

- 使用 120 組偏好資料微調 `Qwen/Qwen2.5-0.5B-Instruct`。
- 優化正確且簡潔的繁體中文回答。
- 30 題目標能力測試。
- 100 題獨立通用能力 benchmark。
- 比較微調前後的目標能力及通用能力。

### 進階

- 使用完整 120 組影像偏好資料微調 `HuggingFaceTB/SmolVLM-500M-Instruct`。
- 優化描述影像時避免提及不存在物件。
- 使用 30 張 hold-out 影像評估。
- 比較原圖、過暗、過亮、模糊及低品質 JPEG 條件。
- 量測集合正確率、hallucination rate 與 missing rate。

#### 為何將 120 組 VLM 資料切分

TRL 的 VLM DPO 會在訓練前預先將影像轉換為 image tokens / pixel values。在 Colab 免費 T4 執行環境中，同時預處理 120 組影像會超出系統 RAM。因此實驗先以固定 `SEED=42` 洗牌，再切成 **4 個互斥批次，每批 30 組**，依序更新同一組 LoRA adapter。每批結束後立即釋放 Trainer、Dataset 與 CUDA cache，以降低記憶體使用。

四個批次合計覆蓋全部 120 組資料，沒有丟棄或重複樣本。但每階段會重建 Trainer、optimizer 與 learning-rate scheduler，因此此 sequential chunk DPO 不完全等同於單一 120 筆標準 DPO epoch；此限制也已在實驗報告中說明。

主要檔案：

- `HW7_C_DPO_LLM_VLM.ipynb`
- `llm_dpo_train.json`
- `llm_target_benchmark.json`
- `llm_general_benchmark.json`
- `vlm_dpo_train.json`
- `vlm_dpo_test.json`
- `vlm_dpo_images.zip`

## 評估輸出

三份 notebook 產生逐題結果、摘要 CSV、分類別統計、比較圖及 Markdown 實驗報告。作業 C 另儲存 LLM 與 VLM 的 LoRA adapter。

| 壓縮檔 | 內容 |
|---|---|
| `outputs_A.zip` | LLM 逐題結果、摘要、分類統計、圖表與報告 |
| `outputs_B.zip` | VLM 逐題結果、模型比較、分類統計、圖表與報告 |
| `outputs_C.zip` | LLM/VLM 微調前後結果、分階段訓練記錄、圖表、報告、環境資訊與 LoRA adapters |

## 結果解讀注意事項

- 作業 A 的 NF4 在本次 CUDA peak allocated 量測中未呈現記憶體下降，結果可能受 bitsandbytes 工作區與量測方式影響。
- 作業 C 的 VLM 在原圖條件下 hallucination rate 為 0% → 0%，未觀察到改善；且模型輸出常缺少完整的顏色—形狀配對，因此 exact accuracy 與 missing rate 應一併解讀。

