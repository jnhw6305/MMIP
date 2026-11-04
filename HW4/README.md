# MMIP Quiz 1–4：時間序列、ViT 與 Image Captioning

本專案完整對應作業要求，採用 PyTorch / Hugging Face，所有資料切分、模型、評估與輸出皆可重現。

## 作業對照

| 題目 | 實作 | 資料集 | 輸出 |
|---|---|---|---|
| Quiz 1 | 三種資料集準備與切分 | IMDb、CIFAR-10、Flickr8k | `data/`（執行時下載） |
| Quiz 2 基礎 | RNN 文字分類與測試推論 | IMDb | accuracy、precision、recall、F1、預測 CSV |
| Quiz 2 進階 | LSTM 與 RNN 比較 | IMDb | `results/sentiment/summary.json` |
| Quiz 3 基礎 | 微調 ViT 並推論 | CIFAR-10 | accuracy、macro-AUC、預測 CSV |
| Quiz 3 進階 | 微調 ResNet，和 ViT 比較 | CIFAR-10 | `results/vision/summary.json` |
| Quiz 4 基礎 | 微調 BLIP、產生描述；Gemini 判讀 | Flickr8k | captions、Gemini 分數與理由 |
| Quiz 4 進階 | BLEU、BERTScore、時間與品質比較 | Flickr8k | `results/caption/summary.json` |

## 安裝

請使用 Python 3.10–3.12（不建議 3.13；部分 Windows `pyarrow` wheel 可能無法載入）。CUDA GPU 可大幅縮短訓練時間。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python verify_env.py
```

## 執行

快速驗證（小型子集、1 epoch）：

```bash
python run_all.py --smoke
```

Windows 若看到 `DLL load failed while importing _compute`，請安裝 Python 3.12 x64 後重建 `.venv`；這是 Python/pyarrow 二進位環境問題，不是模型程式錯誤。

正式實驗：

```bash
python run_all.py
```

若要執行 Gemini 影像內容評審，先設定金鑰（未設定時會保留其他結果並標示 skipped）：

```bash
# PowerShell
$env:GEMINI_API_KEY="你的金鑰"
python src/captioning.py --gemini
```

個別執行：

```bash
python src/sentiment.py --epochs 5
python src/vision.py --epochs 5
python src/captioning.py --epochs 3 --gemini
python src/make_report.py
```

每支程式支援 `--help`。正式模式預設使用可控的資料子集，參數 `--train-size`、`--test-size` 可擴大到完整資料。

## 方法說明

- **公平比較**：同一任務的模型使用相同資料切分、批次大小、epoch 與 seed。
- **Macro-AUC**：對每類執行 one-vs-rest ROC-AUC，再取未加權平均。
- **Captioning 比較**：比較 greedy decoding 與 beam search，記錄平均延遲、BLEU-1/4 與 BERTScore F1。
- **Gemini 評估**：將影像與模型描述一併送入 Gemini，要求回傳 1–5 分與理由；屬 LLM-as-a-judge，並非取代人工評分。
- **資料授權**：資料由官方套件/Hub 於執行時下載，不提交資料、模型權重或 API key。

## 專案結構

```text
src/common.py        共用工具、計時、指標與輸出
src/sentiment.py     RNN / LSTM
src/vision.py        ViT / ResNet
src/captioning.py    BLIP、Gemini、BLEU、BERTScore
src/make_report.py   彙整 Markdown 報告
run_all.py           一鍵執行全部作業
tests/test_metrics.py
```

## 結果解讀

執行後請查看 `REPORT.md` 及 `results/`。`REPORT.md` 由實際 JSON 結果產生，避免手動抄寫造成錯誤。不同硬體、套件版本與資料抽樣會造成數值差異；請勿把尚未實際執行的數字當成實驗結果。
