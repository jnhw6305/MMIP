# MMIP HW7：多模態多輪電商客服分流系統

公開互動展示：<https://mmip-nova-care-hw7.legendaryhunter87.chatgpt.site>

本專案依照課程第 9 週作業三個部分完成，包含快速展示模式與真實 GPU 實驗：

1. 多模態、多輪對話與結構化分流。
2. 比較 Prompt、LoRA SFT、Constraint Decoding、System One 四種節點。
3. 聊天 UI、閱讀時間 UX、離題偵測與文字／圖片攻擊防禦。

## 執行

```powershell
python server.py
```

瀏覽器開啟 <http://127.0.0.1:8000>。這是免模型的快速展示模式。

真實圖片像素分析模式需 Python 3.12、NVIDIA GPU，並先安裝：

```powershell
python -m pip install -r requirements-ml.txt
$env:ENABLE_ML='1'
python server.py
```

## 測試與實驗

```powershell
python -m unittest discover -s tests -v
python benchmark.py
python data/build_dataset.py
python system_one.py
python train_lora.py
python benchmark_real.py
python data/create_vlm_cases.py
python benchmark_vlm.py
python ux_experiment.py
```

## 可重現性

- `ml_config.py` 固定模型名稱與 seed；資料生成、System One 訓練與 LoRA 訓練均使用固定 seed。
- `data/train.jsonl`、`data/validation.jsonl` 與 12 筆正式 Benchmark 分離保存。
- `artifacts/lora-router/training_metrics.json` 保留每個 epoch 的 train／validation loss；大型 adapter、tokenizer 與中間 checkpoint 不納入 Git，請執行 `python train_lora.py` 重建。
- 正式比較保留每次推論延遲、模型原始輸出、硬體與 PyTorch 版本，不只保存彙總表。
- 提交前已執行 8 項單元測試、Python 編譯檢查及 JSON 解析檢查；不同 GPU、驅動與套件版本可能造成些微延遲或浮點差異。

真實實驗輸出位於 `results/real_benchmark_results.json`、`results/vlm_results.json`、`results/ux_results.json`；LoRA 權重及 training metrics 位於 `artifacts/lora-router/`。

本次正式結果（12 案例、每案 10 次）為：A Prompt 33.33% route accuracy／25% schema validity、B LoRA SFT 75%／100%、C Constraint 33.33%／100%、D System One 58.33%／100%。圖片安全集含 13 張實際影像（7 正常、6 攻擊）。

## 重要說明

`benchmark.py` 保留為免 GPU 的快速基線；正式結果請以 `benchmark_real.py` 為準。正式實驗 A 使用 Qwen2.5-0.5B-Instruct zero/few-shot 推論，B 使用實際 LoRA SFT adapter，C 使用 generation-time prefix trie constrained decoding，D 使用訓練後的 character n-gram Naive Bayes。多模態管線由 BLIP 圖像描述與 EasyOCR 圖中文字安全閘道構成。成本欄位若出現估計值會明確標示，不冒充實際帳單。

完整作業說明、架構圖、Spec、Benchmark 與分析見 [REPORT.md](REPORT.md)；逐項繳交核對見 [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)。
