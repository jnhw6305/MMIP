# MMIP HW8：完整 RAG / VideoRAG / GraphRAG 專案

本專案逐項對應課程第 72–75 頁的全部基礎與進階要求。核心模式只使用 Python 標準函式庫；Streamlit 與 RAGAS 使用固定版本，Whisper 僅為錄影後的選配轉錄工具。

## HW8 作業導覽

- [完整作業報告](REPORT.md)
- [評分要求逐項稽核](AUDIT.md)
- [跨主機重現性說明](REPRODUCIBILITY.md)
- [繳交檔案清單](SUBMISSION_MANIFEST.md)
- [Streamlit 介面](app.py)
- [有聲與無聲 Video Corpus](video_corpus/raw/)
- [實驗輸出與 Benchmark](outputs/)
- [自動化測試](tests/)

## 快速開始（Windows PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1
```

Linux / macOS：

```sh
sh scripts/start_app.sh
```

跨主機執行方式：

```sh
docker compose run --rm reproduce
docker compose up ui
```

UI（需先 `pip install streamlit`）：

```powershell
streamlit run app.py
```

完整輸出位於 `outputs/`。Video Corpus 位於 `video_corpus/raw/`，匯入與評估指令如下：

```powershell
python -m src.mmip_rag.cli ingest-video
python -m src.mmip_rag.cli evaluate-video
```

## 評分規準對照

| 作業 | 要求 | 實作與證據 |
|---|---|---|
| 一 基礎 | 自製 QA dataset / embeddings | `data/rag_qa.json`（25 題）、`data/corpus.md`、`src/mmip_rag/rag.py`、`outputs/ragas_dataset.jsonl` |
| 一 基礎 | 多輪 context、回答、reference | CLI / Streamlit，多輪代名詞改寫，回答附 `[D#]` |
| 一 進階 | chunk / overlap / top-k | `experiment-rag` 全組合搜尋，`outputs/rag_grid.csv` |
| 一 進階 | prompting technique | direct / grounded / CoT-plan 消融，`outputs/prompt_ablation.csv` |
| 一 進階 | UI | `app.py` 四分頁介面 |
| 二 基礎 | 有說話、無說話操作影片 | 實際錄製的 speech/visual MP4、校正 sidecar、影片 SHA-256 manifest 與時間區段索引 |
| 二 基礎 | 多輪 VideoRAG | 時間感知檢索、對話改寫、`[V# mm:ss-mm:ss]` 引用 |
| 三 基礎 | RAGAS 評估 | 官方 RAGAS ExactMatch / StringPresence + 本機 context recall、faithfulness proxy、answer similarity |
| 三 進階 | VideoRAG 評估 | temporal IoU、segment recall、modality coverage、引用有效率 |
| 四 基礎 | 論文 Graph 化 | Attention Is All You Need，`data/paper_graph.json` |
| 四 基礎 | 至少 20 QA | `data/graph_benchmark.json`（22 題） |
| 四 基礎 | GraphRAG + benchmark | 問題意圖篩選、關係查詢／最短路徑與 nodes-only baseline 的 fact recall 對照，`outputs/graph_eval.csv` |

## 設計選擇

- Embedding：可重現的 384 維 hashed token vector 搭配 cosine similarity，不用金鑰。
- 生成：抽取式 grounded answer；只回答語料涵蓋的 Transformer 問題，域外問題明確拒答且不附無關引用。
- GraphRAG：實體/關係三元組、問題意圖篩選、關係查詢與最短路徑；圖譜外問題明確拒答。
- VideoRAG：ASR 字幕與視覺操作描述統一成帶時間碼的 segments，影片語料外問題明確拒答。
- RAGAS：固定 0.2.15，使用官方非 LLM metrics，不依賴模型金鑰。

執行官方 RAGAS metrics：

```powershell
python -m src.mmip_rag.ragas_eval
```

完整重現流程與 SHA-256 驗證記錄於 `REPRODUCIBILITY.md`。

## 專案限制

兩支實際錄製 MP4 與校正後的 speech/visual sidecar 均納入 repository；其大小、長度與 SHA-256 記錄於 `video_corpus/corpus_manifest.json`。其他主機可直接觀看原始影片，亦可只使用固定 sidecar 重現 VideoRAG 檢索與評估，不需重新執行 Whisper 或具備 GPU。

## 來源

- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762
- RAGAS 官方 metrics 文件：https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Microsoft GraphRAG：https://github.com/microsoft/graphrag

