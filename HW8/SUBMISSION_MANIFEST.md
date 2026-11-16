# MMIP HW8 GitHub 提交清單

## 作業完成狀態

| 作業要求 | 狀態 | 主要證據 |
|---|---|---|
| 多輪文字 RAG、QA dataset、Embedding、回答與 Reference | 完成 | `data/rag_qa.json`、`data/corpus.md`、`src/mmip_rag/rag.py`、`app.py` |
| Chunk Size、Overlap、Top-K 實驗 | 完成 | `outputs/rag_grid.csv`、`outputs/rag_summary.json` |
| Prompting Technique 消融 | 完成 | `outputs/prompt_ablation.csv` |
| User Interface | 完成 | `app.py` |
| 有聲與無聲 Video Corpus | 完成 | `video_corpus/raw/*.mp4`、`video_corpus/sidecars/*.json` |
| 多輪 VideoRAG 與時間碼 Reference | 完成 | `src/mmip_rag/video.py`、`outputs/video_index.json` |
| RAGAS 文字 RAG 評估 | 完成 | `outputs/ragas_official.csv`、`outputs/ragas_official_summary.json` |
| VideoRAG 進階評估 | 完成 | `data/video_benchmark.json`、`outputs/video_eval.csv` |
| AI 論文 Graph 化 | 完成 | `data/paper_graph.json` |
| 至少 20 題 GraphRAG benchmark | 完成（22 題） | `data/graph_benchmark.json` |
| GraphRAG 回答與 benchmark 評估 | 完成 | `src/mmip_rag/graph.py`、`outputs/graph_eval.csv` |
| 跨主機可重現性 | 已建置；待 GitHub Actions 實際執行 | `Dockerfile`、`compose.yaml`、`.github/workflows/reproduce.yml` |

## 必須上傳至 GitHub

- 根目錄程式與文件：`app.py`、`README.md`、`REPORT.md`、`AUDIT.md`、`REPRODUCIBILITY.md`。
- 執行環境：`requirements.txt`、`requirements-lock.txt`、`requirements-video.txt`、`pyproject.toml`、`Dockerfile`、`compose.yaml`。
- 自動重現：`.github/workflows/reproduce.yml`、`scripts/`、`tests/`。
- 核心程式：`src/mmip_rag/`。
- 資料集與 benchmark：`data/`。
- 完整實驗結果：`outputs/`。
- 有聲與無聲影片：`video_corpus/raw/spoken_transformer.mp4`、`video_corpus/raw/silent_operation.mp4`。
- 校正後影片索引資料：`video_corpus/sidecars/`、`video_corpus/corpus_manifest.json`。
- 有聲影片分享畫面：`output/pdf/transformer_spoken_video_screen.pdf`。
- PDF 產生來源：`scripts/create_spoken_video_pdf.py`。

## 不應上傳

- `.venv/`：本機虛擬環境，可由 lock file 重建。
- `__pycache__/`、`.pytest_cache/`、`*.pyc`：執行快取。
- Whisper 模型快取：可重新下載，且不屬於作業成果。
- 個人 API key、token、登入 Cookie 或其他憑證。

## Video Corpus 驗證

| 檔案 | 長度 | SHA-256 |
|---|---:|---|
| `silent_operation.mp4` | 146 秒 | `5cf34589f9573ba3c4682aeed4f1fbe5353ebbd1009d174b533d0e34438d3fd0` |
| `spoken_transformer.mp4` | 49.01 秒 | `cdb64f6df6f997d2f6dc4c89a8acc7fa5a0aeb4395c0712ac420b2a414dbce1a` |

## 最小重現指令

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1
```

Docker：

```sh
docker compose run --rm reproduce
docker compose up ui
```
