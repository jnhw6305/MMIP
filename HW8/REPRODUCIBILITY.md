# 可重現性說明

## Docker（跨 Windows／macOS／Linux）

執行環境為 Docker Desktop 或 Docker Engine：

```sh
docker compose run --rm reproduce
docker compose up ui
```

第一個指令會在固定的 Linux Python 3.13.13 映像中從零安裝鎖定套件，執行所有實驗、官方 RAGAS 與測試，並把結果寫回 `outputs/`。第二個指令在 <http://localhost:8501> 啟動 UI。

## 不使用 Docker

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

Linux / macOS：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
sh scripts/reproduce.sh
```

腳本會建立隔離環境、安裝固定版本、重跑全部實驗、執行官方 RAGAS 非 LLM 指標、執行測試，最後產生 `outputs/reproducibility_manifest.json`。核心實驗不使用隨機數、不呼叫網路服務、不需要 API key。

## 固定項目

- Docker 基底：Python 3.13.13 + Debian Bookworm slim，並固定 multi-architecture image digest `sha256:355b...a03f`。
- `requirements-lock.txt` 固定完整相依樹，而非只固定頂層套件。
- CI 矩陣：乾淨的 Ubuntu / Windows × Python 3.12 / 3.13。
- Streamlit 1.67.0、RAGAS 0.2.15、langchain-core 0.3.35。
- 所有輸入資料納入專案；參數網格固定，排序具有確定性。
- manifest 記錄輸入與主要輸出的 SHA-256，可比較兩次重跑是否一致。

## 驗證層級

1. SHA-256 驗證同一環境連續重跑完全一致。
2. GitHub Actions 在四種乾淨 runner 組合重跑；彙總 job 比較核心檔案雜湊，任何跨平台差異都會使 workflow 失敗。
3. Docker 將 OS、Python 與套件一併隔離，是報告採用的主要可重現環境。

本機稽核環境未安裝 Docker，因此 repository 內的 Dockerfile 已做靜態檢查，但容器建置成功應以 GitHub Actions 或另一台裝有 Docker 的主機結果為準，不冒稱已在本機執行。

## RAGAS 指標

評估使用官方 `ExactMatch` 與 `StringPresence`，兩者不依賴 evaluator LLM，因此結果可精確重現。專案自訂的 context recall、answer token F1、citation proxy 另存於 `rag_summary.json`，並與 RAGAS 指標分開呈現。

## 影片例外

Whisper 轉錄會受模型版本與硬體浮點運算影響；本專案保存人工校正後的實際錄影 sidecar JSON，能以固定輸入重現檢索與評估，且不要求重現環境持有大型 MP4 或具備 GPU。原始影片的大小、長度與 SHA-256 記錄於 `video_corpus/corpus_manifest.json`。
