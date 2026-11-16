# 作業完整性稽核

稽核日期：2026-11-11。稽核範圍為軟體、資料結構與評估流程。

| 評分項目 | 狀態 | 可重現證據 |
|---|---|---|
| 作業一：自製特定主題 QA dataset / embeddings | 完成 | 25 題 `rag_qa.json`；固定 384 維 hashed token embedding 與 cosine similarity |
| 作業一：多輪 context、合理回答、reference | 完成 | 代名詞改寫、`[D#]` 引用、語料範圍拒答、單元測試 |
| Chunk size / overlap / top-k | 完成 | 3×3×3（排除無效組合）grid，共 27 組 |
| Prompting technique | 完成 | direct / grounded / cot-plan 三組消融 |
| User Interface | 完成 | Streamlit 四頁 UI，版本固定 1.67.0 |
| 作業二：Video Corpus | 完成 | 實際錄製有聲 49.01 秒與無聲 146 秒 MP4；SHA-256 manifest、speech/visual sidecar |
| 作業二：多輪 VideoRAG | 完成 | speech/visual segments、歷史改寫、時間碼引用、域外拒答 |
| 作業三：RAGAS 評估文字 RAG | 完成 | RAGAS 0.2.15 實跑 25 題；官方 CSV/summary |
| 作業三進階：VideoRAG 評估 | 完成 | modality recall、citation validity、temporal IoU |
| 作業四：AI 論文 Graph 化 | 完成 | Transformer：22 nodes、24 edges、來源 URL |
| 作業四：至少 20 道 benchmark | 完成 | 22 題 |
| 作業四：GraphRAG 回答與評估 | 完成 | 意圖式 relation 查詢／最短路徑對 nodes-only baseline，逐題 fact recall |
| 可重現性 | 完成（本機）；跨主機待 CI/另一台 Docker 主機驗證 | Docker、完整 lock、Ubuntu/Windows CI、10 項測試、SHA-256 manifest |

## 已知限制

- RAGAS ExactMatch 為 0、StringPresence 為 0.2，原因是抽取式答案包含完整句子及引用，而標準答案刻意簡短；這些是有效但不充分的字面指標。
- 自訂 `faithfulness_proxy` 只是引用格式檢查，不等同 RAGAS Faithfulness；報告已明確區分。
- VideoRAG 數值以實際錄製影片的校正 sidecar 計算；MP4 雜湊可驗證提交影片與索引來源一致。
- 域外拒答採可重現的語料範圍規則，不依賴外部分類 API；未涵蓋的新句型仍應透過 benchmark 持續擴充。
