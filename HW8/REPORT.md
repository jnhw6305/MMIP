# 實驗結果報告

## 作業一：多輪 RAG

最佳參數：`{'chunk_size': 240, 'overlap': 0, 'top_k': 3, 'context_recall': 1.0, 'answer_similarity': 0.12867826979602723, 'score': 0.5643391348980136}`。平均指標：`{'context_recall': 1.0, 'answer_similarity': 0.12867826979602723, 'faithfulness_proxy': 0.96}`。系統支援多輪代名詞改寫、文件 ID 引用，以及語料範圍檢查；無證據的域外問題會拒答且不顯示無關引用。完整 grid 與 prompt ablation 見 CSV。

## 作業二：VideoRAG

系統匯入實際錄製的 49.01 秒有聲影片與 146 秒無聲操作影片，建立 speech/visual 時間區段索引、多輪查詢與時間碼引用，並拒答影片語料外問題。評估包含 4 題，指標為 modality recall、citation validity 與 temporal IoU；影片大小、長度及 SHA-256 記錄於 `video_corpus/corpus_manifest.json`。

## 作業三：評估

文字 RAG 產生 RAGAS schema，並實際以 RAGAS 0.2.15 的 ExactMatch 與 StringPresence 評估；另計算 context recall、answer similarity、faithfulness proxy。字面指標會懲罰附引用的長答案，因此須與檢索及忠實度指標共同解讀。VideoRAG 另評估時間定位與模態覆蓋。

## 作業四：GraphRAG

選用 Vaswani et al. (2017)〈Attention Is All You Need〉，建立 22 nodes、24 relations 與 22 題 benchmark。系統依問題意圖篩選 relation；關係鏈問題使用最短路徑，圖譜外問題則拒答。`graph_eval.csv` 逐題比較 GraphRAG 與僅回傳 seed nodes 的 baseline；目前平均 fact recall 為 0.9924，baseline 為 0.3232。

## 系統限制

抽取式生成與 hash embedding 具備免費及可重現的特性；其語意泛化能力低於大型 embedding 與生成模型。域外拒答使用固定範圍規則，因此遇到全新問法時仍應持續擴充測試。VideoRAG 評估數值取決於所使用的影片語料與時間區段標註。
