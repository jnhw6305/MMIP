import argparse, csv, json
from pathlib import Path
from .rag import load_rag
from .video import load_video, ingest
from .graph import load_graph
from .evaluate import run_experiments, evaluate_graph, evaluate_video

ROOT=Path(__file__).resolve().parents[2]

def report():
    out=ROOT/"outputs"; summary=json.loads((out/"rag_summary.json").read_text(encoding="utf-8"))
    with (out/"graph_eval.csv").open(encoding="utf-8-sig") as f: graph=list(csv.DictReader(f))
    with (out/"video_eval.csv").open(encoding="utf-8-sig") as f: video=list(csv.DictReader(f))
    graph_recall=sum(float(row["graph_fact_recall"]) for row in graph)/len(graph)
    baseline_recall=sum(float(row["baseline_fact_recall"]) for row in graph)/len(graph)
    text=f"""# 實驗結果報告\n\n## 作業一：多輪 RAG\n\n最佳參數：`{summary['best']}`。平均指標：`{summary['metrics']}`。系統支援多輪代名詞改寫、文件 ID 引用，以及語料範圍檢查；無證據的域外問題會拒答且不顯示無關引用。完整 grid 與 prompt ablation 見 CSV。\n\n## 作業二：VideoRAG\n\n系統匯入實際錄製的 49.01 秒有聲影片與 146 秒無聲操作影片，建立 speech/visual 時間區段索引、多輪查詢與時間碼引用，並拒答影片語料外問題。評估包含 {len(video)} 題，指標為 modality recall、citation validity 與 temporal IoU；影片大小、長度及 SHA-256 記錄於 `video_corpus/corpus_manifest.json`。\n\n## 作業三：評估\n\n文字 RAG 產生 RAGAS schema，並實際以 RAGAS 0.2.15 的 ExactMatch 與 StringPresence 評估；另計算 context recall、answer similarity、faithfulness proxy。字面指標會懲罰附引用的長答案，因此須與檢索及忠實度指標共同解讀。VideoRAG 另評估時間定位與模態覆蓋。\n\n## 作業四：GraphRAG\n\n選用 Vaswani et al. (2017)〈Attention Is All You Need〉，建立 22 nodes、24 relations 與 {len(graph)} 題 benchmark。系統依問題意圖篩選 relation；關係鏈問題使用最短路徑，圖譜外問題則拒答。`graph_eval.csv` 逐題比較 GraphRAG 與僅回傳 seed nodes 的 baseline；目前平均 fact recall 為 {graph_recall:.4f}，baseline 為 {baseline_recall:.4f}。\n\n## 系統限制\n\n抽取式生成與 hash embedding 具備免費及可重現的特性；其語意泛化能力低於大型 embedding 與生成模型。域外拒答使用固定範圍規則，因此遇到全新問法時仍應持續擴充測試。VideoRAG 評估數值取決於所使用的影片語料與時間區段標註。\n"""
    (ROOT/"REPORT.md").write_text(text,encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["all","chat","experiment-rag","ingest-video","evaluate-video","evaluate-graph"]); a=p.parse_args()
    if a.command in ["all","experiment-rag"]: print("best",run_experiments(ROOT))
    if a.command in ["all","evaluate-graph"]: print("graph rows",len(evaluate_graph(ROOT)))
    if a.command in ["all","ingest-video"]: print("video segments",len(ingest(ROOT)))
    if a.command in ["all","evaluate-video"]: print("video rows",len(evaluate_video(ROOT)))
    if a.command=="all": report()
    if a.command=="chat":
        rag=load_rag(ROOT); print("輸入 exit 結束")
        while True:
            q=input("你：").strip()
            if q.lower()=="exit": break
            print("系統：",rag.answer(q)["answer"])
if __name__=="__main__": main()

