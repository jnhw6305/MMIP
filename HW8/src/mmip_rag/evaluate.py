import csv, json, re
from pathlib import Path
from .rag import load_rag, tokens
from .graph import load_graph
from .video import load_video

def f1(a,b):
    x=tokens(a); y=tokens(b)
    if not x or not y:return 0.0
    common=sum(min(x.count(t),y.count(t)) for t in set(x)); p=common/len(x); r=common/len(y)
    return 2*p*r/(p+r) if p+r else 0.0

def fact_recall(answer, expected):
    a=set(tokens(answer)); e=set(tokens(expected))
    return len(a & e)/len(e) if e else 0.0

def evidence_recall(refs, expected): return float(any(e.lower() in r["title"].lower() for e in expected for r in refs))

def evaluate_rag(root,size=420,overlap=80,k=3,prompt="grounded"):
    root=Path(root); qa=json.loads((root/"data/rag_qa.json").read_text(encoding="utf-8")); rag=load_rag(root,size,overlap,k); rows=[]
    for item in qa:
        out=rag.answer(item["q"],prompt); recall=evidence_recall(out["references"],item["e"]); sim=f1(out["answer"],item["a"])
        faith=float(all((f'[{r["id"]}]' in out["answer"] or prompt=="direct") for r in out["references"][:1]))
        rows.append({"question":item["q"],"answer":out["answer"],"ground_truth":item["a"],"contexts":[r["text"] for r in out["references"]],"context_recall":recall,"answer_similarity":sim,"faithfulness_proxy":faith})
    return rows

def mean(rows,key): return sum(float(r[key]) for r in rows)/len(rows)

def run_experiments(root):
    root=Path(root); out=root/"outputs"; out.mkdir(exist_ok=True); grid=[]
    for size in [240,420,650]:
      for overlap in [0,60,120]:
       if overlap>=size: continue
       for k in [1,3,5]:
        rows=evaluate_rag(root,size,overlap,k); grid.append({"chunk_size":size,"overlap":overlap,"top_k":k,"context_recall":mean(rows,"context_recall"),"answer_similarity":mean(rows,"answer_similarity"),"score":(mean(rows,"context_recall")+mean(rows,"answer_similarity"))/2})
    write_csv(out/"rag_grid.csv",grid); prompts=[]
    for p in ["direct","grounded","cot-plan"]:
        rows=evaluate_rag(root,prompt=p); prompts.append({"prompt":p,"context_recall":mean(rows,"context_recall"),"answer_similarity":mean(rows,"answer_similarity"),"faithfulness_proxy":mean(rows,"faithfulness_proxy")})
    write_csv(out/"prompt_ablation.csv",prompts)
    best=max(grid,key=lambda x:x["score"]); final=evaluate_rag(root,best["chunk_size"],best["overlap"],best["top_k"])
    with (out/"ragas_dataset.jsonl").open("w",encoding="utf-8") as f:
        for r in final:f.write(json.dumps({"user_input":r["question"],"response":r["answer"],"retrieved_contexts":r["contexts"],"reference":r["ground_truth"]},ensure_ascii=False)+"\n")
    (out/"rag_summary.json").write_text(json.dumps({"best":best,"metrics":{"context_recall":mean(final,"context_recall"),"answer_similarity":mean(final,"answer_similarity"),"faithfulness_proxy":mean(final,"faithfulness_proxy")}},ensure_ascii=False,indent=2),encoding="utf-8")
    return best

def evaluate_graph(root):
    root=Path(root); g=load_graph(root); bench=json.loads((root/"data/graph_benchmark.json").read_text(encoding="utf-8")); rows=[]
    for q,a in bench:
        o=g.query(q); answer=o["answer"]; baseline=" ".join(g.seeds(q))
        gf=fact_recall(answer,a); bf=fact_recall(baseline,a)
        rows.append({"question":q,"expected":a,"graph_answer":answer,"baseline_nodes_only":baseline,"graph_fact_recall":gf,"baseline_fact_recall":bf,"gain":gf-bf,"path_count":len(o["paths"])})
    write_csv(root/"outputs/graph_eval.csv",rows); return rows

def evaluate_video(root):
    root=Path(root); v=load_video(root); tests=json.loads((root/"data/video_benchmark.json").read_text(encoding="utf-8")); rows=[]
    for item in tests:
        q=item["q"]; m=item["modality"]
        o=v.answer(q); refs=o["references"]; rows.append({"question":q,"expected_modality":m,"modality_recall":float(any(r.get("modality")==m for r in refs)),"citation_validity":float(all(r["end"]>=r["start"] for r in refs)),"segment_count":len(refs)})
        best_iou=0.0
        for r in refs:
            inter=max(0,min(r["end"],item["end"])-max(r["start"],item["start"])); union=max(r["end"],item["end"])-min(r["start"],item["start"])
            best_iou=max(best_iou,inter/union if union else 0)
        rows[-1]["temporal_iou"]=best_iou
    write_csv(root/"outputs/video_eval.csv",rows); return rows

def write_csv(path,rows):
    path.parent.mkdir(exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

