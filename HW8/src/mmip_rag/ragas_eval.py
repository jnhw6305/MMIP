"""Deterministic official RAGAS evaluation; no API key or judge LLM required."""
import json
from pathlib import Path

def main():
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import ExactMatch, StringPresence
    except ImportError as e:
        raise SystemExit("Missing evaluation dependencies. Install requirements-lock.txt.") from e
    root=Path(__file__).resolve().parents[2]
    samples=[]
    for line in (root/"outputs/ragas_dataset.jsonl").read_text(encoding="utf-8").splitlines():
        item=json.loads(line)
        samples.append({"user_input":item["user_input"],"response":item["response"],"reference":item["reference"]})
    dataset=EvaluationDataset.from_list(samples)
    result=evaluate(dataset,metrics=[ExactMatch(),StringPresence()])
    frame=result.to_pandas()
    frame.to_csv(root/"outputs/ragas_official.csv",index=False,encoding="utf-8-sig",lineterminator="\n")
    summary={"ragas_version":__import__("ragas").__version__,"rows":len(frame),"exact_match":float(frame["exact_match"].mean()),"string_present":float(frame["string_present"].mean()),"note":"Deterministic non-LLM RAGAS metrics"}
    (root/"outputs/ragas_official_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
