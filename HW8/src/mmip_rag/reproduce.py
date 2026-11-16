import hashlib, json, platform, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TRACKED=["data/corpus.md","data/rag_qa.json","data/paper_graph.json","data/graph_benchmark.json","data/video_benchmark.json","video_corpus/corpus_manifest.json","video_corpus/sidecars/silent_operation.json","video_corpus/sidecars/spoken_transformer.json","outputs/rag_grid.csv","outputs/prompt_ablation.csv","outputs/rag_summary.json","outputs/graph_eval.csv","outputs/video_eval.csv","outputs/video_index.json","outputs/ragas_dataset.jsonl","outputs/ragas_official.csv","outputs/ragas_official_summary.json"]

def digest(path):
    """Hash semantic content so OS line endings do not create false mismatches."""
    if path.suffix==".json":
        value=json.loads(path.read_text(encoding="utf-8"))
        payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    elif path.suffix==".jsonl":
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload="\n".join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":")) for row in rows).encode("utf-8")
    else:
        payload=path.read_text(encoding="utf-8").replace("\r\n","\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def package_version(name):
    try: return version(name)
    except PackageNotFoundError: return "not installed"

def main():
    subprocess.run([sys.executable,"-m","src.mmip_rag.cli","all"],cwd=ROOT,check=True)
    manifest={"python":sys.version,"platform":platform.platform(),"algorithm":"sha256","files":{x:digest(ROOT/x) for x in TRACKED}}
    manifest["packages"]={"ragas":package_version("ragas"),"streamlit":package_version("streamlit")}
    (ROOT/"outputs/reproducibility_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Wrote outputs/reproducibility_manifest.json")

if __name__=="__main__": main()
