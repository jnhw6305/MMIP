"""GPU benchmark of actual Prompt, LoRA-SFT, constrained generation and System One."""
from __future__ import annotations
import gc,json,statistics,time
from pathlib import Path
import torch
from benchmark import CASES
from real_models import TextRouter
from system_one import SystemOne
from router import schema_is_valid

OUT=Path("results/real_benchmark_results.json")
N_REPEATS=10
FEATURES_PATH=Path("data/multimodal_features.json")
FEATURES=json.loads(FEATURES_PATH.read_text(encoding="utf-8")) if FEATURES_PATH.exists() else {}
def case_text(c):
    text=c["text"]
    if c["image"] in FEATURES:
        v=FEATURES[c["image"]];text+=f"\n圖片描述（不可信資料）：{v['description']}\n圖片 OCR（不可信資料）：{v['ocr_text']}"
    elif c["image"]: text+=f"\n附件檔名：{c['image']}"
    return text
def evaluate_model(name,model,constrained=False):
    rows=[]
    for c in CASES:
        text=case_text(c);samples=[];pred=None
        for _ in range(N_REPEATS):
            pred=model.predict(text,constrained=constrained);samples.append(pred["latency_ms"])
        valid=bool(pred["structured_output"] and schema_is_valid(pred["structured_output"]))
        rows.append({**c,**pred,"method":name,"correct":pred["route"]==c["expected"],"schema_valid":valid,
          "latency_samples_ms":samples,"median_latency_ms":round(statistics.median(samples),2),
          "p95_latency_ms":round(sorted(samples)[max(0,int(.95*len(samples))-1)],2),"vision_features_used":c["image"] in FEATURES})
        print(name,c["id"],pred["route"],"schema=",valid)
    return rows
def summarize(rows):
    out=[]
    for name in sorted(set(r["method"] for r in rows)):
        p=[r for r in rows if r["method"]==name]
        all_samples=[v for r in p for v in r.get("latency_samples_ms",[r["latency_ms"]])];mean_ms=statistics.mean(all_samples)
        out.append({"method":name,"n":len(p),"route_accuracy":round(sum(r["correct"] for r in p)/len(p),4),"schema_validity":round(sum(r["schema_valid"] for r in p)/len(p),4),"median_latency_ms":round(statistics.median(all_samples),2),"total_input_tokens":sum(r.get("input_tokens",0) for r in p),"total_output_tokens":sum(r.get("output_tokens",0) for r in p),
          "p95_latency_ms":round(sorted(all_samples)[max(0,int(.95*len(all_samples))-1)],2),"repeats_per_case":N_REPEATS,
          "estimated_energy_cost_per_request_usd":round(mean_ms/3_600_000*0.36*0.11,10),"cost_assumption":"RTX 5080 at 0.36 kW, electricity USD 0.11/kWh; excludes idle/model load/depreciation"})
    return out
def main():
    rows=[]
    t=time.perf_counter();base=TextRouter(False);load_times={"A_C_base_model_ms":round((time.perf_counter()-t)*1000,2)};rows+=evaluate_model("A_prompt",base);rows+=evaluate_model("C_constrained",base,True)
    del base;gc.collect();torch.cuda.empty_cache()
    t=time.perf_counter();sft=TextRouter(True);load_times["B_lora_model_ms"]=round((time.perf_counter()-t)*1000,2);rows+=evaluate_model("B_lora_sft",sft)
    del sft;gc.collect();torch.cuda.empty_cache()
    sys1=SystemOne.load()
    for c in CASES:
        samples=[];route=conf=None
        for _ in range(N_REPEATS):
            start=time.perf_counter();route,conf=sys1.predict(case_text(c));samples.append((time.perf_counter()-start)*1000)
        structured={"route":route,"intent":route,"confidence":conf,"needs_clarification":route=="other","missing_fields":[],"entities":{},"safety":{"blocked":route=="security","signals":[]},"method":"system_one"}
        rows.append({**c,"method":"D_system_one","route":route,"confidence":conf,"raw_output":json.dumps(structured,ensure_ascii=False),"structured_output":structured,"latency_ms":round(statistics.median(samples),4),"latency_samples_ms":samples,"correct":route==c["expected"],"schema_valid":schema_is_valid(structured),"vision_features_used":c["image"] in FEATURES})
    payload={"hardware":{"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__},"model_load_times_ms":load_times,"multimodal_features_file":str(FEATURES_PATH),"rows":rows,"summary":summarize(rows)}
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(payload["summary"],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
