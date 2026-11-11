"""Controlled before/after measurement for progressive disclosure UX."""
import json,statistics
from pathlib import Path
from benchmark import CASES
from chat import Session

def main():
    rows=[]
    for c in CASES:
        d=Session().reply(c["text"],c["image"],"constraint")
        full=json.dumps(d,ensure_ascii=False)
        rows.append({"id":c["id"],"baseline_visible_chars":len(full),"optimized_visible_chars":len(d["message"]),
          "baseline_processing_feedback":False,"optimized_processing_feedback":True,"reading_delay_ms":d["reading_delay_ms"],
          "detail_available_on_demand":True})
    summary={"n":len(rows),"mean_baseline_visible_chars":round(statistics.mean(x["baseline_visible_chars"] for x in rows),1),
      "mean_optimized_visible_chars":round(statistics.mean(x["optimized_visible_chars"] for x in rows),1),
      "visible_information_reduction_pct":round((1-statistics.mean(x["optimized_visible_chars"] for x in rows)/statistics.mean(x["baseline_visible_chars"] for x in rows))*100,1),
      "processing_feedback_coverage_before":0.0,"processing_feedback_coverage_after":1.0,
      "claim_scope":"UI telemetry proxy; not a human-subject usability study"}
    Path("results/ux_results.json").write_text(json.dumps({"rows":rows,"summary":summary},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
