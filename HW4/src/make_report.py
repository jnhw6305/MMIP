import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "results" / name / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    s, v, c = load("sentiment"), load("vision"), load("caption")
    lines = ["# MMIP Quiz 1–4 實驗報告", "", "本報告由實際執行結果自動產生。", ""]
    if s:
        lines += ["## Quiz 1–2：Sentiment Analysis", "", f"IMDb：train={s['dataset']['train']}、test={s['dataset']['test']}", "",
                  "| 模型 | Accuracy | Macro-F1 | 訓練秒數 |", "|---|---:|---:|---:|"]
        for k in ("rnn", "lstm"): lines.append(f"| {k.upper()} | {s[k]['accuracy']:.4f} | {s[k]['macro_f1']:.4f} | {s[k]['training_seconds']:.1f} |")
        lines.append("")
    if v:
        lines += ["## Quiz 3：Vision Transformer", "", f"CIFAR-10：train={v['dataset']['train']}、test={v['dataset']['test']}", "",
                  "| 模型 | Accuracy | Macro-AUC | 訓練秒數 |", "|---|---:|---:|---:|"]
        for k in ("vit", "resnet"): lines.append(f"| {k.upper()} | {v[k]['accuracy']:.4f} | {v[k]['macro_auc']:.4f} | {v[k]['training_seconds']:.1f} |")
        lines.append("")
    if c:
        lines += ["## Quiz 4：Image Captioning", "", f"Flickr8k：train={c['dataset']['train']}、test={c['dataset']['test']}", "",
                  "| 方法 | BLEU-1 | BLEU-4 | BERTScore F1 | 平均延遲(秒) |", "|---|---:|---:|---:|---:|"]
        for k, x in c["methods"].items(): lines.append(f"| {k} | {x['bleu1']:.4f} | {x['bleu4']:.4f} | {x['bertscore_f1']:.4f} | {x['latency_seconds']:.4f} |")
    if not any((s, v, c)): lines += ["尚無結果。請先執行 `python run_all.py --smoke` 或正式實驗。", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()

