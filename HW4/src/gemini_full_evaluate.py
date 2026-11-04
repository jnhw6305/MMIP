from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from google import genai


def judge(client, image, caption: str, model: str) -> dict:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    prompt = (
        "Evaluate whether the candidate caption accurately describes the image. "
        "Return an integer score from 1 (incorrect) to 5 (fully correct), plus a concise reason. "
        f"Candidate caption: {caption!r}"
    )
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": encoded, "mime_type": "image/jpeg"},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            },
        },
    )
    result = json.loads(interaction.output_text)
    score = int(result["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"Gemini returned an invalid score: {score}")
    return {"score": score, "reason": str(result["reason"])}


def save_outputs(records: dict[int, dict], csv_path: Path, summary_path: Path) -> dict:
    rows = [records[index] for index in sorted(records)]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    successful = [row for row in rows if row.get("status") == "success"]
    summary = {
        "expected": 200,
        "attempted": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean_score": (
            sum(float(row["gemini_score"]) for row in successful) / len(successful)
            if successful else None
        ),
        "model": rows[-1]["gemini_model"] if rows else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def update_report(report_path: Path, summary: dict) -> None:
    if summary["successful"] != summary["expected"]:
        return
    marker = "## Quiz 4：Gemini 全測試集評估"
    section = (
        f"{marker}\n\n"
        f"- 評估模型：`{summary['model']}`\n"
        f"- 成功評估：{summary['successful']}/{summary['expected']} 張\n"
        f"- 平均分數：{summary['mean_score']:.4f}/5\n"
        "- 各圖片分數與理由：`results/caption/gemini_full_evaluation.csv`\n"
    )
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    report_path.write_text(text, encoding="utf-8")


def main(args) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    root = Path(__file__).resolve().parents[1]
    caption_dir = root / "results" / "caption"
    source_path = caption_dir / "captions.csv"
    output_path = caption_dir / "gemini_full_evaluation.csv"
    summary_path = caption_dir / "summary_gemini.json"
    report_path = root / "REPORT.md"

    source = pd.read_csv(source_path)
    beam = source[source["method"] == "beam5"].copy()
    if len(beam) != args.test_size or set(beam["index"].astype(int)) != set(range(args.test_size)):
        raise SystemExit(
            f"Expected beam5 indexes 0..{args.test_size - 1}, found {len(beam)} rows"
        )

    records: dict[int, dict] = {}
    if output_path.exists():
        for row in pd.read_csv(output_path).to_dict("records"):
            records[int(row["index"])] = row

    dataset = load_dataset("jxie/flickr8k", split="train").shuffle(seed=args.seed)
    test_start = min(args.train_size, len(dataset) - 1)
    test = dataset.select(range(test_start, min(test_start + args.test_size, len(dataset))))
    # Disable the SDK's hidden retries. With a small daily free-tier quota,
    # one logical evaluation must consume only one HTTP request per attempt.
    client = genai.Client(
        api_key=api_key,
        http_options={"retry_options": {"attempts": 1}},
    )

    for _, row in beam.sort_values("index").iterrows():
        index = int(row["index"])
        if records.get(index, {}).get("status") == "success":
            print(f"[{index + 1}/{args.test_size}] already complete")
            continue

        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                result = judge(client, test[index]["image"], str(row["prediction"]), args.model)
                records[index] = {
                    "index": index,
                    "method": "beam5",
                    "prediction": row["prediction"],
                    "gemini_score": result["score"],
                    "gemini_reason": result["reason"],
                    "gemini_model": args.model,
                    "status": "success",
                    "error": "",
                }
                print(f"[{index + 1}/{args.test_size}] score={result['score']}")
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"[{index + 1}/{args.test_size}] attempt {attempt} failed: {last_error}")
                lowered = last_error.lower()
                if "429" in lowered or "rate limit" in lowered or "too_many_requests" in lowered:
                    summary = save_outputs(records, output_path, summary_path)
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                    raise SystemExit(
                        "Gemini daily rate limit reached. Run this command again after the quota resets."
                    )
                if attempt < args.retries:
                    time.sleep(args.retry_delay * attempt)
        else:
            records[index] = {
                "index": index,
                "method": "beam5",
                "prediction": row["prediction"],
                "gemini_score": "",
                "gemini_reason": "",
                "gemini_model": args.model,
                "status": "error",
                "error": last_error,
            }

        summary = save_outputs(records, output_path, summary_path)
        time.sleep(args.delay)

    summary = save_outputs(records, output_path, summary_path)
    update_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["successful"] != summary["expected"]:
        raise SystemExit("Some evaluations failed. Run this command again to resume them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    # One original request plus at most one application-level retry for 503s.
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    main(parser.parse_args())
