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


def encode_image(image) -> str:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def judge_batch(client, items: list[dict], model: str) -> list[dict]:
    inputs: list[dict] = [{
        "type": "text",
        "text": (
            "Evaluate every numbered image-caption pair independently. "
            "For each supplied index, return exactly one evaluation with the same index, "
            "an integer score from 1 (incorrect) to 5 (fully correct), and a concise reason. "
            "Do not omit, merge, renumber, or reorder entries."
        ),
    }]
    for item in items:
        inputs.extend([
            {
                "type": "text",
                "text": f"Index {item['index']}; candidate caption: {item['prediction']!r}",
            },
            {
                "type": "image",
                "data": item["image_b64"],
                "mime_type": "image/jpeg",
            },
        ])

    interaction = client.interactions.create(
        model=model,
        input=inputs,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "reason": {"type": "string"},
                            },
                            "required": ["index", "score", "reason"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        },
    )
    output = json.loads(interaction.output_text)["evaluations"]
    expected = {int(item["index"]) for item in items}
    received = {int(item["index"]) for item in output}
    if received != expected or len(output) != len(items):
        raise ValueError(f"Expected indexes {sorted(expected)}, received {sorted(received)}")
    for item in output:
        score = int(item["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"Invalid score {score} for index {item['index']}")
    return output


def save_outputs(records: dict[int, dict], csv_path: Path, summary_path: Path, model: str) -> dict:
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
        "model": model,
        "evaluation_mode": "batched",
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
        f"- 評估方式：批次輸入、逐張獨立評分\n"
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

    source = pd.read_csv(source_path)
    beam = source[source["method"] == "beam5"].copy()
    beam["index"] = beam["index"].astype(int)
    if len(beam) != args.test_size or set(beam["index"]) != set(range(args.test_size)):
        raise SystemExit(f"Expected beam5 indexes 0..{args.test_size - 1}, found {len(beam)} rows")

    records: dict[int, dict] = {}
    if output_path.exists():
        for row in pd.read_csv(output_path).to_dict("records"):
            records[int(row["index"])] = row

    dataset = load_dataset("jxie/flickr8k", split="train").shuffle(seed=args.seed)
    test_start = min(args.train_size, len(dataset) - 1)
    test = dataset.select(range(test_start, min(test_start + args.test_size, len(dataset))))
    client = genai.Client(
        api_key=api_key,
        http_options={"retry_options": {"attempts": 1}},
    )

    pending = [index for index in range(args.test_size) if records.get(index, {}).get("status") != "success"]
    print(f"Already complete: {args.test_size - len(pending)}/{args.test_size}")

    for start in range(0, len(pending), args.batch_size):
        indexes = pending[start : start + args.batch_size]
        batch = []
        for index in indexes:
            row = beam.loc[beam["index"] == index].iloc[0]
            batch.append({
                "index": index,
                "prediction": str(row["prediction"]),
                "image_b64": encode_image(test[index]["image"]),
            })

        for attempt in range(1, args.retries + 1):
            try:
                evaluations = judge_batch(client, batch, args.model)
                by_index = {int(item["index"]): item for item in evaluations}
                for item in batch:
                    result = by_index[item["index"]]
                    records[item["index"]] = {
                        "index": item["index"],
                        "method": "beam5",
                        "prediction": item["prediction"],
                        "gemini_score": int(result["score"]),
                        "gemini_reason": str(result["reason"]),
                        "gemini_model": args.model,
                        "status": "success",
                        "error": "",
                    }
                summary = save_outputs(records, output_path, summary_path, args.model)
                print(
                    f"Batch {indexes[0]}-{indexes[-1]} complete; "
                    f"successful={summary['successful']}/{summary['expected']}"
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"Batch {indexes[0]}-{indexes[-1]} attempt {attempt} failed: {error}")
                lowered = error.lower()
                if "429" in lowered or "rate limit" in lowered or "too_many_requests" in lowered:
                    summary = save_outputs(records, output_path, summary_path, args.model)
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                    raise SystemExit("Gemini rate limit reached. Resume after the quota resets.")
                if attempt < args.retries:
                    time.sleep(args.retry_delay)
        else:
            print("Batch left pending after retry failure; rerun later to resume it.")
            continue
        time.sleep(args.delay)

    summary = save_outputs(records, output_path, summary_path, args.model)
    update_report(root / "REPORT.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["successful"] != summary["expected"]:
        raise SystemExit("Some evaluations remain pending. Run this command again to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    main(parser.parse_args())
