from __future__ import annotations

import argparse
import base64
import io
import os
from statistics import mean

import pandas as pd
import torch
from bert_score import score as bert_score
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from common import ROOT, Timer, device_name, save_json, seed_everything


def collate(rows, processor):
    images = [r["image"].convert("RGB") for r in rows]
    refs = [r.get("caption_0") or r.get("caption") or r.get("text") for r in rows]
    if isinstance(refs[0], list): refs = [x[0] for x in refs]
    enc = processor(images=images, text=refs, padding=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].masked_fill(enc["input_ids"] == processor.tokenizer.pad_token_id, -100)
    return enc, images, refs


def gemini_judge(image, caption, model_name):
    if not os.getenv("GEMINI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "GEMINI_API_KEY not set",
        }

    import json
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"Judge whether this caption matches the image: {caption!r}. "
        "Give a score from 1 to 5 and a short reason."
    )

    interaction = client.interactions.create(
        model=model_name,
        input=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": image_b64,
                "mime_type": "image/jpeg",
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            },
        },
    )

    return json.loads(interaction.output_text)


def run(args):
    seed_everything(args.seed); device = device_name()
    ds = load_dataset("jxie/flickr8k", split="train").shuffle(seed=args.seed)
    train = ds.select(range(min(args.train_size, len(ds))))
    test_start = min(args.train_size, len(ds) - 1)
    test = ds.select(range(test_start, min(test_start + args.test_size, len(ds))))
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device)
    loader = DataLoader(train, args.batch_size, shuffle=True, collate_fn=lambda x: collate(x, processor))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    with Timer() as train_timer:
        for _ in range(args.epochs):
            model.train()
            for enc, _, _ in tqdm(loader, desc="BLIP"):
                opt.zero_grad(); batch = {k: v.to(device) for k, v in enc.items()}
                loss = model(**batch).loss; loss.backward(); opt.step()
    records = []; methods = {"greedy": {"num_beams": 1}, "beam5": {"num_beams": 5}}
    for method, params in methods.items():
        for i, row in enumerate(test):
            image = row["image"].convert("RGB"); ref = row.get("caption_0") or row.get("caption") or row.get("text")
            if isinstance(ref, list): ref = ref[0]
            inputs = processor(images=image, return_tensors="pt").to(device)
            with Timer() as timer:
                ids = model.generate(**inputs, max_new_tokens=35, **params)
            pred = processor.decode(ids[0], skip_special_tokens=True)
            records.append({"method": method, "index": i, "reference": ref, "prediction": pred,
                            "latency_seconds": timer.seconds})
    smooth = SmoothingFunction().method1
    for r in records:
        ref, pred = r["reference"].split(), r["prediction"].split()
        r["bleu1"] = sentence_bleu([ref], pred, weights=(1,0,0,0), smoothing_function=smooth)
        r["bleu4"] = sentence_bleu([ref], pred, smoothing_function=smooth)
    _, _, f1 = bert_score([r["prediction"] for r in records], [r["reference"] for r in records], lang="en")
    for r, value in zip(records, f1.tolist()): r["bertscore_f1"] = value
    if args.gemini:
        for r in records:
            if r["method"] == "beam5": r["gemini"] = gemini_judge(test[r["index"]]["image"], r["prediction"], args.gemini_model)
    summary = {"dataset": {"name": "Flickr8k", "train": len(train), "test": len(test)},
               "training_seconds": train_timer.seconds, "methods": {}}
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        summary["methods"][method] = {k: mean(r[k] for r in subset)
            for k in ("bleu1", "bleu4", "bertscore_f1", "latency_seconds")}
    out = ROOT / "results" / "caption"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "captions.csv", index=False)
    save_json(out / "summary.json", summary); (ROOT / "models").mkdir(exist_ok=True)
    model.save_pretrained(ROOT / "models" / "blip-finetuned"); processor.save_pretrained(ROOT / "models" / "blip-finetuned")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=200); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gemini", action="store_true"); p.add_argument("--gemini-model", default="gemini-3.7-flash")
    run(p.parse_args())

