from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import ROOT, Timer, classification_metrics, device_name, save_json, seed_everything


class TextDataset(Dataset):
    def __init__(self, rows, vocab, max_len=256):
        self.rows, self.vocab, self.max_len = rows, vocab, max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        ids = [self.vocab.get(w, 1) for w in row["text"].lower().split()[: self.max_len]]
        length = max(1, len(ids))
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(row["label"]), torch.tensor(length)


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, kind="rnn", emb=128, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        block = nn.RNN if kind == "rnn" else nn.LSTM
        self.encoder = block(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        output, _ = self.encoder(self.embedding(x))
        batch_index = torch.arange(x.size(0), device=x.device)
        last_valid = output[batch_index, lengths.to(x.device) - 1]
        return self.head(last_valid)


def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []

    with torch.no_grad():
        for x, y, lengths in loader:
            logits = model(x.to(device), lengths)
            p = logits.softmax(-1).cpu()

            ys.extend(y.tolist())
            preds.extend(p.argmax(-1).tolist())
            probs.extend(p[:, 1].tolist())

    metrics = classification_metrics(ys, preds)
    return metrics, ys, preds, probs

def run(args):
    seed_everything(args.seed); device = device_name()
    out = ROOT / "results" / "sentiment"; out.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("imdb")
    train = list(raw["train"].shuffle(seed=args.seed).select(range(min(args.train_size, len(raw["train"])))))
    test = list(raw["test"].shuffle(seed=args.seed).select(range(min(args.test_size, len(raw["test"])))))
    counts = Counter(w for r in train for w in r["text"].lower().split())
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(args.vocab_size - 2))}
    train_loader = DataLoader(TextDataset(train, vocab), args.batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(test, vocab), args.batch_size)
    results = {"dataset": {"name": "IMDb", "train": len(train), "test": len(test)}}
    for kind in ("rnn", "lstm"):
        model = SentimentModel(len(vocab) + 2, kind).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr); loss_fn = nn.CrossEntropyLoss()
        with Timer() as timer:
            for _ in range(args.epochs):
                model.train()

                for x, y, lengths in tqdm(train_loader, desc=kind):
                    opt.zero_grad()
                    logits = model(x.to(device), lengths)
                    loss = loss_fn(logits, y.to(device))
                    loss.backward()
                    opt.step()
        metrics, ys, preds, probs = evaluate(model, test_loader, device)
        metrics["training_seconds"] = timer.seconds
        results[kind] = metrics
        pd.DataFrame({"text": [r["text"] for r in test], "label": ys,
                      "prediction": preds, "positive_probability": probs}).to_csv(
            out / f"{kind}_predictions.csv", index=False)
        (ROOT / "models").mkdir(exist_ok=True); torch.save(model.state_dict(), ROOT / "models" / f"{kind}.pt")
    save_json(out / "summary.json", results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-size", type=int, default=10000); p.add_argument("--test-size", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())
