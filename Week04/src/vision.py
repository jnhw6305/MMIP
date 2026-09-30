from __future__ import annotations

import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from common import ROOT, Timer, classification_metrics, device_name, save_json, seed_everything


def build_model(kind, classes=10):
    if kind == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def evaluate(model, loader, device):
    model.eval(); ys, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).softmax(-1).cpu()
            ys.extend(y.tolist()); preds.extend(p.argmax(-1).tolist()); probs.extend(p.tolist())
    return classification_metrics(ys, preds, probs), ys, preds, probs


def run(args):
    seed_everything(args.seed); device = device_name()
    out = ROOT / "results" / "vision"; out.mkdir(parents=True, exist_ok=True)
    tf_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tf_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    full_train = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=tf_train)
    full_test = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=tf_test)
    g = torch.Generator().manual_seed(args.seed)
    train_ids = torch.randperm(len(full_train), generator=g)[:args.train_size].tolist()
    test_ids = torch.randperm(len(full_test), generator=g)[:args.test_size].tolist()
    train_loader = DataLoader(Subset(full_train, train_ids), args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(full_test, test_ids), args.batch_size, num_workers=0)
    results = {"dataset": {"name": "CIFAR-10", "train": len(train_ids), "test": len(test_ids)}}
    for kind in ("vit", "resnet"):
        model = build_model(kind).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        with Timer() as timer:
            for _ in range(args.epochs):
                model.train()
                for x, y in tqdm(train_loader, desc=kind):
                    opt.zero_grad(); loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device)); loss.backward(); opt.step()
        metrics, ys, preds, probs = evaluate(model, test_loader, device); metrics["training_seconds"] = timer.seconds
        results[kind] = metrics
        pd.DataFrame({"index": test_ids, "label": ys, "prediction": preds,
                      "confidence": [max(p) for p in probs]}).to_csv(
            out / f"{kind}_predictions.csv", index=False)
        (ROOT / "models").mkdir(exist_ok=True); torch.save(model.state_dict(), ROOT / "models" / f"{kind}.pt")
    save_json(out / "summary.json", results); return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-size", type=int, default=10000); p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42); run(p.parse_args())
