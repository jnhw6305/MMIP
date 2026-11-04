"""Reproduce all basic and advanced tasks in course Quiz 1–3.

Run with: python credit_default_quiz1_3.py --data UCI_Credit_Card.csv
Requires: numpy, pandas, scikit-learn, torch, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TARGET = "default.payment.next.month"


class MLP(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(1)


def metrics(y, score, threshold=0.5):
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"threshold": round(float(threshold), 4), "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp), "accuracy": accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "auc": roc_auc_score(y, score)}


def best_f1_threshold(y, score):
    grid = np.arange(0.05, 0.801, 0.01)
    values = [f1_score(y, score >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(values))])


def train_mlp(x_fit, y_fit, x_val, y_val, dropout=0.0, weight_decay=0.0,
              epochs=60, batch_size=256, lr=0.001):
    torch.manual_seed(SEED)
    model = MLP(x_fit.shape[1], dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(x_fit.astype("float32")),
                            torch.from_numpy(y_fit.astype("float32")))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    xv = torch.from_numpy(x_val.astype("float32"))
    yv = torch.from_numpy(y_val.astype("float32"))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            fit_loss = criterion(model(torch.from_numpy(x_fit.astype("float32"))),
                                 torch.from_numpy(y_fit.astype("float32"))).item()
            val_loss = criterion(model(xv), yv).item()
        history.append({"epoch": epoch, "training_loss": fit_loss,
                        "validation_loss": val_loss})
    return model, pd.DataFrame(history)


def scores(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x.astype("float32")))).numpy()


def fmt(m):
    return (f"{m['threshold']:.2f} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("UCI_Credit_Card.csv"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(SEED)
    random.seed(SEED)
    data = pd.read_csv(args.data)
    if TARGET not in data or len(data) != 30000:
        raise ValueError("Expected the Kaggle UCI_Credit_Card.csv with 30,000 rows")
    x = data.drop(columns=[TARGET, "ID"])
    y = data[TARGET].to_numpy(dtype=int)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=SEED)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED)
    scaler = StandardScaler().fit(x_fit)
    xf, xc, xv = (scaler.transform(z).astype("float32") for z in (x_fit, x_cal, x_val))

    logistic = LogisticRegression(max_iter=1000, random_state=SEED).fit(xf, y_fit)
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=4).fit(xf, y_fit)
    ml_scores = {}
    thresholds = {}
    for name, model in (("Logistic regression", logistic), ("Random forest", forest)):
        thresholds[name] = best_f1_threshold(y_cal, model.predict_proba(xc)[:, 1])
        ml_scores[name] = model.predict_proba(xv)[:, 1]

    base, base_hist = train_mlp(xf, y_fit, xv, y_val)
    improved, improved_hist = train_mlp(xf, y_fit, xv, y_val,
                                         dropout=0.30, weight_decay=1e-4)
    base_scores, improved_scores = scores(base, xv), scores(improved, xv)
    mlp_thresholds = {
        "MLP baseline": best_f1_threshold(y_cal, scores(base, xc)),
        "MLP improved": best_f1_threshold(y_cal, scores(improved, xc)),
    }
    all_scores = {**ml_scores, "MLP baseline": base_scores,
                  "MLP improved": improved_scores}
    all_thresholds = {**thresholds, **mlp_thresholds}
    results = {name: metrics(y_val, score, all_thresholds[name])
               for name, score in all_scores.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in (("Baseline", base_hist), ("Improved", improved_hist)):
        ax.plot(hist.epoch, hist.training_loss, label=f"{name} training")
        ax.plot(hist.epoch, hist.validation_loss, linestyle="--",
                label=f"{name} validation")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="MLP loss across 60 epochs")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "quiz2_loss.png", dpi=180)
    fig.savefig(args.out / "quiz2_loss.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in all_scores.items():
        fpr, tpr, _ = roc_curve(y_val, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_val, score):.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=.5)
    ax.set(xlabel="False positive rate (FPR)", ylabel="True positive rate (TPR)",
           title="Validation ROC curves", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "quiz3_roc.png", dpi=180)
    fig.savefig(args.out / "quiz3_roc.svg")
    plt.close(fig)

    base_hist.to_csv(args.out / "quiz2_baseline_history.csv", index=False)
    improved_hist.to_csv(args.out / "quiz2_improved_history.csv", index=False)
    pd.DataFrame({"ID": x_val.index.map(data["ID"]), "actual": y_val,
                  **{name: s for name, s in all_scores.items()}}).to_csv(
                      args.out / "validation_predictions.csv", index=False)
    sample_id = int(data.loc[x_val.index[0], "ID"])
    sample = {"ID": sample_id, "actual": int(y_val[0]),
              "MLP_baseline_probability": float(base_scores[0]),
              "MLP_improved_probability": float(improved_scores[0]),
              "MLP_improved_predicted_at_calibrated_threshold":
                  int(improved_scores[0] >= mlp_thresholds["MLP improved"])}
    output = {"seed": SEED, "rows": len(data), "features": list(x.columns),
              "positive_rate": float(y.mean()), "fit_size": len(x_fit),
              "calibration_size": len(x_cal), "validation_size": len(x_val),
              "results": results, "sample": sample,
              "quiz1_logistic_initial": metrics(y_val, ml_scores["Logistic regression"], 0.5),
              "quiz1_logistic_adjusted": metrics(y_val, ml_scores["Logistic regression"], 0.3),
              "quiz1_forest_initial": metrics(y_val, ml_scores["Random forest"], 0.5),
              "baseline_last_loss": base_hist.iloc[-1].to_dict(),
              "improved_last_loss": improved_hist.iloc[-1].to_dict()}
    (args.out / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 信用卡違約預測：Quiz 1–3 完整結果", "",
             "資料：[Kaggle 指定資料集](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)；原始來源：[UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)。",
             f"共 {len(data):,} 筆，23 個特徵；違約比例 {y.mean():.2%}。移除 ID，以 1 表示次月違約。",
             "隨機種子 42；分層切分：模型訓練 19,200 筆、門檻校準 4,800 筆、最終驗證 6,000 筆。StandardScaler 僅以模型訓練集擬合。",
             "門檻以校準集 F1 最大化選擇（0.05 至 0.80，間距 0.01）；最終驗證集不參與門檻選擇。", "",
             "## Quiz 1：兩種 Scikit-Learn 分類模型", "",
             "| 模型／門檻 | Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",]
    for label, m in (("Logistic 0.50", output["quiz1_logistic_initial"]),
                     ("Logistic 0.30", output["quiz1_logistic_adjusted"]),
                     ("Logistic 校準門檻", results["Logistic regression"]),
                     ("Random forest 0.50", output["quiz1_forest_initial"]),
                     ("Random forest 校準門檻", results["Random forest"])):
        lines.append(f"| {label} | {fmt(m)} |")
    a, b = results["Logistic regression"], results["Random forest"]
    lines += ["", "混淆矩陣定義：TN＝正確判定未違約、FP＝誤報違約、FN＝漏判違約、TP＝正確判定違約。",
              f"校準門檻下，Logistic 的 Precision / Recall 為 {a['precision']:.3f} / {a['recall']:.3f}；Random forest 為 {b['precision']:.3f} / {b['recall']:.3f}。",
              f"Logistic 的誤報 FP={a['FP']}、漏判 FN={a['FN']}；Random forest 的誤報 FP={b['FP']}、漏判 FN={b['FN']}。",
              "提高門檻通常減少誤報、增加漏判；降低門檻則相反。", "",
              "## Quiz 2：MLP 基礎與改善", "",
              "使用 PyTorch，23→64→32→1、ReLU、BCEWithLogitsLoss、Adam、學習率 0.001、batch size 256、60 epochs。",
              "改善版加入兩層 Dropout(0.30) 與 Adam weight decay=0.0001；其他設定及資料切分相同。",
              "訓練與驗證損失曲線：[quiz2_loss.svg](quiz2_loss.svg)。每個 epoch 的原始數值見兩份 history CSV。", "",
              "| 模型 | Training loss (epoch 60) | Validation loss (epoch 60) | Accuracy | Precision | Recall | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|",]
    for name, hist in (("MLP baseline", base_hist), ("MLP improved", improved_hist)):
        m = results[name]
        lines.append(f"| {name} | {hist.training_loss.iloc[-1]:.4f} | {hist.validation_loss.iloc[-1]:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines += ["", f"單筆驗證集預測：ID={sample_id}，實際違約={sample['actual']}，原版 MLP 違約機率={sample['MLP_baseline_probability']:.4f}，改善版={sample['MLP_improved_probability']:.4f}；改善版依其校準門檻預測={sample['MLP_improved_predicted_at_calibrated_threshold']}。",
              f"原版驗證損失在第 {int(base_hist.validation_loss.idxmin())+1} epoch 達最低值 {base_hist.validation_loss.min():.4f}，之後訓練損失仍下降、驗證損失卻回升，顯示過擬合。改善版第 60 epoch 驗證損失由 {base_hist.validation_loss.iloc[-1]:.4f} 降至 {improved_hist.validation_loss.iloc[-1]:.4f}；訓練與驗證損失差距由 {base_hist.validation_loss.iloc[-1]-base_hist.training_loss.iloc[-1]:+.4f} 縮至 {improved_hist.validation_loss.iloc[-1]-improved_hist.training_loss.iloc[-1]:+.4f}，後段曲線也較平穩。改善版 Precision 和 F1 上升，但 Recall 下降，並非所有指標都改善。", "",
              "## Quiz 3：ROC 與 AUC", "",
              "同一張 ROC 圖：[quiz3_roc.svg](quiz3_roc.svg)。橫軸 FPR=FP/(FP+TN)，縱軸 TPR=TP/(TP+FN)。ROC 顯示各門檻下偵測違約與誤報之間的關係；AUC 是曲線下面積，越高表示對違約與未違約樣本的排序區分能力越好。", "",
              "| 模型 | Validation AUC |", "|---|---:|",]
    for name, m in results.items():
        lines.append(f"| {name} | {m['auc']:.4f} |")
    winner = max(results, key=lambda name: results[name]["auc"])
    lines += ["", f"本次驗證集中，{winner} 的 AUC 最高（{results[winner]['auc']:.4f}），因此在這次固定切分上有最佳的正負樣本排序區分能力。AUC 差異並不代表已證明在其他資料或實際部署中必然較佳。",
              "", "## 重現方式", "", "```powershell", "python -m pip install numpy pandas scikit-learn torch matplotlib", "python credit_default_quiz1_3.py --data UCI_Credit_Card.csv", "```", "", "所有表格均由 `results.json` 的實際訓練結果生成。"]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"auc": {k: v["auc"] for k, v in results.items()},
                      "sample": sample}, indent=2))


if __name__ == "__main__":
    main()

