import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pydeps"))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

SEED = 42
rng = np.random.default_rng(SEED)

dataset = fetch_ucirepo(id=350)
X = dataset.data.features.astype(float).copy()
y = dataset.data.targets.iloc[:, 0].astype(int).copy()

# Exactly reproduce the Week02 19,200 / 4,800 / 6,000 split sizes.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=SEED
)


def add_time_features(frame):
    result = frame.copy()
    bill = frame[[f"X{i}" for i in range(12, 18)]].to_numpy()[:, ::-1]
    paid = frame[[f"X{i}" for i in range(18, 24)]].to_numpy()[:, ::-1]
    delay = frame[[f"X{i}" for i in range(6, 12)]].to_numpy()[:, ::-1]
    t = np.arange(6, dtype=float)
    centered = t - t.mean()
    denom = np.square(centered).sum()
    slope = lambda values: ((values - values.mean(axis=1, keepdims=True)) * centered).sum(axis=1) / denom
    limit = np.maximum(frame["X1"].to_numpy(), 1)
    result["TS_bill_slope"] = slope(bill)
    result["TS_payment_slope"] = slope(paid)
    result["TS_bill_volatility"] = bill.std(axis=1)
    result["TS_avg_utilization"] = bill.mean(axis=1) / limit
    result["TS_avg_payment_ratio"] = np.mean(paid / np.maximum(np.abs(bill), 1), axis=1)
    result["TS_delinquency_count"] = (delay > 0).sum(axis=1)
    result["TS_max_delinquency"] = delay.max(axis=1)
    result["TS_recent_bill_vs_avg"] = bill[:, -1] - bill.mean(axis=1)
    result["TS_delay_slope"] = slope(delay)
    result["TS_recent_payment_ratio"] = paid[:, -1] / np.maximum(np.abs(bill[:, -1]), 1)
    return result


def make_synthetic_minority(features, labels, target_minority_share=0.35):
    minority = features.loc[labels.to_numpy() == 1].copy()
    negative_count = int((labels == 0).sum())
    current_positive = len(minority)
    desired_positive = int(target_minority_share / (1 - target_minority_share) * negative_count)
    count = max(0, desired_positive - current_positive)
    sampled = minority.iloc[rng.integers(0, len(minority), size=count)].copy().reset_index(drop=True)

    # Jitter only continuous monetary/age fields; keep categorical and delinquency states intact.
    sampled["X1"] *= rng.lognormal(0, 0.035, count)
    sampled["X5"] += rng.normal(0, 1.2, count)
    monetary = [f"X{i}" for i in range(12, 24)]
    row_factor = rng.lognormal(0, 0.045, count)
    for column in monetary:
        sampled[column] = sampled[column].to_numpy() * row_factor
        sampled[column] += rng.normal(0, max(float(minority[column].std()) * 0.008, 1), count)

    continuous = ["X1", "X5"] + monetary
    for column in continuous:
        low, high = features[column].quantile([0.005, 0.995])
        sampled[column] = sampled[column].clip(low, high).round()
    sampled["X1"] = sampled["X1"].clip(lower=10000)
    sampled["X5"] = sampled["X5"].clip(18, 100)
    return sampled, pd.Series(np.ones(count, dtype=int), name=labels.name)


def fit_and_evaluate(name, train_x, train_y, cal_x, test_x):
    model = RandomForestClassifier(
        n_estimators=260, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
    )
    model.fit(train_x, train_y)
    cal_prob = model.predict_proba(cal_x)[:, 1]
    thresholds = np.arange(0.05, 0.81, 0.01)
    threshold = float(max(thresholds, key=lambda value: f1_score(y_cal, cal_prob >= value)))
    test_prob = model.predict_proba(test_x)[:, 1]
    pred = test_prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "name": name,
        "threshold": round(threshold, 2),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_prob)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    importances = sorted(zip(train_x.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)[:8]
    return metrics, [{"feature": key, "importance": round(float(value), 4)} for key, value in importances]


synthetic_x, synthetic_y = make_synthetic_minority(X_train, y_train)
X_train_synthetic = pd.concat([X_train, synthetic_x], ignore_index=True)
y_train_synthetic = pd.concat([y_train.reset_index(drop=True), synthetic_y], ignore_index=True)

train_time = add_time_features(X_train)
cal_time = add_time_features(X_cal)
test_time = add_time_features(X_test)
synthetic_time = add_time_features(synthetic_x)
train_time_synthetic = pd.concat([train_time, synthetic_time], ignore_index=True)

experiments = []
importance = {}
for name, train_x, train_y, cal_x, test_x in [
    ("Week02 RF baseline", X_train, y_train, X_cal, X_test),
    ("RF + time features", train_time, y_train, cal_time, test_time),
    ("RF + synthetic data", X_train_synthetic, y_train_synthetic, X_cal, X_test),
    ("RF + time + synthetic", train_time_synthetic, y_train_synthetic, cal_time, test_time),
]:
    metrics, top_features = fit_and_evaluate(name, train_x, train_y, cal_x, test_x)
    experiments.append(metrics)
    importance[name] = top_features

real_minority = X_train.loc[y_train.to_numpy() == 1].reset_index(drop=True)
numeric = ["X1", "X5"] + [f"X{i}" for i in range(12, 24)]
ks_values = [ks_2samp(real_minority[col], synthetic_x[col]).statistic for col in numeric]
real_corr = real_minority[numeric].corr().fillna(0)
synthetic_corr = synthetic_x[numeric].corr().fillna(0)
corr_mae = float(np.mean(np.abs(real_corr.to_numpy() - synthetic_corr.to_numpy())))
duplicates = float(synthetic_x.merge(real_minority.drop_duplicates(), how="inner").shape[0] / max(len(synthetic_x), 1))

output = {
    "dataset": {"rows": 30000, "features": 23, "default_rate": round(float(y.mean()), 4)},
    "split": {"train": len(X_train), "calibration": len(X_cal), "test": len(X_test)},
    "synthetic": {
        "method": "Conditional bootstrap with bounded monetary jitter",
        "rows": len(synthetic_x),
        "minority_before": round(float(y_train.mean()), 4),
        "minority_after": round(float(y_train_synthetic.mean()), 4),
        "mean_ks": round(float(np.mean(ks_values)), 4),
        "max_ks": round(float(np.max(ks_values)), 4),
        "correlation_mae": round(corr_mae, 4),
        "exact_match_rate": round(duplicates, 4),
    },
    "experiments": experiments,
    "feature_importance": importance,
}

path = os.path.join(os.path.dirname(__file__), "final_experiment_results.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(output, handle, ensure_ascii=False, indent=2)
print(json.dumps(output, ensure_ascii=False, indent=2))

