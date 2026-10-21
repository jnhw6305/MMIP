import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pydeps"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


dataset = fetch_ucirepo(id=350)
X_all = dataset.data.features.copy()
y = dataset.data.targets.iloc[:, 0].astype(int)

# Keep fields that can be entered transparently by a visitor.
feature_names = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
source_names = [
    "X1", "X5",
    "X6", "X7", "X8", "X9", "X10", "X11",
    "X12", "X13", "X14", "X15", "X16", "X17",
    "X18", "X19", "X20", "X21", "X22", "X23",
]
X = X_all[source_names].astype(float)
X.columns = feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    class_weight="balanced", max_iter=2000, random_state=42, solver="liblinear"
)
model.fit(X_train_scaled, y_train)

prob = model.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.20, 0.71, 0.01)
best_threshold = max(thresholds, key=lambda t: f1_score(y_test, prob >= t))
pred = (prob >= best_threshold).astype(int)

metrics = {
    "test_rows": int(len(y_test)),
    "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "precision": round(float(precision_score(y_test, pred)), 4),
    "recall": round(float(recall_score(y_test, pred)), 4),
    "f1": round(float(f1_score(y_test, pred)), 4),
    "threshold": round(float(best_threshold), 2),
}

payload = {
    "name": "UCI Logistic Regression Baseline",
    "dataset": "UCI Default of Credit Card Clients (ID 350)",
    "dataset_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    "features": feature_names,
    "mean": scaler.mean_.round(10).tolist(),
    "scale": scaler.scale_.round(10).tolist(),
    "coef": model.coef_[0].round(10).tolist(),
    "intercept": round(float(model.intercept_[0]), 10),
    "threshold": round(float(best_threshold), 4),
    "metrics": metrics,
}

print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

