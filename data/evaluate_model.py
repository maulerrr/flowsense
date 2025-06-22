#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate the IsolationForest model on a held-out test set and save
four visualizations to experiment/visualizations/<plot_name>/.

Usage:
    python evaluate_model.py \
      --data_csv data/astana_incidents.csv \
      --model_pkl iforest_model.pkl

Requirements:
    pip install pandas numpy scikit-learn matplotlib
"""

import argparse
import pickle
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

def load_data(path_csv):
    df = pd.read_csv(path_csv, parse_dates=["Timestamp"])
    df["time_frac"] = (
        df["Timestamp"].dt.hour*3600 +
        df["Timestamp"].dt.minute*60 +
        df["Timestamp"].dt.second
    ) / 86400.0
    X = df[["Speed_kmh", "Traffic_Density", "time_frac"]].values
    y = (df["Event_Type"] != "Normal").astype(int).values
    return X, y

def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"✅ Saved {name} to {path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   required=True,
                        help="Path to synthetic traffic CSV")
    parser.add_argument("--model_pkl",  required=True,
                        help="Path to trained IsolationForest pickle")
    args = parser.parse_args()

    # 1) Load model & data
    with open(args.model_pkl, "rb") as f:
        model = pickle.load(f)
    X, y_true = load_data(args.data_csv)

    # 2) Split train/test
    normal_idx = np.where(y_true == 0)[0]
    rng = np.random.default_rng(42)
    train_norm = rng.choice(normal_idx, size=int(0.7*len(normal_idx)), replace=False)
    test_idx = np.setdiff1d(np.arange(len(y_true)), train_norm)

    X_test, y_test = X[test_idx], y_true[test_idx]

    # 3) Scores & preds
    scores = -model.decision_function(X_test)
    threshold = np.percentile(scores, 100*model.contamination)
    y_pred = (scores > threshold).astype(int)

    # 4) Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    f1        = 2*precision*recall/(precision+recall)

    fpr, tpr, _ = roc_curve(y_test, scores); roc_auc = auc(fpr,tpr)
    precs, recs, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recs, precs)

    print("==== Metrics ====")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}")

    # Base output dir
    base = Path("experiment/visualizations")

    # 5) Confusion Matrix
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["Normal","Anomaly"]
    ticks = np.arange(2)
    plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j],
                     ha="center", va="center",
                     color="white" if cm[i,j]>cm.max()/2 else "black")
    save_figure(fig, base/"confusion_matrix", "confusion_matrix")

    # 6) ROC Curve
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
    save_figure(fig, base/"roc_curve", "roc_curve")

    # 7) Precision–Recall Curve
    fig = plt.figure()
    plt.plot(recs, precs, label=f"AUC={pr_auc:.3f}")
    plt.title("Precision–Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(loc="lower left")
    save_figure(fig, base/"pr_curve", "pr_curve")

    # 8) Anomaly Score Distribution
    fig = plt.figure()
    plt.hist(scores[y_test==0], bins=50, alpha=0.6, label="Normal")
    plt.hist(scores[y_test==1], bins=50, alpha=0.6, label="Anomaly")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score (higher = more anomalous)"); plt.ylabel("Count"); plt.legend()
    save_figure(fig, base/"score_distribution", "score_distribution")

if __name__ == "__main__":
    main()
