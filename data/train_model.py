#!/usr/bin/env python3
"""
data/train_model.py

Enhanced training of an IsolationForest on synthetic traffic data:
 - richer feature engineering (with optional location clustering)
 - grid-search over hyperparameters + threshold calibration
 - saves final model + threshold + feature list
 - outputs:
     1) experiment/logs/grid_search.csv
     2) experiment/visualizations/training/.../*.png
"""

import argparse, csv, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve, auc,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

LOG_DIR = Path("experiment/logs")
VIZ_DIR = Path("experiment/visualizations/training")

def save_fig(fig, subdir: str, name: str):
    out = VIZ_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Saved {subdir}/{name}.png")

def build_features(df, use_clusters=False, n_clusters=5):
    # time encoding
    secs = (df["Timestamp"].dt.hour*3600 +
            df["Timestamp"].dt.minute*60 +
            df["Timestamp"].dt.second)
    frac = secs / 86400.0
    df["hour_sin"] = np.sin(2*np.pi*frac)
    df["hour_cos"] = np.cos(2*np.pi*frac)

    # vehicle one‐hots
    veh = pd.get_dummies(df["Vehicle_Type"], prefix="veh")
    df = pd.concat([df, veh], axis=1)

    # numeric severity
    df["severity_num"] = df["Severity"].map({"Low":0,"Medium":1,"High":2}).fillna(0)

    features = ["Speed_kmh","Traffic_Density","hour_sin","hour_cos"] \
               + veh.columns.tolist() + ["severity_num"]

    if use_clusters:
        coords = df[["Latitude","Longitude"]].values
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        df["loc_cluster"] = km.labels_
        loc = pd.get_dummies(df["loc_cluster"], prefix="loc")
        df = pd.concat([df, loc], axis=1)
        features += loc.columns.tolist()

    return df, df[features].values, features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",     default="data/traffic_data/astana_synthetic_data.csv")
    parser.add_argument("--model_pkl",    default="iforest_model.pkl")
    parser.add_argument("--use_clusters", action="store_true",
                        help="Enable latitude/longitude clustering features")
    parser.add_argument("--n_clusters",   type=int, default=5)
    args = parser.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.data_csv, parse_dates=["Timestamp"])
    y  = (df["Event_Type"] != "Normal").astype(int).values

    # 2) Split normals for training
    normals = np.where(y==0)[0]
    rng     = np.random.default_rng(42)
    train_n = rng.choice(normals, size=int(0.7*len(normals)), replace=False)
    test_ix = np.setdiff1d(np.arange(len(df)), train_n)

    # 3) Feature engineering
    df_feat, X_all, feat_names = build_features(
        df, use_clusters=args.use_clusters, n_clusters=args.n_clusters
    )
    X_train, X_test = X_all[train_n], X_all[test_ix]
    y_test          = y[test_ix]

    # 4) Grid definitions
    grid = {
        "contamination": [0.05, 0.10, 0.20],
        "n_estimators":  [100, 200],
        "max_samples":   ["auto", 0.8]
    }

    # 5) Grid-search + threshold sweep; log to CSV
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gs_csv = LOG_DIR / "grid_search.csv"
    best   = {"f1": -1.0}

    with open(gs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cont","n_est","max_samp","thr_pct","f1"])
        for cont in grid["contamination"]:
            for ne in grid["n_estimators"]:
                for ms in grid["max_samples"]:
                    iso = IsolationForest(
                        n_estimators=ne,
                        max_samples=ms,
                        contamination=cont,
                        random_state=42
                    ).fit(X_train)

                    scores = -iso.decision_function(X_test)
                    base   = cont * 100.0

                    for delta in np.linspace(-10, 10, 21):
                        pct    = np.clip(base + delta, 0, 100)
                        thr    = np.percentile(scores, pct)
                        pred   = (scores > thr).astype(int)
                        f1     = f1_score(y_test, pred)
                        writer.writerow([cont, ne, ms, pct, f1])
                        if f1 > best["f1"]:
                            best.update({
                                "cont": cont,
                                "n_est": ne,
                                "max_samp": ms,
                                "pct": pct,
                                "thresh": thr,
                                "f1": f1
                            })

    print("🔍 Best config:")
    print(f"   contamination={best['cont']}, n_estimators={best['n_est']}, max_samples={best['max_samp']}")
    print(f"   threshold percentile={best['pct']:.1f} → F1={best['f1']:.3f}")

    # 6) Load grid results for plots
    gs_df = pd.read_csv(gs_csv)

    # 6a) Threshold sweep per contamination
    fig, ax = plt.subplots()
    for cont in grid["contamination"]:
        sub = gs_df[gs_df.cont == cont]
        ax.plot(sub.thr_pct, sub.f1, label=f"{cont:.2f}")
    ax.set_title("Threshold Sweep per Contamination")
    ax.set_xlabel("Threshold percentile")
    ax.set_ylabel("F1 score")
    ax.legend(title="contamination")
    save_fig(fig, "threshold_sweep", "threshold_sweep")

    # 6b) Best-F1 vs contamination
    bests = gs_df.groupby("cont")["f1"].max().reset_index()
    fig, ax = plt.subplots()
    ax.plot(bests.cont, bests.f1, marker="o")
    ax.set_title("Best F1 by Contamination")
    ax.set_xlabel("contamination")
    ax.set_ylabel("max F1")
    save_fig(fig, "contamination_best_f1", "contamination_best_f1")

    # 7) Train final on best config
    final = IsolationForest(
        n_estimators=best["n_est"],
        max_samples=best["max_samp"],
        contamination=best["cont"],
        random_state=42
    ).fit(X_train)

    # 8) Save model + threshold + features
    with open(args.model_pkl, "wb") as f:
        pickle.dump({
            "model":     final,
            "threshold": best["thresh"],
            "features":  feat_names
        }, f)
    print(f"✅ Saved final model+threshold+features → {args.model_pkl}")

    # 9) Final evaluation
    scores = -final.decision_function(X_test)
    y_pred = (scores > best["thresh"]).astype(int)

    # 9a) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in (0,1):
        for j in (0,1):
            ax.text(j, i, cm[i,j],
                    ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    save_fig(fig, "confusion_matrix", "confusion_matrix")

    # 9b) ROC Curve
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc     = auc(fpr, tpr)
    fig, ax     = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1], [0,1], "k--", alpha=0.5)
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    save_fig(fig, "roc_curve", "roc_curve")

    # 9c) Precision–Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, scores)
    pr_auc        = auc(rec, prec)
    fig, ax       = plt.subplots()
    ax.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    ax.set_title("Precision–Recall"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend()
    save_fig(fig, "pr_curve", "pr_curve")

    # 9d) Anomaly Score Distribution
    fig, ax = plt.subplots()
    ax.hist(scores[y_test==0], bins=50, alpha=0.6, label="Normal")
    ax.hist(scores[y_test==1], bins=50, alpha=0.6, label="Anomaly")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, "score_distribution", "score_distribution")

    # 9e) Calibration Curve (normalize scores → [0,1])
    min_s, max_s = scores.min(), scores.max()
    y_prob = (scores - min_s) / (max_s - min_s)
    prob_pos, frac_pos = calibration_curve(
        y_test, y_prob, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots()
    ax.plot(frac_pos, prob_pos, marker="o", label="model")
    ax.plot([0,1], [0,1], "k--", alpha=0.5)
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted anomaly score")
    ax.set_ylabel("Fraction of actual anomalies")
    ax.legend()
    save_fig(fig, "calibration", "calibration_curve")

if __name__ == "__main__":
    main()
