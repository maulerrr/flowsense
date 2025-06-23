#!/usr/bin/env python3
"""
monolith_api.py

Flask API that loads the trained model+threshold+feature-list and offers
POST /event for real-time inference on single traffic events.
"""

from flask import Flask, request, jsonify
import pickle, time
import pandas as pd
import numpy as np

# 1) Load the full deployment dict
with open("iforest_model.pkl", "rb") as f:
    pkg = pickle.load(f)
model      = pkg["model"]
threshold  = pkg["threshold"]
feat_names = pkg["features"]  # list of strings

# Precompute the expected one-hot columns for vehicle types
veh_cols = [c for c in feat_names if c.startswith("veh_")]

app = Flask(__name__)

def extract_features(evt: dict):
    """
    Build a 1×d feature vector matching feat_names exactly.
    Unknown 'loc_*' cluster columns will be zero-filled.
    """
    # start with zeros
    row = {name: 0.0 for name in feat_names}

    # 1) speed & density
    row["Speed_kmh"]       = float(evt.get("Speed_kmh", 0.0))
    row["Traffic_Density"] = float(evt.get("Traffic_Density", 0.0))

    # 2) time frac → sin/cos
    ts = pd.to_datetime(evt["Timestamp"])
    secs = ts.hour*3600 + ts.minute*60 + ts.second
    tfrac = secs / 86400.0
    row["hour_sin"] = np.sin(2*np.pi*tfrac)
    row["hour_cos"] = np.cos(2*np.pi*tfrac)

    # 3) severity
    sev = evt.get("Severity", "Low")
    sev_map = {"Low":0, "Medium":1, "High":2}
    row["severity_num"] = float(sev_map.get(sev, 0))

    # 4) veh type one-hot
    vt = evt.get("Vehicle_Type", "")
    col = f"veh_{vt}"
    if col in veh_cols:
        row[col] = 1.0

    # 5) leave any loc_* or other columns as zero

    # build numpy array
    X = np.array([row[name] for name in feat_names]).reshape(1, -1)
    return X

@app.route("/event", methods=["POST"])
def infer_event():
    evt = request.json

    # Extract features
    X = extract_features(evt)

    # Run inference and time it
    t0 = time.time()
    raw_score = -model.decision_function(X)[0]  # higher → more anomalous
    latency_ms = (time.time() - t0) * 1000.0

    is_anom = bool(raw_score > threshold)

    # Return the inference result with the original input values
    return jsonify({
        "anomaly":      is_anom,
        "score":        raw_score,
        "threshold":    threshold,
        "inference_ms": latency_ms,
        "input_event":  {
            "Speed_kmh":       float(evt.get("Speed_kmh", 0.0)),
            "Traffic_Density": float(evt.get("Traffic_Density", 0.0)),
            "Timestamp":       evt.get("Timestamp")
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(port=5000)
