#!/usr/bin/env python3
"""
Flask API: POST /event → runs decision_function + threshold.
"""

from flask import Flask, request, jsonify
import pickle, time, pandas as pd

# load model + threshold
with open("iforest_model.pkl","rb") as f:
    data      = pickle.load(f)
    model     = data["model"]
    threshold = data["threshold"]

app = Flask(__name__)

def extract_features(evt):
    ts = pd.to_datetime(evt["Timestamp"])
    time_frac = (ts.hour*3600 + ts.minute*60 + ts.second)/86400.0
    return [[
        float(evt["Speed_kmh"]),
        float(evt["Traffic_Density"]),
        time_frac
    ]]

@app.route("/event", methods=["POST"])
def infer_event():
    evt = request.json
    X = extract_features(evt)
    t0 = time.time()
    score = -model.decision_function(X)[0]
    latency_ms = (time.time() - t0)*1000
    is_anom = bool(score > threshold)

    return jsonify({
        "anomaly":      is_anom,
        "inference_ms": latency_ms,
        "input_event":  {
            "Speed_kmh":       float(evt["Speed_kmh"]),
            "Traffic_Density": float(evt["Traffic_Density"]),
            "Timestamp":       evt["Timestamp"]
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
