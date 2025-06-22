#!/usr/bin/env python3
"""
microservice/detector.py

Consumes from 'raw-events', reconstructs the full feature vector,
runs model.decision_function + thresholding, publishes anomalies to 'predictions'.
"""

import pickle
import json
import time
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# Severity → numeric map
_SEV_MAP = {"Low":0, "Medium":1, "High":2}

def load_artifacts(path="iforest_model.pkl"):
    """Load dict with 'model', 'threshold', 'features'."""
    with open(path, "rb") as f:
        art = pickle.load(f)
    return art["model"], art["threshold"], art["features"]

def make_consumer(topic="raw-events", servers="localhost:9092"):
    return KafkaConsumer(
        topic,
        bootstrap_servers=servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

def make_producer(topic="predictions", servers="localhost:9092"):
    return KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

def build_feature_vector(evt: dict, feat_names: list):
    """
    Given an incoming event dict, compute a single-row feature vector
    with exactly the columns in feat_names (in order).
    """
    ts = pd.to_datetime(evt["Timestamp"])
    secs = ts.hour*3600 + ts.minute*60 + ts.second
    frac = secs / 86400.0

    # base features
    d = {
        "Speed_kmh":        float(evt.get("Speed_kmh", 0)),
        "Traffic_Density":  float(evt.get("Traffic_Density", 0)),
        "hour_sin":         np.sin(2*np.pi*frac),
        "hour_cos":         np.cos(2*np.pi*frac),
        "severity_num":     _SEV_MAP.get(evt.get("Severity","Low"), 0)
    }
    # one-hot vehicle dummies
    v = evt.get("Vehicle_Type","Unknown")
    for feat in feat_names:
        if feat.startswith("veh_"):
            d[feat] = 1.0 if feat == f"veh_{v}" else 0.0

    # fill any missing (e.g. loc clusters if you used them) with 0
    row = [ d.get(col, 0.0) for col in feat_names ]
    return np.array(row).reshape(1, -1)

def process_stream(model, threshold, feat_names, consumer, producer):
    for msg in consumer:
        evt = msg.value

        X = build_feature_vector(evt, feat_names)
        t0 = time.time()
        score = -model.decision_function(X)[0]   # anomaly‐score
        latency = time.time() - t0

        is_anom = bool(score > threshold)
        evt.update({
            "anomaly_score":     float(score),
            "predicted_anomaly": is_anom,
            "inference_latency": float(latency)
        })
        producer.send("predictions", evt)

def main():
    # 1) load model + threshold + feature list
    model, threshold, feat_names = load_artifacts("iforest_model.pkl")
    print(f"Loaded model; threshold={threshold:.4f}; #features={len(feat_names)}")

    # 2) Kafka setup
    consumer = make_consumer("raw-events", "localhost:9092")
    producer = make_producer("predictions", "localhost:9092")

    # 3) Consume → infer → publish
    process_stream(model, threshold, feat_names, consumer, producer)

if __name__ == "__main__":
    main()
