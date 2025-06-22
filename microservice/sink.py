#!/usr/bin/env python3
"""
Consumes from 'predictions', computes metrics & prints results.
"""

import json, time
import numpy as np
from kafka import KafkaConsumer
from sklearn.metrics import precision_score, recall_score, f1_score

consumer = KafkaConsumer(
    "predictions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode()),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

latencies, y_true, y_pred = [], [], []
count = 0
t0_all = time.time()

try:
    for msg in consumer:
        evt = msg.value
        latencies.append(evt["inference_latency"])
        y_true.append(int(evt["Event_Type"] != "Normal"))
        y_pred.append(int(evt["predicted_anomaly"]))
        count += 1
finally:
    total = time.time() - t0_all
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    avg_lat   = np.mean(latencies)*1000
    p50       = np.percentile(latencies,50)*1000
    p95       = np.percentile(latencies,95)*1000
    tp_thru   = count / total

    print("=== Microservices Pipeline ===")
    print(f"Processed    : {count}")
    print(f"Precision    : {precision:.3f}")
    print(f"Recall       : {recall:.3f}")
    print(f"F1-score     : {f1:.3f}")
    print(f"Latency ms   : avg={avg_lat:.1f}, p50={p50:.1f}, p95={p95:.1f}")
    print(f"Throughput   : {tp_thru:.1f} ev/sec")
