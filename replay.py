#!/usr/bin/env python3
"""
replay.py

Replays a CSV into Kafka topic 'raw-events' (real-time or fixed-rate),
then consumes from 'predictions' and computes comparison plots
for multiple replay rates.

Usage:
  # single rate
  python replay.py --input data/traffic_data/astana_synthetic_data.csv \
                   --mode fixed --rate 500

  # multiple rates
  python replay.py --input data/traffic_data/astana_synthetic_data.csv \
                   --mode fixed --rates 500,5000,10000,30000

Outputs:
  experiment/visualizations/experiment_results/metrics.csv
  experiment/visualizations/experiment_results/
      ├─ f1_vs_rate.png
      ├─ latency_vs_rate.png
      ├─ throughput_vs_rate.png
      └─ precision_recall_vs_rate.png
"""
import argparse
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kafka import KafkaProducer, KafkaConsumer
from sklearn.metrics import precision_score, recall_score, f1_score

def run_experiment(events, mode, rate, timeout_s=5.0):
    """
    Publish `events` at given `rate` (events/sec) or real-time,
    then consume exactly len(events) predictions (or until timeout).
    Returns a dict of metrics.
    """
    n_events = len(events)

    # tuned producer for higher throughput
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,       # batch up to 5ms
        batch_size=32 * 1024  # 32 KB
    )

    # consumer starting at 'latest' offset so we only get new predictions
    consumer = KafkaConsumer(
        'predictions',
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
    consumer.subscribe(['predictions'])

    # ----- PRODUCE -----
    print("  → Publishing events…")
    interval = None if mode == 'realtime' else 1.0 / rate
    prev_ts = None
    t_publish_start = time.time()
    for evt in events:
        # convert Timestamp to string for JSON
        ts = evt['Timestamp']
        evt_payload = evt.copy()
        evt_payload['Timestamp'] = ts.strftime("%Y-%m-%d %H:%M:%S")

        t0 = time.time()
        producer.send('raw-events', evt_payload)
        # dynamic sleep
        if mode == 'realtime' and prev_ts is not None:
            delta = (ts - prev_ts).total_seconds()
            if delta > 0:
                time.sleep(delta)
        elif interval is not None:
            elapsed = time.time() - t0
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        prev_ts = ts

    # ensure all messages are sent
    producer.flush()
    t_publish_end = time.time()
    pub_duration = t_publish_end - t_publish_start
    print(f"    → Done publishing {n_events} events in {pub_duration:.2f}s")

    # ----- CONSUME -----
    print("  → Consuming predictions…")
    y_true   = []
    y_pred   = []
    latencies= []
    count    = 0
    last_recv = time.time()

    # keep polling until we've got all or we timeout with no new msgs
    while count < n_events and (time.time() - last_recv) < timeout_s:
        recs = consumer.poll(timeout_ms=500)
        if not recs:
            continue
        for tp, msgs in recs.items():
            for msg in msgs:
                evt = msg.value
                y_true.append(1 if evt.get("Event_Type") != "Normal" else 0)
                y_pred.append(1 if evt.get("predicted_anomaly") else 0)
                latencies.append(evt.get("inference_latency", 0.0))
                count += 1
                last_recv = time.time()
                if count >= n_events:
                    break

    t_consume_end = time.time()
    cons_duration = t_consume_end - t_publish_end
    print(f"    → Received {count} predictions in {cons_duration:.2f}s")

    # metrics
    precision  = precision_score(y_true, y_pred) if y_true else 0.0
    recall     = recall_score(y_true, y_pred)    if y_true else 0.0
    f1         = f1_score(y_true, y_pred)        if y_true else 0.0
    avg_latency= np.mean(latencies)*1000.0 if latencies else 0.0
    throughput = n_events / pub_duration if pub_duration > 0 else 0.0

    return {
        "rate": rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency_ms": avg_latency,
        "throughput_ev_s": throughput
    }

def plot_results(dfm, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) F1 vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.f1, '-o')
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Replay Rate")
    fig.savefig(out_dir/"f1_vs_rate.png", bbox_inches="tight")

    # 2) Latency vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.avg_latency_ms, '-o')
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Avg Inference Latency (ms)")
    ax.set_title("Latency vs Replay Rate")
    fig.savefig(out_dir/"latency_vs_rate.png", bbox_inches="tight")

    # 3) Throughput vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.throughput_ev_s, '-o')
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Achieved Throughput (events/sec)")
    ax.set_title("Throughput vs Replay Rate")
    fig.savefig(out_dir/"throughput_vs_rate.png", bbox_inches="tight")

    # 4) Precision & Recall vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.precision, '-o', label="Precision")
    ax.plot(dfm.rate, dfm.recall,    '-o', label="Recall")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Replay Rate")
    ax.legend()
    fig.savefig(out_dir/"precision_recall_vs_rate.png", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to CSV (with Timestamp, Event_Type, etc.)")
    parser.add_argument("--mode",   choices=["realtime","fixed"], default="fixed",
                        help="Replay mode")
    parser.add_argument("--rate",   type=float,
                        help="events/sec for fixed mode")
    parser.add_argument("--rates",
                        help="comma-separated list of rates (overrides --rate)")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # prepare list of dicts for fast iteration
    records = df.to_dict(orient="records")

    if args.rates:
        rates = [float(r) for r in args.rates.split(",")]
    else:
        if not args.rate:
            parser.error("Must specify --rate or --rates")
        rates = [args.rate]

    results = []
    for r in rates:
        print(f"\n▶ Running replay at {r} events/sec …")
        res = run_experiment(records, args.mode, r)
        print(f"  • Precision={res['precision']:.3f}, "
              f"Recall={res['recall']:.3f}, "
              f"F1={res['f1']:.3f}, "
              f"Latency={res['avg_latency_ms']:.1f}ms, "
              f"Thru={res['throughput_ev_s']:.1f} ev/s")
        results.append(res)

    # Save & plot
    out_dir = Path("experiment/visualizations/experiment_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dfm = pd.DataFrame(results)
    dfm.to_csv(out_dir/"metrics.csv", index=False)
    plot_results(dfm, out_dir)

    print(f"\n✅ Experiment metrics & plots saved under {out_dir.resolve()}")

if __name__ == "__main__":
    main()
