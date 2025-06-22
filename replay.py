#!/usr/bin/env python3
"""
replay.py

Replays a CSV into Kafka topic 'raw-events' (real-time or fixed-rate),
collects results from 'predictions', and at the end produces
comparison plots for multiple rates.

Usage:
  # single rate
  python replay.py --input data/astana_synthetic_data.csv --mode fixed --rate 500

  # multiple rates
  python replay.py --input data/astana_synthetic_data.csv --mode fixed \
      --rates 500,5000,10000,30000

Produces:
  experiment/visualizations/experiment_results/metrics.csv
  experiment/visualizations/experiment_results/
      ├─ f1_vs_rate.png
      ├─ latency_vs_rate.png
      ├─ throughput_vs_rate.png
      └─ precision_recall_vs_rate.png
"""
import argparse, time, json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kafka import KafkaProducer, KafkaConsumer
from sklearn.metrics import precision_score, recall_score, f1_score

def run_experiment(df, mode, rate):
    n_events = len(df)
    # -- set up producer & consumer --
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    consumer = KafkaConsumer(
        "predictions",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    # -- publish events --
    prev_ts = None
    interval = None if mode=="realtime" else 1.0/rate
    start_time = time.time()
    for _, row in df.iterrows():
        evt = row.to_dict()
        evt["Timestamp"] = row.Timestamp.strftime("%Y-%m-%d %H:%M:%S")

        if mode=="realtime" and prev_ts is not None:
            delta = (row.Timestamp - prev_ts).total_seconds()
            if delta>0: time.sleep(delta)
        elif interval:
            time.sleep(interval)

        producer.send("raw-events", evt)
        prev_ts = row.Timestamp

    producer.flush()

    # -- consume predictions --
    y_true, y_pred, latencies = [], [], []
    count = 0
    for msg in consumer:
        evt = msg.value
        y_true.append(1 if evt["Event_Type"]!="Normal" else 0)
        y_pred.append(1 if evt["predicted_anomaly"] else 0)
        latencies.append(evt["inference_latency"])
        count += 1
        if count>=n_events:
            break

    end_time = time.time()
    total_sec = end_time - start_time

    # -- compute metrics --
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    avg_lat   = np.mean(latencies)*1000.0            # ms
    throughput= n_events / total_sec                 # ev/sec

    return {
        "rate": rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency_ms": avg_lat,
        "throughput_ev_s": throughput
    }

def plot_results(dfm, out_dir: Path):
    # 1) F1 vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.f1, marker="o")
    ax.set_title("F1 Score vs Replay Rate")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("F1 Score")
    fig.savefig(out_dir/"f1_vs_rate.png", bbox_inches="tight")

    # 2) Latency vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.avg_latency_ms, marker="o")
    ax.set_title("Avg Inference Latency vs Replay Rate")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Avg Latency (ms)")
    fig.savefig(out_dir/"latency_vs_rate.png", bbox_inches="tight")

    # 3) Throughput vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.throughput_ev_s, marker="o")
    ax.set_title("Achieved Throughput vs Replay Rate")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Throughput (events/sec)")
    fig.savefig(out_dir/"throughput_vs_rate.png", bbox_inches="tight")

    # 4) Precision & Recall vs rate
    fig, ax = plt.subplots()
    ax.plot(dfm.rate, dfm.precision, marker="o", label="Precision")
    ax.plot(dfm.rate, dfm.recall,    marker="o", label="Recall")
    ax.set_title("Precision & Recall vs Replay Rate")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Score")
    ax.legend()
    fig.savefig(out_dir/"precision_recall_vs_rate.png", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to CSV (with Timestamp, Event_Type, etc.)")
    parser.add_argument("--mode",   choices=["realtime","fixed"], default="fixed")
    parser.add_argument("--rate",   type=float,
                        help="events/sec (for fixed mode)")
    parser.add_argument("--rates",
                        help="comma-separated list of rates (overrides --rate)")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if args.rates:
        rates = [float(x) for x in args.rates.split(",")]
    else:
        if not args.rate:
            parser.error("must specify --rate or --rates")
        rates = [args.rate]

    results = []
    for r in rates:
        print(f"\n▶ Running replay at {r} events/sec …")
        res = run_experiment(df, args.mode, r)
        print(f"  • Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, "
              f"F1={res['f1']:.3f}, Latency={res['avg_latency_ms']:.1f}ms, "
              f"Throu={res['throughput_ev_s']:.1f} ev/s")
        results.append(res)

    # Save & plot
    out_dir = Path("experiment/visualizations/experiment_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dfm = pd.DataFrame(results)
    dfm.to_csv(out_dir/"metrics.csv", index=False)
    plot_results(dfm, out_dir)
    print(f"\n✅ Experiment metrics & plots saved under {out_dir.resolve()}")

if __name__=="__main__":
    main()
