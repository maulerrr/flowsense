#!/usr/bin/env python3
"""
replay.py

Replays a CSV into either:
  • Monolith: HTTP POST → localhost:5000/event (with optional concurrency)
  • Microservice: Kafka topic 'raw-events' → consumes 'predictions'

Supports fixed-rate or real-time replay, multiple rates, choice of architecture,
and compares both.

Usage:
  python replay.py --input data/traffic_data/astana_synthetic_data.csv \
                   --mode fixed --rates 5000,10000,15000,30000 \
                   --arch both --concurrency 10

Outputs:
  experiment/visualizations/experiment_results/metrics.csv
  experiment/visualizations/experiment_results/
      ├─ f1_vs_rate.png
      ├─ latency_vs_rate.png
      ├─ throughput_vs_rate.png
      └─ precision_recall_vs_rate.png
"""
import argparse, time, json
from pathlib import Path
import concurrent.futures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from kafka import KafkaProducer, KafkaConsumer
from sklearn.metrics import precision_score, recall_score, f1_score

def run_monolith(events, mode, rate, concurrency, url="http://localhost:5000/event"):
    n = len(events)
    interval = None if mode=="realtime" else 1.0/rate
    prev_ts = None

    def send_request(evt):
        payload = evt.copy()
        payload["Timestamp"] = evt["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=5)
            r.raise_for_status()
            resp = r.json()
            anomaly = bool(resp.get("anomaly", False))
            infer_ms = float(resp.get("inference_ms", (time.time()-t0)*1000))
        except Exception:
            anomaly = False
            infer_ms = (time.time() - t0) * 1000.0
        return evt["Event_Type"], anomaly, infer_ms

    # schedule tasks with pacing
    futures = []
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for evt in events:
            futures.append(executor.submit(send_request, evt))
            # pacing
            if mode=="realtime" and prev_ts:
                delta = (evt["Timestamp"] - prev_ts).total_seconds()
                if delta>0: time.sleep(delta)
            elif interval:
                # subtract the tiny scheduling overhead
                time.sleep(interval)
            prev_ts = evt["Timestamp"]

    total_time = time.time() - t_start

    # collect results
    y_true, y_pred, latencies = [], [], []
    for fut in futures:
        evt_type, anomaly, infer_ms = fut.result()
        y_true.append(1 if evt_type!="Normal" else 0)
        y_pred.append(1 if anomaly else 0)
        latencies.append(infer_ms)

    precision  = precision_score(y_true, y_pred) if y_true else 0.0
    recall     = recall_score(y_true, y_pred)    if y_true else 0.0
    f1         = f1_score(y_true, y_pred)        if y_true else 0.0
    avg_lat    = np.mean(latencies)              if latencies else 0.0
    throughput = n / total_time if total_time>0 else 0.0

    return {"precision":precision,
            "recall":   recall,
            "f1":       f1,
            "avg_latency_ms":avg_lat,
            "throughput_ev_s":throughput}

def run_microservice(events, mode, rate, timeout_s=5.0):
    n = len(events)
    producer = KafkaProducer(
      bootstrap_servers="localhost:9092",
      value_serializer=lambda v: json.dumps(v).encode(),
      linger_ms=5, batch_size=32*1024
    )
    consumer = KafkaConsumer(
      "predictions",
      bootstrap_servers="localhost:9092",
      auto_offset_reset="latest",
      enable_auto_commit=True,
      value_deserializer=lambda m: json.loads(m.decode())
    )
    consumer.subscribe(["predictions"])

    y_true, y_pred, latencies = [], [], []
    interval = None if mode=="realtime" else 1.0/rate
    prev_ts = None
    t_pub_start = time.time()

    # produce
    for evt in events:
        payload = evt.copy()
        payload["Timestamp"] = evt["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        t_req = time.time()
        producer.send("raw-events", payload)

        if mode=="realtime" and prev_ts:
            delta = (evt["Timestamp"]-prev_ts).total_seconds()
            if delta>0: time.sleep(delta)
        elif interval:
            elapsed = time.time() - t_req
            to_sleep = interval - elapsed
            if to_sleep>0: time.sleep(to_sleep)

        prev_ts = evt["Timestamp"]

    producer.flush()
    t_pub_end = time.time()

    # consume
    count = 0
    last_recv = time.time()
    while count < n and (time.time() - last_recv) < timeout_s:
        recs = consumer.poll(timeout_ms=500)
        if not recs: continue
        for tp,msgs in recs.items():
            for msg in msgs:
                ev = msg.value
                y_true.append(1 if ev.get("Event_Type")!="Normal" else 0)
                y_pred.append(1 if ev.get("predicted_anomaly") else 0)
                latencies.append(ev.get("inference_latency",0.0)*1000)
                count += 1
                last_recv = time.time()
                if count>=n: break

    pub_dur = t_pub_end - t_pub_start
    precision  = precision_score(y_true,y_pred) if y_true else 0.0
    recall     = recall_score(y_true,y_pred)    if y_true else 0.0
    f1         = f1_score(y_true,y_pred)        if y_true else 0.0
    avg_lat    = np.mean(latencies)             if latencies else 0.0
    throughput = n/pub_dur if pub_dur>0 else 0.0

    return {"precision":precision,
            "recall":   recall,
            "f1":       f1,
            "avg_latency_ms":avg_lat,
            "throughput_ev_s":throughput}

def plot_comparison(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = df.arch.unique()

    def line_plot(col, ylabel, fname):
        fig,ax = plt.subplots()
        for a in archs:
            sub = df[df.arch==a]
            ax.plot(sub.rate, sub[col], '-o', label=a)
        ax.set_xlabel("Replay Rate (events/sec)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Replay Rate")
        ax.legend()
        fig.savefig(out_dir/fname, bbox_inches="tight")

    line_plot("f1",             "F1 Score",                   "f1_vs_rate.png")
    line_plot("avg_latency_ms", "Avg Inference Latency (ms)", "latency_vs_rate.png")
    line_plot("throughput_ev_s","Throughput (events/sec)",   "throughput_vs_rate.png")

    # Precision & Recall combined
    fig,ax = plt.subplots()
    for a in archs:
        sub = df[df.arch==a]
        ax.plot(sub.rate, sub.precision, '--o', label=f"{a} Precision")
        ax.plot(sub.rate, sub.recall,    '-.s', label=f"{a} Recall")
    ax.set_xlabel("Replay Rate (events/sec)")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Replay Rate")
    ax.legend()
    fig.savefig(out_dir/"precision_recall_vs_rate.png", bbox_inches="tight")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--mode", choices=["realtime","fixed"], default="fixed")
    p.add_argument("--rate", type=float)
    p.add_argument("--rates")
    p.add_argument("--arch", choices=["monolith","microservice","both"], default="both")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max concurrent HTTP requests for monolith")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    events = df.to_dict(orient="records")

    if args.rates:
        rates = [float(r) for r in args.rates.split(",")]
    else:
        if not args.rate:
            p.error("must specify --rate or --rates")
        rates = [args.rate]

    records = []
    for r in rates:
        print(f"\n▶ Rate = {r} events/sec")
        if args.arch in ("monolith","both"):
            print("  • Monolith …")
            m = run_monolith(events, args.mode, r, args.concurrency)
            records.append(dict(arch="monolith", rate=r, **m))
            print(f"    ⇒ F1={m['f1']:.3f}, Lat={m['avg_latency_ms']:.1f}ms, Thr={m['throughput_ev_s']:.1f} ev/s")
        if args.arch in ("microservice","both"):
            print("  • Microservice …")
            m = run_microservice(events, args.mode, r)
            records.append(dict(arch="microservice", rate=r, **m))
            print(f"    ⇒ F1={m['f1']:.3f}, Lat={m['avg_latency_ms']:.1f}ms, Thr={m['throughput_ev_s']:.1f} ev/s")

    out_dir = Path("experiment/visualizations/experiment_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dfm = pd.DataFrame(records)
    dfm.to_csv(out_dir/"metrics.csv", index=False)
    plot_comparison(dfm, out_dir)
    print(f"\n✅ Saved metrics & plots under {out_dir.resolve()}")

if __name__=="__main__":
    main()
