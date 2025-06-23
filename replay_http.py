#!/usr/bin/env python3
"""
replay_http.py

Reads a CSV and POSTs each row to the microservice ingestion endpoint
(http://localhost:5001/ingest) with up to N concurrent workers,
either at a fixed rate or preserving original timestamps.

Usage:
  python replay_http.py --input data/traffic_data/astana_synthetic_data.csv \
                       --mode fixed --rate 500 \
                       [--concurrency 1000]

  python replay_http.py --input data/traffic_data/astana_synthetic_data.csv \
                       --mode realtime \
                       [--concurrency 1000]

Outputs:
  experiment/visualizations/http_replay/metrics.csv
  experiment/visualizations/http_replay/latency_hist.png
  experiment/visualizations/http_replay/latency_over_time.png
  experiment/visualizations/http_replay/throughput_over_time.png
  experiment/visualizations/http_replay/status_codes.png
"""
import argparse, asyncio, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aiohttp

OUT_DIR = Path("experiment/visualizations/http_replay")


async def send_event(session, url, evt, sem, results):
    async with sem:
        t0 = asyncio.get_event_loop().time()
        try:
            async with session.post(url, json=evt, timeout=10) as resp:
                status = resp.status
        except Exception:
            status = 0
        t1 = asyncio.get_event_loop().time()
        latency_ms = (t1 - t0) * 1000
        results.append({
            "timestamp": datetime.utcnow(),
            "latency_ms": latency_ms,
            "status": status
        })


async def run():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True,
                   help="Path to CSV (with Timestamp, Event_Type, etc.)")
    p.add_argument("--mode",       choices=["realtime","fixed"],
                   default="fixed", help="Replay mode")
    p.add_argument("--rate",       type=float,
                   help="events/sec (for fixed mode)")
    p.add_argument("--concurrency", type=int, default=1000,
                   help="Max concurrent HTTP requests")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # pacing
    if args.mode == "fixed":
        if args.rate is None:
            p.error("must specify --rate for fixed mode")
        interval = 1.0 / args.rate
    else:
        interval = None

    url = "http://localhost:5001/ingest"
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prev_ts = None
        for row in df.itertuples(index=False):
            evt = {
                "Event_ID":        row.Event_ID,
                "Timestamp":       row.Timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Vehicle_Type":    row.Vehicle_Type,
                "Speed_kmh":       row.Speed_kmh,
                "Latitude":        row.Latitude,
                "Longitude":       row.Longitude,
                "Event_Type":      row.Event_Type,
                "Severity":        row.Severity,
                "Traffic_Density": row.Traffic_Density
            }

            # pacing
            if args.mode == "realtime" and prev_ts is not None:
                delta = (row.Timestamp - prev_ts).total_seconds()
                if delta > 0:
                    await asyncio.sleep(delta)
            elif interval:
                await asyncio.sleep(interval)

            # fire and forget under semaphore
            asyncio.create_task(send_event(session, url, evt, sem, results))
            prev_ts = row.Timestamp

        # wait for all tasks to finish
        # acquire the full semaphore
        await sem.acquire()
        # release it so the collector tasks can finish
        sem.release()

    # convert to DataFrame
    dfm = pd.DataFrame(results)
    dfm["request_idx"] = np.arange(len(dfm)) + 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(OUT_DIR/"metrics.csv", index=False)
    print(f"✅ Wrote metrics.csv ({len(dfm)} records)")

    # 1) latency histogram
    plt.figure()
    plt.hist(dfm.latency_ms, bins=50)
    plt.title("HTTP Ingest Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.savefig(OUT_DIR/"latency_hist.png", bbox_inches="tight")
    plt.close()

    # 2) latency over time (by request #)
    plt.figure()
    plt.plot(dfm.request_idx, dfm.latency_ms, ".", markersize=2)
    plt.title("Per‐Request Latency Over Sequence")
    plt.xlabel("Request Index")
    plt.ylabel("Latency (ms)")
    plt.savefig(OUT_DIR/"latency_over_time.png", bbox_inches="tight")
    plt.close()

    # 3) cumulative throughput over time
    times = (dfm.timestamp - dfm.timestamp.min()).dt.total_seconds()
    thru = dfm.request_idx / times.clip(lower=1e-3)
    plt.figure()
    plt.plot(times, thru)
    plt.title("Achieved Throughput Over Wall-Clock Time")
    plt.xlabel("Seconds since start")
    plt.ylabel("Throughput (req/sec)")
    plt.savefig(OUT_DIR/"throughput_over_time.png", bbox_inches="tight")
    plt.close()

    # 4) status‐code counts
    counts = dfm.status.value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("HTTP Status Code Counts")
    plt.xlabel("HTTP Status")
    plt.ylabel("Count")
    plt.savefig(OUT_DIR/"status_codes.png", bbox_inches="tight")
    plt.close()

    print(f"✅ Saved visualizations under {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(run())
