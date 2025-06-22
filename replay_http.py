#!/usr/bin/env python3
"""
replay_http.py

Reads a CSV and POSTs each row to the microservice ingestion endpoint:
  http://localhost:5001/ingest

Usage:
  python replay_http.py --input data/traffic_data/astana_synthetic_data.csv \
                       --mode fixed --rate 500
"""

import argparse, time, json
import requests
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--mode",   choices=["realtime","fixed"], default="fixed")
    p.add_argument("--rate",   type=float, help="events/sec for fixed mode")
    args = p.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    url = "http://localhost:5001/ingest"
    interval = None if args.mode=="realtime" else 1.0/args.rate
    prev_ts = None

    print(f"▶ Posting to {url} at {args.mode} rate {args.rate}…")
    for _, row in df.iterrows():
        evt = row.to_dict()
        evt["Timestamp"] = evt["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        # timing control
        if args.mode=="realtime" and prev_ts is not None:
            delta = (row.Timestamp - prev_ts).total_seconds()
            if delta>0: time.sleep(delta)
        elif interval:
            time.sleep(interval)

        # send HTTP
        resp = requests.post(url, json=evt)
        if resp.status_code != 202:
            print("❌ ", resp.status_code, resp.text)

        prev_ts = row.Timestamp

    print("✅ HTTP replay complete.")

if __name__=="__main__":
    main()
