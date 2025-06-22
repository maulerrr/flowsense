#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiment/logs/results.csv")

# throughput vs rate
plt.figure()
for arch in df.arch.unique():
    sub = df[df.arch==arch]
    plt.plot(sub.rate, sub.throughput, label=arch)
plt.xlabel("Injection rate (events/sec)")
plt.ylabel("Processed throughput (events/sec)")
plt.legend()
plt.tight_layout()
plt.savefig("experiment/visualizations/comparison_throughput.png")

# latency vs rate
plt.figure()
for arch in df.arch.unique():
    sub = df[df.arch==arch]
    plt.plot(sub.rate, sub.latency_ms, label=arch)
plt.xlabel("Injection rate")
plt.ylabel("Mean inference latency (ms)")
plt.legend()
plt.tight_layout()
plt.savefig("experiment/visualizations/comparison_latency.png")

# F1-score vs rate
plt.figure()
for arch in df.arch.unique():
    sub = df[df.arch==arch]
    plt.plot(sub.rate, sub.f1, label=arch)
plt.xlabel("Injection rate")
plt.ylabel("F1‐score")
plt.legend()
plt.tight_layout()
plt.savefig("experiment/visualizations/comparison_f1.png")

print("✅ Plots saved under experiment/visualizations/")
