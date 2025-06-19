#!/usr/bin/env python3
"""
train_model.py

Train an IsolationForest on synthetic traffic data, save the model.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

# 1) Load synthetic data
df = pd.read_csv("astana_incidents.csv", parse_dates=["Timestamp"])

# 2) Feature engineering
df["time_frac"] = (
    df["Timestamp"].dt.hour * 3600 +
    df["Timestamp"].dt.minute * 60 +
    df["Timestamp"].dt.second
) / 86400.0

X = df[["Speed_kmh", "Traffic_Density", "time_frac"]]
y = (df["Event_Type"] != "Normal").astype(int)

# 3) Train on Normal events only
normal_mask = df["Event_Type"] == "Normal"
X_train = X[normal_mask]

contamination = y.sum() / len(y)  # fraction of anomalies in entire set

model = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
)
model.fit(X_train)

# 4) Save model
with open("iforest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to iforest_model.pkl")
