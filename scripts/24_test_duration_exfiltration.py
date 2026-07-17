"""
Step 24: Test connection duration as an exfiltration signal.

Different mechanism than byte volume: real data exfiltration is often
a slow, deliberate, long-lived connection rather than one big burst.
We test whether any attack type shows unusually long connection durations
compared to normal, using the same per-type + windowed approach as before.
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/raw/train_test_network.csv"
df = pd.read_csv(path)

print("=== Part A: Full-file duration stats per type ===")
print(df.groupby("type")["duration"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))

# --- Part B: windowed engineered features ---
WINDOW_SIZE = 50
HIGH_DURATION_THRESHOLD = df["duration"].quantile(0.95)
print(f"\nHigh-duration threshold (95th percentile overall): {HIGH_DURATION_THRESHOLD}")

all_windows = []
for attack_type, group in df.groupby("type"):
    group = group.reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE
    for i in range(n_windows):
        chunk = group.iloc[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        vals = chunk["duration"]
        all_windows.append({
            "type": attack_type,
            "dur_max": vals.max(),
            "dur_p95": vals.quantile(0.95),
            "dur_mean": vals.mean(),
            "dur_high_count": (vals > HIGH_DURATION_THRESHOLD).sum(),
        })

windows_df = pd.DataFrame(all_windows)
print(f"\nTotal windows built: {len(windows_df)}")
print("\n=== Part B: Engineered duration features per type ===")
summary = windows_df.groupby("type")[["dur_max", "dur_p95", "dur_mean", "dur_high_count"]].mean()
print(summary.sort_values("dur_p95", ascending=False))

windows_df.to_csv("../data/processed/ton_iot_duration_windows.csv", index=False)
print("\nSaved to ../data/processed/ton_iot_duration_windows.csv")
