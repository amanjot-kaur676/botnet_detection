"""
Step 20: Test the exfiltration hypothesis using outward byte volume.

No pre-existing "exfiltration" label exists in TON_IoT, so instead of looking
for a label, we directly test the feature: does src_bytes (data sent FROM the
device outward) look unusually large for any attack type, compared to normal?

Same approach as the CPU analysis:
1. Full-file stats per type (quick sanity check)
2. Windowed engineered features (max, p95, mean, high-volume count)
   - windows approximate time order the same way as script 19 (no timestamp
     column exists here either)
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/raw/train_test_network.csv"
WINDOW_SIZE = 50

df = pd.read_csv(path)

print(f"Total rows: {len(df)}")
print("\nFull 'type' distribution:")
print(df["type"].value_counts())

print("\n=== Part A: Full-file src_bytes stats per type ===")
print(df.groupby("type")["src_bytes"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))

print("\n=== Part A: Full-file dst_bytes stats per type (for comparison) ===")
print(df.groupby("type")["dst_bytes"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))

# --- Part B: windowed engineered features, same style as CPU analysis ---
HIGH_BYTES_THRESHOLD = df["src_bytes"].quantile(0.95)  # data-driven threshold, not guessed
print(f"\nUsing high-volume threshold (95th percentile of all src_bytes): {HIGH_BYTES_THRESHOLD}")

all_windows = []

for attack_type, group in df.groupby("type"):
    group = group.reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE

    for i in range(n_windows):
        chunk = group.iloc[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        vals = chunk["src_bytes"]

        all_windows.append({
            "type": attack_type,
            "src_bytes_max": vals.max(),
            "src_bytes_p95": vals.quantile(0.95),
            "src_bytes_mean": vals.mean(),
            "src_bytes_high_count": (vals > HIGH_BYTES_THRESHOLD).sum(),
        })

windows_df = pd.DataFrame(all_windows)

print(f"\nTotal windows built: {len(windows_df)}")
print("\n=== Part B: Engineered outward-byte-volume features per type ===")
summary = windows_df.groupby("type")[
    ["src_bytes_max", "src_bytes_p95", "src_bytes_mean", "src_bytes_high_count"]
].mean()
print(summary.sort_values("src_bytes_p95", ascending=False))

windows_df.to_csv("../data/processed/ton_iot_network_bytes_windows.csv", index=False)
print("\nSaved to ../data/processed/ton_iot_network_bytes_windows.csv")
