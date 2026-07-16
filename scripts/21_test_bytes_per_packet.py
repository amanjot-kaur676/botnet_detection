"""
Step 21: Refined exfiltration feature - bytes per packet.

Raw byte volume didn't work (DDoS dominates just by having huge totals).
The real distinguishing shape is:
- DDoS: MANY packets, each fairly small/ordinary  -> LOW bytes-per-packet
- Exfiltration-like: FEWER packets, each carrying more data -> HIGH bytes-per-packet

We compute src_bytes / src_pkts per flow (avoiding divide-by-zero), then
run the same per-type comparison as before.
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/raw/train_test_network.csv"
WINDOW_SIZE = 50

df = pd.read_csv(path)

# avoid divide-by-zero: treat 0 packets as NaN, drop those rows for this ratio
df["bytes_per_pkt"] = df["src_bytes"] / df["src_pkts"].replace(0, np.nan)
df = df.dropna(subset=["bytes_per_pkt"])

print(f"Rows usable after removing zero-packet flows: {len(df)}")

print("\n=== Part A: Full-file bytes_per_pkt stats per type ===")
print(df.groupby("type")["bytes_per_pkt"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))

# --- Part B: windowed engineered features ---
HIGH_RATIO_THRESHOLD = df["bytes_per_pkt"].quantile(0.95)
print(f"\nUsing high-ratio threshold (95th percentile overall): {HIGH_RATIO_THRESHOLD:.2f}")

all_windows = []

for attack_type, group in df.groupby("type"):
    group = group.reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE

    for i in range(n_windows):
        chunk = group.iloc[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        vals = chunk["bytes_per_pkt"]

        all_windows.append({
            "type": attack_type,
            "ratio_max": vals.max(),
            "ratio_p95": vals.quantile(0.95),
            "ratio_mean": vals.mean(),
            "ratio_high_count": (vals > HIGH_RATIO_THRESHOLD).sum(),
        })

windows_df = pd.DataFrame(all_windows)

print(f"\nTotal windows built: {len(windows_df)}")
print("\n=== Part B: Engineered bytes-per-packet features per type ===")
summary = windows_df.groupby("type")[
    ["ratio_max", "ratio_p95", "ratio_mean", "ratio_high_count"]
].mean()
print(summary.sort_values("ratio_p95", ascending=False))

windows_df.to_csv("../data/processed/ton_iot_network_bytes_per_pkt_windows.csv", index=False)
print("\nSaved to ../data/processed/ton_iot_network_bytes_per_pkt_windows.csv")
