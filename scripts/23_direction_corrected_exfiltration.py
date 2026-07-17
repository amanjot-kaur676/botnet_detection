"""
Step 23: Direction-corrected outward byte volume analysis.

Earlier analysis assumed src_bytes = "data leaving the infected device."
We now know that's WRONG for most attack types - src_ip is the ATTACKER
for ddos/dos/injection/password/scanning/xss (IPs .30-.39, per TON_IoT docs).

Fix: define victim_outbound_bytes per row based on which side is actually
the victim/compromised device:
- If src_ip is a known attacker IP -> the victim's outbound data is dst_bytes
  (the response/return traffic FROM victim back TO attacker)
- Otherwise (e.g. backdoor, where src IS the victim) -> victim_outbound_bytes
  is src_bytes as originally assumed
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/raw/train_test_network.csv"
df = pd.read_csv(path)

# data-driven attacker IP set, based on script 22's output
ATTACKER_IPS = {
    "192.168.1.30", "192.168.1.31", "192.168.1.32",
    "192.168.1.36", "192.168.1.38", "192.168.1.39",
}

def victim_outbound_bytes(row):
    if row["src_ip"] in ATTACKER_IPS:
        return row["dst_bytes"]   # victim is dst, so victim's own outbound traffic is dst_bytes
    else:
        return row["src_bytes"]   # victim is src, so victim's outbound traffic is src_bytes

df["victim_outbound_bytes"] = df.apply(victim_outbound_bytes, axis=1)

print("=== Part A: Direction-corrected victim_outbound_bytes stats per type ===")
print(df.groupby("type")["victim_outbound_bytes"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))

# --- Part B: windowed engineered features, same style as before ---
WINDOW_SIZE = 50
HIGH_THRESHOLD = df["victim_outbound_bytes"].quantile(0.95)
print(f"\nHigh-volume threshold (95th percentile): {HIGH_THRESHOLD}")

all_windows = []
for attack_type, group in df.groupby("type"):
    group = group.reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE
    for i in range(n_windows):
        chunk = group.iloc[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        vals = chunk["victim_outbound_bytes"]
        all_windows.append({
            "type": attack_type,
            "vob_max": vals.max(),
            "vob_p95": vals.quantile(0.95),
            "vob_mean": vals.mean(),
            "vob_high_count": (vals > HIGH_THRESHOLD).sum(),
        })

windows_df = pd.DataFrame(all_windows)
print(f"\nTotal windows built: {len(windows_df)}")
print("\n=== Part B: Engineered direction-corrected outbound-byte features per type ===")
summary = windows_df.groupby("type")[["vob_max", "vob_p95", "vob_mean", "vob_high_count"]].mean()
print(summary.sort_values("vob_p95", ascending=False))

windows_df.to_csv("../data/processed/ton_iot_victim_outbound_windows.csv", index=False)
print("\nSaved to ../data/processed/ton_iot_victim_outbound_windows.csv")
