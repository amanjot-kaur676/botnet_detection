"""
Step 25: Figure out what "Okiru" traffic actually looks like, behaviorally.

Okiru is a botnet-family label, not a behavior label, but it's confirmed
Mirai-lineage (scenario 36-1). Rather than leave it wrongly bucketed as
"Impact" by default, we compare its flow characteristics against flows we
ARE confident about (known Scan/C2/Impact patterns) to make an evidence-based
call on where it really belongs.
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/stage_labeled/iot23_CTU-IoT-Malware-Capture-36-1_with_stage.csv"

USE_COLS = ["detailed-label", "stage", "conn_state", "duration",
            "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "id.resp_h"]

CHUNK_SIZE = 200_000
okiru_rows = []
other_rows = []

for chunk in pd.read_csv(path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False):
    okiru_chunk = chunk[chunk["detailed-label"] == "Okiru"]
    other_chunk = chunk[chunk["detailed-label"] != "Okiru"]

    if len(okiru_chunk) > 0:
        okiru_rows.append(okiru_chunk.sample(min(len(okiru_chunk), 2000), random_state=42))
    if len(other_chunk) > 0:
        other_rows.append(other_chunk.sample(min(len(other_chunk), 2000), random_state=42))

okiru_df = pd.concat(okiru_rows, ignore_index=True)
other_df = pd.concat(other_rows, ignore_index=True)

print(f"Okiru sample size: {len(okiru_df)}")
print(f"Other (non-Okiru) sample size: {len(other_df)}")

print("\n=== Okiru: conn_state distribution ===")
print(okiru_df["conn_state"].value_counts(normalize=True).head(10))

print("\n=== Okiru: unique destination count (fan-out signal - high = scanning) ===")
print(f"Unique destinations in Okiru sample: {okiru_df['id.resp_h'].nunique()} out of {len(okiru_df)} rows")

print("\n=== Okiru: duration/bytes/packets stats ===")
print(okiru_df[["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]].describe())

print("\n=== For comparison - other stages already labeled in this file ===")
print(other_df.groupby("stage")["conn_state"].value_counts(normalize=True).groupby(level=0).head(3))

print("\n=== For comparison - other stages: duration/bytes/packets stats ===")
print(other_df.groupby("stage")[["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]].mean())

print("\n=== For comparison - unique destinations per stage ===")
print(other_df.groupby("stage")["id.resp_h"].nunique())
