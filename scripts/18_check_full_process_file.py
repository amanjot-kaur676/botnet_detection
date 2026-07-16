"""
Step 17b: Look at the FULL process file (not just first 2000 rows).
We need to confirm:
1. Does this file include normal/benign rows (label=0), not just attacks?
2. What does CPU usage look like for each type (dos, ddos, normal, etc.)?
"""

import pandas as pd

path = "../data/raw/train_test_linux_process.csv"

df = pd.read_csv(path)

print(f"Total rows: {len(df)}")

print("\nFull 'label' distribution:")
print(df["label"].value_counts())

print("\nFull 'type' distribution:")
print(df["type"].value_counts())

print("\nCPU column basic stats (overall):")
print(df["CPU"].describe())

print("\nAverage CPU usage, grouped by attack type:")
print(df.groupby("type")["CPU"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False))
