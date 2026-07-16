"""
Step 17: Inspect the two TON_IoT Linux host files we just downloaded.
- What columns exist (looking for CPU%, timestamp, IP/host id, label/type)
- How big are they (rows)
- A few sample rows
We do NOT process anything yet - just look first.
"""

import pandas as pd
import os

files_to_check = {
    "process": "../data/raw/train_test_linux_process.csv",
    "memory": "../data/raw/train_test_linux_memory.csv",
}

for name, path in files_to_check.items():
    print("=" * 70)
    print(f"FILE: {name}  ->  {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print("File not found at this path - check the exact filename/location.")
        continue

    # just read first chunk to inspect - these files should be small (train_test tier)
    df = pd.read_csv(path, nrows=2000)

    print(f"\nColumns ({len(df.columns)} total):")
    print(list(df.columns))

    print(f"\nTotal rows in this sample read: {len(df)}")

    # try to get real total row count cheaply
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f) - 1  # minus header
    print(f"Actual total rows in file: {total_lines}")

    print("\nFirst 3 rows:")
    print(df.head(3))

    # check for label/type columns
    for col in ["label", "type"]:
        if col in df.columns:
            print(f"\nUnique values in '{col}':")
            print(df[col].value_counts())

    # flag anything that looks like CPU or timestamp
    likely_cpu_cols = [c for c in df.columns if "cpu" in c.lower()]
    likely_time_cols = [c for c in df.columns if "ts" in c.lower() or "time" in c.lower()]
    likely_ip_cols = [c for c in df.columns if "ip" in c.lower() or "host" in c.lower() or "src" in c.lower()]

    print(f"\nColumns that look CPU-related: {likely_cpu_cols}")
    print(f"Columns that look time-related: {likely_time_cols}")
    print(f"Columns that look IP/host-related: {likely_ip_cols}")
    print()
