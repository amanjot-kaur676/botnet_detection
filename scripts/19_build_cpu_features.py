"""
Step 19: Feature engineering on TON_IoT Linux process CPU data.

No timestamp/session column exists, so we approximate "windows" by grouping
consecutive rows (in file order) into fixed-size blocks, separately within
each attack 'type'. This is an approximation - documented as such.

Engineered features per window:
- cpu_max        : peak CPU value seen in the window
- cpu_p95        : 95th percentile CPU value in the window
- cpu_high_count : how many rows in the window exceed a CPU threshold
"""

import pandas as pd
import numpy as np

# show all columns fully, no truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../data/raw/train_test_linux_process.csv"
WINDOW_SIZE = 50          # rows per window (approximation of a time window)
HIGH_CPU_THRESHOLD = 1.0  # threshold for "process under heavy load"

df = pd.read_csv(path)

all_windows = []

# build windows separately per type so we don't mix dos rows with normal rows in one window
for attack_type, group in df.groupby("type"):
    group = group.reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE

    for i in range(n_windows):
        chunk = group.iloc[i * WINDOW_SIZE : (i + 1) * WINDOW_SIZE]
        cpu_values = chunk["CPU"]

        all_windows.append({
            "type": attack_type,
            "cpu_max": cpu_values.max(),
            "cpu_p95": cpu_values.quantile(0.95),
            "cpu_mean": cpu_values.mean(),
            "cpu_high_count": (cpu_values > HIGH_CPU_THRESHOLD).sum(),
        })

windows_df = pd.DataFrame(all_windows)

print(f"Total windows built: {len(windows_df)}")
print(f"\nWindows per type:")
print(windows_df["type"].value_counts())

print("\nEngineered feature averages, per type (sorted by cpu_max):")
summary = windows_df.groupby("type")[["cpu_max", "cpu_p95", "cpu_mean", "cpu_high_count"]].mean()
print(summary.sort_values("cpu_max", ascending=False))

windows_df.to_csv("../data/processed/ton_iot_linux_cpu_windows.csv", index=False)
print("\nSaved to ../data/processed/ton_iot_linux_cpu_windows.csv")