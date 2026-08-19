import pandas as pd
import numpy as np
import glob
import os
import random
import time
from collections import deque, defaultdict

SEQ_LEN = 10
CHUNK_SIZE = 200_000
CAP_PER_STAGE = 5000

numeric_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
CONN_STATES = ["S0", "SF", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "SHR", "OTH"]
VULNERABLE_PORTS = {23, 2323, 7547, 5555, 37215, 8080, 80, 443}

input_files = sorted(glob.glob("../data/stage_labeled_corrected/iot23_CTU-IoT-Malware-Capture-*_with_stage.csv"))
output_dir = "../data/sequences/flow_seq_mirai_beaconing_v3"
os.makedirs(output_dir, exist_ok=True)

print(f"Found {len(input_files)} corrected stage-labeled files.\n")

VALID_LABELS = ("Scan", "Infect", "C2", "Impact", "Benign")
random.seed(42)


class ReservoirPerLabel:
    def __init__(self, cap):
        self.cap = cap
        self.reservoirs = {label: [] for label in VALID_LABELS}
        self.seen_counts = {label: 0 for label in VALID_LABELS}

    def add(self, label, item):
        self.seen_counts[label] += 1
        reservoir = self.reservoirs[label]
        if len(reservoir) < self.cap:
            reservoir.append(item)
        else:
            j = random.randint(0, self.seen_counts[label] - 1)
            if j < self.cap:
                reservoir[j] = item

    def export(self):
        sequences, labels = [], []
        for label, items in self.reservoirs.items():
            sequences.extend(items)
            labels.extend([label] * len(items))
        return sequences, labels


def compute_device_feature_matrix(combined):
    """
    Compute the full per-row feature matrix for ONE device's flow history,
    ONCE - vectorized, no per-window rebuilding. Returns (n_rows, 20) array.

    20 features = 5 numeric + 11 conn_state one-hot + 1 vulnerable-port flag
    + 1 fan-out (unique destinations) + 1 repeat-contact (max repeats)
    + 1 NEW: beaconing regularity (std of inter-arrival time in window)
    """
    n = len(combined)

    numeric_vals = combined[numeric_cols].to_numpy(dtype=float)  # (n, 5)

    conn_onehot = pd.get_dummies(combined["conn_state"]).reindex(columns=CONN_STATES, fill_value=0).to_numpy(dtype=float)  # (n, 11)

    port_flag = combined["id.resp_p"].isin(VULNERABLE_PORTS).to_numpy(dtype=float).reshape(-1, 1)  # (n, 1)

    # sliding-window unique-destination count (fan-out, high = scanning) AND
    # sliding-window max-repeat-count (repeat-contact, high = C2 target reuse).
    # Uses proper key eviction (del freq[old]) to stay fast - see script 40.
    dests = combined["id.resp_h"].to_numpy()
    unique_counts = np.zeros(n, dtype=float)
    repeat_counts = np.zeros(n, dtype=float)
    window = deque()
    freq = defaultdict(int)
    unique_so_far = 0
    for idx in range(n):
        d = dests[idx]
        if freq[d] == 0:
            unique_so_far += 1
        freq[d] += 1
        window.append(d)
        if len(window) > SEQ_LEN:
            old = window.popleft()
            freq[old] -= 1
            if freq[old] == 0:
                unique_so_far -= 1
                del freq[old]
        unique_counts[idx] = unique_so_far
        repeat_counts[idx] = max(freq.values()) if freq else 0

    # NEW: sliding-window inter-arrival TIMING regularity (beaconing signal).
    # Real C2 beaconing checks in at very consistent intervals - LOW std of
    # inter-arrival time. Scanning/benign traffic is irregular - HIGH std.
    # This looks at WHEN a device connects, not WHERE - a different mechanism
    # than fan-out/repeat-contact, so it should carry non-redundant signal.
    ts_vals = combined["ts"].to_numpy(dtype=float)
    diffs = np.diff(ts_vals, prepend=ts_vals[0])  # diffs[0] is a placeholder (0), not a real gap
    beacon_std = np.zeros(n, dtype=float)
    diff_window = deque()
    d_sum = 0.0
    d_sumsq = 0.0
    for idx in range(n):
        if idx > 0:
            val = diffs[idx]
            diff_window.append(val)
            d_sum += val
            d_sumsq += val * val
            if len(diff_window) > SEQ_LEN - 1:
                old = diff_window.popleft()
                d_sum -= old
                d_sumsq -= old * old
        cnt = len(diff_window)
        if cnt > 0:
            mean = d_sum / cnt
            variance = max(d_sumsq / cnt - mean * mean, 0.0)
            beacon_std[idx] = variance ** 0.5

    return np.concatenate([numeric_vals, conn_onehot, port_flag,
                            unique_counts.reshape(-1, 1),
                            repeat_counts.reshape(-1, 1),
                            beacon_std.reshape(-1, 1)], axis=1)


def process_chunk_per_device(chunk, leftover_by_device, reservoir):
    chunk = chunk.sort_values(by="ts").reset_index(drop=True)
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
    chunk["id.resp_p"] = pd.to_numeric(chunk["id.resp_p"], errors="coerce").fillna(0).astype(int)
    chunk["conn_state"] = chunk["conn_state"].fillna("OTH")
    chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce").fillna(0)

    for device_ip, group in chunk.groupby("id.orig_h", sort=False):
        if device_ip in leftover_by_device:
            combined = pd.concat([leftover_by_device[device_ip], group], ignore_index=True)
        else:
            combined = group.reset_index(drop=True)

        feat_matrix = compute_device_feature_matrix(combined)
        stage_array = combined["stage"].to_numpy()

        n = len(combined)
        for i in range(SEQ_LEN, n):
            label = stage_array[i]
            if label in VALID_LABELS:
                window = feat_matrix[i - SEQ_LEN:i]
                reservoir.add(label, window)

        leftover_by_device[device_ip] = combined.iloc[-(SEQ_LEN - 1):].reset_index(drop=True)


summary = []

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("iot23_", "").replace("_with_stage.csv", "")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    output_path = os.path.join(output_dir, f"flow_seq_{scenario_name}.npz")

    print(f"\nStarting {scenario_name} ({size_mb:.1f} MB)...")

    try:
        reservoir = ReservoirPerLabel(CAP_PER_STAGE)
        leftover_by_device = {}
        needed_cols = numeric_cols + ["conn_state", "id.resp_p", "id.resp_h", "id.orig_h", "stage", "ts"]

        reader = pd.read_csv(file_path, usecols=needed_cols, chunksize=CHUNK_SIZE, low_memory=False)
        chunk_num = 0
        start_time = time.time()
        for chunk in reader:
            chunk_num += 1
            process_chunk_per_device(chunk, leftover_by_device, reservoir)
            if chunk_num % 10 == 0:
                elapsed = time.time() - start_time
                rows_done = chunk_num * CHUNK_SIZE
                rate = rows_done / elapsed
                pct_of_file = min(100, rows_done / (size_mb * 1024 * 1024 / 500) * 100)
                print(f"  ...processed {chunk_num} chunks ({rows_done:,} rows), "
                      f"{elapsed:.0f}s elapsed, ~{rate:,.0f} rows/sec, ~{pct_of_file:.0f}% of file done")

        sequences, labels = reservoir.export()

        if len(sequences) == 0:
            print(f"[SKIP] {scenario_name}: no sequences produced")
            summary.append((scenario_name, "SKIPPED - empty", 0))
            continue

        np.savez(output_path, X=np.array(sequences), y=np.array(labels))
        print(f"[OK] {scenario_name} ({size_mb:.1f} MB): {len(sequences)} sequences saved "
              f"(seen counts: {reservoir.seen_counts}, devices tracked: {len(leftover_by_device)})")
        summary.append((scenario_name, "OK", len(sequences)))

    except Exception as e:
        print(f"[ERROR] {scenario_name}: {e}")
        summary.append((scenario_name, f"ERROR - {e}", 0))

print("\n=== SUMMARY ===")
for name, status, count in summary:
    print(f"{name}: {status} ({count} sequences)")
