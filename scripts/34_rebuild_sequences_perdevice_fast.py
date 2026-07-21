import pandas as pd
import numpy as np
import glob
import os
import random
from collections import deque, defaultdict

SEQ_LEN = 10
CHUNK_SIZE = 200_000
CAP_PER_STAGE = 5000

numeric_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
CONN_STATES = ["S0", "SF", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "SHR", "OTH"]
VULNERABLE_PORTS = {23, 2323, 7547, 5555, 37215, 8080, 80, 443}

input_files = sorted(glob.glob("../data/stage_labeled_corrected/iot23_CTU-IoT-Malware-Capture-*_with_stage.csv"))
output_dir = "../data/sequences/flow_seq_mirai_perdevice"
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
    ONCE - vectorized, no per-window rebuilding. Returns (n_rows, 18) array.
    """
    n = len(combined)

    numeric_vals = combined[numeric_cols].to_numpy(dtype=float)  # (n, 5)

    # vectorized one-hot for conn_state - fast, no python loop per row
    conn_onehot = pd.get_dummies(combined["conn_state"]).reindex(columns=CONN_STATES, fill_value=0).to_numpy(dtype=float)  # (n, 11)

    port_flag = combined["id.resp_p"].isin(VULNERABLE_PORTS).to_numpy(dtype=float).reshape(-1, 1)  # (n, 1)

    # sliding-window unique-destination count (fan-out signal), O(n) single pass
    # using a deque + frequency counter instead of rebuilding a set from scratch each window
    dests = combined["id.resp_h"].to_numpy()
    unique_counts = np.zeros(n, dtype=float)
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
        unique_counts[idx] = unique_so_far

    return np.concatenate([numeric_vals, conn_onehot, port_flag, unique_counts.reshape(-1, 1)], axis=1)


def process_chunk_per_device(chunk, leftover_by_device, reservoir):
    chunk = chunk.sort_values(by="ts").reset_index(drop=True)
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
    chunk["id.resp_p"] = pd.to_numeric(chunk["id.resp_p"], errors="coerce").fillna(0).astype(int)
    chunk["conn_state"] = chunk["conn_state"].fillna("OTH")

    for device_ip, group in chunk.groupby("id.orig_h", sort=False):
        if device_ip in leftover_by_device:
            combined = pd.concat([leftover_by_device[device_ip], group], ignore_index=True)
        else:
            combined = group.reset_index(drop=True)

        feat_matrix = compute_device_feature_matrix(combined)  # ONE computation for the whole device history
        stage_array = combined["stage"].to_numpy()

        n = len(combined)
        for i in range(SEQ_LEN, n):
            label = stage_array[i]
            if label in VALID_LABELS:
                window = feat_matrix[i - SEQ_LEN:i]  # cheap numpy slice, not a rebuild
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
        for chunk in reader:
            chunk_num += 1
            process_chunk_per_device(chunk, leftover_by_device, reservoir)
            if chunk_num % 10 == 0:
                print(f"  ...processed {chunk_num} chunks ({chunk_num * CHUNK_SIZE:,} rows so far)")

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
