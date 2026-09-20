"""
Build a FEATURE SUPERSET dataset - includes both raw Basic-tier fields
(protocol, raw ports, derived rate/ratio features) AND our existing
validated engineered features (conn_state, vulnerable-port flag, fan-out,
repeat-contact, beaconing). This lets us slice out different feature-count
tiers (Basic / Selected / Expanded) from ONE dataset for a fair, controlled
comparison, instead of rebuilding separately for each tier.

New raw/derived fields added this round:
- proto (one-hot: tcp/udp/icmp)
- id.orig_p, id.resp_p (raw port numbers, not just a vulnerable-port flag)
- bytes_per_sec, pkts_per_sec (rate features)
- avg_pkt_size (orig_bytes / orig_pkts)
- fwd_bwd_ratio (orig_bytes / (resp_bytes + 1), forward/backward asymmetry)
"""

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
PROTOS = ["tcp", "udp", "icmp"]
VULNERABLE_PORTS = {23, 2323, 7547, 5555, 37215, 8080, 80, 443}

input_files = sorted(glob.glob("../data/stage_labeled_corrected/iot23_CTU-IoT-Malware-Capture-*_with_stage.csv"))
output_dir = "../data/sequences/flow_seq_mirai_superset"
os.makedirs(output_dir, exist_ok=True)

print(f"Found {len(input_files)} corrected stage-labeled files.\n")

VALID_LABELS = ("Scan", "Infect", "C2", "Impact", "Benign")
random.seed(42)

# Feature layout (for later tier-slicing):
#  0-4   : numeric_cols (duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts)
#  5-7   : proto one-hot (tcp, udp, icmp)
#  8     : id.orig_p (raw source port, scaled by /1000 to keep values small)
#  9     : id.resp_p (raw destination port, scaled by /1000)
#  10    : bytes_per_sec
#  11    : pkts_per_sec
#  12    : avg_pkt_size
#  13    : fwd_bwd_ratio
#  14-24 : conn_state one-hot (11)
#  25    : vulnerable_port flag
#  26    : fan-out count
#  27    : repeat-contact count
#  28    : beaconing std
FEATURE_NAMES = (
    numeric_cols + PROTOS + ["src_port", "dst_port", "bytes_per_sec", "pkts_per_sec",
    "avg_pkt_size", "fwd_bwd_ratio"] + CONN_STATES + ["vuln_port", "fanout_count",
    "repeat_count", "beacon_std"]
)


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
    n = len(combined)

    numeric_vals = combined[numeric_cols].to_numpy(dtype=float)

    proto_onehot = pd.get_dummies(combined["proto"]).reindex(columns=PROTOS, fill_value=0).to_numpy(dtype=float)

    src_port = (combined["id.orig_p"].to_numpy(dtype=float) / 1000.0).reshape(-1, 1)
    dst_port = (combined["id.resp_p"].to_numpy(dtype=float) / 1000.0).reshape(-1, 1)

    duration_safe = np.maximum(combined["duration"].to_numpy(dtype=float), 1e-3)
    bytes_total = combined["orig_bytes"].to_numpy(dtype=float) + combined["resp_bytes"].to_numpy(dtype=float)
    pkts_total = combined["orig_pkts"].to_numpy(dtype=float) + combined["resp_pkts"].to_numpy(dtype=float)
    bytes_per_sec = (bytes_total / duration_safe).reshape(-1, 1)
    pkts_per_sec = (pkts_total / duration_safe).reshape(-1, 1)

    orig_pkts_safe = np.maximum(combined["orig_pkts"].to_numpy(dtype=float), 1)
    avg_pkt_size = (combined["orig_bytes"].to_numpy(dtype=float) / orig_pkts_safe).reshape(-1, 1)

    resp_bytes_safe = combined["resp_bytes"].to_numpy(dtype=float) + 1
    fwd_bwd_ratio = (combined["orig_bytes"].to_numpy(dtype=float) / resp_bytes_safe).reshape(-1, 1)

    conn_onehot = pd.get_dummies(combined["conn_state"]).reindex(columns=CONN_STATES, fill_value=0).to_numpy(dtype=float)

    port_flag = combined["id.resp_p"].isin(VULNERABLE_PORTS).to_numpy(dtype=float).reshape(-1, 1)

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

    ts_vals = combined["ts"].to_numpy(dtype=float)
    diffs = np.diff(ts_vals, prepend=ts_vals[0])
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

    return np.concatenate([
        numeric_vals, proto_onehot, src_port, dst_port, bytes_per_sec, pkts_per_sec,
        avg_pkt_size, fwd_bwd_ratio, conn_onehot, port_flag,
        unique_counts.reshape(-1, 1), repeat_counts.reshape(-1, 1), beacon_std.reshape(-1, 1),
    ], axis=1)


def process_chunk_per_device(chunk, leftover_by_device, reservoir):
    chunk = chunk.sort_values(by="ts").reset_index(drop=True)
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
    chunk["id.resp_p"] = pd.to_numeric(chunk["id.resp_p"], errors="coerce").fillna(0).astype(int)
    chunk["id.orig_p"] = pd.to_numeric(chunk["id.orig_p"], errors="coerce").fillna(0).astype(int)
    chunk["conn_state"] = chunk["conn_state"].fillna("OTH")
    chunk["proto"] = chunk["proto"].fillna("tcp")
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
        needed_cols = numeric_cols + ["conn_state", "proto", "id.resp_p", "id.orig_p",
                                       "id.resp_h", "id.orig_h", "stage", "ts"]

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
                print(f"  ...processed {chunk_num} chunks ({rows_done:,} rows), {elapsed:.0f}s elapsed, ~{rate:,.0f} rows/sec")

        sequences, labels = reservoir.export()

        if len(sequences) == 0:
            print(f"[SKIP] {scenario_name}: no sequences produced")
            summary.append((scenario_name, "SKIPPED - empty", 0))
            continue

        np.savez(output_path, X=np.array(sequences), y=np.array(labels))
        print(f"[OK] {scenario_name} ({size_mb:.1f} MB): {len(sequences)} sequences saved")
        summary.append((scenario_name, "OK", len(sequences)))

    except Exception as e:
        print(f"[ERROR] {scenario_name}: {e}")
        summary.append((scenario_name, f"ERROR - {e}", 0))

print("\n=== SUMMARY ===")
for name, status, count in summary:
    print(f"{name}: {status} ({count} sequences)")

print(f"\nFeature layout ({len(FEATURE_NAMES)} total): {FEATURE_NAMES}")
