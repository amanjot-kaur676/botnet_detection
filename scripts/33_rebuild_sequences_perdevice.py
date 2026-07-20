import pandas as pd
import numpy as np
import glob
import os
import random

SEQ_LEN = 10
CHUNK_SIZE = 200_000
SIZE_THRESHOLD_MB = 50
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


def build_window_features(window_df):
    """Build the 10-timestep feature array for ONE device's window (all rows same device)."""
    numeric_vals = window_df[numeric_cols].to_numpy()
    conn_states = window_df["conn_state"].to_numpy()
    dests = window_df["id.resp_h"].to_numpy()
    ports = window_df["id.resp_p"].to_numpy()

    rows = []
    seen_dests = set()
    for t in range(len(window_df)):
        conn_onehot = [1.0 if conn_states[t] == cs else 0.0 for cs in CONN_STATES]
        is_vuln_port = 1.0 if ports[t] in VULNERABLE_PORTS else 0.0
        seen_dests.add(dests[t])
        unique_dest_count_so_far = float(len(seen_dests))  # NEW: fan-out signal, grows for scanning, flat for C2
        row = np.concatenate([numeric_vals[t], conn_onehot, [is_vuln_port], [unique_dest_count_so_far]])
        rows.append(row)
    return np.array(rows)


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

        stage_array = combined["stage"].to_numpy()

        for i in range(SEQ_LEN, len(combined)):
            label = stage_array[i]
            if label in VALID_LABELS:
                window_df = combined.iloc[i - SEQ_LEN:i]
                feat_window = build_window_features(window_df)
                reservoir.add(label, feat_window)

        # keep only the tail as leftover for next chunk, bounded so memory never grows
        leftover_by_device[device_ip] = combined.iloc[-(SEQ_LEN - 1):].reset_index(drop=True)


summary = []

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("iot23_", "").replace("_with_stage.csv", "")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    output_path = os.path.join(output_dir, f"flow_seq_{scenario_name}.npz")

    try:
        reservoir = ReservoirPerLabel(CAP_PER_STAGE)
        leftover_by_device = {}
        needed_cols = numeric_cols + ["conn_state", "id.resp_p", "id.resp_h", "id.orig_h", "stage", "ts"]

        reader = pd.read_csv(file_path, usecols=needed_cols, chunksize=CHUNK_SIZE, low_memory=False)
        for chunk in reader:
            process_chunk_per_device(chunk, leftover_by_device, reservoir)

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

print(f"\nFeature vector is now 5 numeric + 11 conn_state one-hot + 1 vulnerable-port flag "
      f"+ 1 unique-destination-count (fan-out signal) = 18 features per timestep.")
print("Sequences are now built PER-DEVICE, not globally by time - each window represents one device's own flow history.")
