import pandas as pd
import numpy as np
import glob
import os
import random

SEQ_LEN = 10
CHUNK_SIZE = 200_000
SIZE_THRESHOLD_MB = 50
CAP_PER_STAGE = 5000

# --- Numeric features, same as before ---
numeric_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

# --- NEW: conn_state one-hot categories (common Zeek conn_state values) ---
CONN_STATES = ["S0", "SF", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "SHR", "OTH"]

# --- NEW: known vulnerable ports Mirai/IoT botnets target ---
VULNERABLE_PORTS = {23, 2323, 7547, 5555, 37215, 8080, 80, 443}

input_files = sorted(glob.glob("../data/stage_labeled_corrected/iot23_CTU-IoT-Malware-Capture-*_with_stage.csv"))
output_dir = "../data/sequences/flow_seq_mirai_richfeatures"
os.makedirs(output_dir, exist_ok=True)

print(f"Found {len(input_files)} corrected stage-labeled files.\n")

VALID_LABELS = ("Scan", "Infect", "C2", "Impact", "Benign")
random.seed(42)


def build_feature_row(row_numeric, conn_state_value, port_value):
    """Combine numeric features + one-hot conn_state + vulnerable-port flag into one feature vector."""
    conn_state_onehot = [1.0 if conn_state_value == cs else 0.0 for cs in CONN_STATES]
    is_vuln_port = 1.0 if port_value in VULNERABLE_PORTS else 0.0
    return np.concatenate([row_numeric, conn_state_onehot, [is_vuln_port]])


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


def process_dataframe(df, reservoir):
    df = df.sort_values(by="ts").reset_index(drop=True)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["id.resp_p"] = pd.to_numeric(df["id.resp_p"], errors="coerce").fillna(0).astype(int)
    df["conn_state"] = df["conn_state"].fillna("OTH")

    numeric_array = df[numeric_cols].to_numpy()
    conn_state_array = df["conn_state"].to_numpy()
    port_array = df["id.resp_p"].to_numpy()
    stage_array = df["stage"].to_numpy()

    full_feat_array = np.array([
        build_feature_row(numeric_array[i], conn_state_array[i], port_array[i])
        for i in range(len(df))
    ])

    for i in range(SEQ_LEN, len(df)):
        label = stage_array[i]
        if label in VALID_LABELS:
            reservoir.add(label, full_feat_array[i-SEQ_LEN:i])

    return df.iloc[-SEQ_LEN:].reset_index(drop=True)


summary = []

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("iot23_", "").replace("_with_stage.csv", "")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    output_path = os.path.join(output_dir, f"flow_seq_{scenario_name}.npz")

    try:
        reservoir = ReservoirPerLabel(CAP_PER_STAGE)
        needed_cols = numeric_cols + ["conn_state", "id.resp_p", "stage", "ts"]

        if size_mb <= SIZE_THRESHOLD_MB:
            df = pd.read_csv(file_path, usecols=needed_cols, low_memory=False)
            process_dataframe(df, reservoir)
        else:
            leftover = None
            reader = pd.read_csv(file_path, usecols=needed_cols, chunksize=CHUNK_SIZE, low_memory=False)
            for chunk in reader:
                if leftover is not None:
                    chunk = pd.concat([leftover, chunk], ignore_index=True)
                leftover = process_dataframe(chunk, reservoir)

        sequences, labels = reservoir.export()

        if len(sequences) == 0:
            print(f"[SKIP] {scenario_name}: no sequences produced")
            summary.append((scenario_name, "SKIPPED - empty", 0))
            continue

        np.savez(output_path, X=np.array(sequences), y=np.array(labels))
        method = "full" if size_mb <= SIZE_THRESHOLD_MB else "chunked+capped"
        print(f"[OK] {scenario_name} ({method}, {size_mb:.1f} MB): {len(sequences)} sequences saved "
              f"(seen counts: {reservoir.seen_counts})")
        summary.append((scenario_name, f"OK ({method})", len(sequences)))

    except Exception as e:
        print(f"[ERROR] {scenario_name}: {e}")
        summary.append((scenario_name, f"ERROR - {e}", 0))

print("\n=== SUMMARY ===")
for name, status, count in summary:
    print(f"{name}: {status} ({count} sequences)")

print(f"\nFeature vector is now {len(numeric_cols)} numeric + {len(CONN_STATES)} conn_state one-hot + 1 vulnerable-port flag "
      f"= {len(numeric_cols) + len(CONN_STATES) + 1} features per timestep (was {len(numeric_cols)} before).")
