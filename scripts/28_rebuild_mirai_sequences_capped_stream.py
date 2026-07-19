import pandas as pd
import numpy as np
import glob
import os
import random

SEQ_LEN = 10
CHUNK_SIZE = 200_000
SIZE_THRESHOLD_MB = 50
CAP_PER_STAGE = 5000  # matches the final cap used later, so we never need to hold more than this in memory

feature_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
input_files = sorted(glob.glob("../data/stage_labeled_corrected/iot23_CTU-IoT-Malware-Capture-*_with_stage.csv"))
output_dir = "../data/sequences/flow_seq_mirai_corrected"
os.makedirs(output_dir, exist_ok=True)

print(f"Found {len(input_files)} corrected stage-labeled files.\n")

VALID_LABELS = ("Scan", "Infect", "C2", "Impact", "Benign")

random.seed(42)


class ReservoirPerLabel:
    """Keeps a random sample of at most `cap` sequences per label, seen-so-far correct (Algorithm R)."""
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


summary = []

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("iot23_", "").replace("_with_stage.csv", "")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    output_path = os.path.join(output_dir, f"flow_seq_{scenario_name}.npz")

    try:
        reservoir = ReservoirPerLabel(CAP_PER_STAGE)

        if size_mb <= SIZE_THRESHOLD_MB:
            df = pd.read_csv(file_path, low_memory=False)
            df = df.sort_values(by="ts").reset_index(drop=True)
            for col in feature_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            feat_array = df[feature_cols].to_numpy()
            stage_array = df["stage"].to_numpy()

            for i in range(SEQ_LEN, len(df)):
                label = stage_array[i]
                if label in VALID_LABELS:
                    reservoir.add(label, feat_array[i-SEQ_LEN:i])

        else:
            leftover = None
            reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)

            for chunk in reader:
                chunk = chunk.sort_values(by="ts").reset_index(drop=True)
                for col in feature_cols:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

                if leftover is not None:
                    chunk = pd.concat([leftover, chunk], ignore_index=True)

                feat_array = chunk[feature_cols].to_numpy()
                stage_array = chunk["stage"].to_numpy()

                for i in range(SEQ_LEN, len(chunk)):
                    label = stage_array[i]
                    if label in VALID_LABELS:
                        reservoir.add(label, feat_array[i-SEQ_LEN:i])

                leftover = chunk.iloc[-SEQ_LEN:].reset_index(drop=True)

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
