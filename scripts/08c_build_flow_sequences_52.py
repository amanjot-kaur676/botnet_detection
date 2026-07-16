import pandas as pd
import numpy as np

SEQ_LEN = 10
CHUNK_SIZE = 200_000
input_path = "../data/stage_labeled/iot23_52_with_stage.csv"
feature_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

all_sequences = []
all_labels = []
leftover = None

reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False)

for chunk_num, chunk in enumerate(reader):
    chunk = chunk.sort_values(by="ts").reset_index(drop=True)
    for col in feature_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    if leftover is not None:
        chunk = pd.concat([leftover, chunk], ignore_index=True)

    # Convert to plain numpy ONCE per chunk — this is the speed fix
    feature_array = chunk[feature_cols].to_numpy()
    stage_array = chunk["stage"].to_numpy()

    n = len(chunk)
    for i in range(SEQ_LEN, n):
        label = stage_array[i]
        if label in ("Infect", "C2", "Impact", "Benign"):
            all_sequences.append(feature_array[i-SEQ_LEN:i])
            all_labels.append(label)

    leftover = chunk.iloc[-SEQ_LEN:].reset_index(drop=True)
    print(f"Chunk {chunk_num+1} done, running total sequences: {len(all_sequences)}")

print()
print("Total sequences built (rare stages only):", len(all_sequences))
print(pd.Series(all_labels).value_counts())

np.savez(
    "../data/sequences/flow_sequences_52-1_rare.npz",
    X=np.array(all_sequences),
    y=np.array(all_labels)
)
print("Saved to ../data/sequences/flow_sequences_52-1_rare.npz")