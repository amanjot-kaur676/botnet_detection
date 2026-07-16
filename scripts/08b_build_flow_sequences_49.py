import pandas as pd
import numpy as np

SEQ_LEN = 10
CHUNK_SIZE = 200_000
input_path = "../data/stage_labeled/iot23_49_with_stage.csv"
feature_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

all_sequences = []
all_labels = []
leftover = None  # holds the tail rows of previous chunk to bridge sequences across chunk boundaries

reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False)

for chunk in reader:
    chunk = chunk.sort_values(by="ts").reset_index(drop=True)
    for col in feature_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    if leftover is not None:
        chunk = pd.concat([leftover, chunk], ignore_index=True)

    for i in range(SEQ_LEN, len(chunk)):
        window = chunk.iloc[i-SEQ_LEN:i]
        label = chunk.iloc[i]["stage"]

        # Only keep sequences for rare stages, to avoid an enormous all-Scan file
        if label in ("Infect", "C2", "Impact", "Benign"):
            all_sequences.append(window[feature_cols].values.tolist())
            all_labels.append(label)

    # Keep the last SEQ_LEN rows to bridge into the next chunk
    leftover = chunk.iloc[-SEQ_LEN:].reset_index(drop=True)

print("Total sequences built (rare stages only):", len(all_sequences))
print(pd.Series(all_labels).value_counts())

np.savez(
    "../data/sequences/flow_sequences_49-1_rare.npz",
    X=np.array(all_sequences),
    y=np.array(all_labels)
)
print("Saved to ../data/sequences/flow_sequences_49-1_rare.npz")