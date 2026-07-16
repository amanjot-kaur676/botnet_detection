import pandas as pd
import numpy as np

df = pd.read_csv("../data/stage_labeled/iot23_34-1_with_stage.csv")
df = df.sort_values(by="ts").reset_index(drop=True)

SEQ_LEN = 10  # number of past flows per sequence

feature_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

sequences = []
labels = []

for i in range(SEQ_LEN, len(df)):
    window = df.iloc[i-SEQ_LEN:i]
    seq_features = window[feature_cols].values.tolist()
    label = df.iloc[i]["stage"]

    sequences.append(seq_features)
    labels.append(label)

print("Total sequences built:", len(sequences))
print("Label distribution:")
print(pd.Series(labels).value_counts())

# Save as a compressed numpy file (efficient for sequence data)
np.savez(
    "../data/sequences/flow_sequences_34-1.npz",
    X=np.array(sequences),
    y=np.array(labels)
)
print("Saved to ../data/sequences/flow_sequences_34-1.npz")