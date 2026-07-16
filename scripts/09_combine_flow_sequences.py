import numpy as np

# Load all three saved sequence sets
data_34 = np.load("../data/sequences/flow_sequences_34-1.npz")
data_49 = np.load("../data/sequences/flow_sequences_49-1_rare.npz")
data_52 = np.load("../data/sequences/flow_sequences_52-1_rare.npz")

X_34, y_34 = data_34["X"], data_34["y"]
X_49, y_49 = data_49["X"], data_49["y"]
X_52, y_52 = data_52["X"], data_52["y"]

X_combined = np.concatenate([X_34, X_49, X_52], axis=0)
y_combined = np.concatenate([y_34, y_49, y_52], axis=0)

print("Combined shape:", X_combined.shape)
print("Label distribution:")
import pandas as pd
print(pd.Series(y_combined).value_counts())

np.savez(
    "../data/sequences/flow_sequences_combined.npz",
    X=X_combined,
    y=y_combined
)
print("Saved to ../data/sequences/flow_sequences_combined.npz")