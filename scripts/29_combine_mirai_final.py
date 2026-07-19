import numpy as np
import glob
import os
import pandas as pd

input_files = sorted(glob.glob("../data/sequences/flow_seq_mirai_corrected/flow_seq_*.npz"))

all_X = []
all_y = []
all_scenario = []

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("flow_seq_", "").replace(".npz", "")
    data = np.load(file_path)
    X, y = data["X"], data["y"]

    all_X.append(X)
    all_y.append(y)
    all_scenario.extend([scenario_name] * len(y))

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)

print("Final combined shape:", X_final.shape)
print("\nStage distribution:")
print(pd.Series(y_final).value_counts())
print("\nPer-scenario contribution:")
print(pd.Series(all_scenario).value_counts())

np.savez(
    "../data/sequences/flow_sequences_mirai_final.npz",
    X=X_final, y=y_final, scenario=np.array(all_scenario)
)
print("\nSaved to ../data/sequences/flow_sequences_mirai_final.npz")
