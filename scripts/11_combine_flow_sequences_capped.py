import numpy as np
import glob
import os

MAX_PER_SCENARIO_PER_STAGE = 5000
input_files = sorted(glob.glob("../data/sequences/flow_seq_by_scenario/flow_seq_*.npz"))

all_X = []
all_y = []
all_scenario = []

rng = np.random.default_rng(42)

for file_path in input_files:
    scenario_name = os.path.basename(file_path).replace("flow_seq_", "").replace(".npz", "")
    data = np.load(file_path)
    X, y = data["X"], data["y"]

    for stage in np.unique(y):
        idx = np.where(y == stage)[0]
        if len(idx) > MAX_PER_SCENARIO_PER_STAGE:
            idx = rng.choice(idx, size=MAX_PER_SCENARIO_PER_STAGE, replace=False)

        all_X.append(X[idx])
        all_y.append(y[idx])
        all_scenario.extend([scenario_name] * len(idx))

    print(f"{scenario_name}: kept {sum(len(np.where(y==s)[0][:MAX_PER_SCENARIO_PER_STAGE]) for s in np.unique(y))} sequences (capped)")

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)

print()
print("Final combined shape:", X_final.shape)
import pandas as pd
print(pd.Series(y_final).value_counts())
print()
print("Per-scenario contribution:")
print(pd.Series(all_scenario).value_counts())

np.savez(
    "../data/sequences/flow_sequences_final_combined.npz",
    X=X_final, y=y_final, scenario=np.array(all_scenario)
)
print("\nSaved to ../data/sequences/flow_sequences_final_combined.npz")