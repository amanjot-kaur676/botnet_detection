"""
Step 10c: Filter the combined dataset down to Mirai-lineage scenarios only.

Mirai-lineage = original Mirai + confirmed source-derived variants (Okiru, Hakai).
Everything else is treated as a separate botnet type and excluded here
(but not deleted - it's still in the original combined file if needed later).
"""

import numpy as np

MIRAI_LINEAGE_SCENARIOS = {
    "CTU-IoT-Malware-Capture-34-1",  # Mirai
    "CTU-IoT-Malware-Capture-43-1",  # Mirai
    "CTU-IoT-Malware-Capture-44-1",  # Mirai
    "CTU-IoT-Malware-Capture-49-1",  # Mirai
    "CTU-IoT-Malware-Capture-52-1",  # Mirai
    "CTU-IoT-Malware-Capture-35-1",  # Mirai
    "CTU-IoT-Malware-Capture-48-1",  # Mirai
    "CTU-IoT-Malware-Capture-7-1",   # Linux.Mirai
    "CTU-IoT-Malware-Capture-36-1",  # Okiru (confirmed Mirai variant)
    "CTU-IoT-Malware-Capture-8-1",   # Hakai (confirmed Mirai-based)
}

data = np.load("../data/sequences/flow_sequences_final_combined.npz", allow_pickle=True)
X, y, scenario = data["X"], data["y"], data["scenario"]

mask = np.array([s in MIRAI_LINEAGE_SCENARIOS for s in scenario])

X_mirai = X[mask]
y_mirai = y[mask]
scenario_mirai = scenario[mask]

print(f"Original total: {len(X)}")
print(f"Mirai-lineage total: {len(X_mirai)}")
print(f"Excluded (other botnets): {len(X) - len(X_mirai)}")

print("\nStage distribution within Mirai-lineage data:")
import pandas as pd
print(pd.Series(y_mirai).value_counts())

print("\nPer-scenario contribution within Mirai-lineage data:")
print(pd.Series(scenario_mirai).value_counts())

np.savez(
    "../data/sequences/flow_sequences_mirai_lineage.npz",
    X=X_mirai, y=y_mirai, scenario=scenario_mirai
)
print("\nSaved to ../data/sequences/flow_sequences_mirai_lineage.npz")
