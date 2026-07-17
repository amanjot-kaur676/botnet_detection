"""
Step 26: Reclassify "Okiru" from Impact -> Scan.

Evidence (script 25): Okiru rows are 99.995% conn_state=S0, essentially
every row targets a unique destination, single packet sent with no
response. This is textbook horizontal scanning, not attack impact.

This script updates the stage_labeled CSVs in place (writes corrected
copies) for the Mirai-lineage files only, since that's what matters
going forward.
"""

import pandas as pd
import glob
import os

MIRAI_LINEAGE_FILES = [
    "iot23_CTU-IoT-Malware-Capture-34-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-43-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-44-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-49-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-52-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-35-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-48-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-7-1_with_stage.csv",
    "iot23_CTU-IoT-Malware-Capture-36-1_with_stage.csv",  # Okiru lives here
    "iot23_CTU-IoT-Malware-Capture-8-1_with_stage.csv",
]

INPUT_DIR = "../data/stage_labeled"
OUTPUT_DIR = "../data/stage_labeled_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 200_000

for filename in MIRAI_LINEAGE_FILES:
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(in_path):
        print(f"Skipping missing file: {in_path}")
        continue

    total_reclassified = 0
    first_chunk = True

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        mask = chunk["detailed-label"] == "Okiru"
        total_reclassified += mask.sum()
        chunk.loc[mask, "stage"] = "Scan"

        chunk.to_csv(out_path, mode="w" if first_chunk else "a",
                      header=first_chunk, index=False)
        first_chunk = False

    print(f"{filename}: reclassified {total_reclassified} Okiru rows from Impact -> Scan")

print(f"\nCorrected files saved to {OUTPUT_DIR}")
print("Next: rebuild sequences from these corrected files instead of the originals.")
