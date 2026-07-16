"""
Step 10b: Find out what attack types actually exist inside the "Impact" stage.

Some files are 10+ GB, so we NEVER load a whole file at once.
Instead we read it in small chunks (100,000 rows at a time),
keep only the columns we need, and keep only rows where stage == 'Impact'.

Output: a count of each detailed-label value, ONLY for Impact-stage rows,
combined across all 20 files.
"""

import glob
import pandas as pd

input_files = sorted(glob.glob("../data/stage_labeled/*_with_stage.csv"))

CHUNK_SIZE = 100_000  # rows per chunk - safe for memory

# we only need these two columns for this step - keeps memory low
USE_COLS = ["detailed-label", "stage"]

impact_label_counts = pd.Series(dtype="int64")

for file_path in input_files:
    print(f"Scanning {file_path} ...")
    file_impact_rows = 0

    for chunk in pd.read_csv(file_path, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False):
        impact_chunk = chunk[chunk["stage"] == "Impact"]
        file_impact_rows += len(impact_chunk)

        if len(impact_chunk) > 0:
            counts = impact_chunk["detailed-label"].value_counts()
            impact_label_counts = impact_label_counts.add(counts, fill_value=0)

    print(f"   -> {file_impact_rows} Impact-stage rows found in this file")

print("\n=== FINAL: detailed-label counts, Impact stage only, all files combined ===")
print(impact_label_counts.sort_values(ascending=False))

# save to a small CSV so we don't lose this and don't need to re-scan 40+ GB again
impact_label_counts.sort_values(ascending=False).to_csv(
    "../data/stage_labeled/impact_detailed_label_counts.csv"
)
print("\nSaved summary to ../data/stage_labeled/impact_detailed_label_counts.csv")
