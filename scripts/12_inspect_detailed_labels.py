"""
Step 10 - Part 1: Look at the actual detailed-label values in IoT-23
before we decide how to group them into DoS / DDoS / Exfiltration.

This does NOT change anything. It only prints counts so we can see
what's really in the data.
"""

import glob
import pandas as pd

# adjust this path pattern to match wherever your raw IoT-23 conn.log
# (or the csv you already parsed from it) files live
input_files = sorted(glob.glob("../data/raw/iot23/*/conn.log.labeled*"))

if not input_files:
    print("No files matched. Check the folder path/pattern above and fix it.")
else:
    all_labels = []

    for file_path in input_files:
        # IoT-23 conn.log.labeled files are tab-separated, last column is the label info
        # label info format is usually: label   detailed-label
        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                comment="#",
                header=None,
                low_memory=False,
            )
            # detailed-label is typically the last column
            last_col = df.columns[-1]
            all_labels.append(df[last_col])
            print(f"Read {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    if all_labels:
        combined = pd.concat(all_labels, ignore_index=True)
        print("\n--- Unique detailed-label values and counts ---")
        print(combined.value_counts())
    else:
        print("No labels could be read. Check file format.")
