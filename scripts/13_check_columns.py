"""
Step 10a: Just look at the column names + a few sample rows.
Uses the SMALLEST file (44-1) so this runs instantly and safely.
"""

import pandas as pd

small_file = "../data/stage_labeled/iot23_CTU-IoT-Malware-Capture-44-1_with_stage.csv"

df = pd.read_csv(small_file, nrows=20)

print("Columns found:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nIf there's a 'stage' column, here are its unique values in this sample:")
if "stage" in df.columns:
    print(df["stage"].unique())
else:
    print("No column literally named 'stage' - check the column list above for the real name.")
