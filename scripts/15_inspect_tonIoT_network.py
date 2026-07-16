"""
Step 11a: Look at TON_IoT's train_test_network.csv
- What columns exist
- What attack types exist (looking for dos, exfiltration especially)
This file is small (29.9 MB) so we can load it directly, no chunking needed.
"""

import pandas as pd

file_path = "../data/raw/train_test_network.csv"

df = pd.read_csv(file_path)

print("Columns found:")
print(list(df.columns))

print(f"\nTotal rows: {len(df)}")

# TON_IoT network csv usually has 'label' (0/1) and 'type' (attack category)
for col in ["label", "type"]:
    if col in df.columns:
        print(f"\nUnique values in '{col}':")
        print(df[col].value_counts())
    else:
        print(f"\nNo column named '{col}' - check the column list above.")

print("\nFirst 5 rows:")
print(df.head())
