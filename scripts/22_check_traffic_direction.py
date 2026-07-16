"""
Step 21b: Check traffic direction - which IP is the infected/attacking device,
and which is the target? This determines whether "src_bytes" means
"data leaving the infected device" or something else entirely.
"""

import pandas as pd

path = "../data/raw/train_test_network.csv"
df = pd.read_csv(path)

print("Unique src_ip values overall (top 15):")
print(df["src_ip"].value_counts().head(15))

print("\nUnique dst_ip values overall (top 15):")
print(df["dst_ip"].value_counts().head(15))

print("\n--- Per attack type: most common src_ip and dst_ip ---")
for attack_type in df["type"].unique():
    subset = df[df["type"] == attack_type]
    top_src = subset["src_ip"].value_counts().head(3)
    top_dst = subset["dst_ip"].value_counts().head(3)
    print(f"\nType: {attack_type}")
    print(f"  Top src_ip: {dict(top_src)}")
    print(f"  Top dst_ip: {dict(top_dst)}")
