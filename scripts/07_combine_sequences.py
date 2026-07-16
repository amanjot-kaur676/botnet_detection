import pandas as pd

df_34 = pd.read_csv("../data/sequences/sequences_34-1_standard.csv")
df_49 = pd.read_csv("../data/sequences/sequences_49-1_standard.csv")
df_52 = pd.read_csv("../data/sequences/sequences_52-1_standard.csv")

combined = pd.concat([df_34, df_49, df_52], ignore_index=True)
combined.to_csv("../data/sequences/sequences_combined.csv", index=False)

print("Total combined windows:", len(combined))
print(combined["scenario"].value_counts())
print(combined["dominant_stage"].value_counts())