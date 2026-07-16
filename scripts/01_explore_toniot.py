import pandas as pd

df = pd.read_csv("../data/raw/train_test_network.csv")

print(df.shape)
print(df.columns.tolist())
print(df['type'].value_counts())
print(df['label'].value_counts())