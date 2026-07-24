"""
Random Forest baseline - non-sequential comparison model.

Uses the SAME final dataset as the LSTM (18 features x 10 timesteps),
but flattens each sequence into one flat 180-length vector, ignoring
temporal order. This tests whether sequence modeling is actually needed,
or whether a flat classifier does just as well on this data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42

data = np.load("../data/sequences/flow_sequences_mirai_perdevice_final.npz")
X, y = data["X"], data["y"]

print(f"X shape: {X.shape}")

# Flatten: (n_samples, 10, 18) -> (n_samples, 180)
# This throws away the ORDER of the 10 timesteps - the model just sees
# 180 numbers with no notion of "this came before that."
n_samples, seq_len, n_features = X.shape
X_flat = X.reshape(n_samples, seq_len * n_features)

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# class_weight="balanced" handles imbalance automatically for Random Forest,
# no manual weight tuning needed like we had to do for the LSTM
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,  # use all CPU cores - this is fast enough to not need GPU
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Classification Report (Random Forest baseline) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Confusion Matrix ===")
print("Order:", list(le.classes_))
print(confusion_matrix(y_test, y_pred))

# feature importance - which of the 180 flattened features mattered most
# useful for your paper's discussion section
importances = model.feature_importances_
feature_names = []
base_names = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
              "S0", "SF", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "SHR", "OTH",
              "vuln_port", "fanout_count"]
for t in range(seq_len):
    for f in base_names:
        feature_names.append(f"{f}_t{t}")

importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\n=== Top 15 most important features ===")
print(importance_df.head(15).to_string(index=False))

import joblib
import os
os.makedirs("../data/models", exist_ok=True)
joblib.dump(model, "../data/models/random_forest_baseline.joblib")
print("\nModel saved to ../data/models/random_forest_baseline.joblib")
