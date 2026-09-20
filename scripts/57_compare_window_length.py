"""
Window Length Comparison: does a longer 20-flow window help, specifically
with the persistent C2-vs-Scan confusion that survived every earlier fix?

Uses Random Forest (same fixed model as the tier comparison) so we isolate
the effect of WINDOW LENGTH, not model choice or feature set (same 29
features as the "Selected" tier, just a longer window this time).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42

data = np.load("../data/sequences/flow_sequences_seq20_final.npz")
X, y = data["X"], data["y"]
print(f"X shape: {X.shape}  (should be [n_samples, 20, 29])")

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

n_samples, seq_len, n_features = X.shape
X_flat = X.reshape(n_samples, seq_len * n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=classes))
print("Confusion Matrix:")
print("Order:", list(classes))
print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 70)
print("COMPARISON: 10-flow window (previous 'Selected' tier) vs 20-flow window")
print("=" * 70)
print(f"10-flow window: 71.0% accuracy, 0.680 macro F1")
print(f"20-flow window: {acc:.1%} accuracy, {macro_f1:.3f} macro F1")
print(f"Difference: {(acc - 0.71)*100:+.1f} accuracy points, {(macro_f1 - 0.680):+.3f} macro F1")
