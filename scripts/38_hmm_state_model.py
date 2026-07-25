"""
HMM baseline - the classical "state model" approach.

Unlike LSTM/Random Forest (which learn one function mapping input -> class),
this trains ONE small HMM PER CLASS, each learning "what does a typical
sequence of this stage look like as it evolves." Classification works by
asking each class's HMM "how likely did YOU produce this sequence?" and
picking whichever HMM scores highest.

This is the classical state-space approach to sequence modeling - directly
matches "state model" terminology.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from hmmlearn.hmm import GaussianHMM

SEED = 42
N_HIDDEN_STATES = 4  # each class-HMM has this many internal hidden states

data = np.load("../data/sequences/flow_sequences_mirai_perdevice_final.npz")
X, y = data["X"], data["y"]

n_samples, seq_len, n_features = X.shape
print(f"X shape: {X.shape}")

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", list(classes))

# scale features - same as before, important for numerical stability in HMM too
X_scaled = StandardScaler().fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Train one HMM per class ---
class_models = {}

for class_idx, class_name in enumerate(classes):
    class_sequences = X_train[y_train == class_idx]  # shape (n_this_class, 10, 18)
    print(f"\nTraining HMM for class '{class_name}' on {len(class_sequences)} sequences...")

    # hmmlearn wants all sequences concatenated along time, with a lengths array
    concatenated = class_sequences.reshape(-1, n_features)
    lengths = [seq_len] * len(class_sequences)

    model = GaussianHMM(
        n_components=N_HIDDEN_STATES,
        covariance_type="diag",
        min_covar=1e-3,   # regularization - avoids numerical errors on near-constant features
        n_iter=50,
        random_state=SEED,
    )
    model.fit(concatenated, lengths)
    class_models[class_name] = model

# --- Classify test sequences: score under each class-HMM, pick the best ---
print("\nScoring test sequences under each class HMM (this may take a few minutes)...")

y_pred_names = []
for i in range(len(X_test)):
    seq = X_test[i]  # (10, 18)
    scores = {}
    for class_name, model in class_models.items():
        try:
            scores[class_name] = model.score(seq)  # log-likelihood
        except Exception:
            scores[class_name] = -np.inf
    best_class = max(scores, key=scores.get)
    y_pred_names.append(best_class)

    if (i + 1) % 5000 == 0:
        print(f"  ...scored {i+1}/{len(X_test)} test sequences")

y_pred = le.transform(y_pred_names)

print("\n=== Classification Report (HMM - classical state model) ===")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n=== Confusion Matrix ===")
print("Order:", list(classes))
print(confusion_matrix(y_test, y_pred))

import joblib
import os
os.makedirs("../data/models", exist_ok=True)
joblib.dump(class_models, "../data/models/hmm_per_class_models.joblib")
print("\nModels saved to ../data/models/hmm_per_class_models.joblib")
