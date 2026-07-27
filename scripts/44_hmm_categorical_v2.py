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

"""
HMM baseline v2 - Categorical HMM, a fairer test of the classical "state model" approach.

The original GaussianHMM assumed continuous, bell-curve-shaped features, which was a poor
fit for our mostly one-hot/categorical feature set (12 of 18 features were 0/1 flags) and
led to a near-collapse (39% accuracy, 0% recall on two classes).

This version discretizes ALL features into a small number of bins, then combines them into
ONE categorical symbol per timestep, and uses hmmlearn's CategoricalHMM - which makes no
Gaussian assumption at all. This is a fairer test of whether the classical state-based
approach can work here, given the right emission model for this kind of data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import classification_report, confusion_matrix
from hmmlearn.hmm import CategoricalHMM

SEED = 42
N_HIDDEN_STATES = 4
N_BINS_PER_FEATURE = 3  # discretize each of the 19 features into 3 bins (low/med/high)

data = np.load("../data/sequences/flow_sequences_mirai_v2_final.npz")
X, y = data["X"], data["y"]

n_samples, seq_len, n_features = X.shape
print(f"X shape: {X.shape}")

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", list(classes))

# --- Discretize each feature into bins, then combine into ONE symbol per timestep ---
# e.g. 19 features x 3 bins each -> combined into a single integer symbol (0 to 3^19-1 in theory,
# but we reduce dimensionality first to keep the symbol space manageable)
X_flat_rows = X.reshape(-1, n_features)  # (n_samples*10, n_features)

discretizer = KBinsDiscretizer(n_bins=N_BINS_PER_FEATURE, encode="ordinal", strategy="quantile")
X_binned = discretizer.fit_transform(X_flat_rows).astype(int)  # (n_samples*10, n_features), values 0..2

# combine all features into one symbol using a simple base-3 encoding on a REDUCED feature set
# (using all 19 raw would create an unmanageably large symbol space) - we use the 5 most
# structurally important features based on earlier Random Forest importance: fanout_count,
# repeat_count, duration, orig_pkts, and the S0 conn_state flag
IMPORTANT_FEATURE_IDX = [0, 17, 18, 3, 5]  # duration, fanout_count, repeat_count, orig_pkts, S0
X_binned_reduced = X_binned[:, IMPORTANT_FEATURE_IDX]

symbols = np.zeros(len(X_binned_reduced), dtype=int)
for col in range(X_binned_reduced.shape[1]):
    symbols = symbols * N_BINS_PER_FEATURE + X_binned_reduced[:, col]

symbols = symbols.reshape(n_samples, seq_len)  # back to (n_samples, 10)
n_symbols = N_BINS_PER_FEATURE ** len(IMPORTANT_FEATURE_IDX)
print(f"Combined symbol space size: {n_symbols}")

X_train, X_test, y_train, y_test = train_test_split(
    symbols, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Train one Categorical HMM per class ---
class_models = {}

for class_idx, class_name in enumerate(classes):
    class_sequences = X_train[y_train == class_idx]  # (n_this_class, 10)
    print(f"\nTraining Categorical HMM for class '{class_name}' on {len(class_sequences)} sequences...")

    concatenated = class_sequences.reshape(-1, 1)
    lengths = [seq_len] * len(class_sequences)

    model = CategoricalHMM(
        n_components=N_HIDDEN_STATES,
        n_iter=50,
        random_state=SEED,
        n_features=n_symbols,
    )
    model.fit(concatenated, lengths)
    class_models[class_name] = model

# --- Classify test sequences: score under each class-HMM, pick the best ---
print("\nScoring test sequences under each class HMM...")

y_pred_names = []
for i in range(len(X_test)):
    seq = X_test[i].reshape(-1, 1)
    scores = {}
    for class_name, model in class_models.items():
        try:
            scores[class_name] = model.score(seq)
        except Exception:
            scores[class_name] = -np.inf
    best_class = max(scores, key=scores.get)
    y_pred_names.append(best_class)

    if (i + 1) % 5000 == 0:
        print(f"  ...scored {i+1}/{len(X_test)} test sequences")

y_pred = le.transform(y_pred_names)

print("\n=== Classification Report (Categorical HMM - fairer classical state model test) ===")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n=== Confusion Matrix ===")
print("Order:", list(classes))
print(confusion_matrix(y_test, y_pred))

import joblib
import os
os.makedirs("../data/models", exist_ok=True)
joblib.dump(class_models, "../data/models/hmm_categorical_v2.joblib")
print("\nModels saved to ../data/models/hmm_categorical_v2.joblib")
