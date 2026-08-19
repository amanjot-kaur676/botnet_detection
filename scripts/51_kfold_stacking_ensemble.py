"""
K-Fold Stacking Ensemble - improvement over script 46.

Previous ensemble (script 46) trained base models on only 60% of the data
(train split), reserving 20% purely for generating meta-features. This
wastes data the base models could have learned from.

This version uses 4-fold cross-validation on the train+val portion (80%
of all data): for each fold, base models train on the other 3 folds and
predict on the held-out fold. Stacking these out-of-fold predictions
together gives meta-features built from ALL 80% of training data, with
no leakage (a model never predicts on data it was trained on).

The final base models (used for actual test predictions) are then trained
on the FULL 80% train+val portion, maximizing their own training data too.

NOTE: this trains each neural net (LSTM/GRU/SSM) 4 times (once per fold)
plus once more on the full data = 5x the training of script 46. Expect this
to take considerably longer - likely 45-75 minutes total. Let it run.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
N_FOLDS = 4
EPOCHS = 20  # reduced from 30/40 since we're training many more times total

# =========================================================================
# 1. LOAD DATA: 80% train+val (for CV), 20% test (untouched until the end)
# =========================================================================
data = np.load("../data/sequences/flow_sequences_beaconing_v3_final.npz")
X, y = data["X"], data["y"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", list(classes))

n_samples, seq_len, n_features = X.shape
print(f"X shape: {X.shape}")
X_scaled = StandardScaler().fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
print(f"Train+Val (for CV): {len(X_trainval)}, Test: {len(X_test)}")

n_classes = len(classes)


def make_class_weights(y_subset):
    counts = np.bincount(y_subset, minlength=n_classes)
    raw = 1.0 / np.sqrt(np.maximum(counts, 1))
    raw = raw / raw.min()
    capped = np.clip(raw, 1.0, 5.0)
    return torch.tensor(capped, dtype=torch.float32)


# =========================================================================
# 2. MODEL DEFINITIONS
# =========================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc2(self.relu(self.fc1(h_n[-1])))


class GRUClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc2(self.dropout(self.relu(self.fc1(h_n[-1]))))


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.log(0.1 * torch.ones(d_model)))

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        A = -torch.exp(self.log_A_real)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A_bar = torch.exp(A * dt)
        B_bar = (A_bar - 1) / (A + 1e-8) * self.B
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(-1)
            y_t = (h * self.C.unsqueeze(0)).sum(-1) + self.D.unsqueeze(0) * x_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class SSMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, d_state=32):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.ssm(x)
        x = self.activation(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_torch_model(model_class, X_tr, y_tr, name, fold_label=""):
    model = model_class(n_features, n_classes)
    weights = make_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True, generator=g)

    print(f"  Training {name} {fold_label}...")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def get_torch_probs(model, X_arr):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        return torch.softmax(model(X_t), dim=1).numpy()


# =========================================================================
# 3. GENERATE OUT-OF-FOLD META-FEATURES VIA 4-FOLD CV
# =========================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_meta_features = np.zeros((len(X_trainval), n_classes * 4))  # 4 base models x n_classes
X_trainval_flat = X_trainval.reshape(len(X_trainval), -1)

for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X_trainval_flat, y_trainval)):
    print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} ===")
    X_ft, X_fv = X_trainval[fold_train_idx], X_trainval[fold_val_idx]
    y_ft, y_fv = y_trainval[fold_train_idx], y_trainval[fold_val_idx]
    X_ft_flat, X_fv_flat = X_ft.reshape(len(X_ft), -1), X_fv.reshape(len(X_fv), -1)

    rf = RandomForestClassifier(n_estimators=250, max_depth=20, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_ft_flat, y_ft)
    rf_fold_probs = rf.predict_proba(X_fv_flat)

    lstm = train_torch_model(LSTMClassifier, X_ft, y_ft, "LSTM", f"(fold {fold_idx+1})")
    gru = train_torch_model(GRUClassifier, X_ft, y_ft, "GRU", f"(fold {fold_idx+1})")
    ssm = train_torch_model(SSMClassifier, X_ft, y_ft, "SSM", f"(fold {fold_idx+1})")

    lstm_fold_probs = get_torch_probs(lstm, X_fv)
    gru_fold_probs = get_torch_probs(gru, X_fv)
    ssm_fold_probs = get_torch_probs(ssm, X_fv)

    fold_meta = np.concatenate([rf_fold_probs, lstm_fold_probs, gru_fold_probs, ssm_fold_probs], axis=1)
    oof_meta_features[fold_val_idx] = fold_meta

print("\nOut-of-fold meta-features generated for all training data.")

# =========================================================================
# 4. TRAIN META-CLASSIFIER ON OUT-OF-FOLD PREDICTIONS
# =========================================================================
val_counts = np.bincount(y_trainval)
raw_sw = 1.0 / np.sqrt(val_counts)
raw_sw = raw_sw / raw_sw.min()
capped_sw = np.clip(raw_sw, 1.0, 5.0)
sample_weights = capped_sw[y_trainval]

print("\nTraining meta-classifier (Random Forest, capped weights) on out-of-fold predictions...")
meta_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1)
meta_model.fit(oof_meta_features, y_trainval, sample_weight=sample_weights)

# =========================================================================
# 5. TRAIN FINAL BASE MODELS ON FULL TRAIN+VAL, PREDICT ON TEST
# =========================================================================
print("\n=== Training final base models on FULL train+val data ===")
X_trainval_flat_full = X_trainval.reshape(len(X_trainval), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

rf_final = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_final.fit(X_trainval_flat_full, y_trainval)

lstm_final = train_torch_model(LSTMClassifier, X_trainval, y_trainval, "LSTM", "(final)")
gru_final = train_torch_model(GRUClassifier, X_trainval, y_trainval, "GRU", "(final)")
ssm_final = train_torch_model(SSMClassifier, X_trainval, y_trainval, "SSM", "(final)")

rf_test_probs = rf_final.predict_proba(X_test_flat)
lstm_test_probs = get_torch_probs(lstm_final, X_test)
gru_test_probs = get_torch_probs(gru_final, X_test)
ssm_test_probs = get_torch_probs(ssm_final, X_test)

meta_features_test = np.concatenate([rf_test_probs, lstm_test_probs, gru_test_probs, ssm_test_probs], axis=1)
final_predictions = meta_model.predict(meta_features_test)

# =========================================================================
# 6. RESULTS
# =========================================================================
print("\n" + "=" * 70)
print("INDIVIDUAL FINAL BASE MODEL ACCURACY ON TEST SET")
print("=" * 70)
print(f"Random Forest: {accuracy_score(y_test, np.argmax(rf_test_probs, axis=1)):.4f}")
print(f"LSTM:          {accuracy_score(y_test, np.argmax(lstm_test_probs, axis=1)):.4f}")
print(f"GRU:           {accuracy_score(y_test, np.argmax(gru_test_probs, axis=1)):.4f}")
print(f"SSM:           {accuracy_score(y_test, np.argmax(ssm_test_probs, axis=1)):.4f}")

print("\n" + "=" * 70)
print("K-FOLD STACKING ENSEMBLE - FINAL RESULT")
print("=" * 70)
print(f"Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
print(f"Macro F1: {f1_score(y_test, final_predictions, average='macro'):.4f}")
print(classification_report(y_test, final_predictions, target_names=classes))
print("Confusion Matrix:")
print("Order:", list(classes))
print(confusion_matrix(y_test, final_predictions))

import joblib
import os
os.makedirs("../data/models/ensemble_v2", exist_ok=True)
joblib.dump(rf_final, "../data/models/ensemble_v2/rf_final.joblib")
joblib.dump(meta_model, "../data/models/ensemble_v2/meta_rf_kfold.joblib")
torch.save(lstm_final.state_dict(), "../data/models/ensemble_v2/lstm_final.pt")
torch.save(gru_final.state_dict(), "../data/models/ensemble_v2/gru_final.pt")
torch.save(ssm_final.state_dict(), "../data/models/ensemble_v2/ssm_final.pt")
print("\nFull k-fold ensemble saved to ../data/models/ensemble_v2/")
