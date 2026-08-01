"""
Stacking Ensemble - combines RF, LSTM, GRU, and SSM into ONE final model.

Proper methodology: train/val/test split (60/20/20), not just train/test.
- Base models (RF, LSTM, GRU, SSM) are trained on the TRAIN set only.
- Each base model's predicted probabilities on the VAL set become the
  input features for a meta-classifier (Logistic Regression), which learns
  how to best combine the base models' opinions.
- Final evaluation happens ONLY on the untouched TEST set, using the full
  pipeline (base models -> meta-classifier), so the reported result is a
  fair, honest measure of the whole system.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================================================================
# 1. LOAD DATA AND SPLIT: 60% train, 20% validation, 20% test
# =========================================================================
data = np.load("../data/sequences/flow_sequences_mirai_v2_final.npz")
X, y = data["X"], data["y"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", list(classes))

n_samples, seq_len, n_features = X.shape
X_scaled = StandardScaler().fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

# first split off test (20%), then split the rest into train/val (75/25 -> 60/20 overall)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

n_classes = len(classes)
counts = np.bincount(y_train)
raw_weights = 1.0 / np.sqrt(counts)
raw_weights = raw_weights / raw_weights.min()
capped_weights = np.clip(raw_weights, 1.0, 5.0)
class_weights_tensor = torch.tensor(capped_weights, dtype=torch.float32)


def to_tensor(arr, dtype):
    return torch.tensor(arr, dtype=dtype)


X_train_t = to_tensor(X_train, torch.float32)
y_train_t = to_tensor(y_train, torch.long)
X_val_t = to_tensor(X_val, torch.float32)
X_test_t = to_tensor(X_test, torch.float32)

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, generator=g)

EPOCHS = 30  # slightly reduced from 40 since we're training 3 neural nets in one script


def train_torch_model(model, name):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)

    print(f"\nTraining {name}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  {name} epoch {epoch+1}/{EPOCHS}, loss: {total_loss/len(train_loader):.4f}")
    return model


# =========================================================================
# 2. DEFINE MODEL ARCHITECTURES (same as before)
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


# =========================================================================
# 3. TRAIN BASE MODELS
# =========================================================================
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
rf_model.fit(X_train_flat, y_train)

lstm_model = train_torch_model(LSTMClassifier(n_features, n_classes), "LSTM")
gru_model = train_torch_model(GRUClassifier(n_features, n_classes), "GRU")
ssm_model = train_torch_model(SSMClassifier(n_features, n_classes), "SSM")


# =========================================================================
# 4. GET BASE MODEL PROBABILITIES ON VAL SET -> BUILD META-FEATURES
# =========================================================================
def get_torch_probs(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


print("\nGetting base model predictions on validation set...")
rf_val_probs = rf_model.predict_proba(X_val_flat)
lstm_val_probs = get_torch_probs(lstm_model, X_val_t)
gru_val_probs = get_torch_probs(gru_model, X_val_t)
ssm_val_probs = get_torch_probs(ssm_model, X_val_t)

meta_features_val = np.concatenate([rf_val_probs, lstm_val_probs, gru_val_probs, ssm_val_probs], axis=1)
print(f"Meta-feature shape (val): {meta_features_val.shape}  (4 models x 5 classes = 20 columns)")

# =========================================================================
# 5. TRAIN META-CLASSIFIER ON VALIDATION PREDICTIONS
# =========================================================================
print("\nTraining meta-classifier (Logistic Regression) on base model predictions...")
meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
meta_model.fit(meta_features_val, y_val)

# =========================================================================
# 6. FINAL EVALUATION ON TEST SET (fully untouched until now)
# =========================================================================
print("\nGetting base model predictions on TEST set...")
rf_test_probs = rf_model.predict_proba(X_test_flat)
lstm_test_probs = get_torch_probs(lstm_model, X_test_t)
gru_test_probs = get_torch_probs(gru_model, X_test_t)
ssm_test_probs = get_torch_probs(ssm_model, X_test_t)

meta_features_test = np.concatenate([rf_test_probs, lstm_test_probs, gru_test_probs, ssm_test_probs], axis=1)
final_predictions = meta_model.predict(meta_features_test)

print("\n" + "=" * 70)
print("INDIVIDUAL BASE MODEL ACCURACY ON TEST SET (for reference)")
print("=" * 70)
print(f"Random Forest: {accuracy_score(y_test, np.argmax(rf_test_probs, axis=1)):.4f}")
print(f"LSTM:          {accuracy_score(y_test, np.argmax(lstm_test_probs, axis=1)):.4f}")
print(f"GRU:           {accuracy_score(y_test, np.argmax(gru_test_probs, axis=1)):.4f}")
print(f"SSM:           {accuracy_score(y_test, np.argmax(ssm_test_probs, axis=1)):.4f}")

print("\n" + "=" * 70)
print("FINAL STACKING ENSEMBLE RESULT")
print("=" * 70)
print(classification_report(y_test, final_predictions, target_names=classes))
print("Confusion Matrix:")
print("Order:", list(classes))
print(confusion_matrix(y_test, final_predictions))

# =========================================================================
# 7. SAVE THE FULL ENSEMBLE
# =========================================================================
import joblib
import os
os.makedirs("../data/models/ensemble", exist_ok=True)
joblib.dump(rf_model, "../data/models/ensemble/rf_base.joblib")
joblib.dump(meta_model, "../data/models/ensemble/meta_classifier.joblib")
torch.save(lstm_model.state_dict(), "../data/models/ensemble/lstm_base.pt")
torch.save(gru_model.state_dict(), "../data/models/ensemble/gru_base.pt")
torch.save(ssm_model.state_dict(), "../data/models/ensemble/ssm_base.pt")
print("\nFull ensemble saved to ../data/models/ensemble/")
