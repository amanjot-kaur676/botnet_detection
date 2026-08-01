"""
State Space Model (S4D-style), built from scratch in plain PyTorch.

Unlike LSTM/GRU (gated recurrent networks), this is a genuine state-space model:
each feature channel has its own simple linear dynamical system (a diagonal state
matrix A, input matrix B, output matrix C), discretized and run forward through
time. This is the same family of architecture behind S4/Mamba, simplified to run
on CPU with no special library (avoids the install issues we hit with
mamba-ssm/tensorflow on Python 3.14).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

data = np.load("../data/sequences/flow_sequences_mirai_v2_final.npz")
X, y = data["X"], data["y"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

n_samples, seq_len, n_features = X.shape
print(f"X shape: {X.shape}")
X_scaled = StandardScaler().fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)

# same capped class weighting that worked best for LSTM/GRU
counts = np.bincount(y_train)
raw_weights = 1.0 / np.sqrt(counts)
raw_weights = raw_weights / raw_weights.min()
capped_weights = np.clip(raw_weights, 1.0, 5.0)
print("Class weights (capped):", dict(zip(le.classes_, capped_weights.round(2))))
class_weights_tensor = torch.tensor(capped_weights, dtype=torch.float32)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, generator=g)


class S4DLayer(nn.Module):
    """
    A simplified diagonal state-space layer (S4D-style).
    Each of d_model channels has its own independent state-space system with
    d_state internal state dimensions. Discretized via zero-order hold, then
    run forward through time with a simple sequential scan (fine for our
    short 10-step sequences - no need for the parallel-scan tricks full S4
    implementations use for very long sequences).
    """
    def __init__(self, d_model, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is kept negative (stable dynamics) via -exp(log_A_real)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.log(0.1 * torch.ones(d_model)))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        A = -torch.exp(self.log_A_real)          # (d_model, d_state), always negative -> stable
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)

        # zero-order hold discretization
        A_bar = torch.exp(A * dt)                          # (d_model, d_state)
        B_bar = (A_bar - 1) / (A + 1e-8) * self.B           # (d_model, d_state)

        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(-1)  # (batch, d_model, d_state)
            y_t = (h * self.C.unsqueeze(0)).sum(-1) + self.D.unsqueeze(0) * x_t  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


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
        x = self.input_proj(x)             # (batch, seq_len, d_model)
        x = self.ssm(x)                    # (batch, seq_len, d_model)
        x = self.activation(x)
        x = self.norm(x)
        x = x.mean(dim=1)                  # pool across time (sequence-level summary)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


model = SSMClassifier(n_features, len(le.classes_))
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

print(model)

EPOCHS = 40
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
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).argmax(dim=1).numpy()

print("\n=== Classification Report (S4D-style State Space Model) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Confusion Matrix ===")
print("Order:", list(le.classes_))
print(confusion_matrix(y_test, y_pred))

import os
os.makedirs("../data/models", exist_ok=True)
torch.save(model.state_dict(), "../data/models/ssm_s4d_v2.pt")
print("\nModel saved to ../data/models/ssm_s4d_v2.pt")
