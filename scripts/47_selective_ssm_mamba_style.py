"""
Selective State Space Model (Mamba-style "S6" layer), simplified and built
from scratch in plain PyTorch.

Difference from our earlier S4D layer: there, the dynamics (A, B, C, dt) were
FIXED - the same "physics" applied to every input regardless of content.
Here, B, C, and the discretization step (dt) are computed FROM THE INPUT
at every timestep via small learned projections - this is the "selective"
mechanism that lets Mamba dynamically decide how much to "pay attention to"
or "let through" at each step, rather than treating every timestep uniformly.

This is a simplified version of the real S6 layer (no hardware-aware parallel
scan, since our sequences are only 10 steps long - a plain sequential loop
is fine here), not the literal mamba-ssm package (which needs custom CUDA
kernels and won't run on a Windows CPU-only setup).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SelectiveSSMLayer(nn.Module):
    """
    Simplified Mamba-style selective SSM layer.
    A stays fixed per-channel (like S4D), but B, C, and dt are computed
    FROM THE INPUT at every timestep - this is the key "selective" idea.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A fixed per channel, kept negative for stability (same idea as S4D)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.D = nn.Parameter(torch.ones(d_model))

        # selective projections - these compute B, C, and dt FROM the input
        # at each timestep, instead of using fixed values
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        A = -torch.exp(self.log_A_real)  # (d_model, d_state), fixed

        # selective, input-dependent parameters - computed per timestep
        dt = F.softplus(self.dt_proj(x))       # (batch, seq_len, d_model) - how big a "step" to take, per input
        B_t = self.B_proj(x)                    # (batch, seq_len, d_state) - what to let INTO the state
        C_t = self.C_proj(x)                    # (batch, seq_len, d_state) - what to read OUT of the state

        h = torch.zeros(batch, d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]            # (batch, d_model)
            dt_t = dt[:, t, :]          # (batch, d_model)
            B_now = B_t[:, t, :]        # (batch, d_state)
            C_now = C_t[:, t, :]        # (batch, d_state)

            # discretize dynamically using this timestep's own dt
            A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))          # (batch, d_model, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_now.unsqueeze(1)                 # (batch, d_model, d_state) - simplified Euler approx, same as Mamba uses

            h = A_bar * h + B_bar * x_t.unsqueeze(-1)                       # (batch, d_model, d_state)
            y_t = (h * C_now.unsqueeze(1)).sum(-1) + self.D.unsqueeze(0) * x_t  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class SelectiveSSMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.ssm = SelectiveSSMLayer(d_model, d_state)
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


model = SelectiveSSMClassifier(n_features, len(le.classes_))
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

print("\n=== Classification Report (Selective SSM - Mamba-style) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Confusion Matrix ===")
print("Order:", list(le.classes_))
print(confusion_matrix(y_test, y_pred))

import os
os.makedirs("../data/models", exist_ok=True)
torch.save(model.state_dict(), "../data/models/selective_ssm_mamba_style.pt")
print("\nModel saved to ../data/models/selective_ssm_mamba_style.pt")
