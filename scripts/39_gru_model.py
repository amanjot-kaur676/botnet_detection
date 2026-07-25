"""
GRU model - a lighter, faster alternative to LSTM for sequence modeling.
Same data, same capped class-weighting approach that worked best for LSTM,
just a different recurrent architecture (fewer internal gates than LSTM,
so it trains faster and has fewer parameters).
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

data = np.load("../data/sequences/flow_sequences_mirai_perdevice_final.npz")
X, y = data["X"], data["y"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

n_samples, seq_len, n_features = X.shape
X_scaled = StandardScaler().fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)

# same capped weighting scheme that worked best for the LSTM
counts = np.bincount(y_train)
raw_weights = 1.0 / np.sqrt(counts)
raw_weights = raw_weights / raw_weights.min()
MAX_RATIO = 5.0
capped_weights = np.clip(raw_weights, 1.0, MAX_RATIO)
print("Class weights (capped):", dict(zip(le.classes_, capped_weights.round(2))))
class_weights_tensor = torch.tensor(capped_weights, dtype=torch.float32)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, generator=g)


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
        x = self.relu(self.fc1(h_n[-1]))
        x = self.dropout(x)
        return self.fc2(x)


model = GRUClassifier(n_features, len(le.classes_))
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

print("\n=== Classification Report (GRU, capped weights, 40 epochs) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Confusion Matrix ===")
print("Order:", list(le.classes_))
print(confusion_matrix(y_test, y_pred))

import os
os.makedirs("../data/models", exist_ok=True)
torch.save(model.state_dict(), "../data/models/gru_capped_weights.pt")
print("\nModel saved to ../data/models/gru_capped_weights.pt")
