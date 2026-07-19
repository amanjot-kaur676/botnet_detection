"""
Step 30 (PyTorch version): Baseline state-sequence model (LSTM) for stage classification.

Same design as the TensorFlow version - just using PyTorch since TensorFlow
doesn't yet support Python 3.14.

Input: flow_sequences_mirai_final.npz
  X shape: (102698, 10, 5)  -> 10 timesteps, 5 features per timestep
  y: stage labels (Scan/Infect/C2/Impact/Benign)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Fix random seeds everywhere for reproducible results ---
# Without this, results change noticeably between runs of identical code,
# which makes it impossible to trust comparisons between model versions.
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# --- Load data ---
data = np.load("../data/sequences/flow_sequences_mirai_final.npz")
X, y = data["X"], data["y"]

print(f"X shape: {X.shape}")
print(f"y distribution:\n{pd.Series(y).value_counts()}")

# --- Encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# --- Scale features ---
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)

# --- Train/test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --- Convert to PyTorch tensors ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=g)


# --- Define the LSTM model ---
class StageLSTM(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        # use the final hidden state (summary of the whole sequence)
        x = h_n[-1]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


n_classes = len(label_encoder.classes_)
model = StageLSTM(n_features=n_features, n_classes=n_classes).to(device)

# --- Class weighting to fix severe imbalance (Infect was ~2.7% of data and
# was never predicted at all in the unweighted run) ---
# Weight = inverse of class frequency, so rare classes matter more to the loss
class_counts = pd.Series(y_train).value_counts().sort_index()
class_weights = (1.0 / class_counts).values
class_weights = class_weights / class_weights.sum() * n_classes  # normalize
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\nClass weights (order: {list(label_encoder.classes_)}): {class_weights_t.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# --- Train ---
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}")

# --- Evaluate ---
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

y_test_np = y_test_t.cpu().numpy()

print("\n=== Classification Report ===")
print(classification_report(y_test_np, y_pred, target_names=label_encoder.classes_, zero_division=0))

print("\n=== Confusion Matrix ===")
print("Rows = actual, Columns = predicted")
print("Order:", list(label_encoder.classes_))
print(confusion_matrix(y_test_np, y_pred))

# --- Save the model ---
import os
os.makedirs("../data/models", exist_ok=True)
torch.save(model.state_dict(), "../data/models/baseline_lstm_stage_classifier.pt")
print("\nModel saved to ../data/models/baseline_lstm_stage_classifier.pt")