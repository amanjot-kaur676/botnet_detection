"""
Step 30: Baseline state-sequence model (LSTM) for stage classification.

Input: flow_sequences_mirai_final.npz
  X shape: (102698, 10, 5)  -> 10 timesteps, 5 features per timestep
  y: stage labels (Scan/Infect/C2/Impact/Benign)

This is a FIRST baseline - simple architecture, just to get an end-to-end
working pipeline and real numbers. We tune/improve after seeing results.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Load data ---
data = np.load("../data/sequences/flow_sequences_mirai_final.npz")
X, y = data["X"], data["y"]

print(f"X shape: {X.shape}")
print(f"y distribution:\n{pd.Series(y).value_counts()}")

# --- Encode labels (Scan/Infect/C2/Impact/Benign -> 0/1/2/3/4) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# --- Scale features (important for neural nets - keeps values in a similar range) ---
# reshape to 2D for the scaler, then back to 3D sequence shape
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)

# --- Train/test split (stratified so rare stages like Infect are represented in both) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --- Build a simple LSTM model ---
n_classes = len(label_encoder.classes_)

model = models.Sequential([
    layers.Input(shape=(n_timesteps, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    verbose=1,
)

# --- Evaluate ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n=== Confusion Matrix ===")
print("Rows = actual, Columns = predicted")
print("Order:", list(label_encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# --- Save the model for later use ---
model.save("../data/models/baseline_lstm_stage_classifier.keras")
print("\nModel saved to ../data/models/baseline_lstm_stage_classifier.keras")
