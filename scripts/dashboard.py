"""
Project Dashboard - Mirai-Lineage IoT Botnet Stage Detection

Run with: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

st.set_page_config(page_title="Mirai Botnet Stage Detection", layout="wide")

# =========================================================================
# MODEL ARCHITECTURE DEFINITIONS (needed to load saved PyTorch models)
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


CLASS_NAMES = ["Benign", "C2", "Impact", "Infect", "Scan"]
N_FEATURES = 19


# =========================================================================
# LOAD DATA AND MODELS (cached so this only runs once)
# =========================================================================
@st.cache_resource
def load_everything():
    data = np.load("../data/sequences/flow_sequences_mirai_v2_final.npz")
    X, y = data["X"], data["y"]

    rf_model = joblib.load("../data/models/ensemble/rf_base.joblib")
    meta_model = joblib.load("../data/models/ensemble/meta_rf.joblib")

    lstm_model = LSTMClassifier(N_FEATURES, len(CLASS_NAMES))
    lstm_model.load_state_dict(torch.load("../data/models/ensemble/lstm_base.pt"))
    lstm_model.eval()

    gru_model = GRUClassifier(N_FEATURES, len(CLASS_NAMES))
    gru_model.load_state_dict(torch.load("../data/models/ensemble/gru_base.pt"))
    gru_model.eval()

    ssm_model = SSMClassifier(N_FEATURES, len(CLASS_NAMES))
    ssm_model.load_state_dict(torch.load("../data/models/ensemble/ssm_base.pt"))
    ssm_model.eval()

    return X, y, rf_model, meta_model, lstm_model, gru_model, ssm_model


X, y, rf_model, meta_model, lstm_model, gru_model, ssm_model = load_everything()


def get_torch_probs(model, x_tensor):
    with torch.no_grad():
        logits = model(x_tensor)
        return torch.softmax(logits, dim=1).numpy()[0]


# =========================================================================
# PAGE LAYOUT
# =========================================================================
st.title("IoT Botnet Detection: Mirai & Mirai-Variant Lifecycle Classification")
st.markdown("""
This project detects which stage of the attack lifecycle (**Scan → Infect → C2 → Impact**)
a Mirai-family IoT botnet infection is in, using a sequence of network flow features.
""")

tab1, tab2, tab3 = st.tabs(["Project Overview", "Model Comparison", "Live Classification Demo"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.header("Attack Lifecycle")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1. Scan", "Probing for targets")
    col2.metric("2. Infect", "Breaking in")
    col3.metric("3. C2", "Awaiting commands")
    col4.metric("4. Impact", "Attacking / exfiltrating")

    st.header("Datasets Used")
    st.markdown("""
    - **IoT-23**: real network captures of Mirai-lineage botnet-infected IoT devices
    - **TON_IoT**: network + host telemetry, used to validate CPU/byte-volume detection hypotheses
    """)

    st.header("Final Dataset")
    st.write(f"**{X.shape[0]:,}** sequences, **{X.shape[1]}** timesteps per sequence, **{X.shape[2]}** features per timestep")
    st.write("Stage distribution:")
    st.bar_chart(pd.Series(y).value_counts())

# --- TAB 2: MODEL COMPARISON ---
with tab2:
    st.header("7 Models Tested")
    results_df = pd.DataFrame({
        "Model": ["Stacking Ensemble", "Random Forest", "LSTM", "GRU",
                  "S4D State Space Model", "Selective SSM (Mamba-style)",
                  "Categorical HMM", "Gaussian HMM"],
        "Accuracy": [0.65, 0.63, 0.61, 0.61, 0.60, 0.60, 0.46, 0.39],
        "Type": ["Ensemble", "Non-sequential", "Sequential", "Sequential",
                 "Sequential (SSM)", "Sequential (SSM)", "Classical", "Classical"],
    })
    st.dataframe(results_df, use_container_width=True)
    st.bar_chart(results_df.set_index("Model")["Accuracy"])

    st.markdown("""
    **Key finding:** Four independent deep sequence architectures (LSTM, GRU, S4D, Selective SSM)
    converged to nearly identical performance (~60-61%), while the non-sequential Random Forest
    outperformed all of them individually (63%). The final stacking ensemble, combining all four,
    achieved the best result (65%) by leveraging each model's complementary strengths.
    """)

# --- TAB 3: LIVE DEMO ---
with tab3:
    st.header("Try It: Classify a Random Sequence")
    st.write("Pick a random real sequence from the test data and see what each model predicts.")

    if st.button("Pick a random sequence"):
        idx = np.random.randint(0, len(X))
        sample_X = X[idx]
        true_label = y[idx]

        st.session_state["sample_X"] = sample_X
        st.session_state["true_label"] = true_label

    if "sample_X" in st.session_state:
        sample_X = st.session_state["sample_X"]
        true_label = st.session_state["true_label"]

        st.subheader(f"True stage: **{true_label}**")

        x_flat = sample_X.reshape(1, -1)
        x_tensor = torch.tensor(sample_X, dtype=torch.float32).unsqueeze(0)

        rf_probs = rf_model.predict_proba(x_flat)[0]
        lstm_probs = get_torch_probs(lstm_model, x_tensor)
        gru_probs = get_torch_probs(gru_model, x_tensor)
        ssm_probs = get_torch_probs(ssm_model, x_tensor)

        meta_input = np.concatenate([rf_probs, lstm_probs, gru_probs, ssm_probs]).reshape(1, -1)
        ensemble_pred = CLASS_NAMES[meta_model.predict(meta_input)[0]]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Individual Model Predictions")
            for name, probs in [("Random Forest", rf_probs), ("LSTM", lstm_probs),
                                 ("GRU", gru_probs), ("SSM", ssm_probs)]:
                pred_class = CLASS_NAMES[np.argmax(probs)]
                confidence = np.max(probs)
                correct = "✅" if pred_class == true_label else "❌"
                st.write(f"**{name}**: {pred_class} ({confidence:.0%} confidence) {correct}")

        with col2:
            st.markdown("### Final Ensemble Prediction")
            correct = "✅ Correct!" if ensemble_pred == true_label else "❌ Incorrect"
            st.markdown(f"## {ensemble_pred}")
            st.write(correct)

        st.markdown("### Probability Breakdown (Ensemble)")
        prob_df = pd.DataFrame({
            "Stage": CLASS_NAMES,
            "Random Forest": rf_probs,
            "LSTM": lstm_probs,
            "GRU": gru_probs,
            "SSM": ssm_probs,
        }).set_index("Stage")
        st.bar_chart(prob_df)
