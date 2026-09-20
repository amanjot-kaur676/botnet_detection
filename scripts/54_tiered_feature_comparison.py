"""
Tiered Feature Comparison - Basic vs Selected vs Expanded(-lite).

Tests whether more features actually earn their complexity, using Random
Forest (fast, reliable, no weighting overcorrection issues) as the fixed
model across all tiers - isolating the FEATURE effect from the MODEL effect.

Feature layout in the superset (29 features, indices 0-28):
  0-4   : duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts
  5-7   : proto one-hot (tcp, udp, icmp)
  8-9   : src_port, dst_port (raw, scaled)
  10-13 : bytes_per_sec, pkts_per_sec, avg_pkt_size, fwd_bwd_ratio
  14-24 : conn_state one-hot (11)
  25    : vuln_port flag
  26    : fanout_count
  27    : repeat_count
  28    : beacon_std

Tier 1 (Basic, 14 features):   indices 0-13   (raw network/flow + traffic behavior)
Tier 2 (Selected, 29 features): indices 0-28  (everything - Basic + conn_state/engineered)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42

data = np.load("../data/sequences/flow_sequences_superset_final.npz")
X, y = data["X"], data["y"]
print(f"Full X shape: {X.shape}")

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

n_samples, seq_len, n_features = X.shape
print(f"Total features available: {n_features}")

TIERS = {
    "Tier 1: Basic (14 features - raw network/flow + traffic behavior)": slice(0, 14),
    "Tier 2: Selected (29 features - Basic + conn_state + engineered signals)": slice(0, 29),
}

results_summary = []

for tier_name, feat_slice in TIERS.items():
    print("\n" + "=" * 70)
    print(tier_name)
    print("=" * 70)

    X_tier = X[:, :, feat_slice]
    n_tier_features = X_tier.shape[2]
    print(f"Using {n_tier_features} features per timestep")

    X_flat = X_tier.reshape(n_samples, seq_len * n_tier_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    # false positives/negatives per class: FP = predicted as class but wasn't; FN = was class but predicted as something else
    fp_per_class = cm.sum(axis=0) - np.diag(cm)
    fn_per_class = cm.sum(axis=1) - np.diag(cm)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("Confusion Matrix:")
    print("Order:", list(classes))
    print(cm)
    print("\nFalse Positives per class:", dict(zip(classes, fp_per_class)))
    print("False Negatives per class:", dict(zip(classes, fn_per_class)))

    results_summary.append({
        "tier": tier_name, "n_features": n_tier_features,
        "accuracy": acc, "macro_f1": macro_f1,
        "total_fp": fp_per_class.sum(), "total_fn": fn_per_class.sum(),
    })

print("\n" + "=" * 70)
print("TIER COMPARISON SUMMARY")
print("=" * 70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

print("""
Interpretation guide:
- If Tier 2 (Selected) beats Tier 1 (Basic) by a meaningful margin on
  macro F1 and reduces false negatives, the extra engineered features
  (conn_state, fan-out, repeat-contact, beaconing) are earning their
  complexity.
- If Tier 2 is only marginally better (or worse), Tier 1's simpler,
  more explainable feature set is the better choice per the
  "don't add complexity that doesn't pay for itself" principle.
""")
