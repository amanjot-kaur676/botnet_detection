"""
Generate confusion matrix visualizations for all models tested.
Saves PNG figures for use in your report/presentation.

Uses the confusion matrices already recorded from each model's run
(hardcoded from your actual results, so no need to retrain anything).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("../data/figures", exist_ok=True)

CLASS_NAMES = ["Benign", "C2", "Impact", "Infect", "Scan"]

# actual confusion matrices from your runs
confusion_matrices = {
    "Random Forest": np.array([
        [1347, 2594, 0, 247, 472],
        [202, 4398, 3, 0, 487],
        [1, 10, 3981, 169, 6],
        [7, 1, 0, 534, 17],
        [123, 2145, 0, 1010, 2747],
    ]),
    "LSTM": np.array([
        [798, 450, 2, 610, 2800],
        [231, 1918, 2, 6, 2933],
        [17, 2, 3975, 83, 90],
        [25, 2, 0, 354, 178],
        [107, 12, 0, 483, 5423],
    ]),
    "GRU": np.array([
        [747, 469, 1, 635, 2808],
        [242, 1908, 3, 5, 2932],
        [12, 1, 3978, 90, 86],
        [25, 0, 0, 364, 170],
        [82, 12, 0, 496, 5435],
    ]),
    "S4D SSM": np.array([
        [489, 719, 1, 655, 2796],
        [106, 2056, 2, 11, 2915],
        [11, 1, 3978, 98, 79],
        [14, 5, 0, 382, 158],
        [100, 13, 0, 499, 5413],
    ]),
    "Selective SSM (Mamba-style)": np.array([
        [605, 519, 0, 713, 2823],
        [130, 1959, 2, 11, 2988],
        [1, 1, 3980, 111, 74],
        [4, 0, 0, 404, 151],
        [114, 13, 0, 516, 5382],
    ]),
    "Categorical HMM": np.array([
        [284, 3252, 116, 875, 133],
        [209, 4288, 371, 141, 81],
        [36, 492, 3471, 156, 12],
        [17, 8, 0, 503, 31],
        [401, 2993, 20, 1679, 932],
    ]),
    "Stacking Ensemble (RF meta)": np.array([
        [1339, 2578, 1, 43, 699],
        [225, 4372, 2, 0, 491],
        [9, 10, 3981, 23, 144],
        [25, 1, 0, 62, 471],
        [111, 2154, 0, 30, 3730],
    ]),
}

# --- Individual confusion matrix plots ---
for model_name, cm in confusion_matrices.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} - Confusion Matrix (row-normalized)")
    plt.tight_layout()

    filename = f"../data/figures/confusion_matrix_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# --- Combined comparison bar chart ---
accuracy_data = {
    "Stacking Ensemble": 0.6577,
    "Random Forest": 0.63,
    "LSTM": 0.6071,
    "GRU": 0.6057,
    "S4D SSM": 0.6043,
    "Selective SSM": 0.60,
    "Categorical HMM": 0.46,
    "Gaussian HMM": 0.39,
}

fig, ax = plt.subplots(figsize=(10, 6))
names = list(accuracy_data.keys())
values = list(accuracy_data.values())
colors = ["#2ecc71" if n == "Stacking Ensemble" else "#3498db" for n in names]
bars = ax.bar(names, values, color=colors)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison - Test Set Accuracy")
ax.set_ylim(0, 0.8)
ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="Random guessing (5 classes)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.0%}", ha="center", fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("../data/figures/model_comparison_bar_chart.png", dpi=150)
plt.close()
print("Saved: ../data/figures/model_comparison_bar_chart.png")

print("\nAll figures saved to ../data/figures/")
