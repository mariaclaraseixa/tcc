import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import RESULTS_DIR

MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"]
STRATEGIES = ["zero_shot", "cot"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku": "Claude Haiku",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
STRATEGY_LABELS = {"zero_shot": "Zero-Shot", "cot": "CoT"}

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("V1 — Confusion Matrices per Model (rows=strategy, cols=model)",
             fontsize=12, fontweight="bold", y=1.01)

for col, model in enumerate(MODELS):
    for row, strategy in enumerate(STRATEGIES):
        ax = axes[row][col]
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"v1_{model}_{strategy}.csv"))
        v = df[df["prediction"].isin(["human", "ai"])]
        labels = ["ai", "human"]
        cm = confusion_matrix(v["true_label"], v["prediction"], labels=labels)

        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["AI", "Human"], fontsize=9)
        ax.set_yticklabels(["AI", "Human"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(f"{MODEL_LABELS[model]}\n{STRATEGY_LABELS[strategy]}", fontsize=9.5)

        thresh = cm.max() / 2.0
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]),
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[ii, jj] > thresh else "black")

out_grid = os.path.join(RESULTS_DIR, "metrics", "confusion_matrices_v1.png")
fig.tight_layout()
fig.savefig(out_grid, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grid 2x3 salvo: {out_grid}")

# --- matriz por modelo (soma zero_shot + cot) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5))
fig2.suptitle("V1 — Confusion Matrices per Model (Zero-Shot + CoT combined)",
              fontsize=12, fontweight="bold")

for col, model in enumerate(MODELS):
    ax = axes2[col]
    frames = [
        pd.read_csv(os.path.join(RESULTS_DIR, f"v1_{model}_{s}.csv"))
        for s in STRATEGIES
    ]
    combined = pd.concat(frames)
    v = combined[combined["prediction"].isin(["human", "ai"])]
    labels = ["ai", "human"]
    cm = confusion_matrix(v["true_label"], v["prediction"], labels=labels)

    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["AI", "Human"], fontsize=10)
    ax.set_yticklabels(["AI", "Human"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{MODEL_LABELS[model]}\n(n={len(v)})", fontsize=10.5)

    thresh = cm.max() / 2.0
    for ii in range(2):
        for jj in range(2):
            pct = cm[ii, jj] / cm.sum() * 100
            ax.text(jj, ii, f"{cm[ii, jj]}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[ii, jj] > thresh else "black")

out_combined = os.path.join(RESULTS_DIR, "metrics", "confusion_matrices_v1_per_model.png")
fig2.tight_layout()
fig2.savefig(out_combined, dpi=150, bbox_inches="tight")
plt.close()
print(f"Por modelo salvo: {out_combined}")
