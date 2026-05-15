import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config import RESULTS_DIR

MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"]
STRATEGIES = ["zero_shot", "cot"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku": "Claude Haiku",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
STRATEGY_LABELS = {"zero_shot": "Zero-Shot", "cot": "CoT"}
METRICS = [("precision", "Precision"), ("recall", "Recall"),
           ("f1", "F1"), ("accuracy", "Accuracy")]
METRIC_COLORS = ["#4472C4", "#ED7D31", "#A9D18E", "#C00000"]


def calc(v, key):
    if key == "precision":
        return precision_score(v["true_label"], v["prediction"], pos_label="ai", zero_division=0)
    if key == "recall":
        return recall_score(v["true_label"], v["prediction"], pos_label="ai", zero_division=0)
    if key == "f1":
        return f1_score(v["true_label"], v["prediction"], pos_label="ai", zero_division=0)
    return accuracy_score(v["true_label"], v["prediction"])


rows = []
for model in MODELS:
    for strategy in STRATEGIES:
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"v1_{model}_{strategy}.csv"))
        v = df[df["prediction"].isin(["human", "ai"])]
        row = {
            "model": model,
            "strategy": strategy,
            "label": MODEL_LABELS[model] + "\n" + STRATEGY_LABELS[strategy],
            "n_valid": len(v),
            "n_unknown": len(df) - len(v),
        }
        for key, _ in METRICS:
            row[key] = calc(v, key)
        rows.append(row)
mdf = pd.DataFrame(rows)

# --- aggregate ---
all_frames = [
    pd.read_csv(os.path.join(RESULTS_DIR, f"v1_{m}_{s}.csv"))
    for m in MODELS for s in STRATEGIES
]
all_preds = pd.concat(all_frames)
v_all = all_preds[all_preds["prediction"].isin(["human", "ai"])]
g = {key: calc(v_all, key) for key, _ in METRICS}

print("=" * 62)
print("  RESUMO GERAL V1")
print("=" * 62)
n_total = len(all_preds)
n_valid = len(v_all)
print(f"  Total predicoes : {n_total} | Validas: {n_valid} | Unknowns: {n_total - n_valid}")
print(f"  Precision global: {g['precision']:.3f}")
print(f"  Recall    global: {g['recall']:.3f}")
print(f"  F1        global: {g['f1']:.3f}")
print(f"  Accuracy  global: {g['accuracy']:.3f}")
print()
print("  Media por modelo:")
for model in MODELS:
    sub = mdf[mdf["model"] == model]
    name = MODEL_LABELS[model]
    f1m = sub["f1"].mean()
    accm = sub["accuracy"].mean()
    pm = sub["precision"].mean()
    rm = sub["recall"].mean()
    print(f"    {name:<22}  F1={f1m:.3f}  Acc={accm:.3f}  Prec={pm:.3f}  Rec={rm:.3f}")
print("=" * 62)

# --- plot ---
fig, ax = plt.subplots(figsize=(14, 5.5))
x = np.arange(len(mdf))
width = 0.18
for i, (metric, mlabel) in enumerate(METRICS):
    offset = (i - 1.5) * width
    vals = mdf[metric].tolist()
    bars = ax.bar(x + offset, vals, width, label=mlabel,
                  color=METRIC_COLORS[i], edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(mdf["label"].tolist(), fontsize=8.5)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_title("V1 -- All Metrics by Model and Strategy", fontsize=13, fontweight="bold")
glabel = f"Global Accuracy ({g['accuracy']:.3f})"
ax.axhline(g["accuracy"], color="gray", linestyle=":", linewidth=1.2, label=glabel)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for sep in [1.5, 3.5]:
    ax.axvline(sep, color="#cccccc", linewidth=1.2, linestyle="--")

out = os.path.join(RESULTS_DIR, "metrics", "v1_overview.png")
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close()
print(f"\n  Grafico salvo: {out}")
