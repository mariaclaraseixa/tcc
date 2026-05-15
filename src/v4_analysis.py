"""
Analises especificas para os resultados V4:
1. accuracy_by_size_model.csv  — acuracia por quartil de tamanho (Small/Medium/Large) x modelo x estrategia
2. accuracy_by_language_model.csv — acuracia por linguagem x modelo x estrategia
3. tabela resumo V4 (precision, recall, F1, accuracy)
Gera graficos PNG e salva CSVs em results/metrics/
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score,
)

from config import RESULTS_DIR, SAMPLES_PER_CLASS_PER_LANG
from src.data_loader import load_data

logger = logging.getLogger(__name__)

MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "claude-haiku"]
STRATEGIES = ["zero_shot", "cot"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku": "Claude Haiku",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
STRATEGY_LABELS = {"zero_shot": "Zero-Shot", "cot": "CoT"}
COLORS = {"zero_shot": "#4472C4", "cot": "#ED7D31"}


def _load_v4() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for strategy in STRATEGIES:
            path = os.path.join(RESULTS_DIR, f"v4_{model}_{strategy}.csv")
            if not os.path.exists(path):
                logger.warning("Arquivo nao encontrado: %s", path)
                continue
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _metrics(df: pd.DataFrame) -> dict:
    v = df[df["prediction"].isin(["human", "ai"])].copy()
    if len(v) == 0:
        return dict(precision=0, recall=0, f1=0, accuracy=0, mcc=0, auc=float("nan"), n=len(df))

    y_true = (v["true_label"] == "ai").astype(int)
    y_pred = (v["prediction"] == "ai").astype(int)

    # AUC-ROC: confidence como score do lado AI
    # se prediction=ai  → score = confidence/100
    # se prediction=human → score = 1 - confidence/100
    v["ai_score"] = v.apply(
        lambda r: r["confidence"] / 10 if r["prediction"] == "ai" else 1 - r["confidence"] / 10,
        axis=1,
    )
    try:
        auc = roc_auc_score(y_true, v["ai_score"])
    except Exception:
        auc = float("nan")

    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        auc=auc,
        n=len(v),
    )


def _n_lines(code: str) -> int:
    return len(code.strip().splitlines())


def run(output_dir: str | None = None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    preds = _load_v4()
    valid = preds[preds["prediction"].isin(["human", "ai"])].copy()
    valid["correct"] = (valid["prediction"] == valid["true_label"]).astype(int)

    # --- tamanho dos snippets (mesma amostra do experimento) ---
    df_data = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    df_data["n_lines"] = df_data["code"].apply(_n_lines)
    size_info = df_data[["n_lines"]].copy()

    q1 = size_info["n_lines"].quantile(0.25)
    q3 = size_info["n_lines"].quantile(0.75)

    def size_label(n):
        if n <= q1:
            return "Small"
        elif n <= q3:
            return "Medium"
        else:
            return "Large"

    size_info["size_group"] = size_info["n_lines"].apply(size_label)
    valid = valid.merge(size_info[["size_group"]], left_on="id", right_index=True, how="left")

    # ================================================================
    # 1. ACURACIA POR TAMANHO x MODELO x ESTRATEGIA
    # ================================================================
    size_rows = []
    size_order = ["Small", "Medium", "Large"]
    for model in MODELS:
        for strategy in STRATEGIES:
            sub = valid[(valid["model"] == model) & (valid["strategy"] == strategy)]
            for size in size_order:
                grp = sub[sub["size_group"] == size]
                acc = grp["correct"].mean() if len(grp) > 0 else float("nan")
                size_rows.append({
                    "model": model, "strategy": strategy,
                    "size_group": size, "n": len(grp), "accuracy": acc,
                })
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(os.path.join(output_dir, "accuracy_by_size_model.csv"), index=False)

    # plot size
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=True)
    x = np.arange(len(size_order))
    width = 0.35
    for col, model in enumerate(MODELS):
        ax = axes[col]
        for i, strategy in enumerate(STRATEGIES):
            sub = size_df[(size_df["model"] == model) & (size_df["strategy"] == strategy)]
            vals = [sub[sub["size_group"] == s]["accuracy"].values[0] for s in size_order]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=STRATEGY_LABELS[strategy], color=COLORS[strategy], edgecolor="white")
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(size_order, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        if col == 1:
            ax.legend(fontsize=8)
    fig.suptitle("V4 — Accuracy by Snippet Size (Small=Q1, Medium=Q1–Q3, Large=Q3+)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_by_size_model.png"), dpi=150)
    plt.close(fig)

    # ================================================================
    # 2. METRICAS POR LINGUAGEM x MODELO x ESTRATEGIA
    # ================================================================
    languages = sorted(valid["language"].dropna().unique())
    lang_rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            sub = preds[(preds["model"] == model) & (preds["strategy"] == strategy)]
            for lang in languages:
                grp = sub[sub["language"] == lang]
                m = _metrics(grp)
                lang_rows.append({
                    "model": MODEL_LABELS[model], "strategy": STRATEGY_LABELS[strategy],
                    "language": lang, **m,
                })
    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(os.path.join(output_dir, "accuracy_by_language_model.csv"), index=False)

    # plot por modelo x linguagem
    fig2, axes2 = plt.subplots(1, len(MODELS), figsize=(13, 5), sharey=True)
    x2 = np.arange(len(languages))
    for col, model in enumerate(MODELS):
        ax = axes2[col]
        sub_m = lang_df[lang_df["model"] == MODEL_LABELS[model]]
        for i, strategy in enumerate(STRATEGIES):
            sub_s = sub_m[sub_m["strategy"] == STRATEGY_LABELS[strategy]]
            vals = [sub_s[sub_s["language"] == l]["accuracy"].values[0] for l in languages]
            bars = ax.bar(x2 + (i - 0.5) * width, vals, width,
                          label=STRATEGY_LABELS[strategy], color=COLORS[strategy], edgecolor="white")
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xticks(x2)
        ax.set_xticklabels(languages, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        if col == 1:
            ax.legend(fontsize=8)
    fig2.suptitle("V4 — Accuracy by Programming Language (per model/strategy)",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "accuracy_by_language_model.png"), dpi=150)
    plt.close(fig2)

    # plot agregado por linguagem (todos modelos/estrategias juntos)
    agg_lang_rows = []
    for lang in languages:
        grp = preds[preds["language"] == lang]
        m = _metrics(grp)
        agg_lang_rows.append({"language": lang, **m})
    agg_lang_df = pd.DataFrame(agg_lang_rows)

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    x3 = np.arange(len(languages))
    bars3 = ax3.bar(x3, agg_lang_df["accuracy"], width=0.4,
                    color=["#4472C4", "#ED7D31"], edgecolor="white")
    for bar, (_, row) in zip(bars3, agg_lang_df.iterrows()):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{row['accuracy']:.3f}\n(n={row['n']})",
                 ha="center", va="bottom", fontsize=10)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(languages, fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Accuracy", fontsize=11)
    ax3.set_title("V4 — Accuracy by Language (all models aggregated)", fontsize=11, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "accuracy_by_language.png"), dpi=150)
    plt.close(fig3)

    # ================================================================
    # 3. TABELA RESUMO V4
    # ================================================================
    summary_rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            sub = preds[(preds["model"] == model) & (preds["strategy"] == strategy)]
            m = _metrics(sub)
            summary_rows.append({
                "model": MODEL_LABELS[model],
                "strategy": STRATEGY_LABELS[strategy],
                **m,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "v4_results_table.csv"), index=False)

    # print tabela
    print("\n" + "=" * 70)
    print("  TABELA DE RESULTADOS V4")
    print("=" * 70)
    print(f"  {'Modelo':<22} {'Estrategia':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} {'MCC':>7} {'AUC':>7} {'N':>6}")
    print("  " + "-" * 80)
    for _, row in summary_df.iterrows():
        auc_str = f"{row['auc']:>7.3f}" if row['auc'] == row['auc'] else "    n/a"
        print(f"  {row['model']:<22} {row['strategy']:<12} "
              f"{row['precision']:>7.3f} {row['recall']:>7.3f} "
              f"{row['f1']:>7.3f} {row['accuracy']:>7.3f} "
              f"{row['mcc']:>7.3f} {auc_str} {int(row['n']):>6}")
    print("=" * 70)
    print(f"\n  Quartis de tamanho: Q1={q1:.0f} linhas, Q3={q3:.0f} linhas")
    print(f"  Small (<= {q1:.0f}), Medium ({q1:.0f}-{q3:.0f}), Large (> {q3:.0f})")

    print(f"\n  Arquivos salvos em {output_dir}:")
    for f in ["accuracy_by_size_model.csv", "accuracy_by_size_model.png",
              "accuracy_by_language_model.csv", "accuracy_by_language_model.png",
              "v4_results_table.csv"]:
        print(f"    {f}")


def main():
    from src.logging_config import setup_logging
    setup_logging()
    run()


if __name__ == "__main__":
    main()
