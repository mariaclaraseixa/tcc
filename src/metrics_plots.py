"""
Gera graficos de metricas (Precision, F1) e matrizes de confusao para os resultados V4.
Tambem imprime resumo de amostras por modelo/estrategia.
"""
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
)

from config import RESULTS_DIR

logger = logging.getLogger(__name__)

MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"]
STRATEGIES = ["zero_shot", "cot"]
MODEL_LABELS = {
    "gpt-4o-mini":       "GPT-4o-mini",
    "claude-haiku":      "Claude Haiku",
    "gemini-2.5-flash":  "Gemini 2.5 Flash",
}
STRATEGY_LABELS = {"zero_shot": "Zero-Shot", "cot": "CoT"}

COLORS = {
    "zero_shot": "#4472C4",
    "cot":       "#ED7D31",
}

FILE_PREFIX = {"v1": "v1_", "v4": ""}


def _load_results(version: str = "v1") -> pd.DataFrame:
    prefix = FILE_PREFIX[version]
    frames = []
    for model in MODELS:
        for strategy in STRATEGIES:
            path = os.path.join(RESULTS_DIR, f"{prefix}{model}_{strategy}.csv")
            if not os.path.exists(path):
                logger.warning("Arquivo nao encontrado: %s", path)
                continue
            df = pd.read_csv(path)
            df["model"] = model
            df["strategy"] = strategy
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _compute_metrics(df: pd.DataFrame) -> dict:
    valid = df[df["prediction"].isin(["human", "ai"])]
    n_total = len(df)
    n_valid = len(valid)
    n_unknown = n_total - n_valid
    if n_valid == 0:
        return dict(precision=0, recall=0, f1=0, accuracy=0,
                    n_total=n_total, n_valid=n_valid, n_unknown=n_unknown)
    return dict(
        precision=precision_score(valid["true_label"], valid["prediction"],
                                  pos_label="ai", zero_division=0),
        recall=recall_score(valid["true_label"], valid["prediction"],
                            pos_label="ai", zero_division=0),
        f1=f1_score(valid["true_label"], valid["prediction"],
                    pos_label="ai", zero_division=0),
        accuracy=accuracy_score(valid["true_label"], valid["prediction"]),
        n_total=n_total, n_valid=n_valid, n_unknown=n_unknown,
    )


def _plot_metric_bars(metrics_df: pd.DataFrame, metric: str, version: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.35

    for i, strategy in enumerate(STRATEGIES):
        vals = [
            metrics_df.loc[(metrics_df["model"] == m) & (metrics_df["strategy"] == strategy),
                           metric].values[0]
            for m in MODELS
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[strategy],
                      color=COLORS[strategy], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.set_title(f"{metric.capitalize()} by Model and Prompt Strategy ({version.upper()})", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(output_dir, f"{metric}_by_model_strategy_{version}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Salvo: %s", path)
    return path


def _plot_confusion_matrices(all_data: pd.DataFrame, version: str, output_dir: str):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row_i, strategy in enumerate(STRATEGIES):
        for col_i, model in enumerate(MODELS):
            ax = fig.add_subplot(gs[row_i, col_i])
            sub = all_data[(all_data["model"] == model) & (all_data["strategy"] == strategy)]
            valid = sub[sub["prediction"].isin(["human", "ai"])]

            if len(valid) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            labels = ["ai", "human"]
            cm = confusion_matrix(valid["true_label"], valid["prediction"], labels=labels)

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
                            ha="center", va="center", fontsize=13, fontweight="bold",
                            color="white" if cm[ii, jj] > thresh else "black")

    fig.suptitle(f"Confusion Matrices — {version.upper()} Results", fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(output_dir, f"confusion_matrices_{version}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", path)
    return path


def _print_sample_summary(metrics_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("  RESUMO DE AMOSTRAS POR MODELO / ESTRATEGIA")
    print("=" * 72)
    print(f"  {'Modelo':<22} {'Estrategia':<12} {'Total':>7} {'Validas':>8} {'Unknown':>8}")
    print("  " + "-" * 60)
    for _, row in metrics_df.iterrows():
        print(f"  {MODEL_LABELS[row['model']]:<22} {STRATEGY_LABELS[row['strategy']]:<12} "
              f"{row['n_total']:>7} {row['n_valid']:>8} {row['n_unknown']:>8}")
    total = metrics_df["n_total"].sum()
    valid = metrics_df["n_valid"].sum()
    unk = metrics_df["n_unknown"].sum()
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<22} {'':12} {total:>7} {valid:>8} {unk:>8}")
    print(f"\n  Dataset: 200 snippets (50 por classe por linguagem: Python/C x ai/human)")
    print(f"  Combinacoes: 3 modelos x 2 estrategias = 6 rodadas")
    print(f"  Predicoes totais: {total} | Validas: {valid} | Unknowns: {unk}")
    print("=" * 72)


def _print_metrics_table(metrics_df: pd.DataFrame, version: str = "v1"):
    print("\n" + "=" * 72)
    print(f"  METRICAS COMPLETAS -- {version.upper()}")
    print("=" * 72)
    print(f"  {'Modelo':<22} {'Estrategia':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7}")
    print("  " + "-" * 64)
    for _, row in metrics_df.iterrows():
        print(f"  {MODEL_LABELS[row['model']]:<22} {STRATEGY_LABELS[row['strategy']]:<12} "
              f"{row['precision']:>7.3f} {row['recall']:>7.3f} "
              f"{row['f1']:>7.3f} {row['accuracy']:>7.3f}")
    print("=" * 72)


def run(version: str = "v1", output_dir: str | None = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    all_data = _load_results(version)

    rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            sub = all_data[(all_data["model"] == model) & (all_data["strategy"] == strategy)]
            m = _compute_metrics(sub)
            rows.append({"model": model, "strategy": strategy, **m})
    metrics_df = pd.DataFrame(rows)

    _print_sample_summary(metrics_df)
    _print_metrics_table(metrics_df, version)

    p1 = _plot_metric_bars(metrics_df, "precision", version, output_dir)
    p2 = _plot_metric_bars(metrics_df, "f1", version, output_dir)
    p3 = _plot_confusion_matrices(all_data, version, output_dir)

    print(f"\n  Graficos salvos:")
    print(f"    {p1}")
    print(f"    {p2}")
    print(f"    {p3}")

    metrics_df.to_csv(os.path.join(output_dir, f"metrics_summary_{version}.csv"), index=False)
    return metrics_df


def main():
    import argparse
    from src.logging_config import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", choices=["v1", "v4"])
    args = parser.parse_args()
    run(version=args.version)


if __name__ == "__main__":
    main()
