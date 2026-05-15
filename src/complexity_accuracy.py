"""
Cruza complexity_score com acuracia de classificacao (V4).
- Faixas: Low 0-3, Medium 4-6, High 7-10
- Barplot com linha na acuracia global
- Spearman entre complexity_score e correct
"""
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from config import RESULTS_DIR

logger = logging.getLogger(__name__)

MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"]
STRATEGIES = ["zero_shot", "cot"]
FILE_PREFIX = {"v1": "v1_", "v4": ""}

LEVEL_BINS  = [-1, 3, 6, 10]
LEVEL_LABELS = ["Low (0-3)", "Medium (4-7)", "High (7-10)"]
LEVEL_COLORS = ["#5B9BD5", "#ED7D31", "#A9D18E"]


def _load_predictions(version: str = "v1") -> pd.DataFrame:
    prefix = FILE_PREFIX[version]
    frames = []
    for model in MODELS:
        for strategy in STRATEGIES:
            path = os.path.join(RESULTS_DIR, f"{prefix}{model}_{strategy}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
            else:
                logger.warning("Arquivo nao encontrado: %s", path)
    return pd.concat(frames, ignore_index=True)


def run(version: str = "v1", output_dir: str | None = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    # --- carrega dados ---
    complexity = pd.read_csv(os.path.join(RESULTS_DIR, "complexity.csv"))
    preds = _load_predictions(version)

    # --- complexidade media por snippet (media dos 3 modelos, descarta erros) ---
    comp_valid = complexity[complexity["complexity_score"] >= 0].copy()
    comp_mean = (
        comp_valid.groupby("id")["complexity_score"]
        .mean()
        .reset_index()
        .rename(columns={"complexity_score": "complexity_mean"})
    )

    # --- join predicoes com complexidade media ---
    valid_preds = preds[preds["prediction"].isin(["human", "ai"])].copy()
    valid_preds["correct"] = (valid_preds["prediction"] == valid_preds["true_label"]).astype(int)
    merged = valid_preds.merge(comp_mean, on="id", how="inner")
    logger.info("Predicoes validas com complexidade: %d", len(merged))

    # --- faixas de complexidade ---
    merged["complexity_level"] = pd.cut(
        merged["complexity_mean"],
        bins=LEVEL_BINS,
        labels=LEVEL_LABELS,
        include_lowest=True,
    )

    # --- acuracia por faixa ---
    by_level = (
        merged.groupby("complexity_level", observed=True)["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    global_acc = merged["correct"].mean()

    # --- spearman ---
    rho, pval = spearmanr(merged["complexity_mean"], merged["correct"])

    # --- estatisticas de complexity_score por modelo ---
    comp_stats = (
        comp_valid.groupby("model")["complexity_score"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        [["mean", "std", "25%", "50%", "75%"]]
        .rename(columns={"25%": "Q1", "50%": "median", "75%": "Q3"})
        .reset_index()
    )

    # --- spearman por modelo (join direto id+model para manter score individual) ---
    preds_with_comp = valid_preds.merge(
        comp_valid[["id", "model", "complexity_score"]], on=["id", "model"], how="inner"
    )
    spearman_by_model = []
    for model, grp in preds_with_comp.groupby("model"):
        r, p = spearmanr(grp["complexity_score"], grp["correct"])
        spearman_by_model.append({"model": model, "rho": r, "pval": p})
    sp_df = pd.DataFrame(spearman_by_model)

    # --- print ---
    print("\n" + "=" * 60)
    print("  ESTATISTICAS DE COMPLEXITY_SCORE POR MODELO")
    print("=" * 60)
    print(f"\n  {'Modelo':<22} {'mean':>6} {'std':>6} {'Q1':>6} {'median':>8} {'Q3':>6}")
    print("  " + "-" * 58)
    for _, row in comp_stats.iterrows():
        print(f"  {row['model']:<22} {row['mean']:>6.2f} {row['std']:>6.2f} "
              f"{row['Q1']:>6.1f} {row['median']:>8.1f} {row['Q3']:>6.1f}")

    print("\n" + "=" * 60)
    print("  SPEARMAN (complexity_score x correct) POR MODELO")
    print("=" * 60)
    print(f"\n  {'Modelo':<22} {'rho':>7} {'p-valor':>10}  sig")
    print("  " + "-" * 48)
    for _, row in sp_df.iterrows():
        sig = "*" if row["pval"] < 0.05 else ""
        print(f"  {row['model']:<22} {row['rho']:>+7.3f} {row['pval']:>10.4f}  {sig}")

    print("\n" + "=" * 60)
    print(f"  ACURACIA POR FAIXA DE COMPLEXIDADE ({version.upper()})")
    print("=" * 60)
    print(f"\n  {'Faixa':<14} {'N':>6} {'Acuracia':>10}")
    print("  " + "-" * 32)
    for _, row in by_level.iterrows():
        print(f"  {str(row['complexity_level']):<14} {row['n']:>6} {row['accuracy']:>10.3f}")
    print(f"\n  Acuracia global: {global_acc:.3f}")
    print(f"  Spearman global (score x correct): rho={rho:+.3f}  p={pval:.4f}")
    if pval < 0.05:
        direction = "positiva" if rho > 0 else "negativa"
        print(f"  -> Correlacao {direction} significativa (p < 0.05)")
    else:
        print("  -> Sem correlacao significativa (p >= 0.05)")
    print("=" * 60)

    # --- barplot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        by_level["complexity_level"].astype(str),
        by_level["accuracy"],
        color=LEVEL_COLORS[:len(by_level)],
        edgecolor="white",
        width=0.5,
    )
    ax.axhline(global_acc, color="#C00000", linewidth=1.5, linestyle="--",
               label=f"Global accuracy ({global_acc:.3f})")

    # anotacoes de valor nas barras
    for bar, (_, row) in zip(bars, by_level.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{row['accuracy']:.3f}\n(n={row['n']})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylim(0, min(1.0, by_level["accuracy"].max() + 0.12))
    ax.set_xlabel("Complexity Level", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        f"Classification Accuracy by Code Complexity ({version.upper()})\n"
        f"Spearman rho={rho:+.3f}, p={pval:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_path = os.path.join(output_dir, f"accuracy_by_complexity_{version}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: %s", plot_path)
    print(f"\n  Grafico salvo: {plot_path}")

    # --- salva CSVs ---
    by_level.to_csv(os.path.join(output_dir, f"accuracy_by_complexity_{version}.csv"), index=False)
    comp_stats.to_csv(os.path.join(output_dir, "complexity_stats_by_model.csv"), index=False)
    sp_df.to_csv(os.path.join(output_dir, f"spearman_complexity_by_model_{version}.csv"), index=False)
    logger.info("CSVs salvos em %s", output_dir)

    return by_level


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
