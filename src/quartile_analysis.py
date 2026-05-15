"""
Analisa acurácia de classificação por quartil de tamanho de código (nº de linhas).
Usa os resultados V1 (200 amostras, 6 arquivos).
Saída: results/<experimento>/metrics/accuracy_by_quartile.csv
"""
import logging
import os

import pandas as pd
from scipy.stats import spearmanr

from config import RESULTS_DIR, SAMPLES_PER_CLASS_PER_LANG
from src.data_loader import load_data

logger = logging.getLogger(__name__)

V1_FILES = [
    "v1_gpt-4o-mini_zero_shot.csv",
    "v1_gpt-4o-mini_cot.csv",
    "v1_claude-haiku_zero_shot.csv",
    "v1_claude-haiku_cot.csv",
    "v1_gemini-2.5-flash_zero_shot.csv",
    "v1_gemini-2.5-flash_cot.csv",
]

V4_FILES = [
    "gpt-4o-mini_zero_shot.csv",
    "gpt-4o-mini_cot.csv",
    "claude-haiku_zero_shot.csv",
    "claude-haiku_cot.csv",
    "gemini-2.5-flash_zero_shot.csv",
    "gemini-2.5-flash_cot.csv",
]


def _load_results(version: str = "v1") -> pd.DataFrame:
    files = V1_FILES if version == "v1" else V4_FILES
    frames = []
    for fname in files:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            logger.warning("Arquivo nao encontrado: %s", path)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"Nenhum arquivo {version} encontrado em " + RESULTS_DIR)
    return pd.concat(frames, ignore_index=True)


def _compute_n_lines(code: str) -> int:
    return len(code.strip().splitlines())


def run(version: str = "v1", output_dir: str | None = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    # --- carrega predições ---
    preds = _load_results(version)
    logger.info("Predicoes %s carregadas: %d linhas", version.upper(), len(preds))

    # --- carrega dataset com mesma amostragem usada no experimento ---
    df_data = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    df_data["n_lines"] = df_data["code"].apply(_compute_n_lines)
    code_info = df_data[["n_lines", "language_name"]].copy()

    # --- junta predições com tamanho de código ---
    merged = preds.merge(code_info, left_on="id", right_index=True, how="left")
    n_missing = merged["n_lines"].isna().sum()
    if n_missing:
        logger.warning("%d predições sem tamanho de código — removidas.", n_missing)
    merged = merged.dropna(subset=["n_lines"])
    merged["n_lines"] = merged["n_lines"].astype(int)

    # --- descarta unknowns para métrica ---
    valid = merged[merged["prediction"].isin(["human", "ai"])].copy()
    valid["correct"] = (valid["prediction"] == valid["true_label"]).astype(int)

    # --- quartis globais (sobre os 200 snippets únicos) ---
    snippet_sizes = code_info["n_lines"].reset_index(drop=True)
    quartile_bins = pd.qcut(snippet_sizes, q=4, retbins=True, duplicates="drop")[1]
    labels = [f"Q{i+1}" for i in range(len(quartile_bins) - 1)]
    valid["quartile"] = pd.cut(valid["n_lines"], bins=quartile_bins, labels=labels, include_lowest=True)

    # --- acurácia por quartil (agregado: todos modelos/estratégias) ---
    overall = (
        valid.groupby("quartile", observed=True)
        .agg(
            n_snippets=("id", "nunique"),
            n_predictions=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    # adiciona intervalo de linhas por quartil
    bin_edges = quartile_bins
    overall["lines_range"] = [
        f"{int(bin_edges[i])+1}–{int(bin_edges[i+1])}" if i > 0
        else f"{int(bin_edges[i])}–{int(bin_edges[i+1])}"
        for i in range(len(labels))
    ]
    overall = overall[["quartile", "lines_range", "n_snippets", "n_predictions", "accuracy"]]

    # --- acurácia por quartil × modelo ---
    by_model = (
        valid.groupby(["model", "quartile"], observed=True)["correct"]
        .mean()
        .unstack("quartile")
        .reset_index()
    )
    by_model.columns.name = None

    # --- correlação de Spearman (n_lines vs correct, por predição individual) ---
    rho, pval = spearmanr(valid["n_lines"], valid["correct"])

    # --- salva ---
    out_overall = os.path.join(output_dir, f"accuracy_by_quartile_{version}.csv")
    out_model = os.path.join(output_dir, f"accuracy_by_quartile_model_{version}.csv")
    overall.to_csv(out_overall, index=False)
    by_model.to_csv(out_model, index=False)
    logger.info("Salvo: %s", out_overall)
    logger.info("Salvo: %s", out_model)

    # --- imprime ---
    print("\n" + "=" * 60)
    print(f"  ACURACIA POR QUARTIL DE TAMANHO (n linhas)  -- {version.upper()}")
    print("  Quartis globais (sobre 200 snippets unicos)")
    print("=" * 60)
    print(f"\n  {'Quartil':<8} {'Linhas':<12} {'Snippets':>9} {'Predicoes':>10} {'Acuracia':>10}")
    print("  " + "-" * 52)
    for _, row in overall.iterrows():
        print(f"  {row['quartile']:<8} {row['lines_range']:<12} {row['n_snippets']:>9} "
              f"{row['n_predictions']:>10} {row['accuracy']:>10.3f}")

    print(f"\n  Spearman (n_linhas x acerto): rho={rho:+.3f}  p={pval:.4f}")
    if pval < 0.05:
        direction = "positiva" if rho > 0 else "negativa"
        print(f"  -> Correlacao {direction} significativa (p < 0.05)")
    else:
        print("  -> Sem correlacao significativa (p >= 0.05)")

    print("\n  Acurácia por modelo × quartil:")
    print("  " + by_model.to_string(index=False))
    print("=" * 60)

    return overall


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
