"""
Gera tabela comparativa de desempenho entre modelos para o experimento atual.
Lê os resultados do experimento e produz:
  - Tabela pivot por métrica (modelo x estratégia)
  - CSV e HTML com a comparação completa
  - Ranking final dos modelos (média de F1 entre estratégias)
"""
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import METRICS_DIR, CURRENT_EXPERIMENT
from src.logging_config import setup_logging
from src.metrics import load_all_results, compute_metrics

logger = logging.getLogger(__name__)

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1-Score",
    "precision": "Precision",
    "recall": "Recall",
}


def _pivot_metric(metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = metrics_df.pivot(index="strategy", columns="model", values=metric)
    pivot.index.name = "Estratégia"
    pivot.columns.name = "Modelo"
    return pivot.round(4)


def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string())
    print()


def _save_html(tables: dict, out_path: str, experiment: str) -> None:
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; margin-top: 30px; }",
        "table { border-collapse: collapse; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ccc; padding: 8px 14px; text-align: center; }",
        "th { background-color: #4472C4; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        ".best { font-weight: bold; color: #0a7c0a; }",
        "</style></head><body>",
        f"<h1>Comparação de Modelos — {experiment}</h1>",
    ]

    for title, df in tables.items():
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append(df.to_html(classes=""))

    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    logger.info("Relatório HTML salvo: %s", out_path)


def _plot_heatmap(metrics_df: pd.DataFrame, metric: str, out_path: str) -> None:
    pivot = _pivot_metric(metrics_df, metric)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(4, len(pivot) * 1.2)))
    import seaborn as sns
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title(f"{_METRIC_LABELS.get(metric, metric)} — {CURRENT_EXPERIMENT}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info("Heatmap salvo: %s", out_path)


def generate_comparison(metrics_df: pd.DataFrame) -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)

    tables = {}

    for metric, label in _METRIC_LABELS.items():
        pivot = _pivot_metric(metrics_df, metric)
        tables[label] = pivot
        _print_table(label, pivot)

        csv_path = os.path.join(METRICS_DIR, f"compare_{metric}.csv")
        pivot.to_csv(csv_path)
        logger.info("Tabela '%s' salva: %s", label, csv_path)

        heatmap_path = os.path.join(METRICS_DIR, f"heatmap_{metric}.png")
        _plot_heatmap(metrics_df, metric, heatmap_path)

    # Ranking geral: média de F1 por modelo
    ranking = (
        metrics_df.groupby("model")["f1"]
        .mean()
        .sort_values(ascending=False)
        .rename("F1 médio")
        .to_frame()
        .round(4)
    )
    ranking.index.name = "Modelo"
    ranking["Rank"] = range(1, len(ranking) + 1)
    tables["Ranking Geral (F1 médio)"] = ranking
    _print_table("Ranking Geral (F1 médio entre estratégias)", ranking)

    ranking_path = os.path.join(METRICS_DIR, "ranking_models.csv")
    ranking.to_csv(ranking_path)
    logger.info("Ranking salvo: %s", ranking_path)

    html_path = os.path.join(METRICS_DIR, "comparison_report.html")
    _save_html(tables, html_path, CURRENT_EXPERIMENT)


def main() -> None:
    setup_logging()
    logger.info("=== Gerando tabela comparativa — %s ===", CURRENT_EXPERIMENT)

    df = load_all_results()
    if df.empty:
        logger.error("Nenhum resultado encontrado. Execute o experimento primeiro.")
        return

    metrics_df = compute_metrics(df)
    if metrics_df.empty:
        logger.error("Não foi possível calcular métricas.")
        return

    generate_comparison(metrics_df)
    logger.info("=== Comparação concluída. Resultados em: %s ===", METRICS_DIR)


if __name__ == "__main__":
    main()
