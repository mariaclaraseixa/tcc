import logging
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

from config import RESULTS_DIR, METRICS_DIR
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, pos_label="ai", zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label="ai", zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label="ai", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["human", "ai"]),
    }


def load_all_results(files: list = None) -> pd.DataFrame:
    if files is None:
        files = [f for f in glob.glob(f"{RESULTS_DIR}/*.csv") if "summary" not in os.path.basename(f)]
    logger.info("Carregando %d arquivo(s) de resultado.", len(files))

    if not files:
        logger.error(
            "Nenhum CSV encontrado em '%s'. Execute experiment.py antes de rodar metrics.py.",
            RESULTS_DIR,
        )
        return pd.DataFrame()

    dfs = []
    for f in files:
        logger.debug("Lendo: %s", f)
        dfs.append(pd.read_csv(f))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Total de predições carregadas: %d", len(combined))
    return combined


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando métricas por (modelo, estratégia)...")
    rows = []

    for (model, strategy), group in df.groupby(["model", "strategy"]):
        valid = group[group["prediction"].isin(["human", "ai"])]
        n_invalid = len(group) - len(valid)

        if len(valid) == 0:
            logger.warning(
                "Nenhuma predição válida para %s | %s — grupo ignorado.", model, strategy,
            )
            continue

        m = _calc_metrics(valid["true_label"], valid["prediction"])

        logger.info(
            "  %s | %-10s — Acc=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  (inválidos=%d)",
            model, strategy, m["accuracy"], m["f1"], m["precision"], m["recall"], n_invalid,
        )
        rows.append({
            "model": model, "strategy": strategy,
            "accuracy": m["accuracy"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
            "n_samples": len(valid), "n_unknown": n_invalid,
        })

    metrics_df = pd.DataFrame(rows, columns=[
        "model", "strategy", "accuracy", "f1", "precision", "recall", "n_samples", "n_unknown"
    ])
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values("f1", ascending=False)
    logger.info("Métricas calculadas para %d combinação(ões).", len(metrics_df))
    return metrics_df


def plot_confusion_matrices(df: pd.DataFrame) -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    logger.info("Gerando matrizes de confusão...")
    count = 0

    for (model, strategy), group in df.groupby(["model", "strategy"]):
        valid = group[group["prediction"].isin(["human", "ai"])]
        if len(valid) == 0:
            logger.warning(
                "Pulando matriz de confusão para %s | %s (sem predições válidas).",
                model, strategy,
            )
            continue

        cm = _calc_metrics(valid["true_label"], valid["prediction"])["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["human", "ai"], yticklabels=["human", "ai"], ax=ax)
        ax.set_title(f"{model} | {strategy}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        fname = f"{METRICS_DIR}/cm_{model.replace('/', '_')}_{strategy}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        logger.debug("Matriz de confusão salva: %s", fname)
        count += 1

    logger.info("%d matriz(es) de confusão gerada(s).", count)


def plot_f1_comparison(metrics_df: pd.DataFrame) -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    logger.info("Gerando gráfico comparativo de F1-Score...")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = metrics_df.pivot(index="strategy", columns="model", values="f1")
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("F1-Score por Estratégia e Modelo")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Modelo")
    plt.tight_layout()

    fname = f"{METRICS_DIR}/f1_comparison.png"
    plt.savefig(fname)
    plt.close()
    logger.info("Gráfico F1 salvo: %s", fname)


def main() -> None:
    setup_logging()
    logger.info("=== Iniciando análise de métricas ===")

    df = load_all_results()
    if df.empty:
        logger.critical("Nenhum dado carregado — encerrando sem gerar resultados.")
        return

    metrics_df = compute_metrics(df)

    os.makedirs(METRICS_DIR, exist_ok=True)
    summary_path = f"{METRICS_DIR}/summary.csv"
    metrics_df.to_csv(summary_path, index=False)
    logger.info("CSV de resumo salvo: %s", summary_path)

    plot_confusion_matrices(df)
    plot_f1_comparison(metrics_df)

    logger.info("=== Análise concluída. Resultados em: %s ===", METRICS_DIR)


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    main()
