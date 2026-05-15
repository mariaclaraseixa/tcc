import logging
import sys

from config import RESULTS_DIR
from src.logging_config import setup_logging
from src.metrics import load_all_results, compute_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = load_all_results()
    if df.empty:
        logger.error("Nenhum dado carregado. Execute o experimento primeiro.")
        sys.exit(1)

    metrics_df = compute_metrics(df)
    if metrics_df.empty:
        logger.error("Nenhuma métrica calculada — verifique os arquivos em '%s'.", RESULTS_DIR)
        sys.exit(1)

    out = f"{RESULTS_DIR}/summary_metrics.csv"
    metrics_df.to_csv(out, index=False)
    logger.info("Resumo salvo em %s (%d linhas).", out, len(metrics_df))


if __name__ == "__main__":
    main()
