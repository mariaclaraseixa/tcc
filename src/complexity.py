"""
Analisa complexidade de código para todos os snippets de all_predictions.csv,
rodando os 3 modelos em paralelo.
Saída: results/<experimento>/complexity.csv
Colunas: id, language, model, complexity_score, technical_justification, identified_elements
"""
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    MODELS, MAX_CODE_CHARS, RESULTS_DIR, PROMPTS_DIR,
    SAMPLES_PER_CLASS_PER_LANG, MAX_WORKERS, MAX_TOKENS_DEFAULT,
)
from src.data_loader import load_data
from src.logging_config import setup_logging
from src.providers import get_provider

load_dotenv()

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _load_prompt() -> str:
    path = f"{PROMPTS_DIR}/complexity.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_thread_provider(model_name: str):
    if not hasattr(_thread_local, "providers"):
        _thread_local.providers = {}
    if model_name not in _thread_local.providers:
        _thread_local.providers[model_name] = get_provider(model_name)
    return _thread_local.providers[model_name]


def _parse(text: str) -> tuple:
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        score = int(data.get("complexity_score", -1))
        justification = str(data.get("technical_justification", ""))
        elements = data.get("identified_elements", [])
        if not isinstance(elements, list):
            elements = [str(elements)]
        return score, justification, json.dumps(elements)
    except Exception as e:
        logger.debug("Falha ao parsear resposta de complexidade: %s | %r", e, text)
        return -1, "parse_error", "[]"


def run_complexity_for_model(df: pd.DataFrame, model_name: str, prompt_template: str) -> list:
    total = len(df)
    logger.info("Complexidade — modelo=%s | amostras=%d", model_name, total)
    rows = list(df.itertuples())
    results = [None] * total

    def process_sample(i: int, row) -> tuple:
        code = str(row.code)[:MAX_CODE_CHARS]
        prompt = prompt_template.replace("{code}", code)
        provider = _get_thread_provider(model_name)

        if not provider.is_available:
            logger.warning("[complexity|%s] Provider indisponível — id=%s", model_name, row.Index)
            return i, {
                "id": row.Index, "language": row.language_name,
                "model": model_name,
                "complexity_score": -1,
                "technical_justification": "provider_unavailable",
                "identified_elements": "[]",
            }

        try:
            raw = provider.call(prompt, max_tokens=MAX_TOKENS_DEFAULT)
            if not raw or not raw.strip():
                score, justification, elements = -1, "empty_response", "[]"
            else:
                score, justification, elements = _parse(raw)
                logger.debug("[complexity|%s] id=%s → score=%d", model_name, row.Index, score)
        except Exception as e:
            logger.error("[complexity|%s] Erro API id=%s: %s", model_name, row.Index, e)
            score, justification, elements = -1, f"api_error: {e}", "[]"

        return i, {
            "id": row.Index, "language": row.language_name,
            "model": model_name,
            "complexity_score": score,
            "technical_justification": justification,
            "identified_elements": elements,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, i, row): i for i, row in enumerate(rows)}
        with tqdm(total=total, desc=f"complexity | {model_name}",
                  mininterval=8, unit="amostra") as pbar:
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result
                pbar.update(1)

    n_valid = sum(1 for r in results if r["complexity_score"] >= 0)
    logger.info("[%s] Complexidade concluída — válidas=%d | erros=%d", model_name, n_valid, total - n_valid)
    return results


def main():
    setup_logging()

    output_file = f"{RESULTS_DIR}/complexity.csv"
    if os.path.exists(output_file):
        logger.info("Arquivo já existe, pulando: %s", output_file)
        return

    # Carrega IDs únicos de all_predictions.csv para garantir mesmos snippets
    all_preds_path = f"{RESULTS_DIR}/all_predictions.csv"
    if not os.path.exists(all_preds_path):
        logger.error("Arquivo não encontrado: %s — execute o experimento primeiro.", all_preds_path)
        return

    valid_ids = set(pd.read_csv(all_preds_path)["id"].unique())
    logger.info("IDs únicos em all_predictions.csv: %d", len(valid_ids))

    df = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    df = df[df.index.isin(valid_ids)].copy()
    logger.info("Snippets carregados para análise: %d", len(df))

    prompt_template = _load_prompt()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for model_name in MODELS:
        model_results = run_complexity_for_model(df, model_name, prompt_template)
        all_results.extend(model_results)

    out_df = pd.DataFrame(all_results, columns=[
        "id", "language", "model", "complexity_score",
        "technical_justification", "identified_elements",
    ])
    out_df.to_csv(output_file, index=False)
    logger.info("Complexidade salva: %s (%d linhas)", output_file, len(out_df))


if __name__ == "__main__":
    main()
