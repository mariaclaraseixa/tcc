"""
Gera complexity_scores.csv com colunas: id, language, true_label, model, complexity_score.
- Le IDs dos CSVs de resultados existentes (zero_shot de cada modelo)
- Recupera os snippets do dataset HuggingFace
- Envia o prompt de complexidade V1 para os 3 modelos via API
- Resistente a interrupcoes: pula IDs ja processados
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
    MAX_WORKERS, MAX_CODE_CHARS, MAX_TOKENS_DEFAULT,
    RESULTS_DIR, PROMPTS_DIR, SAMPLES_PER_CLASS_PER_LANG,
)
from src.data_loader import load_data
from src.logging_config import setup_logging
from src.providers import get_provider

load_dotenv()
logger = logging.getLogger(__name__)

MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "claude-haiku"]
OUTPUT_FILE = os.path.join(RESULTS_DIR, "complexity_scores.csv")

_tl = threading.local()


def _load_prompt() -> str:
    with open(os.path.join(PROMPTS_DIR, "complexity.txt"), "r", encoding="utf-8") as f:
        return f.read()


def _provider(model_name: str):
    if not hasattr(_tl, "p"):
        _tl.p = {}
    if model_name not in _tl.p:
        _tl.p[model_name] = get_provider(model_name)
    return _tl.p[model_name]


def _parse(text: str) -> int:
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            idx = cleaned.rfind('"complexity_score"')
            if idx == -1:
                return -1
            start = cleaned.rfind('{', 0, idx)
            depth = end = 0
            for i, ch in enumerate(cleaned[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            data = json.loads(cleaned[start:end])
        return int(data.get("complexity_score", -1))
    except Exception as e:
        logger.debug("Parse error: %s", e)
        return -1


def _load_existing() -> set:
    if not os.path.exists(OUTPUT_FILE):
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    return set(zip(df["id"], df["model"]))


def _run_model(df: pd.DataFrame, model_name: str, prompt_tpl: str, existing: set) -> list:
    rows = [r for r in df.itertuples() if (r.Index, model_name) not in existing]
    if not rows:
        logger.info("[%s] Todos os IDs ja processados.", model_name)
        return []

    results = []

    def process(row):
        prompt = prompt_tpl.replace("{code}", str(row.code)[:MAX_CODE_CHARS])
        prov = _provider(model_name)
        try:
            raw = prov.call(prompt, max_tokens=MAX_TOKENS_DEFAULT)
            score = _parse(raw) if raw and raw.strip() else -1
        except Exception as e:
            logger.error("[%s] id=%s: %s", model_name, row.Index, e)
            score = -1
        return {
            "id": row.Index,
            "language": row.language_name,
            "true_label": row.label,
            "model": model_name,
            "complexity_score": score,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        with tqdm(total=len(rows), desc=f"{model_name} (complexity)", unit="snippet") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    return results


def main():
    setup_logging()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    logger.info("Snippets carregados: %d", len(df))

    prompt_tpl = _load_prompt()
    existing = _load_existing()
    logger.info("Pares (id, model) ja processados: %d", len(existing))

    all_new = []
    for model_name in MODELS:
        all_new.extend(_run_model(df, model_name, prompt_tpl, existing))

    if all_new:
        new_df = pd.DataFrame(all_new)
        if os.path.exists(OUTPUT_FILE):
            old_df = pd.read_csv(OUTPUT_FILE)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(OUTPUT_FILE, index=False)
        logger.info("Salvo: %s (%d linhas)", OUTPUT_FILE, len(combined))
    else:
        logger.info("Nenhum novo resultado — arquivo nao alterado.")

    # resumo
    final = pd.read_csv(OUTPUT_FILE)
    valid = final[final["complexity_score"] >= 0]
    print(f"\nTotal: {len(final)} | Validos: {len(valid)} | Erros: {len(final)-len(valid)}")
    print(valid.groupby("model")["complexity_score"].describe().round(2))


if __name__ == "__main__":
    main()
