"""
Executa o experimento com os prompts V4 (domain principle / asymmetric fingerprint).
Salva em results/experiment1/v4_{model}_{strategy}.csv
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
    MAX_WORKERS, MAX_CODE_CHARS, MAX_TOKENS_DEFAULT, MAX_TOKENS_COT,
    RESULTS_DIR, PROMPTS_DIR, SAMPLES_PER_CLASS_PER_LANG,
)
from src.data_loader import load_data
from src.logging_config import setup_logging
from src.providers import get_provider

load_dotenv()
logger = logging.getLogger(__name__)

MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "claude-haiku"]
STRATEGIES = ["zero_shot", "cot"]
MAX_TOKENS = {"zero_shot": MAX_TOKENS_DEFAULT, "cot": MAX_TOKENS_COT}

_tl = threading.local()


def _load_prompt(strategy: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{strategy}_v4.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build(template: str, code: str) -> str:
    return template.replace("{code}", code[:MAX_CODE_CHARS])


def _provider(model_name: str):
    if not hasattr(_tl, "p"):
        _tl.p = {}
    if model_name not in _tl.p:
        _tl.p[model_name] = get_provider(model_name)
    return _tl.p[model_name]


def _parse(text: str):
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            idx = cleaned.rfind('"prediction"')
            if idx == -1:
                raise ValueError("no prediction key")
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
        pred = str(data.get("prediction", "")).lower().strip()
        try:
            conf = int(data.get("confidence", 50))
        except Exception:
            conf = 50
        if pred not in ["human", "ai"]:
            pred = "unknown"
        return pred, conf, str(data.get("explanation", ""))
    except Exception as e:
        logger.debug("Parse error: %s", e)
        return "unknown", 50, "parse_error"


def _run(df: pd.DataFrame, model_name: str, strategy: str, template: str) -> list:
    rows = list(df.itertuples())
    results = [None] * len(rows)

    def process(i, row):
        prompt = _build(template, str(row.code))
        prov = _provider(model_name)
        try:
            raw = prov.call(prompt, max_tokens=MAX_TOKENS[strategy])
            if not raw or not raw.strip():
                pred, conf, expl = "unknown", 50, "empty_response"
            else:
                pred, conf, expl = _parse(raw)
        except Exception as e:
            logger.error("[%s|%s] id=%s: %s", model_name, strategy, row.Index, e)
            pred, conf, expl = "unknown", 50, f"api_error: {e}"
        return i, {
            "id": row.Index, "language": row.language_name,
            "true_label": row.label, "model": model_name,
            "strategy": strategy, "prediction": pred,
            "confidence": conf, "explanation": expl,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process, i, row): i for i, row in enumerate(rows)}
        with tqdm(total=len(rows), desc=f"{model_name} | {strategy} (v4)", unit="snippet") as pbar:
            for future in as_completed(futures):
                i, res = future.result()
                results[i] = res
                pbar.update(1)
    return results


def main():
    setup_logging()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    logger.info("Snippets carregados: %d", len(df))

    prompts = {s: _load_prompt(s) for s in STRATEGIES}

    for model_name in MODELS:
        for strategy in STRATEGIES:
            out = os.path.join(RESULTS_DIR, f"v4_{model_name.replace('/', '_')}_{strategy}.csv")
            if os.path.exists(out):
                logger.info("Ja existe, pulando: %s", out)
                continue
            rows = _run(df, model_name, strategy, prompts[strategy])
            pd.DataFrame(rows).to_csv(out, index=False)
            logger.info("Salvo: %s", out)


if __name__ == "__main__":
    main()
