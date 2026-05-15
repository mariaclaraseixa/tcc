import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    STRATEGIES, MODELS, MAX_CODE_CHARS,
    PROMPTS_DIR, RESULTS_DIR, API_SLEEP_S, SAMPLES_PER_CLASS_PER_LANG, MAX_WORKERS,
    MAX_TOKENS_DEFAULT, MAX_TOKENS_COT,
)
from src.data_loader import load_data
from src.logging_config import setup_logging
from src.providers import get_provider

_thread_local = threading.local()


def _get_thread_provider(model_name: str):
    if not hasattr(_thread_local, "providers"):
        _thread_local.providers = {}
    if model_name not in _thread_local.providers:
        _thread_local.providers[model_name] = get_provider(model_name)
    return _thread_local.providers[model_name]


load_dotenv()

logger = logging.getLogger(__name__)

_SIGNALS = (
    "POSITIVE HUMAN SIGNALS (non-exhaustive)\n"
    '- REPL/interactive artifacts: ">>>", "In [1]:", IPython prompts, traceback fragments\n'
    '- Version, authorship, or dated headers: "# v0.3 - fixed bug", "# author: jdoe", "// 2019-04-12"\n'
    "- Idiosyncratic or inconsistent style within the same file (mixed camelCase/snake_case, irregular indentation)\n"
    "- Commented-out experimental code, dead branches, leftover debug prints, hardcoded local paths\n"
    "- Personal/colloquial comments, typos, non-English comments, informal TODO/FIXME\n"
    "- Language idioms used distinctively: Haskell-style type signature in a comment, Perl one-liner condensation, C macro tricks\n"
    "- Partial or Stack-Overflow-style fragments, magic numbers without explanation\n\n"
    "POSITIVE AI SIGNALS (non-exhaustive)\n"
    "- Pedagogical comments restating what the next obvious line does\n"
    "- Uniform docstrings/type hints applied evenly even to trivial helpers\n"
    "- Defensive structure beyond task needs: broad try/except, redundant validation\n"
    '- Demonstration scaffolding appended to library code ("Example usage:", boilerplate `if __name__ == "__main__":`)\n'
    '- Overly generic, consistently placeholder-flavored names ("data", "result", "process_data")\n'
    "- Exhaustive edge-case coverage that the apparent task does not require\n"
    '- Residual chat framing leaking into comments/strings ("Here is the solution:", "Sure!", "Note that...")\n\n'
)

_JSON_OUT = (
    "Return ONLY this JSON object, with no surrounding text:\n"
    '{{"prediction": "human" or "ai", "confidence": <integer 0-100>,'
    ' "explanation": "<1-2 sentences citing the specific fingerprint(s) observed, or stating that no human fingerprints were found>"}}\n\n'
    "CODE SNIPPET:\n{code}"
)

_V4_HEADER = (
    'You are a forensic code analyst. Classify the following code snippet as "human" (human-written)'
    ' or "ai" (AI-generated).\n\n'
    "DOMAIN PRINCIPLE\n"
    "Human-authored code carries the fingerprints of lived experience: debugging sessions, evolving"
    " understanding, personal habits, shortcuts, idiosyncrasies accumulated over time."
    " AI-generated code lacks these fingerprints — it is structurally clean, stylistically uniform,"
    " and free of the residue of authorship.\n\n"
    "Your task is therefore asymmetric: look for human fingerprints. If you find them, classify as human."
    " If you cannot find them, classify as ai — because human code is distinguishable by what it carries,"
    " and AI code is the featureless default.\n\n"
)

_V4_HUMAN = (
    "HUMAN FINGERPRINTS (any one is sufficient evidence)\n"
    '- REPL/interactive artifacts: ">>>", "In [1]:", IPython prompts, traceback fragments mixed with code\n'
    '- Version, authorship, or dated annotations: "# v0.3 - fixed bug", "# author: jdoe", "// 2019-04-12"\n'
    "- Idiosyncratic or inconsistent style within the same file: mixed camelCase/snake_case,"
    " irregular indentation, inconsistent quote style\n"
    "- Commented-out experimental code, dead branches, leftover debug prints, hardcoded local paths\n"
    "- Personal, colloquial, or non-English comments; informal TODO/FIXME/HACK; typos\n"
    "- Distinctive language idioms: Haskell-style type signatures in comments, Perl one-liner"
    " condensation, C macro tricks, pointer-arithmetic offsets\n"
    "- Stack-Overflow-style fragments, magic numbers used without explanation\n\n"
)

_V4_AI = (
    "AI FINGERPRINTS (reinforce the default; not required to classify as ai)\n"
    "- Pedagogical comments restating what the immediately following line does\n"
    "- Uniform docstrings or type hints applied identically to every function including trivial ones\n"
    "- Defensive structure beyond what the task requires: broad try/except, redundant null checks\n"
    '- Demonstration scaffolding: "Example usage:" blocks, boilerplate'
    ' `if __name__ == "__main__":` around trivial code\n'
    '- Consistently placeholder-flavored names throughout: "data", "result", "process_data", "helper"\n'
    "- Exhaustive edge-case coverage for a task that does not require it\n"
    '- Residual chat framing in comments or strings: "Here is the solution:", "Sure!", "Note that..."\n\n'
)

_V4_LOGIC = (
    "DECISION LOGIC\n"
    "1. Scan for human fingerprints. Found at least one?\n"
    "   Strong, distinctive fingerprint: predict \"human\", confidence 75-90.\n"
    "   Weak or ambiguous fingerprint: predict \"human\", confidence 55-70.\n"
    "2. No human fingerprints found?\n"
    "   Predict \"ai\". Human code is distinguishable; absence of fingerprints is meaningful.\n"
    "   Also found AI fingerprints: confidence 70-85.\n"
    "   No AI fingerprints either (clean but anonymous code): confidence 60-70.\n"
    "3. Fingerprints AND strong AI markers both present?\n"
    "   Weigh specificity and quantity. Pick the stronger side. If truly tied: predict \"ai\", confidence 55-60.\n\n"
    "CONFIDENCE CALIBRATION\n"
    "- 85-95: multiple clear fingerprints or AI markers on the chosen side\n"
    "- 70-84: one clear fingerprint or AI marker\n"
    "- 55-69: weak or ambiguous evidence\n"
    "- <55:   do not use\n\n"
)

_V4_COT_STEPS = (
    "REASONING STEPS\n"
    "Step 1 — Fingerprint scan: go through the human fingerprints list. For each one present,"
    " quote the specific evidence briefly. If none found, write \"no human fingerprints observed\".\n"
    "Step 2 — AI marker scan: go through the AI fingerprints list. For each one present, note it briefly.\n"
    "Step 3 — Decide using the asymmetric rule:\n"
    "  Human fingerprints found: predict \"human\", confidence by strength (75-90 clear, 55-70 weak).\n"
    "  No fingerprints found: predict \"ai\". AI markers: confidence 70-85. No markers: confidence 60-70.\n"
    "  Both present and comparable: weigh; if tied predict \"ai\", confidence 55-60.\n\n"
    "CONFIDENCE CALIBRATION\n"
    "- 85-95: multiple clear fingerprints or AI markers on the chosen side\n"
    "- 70-84: one clear fingerprint or AI marker\n"
    "- 55-69: weak or ambiguous evidence\n"
    "- <55:   do not use\n\n"
)

_V4_ZS = _V4_HEADER + _V4_HUMAN + _V4_AI + _V4_LOGIC + _JSON_OUT
_V4_COT = (
    'You are a forensic code analyst. Classify the following code snippet as "human" (human-written)'
    ' or "ai" (AI-generated). Reason step by step before answering.\n\n'
    + _V4_HEADER[_V4_HEADER.find("DOMAIN PRINCIPLE"):]
    + _V4_HUMAN + _V4_AI + _V4_COT_STEPS
    + "CRITICAL OUTPUT REQUIREMENT: Execute all reasoning steps internally. Output ONLY the raw JSON"
    " object below — no markdown, no headers, no step text in the response:\n"
    '{{"prediction": "human" or "ai", "confidence": <integer 0-100>,'
    ' "explanation": "<1-2 sentences citing the specific fingerprint(s) observed, or stating that no human fingerprints were found>"}}\n\n'
    "CODE SNIPPET:\n{code}"
)

_COT_STEPS = (
    "REASONING PROCEDURE\n"
    "Step 1 — Human evidence inventory: list every positive human signal you actually observe,"
    " briefly quoting or pointing to the snippet. If none, write \"none observed\".\n"
    "Step 2 — AI evidence inventory: list every positive AI signal you actually observe,"
    " briefly quoting or pointing to the snippet. If none, write \"none observed\".\n"
    "Step 3 — Weigh: compare strength, distinctiveness and quantity on each side;"
    " note any signals that are ambiguous.\n"
)

_COT_OUT = (
    "CRITICAL OUTPUT REQUIREMENT: Do NOT write your reasoning steps in the response."
    " Execute all steps internally, then output ONLY the raw JSON object below"
    " — no markdown, no headers, no step labels, no other text:\n"
    '{{"prediction": "human" or "ai", "confidence": <integer 0-100>,'
    ' "explanation": "<1-2 sentences citing the specific positive signals that drove the decision>"}}\n\n'
    "CODE SNIPPET:\n{code}"
)

# ---------------------------------------------------------------------------
# Prompts model-specific (mesmo catálogo de sinais, priors e regras distintos)
# ---------------------------------------------------------------------------

PROMPTS = {
    # V4: prompt universal baseado em princípio de domínio (igual para todos os modelos)
    "gpt-4o-mini": {
        "zero_shot": _V4_ZS,
        "cot": _V4_COT,
    },
    "gemini-2.5-flash": {
        "zero_shot": _V4_ZS,
        "cot": _V4_COT,
    },
    "claude-haiku": {
        "zero_shot": _V4_ZS,
        "cot": _V4_COT,
    },
}

_PROMPTS_LEGACY = {
    "_unused": {
        "zero_shot": (
            'You are a forensic code analyst. Classify a code snippet as either "human" (human-written)'
            ' or "ai" (AI-generated) using only evidence visible inside the snippet itself.\n\n'
            "CRITICAL RULES\n"
            "1. No prior assumption about class balance. The true distribution is unknown and may be highly"
            " skewed. Decide strictly by the strength of evidence you actually observe — never by elimination.\n"
            "2. Look for POSITIVE EVIDENCE on BOTH sides independently. Absence of human signals is NOT"
            " evidence of AI authorship, and vice versa. Absence of AI signals is NOT evidence of human authorship.\n"
            "3. If neither side has strong positive evidence, choose the class with the weakest contraindications"
            ' and report confidence below 60. Do NOT default to "human" as a safe answer'
            " — both classes are equally likely a priori in this task.\n"
            '4. When indicators are roughly balanced on both sides, predict "ai". The pre-training distribution'
            " biases models toward predicting \"human\" under uncertainty; this rule counteracts that bias.\n\n"
            + _SIGNALS
            + "CONFIDENCE CALIBRATION\n"
            "- 90-100: multiple strong, distinctive signals on the chosen side\n"
            "- 70-89:  clear signals but limited in number or somewhat ambiguous\n"
            "- 50-69:  weak or conflicting signals; near toss-up\n"
            "- <50:    do not use\n\n"
            + _JSON_OUT
        ),
        "cot": (
            'You are a forensic code analyst. Classify a code snippet as either "human" (human-written)'
            ' or "ai" (AI-generated) using only evidence visible inside the snippet itself.'
            " Reason step by step before producing the final answer.\n\n"
            "CRITICAL RULES\n"
            "1. No prior assumption about class balance. The true distribution is unknown and may be highly"
            " skewed. Decide strictly by the strength of evidence you actually observe — never by elimination.\n"
            "2. Inventory POSITIVE EVIDENCE for BOTH classes independently. Absence of one side's signals is"
            " NOT evidence for the other side.\n"
            "3. If neither side has strong positive evidence, choose the class with the weakest contraindications"
            ' and report confidence below 60. Do NOT default to "human" as a safe answer'
            " — both classes are equally likely a priori in this task.\n"
            '4. When indicators are roughly balanced on both sides, predict "ai". The pre-training distribution'
            " biases models toward predicting \"human\" under uncertainty; this rule counteracts that bias.\n\n"
            + _SIGNALS
            + _COT_STEPS
            + "Step 4 — Decide: pick the class with stronger positive evidence."
            ' If evidence is balanced or weak on both sides, predict "ai" (counteracting the human-default bias)'
            " and report confidence below 60.\n\n"
            "CONFIDENCE CALIBRATION\n"
            "- 90-100: multiple strong, distinctive signals on the chosen side\n"
            "- 70-89:  clear signals but limited or somewhat ambiguous\n"
            "- 50-69:  weak or conflicting signals; near toss-up\n"
            "- <50:    do not use\n\n"
            + _COT_OUT
        ),
    },

    # Gemini 2.5 Flash: V3.1 (melhor resultado observado para este modelo)
    "gemini-2.5-flash": {
        "zero_shot": (
            'You are a forensic code analyst. Classify a code snippet as either "human" (human-written)'
            ' or "ai" (AI-generated) using only evidence visible inside the snippet itself.\n\n'
            "CRITICAL RULES\n"
            "1. No prior assumption about class balance. The true distribution is unknown and may be highly"
            " skewed. Decide strictly by the strength of evidence you actually observe — never by elimination.\n"
            "2. Look for POSITIVE EVIDENCE on BOTH sides independently. Absence of human signals is NOT"
            " evidence of AI authorship, and vice versa. Absence of AI signals is NOT evidence of human authorship.\n"
            "3. If neither side has strong positive evidence, choose the class with the weakest contraindications"
            ' and report confidence below 60. Do NOT default to "human" as a safe answer'
            " — both classes are equally likely a priori in this task.\n"
            '4. When indicators are roughly balanced on both sides, predict "ai". The pre-training distribution'
            " biases models toward predicting \"human\" under uncertainty; this rule counteracts that bias.\n\n"
            + _SIGNALS
            + "CONFIDENCE CALIBRATION\n"
            "- 90-100: multiple strong, distinctive signals on the chosen side\n"
            "- 70-89:  clear signals but limited in number or somewhat ambiguous\n"
            "- 50-69:  weak or conflicting signals; near toss-up\n"
            "- <50:    do not use\n\n"
            + _JSON_OUT
        ),
        "cot": (
            'You are a forensic code analyst. Classify a code snippet as either "human" (human-written)'
            ' or "ai" (AI-generated) using only evidence visible inside the snippet itself.'
            " Reason step by step before producing the final answer.\n\n"
            "CRITICAL RULES\n"
            "1. No prior assumption about class balance. The true distribution is unknown and may be highly"
            " skewed. Decide strictly by the strength of evidence you actually observe — never by elimination.\n"
            "2. Inventory POSITIVE EVIDENCE for BOTH classes independently. Absence of one side's signals is"
            " NOT evidence for the other side.\n"
            "3. If neither side has strong positive evidence, choose the class with the weakest contraindications"
            ' and report confidence below 60. Do NOT default to "human" as a safe answer'
            " — both classes are equally likely a priori in this task.\n"
            '4. When indicators are roughly balanced on both sides, predict "ai". The pre-training distribution'
            " biases models toward predicting \"human\" under uncertainty; this rule counteracts that bias.\n\n"
            + _SIGNALS
            + _COT_STEPS
            + "Step 4 — Decide: pick the class with stronger positive evidence."
            ' If evidence is balanced or weak on both sides, predict "ai" (counteracting the human-default bias)'
            " and report confidence below 60.\n\n"
            "CONFIDENCE CALIBRATION\n"
            "- 90-100: multiple strong, distinctive signals on the chosen side\n"
            "- 70-89:  clear signals but limited or somewhat ambiguous\n"
            "- 50-69:  weak or conflicting signals; near toss-up\n"
            "- <50:    do not use\n\n"
            + _COT_OUT
        ),
    },

    # Claude Haiku: recall 0.165 → prior AI explícito e framing unidirecional
    "claude-haiku": {
        "zero_shot": (
            'You are a forensic code analyst. Your task: determine if a code snippet was written by a human'
            ' or generated by AI.\n\n'
            "YOUR PRIOR IS: AI-GENERATED. You are verifying whether the code carries unmistakable"
            " fingerprints of human authorship. If you cannot find those fingerprints, your prior stands.\n\n"
            "HUMAN AUTHORSHIP FINGERPRINTS — look for these specifically:\n"
            '- REPL/interactive artifacts: ">>>", "In [1]:", IPython prompts, traceback fragments\n'
            '- Version, authorship, or dated headers: "# v0.3 - fixed bug", "# author: jdoe", "// 2019-04-12"\n'
            "- Idiosyncratic or inconsistent style within the same file (mixed camelCase/snake_case, irregular indentation)\n"
            "- Commented-out experimental code, dead branches, leftover debug prints, hardcoded local paths\n"
            "- Personal/colloquial comments, typos, non-English comments, informal TODO/FIXME\n"
            "- Language idioms used distinctively: Haskell-style type signature in a comment, Perl one-liner condensation, C macro tricks\n"
            "- Partial or Stack-Overflow-style fragments, magic numbers without explanation\n\n"
            "AI INDICATORS — these confirm your prior:\n"
            "- Pedagogical comments restating what the next obvious line does\n"
            "- Uniform docstrings/type hints applied evenly even to trivial helpers\n"
            "- Defensive structure beyond task needs: broad try/except, redundant validation\n"
            '- Demonstration scaffolding ("Example usage:", boilerplate `if __name__ == "__main__":`)\n'
            '- Overly generic placeholder names ("data", "result", "process_data")\n'
            "- Exhaustive edge-case coverage that the apparent task does not require\n"
            '- Residual chat framing in comments/strings ("Here is the solution:", "Sure!", "Note that...")\n\n'
            "DECISION\n"
            "- Human fingerprints found AND no strong AI indicators? → predict \"human\", confidence 70-90.\n"
            "- Human fingerprints found BUT AI indicators also present? → predict \"human\" if fingerprints are"
            " stronger, else predict \"ai\".\n"
            "- No human fingerprints found, regardless of AI indicators? → predict \"ai\","
            " confidence 65-80 (prior confirmed).\n"
            "- Ambiguous or very weak signals on both sides? → predict \"ai\", confidence 60-65.\n\n"
            "CONFIDENCE CALIBRATION\n"
            "- 85-95: multiple clear fingerprints or indicators on the chosen side\n"
            "- 70-84: one clear fingerprint or indicator\n"
            "- 60-69: prior maintained with no fingerprints found\n"
            "- <60:   do not use\n\n"
            + _JSON_OUT
        ),
        "cot": (
            'You are a forensic code analyst. Determine if this code was written by a human or generated by AI.'
            " Reason step by step.\n\n"
            "YOUR PRIOR IS: AI-GENERATED. You are verifying whether the code carries unmistakable"
            " fingerprints of human authorship. If you cannot find those fingerprints, your prior stands.\n\n"
            "HUMAN AUTHORSHIP FINGERPRINTS:\n"
            '- REPL/interactive artifacts: ">>>", "In [1]:", IPython prompts, traceback fragments\n'
            '- Version, authorship, or dated headers: "# v0.3 - fixed bug", "# author: jdoe", "// 2019-04-12"\n'
            "- Idiosyncratic or inconsistent style within the same file (mixed camelCase/snake_case, irregular indentation)\n"
            "- Commented-out experimental code, dead branches, leftover debug prints, hardcoded local paths\n"
            "- Personal/colloquial comments, typos, non-English comments, informal TODO/FIXME\n"
            "- Language idioms used distinctively: Haskell-style type signature in a comment, Perl one-liner condensation, C macro tricks\n"
            "- Partial or Stack-Overflow-style fragments, magic numbers without explanation\n\n"
            "AI INDICATORS:\n"
            "- Pedagogical comments restating what the next obvious line does\n"
            "- Uniform docstrings/type hints applied evenly even to trivial helpers\n"
            "- Defensive structure beyond task needs: broad try/except, redundant validation\n"
            '- Demonstration scaffolding ("Example usage:", boilerplate if __name__ == "__main__":)\n'
            '- Overly generic placeholder names ("data", "result", "process_data")\n'
            "- Exhaustive edge-case coverage that the apparent task does not require\n"
            '- Residual chat framing in comments/strings ("Here is the solution:", "Sure!", "Note that...")\n\n'
            "STEPS\n"
            "Step 1 — Search for human fingerprints. List each found, quoting the evidence. If none, write \"none found\".\n"
            "Step 2 — Search for AI indicators. List each found, quoting the evidence. If none, write \"none found\".\n"
            "Step 3 — Decide: human fingerprints found → lean human; none found → AI prior stands.\n\n"
            "DECISION RULES\n"
            "- Human fingerprints found, no AI indicators: predict \"human\", confidence 70-90.\n"
            "- Human fingerprints found, AI indicators also present: pick stronger side.\n"
            "- No human fingerprints, regardless of AI indicators: predict \"ai\", confidence 65-80.\n"
            "- Ambiguous: predict \"ai\", confidence 60-65.\n\n"
            "CONFIDENCE CALIBRATION\n"
            "- 85-95: multiple clear signals on the chosen side\n"
            "- 70-84: one clear signal\n"
            "- 60-69: prior maintained, no fingerprints\n"
            "- <60:   do not use\n\n"
            + _COT_OUT
        ),
    },
}

# Fallback para modelos sem prompt específico
def _get_prompt(model_name: str, strategy: str) -> str:
    if model_name in PROMPTS:
        return PROMPTS[model_name][strategy]
    return PROMPTS["gemini-2.5-flash"][strategy]


def build_prompt(model_name: str, strategy: str, code: str, max_chars: int) -> str:
    return _get_prompt(model_name, strategy).format(code=code[:max_chars])


def parse_response(text: str):
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()

        # Tenta parse direto primeiro
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback robusto: localiza o último objeto JSON que contém "prediction"
            # usando busca pelo último { antes de "prediction" e balanceamento de chaves
            key_idx = cleaned.rfind('"prediction"')
            if key_idx == -1:
                raise ValueError("Chave 'prediction' não encontrada na resposta")
            start = cleaned.rfind('{', 0, key_idx)
            if start == -1:
                raise ValueError("Abertura de objeto JSON não encontrada")
            depth, end = 0, -1
            for i, ch in enumerate(cleaned[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                raise ValueError("Fechamento de objeto JSON não encontrado")
            data = json.loads(cleaned[start:end])

        prediction = str(data.get("prediction", "")).lower().strip()
        try:
            confidence = int(data.get("confidence", 50))
        except (ValueError, TypeError):
            confidence = 50
        explanation = str(data.get("explanation", ""))
        if prediction not in ["human", "ai"]:
            prediction = "unknown"
        return prediction, confidence, explanation
    except Exception as e:
        logger.debug("Falha ao parsear resposta da API: %s | texto: %r", e, text)
        return "unknown", 50, "parse_error"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}min{s:02d}s" if m else f"{s}s"


def run_experiment(df: pd.DataFrame, model_name: str, strategy: str,
                   combo_num: int = None, total_combos: int = None) -> list:
    total = len(df)
    logger.info(
        "Iniciando experimento — modelo=%s | estratégia=%s | amostras=%d",
        model_name, strategy, total,
    )

    prefix = f"[{combo_num}/{total_combos}] " if combo_num and total_combos else ""

    max_tokens = MAX_TOKENS_COT if strategy == "cot" else MAX_TOKENS_DEFAULT
    rows = list(df.itertuples())
    results = [None] * total

    def process_sample(i: int, row) -> tuple:
        sample_id = row.Index
        lang = row.language_name
        true_label = row.label

        logger.debug(
            "[%s|%s] Amostra %d/%d — id=%s, lang=%s, label=%s",
            model_name, strategy, i + 1, total, sample_id, lang, true_label,
        )

        provider = _get_thread_provider(model_name)

        if not provider.is_available:
            logger.warning(
                "[%s|%s] Provider indisponível — pulando amostra id=%s (%d/%d).",
                model_name, strategy, sample_id, i + 1, total,
            )
            prediction, confidence, explanation = "unknown", 0, "provider_unavailable"
        else:
            prompt = build_prompt(model_name, strategy, str(row.code), MAX_CODE_CHARS)
            try:
                raw = provider.call(prompt, max_tokens=max_tokens)
                if not raw or not raw.strip():
                    logger.warning(
                        "[%s|%s] API retornou resposta vazia para id=%s.",
                        model_name, strategy, sample_id,
                    )
                    prediction, confidence, explanation = "unknown", 50, "empty_response"
                else:
                    prediction, confidence, explanation = parse_response(raw)
                    if prediction == "unknown":
                        logger.warning(
                            "[%s|%s] Resposta não parseável para id=%s. Prévia: %r",
                            model_name, strategy, sample_id, raw[:120],
                        )
                    else:
                        logger.debug(
                            "[%s|%s] id=%s → predição=%s (confiança=%d)",
                            model_name, strategy, sample_id, prediction, confidence,
                        )
            except Exception as e:
                logger.error(
                    "[%s|%s] Erro ao chamar API para id=%s: %s",
                    model_name, strategy, sample_id, e,
                )
                prediction, confidence, explanation = "unknown", 50, f"api_error: {e}"

        time.sleep(API_SLEEP_S)
        return i, {
            "id": sample_id,
            "language": lang,
            "true_label": true_label,
            "model": model_name,
            "strategy": strategy,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, i, row): i for i, row in enumerate(rows)}
        with tqdm(total=total, desc=f"{prefix}{model_name} | {strategy}",
                  mininterval=8, unit="amostra") as pbar:
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result
                pbar.update(1)

    n_valid = sum(1 for r in results if r["prediction"] not in ("unknown",))
    n_unknown = sum(1 for r in results if r["prediction"] == "unknown" and "provider_unavailable" not in r.get("explanation", ""))
    n_skipped = sum(1 for r in results if "provider_unavailable" in r.get("explanation", ""))

    logger.info(
        "[%s|%s] Concluído — válidas=%d | parse_error/api_error=%d | puladas (provider down)=%d",
        model_name, strategy, n_valid, n_unknown, n_skipped,
    )
    if n_skipped > 0:
        logger.warning(
            "[%s|%s] %d amostra(s) pulada(s) por indisponibilidade do provider.",
            model_name, strategy, n_skipped,
        )

    return results


def main():
    log_path = setup_logging()

    logger.info("=" * 60)
    logger.info("PIPELINE DE EXPERIMENTOS INICIADO")
    logger.info("Modelos    : %s", MODELS)
    logger.info("Estratégias: %s", STRATEGIES)
    logger.info("=" * 60)

    df = load_data(sample_per_class_per_lang=SAMPLES_PER_CLASS_PER_LANG)
    n_amostras = len(df)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_combos = len(MODELS) * len(STRATEGIES)
    n_executadas = n_puladas = 0
    elapsed_times: list[float] = []
    pipeline_start = time.time()

    for model in MODELS:
        for strategy in STRATEGIES:
            output_file = f"{RESULTS_DIR}/{model.replace('/', '_')}_{strategy}.csv"

            if os.path.exists(output_file):
                logger.info("Arquivo já existe, pulando: %s", output_file)
                n_puladas += 1
                continue

            combo_num = n_executadas + n_puladas + 1
            logger.info("-" * 60)
            logger.info("[%d/%d] Iniciando: %s | %s", combo_num, total_combos, model, strategy)

            t0 = time.time()
            results = run_experiment(df, model, strategy, combo_num, total_combos)
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)

            pd.DataFrame(results).to_csv(output_file, index=False)

            sps = elapsed / n_amostras
            remaining = total_combos - combo_num
            if remaining > 0:
                eta_s = remaining * (sum(elapsed_times) / len(elapsed_times))
                eta_str = f"~{_fmt_time(eta_s)}"
            else:
                eta_str = "—"

            logger.info(
                "[%d/%d] ✓ %s | %s — %s (%.1fs/amostra) | restam %d combos, ETA: %s",
                combo_num, total_combos, model, strategy,
                _fmt_time(elapsed), sps, remaining, eta_str,
            )
            n_executadas += 1

    total_elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(
        "PIPELINE CONCLUÍDO — executadas=%d | puladas=%d | total=%d | tempo total: %s",
        n_executadas, n_puladas, total_combos, _fmt_time(total_elapsed),
    )
    if log_path:
        logger.info("Log completo: %s", log_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
