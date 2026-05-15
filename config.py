# Dataset
DATASET_NAME = "isThisYouLLM/H-AIRosettaMP"
LANGUAGES = ["Python", "C", "C++"]
SAMPLES_PER_CLASS_PER_LANG = 50

# Experiment
CURRENT_EXPERIMENT = "experiment1"
STRATEGIES = ["zero_shot", "cot"]
MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"]
COMPLEXITY_MODEL = "gpt-4o-mini"
MAX_CODE_CHARS = 3000

# Tokens de saída por estratégia — CoT precisa de mais espaço para cobrir 5 dimensões
MAX_TOKENS_DEFAULT = 300
MAX_TOKENS_COT = 500

# Paths (derivados do experimento atual)
_EXPERIMENT_BASE = f"experiments/{CURRENT_EXPERIMENT}"
PROMPTS_DIR = f"{_EXPERIMENT_BASE}/prompts"
RESULTS_DIR = f"results/{CURRENT_EXPERIMENT}"
METRICS_DIR = f"results/{CURRENT_EXPERIMENT}/metrics"
LOG_DIR = f"logs/{CURRENT_EXPERIMENT}"

# API timing
API_SLEEP_S = 0.3
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_WAIT_S = 15
MAX_CONSECUTIVE_FAILURES = 3
MAX_WORKERS = 5

# Rate limits por modelo — fonte: headers de resposta + mensagens 429 + documentação oficial
# Anthropic Tier 1: 50 RPM | 50 000 ITPM | 10 000 OTPM
# Com max_tokens=300 e CoT (~250 tokens/resposta): 10000/250 = 40 req/min → usar 30 (margem)
# OpenAI Tier 1: 10 000 RPM | 200 000 TPM → sem restrição prática
# Gemini pay-as-you-go: sem 429 com 5 workers → ≥ 1 000 RPM → usar 500 (conservador)
MODEL_RPM = {
    "gpt-4o-mini":      500,   # real: 10 000 RPM
    "claude-haiku":      30,   # real: 50 RPM (RPM) / ~40 RPM (OTPM com CoT) → 30 com margem
    "gemini-2.5-flash": 500,   # real: ≥ 1 000 RPM (tier pago)
}

# Actual API model IDs (podem diferir dos nomes usados como chave em MODELS)
OPENAI_MODEL_ID = "gpt-4o-mini"
GEMINI_MODEL_ID = "gemini-2.5-flash"
CLAUDE_MODEL_ID = "claude-haiku-4-5-20251001"
