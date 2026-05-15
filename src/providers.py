import logging
import os
import threading
import time
from abc import ABC, abstractmethod

import anthropic
from openai import OpenAI
from google import genai as google_genai

from config import (
    OPENAI_MODEL_ID, GEMINI_MODEL_ID, CLAUDE_MODEL_ID,
    RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_BASE_WAIT_S, MAX_CONSECUTIVE_FAILURES,
    MODEL_RPM, MAX_TOKENS_DEFAULT,
)

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Throttle thread-safe: garante no máximo `rpm` chamadas por minuto."""
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()


_rate_limiters: dict[str, _RateLimiter] = {
    model: _RateLimiter(rpm) for model, rpm in MODEL_RPM.items()
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Variável de ambiente '{name}' não definida. Verifique o arquivo .env.")
    return value


class AIProviderStrategy(ABC):
    @abstractmethod
    def call(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        pass


class OpenAIStrategy(AIProviderStrategy):
    def __init__(self, model: str = OPENAI_MODEL_ID, client=None):
        self.model = model
        self.client = client or OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        logger.debug("OpenAIStrategy criada (model=%s)", self.model)

    def call(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class GeminiStrategy(AIProviderStrategy):
    def __init__(self, model: str = GEMINI_MODEL_ID, client=None):
        self.model = model
        self.client = client or google_genai.Client(api_key=_require_env("GOOGLE_API_KEY"))
        logger.debug("GeminiStrategy criada (model=%s)", self.model)

    def call(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text.strip()


class ClaudeStrategy(AIProviderStrategy):
    def __init__(self, model: str = CLAUDE_MODEL_ID, client=None):
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
        logger.debug("ClaudeStrategy criada (model=%s)", self.model)

    def call(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


def _is_rate_limit(e: Exception) -> bool:
    if isinstance(e, anthropic.RateLimitError):
        return True
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resource exhausted" in msg or "rate limit" in msg


def _is_network_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in [
        "getaddrinfo failed", "connect error", "read error", "write error",
        "connection reset", "connection aborted", "connection refused",
        "network unreachable", "10053", "10054", "11001", "eof occurred",
        "ssl", "timed out", "timeout",
    ])


_NETWORK_RETRY_MAX = 5
_NETWORK_RETRY_WAIT_S = 10


class ModelState(ABC):
    @abstractmethod
    def call(self, provider: "StatefulProvider", prompt: str,
             max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        pass


class AvailableState(ModelState):
    def call(self, provider: "StatefulProvider", prompt: str,
             max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        logger.debug("[%s] Chamando API (estado: Available)", provider.model_name)
        limiter = _rate_limiters.get(provider.model_name)
        if limiter:
            limiter.acquire()
        for network_attempt in range(_NETWORK_RETRY_MAX + 1):
            try:
                result = provider.strategy.call(prompt, max_tokens=max_tokens)
                if provider._consecutive_failures > 0:
                    logger.info(
                        "[%s] Chamada bem-sucedida após %d falha(s) anterior(es).",
                        provider.model_name, provider._consecutive_failures,
                    )
                provider._consecutive_failures = 0
                return result
            except Exception as e:
                if _is_rate_limit(e):
                    logger.warning(
                        "[%s] Rate limit detectado — ativando backoff. Detalhe: %s",
                        provider.model_name, e,
                    )
                    provider.set_state(RateLimitedState())
                    return provider.state.call(provider, prompt, max_tokens=max_tokens)

                if _is_network_error(e) and network_attempt < _NETWORK_RETRY_MAX:
                    logger.warning(
                        "[%s] Erro de rede (tentativa %d/%d), aguardando %ds... Detalhe: %s",
                        provider.model_name, network_attempt + 1, _NETWORK_RETRY_MAX,
                        _NETWORK_RETRY_WAIT_S, e,
                    )
                    time.sleep(_NETWORK_RETRY_WAIT_S)
                    continue

                provider._consecutive_failures += 1
                logger.error(
                    "[%s] Erro de API (falha consecutiva %d/%d): %s",
                    provider.model_name, provider._consecutive_failures, provider.max_failures, e,
                    exc_info=True,
                )
                if provider._consecutive_failures >= provider.max_failures:
                    logger.critical(
                        "[%s] %d falhas consecutivas atingidas — modelo marcado como INDISPONÍVEL.",
                        provider.model_name, provider._consecutive_failures,
                    )
                    provider.set_state(UnavailableState())
                raise


class RateLimitedState(ModelState):
    _MAX_RETRIES = RATE_LIMIT_MAX_RETRIES
    _BASE_WAIT_S = RATE_LIMIT_BASE_WAIT_S

    def __init__(self) -> None:
        self._retries = 0

    def call(self, provider: "StatefulProvider", prompt: str,
             max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        while self._retries < self._MAX_RETRIES:
            wait = self._BASE_WAIT_S * (2 ** self._retries)
            logger.warning(
                "[%s] Aguardando %ds antes de retentar (tentativa %d/%d)...",
                provider.model_name, wait, self._retries + 1, self._MAX_RETRIES,
            )
            time.sleep(wait)
            self._retries += 1

            try:
                result = provider.strategy.call(prompt, max_tokens=max_tokens)
                logger.info(
                    "[%s] Recuperado após %ds de backoff — retornando para estado Available.",
                    provider.model_name, wait,
                )
                provider.set_state(AvailableState())
                provider._consecutive_failures = 0
                return result
            except Exception as e:
                if _is_rate_limit(e):
                    logger.warning(
                        "[%s] Rate limit persistente na tentativa %d.",
                        provider.model_name, self._retries,
                    )
                    continue
                logger.error(
                    "[%s] Erro não relacionado a rate limit durante backoff: %s",
                    provider.model_name, e,
                    exc_info=True,
                )
                provider.set_state(UnavailableState())
                raise

        logger.critical(
            "[%s] Máximo de %d retentativas de rate limit atingido — modelo marcado como INDISPONÍVEL.",
            provider.model_name, self._MAX_RETRIES,
        )
        provider.set_state(UnavailableState())
        raise RuntimeError(
            f"Rate limit persistente em '{provider.model_name}' após {self._MAX_RETRIES} tentativas."
        )


class UnavailableState(ModelState):
    def call(self, provider: "StatefulProvider", prompt: str,
             max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        logger.debug("[%s] Chamada bloqueada — provider indisponível.", provider.model_name)
        raise RuntimeError(f"Modelo '{provider.model_name}' está indisponível.")


class StatefulProvider:
    def __init__(self, strategy: AIProviderStrategy, model_name: str, max_failures: int = MAX_CONSECUTIVE_FAILURES):
        self.strategy = strategy
        self.model_name = model_name
        self.max_failures = max_failures
        self._state: ModelState = AvailableState()
        self._consecutive_failures = 0
        logger.debug("StatefulProvider criado: %s (max_failures=%d)", model_name, max_failures)

    def set_state(self, state: ModelState) -> None:
        old = type(self._state).__name__
        new = type(state).__name__
        logger.debug("[%s] Transição de estado: %s → %s", self.model_name, old, new)
        self._state = state

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def is_available(self) -> bool:
        return not isinstance(self._state, UnavailableState)

    def call(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        return self._state.call(self, prompt, max_tokens=max_tokens)


_PROVIDER_MAP = {
    "gpt-4o-mini":      lambda: StatefulProvider(OpenAIStrategy(OPENAI_MODEL_ID), "gpt-4o-mini"),
    "gemini-2.5-flash": lambda: StatefulProvider(GeminiStrategy(GEMINI_MODEL_ID), "gemini-2.5-flash"),
    "claude-haiku":     lambda: StatefulProvider(ClaudeStrategy(CLAUDE_MODEL_ID), "claude-haiku"),
}


def get_provider(model_name: str, _registry: dict = None) -> StatefulProvider:
    registry = _registry if _registry is not None else _PROVIDER_MAP
    if model_name not in registry:
        logger.error(
            "Modelo desconhecido solicitado: '%s'. Opções válidas: %s",
            model_name, list(registry),
        )
        raise ValueError(f"Modelo desconhecido: '{model_name}'. Opções: {list(registry)}")
    logger.debug("Criando provider: %s", model_name)
    return registry[model_name]()
