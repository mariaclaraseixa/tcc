import logging
import logging.handlers
import os
from datetime import datetime

from config import LOG_DIR

_FMT = "%(asctime)s [%(levelname)-8s] %(name)-22s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Bibliotecas de terceiros que poluem o console com INFO de HTTP
_NOISY_LIBS = {"httpx", "httpcore", "openai", "google_genai",
               "huggingface_hub", "hf_hub", "anthropic"}


class _ConsoleFilter(logging.Filter):
    """Bloqueia logs INFO/DEBUG de libs externas no console."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.split(".")[0] not in _NOISY_LIBS


def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> str:
    root = logging.getLogger()
    if root.handlers:
        return ""

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"experiment_{timestamp}.log")

    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    console.addFilter(_ConsoleFilter())
    root.addHandler(console)

    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger(__name__).info("Logging iniciado. Arquivo de log: %s", log_path)
    return log_path
