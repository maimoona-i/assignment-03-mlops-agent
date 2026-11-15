"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "mlops_agent"

_SENSITIVE_KEY = re.compile(r"(token|password|passwd|secret|api[_-]?key|authorization|credential|kubeconfig)", re.I)
_SECRET_VALUE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+)"
)
REDACTED = "***REDACTED***"


def redact_text(text: str) -> str:
    return _SECRET_VALUE.sub(REDACTED, text)


def redact(obj: Any) -> Any:
    """Recursively redact sensitive keys and secret-looking values."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _SENSITIVE_KEY.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if os.environ.get("LOG_FILE"):
        handlers.append(logging.FileHandler(os.environ["LOG_FILE"]))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger._configured = True  # type: ignore[attr-defined]
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured log line. Fields are always redacted."""
    get_logger().info(event, extra={"fields": redact(fields)})
