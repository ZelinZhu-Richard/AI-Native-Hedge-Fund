"""Structured logging configuration using structlog.

Usage:
    from meridian.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("fetching data", ticker="AAPL", bars=252)
"""

from __future__ import annotations

import os

import structlog

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    log_format = os.environ.get("MERIDIAN_LOGGING__FORMAT", "pretty").lower()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to the given name."""
    _configure()
    return structlog.get_logger(name)  # type: ignore[return-value]


def reset_logging() -> None:
    """Reset logging configuration. Useful for tests."""
    global _configured
    _configured = False
    structlog.reset_defaults()
