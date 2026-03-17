"""Meridian exception hierarchy.

All exceptions carry a context dict for structured logging:
    try:
        ...
    except DataProviderError as e:
        logger.error("fetch failed", **e.context)
"""

from __future__ import annotations

from typing import Any


class MeridianError(Exception):
    """Base exception for all Meridian errors."""

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context: dict[str, Any] = context
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class DataProviderError(MeridianError):
    """API failures, rate limits, network errors."""


class DataValidationError(MeridianError):
    """Data quality check failures."""


class StorageError(MeridianError):
    """Database read/write failures."""


class ConfigurationError(MeridianError):
    """Invalid settings or missing environment variables."""


class IngestionError(MeridianError):
    """Orchestration-level failures during data ingestion."""


class FeatureComputationError(MeridianError):
    """Feature computation failures (bad input, numerical errors)."""


class RegimeDetectionError(MeridianError):
    """Regime detection failures (insufficient data, convergence)."""
