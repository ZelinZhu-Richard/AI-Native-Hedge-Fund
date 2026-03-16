"""Application settings via Pydantic BaseSettings.

All settings are configurable via environment variables with the MERIDIAN_ prefix
and nested delimiter __ (e.g., MERIDIAN_DATABASE__PATH).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MERIDIAN_DATABASE__")

    path: Path = Field(default=Path("data/meridian.duckdb"))
    read_only: bool = False


class DataProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MERIDIAN_DATA_PROVIDER__")

    alpha_vantage_api_key: str = ""
    yahoo_rate_limit_per_second: float = 2.0


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MERIDIAN_INGESTION__")

    default_batch_size: int = 50
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MERIDIAN_LOGGING__")

    level: str = "INFO"
    format: str = "pretty"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MERIDIAN_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    data_provider: DataProviderSettings = Field(default_factory=DataProviderSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton. Useful for tests."""
    global _settings
    _settings = None
