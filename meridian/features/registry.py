"""Feature registry for tracking all registered features.

Singleton pattern ensures a single source of truth for feature metadata.
Used for cache invalidation, dependency resolution, and documentation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar

from meridian.config.constants import DEFAULT_FEATURE_VERSION


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable descriptor for a single feature."""

    name: str
    category: str
    lookback_days: int
    version: int = DEFAULT_FEATURE_VERSION
    description: str = ""
    dependencies: tuple[str, ...] = ()


class FeatureRegistry:
    """Singleton tracking all registered features."""

    _instance: ClassVar[FeatureRegistry | None] = None

    def __init__(self) -> None:
        self._features: dict[str, FeatureConfig] = {}

    @classmethod
    def instance(cls) -> FeatureRegistry:
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. For tests."""
        cls._instance = None

    def register(self, config: FeatureConfig) -> None:
        """Register a feature config. Overwrites if same name exists."""
        self._features[config.name] = config

    def get(self, name: str) -> FeatureConfig:
        """Get a feature config by name."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not registered")
        return self._features[name]

    def list_features(self, category: str | None = None) -> list[FeatureConfig]:
        """List all features, optionally filtered by category."""
        features = list(self._features.values())
        if category is not None:
            features = [f for f in features if f.category == category]
        return sorted(features, key=lambda f: (f.category, f.name))

    def max_lookback(self) -> int:
        """Max lookback across all registered features."""
        if not self._features:
            return 0
        return max(f.lookback_days for f in self._features.values())

    def version_hash(self) -> str:
        """Hash of all configs for cache invalidation."""
        parts = []
        for config in sorted(self._features.values(), key=lambda f: f.name):
            parts.append(f"{config.name}:{config.version}:{config.lookback_days}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
