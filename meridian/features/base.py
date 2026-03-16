"""Base classes for feature computers.

All feature computers inherit from FeatureComputer and register their
features with the singleton FeatureRegistry on instantiation.
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd

from meridian.core.logging import get_logger
from meridian.features.registry import FeatureConfig, FeatureRegistry


class FeatureComputer(abc.ABC):
    """Abstract base class for all feature computers."""

    def __init__(self) -> None:
        self.logger = get_logger(f"features.{self.category}")
        self._register_features()

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Feature category name (e.g. 'technical', 'volatility')."""

    @abc.abstractmethod
    def feature_configs(self) -> list[FeatureConfig]:
        """Return list of feature configs this computer produces."""

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute features from input DataFrame.

        Input: OHLCV DataFrame sorted by date.
        Output: DataFrame with same index, one column per feature.
        NaN where lookback is insufficient.
        """

    @property
    @abc.abstractmethod
    def required_columns(self) -> list[str]:
        """Columns required in the input DataFrame."""

    def _register_features(self) -> None:
        registry = FeatureRegistry.instance()
        for config in self.feature_configs():
            registry.register(config)


class SingleTickerFeature(FeatureComputer):
    """Feature computer that operates on a single ticker's data."""


class CrossSectionalFeature(FeatureComputer):
    """Feature computer that operates across the full universe."""
