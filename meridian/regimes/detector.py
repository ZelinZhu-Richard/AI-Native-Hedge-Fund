"""Unified regime detection pipeline: features -> PCA -> clustering -> RegimeResult."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel

from meridian.core.exceptions import RegimeDetectionError
from meridian.regimes.analysis import RegimeAnalyzer
from meridian.regimes.clustering import HMMRegimeDetector, KMeansRegimeDetector
from meridian.regimes.pca import RollingPCA


class RegimeResult(BaseModel):
    """Immutable result of regime detection. JSON-serializable."""

    labels: dict[str, int]  # ISO date str -> regime label
    probabilities: dict[str, dict[str, float]]  # date -> {regime_0: prob}
    current_regime: int
    current_confidence: float  # max prob of current regime
    transition_probability: float  # prob of leaving current regime
    regime_stats: dict[str, Any]  # per-regime summary
    transition_dates: list[str]  # ISO dates where regime changed


class RegimeDetector:
    """Full pipeline: features -> PCA -> clustering -> RegimeResult.

    Orchestrates rolling PCA and clustering with synchronized
    refit cadence. This is the interface that the gating network
    (Day 11) and risk management (Sprint 2) will import.
    """

    def __init__(
        self,
        pca: RollingPCA,
        hmm: HMMRegimeDetector,
        kmeans: KMeansRegimeDetector,
        method: str = "hmm",
    ) -> None:
        if method not in ("hmm", "kmeans", "ensemble"):
            raise ValueError(f"method must be hmm, kmeans, or ensemble. Got: {method}")
        self.pca = pca
        self.hmm = hmm
        self.kmeans = kmeans
        self.method = method
        self._analyzer = RegimeAnalyzer()

    def detect(self, feature_matrix: pd.DataFrame) -> RegimeResult:
        """Full pipeline: features -> PCA -> clustering -> RegimeResult.

        1. pca.fit_transform(feature_matrix) -> components
        2. Fit clustering on components
        3. Build RegimeResult with labels, probs, transitions
        """
        # Step 1: PCA
        pca_result = self.pca.fit_transform(feature_matrix)
        components = pca_result["components"]

        if len(components) == 0:
            raise RegimeDetectionError("PCA produced no valid components.")

        # Step 2: Fit and predict with selected method
        if self.method == "hmm":
            labels, proba = self._detect_hmm(components)
        elif self.method == "kmeans":
            labels, proba = self._detect_kmeans(components)
        else:  # ensemble
            labels, proba = self._detect_ensemble(components)

        # Step 3: Build result
        return self._build_result(labels, proba, feature_matrix, components)

    def _detect_hmm(
        self, components: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        self.hmm.fit(components)
        labels = self.hmm.predict(components)
        proba = self.hmm.predict_proba(components)
        return labels, proba

    def _detect_kmeans(
        self, components: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        self.kmeans.fit(components)
        labels = self.kmeans.predict(components)

        # Convert distances to pseudo-probabilities via softmax
        distances = self.kmeans.predict_distance(components)
        # Inverse distance weighting: closer = higher probability
        inv_dist = 1.0 / (distances + 1e-10)
        proba_values = inv_dist.div(inv_dist.sum(axis=1), axis=0)
        proba = proba_values.rename(
            columns={
                f"distance_{i}": f"regime_{i}"
                for i in range(self.kmeans.n_regimes)
            }
        )
        return labels, proba

    def _detect_ensemble(
        self, components: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Average HMM and KMeans probabilities, pick argmax."""
        _, hmm_proba = self._detect_hmm(components)
        _, kmeans_proba = self._detect_kmeans(components)

        # Average probabilities
        avg_proba = (hmm_proba + kmeans_proba) / 2.0
        labels = avg_proba.idxmax(axis=1).str.replace("regime_", "").astype(int)
        labels.name = "regime"
        return labels, avg_proba

    def _build_result(
        self,
        labels: pd.Series,
        proba: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        components: pd.DataFrame,
    ) -> RegimeResult:
        """Assemble RegimeResult from raw outputs."""
        # Convert labels to dict with ISO date strings
        labels_dict: dict[str, int] = {}
        for dt, label in labels.items():
            key = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            labels_dict[key] = int(label)

        # Convert probabilities to nested dict
        proba_dict: dict[str, dict[str, float]] = {}
        for dt in proba.index:
            key = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            proba_dict[key] = {
                col: float(proba.loc[dt, col]) for col in proba.columns
            }

        # Current regime info
        current_regime = int(labels.iloc[-1])
        current_proba = proba.iloc[-1]
        current_confidence = float(current_proba.max())

        # Transition probability = 1 - P(staying in current regime)
        current_regime_col = f"regime_{current_regime}"
        if current_regime_col in current_proba.index:
            transition_probability = 1.0 - float(
                current_proba[current_regime_col]
            )
        else:
            transition_probability = 0.0

        # Find transition dates
        transition_dates: list[str] = []
        label_values = labels.values
        for i in range(1, len(label_values)):
            if label_values[i] != label_values[i - 1]:
                dt = labels.index[i]
                key = (
                    dt.strftime("%Y-%m-%d")
                    if hasattr(dt, "strftime")
                    else str(dt)
                )
                transition_dates.append(key)

        # Compute regime stats
        regime_stats = self._analyzer.characterize_regimes(
            regime_labels=labels,
            feature_matrix=feature_matrix.reindex(labels.index),
        )

        return RegimeResult(
            labels=labels_dict,
            probabilities=proba_dict,
            current_regime=current_regime,
            current_confidence=current_confidence,
            transition_probability=transition_probability,
            regime_stats=regime_stats,
            transition_dates=transition_dates,
        )

    def detect_live(
        self, features_today: pd.Series, as_of_date: date
    ) -> dict:
        """Single-day detection for paper/live trading.

        Uses pre-fitted PCA + clustering models.
        """
        # Transform today's features through PCA
        pca_values = self.pca.transform_single(features_today, as_of_date)

        # Create a single-row DataFrame for clustering
        component_names = [f"PC{i+1}" for i in range(len(pca_values))]
        components_df = pd.DataFrame(
            [pca_values],
            index=[pd.Timestamp(as_of_date)],
            columns=component_names,
        )

        if self.method == "hmm":
            labels = self.hmm.predict(components_df)
            proba = self.hmm.predict_proba(components_df)
        elif self.method == "kmeans":
            labels = self.kmeans.predict(components_df)
            distances = self.kmeans.predict_distance(components_df)
            inv_dist = 1.0 / (distances + 1e-10)
            proba = inv_dist.div(inv_dist.sum(axis=1), axis=0)
            proba = proba.rename(
                columns={
                    f"distance_{i}": f"regime_{i}"
                    for i in range(self.kmeans.n_regimes)
                }
            )
        else:
            raise RegimeDetectionError(
                "Ensemble method requires batch detection."
            )

        regime = int(labels.iloc[0])
        proba_row = proba.iloc[0]

        return {
            "regime": regime,
            "confidence": float(proba_row.max()),
            "probabilities": {
                col: float(proba_row[col]) for col in proba.columns
            },
            "pca_components": pca_values.tolist(),
        }
