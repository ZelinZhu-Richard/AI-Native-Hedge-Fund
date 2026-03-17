"""Regime clustering via HMM and KMeans on PCA components."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

from meridian.config.constants import DEFAULT_N_REGIMES
from meridian.core.exceptions import RegimeDetectionError

logger = logging.getLogger(__name__)


def _relabel_for_consistency(
    labels: np.ndarray,
    data: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """Sort regime labels by mean of first component.

    Ensures regime 0 = lowest mean, regime N-1 = highest.
    Gives semantic stability across refits.
    """
    means = []
    for r in range(n_regimes):
        mask = labels == r
        if mask.any():
            means.append((r, data[mask, 0].mean()))
        else:
            means.append((r, float("inf")))

    # Sort by mean value
    sorted_regimes = [r for r, _ in sorted(means, key=lambda x: x[1])]

    # Create mapping from old label to new label
    mapping = {old: new for new, old in enumerate(sorted_regimes)}
    return np.array([mapping[label] for label in labels])


class HMMRegimeDetector:
    """Gaussian HMM for regime detection on PCA components.

    HMM is natural for markets: hidden states (regimes) generate
    observable patterns. Learns unsupervised — we don't pre-label.
    """

    def __init__(
        self,
        n_regimes: int = DEFAULT_N_REGIMES,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model: GaussianHMM | None = None
        self._label_mapping: dict[int, int] | None = None

    def fit(self, pca_components: pd.DataFrame) -> HMMRegimeDetector:
        """Fit HMM on PCA component time series.

        Args:
            pca_components: DataFrame with dates as index,
                PCA components as columns.

        Returns:
            self for method chaining.
        """
        data = pca_components.values

        if len(data) < self.n_regimes * 2:
            raise RegimeDetectionError(
                f"Need at least {self.n_regimes * 2} observations "
                f"to fit {self.n_regimes} regimes, got {len(data)}",
            )

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=200,
            random_state=self.random_state,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore", message=".*did not converge.*"
            )
            try:
                model.fit(data)
            except Exception as e:
                raise RegimeDetectionError(
                    f"HMM fitting failed: {e}",
                ) from e

        self._model = model

        # Build relabeling mapping
        raw_labels = model.predict(data)
        relabeled = _relabel_for_consistency(
            raw_labels, data, self.n_regimes
        )
        # Store the mapping
        self._label_mapping = {}
        for old, new in zip(raw_labels, relabeled, strict=False):
            self._label_mapping[int(old)] = int(new)

        return self

    def _relabel(self, raw_labels: np.ndarray) -> np.ndarray:
        """Apply stored label mapping."""
        if self._label_mapping is None:
            return raw_labels
        return np.array(
            [self._label_mapping.get(lbl, lbl) for lbl in raw_labels]
        )

    def predict(self, pca_components: pd.DataFrame) -> pd.Series:
        """Predict regime labels for each date.

        Args:
            pca_components: DataFrame with PCA components.

        Returns:
            Series with regime labels (0..n_regimes-1), indexed by date.
        """
        if self._model is None:
            raise RegimeDetectionError("HMM not fitted. Call fit() first.")

        raw = self._model.predict(pca_components.values)
        labels = self._relabel(raw)
        return pd.Series(labels, index=pca_components.index, name="regime")

    def predict_proba(self, pca_components: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities for each date.

        Columns: regime_0..regime_{n-1}, rows sum to 1.0.
        This is what the gating network consumes.
        """
        if self._model is None:
            raise RegimeDetectionError("HMM not fitted. Call fit() first.")

        raw_proba = self._model.predict_proba(pca_components.values)

        # Reorder columns to match relabeled regime ordering
        if self._label_mapping is not None:
            reordered = np.zeros_like(raw_proba)
            for old, new in self._label_mapping.items():
                reordered[:, new] = raw_proba[:, old]
            raw_proba = reordered

        columns = [f"regime_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(
            raw_proba, index=pca_components.index, columns=columns
        )

    def get_transition_matrix(self) -> np.ndarray:
        """Return the HMM transition probability matrix.

        Entry [i, j] = P(regime_j at t+1 | regime_i at t).
        """
        if self._model is None:
            raise RegimeDetectionError("HMM not fitted. Call fit() first.")

        transmat = self._model.transmat_.copy()

        # Reorder rows and columns to match relabeled regime ordering
        if self._label_mapping is not None:
            n = self.n_regimes
            reordered = np.zeros((n, n))
            for old_i, new_i in self._label_mapping.items():
                for old_j, new_j in self._label_mapping.items():
                    reordered[new_i, new_j] = transmat[old_i, old_j]
            return reordered

        return transmat


class KMeansRegimeDetector:
    """KMeans clustering on PCA components.

    Simpler than HMM (no temporal dynamics), but more robust.
    Sanity check against HMM and stable fallback.
    """

    def __init__(
        self,
        n_regimes: int = DEFAULT_N_REGIMES,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._model: KMeans | None = None
        self._label_mapping: dict[int, int] | None = None

    def fit(
        self, pca_components: pd.DataFrame
    ) -> KMeansRegimeDetector:
        """Fit KMeans on PCA component data.

        Args:
            pca_components: DataFrame with dates as index,
                PCA components as columns.

        Returns:
            self for method chaining.
        """
        data = pca_components.values

        if len(data) < self.n_regimes:
            raise RegimeDetectionError(
                f"Need at least {self.n_regimes} observations "
                f"to fit {self.n_regimes} clusters, got {len(data)}",
            )

        model = KMeans(
            n_clusters=self.n_regimes,
            n_init=10,
            random_state=self.random_state,
        )
        model.fit(data)
        self._model = model

        # Build relabeling mapping
        raw_labels = model.predict(data)
        relabeled = _relabel_for_consistency(
            raw_labels, data, self.n_regimes
        )
        self._label_mapping = {}
        for old, new in zip(raw_labels, relabeled, strict=False):
            self._label_mapping[int(old)] = int(new)

        return self

    def _relabel(self, raw_labels: np.ndarray) -> np.ndarray:
        """Apply stored label mapping."""
        if self._label_mapping is None:
            return raw_labels
        return np.array(
            [self._label_mapping.get(lbl, lbl) for lbl in raw_labels]
        )

    def predict(self, pca_components: pd.DataFrame) -> pd.Series:
        """Predict regime labels for each date.

        Args:
            pca_components: DataFrame with PCA components.

        Returns:
            Series with regime labels (0..n_regimes-1), indexed by date.
        """
        if self._model is None:
            raise RegimeDetectionError(
                "KMeans not fitted. Call fit() first."
            )

        raw = self._model.predict(pca_components.values)
        labels = self._relabel(raw)
        return pd.Series(labels, index=pca_components.index, name="regime")

    def predict_distance(
        self, pca_components: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute distance from each point to each centroid.

        Columns: distance_0..distance_{n-1}. Non-negative.
        Far from ALL centroids = potentially new regime.
        """
        if self._model is None:
            raise RegimeDetectionError(
                "KMeans not fitted. Call fit() first."
            )

        distances = self._model.transform(pca_components.values)

        # Reorder columns to match relabeled regime ordering
        if self._label_mapping is not None:
            reordered = np.zeros_like(distances)
            for old, new in self._label_mapping.items():
                reordered[:, new] = distances[:, old]
            distances = reordered

        columns = [f"distance_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(
            distances, index=pca_components.index, columns=columns
        )
