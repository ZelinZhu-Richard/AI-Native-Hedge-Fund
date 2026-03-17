"""Tests for the unified RegimeDetector pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.regimes.clustering import HMMRegimeDetector, KMeansRegimeDetector
from meridian.regimes.detector import RegimeDetector, RegimeResult
from meridian.regimes.pca import RollingPCA


@pytest.fixture
def regime_feature_matrix():
    """500 rows x 20 features with embedded regime shift at row 250."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)

    # First half: low volatility regime
    f1 = np.random.randn(250, 20) * 0.5
    # Second half: high volatility regime with mean shift
    f2 = np.random.randn(250, 20) * 2.0 + 1.0

    features = np.vstack([f1, f2])
    return pd.DataFrame(
        features,
        index=dates,
        columns=[f"feat_{i}" for i in range(20)],
    )


@pytest.fixture
def detector(regime_feature_matrix):
    """Pre-configured RegimeDetector."""
    pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
    hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
    kmeans = KMeansRegimeDetector(n_regimes=3, random_state=42)
    return RegimeDetector(pca, hmm, kmeans, method="hmm")


class TestRegimeDetector:
    def test_full_pipeline_returns_regime_result(
        self, detector, regime_feature_matrix
    ):
        """detect() returns valid RegimeResult."""
        result = detector.detect(regime_feature_matrix)
        assert isinstance(result, RegimeResult)

    def test_detect_live_returns_valid_result(
        self, regime_feature_matrix
    ):
        """Live detection returns valid result with correct structure."""
        # Use kmeans for live — it's point-based (no sequence context)
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        kmeans = KMeansRegimeDetector(n_regimes=3, random_state=42)
        det = RegimeDetector(pca, hmm, kmeans, method="kmeans")

        det.detect(regime_feature_matrix)

        last_date = regime_feature_matrix.index[-1]
        last_features = regime_feature_matrix.iloc[-1]

        live = det.detect_live(last_features, last_date.date())
        assert "regime" in live
        assert "confidence" in live
        assert "probabilities" in live
        assert "pca_components" in live
        assert 0 <= live["regime"] <= 2
        assert 0.0 <= live["confidence"] <= 1.0

    def test_regime_result_serialization(
        self, detector, regime_feature_matrix
    ):
        """model_dump() + model_validate() roundtrip."""
        result = detector.detect(regime_feature_matrix)

        dumped = result.model_dump()
        restored = RegimeResult.model_validate(dumped)

        assert restored.current_regime == result.current_regime
        assert restored.labels == result.labels
        assert restored.transition_dates == result.transition_dates

    def test_method_switching(self, regime_feature_matrix):
        """Both hmm and kmeans methods produce valid results."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        kmeans = KMeansRegimeDetector(n_regimes=3, random_state=42)

        for method in ("hmm", "kmeans"):
            det = RegimeDetector(pca, hmm, kmeans, method=method)
            result = det.detect(regime_feature_matrix)
            assert isinstance(result, RegimeResult)
            assert len(result.labels) > 0

    def test_all_fields_populated(
        self, detector, regime_feature_matrix
    ):
        """No None values in RegimeResult."""
        result = detector.detect(regime_feature_matrix)

        assert result.labels is not None
        assert result.probabilities is not None
        assert result.current_regime is not None
        assert result.current_confidence is not None
        assert result.transition_probability is not None
        assert result.regime_stats is not None
        assert result.transition_dates is not None

        assert len(result.labels) > 0
        assert len(result.probabilities) > 0
        assert 0.0 <= result.current_confidence <= 1.0
        assert 0.0 <= result.transition_probability <= 1.0

    def test_transition_dates_are_actual_changes(
        self, detector, regime_feature_matrix
    ):
        """Each transition date has different label vs previous date."""
        result = detector.detect(regime_feature_matrix)

        if len(result.transition_dates) == 0:
            pytest.skip("No transitions detected")

        sorted_dates = sorted(result.labels.keys())
        for trans_date in result.transition_dates:
            idx = sorted_dates.index(trans_date)
            assert idx > 0, "Transition can't be on first date"
            prev_date = sorted_dates[idx - 1]
            assert (
                result.labels[trans_date] != result.labels[prev_date]
            ), f"No actual change at {trans_date}"
