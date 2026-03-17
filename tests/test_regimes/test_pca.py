"""Tests for rolling PCA with anti-lookahead guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.core.exceptions import RegimeDetectionError
from meridian.regimes.pca import MarketPCA, RollingPCA


@pytest.fixture
def pca_feature_matrix():
    """500 rows x 20 features with 3 latent factors + noise."""
    np.random.seed(42)
    n_dates = 500
    n_features = 20
    dates = pd.bdate_range("2020-01-02", periods=n_dates)

    # 3 latent factors
    factors = np.random.randn(n_dates, 3)
    # Random loadings
    loadings = np.random.randn(3, n_features)
    # Features = factors @ loadings + noise
    features = factors @ loadings + np.random.randn(n_dates, n_features) * 0.5

    return pd.DataFrame(
        features,
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )


@pytest.fixture
def returns_matrix():
    """300 rows x 10 tickers for MarketPCA testing."""
    np.random.seed(42)
    n_dates = 400
    n_tickers = 10
    dates = pd.bdate_range("2020-01-02", periods=n_dates)

    returns = np.random.randn(n_dates, n_tickers) * 0.02
    return pd.DataFrame(
        returns,
        index=dates,
        columns=[f"TICK_{i}" for i in range(n_tickers)],
    )


class TestRollingPCA:
    def test_correct_number_of_components(self, pca_feature_matrix):
        """RollingPCA(n_components=5) produces 5 columns."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result = pca.fit_transform(pca_feature_matrix)
        assert result["components"].shape[1] == 5
        assert list(result["components"].columns) == [
            "PC1", "PC2", "PC3", "PC4", "PC5"
        ]

    def test_explained_variance_bounded(self, pca_feature_matrix):
        """Each variance ratio in [0,1], sum <= 1.0."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result = pca.fit_transform(pca_feature_matrix)

        ev = result["explained_variance"]
        assert (ev >= 0).all().all()
        assert (ev <= 1).all().all()
        assert (result["total_variance_explained"] <= 1.0 + 1e-10).all()

    def test_rolling_window_anti_lookahead(self, pca_feature_matrix):
        """Truncation invariance: full vs truncated must match on overlap."""
        pca_full = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_full = pca_full.fit_transform(pca_feature_matrix)

        # Truncate to first 350 rows
        truncated = pca_feature_matrix.iloc[:350]
        pca_trunc = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_trunc = pca_trunc.fit_transform(truncated)

        # Overlapping dates must match exactly
        common_dates = result_full["components"].index.intersection(
            result_trunc["components"].index
        )
        assert len(common_dates) > 0

        full_vals = result_full["components"].loc[common_dates]
        trunc_vals = result_trunc["components"].loc[common_dates]
        np.testing.assert_array_almost_equal(
            full_vals.values, trunc_vals.values, decimal=10
        )

    def test_scaler_fit_within_window(self, pca_feature_matrix):
        """Scaler mean matches rolling window mean, NOT global mean."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        pca.fit_transform(pca_feature_matrix)

        clean = pca_feature_matrix.dropna(how="all").ffill().dropna()

        # Check the first refit
        refit_indices = pca.get_refit_indices()
        assert len(refit_indices) > 0

        first_refit = refit_indices[0]
        scaler = pca.get_fitted_scaler(first_refit)
        assert scaler is not None

        # Scaler mean should match the window mean
        window = clean.iloc[first_refit - 100 : first_refit + 1]
        expected_mean = window.values.mean(axis=0)
        np.testing.assert_array_almost_equal(
            scaler.mean_, expected_mean, decimal=10
        )

        # Should NOT match global mean
        global_mean = clean.values.mean(axis=0)
        assert not np.allclose(scaler.mean_, global_mean, atol=1e-6)

    def test_refit_frequency(self, pca_feature_matrix):
        """Loadings change every refit_frequency days."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        pca.fit_transform(pca_feature_matrix)

        refit_indices = pca.get_refit_indices()
        # Check spacing between refits
        for i in range(1, len(refit_indices)):
            gap = refit_indices[i] - refit_indices[i - 1]
            assert gap == 21

    def test_between_refits_same_model(self, pca_feature_matrix):
        """Between refits, the same scaler/PCA model is used."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        pca.fit_transform(pca_feature_matrix)

        refit_indices = pca.get_refit_indices()
        # Only refit indices should have entries
        for idx in refit_indices:
            assert pca.get_fitted_scaler(idx) is not None

        # Non-refit indices should not have entries
        all_indices = set(range(100, len(pca_feature_matrix)))
        non_refit = all_indices - set(refit_indices)
        for idx in list(non_refit)[:10]:
            assert pca.get_fitted_scaler(idx) is None

    def test_transform_single_matches_batch(self, pca_feature_matrix):
        """Single-date transform matches batch row."""
        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result = pca.fit_transform(pca_feature_matrix)

        components = result["components"]
        last_date = components.index[-1]
        last_features = pca_feature_matrix.loc[last_date]

        single = pca.transform_single(
            last_features, last_date.date()
        )

        np.testing.assert_array_almost_equal(
            single[:5], components.iloc[-1].values, decimal=10
        )

    def test_synthetic_known_factors(self):
        """3 known factors should be recovered with high explained variance."""
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)

        # 3 orthogonal factors with large variance
        f1 = np.sin(np.linspace(0, 10, n)) * 5
        f2 = np.cos(np.linspace(0, 8, n)) * 4
        f3 = np.linspace(-3, 3, n)

        factors = np.column_stack([f1, f2, f3])
        loadings = np.random.randn(3, 15)
        features = factors @ loadings + np.random.randn(n, 15) * 0.1

        fm = pd.DataFrame(
            features,
            index=dates,
            columns=[f"f_{i}" for i in range(15)],
        )

        pca = RollingPCA(n_components=3, window_days=100, refit_frequency=21)
        result = pca.fit_transform(fm)

        # 3 components should capture >90% of variance
        assert result["total_variance_explained"].mean() > 0.90

    def test_min_observations_guard(self):
        """Raises RegimeDetectionError with too few rows."""
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", periods=30)
        features = pd.DataFrame(
            np.random.randn(30, 10),
            index=dates,
            columns=[f"f_{i}" for i in range(10)],
        )

        pca = RollingPCA(n_components=5, window_days=252)
        with pytest.raises(RegimeDetectionError):
            pca.fit_transform(features)

    def test_handles_nan_features(self, pca_feature_matrix):
        """NaN rows at the start are handled gracefully."""
        # Insert NaNs in first 10 rows
        modified = pca_feature_matrix.copy()
        modified.iloc[:10] = np.nan

        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result = pca.fit_transform(modified)

        # Should still produce valid output
        assert len(result["components"]) > 0
        assert not result["components"].isna().any().any()


class TestMarketPCA:
    def test_market_pca_shape(self, returns_matrix):
        """MarketPCA produces correct (dates x n_factors) output."""
        mpca = MarketPCA(n_factors=5, window_days=100, refit_frequency=21)
        result = mpca.fit_transform(returns_matrix)

        assert result["factors"].shape[1] == 5
        assert len(result["factors"]) > 0

    def test_market_pca_anti_lookahead(self, returns_matrix):
        """Truncation invariance for MarketPCA."""
        mpca_full = MarketPCA(n_factors=3, window_days=100, refit_frequency=21)
        result_full = mpca_full.fit_transform(returns_matrix)

        truncated = returns_matrix.iloc[:300]
        mpca_trunc = MarketPCA(n_factors=3, window_days=100, refit_frequency=21)
        result_trunc = mpca_trunc.fit_transform(truncated)

        common = result_full["factors"].index.intersection(
            result_trunc["factors"].index
        )
        assert len(common) > 0

        np.testing.assert_array_almost_equal(
            result_full["factors"].loc[common].values,
            result_trunc["factors"].loc[common].values,
            decimal=10,
        )
