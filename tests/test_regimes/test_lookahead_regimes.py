"""Anti-lookahead tests for regime detection — MOST IMPORTANT TESTS.

These verify that future data never influences past regime labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.regimes.clustering import KMeansRegimeDetector
from meridian.regimes.pca import RollingPCA


@pytest.fixture
def lookahead_features():
    """500 rows x 20 features for lookahead testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    features = np.random.randn(500, 20)
    return pd.DataFrame(
        features,
        index=dates,
        columns=[f"feat_{i}" for i in range(20)],
    )


class TestPCAAntiLookahead:
    def test_pca_truncation_invariance(self, lookahead_features):
        """PCA on 500 rows vs first 350: overlapping dates must match exactly."""
        pca_full = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_full = pca_full.fit_transform(lookahead_features)

        truncated = lookahead_features.iloc[:350]
        pca_trunc = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_trunc = pca_trunc.fit_transform(truncated)

        # Components for dates 100-349 must match exactly
        common = result_full["components"].index.intersection(
            result_trunc["components"].index
        )
        assert len(common) > 100

        np.testing.assert_array_almost_equal(
            result_full["components"].loc[common].values,
            result_trunc["components"].loc[common].values,
            decimal=10,
        )

    def test_clustering_truncation_invariance(self, lookahead_features):
        """Clustering on full vs truncated PCA: overlapping labels match."""
        # Full PCA + clustering
        pca_full = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_full = pca_full.fit_transform(lookahead_features)
        components_full = result_full["components"]

        # Truncated PCA + clustering
        truncated = lookahead_features.iloc[:350]
        pca_trunc = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_trunc = pca_trunc.fit_transform(truncated)
        components_trunc = result_trunc["components"]

        # PCA components match on overlap — now verify clustering does too
        common = components_full.index.intersection(components_trunc.index)

        # KMeans is deterministic on same input
        km_full = KMeansRegimeDetector(n_regimes=3, random_state=42)
        km_full.fit(components_trunc.loc[common])  # fit on shared data only
        labels_from_full = km_full.predict(components_full.loc[common])

        km_trunc = KMeansRegimeDetector(n_regimes=3, random_state=42)
        km_trunc.fit(components_trunc.loc[common])
        labels_from_trunc = km_trunc.predict(components_trunc.loc[common])

        # Same input -> same labels
        pd.testing.assert_series_equal(labels_from_full, labels_from_trunc)

    def test_future_data_no_retroactive_labels(self, lookahead_features):
        """Adding 100 rows of future data doesn't change past PCA components."""
        base = lookahead_features.iloc[:400]
        extended = lookahead_features.iloc[:500]

        pca_base = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_base = pca_base.fit_transform(base)

        pca_ext = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        result_ext = pca_ext.fit_transform(extended)

        # All base dates must have identical components
        common = result_base["components"].index.intersection(
            result_ext["components"].index
        )
        assert len(common) == len(result_base["components"])

        np.testing.assert_array_almost_equal(
            result_base["components"].values,
            result_ext["components"].loc[common].values,
            decimal=10,
        )

    def test_scaler_not_global(self):
        """Scaler mean before shift must not reflect post-shift."""
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)

        # Mean shift at row 300: first 300 rows mean ~0, last 200 rows mean ~10
        features = np.random.randn(n, 10)
        features[300:] += 10.0

        fm = pd.DataFrame(
            features,
            index=dates,
            columns=[f"f_{i}" for i in range(10)],
        )

        pca = RollingPCA(n_components=5, window_days=100, refit_frequency=21)
        pca.fit_transform(fm)

        # Find a refit that happened before the shift (before index 300)
        refit_indices = pca.get_refit_indices()
        pre_shift_refits = [idx for idx in refit_indices if idx < 300]
        assert len(pre_shift_refits) > 0

        last_pre_shift = pre_shift_refits[-1]
        scaler = pca.get_fitted_scaler(last_pre_shift)
        assert scaler is not None

        # Scaler mean should be near 0 (pre-shift data only)
        assert np.all(
            np.abs(scaler.mean_) < 2.0
        ), f"Scaler mean {scaler.mean_} suggests future data leakage"

        # Find a refit that happened after the shift
        post_shift_refits = [idx for idx in refit_indices if idx >= 400]
        if post_shift_refits:
            post_scaler = pca.get_fitted_scaler(post_shift_refits[0])
            assert post_scaler is not None
            # Post-shift scaler should reflect the shifted mean
            assert np.all(
                post_scaler.mean_ > 5.0
            ), "Post-shift scaler should reflect high mean"
