"""Tests for HMM and KMeans regime clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.regimes.clustering import HMMRegimeDetector, KMeansRegimeDetector


@pytest.fixture
def pca_components():
    """400 rows x 3 PCA components for clustering tests."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=400)
    data = np.random.randn(400, 3)
    return pd.DataFrame(data, index=dates, columns=["PC1", "PC2", "PC3"])


@pytest.fixture
def separable_regimes():
    """400 rows x 3 dims with 2 clearly separable clusters."""
    np.random.seed(42)
    n_per_cluster = 200
    dates = pd.bdate_range("2020-01-02", periods=n_per_cluster * 2)

    # Cluster 0: centered at (-5, -5, -5)
    c0 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([-5, -5, -5])
    # Cluster 1: centered at (5, 5, 5)
    c1 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([5, 5, 5])

    data = np.vstack([c0, c1])
    true_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    df = pd.DataFrame(data, index=dates, columns=["PC1", "PC2", "PC3"])
    return df, true_labels


class TestHMMRegimeDetector:
    def test_correct_regime_count(self, pca_components):
        """Labels in {0, ..., n_regimes-1}."""
        hmm = HMMRegimeDetector(n_regimes=4, random_state=42)
        hmm.fit(pca_components)
        labels = hmm.predict(pca_components)

        unique = set(labels.unique())
        assert unique.issubset({0, 1, 2, 3})

    def test_predict_proba_sums_to_one(self, pca_components):
        """Each row sums to 1.0 (atol=1e-6)."""
        hmm = HMMRegimeDetector(n_regimes=4, random_state=42)
        hmm.fit(pca_components)
        proba = hmm.predict_proba(pca_components)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_transition_matrix_valid(self, pca_components):
        """Rows sum to 1.0, all values in [0,1]."""
        hmm = HMMRegimeDetector(n_regimes=4, random_state=42)
        hmm.fit(pca_components)
        transmat = hmm.get_transition_matrix()

        assert transmat.shape == (4, 4)
        assert (transmat >= 0).all()
        assert (transmat <= 1).all()
        np.testing.assert_allclose(
            transmat.sum(axis=1), 1.0, atol=1e-6
        )

    def test_detects_synthetic_regimes(self, separable_regimes):
        """On separable data, >90% accuracy (allowing label permutation)."""
        data, true_labels = separable_regimes
        hmm = HMMRegimeDetector(n_regimes=2, random_state=42)
        hmm.fit(data)
        pred = hmm.predict(data).values

        # Check accuracy with both possible label assignments
        acc_direct = (pred == true_labels).mean()
        acc_flipped = (pred == (1 - true_labels)).mean()
        best_acc = max(acc_direct, acc_flipped)

        assert best_acc > 0.90, f"Best accuracy was {best_acc:.2%}"

    def test_deterministic_with_seed(self, pca_components):
        """Same seed = same labels."""
        hmm1 = HMMRegimeDetector(n_regimes=3, random_state=42)
        hmm1.fit(pca_components)
        labels1 = hmm1.predict(pca_components)

        hmm2 = HMMRegimeDetector(n_regimes=3, random_state=42)
        hmm2.fit(pca_components)
        labels2 = hmm2.predict(pca_components)

        pd.testing.assert_series_equal(labels1, labels2)


class TestKMeansRegimeDetector:
    def test_assigns_every_date(self, pca_components):
        """No missing labels."""
        km = KMeansRegimeDetector(n_regimes=4, random_state=42)
        km.fit(pca_components)
        labels = km.predict(pca_components)

        assert len(labels) == len(pca_components)
        assert not labels.isna().any()

    def test_distances_non_negative(self, pca_components):
        """All distances >= 0."""
        km = KMeansRegimeDetector(n_regimes=4, random_state=42)
        km.fit(pca_components)
        distances = km.predict_distance(pca_components)

        assert (distances >= 0).all().all()

    def test_detects_synthetic_regimes(self, separable_regimes):
        """>90% accuracy on separable data."""
        data, true_labels = separable_regimes
        km = KMeansRegimeDetector(n_regimes=2, random_state=42)
        km.fit(data)
        pred = km.predict(data).values

        acc_direct = (pred == true_labels).mean()
        acc_flipped = (pred == (1 - true_labels)).mean()
        best_acc = max(acc_direct, acc_flipped)

        assert best_acc > 0.90, f"Best accuracy was {best_acc:.2%}"

    def test_regime_label_range(self, pca_components):
        """Labels in [0, n_regimes-1]."""
        km = KMeansRegimeDetector(n_regimes=4, random_state=42)
        km.fit(pca_components)
        labels = km.predict(pca_components)

        assert labels.min() >= 0
        assert labels.max() <= 3

    def test_deterministic_with_seed(self, pca_components):
        """Same seed = same labels."""
        km1 = KMeansRegimeDetector(n_regimes=3, random_state=42)
        km1.fit(pca_components)
        labels1 = km1.predict(pca_components)

        km2 = KMeansRegimeDetector(n_regimes=3, random_state=42)
        km2.fit(pca_components)
        labels2 = km2.predict(pca_components)

        pd.testing.assert_series_equal(labels1, labels2)
