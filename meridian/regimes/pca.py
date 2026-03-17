"""Rolling PCA with strict anti-lookahead guarantees.

At each date t, StandardScaler and PCA are fit on features from
[t - window_days, t]. This ensures no future information leaks
into historical regime labels.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from meridian.config.constants import (
    MIN_PCA_OBSERVATIONS,
    PCA_DEFAULT_N_COMPONENTS,
    PCA_DEFAULT_REFIT_FREQUENCY,
    PCA_DEFAULT_WINDOW_DAYS,
)
from meridian.core.exceptions import RegimeDetectionError

logger = logging.getLogger(__name__)


class RollingPCA:
    """Rolling PCA with strict anti-lookahead guarantees.

    At each date t, StandardScaler and PCA are fit on features
    from [t - window_days, t]. This ensures no lookahead.
    Models refit every refit_frequency days; between refits,
    the existing model transforms new data.
    """

    def __init__(
        self,
        n_components: int = PCA_DEFAULT_N_COMPONENTS,
        window_days: int = PCA_DEFAULT_WINDOW_DAYS,
        refit_frequency: int = PCA_DEFAULT_REFIT_FREQUENCY,
    ) -> None:
        self.n_components = n_components
        self.window_days = window_days
        self.refit_frequency = refit_frequency
        self._fitted_models: dict[int, tuple[StandardScaler, PCA]] = {}
        self._last_refit_idx: int | None = None

    def fit_transform(self, feature_matrix: pd.DataFrame) -> dict:
        """Run rolling PCA over the feature matrix.

        Args:
            feature_matrix: DatetimeIndex, columns = feature names.
                Single-ticker or pre-aggregated features.

        Returns:
            dict with keys:
                'components': DataFrame (n_dates x n_components)
                'explained_variance': DataFrame per component per date
                'loadings': dict[date, ndarray] at each refit date
                'total_variance_explained': Series per date
        """
        # Drop fully-NaN rows (feature warmup period)
        clean = feature_matrix.dropna(how="all")
        # Forward-fill remaining NaNs within individual features
        clean = clean.ffill()
        # Drop any rows still containing NaN (start of series)
        clean = clean.dropna()

        if len(clean) < self.window_days:
            raise RegimeDetectionError(
                f"Need at least {self.window_days} valid rows, "
                f"got {len(clean)}",
                n_rows=len(clean),
                window_days=self.window_days,
            )

        n_dates = len(clean)
        component_names = [f"PC{i+1}" for i in range(self.n_components)]

        # Output arrays
        components_data = np.full((n_dates, self.n_components), np.nan)
        variance_data = np.full((n_dates, self.n_components), np.nan)
        total_var = np.full(n_dates, np.nan)
        loadings: dict[date, np.ndarray] = {}

        self._fitted_models.clear()
        self._last_refit_idx = None
        current_scaler: StandardScaler | None = None
        current_pca: PCA | None = None

        for i in range(self.window_days, n_dates):
            # Determine if we need to refit
            needs_refit = (
                self._last_refit_idx is None
                or (i - self._last_refit_idx) >= self.refit_frequency
            )

            if needs_refit:
                window_start = i - self.window_days
                window = clean.iloc[window_start : i + 1]

                # Validate minimum observations
                valid_rows = window.dropna().shape[0]
                if valid_rows < MIN_PCA_OBSERVATIONS:
                    raise RegimeDetectionError(
                        f"Window has {valid_rows} valid rows, "
                        f"need {MIN_PCA_OBSERVATIONS}",
                        window_start=window_start,
                        window_end=i,
                    )

                # Fit scaler and PCA on this window only
                current_scaler = StandardScaler()
                scaled = current_scaler.fit_transform(window.values)

                n_comp = min(self.n_components, scaled.shape[1], scaled.shape[0])
                current_pca = PCA(n_components=n_comp)
                current_pca.fit(scaled)

                self._fitted_models[i] = (current_scaler, current_pca)
                self._last_refit_idx = i

                # Store loadings at refit dates
                dt = clean.index[i]
                if hasattr(dt, "date"):
                    dt = dt.date()
                loadings[dt] = current_pca.components_.copy()

            # Transform current row using current model
            if current_scaler is not None and current_pca is not None:
                row = clean.iloc[i : i + 1].values
                scaled_row = current_scaler.transform(row)
                transformed = current_pca.transform(scaled_row)

                n_comp = current_pca.n_components_
                components_data[i, :n_comp] = transformed[0]
                variance_data[i, :n_comp] = (
                    current_pca.explained_variance_ratio_
                )
                total_var[i] = current_pca.explained_variance_ratio_.sum()

        # Build output DataFrames — only non-NaN rows
        valid_mask = ~np.isnan(components_data[:, 0])
        valid_dates = clean.index[valid_mask]

        components_df = pd.DataFrame(
            components_data[valid_mask],
            index=valid_dates,
            columns=component_names,
        )
        variance_df = pd.DataFrame(
            variance_data[valid_mask],
            index=valid_dates,
            columns=component_names,
        )
        total_var_series = pd.Series(
            total_var[valid_mask], index=valid_dates, name="total_variance"
        )

        return {
            "components": components_df,
            "explained_variance": variance_df,
            "loadings": loadings,
            "total_variance_explained": total_var_series,
        }

    def transform_single(
        self, features_at_t: pd.Series, as_of_date: date
    ) -> np.ndarray:
        """For live trading. Uses most recently fitted model.

        Args:
            features_at_t: Feature values for a single date.
            as_of_date: The date for which to transform (unused,
                kept for API consistency and logging).

        Returns:
            1D array of PCA component values.
        """
        if not self._fitted_models:
            raise RegimeDetectionError(
                "No fitted models available. Call fit_transform first.",
                as_of_date=str(as_of_date),
            )

        # Use the most recently fitted model
        last_key = max(self._fitted_models.keys())
        scaler, pca = self._fitted_models[last_key]

        row = features_at_t.values.reshape(1, -1)
        scaled = scaler.transform(row)
        return pca.transform(scaled)[0]

    def get_fitted_scaler(self, refit_idx: int) -> StandardScaler | None:
        """Access a fitted scaler for testing anti-lookahead guarantees."""
        if refit_idx in self._fitted_models:
            return self._fitted_models[refit_idx][0]
        return None

    def get_refit_indices(self) -> list[int]:
        """Return all indices at which a refit occurred."""
        return sorted(self._fitted_models.keys())


class MarketPCA:
    """Cross-sectional PCA: tickers as columns, dates as rows.

    Extracts market-wide factors (analogous to Barra/APT
    statistical factors) without pre-specifying what they are.
    """

    def __init__(
        self,
        n_factors: int = 5,
        window_days: int = PCA_DEFAULT_WINDOW_DAYS,
        refit_frequency: int = PCA_DEFAULT_REFIT_FREQUENCY,
    ) -> None:
        self.n_factors = n_factors
        self.window_days = window_days
        self.refit_frequency = refit_frequency
        self._fitted_models: dict[int, tuple[StandardScaler, PCA]] = {}
        self._last_refit_idx: int | None = None

    def fit_transform(self, returns_matrix: pd.DataFrame) -> dict:
        """Run cross-sectional rolling PCA on returns.

        Args:
            returns_matrix: dates as index, tickers as columns,
                returns as values.

        Returns:
            dict with keys:
                'factors': DataFrame (n_dates x n_factors)
                'factor_exposures': dict[date, DataFrame] per refit
                'explained_variance': per factor per date
        """
        clean = returns_matrix.dropna(how="all")
        clean = clean.ffill()
        clean = clean.dropna()

        if len(clean) < self.window_days:
            raise RegimeDetectionError(
                f"Need at least {self.window_days} valid rows, "
                f"got {len(clean)}",
                n_rows=len(clean),
                window_days=self.window_days,
            )

        n_dates = len(clean)
        factor_names = [f"Factor{i+1}" for i in range(self.n_factors)]

        factors_data = np.full((n_dates, self.n_factors), np.nan)
        variance_data = np.full((n_dates, self.n_factors), np.nan)
        factor_exposures: dict[date, pd.DataFrame] = {}

        self._fitted_models.clear()
        self._last_refit_idx = None
        current_scaler: StandardScaler | None = None
        current_pca: PCA | None = None

        for i in range(self.window_days, n_dates):
            needs_refit = (
                self._last_refit_idx is None
                or (i - self._last_refit_idx) >= self.refit_frequency
            )

            if needs_refit:
                window_start = i - self.window_days
                window = clean.iloc[window_start : i + 1]

                valid_rows = window.dropna().shape[0]
                if valid_rows < MIN_PCA_OBSERVATIONS:
                    raise RegimeDetectionError(
                        f"Window has {valid_rows} valid rows, "
                        f"need {MIN_PCA_OBSERVATIONS}",
                        window_start=window_start,
                        window_end=i,
                    )

                current_scaler = StandardScaler()
                scaled = current_scaler.fit_transform(window.values)

                n_comp = min(
                    self.n_factors, scaled.shape[1], scaled.shape[0]
                )
                current_pca = PCA(n_components=n_comp)
                current_pca.fit(scaled)

                self._fitted_models[i] = (current_scaler, current_pca)
                self._last_refit_idx = i

                # Store factor exposures (loadings) at refit dates
                dt = clean.index[i]
                if hasattr(dt, "date"):
                    dt = dt.date()
                factor_exposures[dt] = pd.DataFrame(
                    current_pca.components_[:n_comp],
                    index=factor_names[:n_comp],
                    columns=clean.columns,
                )

            if current_scaler is not None and current_pca is not None:
                row = clean.iloc[i : i + 1].values
                scaled_row = current_scaler.transform(row)
                transformed = current_pca.transform(scaled_row)

                n_comp = current_pca.n_components_
                factors_data[i, :n_comp] = transformed[0]
                variance_data[i, :n_comp] = (
                    current_pca.explained_variance_ratio_
                )

        valid_mask = ~np.isnan(factors_data[:, 0])
        valid_dates = clean.index[valid_mask]

        factors_df = pd.DataFrame(
            factors_data[valid_mask],
            index=valid_dates,
            columns=factor_names,
        )
        variance_df = pd.DataFrame(
            variance_data[valid_mask],
            index=valid_dates,
            columns=factor_names,
        )

        return {
            "factors": factors_df,
            "factor_exposures": factor_exposures,
            "explained_variance": variance_df,
        }
