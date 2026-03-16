"""Feature store — DuckDB-backed storage for computed features.

Long format in storage (ticker, date, feature_name, value).
get_feature_matrix() pivots to wide format for model consumption.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd

from meridian.core.exceptions import StorageError
from meridian.core.logging import get_logger
from meridian.data.storage.database import MeridianDatabase
from meridian.features.registry import FeatureConfig

logger = get_logger("features.store")

_FEATURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    lookback_days INTEGER NOT NULL,
    description VARCHAR,
    last_computed TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_values (
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    feature_name VARCHAR NOT NULL,
    value DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, date, feature_name)
);
"""


class FeatureStore:
    """DuckDB-backed storage for computed features."""

    def __init__(self, db: MeridianDatabase) -> None:
        self.db = db

    def create_schema(self) -> None:
        """Create feature tables. Idempotent."""
        try:
            self.db.conn.execute(_FEATURE_SCHEMA_SQL)
            logger.info("feature schema created")
        except Exception as e:
            raise StorageError(
                "Failed to create feature schema", error=str(e)
            ) from e

    def store_features(self, ticker: str, features_df: pd.DataFrame) -> int:
        """Melt wide DataFrame to long format and store.

        Args:
            ticker: Ticker symbol.
            features_df: DataFrame with DatetimeIndex/date index and
                feature names as columns.

        Returns:
            Number of rows stored.
        """
        if features_df.empty:
            return 0

        try:
            # Prepare long-format DataFrame
            df = features_df.copy()
            df.index.name = "date"
            melted = df.reset_index().melt(
                id_vars=["date"],
                var_name="feature_name",
                value_name="value",
            )
            # Drop NaN values — no point storing them
            melted = melted.dropna(subset=["value"])
            if melted.empty:
                return 0

            now = datetime.datetime.now(datetime.UTC)

            # INSERT OR REPLACE via executemany for robustness
            params = [
                (ticker, row["date"], row["feature_name"], row["value"], now)
                for _, row in melted.iterrows()
            ]
            self.db.conn.executemany(
                """
                INSERT OR REPLACE INTO feature_values
                    (ticker, date, feature_name, value, computed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                params,
            )
            count = len(params)
            logger.info("features stored", ticker=ticker, rows=count)
            return count
        except Exception as e:
            raise StorageError(
                "Failed to store features",
                ticker=ticker,
                error=str(e),
            ) from e

    def store_feature_metadata(self, configs: list[FeatureConfig]) -> None:
        """Store or update feature metadata."""
        try:
            for config in configs:
                self.db.conn.execute(
                    """
                    INSERT OR REPLACE INTO feature_metadata
                        (feature_name, category, version, lookback_days,
                         description, last_computed)
                    VALUES (?, ?, ?, ?, ?, current_timestamp)
                    """,
                    [
                        config.name,
                        config.category,
                        config.version,
                        config.lookback_days,
                        config.description,
                    ],
                )
        except Exception as e:
            raise StorageError(
                "Failed to store feature metadata", error=str(e)
            ) from e

    def get_features(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> pd.DataFrame:
        """Get features for a single ticker in long format."""
        conditions = ["ticker = ?"]
        params: list[Any] = [ticker]

        if feature_names:
            placeholders = ", ".join(["?"] * len(feature_names))
            conditions.append(f"feature_name IN ({placeholders})")
            params.extend(feature_names)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        where = " AND ".join(conditions)
        return self.db.conn.execute(
            f"SELECT ticker, date, feature_name, value FROM feature_values "
            f"WHERE {where} ORDER BY date, feature_name",
            params,
        ).fetchdf()

    def get_feature_matrix(
        self,
        tickers: list[str],
        feature_names: list[str] | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> pd.DataFrame:
        """Wide format: MultiIndex (ticker, date), columns = features."""
        conditions = []
        params: list[Any] = []

        ticker_ph = ", ".join(["?"] * len(tickers))
        conditions.append(f"ticker IN ({ticker_ph})")
        params.extend(tickers)

        if feature_names:
            feat_ph = ", ".join(["?"] * len(feature_names))
            conditions.append(f"feature_name IN ({feat_ph})")
            params.extend(feature_names)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        where = " AND ".join(conditions)
        long_df = self.db.conn.execute(
            f"SELECT ticker, date, feature_name, value FROM feature_values "
            f"WHERE {where} ORDER BY ticker, date",
            params,
        ).fetchdf()

        if long_df.empty:
            return pd.DataFrame()

        # Pivot to wide format
        wide = long_df.pivot_table(
            index=["ticker", "date"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        )
        wide.columns.name = None
        return wide

    def get_latest_feature_date(self, ticker: str) -> datetime.date | None:
        """Get the most recent feature date for a ticker."""
        result = self.db.conn.execute(
            "SELECT MAX(date) FROM feature_values WHERE ticker = ?", [ticker]
        ).fetchone()
        if result and result[0] is not None:
            val = result[0]
            if isinstance(val, datetime.date):
                return val
            return pd.Timestamp(val).date()
        return None

    def get_available_features(self) -> list[str]:
        """Get list of all stored feature names."""
        result = self.db.conn.execute(
            "SELECT DISTINCT feature_name FROM feature_values ORDER BY feature_name"
        ).fetchall()
        return [row[0] for row in result]

    def get_feature_coverage(self) -> dict[str, Any]:
        """Get summary stats about feature coverage."""
        ticker_count = self.db.conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM feature_values"
        ).fetchone()
        feature_count = self.db.conn.execute(
            "SELECT COUNT(DISTINCT feature_name) FROM feature_values"
        ).fetchone()
        row_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM feature_values"
        ).fetchone()
        date_range = self.db.conn.execute(
            "SELECT MIN(date), MAX(date) FROM feature_values"
        ).fetchone()

        return {
            "tickers": ticker_count[0] if ticker_count else 0,
            "features": feature_count[0] if feature_count else 0,
            "total_rows": row_count[0] if row_count else 0,
            "earliest_date": (
                str(date_range[0])
                if date_range and date_range[0]
                else None
            ),
            "latest_date": (
                str(date_range[1])
                if date_range and date_range[1]
                else None
            ),
        }
