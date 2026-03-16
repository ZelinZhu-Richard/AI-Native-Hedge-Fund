#!/usr/bin/env python3
"""Initialize the Meridian database and create schema.

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

from pathlib import Path

from meridian.config.settings import get_settings
from meridian.core.logging import get_logger
from meridian.data.storage.database import MeridianDatabase

logger = get_logger("setup_db")


def main() -> None:
    settings = get_settings()
    db_path = settings.database.path

    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("initializing database", path=str(db_path))

    with MeridianDatabase(db_path) as db:
        db.create_schema()

        # Verify tables exist
        tables = db.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        expected = ["ohlcv", "data_quality", "ticker_metadata", "ingestion_log"]
        for table in expected:
            if table in table_names:
                logger.info("table verified", table=table)
            else:
                logger.error("table missing", table=table)
                raise RuntimeError(f"Table {table} was not created")

    logger.info("database setup complete", path=str(db_path))


if __name__ == "__main__":
    main()
