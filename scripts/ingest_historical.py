#!/usr/bin/env python3
"""Ingest historical market data from a data provider.

Usage:
    python scripts/ingest_historical.py --universe test --start-date 2024-01-01 --end-date 2024-03-01
    python scripts/ingest_historical.py --universe sp500 --start-date 2019-01-01 --end-date 2024-12-31 --batch-size 25

Exit codes:
    0 - All tickers ingested successfully
    1 - Partial success (some tickers failed)
    2 - Fatal error
"""

from __future__ import annotations

import argparse
import datetime
import sys

from rich.console import Console

from meridian.config.settings import get_settings
from meridian.config.universe import SP500_TICKERS, TEST_UNIVERSE, get_sp500_tickers
from meridian.core.logging import get_logger
from meridian.data.ingest import ingest_historical
from meridian.data.providers.alpha_vantage import AlphaVantageProvider
from meridian.data.providers.yahoo import YahooFinanceProvider
from meridian.data.storage.database import MeridianDatabase

logger = get_logger("scripts.ingest")
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest historical market data"
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.date.fromisoformat(s),
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.date.fromisoformat(s),
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe",
        choices=["test", "sp500"],
        default="test",
        help="Ticker universe to ingest (default: test)",
    )
    parser.add_argument(
        "--provider",
        choices=["yahoo", "alpha_vantage"],
        default="yahoo",
        help="Data provider (default: yahoo)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of tickers per batch (default: from settings)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous incomplete ingestion",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    # Select universe
    if args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = list(TEST_UNIVERSE)

    console.print(
        f"[bold]Meridian Data Ingestion[/bold]\n"
        f"  Universe: {args.universe} ({len(tickers)} tickers)\n"
        f"  Period: {args.start_date} to {args.end_date}\n"
        f"  Provider: {args.provider}\n"
    )

    # Create provider
    if args.provider == "alpha_vantage":
        provider = AlphaVantageProvider(
            api_key=settings.data_provider.alpha_vantage_api_key,
        )
    else:
        provider = YahooFinanceProvider(
            rate_limit_per_second=settings.data_provider.yahoo_rate_limit_per_second,
        )

    # Run ingestion
    try:
        with MeridianDatabase(settings.database.path) as db:
            db.create_schema()

            result = ingest_historical(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                provider=provider,
                db=db,
                batch_size=args.batch_size,
            )

        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Total:     {result['total']}")
        console.print(f"  Succeeded: {result['succeeded']}")
        console.print(f"  Failed:    {result['failed']}")
        console.print(f"  Skipped:   {result['skipped']}")

        if "clean_tickers" in result:
            console.print(f"  Clean:     {result['clean_tickers']}")
            console.print(f"  Dirty:     {result['dirty_tickers']}")

        if result["failed"] > 0 and result["succeeded"] > 0:
            return 1  # Partial success
        elif result["failed"] > 0 and result["succeeded"] == 0:
            return 2  # All failed
        return 0

    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.error("fatal ingestion error", error=str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
