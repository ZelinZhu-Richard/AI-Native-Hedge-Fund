"""Named constants with financial rationale.

Every constant documents WHY it has the value it does.
"""

# Largest S&P 500 single-day moves are ~12% (Oct 2008, Mar 2020).
# A >50% single-day return almost certainly indicates a data error
# or an unadjusted stock split rather than actual price movement.
MAX_SINGLE_DAY_RETURN = 0.50

# If a stock's closing price is identical for 5+ consecutive trading days,
# the data is likely stale (feed issue) rather than genuine zero-movement.
STALE_PRICE_THRESHOLD_DAYS = 5

# A >40% single-day price change strongly suggests an unadjusted stock split.
# This threshold catches common splits: 2:1 (-50%), 3:1 (-67%), 4:1 (-75%).
SPLIT_DETECTION_THRESHOLD = 0.40

# Default number of tickers per API batch request.
DEFAULT_BATCH_SIZE = 50

# Maximum retry attempts for transient failures (network, rate limit).
MAX_RETRIES = 3

# Base delay in seconds for exponential backoff between retries.
RETRY_BASE_DELAY_SECONDS = 1.0

# NYSE exchange code per ISO 10383 (Market Identifier Code).
NYSE_EXCHANGE_CODE = "XNYS"

# Minimum number of data points required before running statistical
# validation checks (outlier detection, stale data, etc.). Prevents
# false positives on newly listed or recently added tickers.
MIN_DATA_POINTS_FOR_VALIDATION = 30
