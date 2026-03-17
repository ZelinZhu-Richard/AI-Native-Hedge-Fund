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

# --- Feature engineering constants ---

# 200-day SMA is the longest lookback in technical features.
# Features within this warmup window are NaN, not partially computed.
MAX_FEATURE_LOOKBACK_DAYS = 200

# Trading days are ~71% of calendar days. When querying data for feature
# computation, multiply lookback by this to ensure enough trading days.
LOOKBACK_CALENDAR_BUFFER_MULTIPLIER = 1.5

# Initial version tag for all features.
DEFAULT_FEATURE_VERSION = 1

# Annualization factor: ~252 trading days per year.
TRADING_DAYS_PER_YEAR = 252

# --- Regime detection constants ---

# Default PCA rolling window: 1 trading year.
# 252 days gives stable covariance estimation while adapting
# to structural breaks (COVID, rate hiking cycles).
PCA_DEFAULT_WINDOW_DAYS = 252

# PCA refits every 21 trading days (~1 month).
# Monthly refits balance adaptation speed vs computational cost.
# Daily refit is noisy; quarterly is too slow to catch transitions.
PCA_DEFAULT_REFIT_FREQUENCY = 21

# Default PCA components to retain.
# 10 components typically capture 70-85% of variance in equity
# features. Empirical research shows 10-15 is optimal.
PCA_DEFAULT_N_COMPONENTS = 10

# Default number of market regimes for clustering.
# 4 regimes map empirically to: low-vol bull, high-vol bull,
# low-vol bear, high-vol bear (crisis).
DEFAULT_N_REGIMES = 4

# Minimum observations for a valid PCA fit.
# Need at least 2x n_components to avoid singular covariance.
MIN_PCA_OBSERVATIONS = 50
