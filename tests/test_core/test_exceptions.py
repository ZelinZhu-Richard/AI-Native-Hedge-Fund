"""Tests for the exception hierarchy."""

from __future__ import annotations

from meridian.core.exceptions import (
    ConfigurationError,
    DataProviderError,
    DataValidationError,
    IngestionError,
    MeridianError,
    StorageError,
)


class TestMeridianError:
    def test_basic_message(self):
        err = MeridianError("something failed")
        assert str(err) == "something failed"
        assert err.message == "something failed"
        assert err.context == {}

    def test_with_context(self):
        err = MeridianError("fetch failed", ticker="AAPL", status=404)
        assert "ticker='AAPL'" in str(err)
        assert "status=404" in str(err)
        assert err.context == {"ticker": "AAPL", "status": 404}

    def test_is_exception(self):
        err = MeridianError("test")
        assert isinstance(err, Exception)

    def test_context_available_for_logging(self):
        err = MeridianError("db error", table="ohlcv", rows=100)
        # Context should be usable with logger.bind(**err.context)
        assert "table" in err.context
        assert err.context["rows"] == 100


class TestExceptionHierarchy:
    def test_data_provider_error_inherits(self):
        err = DataProviderError("timeout", provider="yahoo")
        assert isinstance(err, MeridianError)
        assert isinstance(err, Exception)
        assert err.context["provider"] == "yahoo"

    def test_data_validation_error(self):
        err = DataValidationError("bad data", ticker="AAPL", field="volume")
        assert isinstance(err, MeridianError)
        assert err.context["ticker"] == "AAPL"

    def test_storage_error(self):
        err = StorageError("write failed", table="ohlcv")
        assert isinstance(err, MeridianError)
        assert "table='ohlcv'" in str(err)

    def test_configuration_error(self):
        err = ConfigurationError("missing key", var="API_KEY")
        assert isinstance(err, MeridianError)

    def test_ingestion_error(self):
        err = IngestionError("batch failed", batch=3, tickers=50)
        assert isinstance(err, MeridianError)
        assert err.context["batch"] == 3

    def test_can_catch_by_base(self):
        """All specific errors should be catchable via MeridianError."""
        errors = [
            DataProviderError("test"),
            DataValidationError("test"),
            StorageError("test"),
            ConfigurationError("test"),
            IngestionError("test"),
        ]
        for err in errors:
            try:
                raise err
            except MeridianError:
                pass  # Should be caught
