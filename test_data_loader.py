"""
tests/test_data_loader.py
==========================
Unit tests for src/data_loader.py

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loader() -> DataLoader:
    """Return a DataLoader instance for MSFT vs SPY."""
    return DataLoader(ticker="MSFT", benchmark="SPY")


@pytest.fixture(scope="module")
def returns_df(loader: DataLoader) -> pd.DataFrame:
    """Fetch one year of data — cached for the whole test module."""
    return loader.fetch(start="2023-01-01", end="2024-01-01")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataLoaderInit:
    def test_ticker_uppercased(self) -> None:
        loader = DataLoader(ticker="aapl")
        assert loader.ticker == "AAPL"

    def test_benchmark_uppercased(self) -> None:
        loader = DataLoader(ticker="AAPL", benchmark="spy")
        assert loader.benchmark == "SPY"

    def test_default_benchmark(self) -> None:
        loader = DataLoader(ticker="GOOG")
        assert loader.benchmark == "SPY"

    def test_repr(self) -> None:
        loader = DataLoader(ticker="TSLA")
        assert "TSLA" in repr(loader)


class TestDataLoaderFetch:
    def test_returns_dataframe(self, returns_df: pd.DataFrame) -> None:
        assert isinstance(returns_df, pd.DataFrame)

    def test_expected_columns_present(self, returns_df: pd.DataFrame) -> None:
        expected = {"close_MSFT", "close_SPY", "log_ret_MSFT", "log_ret_SPY"}
        assert expected.issubset(set(returns_df.columns))

    def test_no_missing_values(self, returns_df: pd.DataFrame) -> None:
        assert returns_df.isna().sum().sum() == 0, \
            "DataFrame should contain no NaN values after cleaning."

    def test_index_is_datetime(self, returns_df: pd.DataFrame) -> None:
        assert isinstance(returns_df.index, pd.DatetimeIndex)

    def test_log_returns_are_finite(self, returns_df: pd.DataFrame) -> None:
        assert np.isfinite(returns_df["log_ret_MSFT"]).all()
        assert np.isfinite(returns_df["log_ret_SPY"]).all()

    def test_log_returns_reasonable_range(self, returns_df: pd.DataFrame) -> None:
        # Daily log returns for large caps should rarely exceed ±20%.
        assert returns_df["log_ret_MSFT"].abs().max() < 0.20

    def test_non_empty(self, returns_df: pd.DataFrame) -> None:
        assert len(returns_df) > 100, \
            "Expected at least 100 trading days for a 1-year window."


class TestDataLoaderValidation:
    def test_raises_on_bad_ticker(self) -> None:
        loader = DataLoader(ticker="XXXXXX_INVALID")
        with pytest.raises(ValueError):
            loader.fetch(start="2023-01-01", end="2024-01-01")
