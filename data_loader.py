"""
src/data_loader.py
==================
Market Data Pipeline
--------------------
Downloads historical OHLCV price data from Yahoo Finance for a target
ticker and an optional benchmark (default: SPY). Computes daily log
returns and exposes a clean, analysis-ready DataFrame.

Design decisions
~~~~~~~~~~~~~~~~
* **Log returns** are used throughout because they are time-additive,
  approximately Gaussian, and numerically stable — all properties that
  matter in event-study and regression contexts.
* Missing values introduced by non-overlapping trading calendars are
  forward-filled then back-filled so the returned DataFrame always has
  a contiguous date index.
* The module is intentionally side-effect free: no global state, no
  files written unless ``save_raw`` is set to ``True``.

Usage example
~~~~~~~~~~~~~
    >>> from src.data_loader import DataLoader
    >>> loader = DataLoader(ticker="AAPL", benchmark="SPY")
    >>> df = loader.fetch(start="2022-01-01", end="2024-01-01")
    >>> print(df.head())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── Module-level logger ──────────────────────────────────────────────────────
# Each module owns its own logger so log messages are easy to trace.
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_BENCHMARK: str = "SPY"
RAW_DATA_DIR: Path = Path("data/raw")


class DataLoader:
    """Downloads and preprocesses market price data for a single ticker.

    Parameters
    ----------
    ticker : str
        The primary equity ticker symbol (e.g. ``"AAPL"``).
    benchmark : str, optional
        Benchmark ticker used to compute market returns for the event
        study's market model. Defaults to ``"SPY"`` (S&P 500 ETF).

    Attributes
    ----------
    ticker : str
        Upper-cased primary ticker.
    benchmark : str
        Upper-cased benchmark ticker.
    _raw_data : pd.DataFrame or None
        Cached raw download; populated after the first call to
        :meth:`fetch`.
    """

    def __init__(
        self,
        ticker: str,
        benchmark: str = DEFAULT_BENCHMARK,
    ) -> None:
        self.ticker: str = ticker.upper().strip()
        self.benchmark: str = benchmark.upper().strip()
        self._raw_data: Optional[pd.DataFrame] = None

        logger.info(
            "DataLoader initialised — ticker=%s  benchmark=%s",
            self.ticker,
            self.benchmark,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch(
        self,
        start: str,
        end: str,
        save_raw: bool = False,
    ) -> pd.DataFrame:
        """Download price data and return a processed returns DataFrame.

        Parameters
        ----------
        start : str
            Start date in ``"YYYY-MM-DD"`` format (inclusive).
        end : str
            End date in ``"YYYY-MM-DD"`` format (exclusive, Yahoo
            Finance convention).
        save_raw : bool, optional
            If ``True``, persists the raw adjusted-close prices to
            ``data/raw/<ticker>_<benchmark>_raw.csv``. Defaults to
            ``False``.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by ``Date`` (``DatetimeIndex``) with
            the following columns:

            * ``close_<TICKER>``   — adjusted closing price
            * ``close_<BENCHMARK>`` — adjusted closing price
            * ``log_ret_<TICKER>``  — daily log return  (target)
            * ``log_ret_<BENCHMARK>`` — daily log return (benchmark)

        Raises
        ------
        ValueError
            If ``yfinance`` returns an empty DataFrame for either
            ticker (e.g., bad symbol or date range outside trading
            history).
        """
        logger.info(
            "Fetching price data for %s and %s from %s to %s …",
            self.ticker,
            self.benchmark,
            start,
            end,
        )

        # Download both tickers in a single network call for efficiency.
        symbols: list[str] = [self.ticker, self.benchmark]
        raw: pd.DataFrame = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            auto_adjust=True,   # Adjusts for splits & dividends automatically
            progress=False,     # Suppress tqdm bar — cleaner in notebooks
        )

        self._validate_download(raw, symbols)

        # yfinance returns a MultiIndex when multiple tickers are requested.
        # We extract only the "Close" price level.
        close_prices: pd.DataFrame = self._extract_close(raw, symbols)

        if save_raw:
            self._persist_raw(close_prices)

        # Compute log returns and merge into the final DataFrame.
        result: pd.DataFrame = self._build_returns_frame(close_prices)

        logger.info(
            "Data pipeline complete — %d trading days returned.",
            len(result),
        )
        return result

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _validate_download(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
    ) -> None:
        """Raise ``ValueError`` if the download result is unusable.

        Parameters
        ----------
        raw : pd.DataFrame
            The raw DataFrame returned by ``yf.download``.
        symbols : list[str]
            Expected ticker symbols.
        """
        if raw.empty:
            raise ValueError(
                f"yfinance returned an empty DataFrame for {symbols}. "
                "Check your ticker symbols and date range."
            )

        logger.info("Download successful — raw shape: %s", raw.shape)

    def _extract_close(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Isolate adjusted close prices and rename columns clearly.

        Parameters
        ----------
        raw : pd.DataFrame
            MultiIndex DataFrame from ``yf.download``.
        symbols : list[str]
            Ticker symbols in the order they were requested.

        Returns
        -------
        pd.DataFrame
            Single-level column DataFrame: ``close_<TICKER>`` and
            ``close_<BENCHMARK>``.
        """
        # Handle both single-ticker (flat) and multi-ticker (MultiIndex) cases.
        if isinstance(raw.columns, pd.MultiIndex):
            close: pd.DataFrame = raw["Close"].copy()
        else:
            # Fallback: single ticker (shouldn't occur here, but defensive)
            close = raw[["Close"]].copy()
            close.columns = symbols

        # Rename to human-readable column names.
        rename_map: dict[str, str] = {
            sym: f"close_{sym}" for sym in symbols
        }
        close.rename(columns=rename_map, inplace=True)

        # Handle gaps: forward-fill (carry last known price) then
        # back-fill (handles leading NaNs at the very start).
        n_missing_before: int = close.isna().sum().sum()
        close.ffill(inplace=True)
        close.bfill(inplace=True)
        n_missing_after: int = close.isna().sum().sum()

        if n_missing_before > 0:
            logger.warning(
                "Imputed %d missing close price(s) via ffill/bfill.",
                n_missing_before - n_missing_after,
            )

        return close

    def _build_returns_frame(
        self,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute daily log returns and attach them to the price frame.

        Log return formula:  r_t = ln(P_t / P_{t-1})

        The first row is dropped because the return on day 0 is
        undefined (no prior price).

        Parameters
        ----------
        close : pd.DataFrame
            Adjusted close prices with columns ``close_<TICKER>`` and
            ``close_<BENCHMARK>``.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame of prices and log returns.
        """
        log_returns: pd.DataFrame = np.log(close / close.shift(1))

        # Rename the return columns.
        log_returns.columns = [
            col.replace("close_", "log_ret_") for col in log_returns.columns
        ]

        # Merge prices and returns side-by-side, drop the first NaN row.
        combined: pd.DataFrame = pd.concat(
            [close, log_returns], axis=1
        ).dropna()

        # Ensure the index is a proper DatetimeIndex with UTC stripped.
        combined.index = pd.to_datetime(combined.index).tz_localize(None)
        combined.index.name = "Date"

        return combined

    def _persist_raw(self, close: pd.DataFrame) -> None:
        """Save raw adjusted close prices to CSV.

        Parameters
        ----------
        close : pd.DataFrame
            Adjusted close prices to persist.
        """
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename: Path = RAW_DATA_DIR / f"{self.ticker}_{self.benchmark}_raw.csv"
        close.to_csv(filename)
        logger.info("Raw data saved → %s", filename)

    # ── Dunder Helpers ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DataLoader(ticker='{self.ticker}', "
            f"benchmark='{self.benchmark}')"
        )
