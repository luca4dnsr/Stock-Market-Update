import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

import pandas as pd

from config import PRICE_HISTORY_PERIOD
from fetcher import (
    BENCHMARK_TICKER,
    UpstreamNotReady,
    _extract_batch_prices,
    _retry_inconsistent_prices,
    _validate_latest_date_coverage,
    _validate_return_history_coverage,
    fetch_benchmark_latest_date,
    fetch_all_data,
    fetch_price_data,
    get_expected_market_date,
)


def _price_series(row_count: int, end: str = "2026-07-24") -> pd.Series:
    return pd.Series(
        range(100, 100 + row_count),
        index=pd.bdate_range(end=end, periods=row_count),
        dtype=float,
    )


class FetcherTest(unittest.TestCase):
    def test_expected_market_date_uses_last_completed_xnys_session(self):
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 7, 27, 4, 8, tzinfo=timezone.utc)
            ),
            date(2026, 7, 24),
        )

    def test_expected_market_date_applies_close_grace(self):
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 7, 24, 20, 29, tzinfo=timezone.utc)
            ),
            date(2026, 7, 23),
        )
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 7, 24, 20, 30, tzinfo=timezone.utc)
            ),
            date(2026, 7, 24),
        )

    def test_expected_market_date_handles_holiday_and_early_close(self):
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 7, 3, 22, 0, tzinfo=timezone.utc)
            ),
            date(2026, 7, 2),
        )
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 11, 27, 18, 29, tzinfo=timezone.utc)
            ),
            date(2026, 11, 25),
        )
        self.assertEqual(
            get_expected_market_date(
                datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc)
            ),
            date(2026, 11, 27),
        )

    @patch("fetcher.time.sleep")
    @patch("fetcher.yf.download")
    def test_fetch_price_data_uses_buffered_period(self, mock_download, _mock_sleep):
        mock_download.return_value = pd.DataFrame(
            {"Adj Close": [100.0, 101.0], "Close": [101.0, 102.0]},
            index=pd.bdate_range("2026-07-23", periods=2),
        )

        prices = fetch_price_data(["ABC"])

        self.assertEqual(mock_download.call_args.kwargs["period"], PRICE_HISTORY_PERIOD)
        self.assertEqual(PRICE_HISTORY_PERIOD, "6mo")
        self.assertFalse(mock_download.call_args.kwargs["auto_adjust"])
        self.assertEqual(len(prices["ABC"]), 2)

    @patch("fetcher.time.sleep")
    @patch("fetcher.yf.download")
    def test_fetch_price_data_sorts_and_deduplicates_index(
        self, mock_download, _mock_sleep
    ):
        mock_download.return_value = pd.DataFrame(
            {"Adj Close": [102.0, 100.0, 103.0, 999.0]},
            index=pd.DatetimeIndex(
                ["2026-07-24", "2026-07-23", "2026-07-24", pd.NaT]
            ),
        )

        with self.assertLogs("fetcher", level="INFO") as captured:
            prices = fetch_price_data(["ABC"])

        self.assertEqual(
            list(prices["ABC"].index),
            list(pd.DatetimeIndex(["2026-07-23", "2026-07-24"])),
        )
        self.assertEqual(prices["ABC"].tolist(), [100.0, 103.0])
        logs = "\n".join(captured.output)
        self.assertIn("NaT=1", logs)
        self.assertIn("duplicate_index=1", logs)

    @patch("fetcher.time.sleep")
    @patch("fetcher.yf.download")
    def test_fetch_price_data_rejects_unadjusted_close_only_response(
        self, mock_download, _mock_sleep
    ):
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.bdate_range("2026-07-23", periods=2),
        )

        with self.assertLogs("fetcher", level="WARNING") as captured:
            prices = fetch_price_data(["ABC"])

        self.assertEqual(prices, {})
        self.assertIn("required=Adj Close", "\n".join(captured.output))

    @patch("fetcher.time.sleep")
    @patch("fetcher.yf.download")
    def test_fetch_price_data_does_not_mix_close_into_latest_adj_close_gap(
        self, mock_download, _mock_sleep
    ):
        columns = pd.MultiIndex.from_product(
            [["Adj Close", "Close"], ["AAA", "BBB"]],
            names=["Price", "Ticker"],
        )
        mock_download.return_value = pd.DataFrame(
            [
                [100.0, 200.0, 101.0, 201.0],
                [102.0, 202.0, 103.0, 203.0],
                [float("nan"), 204.0, 105.0, 205.0],
            ],
            index=pd.date_range("2026-07-22", periods=3),
            columns=columns,
        )

        with self.assertLogs("fetcher", level="INFO") as captured:
            prices = fetch_price_data(["AAA", "BBB"])

        self.assertEqual(prices["AAA"].index.max().date(), date(2026, 7, 23))
        self.assertEqual(prices["AAA"].iloc[-1], 102.0)
        self.assertEqual(prices["BBB"].index.max().date(), date(2026, 7, 24))
        logs = "\n".join(captured.output)
        self.assertIn("selected=Adj Close", logs)
        self.assertIn("Adj Close 최신값 결측 1건", logs)
        self.assertIn("비조정 Close로 대체하지 않고", logs)

    @patch("fetcher.yf.download")
    def test_benchmark_latest_date_uses_latest_close_date(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {
                ("Adj Close", BENCHMARK_TICKER): [6500.0, float("nan")],
                ("Close", BENCHMARK_TICKER): [6500.0, 6510.0],
            },
            index=pd.DatetimeIndex(["2026-07-23", "2026-07-24"]),
        )
        mock_download.return_value.columns = pd.MultiIndex.from_tuples(
            mock_download.return_value.columns,
            names=["Price", "Ticker"],
        )

        latest_date = fetch_benchmark_latest_date(date(2026, 7, 24))

        self.assertEqual(latest_date, date(2026, 7, 24))
        self.assertEqual(mock_download.call_args.args[0], BENCHMARK_TICKER)
        self.assertEqual(mock_download.call_args.kwargs["period"], "1mo")
        self.assertFalse(mock_download.call_args.kwargs["auto_adjust"])

    @patch("fetcher.yf.download")
    def test_benchmark_does_not_regress_when_expected_close_is_missing(
        self, mock_download
    ):
        mock_download.return_value = pd.DataFrame(
            {
                ("Adj Close", BENCHMARK_TICKER): [6500.0, 6501.0],
                ("Close", BENCHMARK_TICKER): [6500.0, float("nan")],
            },
            index=pd.DatetimeIndex(["2026-07-23", "2026-07-24"]),
        )
        mock_download.return_value.columns = pd.MultiIndex.from_tuples(
            mock_download.return_value.columns,
            names=["Price", "Ticker"],
        )

        with self.assertRaises(UpstreamNotReady) as captured:
            fetch_benchmark_latest_date(date(2026, 7, 24))

        self.assertEqual(captured.exception.reason, "benchmark_close_missing")
        self.assertEqual(captured.exception.expected_date, date(2026, 7, 24))
        self.assertEqual(captured.exception.raw_latest_date, date(2026, 7, 24))
        self.assertEqual(captured.exception.observed_date, date(2026, 7, 23))

    @patch("fetcher.yf.download", return_value=pd.DataFrame())
    def test_empty_benchmark_response_is_retryable_source_lag(
        self, _mock_download
    ):
        with self.assertRaises(UpstreamNotReady) as captured:
            fetch_benchmark_latest_date(date(2026, 7, 24))

        self.assertEqual(captured.exception.reason, "benchmark_empty")
        self.assertEqual(captured.exception.expected_date, date(2026, 7, 24))
        self.assertEqual(captured.exception.missing_count, 1)

    @patch("fetcher.get_sp500_components")
    @patch(
        "fetcher.fetch_benchmark_latest_date",
        side_effect=UpstreamNotReady(
            "benchmark_close_missing",
            date(2026, 7, 24),
        ),
    )
    def test_fetch_all_data_checks_benchmark_before_components(
        self, _mock_benchmark, mock_components
    ):
        with self.assertRaises(UpstreamNotReady):
            fetch_all_data(date(2026, 7, 24))

        mock_components.assert_not_called()

    def test_batch_union_shape_keeps_502_of_503_canonical_adjusted_series(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        columns = pd.MultiIndex.from_product(
            [["Adj Close", "Close"], tickers],
            names=["Price", "Ticker"],
        )
        index = pd.DatetimeIndex(
            ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-24", pd.NaT]
        )
        raw = pd.DataFrame(100.0, index=index, columns=columns)
        raw.loc[:, ("Adj Close", tickers[-1])] = [
            98.0,
            99.0,
            float("nan"),
            float("nan"),
            999.0,
        ]
        raw.loc[:, ("Close", tickers[-1])] = [98.0, 99.0, 100.0, 101.0, 999.0]

        prices = _extract_batch_prices(raw, tickers, "503종목 회귀")
        canonical, coverage = _validate_latest_date_coverage(
            prices, tickers, date(2026, 7, 24)
        )

        self.assertEqual(len(prices), 503)
        self.assertEqual(len(canonical), 502)
        self.assertGreater(coverage, 0.99)
        self.assertEqual(
            prices[tickers[-1]].index.max().date(),
            date(2026, 7, 23),
        )

    @patch("fetcher.fetch_market_caps")
    @patch("fetcher.fetch_benchmark_latest_date", return_value=date(2026, 7, 24))
    @patch("fetcher.fetch_price_data")
    @patch("fetcher.get_sp500_components")
    def test_fetch_all_data_rejects_insufficient_three_month_history(
        self, mock_components, mock_prices, _mock_benchmark, mock_market_caps
    ):
        mock_components.return_value = pd.DataFrame(
            [
                {"ticker": "AAA", "name": "A", "sector": "Tech", "sub_sector": "A"},
                {"ticker": "BBB", "name": "B", "sector": "Tech", "sub_sector": "B"},
            ]
        )
        mock_prices.return_value = {
            "AAA": _price_series(63),
            "BBB": _price_series(63),
        }

        with self.assertRaisesRegex(RuntimeError, "3개월"):
            fetch_all_data(date(2026, 7, 24))

        mock_market_caps.assert_not_called()

    @patch("fetcher.fetch_market_caps")
    @patch("fetcher.fetch_benchmark_latest_date", return_value=date(2026, 7, 24))
    @patch("fetcher.fetch_price_data")
    @patch("fetcher.get_sp500_components")
    def test_fetch_all_data_accepts_exact_three_month_history_boundary(
        self, mock_components, mock_prices, _mock_benchmark, mock_market_caps
    ):
        components = pd.DataFrame(
            [
                {"ticker": "AAA", "name": "A", "sector": "Tech", "sub_sector": "A"},
                {"ticker": "BBB", "name": "B", "sector": "Tech", "sub_sector": "B"},
            ]
        )
        mock_components.return_value = components
        mock_prices.return_value = {
            "AAA": _price_series(64),
            "BBB": _price_series(64),
        }
        mock_market_caps.return_value = {"AAA": 100.0, "BBB": 200.0}

        result_components, prices, market_caps = fetch_all_data(
            date(2026, 7, 24)
        )

        self.assertIs(result_components, components)
        self.assertEqual(set(prices), {"AAA", "BBB"})
        self.assertEqual(market_caps, {"AAA": 100.0, "BBB": 200.0})

    def test_latest_date_contract_accepts_identical_503_series(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        prices = {ticker: _price_series(64) for ticker in tickers}

        canonical, coverage = _validate_latest_date_coverage(
            prices, tickers, date(2026, 7, 24)
        )

        self.assertEqual(len(canonical), 503)
        self.assertEqual(coverage, 1.0)

    def test_latest_date_contract_clips_future_partial_row(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        prices = {ticker: _price_series(64) for ticker in tickers}
        prices[tickers[-1]] = _price_series(64, end="2026-07-27")

        canonical, coverage = _validate_latest_date_coverage(
            prices, tickers, date(2026, 7, 24)
        )

        self.assertEqual(len(canonical), 503)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(
            canonical[tickers[-1]].index.max().date(),
            date(2026, 7, 24),
        )

    def test_latest_date_contract_quarantines_future_only_ticker(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        prices = {ticker: _price_series(64) for ticker in tickers}
        prices[tickers[-1]] = pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(["2026-07-27", "2026-07-28"]),
        )

        canonical, coverage = _validate_latest_date_coverage(
            prices, tickers, date(2026, 7, 24)
        )

        self.assertEqual(len(canonical), 502)
        self.assertNotIn(tickers[-1], canonical)
        self.assertGreaterEqual(coverage, 0.98)

    def test_latest_date_contract_preserves_full_503_ticker_threshold(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        passing = {
            ticker: _price_series(
                64, end="2026-07-24" if index < 493 else "2026-07-23"
            )
            for index, ticker in enumerate(tickers)
        }
        failing = {
            ticker: _price_series(
                64, end="2026-07-24" if index < 492 else "2026-07-23"
            )
            for index, ticker in enumerate(tickers)
        }

        canonical, coverage = _validate_latest_date_coverage(
            passing, tickers, date(2026, 7, 24)
        )
        self.assertEqual(len(canonical), 493)
        self.assertGreaterEqual(coverage, 0.98)

        with self.assertRaises(UpstreamNotReady) as captured:
            _validate_latest_date_coverage(
                failing, tickers, date(2026, 7, 24)
            )
        self.assertEqual(
            captured.exception.reason,
            "constituent_adj_close_lag",
        )

    def test_latest_date_contract_clips_future_partial_majority(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        prices = {
            ticker: _price_series(
                64, end="2026-07-27" if index < 493 else "2026-07-24"
            )
            for index, ticker in enumerate(tickers)
        }

        canonical, coverage = _validate_latest_date_coverage(
            prices, tickers, date(2026, 7, 24)
        )

        self.assertEqual(len(canonical), 503)
        self.assertEqual(coverage, 1.0)
        self.assertTrue(
            all(
                series.index.max().date() == date(2026, 7, 24)
                for series in canonical.values()
            )
        )

    def test_latest_date_contract_rejects_502_stale_as_source_lag(self):
        tickers = [f"T{index:03d}" for index in range(503)]
        prices = {
            ticker: _price_series(
                64, end="2026-07-23" if index < 502 else "2026-07-24"
            )
            for index, ticker in enumerate(tickers)
        }

        with self.assertRaises(UpstreamNotReady) as captured:
            _validate_latest_date_coverage(
                prices, tickers, date(2026, 7, 24)
            )

        self.assertEqual(captured.exception.aligned, 1)
        self.assertEqual(captured.exception.stale_count, 502)
        self.assertEqual(captured.exception.observed_date, date(2026, 7, 23))

    @patch("fetcher.fetch_price_data")
    def test_retry_targets_one_off_date_ticker_when_coverage_passes(
        self, mock_prices
    ):
        tickers = [f"T{index:03d}" for index in range(503)]
        initial = {
            ticker: _price_series(
                64, end="2026-07-23" if index == 502 else "2026-07-24"
            )
            for index, ticker in enumerate(tickers)
        }
        mock_prices.return_value = {
            tickers[-1]: _price_series(64),
        }

        prices = _retry_inconsistent_prices(
            initial, tickers, date(2026, 7, 24)
        )

        mock_prices.assert_called_once_with([tickers[-1]])
        self.assertEqual(
            prices[tickers[-1]].index.max().date(),
            date(2026, 7, 24),
        )

    @patch("fetcher.fetch_price_data")
    def test_retry_targets_small_mixed_gap_below_coverage_threshold(
        self, mock_prices
    ):
        tickers = [f"T{index:03d}" for index in range(503)]
        initial = {
            ticker: _price_series(
                64,
                end="2026-07-23" if 492 <= index < 498 else "2026-07-24",
            )
            for index, ticker in enumerate(tickers[:498])
        }
        retry_tickers = tickers[492:]
        mock_prices.return_value = {
            ticker: _price_series(64) for ticker in retry_tickers
        }

        prices = _retry_inconsistent_prices(
            initial, tickers, date(2026, 7, 24)
        )

        mock_prices.assert_called_once_with(retry_tickers)
        self.assertEqual(len(prices), 503)

    @patch("fetcher.fetch_market_caps")
    @patch("fetcher.fetch_benchmark_latest_date", return_value=date(2026, 7, 24))
    @patch("fetcher.fetch_price_data")
    @patch("fetcher.get_sp500_components")
    def test_fetch_all_data_skips_mass_retry_for_systemic_stale_prices(
        self,
        mock_components,
        mock_prices,
        _mock_benchmark,
        mock_market_caps,
    ):
        mock_components.return_value = pd.DataFrame(
            [
                {"ticker": "AAA", "name": "A", "sector": "Tech", "sub_sector": "A"},
                {"ticker": "BBB", "name": "B", "sector": "Tech", "sub_sector": "B"},
            ]
        )
        stale = _price_series(64, end="2026-07-23")
        mock_prices.return_value = {
            "AAA": _price_series(64),
            "BBB": stale,
        }

        with self.assertRaises(UpstreamNotReady):
            fetch_all_data(date(2026, 7, 24))

        mock_prices.assert_called_once_with(["AAA", "BBB"])
        mock_market_caps.assert_not_called()

    def test_return_history_coverage_uses_full_503_ticker_threshold(self):
        valid = _price_series(64)
        insufficient = _price_series(63)
        passing = {
            f"T{index:03d}": valid if index < 493 else insufficient
            for index in range(503)
        }
        failing = {
            f"T{index:03d}": valid if index < 492 else insufficient
            for index in range(503)
        }

        _validate_return_history_coverage(passing, 503)
        with self.assertRaisesRegex(RuntimeError, "3개월"):
            _validate_return_history_coverage(failing, 503)

    def test_return_history_coverage_rejects_zero_return_baseline(self):
        prices = _price_series(64)
        prices.iloc[0] = 0

        with self.assertRaisesRegex(RuntimeError, "3개월"):
            _validate_return_history_coverage({"AAA": prices}, 1)


if __name__ == "__main__":
    unittest.main()
