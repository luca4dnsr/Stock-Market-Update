import unittest
from unittest.mock import patch

import pandas as pd

from config import PRICE_HISTORY_PERIOD
from fetcher import (
    _validate_return_history_coverage,
    fetch_all_data,
    fetch_price_data,
)


def _price_series(row_count: int) -> pd.Series:
    return pd.Series(
        range(100, 100 + row_count),
        index=pd.bdate_range("2026-01-02", periods=row_count),
        dtype=float,
    )


class FetcherTest(unittest.TestCase):
    @patch("fetcher.time.sleep")
    @patch("fetcher.yf.download")
    def test_fetch_price_data_uses_buffered_period(self, mock_download, _mock_sleep):
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.bdate_range("2026-07-23", periods=2),
        )

        prices = fetch_price_data(["ABC"])

        self.assertEqual(mock_download.call_args.kwargs["period"], PRICE_HISTORY_PERIOD)
        self.assertEqual(PRICE_HISTORY_PERIOD, "6mo")
        self.assertEqual(len(prices["ABC"]), 2)

    @patch("fetcher.fetch_market_caps")
    @patch("fetcher.fetch_price_data")
    @patch("fetcher.get_sp500_components")
    def test_fetch_all_data_rejects_insufficient_three_month_history(
        self, mock_components, mock_prices, mock_market_caps
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
            fetch_all_data()

        mock_market_caps.assert_not_called()

    @patch("fetcher.fetch_market_caps")
    @patch("fetcher.fetch_price_data")
    @patch("fetcher.get_sp500_components")
    def test_fetch_all_data_accepts_exact_three_month_history_boundary(
        self, mock_components, mock_prices, mock_market_caps
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

        result_components, prices, market_caps = fetch_all_data()

        self.assertIs(result_components, components)
        self.assertEqual(set(prices), {"AAA", "BBB"})
        self.assertEqual(market_caps, {"AAA": 100.0, "BBB": 200.0})

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
