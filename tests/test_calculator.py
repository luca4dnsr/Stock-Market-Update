import unittest

import pandas as pd

from calculator import calculate_returns


def _price_series(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.bdate_range("2026-01-02", periods=len(values)),
        dtype=float,
    )


class CalculatorTest(unittest.TestCase):
    def test_three_month_return_is_missing_with_only_63_closes(self):
        returns = calculate_returns({"ABC": _price_series([100.0] * 63)})

        self.assertTrue(pd.isna(returns.loc[0, "return_3m"]))

    def test_three_month_return_uses_64_closes_for_63_sessions(self):
        values = [100.0] + [100.0] * 62 + [110.0]

        returns = calculate_returns({"ABC": _price_series(values)})

        self.assertEqual(returns.loc[0, "return_3m"], 10.0)


if __name__ == "__main__":
    unittest.main()
