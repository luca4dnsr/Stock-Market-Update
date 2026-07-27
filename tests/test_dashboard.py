import unittest

import pandas as pd

from dashboard import HTML_TEMPLATE, _build_stock_rows


class DashboardTest(unittest.TestCase):
    def test_stock_rows_render_safe_source_links_and_raw_market_cap_sort_values(self):
        stocks = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "name": "AAA Corp",
                    "market_cap_b": 2500.0,
                    "mc_rank": 1,
                    "day_rank": 1,
                    "sector": "Technology",
                    "business_summary": "사업 설명",
                    "move_reason": "등락 이유",
                    "return_1d": 2.0,
                    "return_1w": 1.0,
                    "return_1m": 3.0,
                    "return_3m": 4.0,
                    "source_titles": [
                        "차단 대상",
                        "<script>alert(1)</script>",
                        "두 번째 기사",
                        "세 번째 기사",
                        "네 번째 기사",
                    ],
                    "source_urls": [
                        "javascript:alert(1)",
                        'https://example.com/one?q="quoted"',
                        "http://example.com/two",
                        "HTTPS://example.com/three",
                        "https://example.com/four",
                    ],
                },
                {
                    "ticker": "BBB",
                    "name": "BBB Corp",
                    "market_cap_b": 900.0,
                    "mc_rank": 2,
                    "day_rank": 2,
                    "sector": "Financials",
                    "business_summary": "사업 설명",
                    "move_reason": "등락 이유",
                    "return_1d": 1.0,
                    "return_1w": 1.0,
                    "return_1m": 1.0,
                    "return_3m": 1.0,
                    "source_titles": [],
                    "source_urls": [],
                },
            ]
        )

        html = _build_stock_rows(stocks, "top-row")

        self.assertEqual(html.count('rel="noopener noreferrer"'), 3)
        self.assertNotIn("javascript:alert", html)
        self.assertNotIn("https://example.com/four", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("q=&quot;quoted&quot;", html)
        self.assertIn('data-sort-value="2500">$2.50T', html)
        self.assertIn('data-sort-value="900">$900.0B', html)
        self.assertIn("aCell?.dataset.sortValue", HTML_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
