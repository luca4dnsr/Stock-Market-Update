import unittest

import pandas as pd

from dashboard import HTML_TEMPLATE, _build_market_summary_html, _build_stock_rows


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
                    "evidence_outcome": "related_news_no_direct_catalyst",
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
        self.assertIn("관련 기사:", html)
        self.assertIn("aCell?.dataset.sortValue", HTML_TEMPLATE)

    def test_market_summary_separates_direct_and_context_sources(self):
        html = _build_market_summary_html(
            {
                "headline": "시장 요약",
                "observation": "시장 폭 관측",
                "interpretation": "직접 근거 해석",
                "recent_context": "최근 거시·섹터 맥락",
                "korea_market_scenario": {
                    "session_date": "2026-07-27",
                    "base_case": "조건부 기본 시나리오",
                    "positive_conditions": ["반도체 낙폭 축소"],
                    "risk_conditions": ["금리 상승 지속"],
                    "watch_items": ["반도체", "국채금리"],
                },
                "direct_source_titles": ["직접 기사"],
                "direct_source_urls": ["https://example.com/direct"],
                "direct_source_dates": ["2026-07-24"],
                "context_source_titles": ["맥락 기사"],
                "context_source_urls": ["https://example.com/context"],
                "context_source_dates": ["2026-07-20"],
                "rag_status": "rag_success",
                "provider": "Gemini + Finnhub RAG",
                "disclaimer": "투자 조언이 아닙니다.",
            }
        )

        self.assertIn("<strong>직접 근거</strong>", html)
        self.assertIn("<strong>맥락 근거</strong>", html)
        self.assertIn("<strong>최근 맥락</strong>", html)
        self.assertIn("</h2>\n    <br>\n    <p><strong>관측</strong>", html)
        self.assertIn("2026-07-24", html)
        self.assertIn("한국 증시 확인 조건과 시나리오", html)
        self.assertIn("반도체 낙폭 축소", html)
        self.assertIn("금리 상승 지속", html)
        self.assertIn("RAG rag_success", html)


if __name__ == "__main__":
    unittest.main()
