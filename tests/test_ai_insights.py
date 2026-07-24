from datetime import date
import unittest

from ai_insights import (
    _normalise_company_articles,
    _normalise_stock_batch,
    _parse_json,
    _select_market_articles,
)


def _article(article_id: str, headline: str, source: str, published_date: str) -> dict:
    return {
        "article_id": article_id,
        "headline": headline,
        "summary": headline,
        "source": source,
        "published_at": f"{published_date}T16:00:00-04:00",
        "published_date": published_date,
        "url": f"https://example.com/{article_id}",
        "related": [],
    }


class AiInsightsTest(unittest.TestCase):
    def test_market_sources_are_selected_by_code(self):
        end = date(2026, 7, 23)
        articles = [
            _article("1", "S&P 500 falls after Federal Reserve rate comments", "Reuters", "2026-07-23"),
            _article("2", "Nasdaq stocks react to earnings guidance", "Reuters", "2026-07-23"),
            _article("3", "Treasury bond yields rise after inflation data", "AP", "2026-07-22"),
            _article("4", "Local sports team signs a new player", "Sports Wire", "2026-07-23"),
        ]

        selected = _select_market_articles(articles, end)

        self.assertCountEqual([article["article_id"] for article in selected], ["1", "2", "3"])

    def test_stock_sources_do_not_depend_on_model_article_ids(self):
        items = [
            {
                "ticker": "ABC",
                "business_source_en": "Example business",
                "selected_finnhub_articles": [
                    _article("100", "ABC raises revenue guidance", "Reuters", "2026-07-23")
                ],
            }
        ]
        generated = {
            "items": [
                {
                    "ticker": "ABC",
                    "business_ko": "예시 사업을 영위하는 기업",
                    "move_reason_ko": "매출 가이던스 상향을 발표했습니다.",
                    "evidence_status": "verified",
                }
            ]
        }

        entries = _normalise_stock_batch(generated, items, "Gemini + Finnhub")

        self.assertEqual(entries["ABC"]["source_urls"], ["https://example.com/100"])
        self.assertEqual(entries["ABC"]["provider"], "Gemini + Finnhub")
        self.assertEqual(entries["ABC"]["model_verdict"], "verified")

    def test_company_news_reports_pre_cap_filter_count(self):
        start = date(2026, 6, 23)
        end = date(2026, 7, 24)
        raw_items = [
            {
                "id": index,
                "headline": f"ABC news {index}",
                "url": f"https://example.com/{index}",
                "datetime": 1782497600 + index,
                "related": "ABC",
            }
            for index in range(1, 5)
        ]

        selected, passed_count = _normalise_company_articles("ABC", raw_items, start, end)

        self.assertEqual(passed_count, 4)
        self.assertEqual(len(selected), 3)

    def test_company_news_prioritises_catalyst_over_newer_generic_article(self):
        start = date(2026, 6, 23)
        end = date(2026, 7, 24)
        raw_items = [
            {
                "id": "1",
                "headline": "ABC shares move in afternoon trading",
                "url": "https://example.com/1",
                "datetime": 1782497603,
                "related": "ABC",
            },
            {
                "id": "2",
                "headline": "ABC raises full-year guidance after earnings beat",
                "url": "https://example.com/2",
                "datetime": 1782497602,
                "related": "ABC",
            },
            {
                "id": "3",
                "headline": "ABC company profile update",
                "url": "https://example.com/3",
                "datetime": 1782497601,
                "related": "ABC",
            },
            {
                "id": "4",
                "headline": "ABC hosts investor conference",
                "url": "https://example.com/4",
                "datetime": 1782497600,
                "related": "ABC",
            },
        ]

        selected, passed_count = _normalise_company_articles("ABC", raw_items, start, end)

        self.assertEqual(passed_count, 4)
        self.assertEqual(selected[0]["article_id"], "2")
        self.assertNotIn("4", [article["article_id"] for article in selected])

    def test_json_repair_adds_missing_array_item_comma(self):
        malformed = '{"items":[{"ticker":"AAA"}\n{"ticker":"BBB"}]}'

        parsed = _parse_json(malformed)

        self.assertEqual([item["ticker"] for item in parsed["items"]], ["AAA", "BBB"])


if __name__ == "__main__":
    unittest.main()
