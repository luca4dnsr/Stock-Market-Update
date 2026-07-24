from datetime import date
import unittest

from ai_insights import _normalise_stock_batch, _select_market_articles


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


if __name__ == "__main__":
    unittest.main()
