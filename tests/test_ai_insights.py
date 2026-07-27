from datetime import date
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ai_insights import (
    enrich_with_ai,
    _build_market_summary,
    _fallback_market_prompt,
    _fallback_stock_entries,
    _fallback_stock_prompt,
    _is_non_retryable_gemini_error,
    _normalise_company_articles,
    _normalise_stock_batch,
    _parse_json,
    _request_gemini_json,
    _request_nim_json,
    _research_market_summary,
    _select_market_articles,
    _valid_stock_response_items,
)
from config import AI_INSIGHTS_CACHE_VERSION


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


def _stock_input(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "name": f"{ticker} Inc.",
        "sector": "Information Technology",
        "return_1d": 1.0,
        "business_source_en": f"{ticker} business",
        "selected_finnhub_articles": [],
    }


def _nim_stock_item(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "business_ko": f"{ticker} 사업을 영위하는 기업",
        "move_reason_ko": "",
        "evidence_status": "limited",
    }


class AiInsightsTest(unittest.TestCase):
    @patch("ai_insights.requests.post")
    def test_gemini_uses_api_enum_for_json_mime_type(self, mock_post):
        response = Mock(ok=True)
        response.json.return_value = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": '{"items": []}'}]},
                }
            ]
        }
        mock_post.return_value = response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            generated = _request_gemini_json("prompt", {"type": "OBJECT"})

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            payload["generationConfig"]["responseFormat"]["text"]["mimeType"],
            "APPLICATION_JSON",
        )
        self.assertEqual(generated, {"items": []})

    @patch("ai_insights.requests.post")
    def test_nim_logs_response_shape_without_logging_content(self, mock_post):
        response = Mock(ok=True)
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": (
                            '{"items":[{"ticker":"AAA","business_ko":"TOP SECRET"}],'
                            '"LEAK KEY":"hidden"}'
                        ),
                        "reasoning_content": "private reasoning",
                    },
                }
            ]
        }
        mock_post.return_value = response

        with (
            patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}),
            self.assertLogs("ai_insights", level="INFO") as captured,
        ):
            generated = _request_nim_json("model", "system", "prompt")

        logs = "\n".join(captured.output)
        self.assertEqual(generated["items"][0]["ticker"], "AAA")
        self.assertIn("finish_reason=length", logs)
        self.assertIn("top_level_keys=items", logs)
        self.assertIn("unknown_key_count=1", logs)
        self.assertNotIn("TOP SECRET", logs)
        self.assertNotIn("private reasoning", logs)
        self.assertNotIn("LEAK KEY", logs)

    def test_nim_rejects_null_and_non_string_stock_fields(self):
        items = [_stock_input("AAA"), _stock_input("BBB"), _stock_input("CCC")]
        generated = {
            "items": [
                {
                    **_nim_stock_item("AAA"),
                    "business_ko": None,
                },
                {
                    **_nim_stock_item("BBB"),
                    "move_reason_ko": None,
                },
                {
                    **_nim_stock_item("CCC"),
                    "evidence_status": ["limited"],
                },
                {
                    **_nim_stock_item("LEAK TICKER"),
                },
            ]
        }

        with self.assertLogs("ai_insights", level="INFO") as captured:
            valid = _valid_stock_response_items(generated, items, attempt=1)

        self.assertEqual(valid, {})
        logs = "\n".join(captured.output)
        self.assertIn("unexpected_count=1", logs)
        self.assertNotIn("LEAK TICKER", logs)

    def test_nim_prompts_require_exact_top_level_objects(self):
        start = date(2026, 6, 24)
        end = date(2026, 7, 24)

        stock_prompt = _fallback_stock_prompt(
            [_stock_input("AAA")], "2026-07-24", start, end
        )
        market_prompt = _fallback_market_prompt(
            {}, [], "2026-07-24", start, end
        )

        self.assertIn('{"items":[', stock_prompt)
        for field in ("ticker", "business_ko", "move_reason_ko", "evidence_status"):
            self.assertIn(f'"{field}"', stock_prompt)
        for field in ("headline", "observation", "interpretation"):
            self.assertIn(f'"{field}"', market_prompt)

    @patch("ai_insights._request_nim_json")
    def test_nim_retries_only_incomplete_ticker_and_merges_result(self, mock_request):
        items = [_stock_input("AAA"), _stock_input("BBB")]
        mock_request.side_effect = [
            {"items": [_nim_stock_item("AAA")]},
            {"items": [_nim_stock_item("BBB")]},
        ]

        entries = _fallback_stock_entries(
            items, "2026-07-24", date(2026, 6, 24), date(2026, 7, 24)
        )

        self.assertEqual(set(entries), {"AAA", "BBB"})
        retry_prompt = mock_request.call_args_list[1].args[2]
        self.assertIn('"ticker": "BBB"', retry_prompt)
        self.assertNotIn('"ticker": "AAA"', retry_prompt)

    @patch("ai_insights._request_nim_json")
    def test_nim_retry_failure_preserves_first_valid_ticker(self, mock_request):
        items = [_stock_input("AAA"), _stock_input("BBB")]
        mock_request.side_effect = [
            {"items": [_nim_stock_item("AAA")]},
            RuntimeError("retry failed"),
        ]

        entries = _fallback_stock_entries(
            items, "2026-07-24", date(2026, 6, 24), date(2026, 7, 24)
        )

        self.assertEqual(set(entries), {"AAA"})

    def test_market_response_reports_exact_missing_fields(self):
        with self.assertRaisesRegex(
            ValueError, "observation, interpretation"
        ):
            _build_market_summary(
                {"headline": "시장 요약"}, [], "NVIDIA NIM GPT-OSS 120B"
            )

    def test_market_response_rejects_null_and_non_string_fields(self):
        with self.assertRaisesRegex(
            ValueError, "headline, observation, interpretation"
        ):
            _build_market_summary(
                {
                    "headline": None,
                    "observation": 123,
                    "interpretation": ["해석"],
                },
                [],
                "NVIDIA NIM GPT-OSS 120B",
            )

    def test_gemini_only_disables_global_request_errors(self):
        self.assertTrue(
            _is_non_retryable_gemini_error(
                RuntimeError(
                    "Gemini HTTP 400: Invalid value at "
                    "'generation_config.response_format.text.mime_type'"
                )
            )
        )
        self.assertFalse(
            _is_non_retryable_gemini_error(
                RuntimeError("Gemini HTTP 400: batch payload is too large")
            )
        )

    @patch("ai_insights._fallback_market_summary")
    @patch("ai_insights._request_gemini_json")
    def test_market_skips_gemini_after_common_stock_request_error(
        self, mock_gemini, mock_fallback
    ):
        sources = [
            _article(str(index), f"Market article {index}", "Reuters", "2026-07-24")
            for index in range(3)
        ]
        mock_fallback.return_value = {"provider": "NVIDIA NIM GPT-OSS 120B"}

        result = _research_market_summary(
            {"headline": "기본 요약"},
            sources,
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            "Gemini HTTP 400: Invalid value at "
            "'generation_config.response_format.text.mime_type'",
        )

        mock_gemini.assert_not_called()
        mock_fallback.assert_called_once()
        self.assertEqual(result["provider"], "NVIDIA NIM GPT-OSS 120B")

    @patch("ai_insights._fallback_stock_entries")
    @patch("ai_insights._request_gemini_json")
    @patch("ai_insights._collect_company_news")
    @patch("ai_insights._save_cache")
    @patch("ai_insights._load_cache")
    def test_gemini_http_400_disables_only_later_gemini_batches(
        self,
        mock_load_cache,
        _mock_save_cache,
        mock_collect_news,
        mock_gemini,
        mock_fallback,
    ):
        data_date = "2026-07-24"
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        mock_load_cache.return_value = {
            f"{AI_INSIGHTS_CACHE_VERSION}:market:{data_date}": {
                "headline": "시장 요약",
                "observation": "관측",
                "interpretation": "해석",
                "disclaimer": "면책",
                "source_urls": [],
                "source_titles": [],
            }
        }
        mock_collect_news.return_value = (
            {ticker: [] for ticker in tickers},
            {
                ticker: {
                    "finnhub_collected": 0,
                    "finnhub_filter_passed": 0,
                    "finnhub_selected": 0,
                    "finnhub_status": "ok",
                }
                for ticker in tickers
            },
        )
        mock_gemini.side_effect = RuntimeError(
            "Gemini HTTP 400: Invalid value at "
            "'generation_config.response_format.text.mime_type'"
        )
        mock_fallback.side_effect = lambda batch, *_args: {
            str(item["ticker"]): {
                "business_summary": f"{item['ticker']} 사업",
                "move_reason": "제한 문구",
                "source_urls": [],
                "source_titles": [],
                "provider": "NVIDIA NIM GPT-OSS 120B (Finnhub 근거 부족)",
                "model_verdict": "limited",
            }
            for item in batch
        }
        stocks = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "name": f"{ticker} Inc.",
                    "sector": "Information Technology",
                    "return_1d": 1.0,
                    "business_summary": f"{ticker} business",
                }
                for ticker in tickers
            ]
        )

        enrich_with_ai(stocks, data_date, {"headline": "기본 요약"})

        self.assertEqual(mock_gemini.call_count, 1)
        self.assertEqual(mock_fallback.call_count, 2)

    def test_enrich_passes_cached_stock_sources_to_dataframe(self):
        data_date = "2026-07-23"
        cache = {
            f"{AI_INSIGHTS_CACHE_VERSION}:{data_date}:ABC": {
                "business_summary": "예시 사업",
                "move_reason": "가이던스를 상향했습니다.",
                "source_urls": ["https://example.com/100"],
                "source_titles": ["ABC raises guidance"],
            },
            f"{AI_INSIGHTS_CACHE_VERSION}:market:{data_date}": {
                "headline": "시장 요약",
                "observation": "관측",
                "interpretation": "해석",
                "disclaimer": "면책",
                "source_urls": [],
                "source_titles": [],
            },
        }
        stocks = pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "name": "Example",
                    "sector": "Technology",
                    "return_1d": 1.0,
                    "business_summary": "Example business",
                }
            ]
        )

        with patch("ai_insights._load_cache", return_value=cache):
            enriched, _ = enrich_with_ai(stocks, data_date, {"headline": "기본 요약"})

        self.assertEqual(enriched.loc[0, "source_urls"], ["https://example.com/100"])
        self.assertEqual(enriched.loc[0, "source_titles"], ["ABC raises guidance"])

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
