import copy
from datetime import date, datetime, timezone
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ai_insights import (
    enrich_with_ai,
    _build_market_summary,
    _fallback_market_prompt,
    _fallback_stock_entries,
    _fallback_stock_prompt,
    _finnhub_get,
    _is_non_retryable_gemini_error,
    _legacy_market_retrieval,
    _market_retrieval_fingerprint,
    _market_prompt,
    _normalise_company_articles,
    _normalise_stock_batch,
    _parse_json,
    _request_gemini_json,
    _request_nim_json,
    _research_market_summary,
    _retrieve_market_evidence,
    _select_market_articles,
    _stock_cache_key,
    _stock_evidence_outcome,
    _stock_prompt,
    _valid_stock_response_items,
    _workflow_news_cutoff,
)
from config import AI_INSIGHTS_CACHE_VERSION


def _article(article_id: str, headline: str, source: str, published_date: str) -> dict:
    return {
        "article_id": article_id,
        "headline": headline,
        "summary": headline,
        "source": source,
        "published_at": f"{published_date}T15:00:00-04:00",
        "published_date": published_date,
        "url": f"https://example.com/{article_id}",
        "related": [],
    }


def _market_retrieval() -> dict:
    direct = [
        {
            **_article(str(index), f"Direct market article {index}", "Reuters", "2026-07-24"),
            "evidence_id": f"D{index}",
            "session_phase": "regular_session",
        }
        for index in range(1, 4)
    ]
    context = [
        {
            **_article("10", "Recent inflation and sector context", "AP", "2026-07-20"),
            "evidence_id": "C1",
        }
    ]
    return {
        "direct_evidence": direct,
        "historical_context": context,
        "rag_status": "ready",
        "retriever_version": "test-v1",
        "market_close_cutoff": "2026-07-24T16:00:00-04:00",
        "news_cutoff": "2026-07-24T22:00:00+00:00",
        "retrieval_as_of": "2026-07-27T01:00:00+00:00",
        "corpus_status": {"document_count": 4},
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


def _korea_scenario(session_date: str = "2026-07-27") -> dict:
    return {
        "session_date": session_date,
        "base_case": "미국장 흐름을 반영한 조건부 기본 시나리오",
        "positive_conditions": ["위험선호 회복 확인"],
        "risk_conditions": ["약세 지속 여부 확인"],
        "watch_items": ["반도체와 금리"],
    }


class AiInsightsTest(unittest.TestCase):
    def test_stock_evidence_outcome_distinguishes_display_states(self):
        base = {
            "finnhub_status": "ok",
            "finnhub_selected": 3,
            "finnhub_post_close_selected": 0,
        }
        self.assertEqual(
            _stock_evidence_outcome(base, "verified"),
            "verified_direct_catalyst",
        )
        self.assertEqual(
            _stock_evidence_outcome(base, "limited"),
            "related_news_no_direct_catalyst",
        )
        self.assertEqual(
            _stock_evidence_outcome(
                {**base, "finnhub_post_close_selected": 3},
                "limited",
            ),
            "post_close_only",
        )
        self.assertEqual(
            _stock_evidence_outcome(
                {**base, "finnhub_selected": 0},
                "limited",
            ),
            "no_eligible_articles",
        )
        self.assertEqual(
            _stock_evidence_outcome(
                {**base, "finnhub_status": "error:RuntimeError"},
                "limited",
            ),
            "generation_failure",
        )

    def test_workflow_news_cutoff_is_kst_0700_and_never_future(self):
        self.assertEqual(
            _workflow_news_cutoff(
                "2026-07-24",
                datetime(2026, 7, 24, 23, tzinfo=timezone.utc),
            ),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )
        self.assertEqual(
            _workflow_news_cutoff(
                "2026-07-24",
                datetime(2026, 7, 24, 21, tzinfo=timezone.utc),
            ),
            datetime(2026, 7, 24, 21, tzinfo=timezone.utc),
        )

    @patch("ai_insights.requests.get")
    def test_finnhub_request_error_does_not_expose_api_key(self, mock_get):
        mock_get.side_effect = requests.ConnectionError(
            "failed https://finnhub.io/api/v1/news?token=secret-key"
        )

        with (
            patch.dict("os.environ", {"FINNHUB_API_KEY": "secret-key"}),
            self.assertRaises(RuntimeError) as captured,
        ):
            _finnhub_get("news", {"category": "general"})

        message = str(captured.exception)
        self.assertNotIn("secret-key", message)
        self.assertNotIn("token=", message)

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
    def test_gemini_logs_incomplete_json_boundary_and_missing_ticker(self, mock_post):
        response = Mock(ok=True)
        response.json.return_value = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": '{"items":[{"ticker":"AAA"}'}]
                    },
                }
            ]
        }
        mock_post.return_value = response

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            self.assertLogs("ai_insights", level="INFO") as captured,
        ):
            generated = _request_gemini_json(
                "prompt",
                {"type": "OBJECT"},
                expected_tickers=["AAA", "BBB"],
            )

        self.assertEqual(generated["items"], [{"ticker": "AAA"}])
        logs = "\n".join(captured.output)
        self.assertIn("outcome=incomplete_json_boundary", logs)
        self.assertIn("closed_containers=2", logs)
        self.assertIn("expected=AAA,BBB", logs)
        self.assertIn("returned=AAA", logs)
        self.assertIn("missing=BBB", logs)

    @patch("ai_insights.requests.post")
    def test_gemini_logs_model_omission_for_complete_json(self, mock_post):
        response = Mock(ok=True)
        response.json.return_value = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": '{"items":[{"ticker":"AAA"}]}'}]
                    },
                }
            ]
        }
        mock_post.return_value = response

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            self.assertLogs("ai_insights", level="INFO") as captured,
        ):
            _request_gemini_json(
                "prompt",
                {"type": "OBJECT"},
                expected_tickers=["AAA", "BBB"],
            )

        logs = "\n".join(captured.output)
        self.assertIn("outcome=model_omitted_expected_ticker", logs)
        self.assertIn("closed_containers=0", logs)
        self.assertIn("missing=BBB", logs)

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

    @patch("ai_insights.requests.post")
    def test_nim_recognises_market_response_keys(self, mock_post):
        response = Mock(ok=True)
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"headline":"Market close","observation":"Breadth improved",'
                            '"interpretation":"Rates helped","recent_context":"",'
                            '"direct_evidence_ids":["D1","D2","D3"],'
                            '"context_evidence_ids":[]}'
                        ),
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

        self.assertEqual(generated["headline"], "Market close")
        logs = "\n".join(captured.output)
        self.assertIn("unknown_key_count=0", logs)
        for field in (
            "headline",
            "observation",
            "interpretation",
            "recent_context",
            "direct_evidence_ids",
            "context_evidence_ids",
        ):
            self.assertIn(field, logs)

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
            {}, _market_retrieval(), "2026-07-24"
        )

        self.assertIn('{"items":[', stock_prompt)
        for field in ("ticker", "business_ko", "move_reason_ko", "evidence_status"):
            self.assertIn(f'"{field}"', stock_prompt)
        self.assertIn("post_close", stock_prompt)
        self.assertIn("정규장 등락의 원인으로 표현하지", stock_prompt)
        for field in (
            "headline",
            "observation",
            "interpretation",
            "recent_context",
            "direct_evidence_ids",
            "context_evidence_ids",
        ):
            self.assertIn(f'"{field}"', market_prompt)
        self.assertIn("post_close", market_prompt)
        self.assertIn("정규장 움직임의 원인으로 표현하지", market_prompt)

    def test_gemini_stock_prompt_requires_each_expected_ticker_once(self):
        prompt = _stock_prompt(
            [_stock_input("AAA"), _stock_input("BBB")],
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
        )

        self.assertIn('"expected_count": 2', prompt)
        self.assertIn('"expected_tickers": ["AAA", "BBB"]', prompt)
        self.assertIn("모든 ticker를 입력 순서대로 정확히 한 번씩", prompt)
        self.assertIn("배열 길이는 반드시 expected_count와 같아야", prompt)

    def test_market_prompt_structures_korea_session_scenario(self):
        prompt = _market_prompt(
            {"headline": "기본 요약"},
            _market_retrieval(),
            "2026-07-24",
        )

        self.assertIn('"korea_session_date": "2026-07-27"', prompt)
        self.assertIn("korea_market_scenario", prompt)
        self.assertIn("positive_conditions", prompt)
        self.assertIn("risk_conditions", prompt)
        self.assertIn("한국 개별 종목 추천", prompt)

    def test_nim_market_prompt_uses_production_evidence_ids(self):
        retrieval = _market_retrieval()
        for index, article in enumerate(retrieval["direct_evidence"], start=101):
            article["evidence_id"] = f"finnhub:{index}"
        retrieval["historical_context"][0]["evidence_id"] = "finnhub:201"

        prompt = _fallback_market_prompt(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        for evidence_id in (
            "finnhub:101",
            "finnhub:102",
            "finnhub:103",
            "finnhub:201",
        ):
            self.assertIn(f'"{evidence_id}"', prompt)
        self.assertNotIn('"D1"', prompt)
        self.assertNotIn('"C1"', prompt)

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

    def test_nim_rejects_post_close_reason_without_timing_label(self):
        source = _article(
            "100",
            "AAA raises guidance",
            "Reuters",
            "2026-07-24",
        )
        source["session_phase"] = "post_close"
        expected = _stock_input("AAA")
        expected["selected_finnhub_articles"] = [source]
        generated = {
            "items": [
                {
                    "ticker": "AAA",
                    "business_ko": "예시 사업을 영위하는 기업",
                    "move_reason_ko": "가이던스를 상향 발표했습니다.",
                    "evidence_status": "verified",
                }
            ]
        }

        valid = _valid_stock_response_items(
            generated,
            [expected],
            attempt=1,
        )

        self.assertEqual(valid, {})

    def test_market_response_reports_exact_missing_fields(self):
        with self.assertRaisesRegex(
            ValueError, "observation, interpretation, recent_context"
        ):
            _build_market_summary(
                {"headline": "시장 요약"}, [], "NVIDIA NIM GPT-OSS 120B"
            )

    def test_market_response_rejects_null_and_non_string_fields(self):
        with self.assertRaisesRegex(
            ValueError, "headline, observation, interpretation, recent_context"
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

    def test_market_response_keeps_direct_and_context_sources_separate(self):
        generated = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "최근 물가와 섹터 맥락",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["D1", "D2", "D3"],
            "context_evidence_ids": ["C1"],
        }

        summary = _build_market_summary(
            generated, _market_retrieval(), "Gemini + Finnhub RAG"
        )

        self.assertEqual(summary["direct_evidence_ids"], ["D1", "D2", "D3"])
        self.assertEqual(summary["context_evidence_ids"], ["C1"])
        self.assertEqual(len(summary["direct_source_urls"]), 3)
        self.assertEqual(len(summary["context_source_urls"]), 1)
        self.assertEqual(summary["rag_status"], "rag_success")
        self.assertEqual(
            summary["korea_market_scenario"]["session_date"],
            "2026-07-27",
        )

    def test_market_response_rejects_context_id_as_direct_evidence(self):
        generated = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "최근 물가와 섹터 맥락",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["C1", "D2", "D3"],
            "context_evidence_ids": ["C1"],
        }

        with self.assertRaisesRegex(ValueError, "입력에 없는 근거 ID"):
            _build_market_summary(
                generated, _market_retrieval(), "Gemini + Finnhub RAG"
            )

    def test_direct_only_market_response_requires_empty_recent_context(self):
        retrieval = _market_retrieval()
        retrieval["historical_context"] = []
        generated = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["D1", "D2", "D3"],
            "context_evidence_ids": [],
        }

        summary = _build_market_summary(
            generated, retrieval, "Gemini + Finnhub RAG"
        )

        self.assertEqual(summary["recent_context"], "")
        self.assertEqual(summary["context_evidence_ids"], [])
        self.assertEqual(summary["rag_status"], "direct_only")

        generated["recent_context"] = "출처 없는 최근 맥락"
        with self.assertRaisesRegex(ValueError, "근거 없는 최근 맥락"):
            _build_market_summary(
                generated, retrieval, "Gemini + Finnhub RAG"
            )

    def test_market_response_rejects_fewer_than_three_direct_citations(self):
        generated = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "최근 물가와 섹터 맥락",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["D1", "D2"],
            "context_evidence_ids": ["C1"],
        }

        with self.assertRaisesRegex(ValueError, "직접 근거가 기준보다 적습니다"):
            _build_market_summary(
                generated,
                _market_retrieval(),
                "Gemini + Finnhub RAG",
            )

    def test_market_response_requires_three_regular_session_citations(self):
        retrieval = _market_retrieval()
        retrieval["direct_evidence"][2]["session_phase"] = "post_close"
        generated = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "최근 물가와 섹터 맥락",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["D1", "D2", "D3"],
            "context_evidence_ids": ["C1"],
        }

        with self.assertRaisesRegex(ValueError, "정규장 직접 근거가 기준보다 적습니다"):
            _build_market_summary(
                generated,
                retrieval,
                "Gemini + Finnhub RAG",
            )

    def test_limited_market_summary_marks_rag_as_attempted(self):
        retrieval = _market_retrieval()
        retrieval["direct_evidence"] = retrieval["direct_evidence"][:2]

        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        self.assertTrue(summary["rag_attempted"])
        self.assertEqual(summary["fallback_stage"], "rule_based")
        self.assertEqual(summary["rag_status"], "insufficient_direct_evidence")

        retrieval["direct_evidence"] = []
        retrieval["rag_status"] = "unavailable"
        retrieval["corpus_status"] = {"status": "corrupt"}
        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )
        self.assertEqual(summary["rag_status"], "corpus_corrupt")

    @patch("ai_insights._fallback_market_summary")
    @patch("ai_insights._request_gemini_json")
    def test_post_close_only_market_evidence_skips_ai_generation(
        self, mock_gemini, mock_fallback
    ):
        retrieval = _market_retrieval()
        for source in retrieval["direct_evidence"]:
            source["session_phase"] = "post_close"

        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        mock_gemini.assert_not_called()
        mock_fallback.assert_not_called()
        self.assertEqual(summary["fallback_stage"], "rule_based")
        self.assertEqual(summary["rag_status"], "insufficient_direct_evidence")

    @patch("ai_insights._collect_market_news")
    def test_legacy_market_retrieval_excludes_old_articles_from_direct_evidence(
        self, mock_collect
    ):
        post_close = _article(
            "post-close",
            "S&P 500 futures react to Federal Reserve comments",
            "CNBC",
            "2026-07-24",
        )
        post_close["published_at"] = "2026-07-24T17:00:00-04:00"
        after_cutoff = _article(
            "after-cutoff",
            "Wall Street outlook changes late in the evening",
            "Bloomberg",
            "2026-07-24",
        )
        after_cutoff["published_at"] = "2026-07-24T18:01:00-04:00"
        mock_collect.return_value = (
            [
                _article(
                    "old",
                    "S&P 500 Federal Reserve inflation market update",
                    "Reuters",
                    "2026-07-01",
                ),
                _article(
                    "recent-1",
                    "S&P 500 moves after Federal Reserve comments",
                    "Reuters",
                    "2026-07-24",
                ),
                _article(
                    "recent-2",
                    "Nasdaq stocks react to interest rate outlook",
                    "Reuters",
                    "2026-07-23",
                ),
                _article(
                    "recent-3",
                    "Treasury yields influence stock market sectors",
                    "AP",
                    "2026-07-22",
                ),
                post_close,
                after_cutoff,
            ],
            {"status": "ok", "error": ""},
        )

        retrieval = _legacy_market_retrieval(
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )

        ids = {
            article["article_id"]
            for article in retrieval["direct_evidence"]
        }
        self.assertEqual(
            ids,
            {"recent-1", "recent-2", "recent-3", "post-close"},
        )
        self.assertNotIn("old", ids)
        self.assertNotIn("after-cutoff", ids)
        phases = {
            article["article_id"]: article["session_phase"]
            for article in retrieval["direct_evidence"]
        }
        self.assertEqual(phases["post-close"], "post_close")
        self.assertEqual(
            retrieval["news_cutoff"],
            "2026-07-24T22:00:00+00:00",
        )

    @patch("ai_insights._collect_market_news")
    def test_legacy_market_retrieval_rejects_company_only_earnings(
        self, mock_collect
    ):
        mock_collect.return_value = (
            [
                _article(
                    f"company-{index}",
                    f"Acme {index} earnings guidance beats estimates",
                    source,
                    "2026-07-24",
                )
                for index, source in enumerate(("Reuters", "AP", "CNBC"), start=1)
            ],
            {"status": "ok", "error": ""},
        )

        retrieval = _legacy_market_retrieval(
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )

        self.assertEqual(retrieval["direct_evidence"], [])

    @patch("ai_insights._save_market_news_pool")
    @patch("ai_insights._load_market_news_pool")
    @patch("ai_insights._finnhub_get")
    def test_legacy_market_retrieval_uses_pool_when_live_fetch_fails(
        self, mock_finnhub, mock_load_pool, mock_save_pool
    ):
        mock_finnhub.return_value = {"error": "temporary outage"}
        mock_load_pool.return_value = [
            _article(
                "cached-1",
                "S&P 500 moves after Federal Reserve comments",
                "Reuters",
                "2026-07-24",
            ),
            _article(
                "cached-2",
                "Nasdaq stocks react to interest rate outlook",
                "AP",
                "2026-07-23",
            ),
            _article(
                "cached-3",
                "Treasury yields influence stock market sectors",
                "CNBC",
                "2026-07-22",
            ),
        ]

        with self.assertLogs("ai_insights", level="WARNING") as captured:
            retrieval = _legacy_market_retrieval(
                "2026-07-24",
                date(2026, 6, 24),
                date(2026, 7, 24),
                datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
            )

        self.assertEqual(len(retrieval["direct_evidence"]), 3)
        self.assertEqual(retrieval["corpus_status"]["status"], "degraded")
        self.assertEqual(retrieval["corpus_status"]["error"], "ValueError")
        self.assertIn("rolling pool만 사용", "\n".join(captured.output))
        mock_save_pool.assert_called_once()

    @patch("ai_insights._legacy_market_retrieval")
    @patch("market_rag.retrieve_market_context")
    def test_legacy_direct_fallback_preserves_rag_historical_context(
        self, mock_retrieve, mock_legacy
    ):
        rag = _market_retrieval()
        rag["direct_evidence"] = rag["direct_evidence"][:2]
        legacy = _market_retrieval()
        legacy["historical_context"] = []
        legacy["retriever_version"] = "legacy-keyword-v1"
        mock_retrieve.return_value = rag
        mock_legacy.return_value = legacy

        result = _retrieve_market_evidence(
            {"sector_returns": []},
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )

        self.assertEqual(
            [item["evidence_id"] for item in result["historical_context"]],
            ["C1"],
        )
        self.assertEqual(result["rag_status"], "hybrid")

    @patch("ai_insights._legacy_market_retrieval")
    @patch("market_rag.retrieve_market_context")
    def test_post_close_rag_evidence_does_not_skip_legacy_fallback(
        self, mock_retrieve, mock_legacy
    ):
        rag = _market_retrieval()
        rag["direct_evidence"][2]["session_phase"] = "post_close"
        legacy = _market_retrieval()
        legacy["historical_context"] = []
        legacy["retriever_version"] = "legacy-keyword-v1"
        mock_retrieve.return_value = rag
        mock_legacy.return_value = legacy

        result = _retrieve_market_evidence(
            {"sector_returns": []},
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )

        mock_legacy.assert_called_once()
        self.assertEqual(
            [item["evidence_id"] for item in result["historical_context"]],
            ["C1"],
        )
        self.assertEqual(result["rag_status"], "hybrid")

    @patch("ai_insights._legacy_market_retrieval")
    @patch("market_rag.retrieve_market_context")
    def test_legacy_fallback_preserves_rag_corpus_failure_status(
        self, mock_retrieve, mock_legacy
    ):
        rag = _market_retrieval()
        rag["direct_evidence"] = []
        rag["historical_context"] = []
        rag["rag_status"] = "unavailable"
        rag["corpus_status"] = {"ok": False, "status": "corrupt"}
        legacy = _market_retrieval()
        legacy["direct_evidence"] = []
        legacy["historical_context"] = []
        legacy["rag_status"] = "corpus_empty"
        legacy["retriever_version"] = "legacy-keyword-v1"
        legacy["corpus_status"] = {"fallback": "legacy_market_news_pool"}
        mock_retrieve.return_value = rag
        mock_legacy.return_value = legacy

        retrieval = _retrieve_market_evidence(
            {"sector_returns": []},
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )
        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        self.assertEqual(
            retrieval["corpus_status"]["rag"]["status"],
            "corrupt",
        )
        self.assertEqual(summary["rag_status"], "corpus_corrupt")

    @patch("ai_insights._legacy_market_retrieval")
    @patch("market_rag.retrieve_market_context")
    def test_legacy_fallback_preserves_unexpected_rag_retrieval_failure(
        self, mock_retrieve, mock_legacy
    ):
        mock_retrieve.side_effect = RuntimeError("unexpected retrieval failure")
        legacy = _market_retrieval()
        legacy["direct_evidence"] = []
        legacy["historical_context"] = []
        legacy["rag_status"] = "corpus_empty"
        legacy["retriever_version"] = "legacy-keyword-v1"
        legacy["corpus_status"] = {"fallback": "legacy_market_news_pool"}
        mock_legacy.return_value = legacy

        retrieval = _retrieve_market_evidence(
            {"sector_returns": []},
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )
        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        self.assertEqual(
            retrieval["corpus_status"]["rag"]["status"],
            "retrieval_failure",
        )
        self.assertEqual(summary["rag_status"], "corpus_retrieval_failure")

    def test_market_cache_fingerprint_tracks_prompt_data_and_article_content(self):
        retrieval = _market_retrieval()
        original = _market_retrieval_fingerprint(
            retrieval,
            {"breadth": {"advances": 300, "declines": 200}},
        )
        changed_article = copy.deepcopy(retrieval)
        changed_article["direct_evidence"][0]["summary"] = "updated summary"

        self.assertNotEqual(
            original,
            _market_retrieval_fingerprint(
                changed_article,
                {"breadth": {"advances": 300, "declines": 200}},
            ),
        )
        self.assertNotEqual(
            original,
            _market_retrieval_fingerprint(
                retrieval,
                {"breadth": {"advances": 250, "declines": 250}},
            ),
        )

    @patch("ai_insights._request_gemini_json")
    def test_research_labels_pure_legacy_retrieval_without_rag(
        self, mock_gemini
    ):
        retrieval = _market_retrieval()
        retrieval["historical_context"] = []
        retrieval["retriever_version"] = "legacy-keyword-v1"
        mock_gemini.return_value = {
            "headline": "시장 요약",
            "observation": "시장 폭 관측",
            "interpretation": "직접 기사에 근거한 해석",
            "recent_context": "",
            "korea_market_scenario": _korea_scenario(),
            "direct_evidence_ids": ["D1", "D2", "D3"],
            "context_evidence_ids": [],
        }

        summary = _research_market_summary(
            {"headline": "기본 요약"},
            retrieval,
            "2026-07-24",
        )

        self.assertEqual(summary["provider"], "Gemini + Finnhub")
        self.assertEqual(summary["rag_status"], "legacy_direct")

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
    @patch("ai_insights._retrieve_market_evidence")
    @patch("ai_insights._save_cache")
    @patch("ai_insights._load_cache")
    def test_gemini_http_400_disables_only_later_gemini_batches(
        self,
        mock_load_cache,
        _mock_save_cache,
        mock_retrieve_market,
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
        mock_retrieve_market.return_value = {
            "direct_evidence": [],
            "historical_context": [],
            "rag_status": "empty",
            "retriever_version": "test-v1",
            "retrieval_as_of": "2026-07-24T22:00:00Z",
        }
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

    @patch("ai_insights._fallback_stock_entries")
    @patch("ai_insights._request_gemini_json")
    @patch("ai_insights._collect_company_news")
    @patch("ai_insights._retrieve_market_evidence")
    @patch("ai_insights._save_cache")
    @patch("ai_insights._load_cache", return_value={})
    def test_gemini_valid_item_is_preserved_and_only_missing_ticker_falls_back(
        self,
        _mock_load_cache,
        _mock_save_cache,
        mock_retrieve_market,
        mock_collect_news,
        mock_gemini,
        mock_fallback,
    ):
        data_date = "2026-07-24"
        sources = {
            ticker: [
                {
                    **_article(
                        ticker,
                        f"{ticker} raises guidance",
                        "Reuters",
                        data_date,
                    ),
                    "session_phase": "regular_session",
                }
            ]
            for ticker in ("AAA", "BBB")
        }
        mock_collect_news.return_value = (
            sources,
            {
                ticker: {
                    "finnhub_collected": 1,
                    "finnhub_filter_passed": 1,
                    "finnhub_selected": 1,
                    "finnhub_post_close_selected": 0,
                    "finnhub_status": "ok",
                }
                for ticker in sources
            },
        )
        mock_gemini.return_value = {
            "items": [
                {
                    "ticker": "AAA",
                    "business_ko": "AAA 사업을 영위하는 기업",
                    "move_reason_ko": "AAA가 가이던스를 상향했습니다.",
                    "evidence_status": "verified",
                }
            ]
        }
        mock_fallback.return_value = {
            "BBB": {
                "business_summary": "BBB 사업을 영위하는 기업",
                "move_reason": "BBB가 가이던스를 상향했습니다.",
                "source_urls": ["https://example.com/BBB"],
                "source_titles": ["BBB raises guidance"],
                "provider": "NVIDIA NIM GPT-OSS 120B",
                "model_verdict": "verified",
            }
        }
        mock_retrieve_market.return_value = {
            "direct_evidence": [],
            "historical_context": [],
            "rag_status": "empty",
            "retriever_version": "test-v1",
            "retrieval_as_of": "2026-07-27T01:00:00Z",
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
                for ticker in ("AAA", "BBB")
            ]
        )

        with self.assertLogs("ai_insights", level="INFO") as captured:
            enriched, _ = enrich_with_ai(
                stocks,
                data_date,
                {"headline": "기본 요약"},
                retrieval_as_of=datetime(
                    2026,
                    7,
                    27,
                    1,
                    tzinfo=timezone.utc,
                ),
            )

        fallback_batch = mock_fallback.call_args.args[0]
        self.assertEqual(
            [item["ticker"] for item in fallback_batch],
            ["BBB"],
        )
        self.assertEqual(
            enriched.set_index("ticker").loc["AAA", "move_reason"],
            "AAA가 가이던스를 상향했습니다.",
        )
        self.assertEqual(
            enriched.set_index("ticker").loc["BBB", "move_reason"],
            "BBB가 가이던스를 상향했습니다.",
        )
        logs = "\n".join(captured.output)
        self.assertIn(
            "ticker=AAA",
            logs,
        )
        self.assertIn("GPT fallback=not_used", logs)
        self.assertIn("결과=verified_direct_catalyst", logs)

    @patch("ai_insights._fallback_stock_entries")
    @patch("ai_insights._request_gemini_json")
    @patch("ai_insights._collect_company_news")
    @patch("ai_insights._retrieve_market_evidence")
    @patch("ai_insights._save_cache")
    @patch("ai_insights._load_cache", return_value={})
    def test_gemini_post_close_timing_failure_uses_nim_fallback(
        self,
        _mock_load_cache,
        mock_save_cache,
        mock_retrieve_market,
        mock_collect_news,
        mock_gemini,
        mock_fallback,
    ):
        data_date = "2026-07-24"
        source = _article(
            "100",
            "AAA raises guidance",
            "Reuters",
            data_date,
        )
        source["session_phase"] = "post_close"
        mock_collect_news.return_value = (
            {"AAA": [source]},
            {
                "AAA": {
                    "finnhub_collected": 1,
                    "finnhub_filter_passed": 1,
                    "finnhub_selected": 1,
                    "finnhub_status": "ok",
                }
            },
        )
        mock_gemini.return_value = {
            "items": [
                {
                    "ticker": "AAA",
                    "business_ko": "예시 사업을 영위하는 기업",
                    "move_reason_ko": "가이던스를 상향 발표했습니다.",
                    "evidence_status": "verified",
                }
            ]
        }
        mock_fallback.return_value = {
            "AAA": {
                "business_summary": "예시 사업을 영위하는 기업",
                "move_reason": "장 마감 후 가이던스를 상향 발표했습니다.",
                "source_urls": ["https://example.com/100"],
                "source_titles": ["AAA raises guidance"],
                "provider": "NVIDIA NIM GPT-OSS 120B",
                "model_verdict": "verified",
            }
        }
        mock_retrieve_market.return_value = {
            "direct_evidence": [],
            "historical_context": [],
            "rag_status": "empty",
            "retriever_version": "test-v1",
            "retrieval_as_of": "2026-07-24T22:00:00Z",
        }
        stocks = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "name": "AAA Inc.",
                    "sector": "Information Technology",
                    "return_1d": 1.0,
                    "business_summary": "AAA business",
                }
            ]
        )

        enriched, _ = enrich_with_ai(
            stocks,
            data_date,
            {"headline": "기본 요약"},
            retrieval_as_of=datetime(
                2026,
                7,
                24,
                22,
                tzinfo=timezone.utc,
            ),
        )

        mock_fallback.assert_called_once()
        self.assertEqual(
            enriched.loc[0, "move_reason"],
            "관련 기사는 장 마감 후 발표되어 해당 거래일 정규장 등락 원인으로 사용할 수 없습니다.",
        )
        self.assertEqual(enriched.loc[0, "evidence_outcome"], "post_close_only")
        self.assertTrue(mock_save_cache.called)

    @patch("ai_insights._retrieve_market_evidence")
    @patch("ai_insights._request_gemini_json")
    @patch("ai_insights.requests.get")
    @patch("ai_insights._save_cache")
    @patch("ai_insights._load_cache", return_value={})
    def test_finnhub_dict_error_result_is_used_but_not_cached(
        self,
        _mock_load_cache,
        mock_save_cache,
        mock_get,
        mock_gemini,
        mock_retrieve_market,
    ):
        data_date = "2026-07-24"
        response = Mock(ok=True)
        response.json.return_value = {"error": "rate limit"}
        mock_get.return_value = response
        mock_gemini.return_value = {
            "items": [
                {
                    "ticker": "AAA",
                    "business_ko": "예시 사업을 영위하는 기업",
                    "move_reason_ko": "",
                    "evidence_status": "limited",
                }
            ]
        }
        mock_retrieve_market.return_value = {
            "direct_evidence": [],
            "historical_context": [],
            "rag_status": "empty",
            "retriever_version": "test-v1",
            "retrieval_as_of": "2026-07-24T22:00:00Z",
        }
        stocks = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "name": "AAA Inc.",
                    "sector": "Information Technology",
                    "return_1d": 1.0,
                    "business_summary": "AAA business",
                }
            ]
        )

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}):
            enriched, _ = enrich_with_ai(
                stocks,
                data_date,
                {"headline": "기본 요약"},
            )

        self.assertEqual(
            enriched.loc[0, "move_reason"],
            "관련 기사 판정 생성에 실패해 해당 거래일의 직접 촉매를 확인하지 못했습니다.",
        )
        self.assertEqual(enriched.loc[0, "evidence_outcome"], "generation_failure")
        mock_get.assert_called_once()
        mock_save_cache.assert_not_called()

    def test_enrich_passes_cached_stock_sources_to_dataframe(self):
        data_date = "2026-07-23"
        news_cutoff = datetime(2026, 7, 23, 22, tzinfo=timezone.utc)
        cache = {
            _stock_cache_key(data_date, "ABC", news_cutoff): {
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

        with (
            patch("ai_insights._load_cache", return_value=cache),
            patch(
                "ai_insights._retrieve_market_evidence",
                return_value={
                    "direct_evidence": [],
                    "historical_context": [],
                    "rag_status": "empty",
                    "retriever_version": "test-v1",
                    "retrieval_as_of": "2026-07-23T22:00:00Z",
                },
            ),
        ):
            enriched, _ = enrich_with_ai(
                stocks,
                data_date,
                {"headline": "기본 요약"},
                retrieval_as_of=news_cutoff,
            )

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

    def test_market_selection_reserves_three_pre_close_sources(self):
        end = date(2026, 7, 24)
        regular = [
            _article(
                f"regular-{index}",
                f"Stock market update {index}",
                f"Regular Source {index}",
                end.isoformat(),
            )
            for index in range(1, 4)
        ]
        post_close = [
            _article(
                f"post-{index}",
                (
                    "S&P 500 stock market reacts to Federal Reserve "
                    f"inflation and Treasury outlook {index}"
                ),
                f"Post Source {index}",
                end.isoformat(),
            )
            for index in range(1, 4)
        ]
        for article in post_close:
            article["published_at"] = "2026-07-24T17:00:00-04:00"

        selected = _select_market_articles(
            [*regular, *post_close],
            end,
            datetime(2026, 7, 24, 20, tzinfo=timezone.utc),
        )

        selected_ids = {article["article_id"] for article in selected}
        self.assertEqual(len(selected), 5)
        self.assertTrue(
            {article["article_id"] for article in regular}
            <= selected_ids
        )

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

    def test_post_close_stock_reason_requires_explicit_timing(self):
        source = _article(
            "100",
            "ABC raises revenue guidance",
            "Reuters",
            "2026-07-24",
        )
        source["session_phase"] = "post_close"
        items = [
            {
                "ticker": "ABC",
                "business_source_en": "Example business",
                "selected_finnhub_articles": [source],
            }
        ]
        generated = {
            "items": [
                {
                    "ticker": "ABC",
                    "business_ko": "예시 사업을 영위하는 기업",
                    "move_reason_ko": "매출 가이던스를 상향 발표했습니다.",
                    "evidence_status": "verified",
                }
            ]
        }

        invalid = _normalise_stock_batch(
            generated,
            items,
            "Gemini + Finnhub",
        )
        self.assertEqual(
            invalid["ABC"]["model_verdict"],
            "invalid_post_close_timing",
        )
        self.assertEqual(invalid["ABC"]["source_urls"], [])

        generated["items"][0][
            "move_reason_ko"
        ] = "장 마감 후 매출 가이던스를 상향 발표했습니다."
        valid = _normalise_stock_batch(
            generated,
            items,
            "Gemini + Finnhub",
        )
        self.assertEqual(valid["ABC"]["model_verdict"], "verified")
        self.assertEqual(valid["ABC"]["source_urls"], ["https://example.com/100"])

    def test_company_news_reports_pre_cap_filter_count(self):
        start = date(2026, 6, 23)
        end = date(2026, 7, 24)
        news_cutoff = datetime(2026, 7, 24, 22, tzinfo=timezone.utc)
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

        selected, passed_count = _normalise_company_articles(
            "ABC",
            raw_items,
            start,
            end,
            news_cutoff,
        )

        self.assertEqual(passed_count, 4)
        self.assertEqual(len(selected), 3)

    def test_company_news_prioritises_catalyst_over_newer_generic_article(self):
        start = date(2026, 6, 23)
        end = date(2026, 7, 24)
        news_cutoff = datetime(2026, 7, 24, 22, tzinfo=timezone.utc)
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

        selected, passed_count = _normalise_company_articles(
            "ABC",
            raw_items,
            start,
            end,
            news_cutoff,
        )

        self.assertEqual(passed_count, 4)
        self.assertEqual(selected[0]["article_id"], "2")
        self.assertNotIn("4", [article["article_id"] for article in selected])

    def test_company_news_includes_post_close_until_workflow_cutoff(self):
        start = date(2026, 6, 24)
        end = date(2026, 7, 24)
        news_cutoff = datetime(2026, 7, 24, 22, tzinfo=timezone.utc)

        def timestamp(day: int, hour: int) -> int:
            return int(
                datetime(
                    2026,
                    7,
                    day,
                    hour,
                    tzinfo=ZoneInfo("America/New_York"),
                ).timestamp()
            )

        raw_items = [
            {
                "id": "before-close",
                "headline": "ABC raises guidance",
                "url": "https://example.com/before",
                "datetime": timestamp(24, 15),
                "related": "ABC",
            },
            {
                "id": "after-close",
                "headline": "ABC announces acquisition",
                "url": "https://example.com/after",
                "datetime": timestamp(24, 17),
                "related": "ABC",
            },
            {
                "id": "next-day",
                "headline": "ABC reports earnings",
                "url": "https://example.com/next",
                "datetime": timestamp(25, 9),
                "related": "ABC",
            },
            {
                "id": "after-cutoff",
                "headline": "ABC signs a contract late in the evening",
                "url": "https://example.com/late",
                "datetime": timestamp(24, 19),
                "related": "ABC",
            },
        ]

        selected, passed_count = _normalise_company_articles(
            "ABC",
            raw_items,
            start,
            end,
            news_cutoff,
        )

        self.assertEqual(passed_count, 2)
        self.assertEqual(
            {item["article_id"] for item in selected},
            {"before-close", "after-close"},
        )
        self.assertEqual(
            {
                item["article_id"]: item["session_phase"]
                for item in selected
            },
            {
                "before-close": "regular_session",
                "after-close": "post_close",
            },
        )

    def test_json_repair_adds_missing_array_item_comma(self):
        malformed = '{"items":[{"ticker":"AAA"}\n{"ticker":"BBB"}]}'

        parsed = _parse_json(malformed)

        self.assertEqual([item["ticker"] for item in parsed["items"]], ["AAA", "BBB"])


if __name__ == "__main__":
    unittest.main()
