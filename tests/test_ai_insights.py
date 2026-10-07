from datetime import date, datetime, timezone
import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

from ai_insights import (
    _build_market_summary,
    _finnhub_get,
    _market_prompt,
    _normalise_company_articles,
    _normalise_stock_batch,
    _parse_json,
    _request_glm_json,
    _select_market_articles,
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


def _stock_input(ticker: str, articles: list[dict] | None = None) -> dict:
    return {
        "ticker": ticker,
        "name": f"{ticker} Inc.",
        "sector": "Information Technology",
        "return_1d": 1.0,
        "business_source_en": f"{ticker} business",
        "selected_finnhub_articles": articles or [],
    }


def _stock_response(ticker: str, status: str = "limited") -> dict:
    return {
        "ticker": ticker,
        "business_ko": f"{ticker} 사업을 영위하는 기업",
        "move_reason_ko": "",
        "evidence_status": status,
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
    return {
        "direct_evidence": direct,
        "historical_context": [],
        "rag_status": "ready",
        "retriever_version": "test-v1",
        "market_close_cutoff": "2026-07-24T16:00:00-04:00",
        "news_cutoff": "2026-07-24T22:00:00+00:00",
        "retrieval_as_of": "2026-07-27T01:00:00+00:00",
        "corpus_status": {"document_count": 3},
    }


class AiInsightsTest(unittest.TestCase):
    def test_glm_request_uses_nvidia_nim_json_contract(self):
        response = Mock(ok=True)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"items": []}'}}]
        }

        with (
            patch("ai_insights.requests.post", return_value=response) as mock_post,
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
        ):
            generated = _request_glm_json("prompt")

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )
        self.assertEqual(payload["model"], "z-ai/glm-5.3")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 1800)
        self.assertEqual(generated, {"items": []})

    def test_glm_timeout_is_not_retried(self):
        with (
            patch(
                "ai_insights.requests.post",
                side_effect=requests.ReadTimeout("slow response"),
            ) as mock_post,
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
            self.assertRaises(requests.ReadTimeout),
        ):
            _request_glm_json("prompt")

        self.assertEqual(mock_post.call_count, 1)

    def test_glm_request_retries_empty_response_with_repair_prompt(self):
        empty_response = Mock(ok=True)
        empty_response.json.return_value = {
            "choices": [{"message": {"content": ""}}],
        }
        valid_response = Mock(ok=True)
        valid_response.json.return_value = {
            "choices": [{"message": {"content": '{"items": []}'}}],
        }

        with (
            patch(
                "ai_insights.requests.post",
                side_effect=[empty_response, valid_response],
            ) as mock_post,
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
        ):
            generated = _request_glm_json("prompt")

        self.assertEqual(generated, {"items": []})
        self.assertEqual(mock_post.call_count, 2)
        retry_prompt = mock_post.call_args_list[1].kwargs["json"]["messages"][-1]["content"]
        self.assertIn("JSON 객체만", retry_prompt)

    def test_glm_request_redacts_no_secret_in_errors(self):
        response = Mock(ok=False, status_code=401, text="token=test-key")
        with (
            patch("ai_insights.requests.post", return_value=response),
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
            self.assertRaises(RuntimeError) as captured,
        ):
            _request_glm_json("prompt")
        self.assertIn("GLM 5.3 HTTP 401", str(captured.exception))

    @patch("ai_insights.requests.get")
    def test_finnhub_request_error_does_not_expose_api_key(self, mock_get):
        mock_get.side_effect = requests.ConnectionError(
            "failed https://finnhub.io/api/v1/news?token=secret-key"
        )
        with (
            patch.dict(os.environ, {"FINNHUB_API_KEY": "secret-key"}),
            self.assertRaises(RuntimeError) as captured,
        ):
            _finnhub_get("news", {"category": "general"})
        self.assertNotIn("secret-key", str(captured.exception))

    def test_json_parser_repairs_closed_container(self):
        self.assertEqual(_parse_json('{"items":[{"ticker":"AAA"}'), {
            "items": [{"ticker": "AAA"}],
        })

    def test_workflow_news_cutoff_is_kst_seven(self):
        self.assertEqual(
            _workflow_news_cutoff(
                "2026-07-24",
                datetime(2026, 7, 24, 23, tzinfo=timezone.utc),
            ),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )

    def test_company_articles_require_url_ticker_and_cutoff(self):
        valid = {
            "id": "1",
            "headline": "AAA raises guidance",
            "summary": "AAA raises guidance",
            "source": "Reuters",
            "datetime": int(datetime(2026, 7, 24, 15, tzinfo=timezone.utc).timestamp()),
            "url": "https://example.com/1",
            "related": "AAA",
        }
        invalid = {**valid, "id": "2", "related": "BBB"}
        selected, passed = _normalise_company_articles(
            "AAA",
            [valid, invalid],
            date(2026, 6, 24),
            date(2026, 7, 24),
            datetime(2026, 7, 24, 22, tzinfo=timezone.utc),
        )
        self.assertEqual(passed, 1)
        self.assertEqual([item["article_id"] for item in selected], ["1"])

    def test_stock_validation_rejects_missing_and_unexpected_tickers(self):
        items = [_stock_input("AAA"), _stock_input("BBB")]
        generated = {"items": [_stock_response("AAA"), _stock_response("LEAK")]}
        valid = _valid_stock_response_items(generated, items, attempt=1)
        self.assertEqual(set(valid), {"AAA"})

    def test_stock_normalisation_verifies_direct_catalyst(self):
        source = _article("1", "AAA raises guidance", "Reuters", "2026-07-24")
        item = _stock_input("AAA", [source])
        generated = {
            "items": [{
                "ticker": "AAA",
                "business_ko": "예시 사업 기업",
                "move_reason_ko": "가이던스를 상향했습니다.",
                "evidence_status": "verified",
            }]
        }
        result = _normalise_stock_batch(generated, [item], "GLM 5.3 + Finnhub")
        self.assertEqual(result["AAA"]["model_verdict"], "verified")
        self.assertEqual(result["AAA"]["provider"], "GLM 5.3 + Finnhub")

    def test_stock_evidence_outcomes_distinguish_display_states(self):
        base = {
            "finnhub_status": "ok",
            "finnhub_selected": 3,
            "finnhub_post_close_selected": 0,
        }
        self.assertEqual(_stock_evidence_outcome(base, "verified"), "verified_direct_catalyst")
        self.assertEqual(_stock_evidence_outcome(base, "limited"), "related_news_no_direct_catalyst")
        self.assertEqual(
            _stock_evidence_outcome({**base, "finnhub_selected": 0}, "limited"),
            "no_eligible_articles",
        )
        self.assertEqual(
            _stock_evidence_outcome({**base, "finnhub_status": "error:RuntimeError"}, "limited"),
            "generation_failure",
        )

    def test_post_close_verified_reason_is_rejected(self):
        source = _article("1", "AAA raises guidance", "Reuters", "2026-07-24")
        source["session_phase"] = "post_close"
        item = _stock_input("AAA", [source])
        generated = {
            "items": [{
                "ticker": "AAA",
                "business_ko": "예시 사업 기업",
                "move_reason_ko": "가이던스를 상향했습니다.",
                "evidence_status": "verified",
            }]
        }
        self.assertEqual(_valid_stock_response_items(generated, [item], 1), {})

    def test_stock_prompt_requires_all_tickers_and_timing_rules(self):
        prompt = _stock_prompt(
            [_stock_input("AAA"), _stock_input("BBB")],
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
        )
        self.assertIn('"expected_count": 2', prompt)
        self.assertIn('"expected_tickers": ["AAA", "BBB"]', prompt)
        self.assertIn("post_close", prompt)
        self.assertIn("정규장 등락의 원인으로 표현하지", prompt)

    def test_stock_prompt_compacts_article_summaries(self):
        article = _article("1", "AAA raises guidance", "Reuters", "2026-07-24")
        article["summary"] = "x" * 900
        prompt = _stock_prompt(
            [_stock_input("AAA", [article])],
            "2026-07-24",
            date(2026, 6, 24),
            date(2026, 7, 24),
        )
        self.assertNotIn("x" * 401, prompt)

    def test_market_prompt_contains_korea_scenario_and_evidence_rules(self):
        prompt = _market_prompt(
            {"headline": "기본 요약"},
            _market_retrieval(),
            "2026-07-24",
        )
        self.assertIn("korea_market_scenario", prompt)
        self.assertIn("direct_evidence_ids", prompt)
        self.assertIn("그날 정규장 움직임의 원인으로", prompt)

    def test_market_selection_requires_regular_session_evidence(self):
        articles = _market_retrieval()["direct_evidence"]
        for index, article in enumerate(articles):
            article["headline"] = "S&P 500 stocks react to earnings"
            article["summary"] = article["headline"]
            article["source"] = ("Reuters", "AP", "Bloomberg")[index]
        selected = _select_market_articles(
            articles,
            date(2026, 7, 24),
        )
        self.assertEqual(len(selected), 3)

    def test_market_summary_validates_required_fields_and_provider(self):
        generated = {
            "headline": "시장 요약",
            "observation": "상승 종목이 우세했습니다.",
            "interpretation": "직접 근거에 한정한 해석입니다.",
            "recent_context": "",
            "korea_market_scenario": {
                "session_date": "2026-07-27",
                "base_case": "조건부 시나리오",
                "positive_conditions": ["조건"],
                "risk_conditions": ["위험"],
                "watch_items": ["금리"],
            },
            "direct_evidence_ids": ["D1", "D2", "D3"],
            "context_evidence_ids": [],
        }
        summary = _build_market_summary(
            generated,
            _market_retrieval(),
            "GLM 5.3 + Finnhub",
        )
        self.assertEqual(summary["provider"], "GLM 5.3 + Finnhub")
        self.assertEqual(summary["fallback_stage"], "none")
        self.assertEqual(summary["direct_evidence_ids"], ["D1", "D2", "D3"])

    def test_cache_version_is_glm_specific(self):
        self.assertIn("glm-5.3", AI_INSIGHTS_CACHE_VERSION)


if __name__ == "__main__":
    unittest.main()
