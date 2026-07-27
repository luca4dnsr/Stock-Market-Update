import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import requests

from market_rag import (
    MarketRagError,
    database_status,
    ingest_finnhub_news,
    init_db,
    main,
    quick_check,
    record_market_snapshot,
    retrieve_market_context,
    upsert_articles,
)


UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def article(
    article_id,
    headline,
    published_at,
    *,
    summary="",
    source="Reuters",
    url=None,
):
    return {
        "id": article_id,
        "headline": headline,
        "summary": summary,
        "source": source,
        "url": url or f"https://example.com/{article_id}",
        "published_at": published_at,
    }


class MarketRagTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "market_rag.sqlite3"

    def test_init_requires_fts5_and_creates_external_content_index(self):
        result = init_db(self.db_path)

        self.assertTrue(result["ok"])
        check = quick_check(self.db_path)
        self.assertEqual(check["result"], "ok")
        self.assertEqual(check["quick_check"], "ok")
        self.assertEqual(check["article_count"], 0)
        connection = sqlite3.connect(self.db_path)
        try:
            sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='articles_fts'"
            ).fetchone()[0]
            self.assertIn("fts5", sql.lower())
            self.assertIn("content='articles'", sql)
        finally:
            connection.close()

    def test_upsert_is_idempotent_and_deduplicates_by_id_url_and_content(self):
        seen = datetime(2026, 7, 24, 19, tzinfo=UTC)
        original = article(
            "a-1",
            "Federal Reserve keeps interest rate unchanged",
            "2026-07-24T18:00:00Z",
            summary="Stocks considered the FOMC decision.",
            url="https://example.com/rates?utm_source=test",
        )

        first = upsert_articles([original], self.db_path, seen_at=seen)
        second = upsert_articles([original], self.db_path, seen_at=seen)
        same_content = dict(
            original,
            id="different-id",
            url="https://another.example/same-story",
        )
        third = upsert_articles([same_content], self.db_path, seen_at=seen)
        same_url = dict(
            original,
            id="third-id",
            headline="Federal Reserve decision supports stocks",
        )
        fourth = upsert_articles([same_url], self.db_path, seen_at=seen)

        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["updated"], 1)
        self.assertEqual(third["article_count"], 1)
        self.assertEqual(fourth["article_count"], 1)
        self.assertGreaterEqual(third["deduplicated"], 1)

    def test_later_article_revision_does_not_leak_into_earlier_as_of(self):
        original = article(
            "revision",
            "Stock market watches Federal Reserve policy",
            "2026-07-24T18:00:00Z",
            summary="Original pre-close summary.",
        )
        revised = dict(
            original,
            headline="Federal Reserve surprises markets in a later revision",
            summary="Content added after the original observation.",
        )
        upsert_articles(
            [original],
            self.db_path,
            seen_at="2026-07-24T18:30:00Z",
        )
        upsert_articles(
            [revised],
            self.db_path,
            seen_at="2026-07-25T00:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T19:00:00Z",
            db_path=self.db_path,
        )

        selected = next(
            item
            for item in result["direct_evidence"]
            if item["article_id"] == "revision"
        )
        self.assertEqual(selected["headline"], original["headline"])
        self.assertEqual(selected["summary"], original["summary"])

    def test_retrieval_includes_post_close_until_workflow_cutoff(self):
        before_close = article(
            "valid",
            "Federal Reserve outlook lifts stock market",
            "2026-07-24T19:00:00Z",
        )
        after_close = article(
            "future-published",
            "Fed comments after the closing bell",
            "2026-07-24T20:30:00Z",
        )
        after_cutoff = article(
            "after-cutoff",
            "Wall Street outlook changes late in the evening",
            "2026-07-24T22:01:00Z",
        )
        late_seen = article(
            "future-seen",
            "Wall Street reacts to Federal Reserve policy",
            "2026-07-24T18:30:00Z",
        )
        company_only = article(
            "company-only",
            "Acme Software earnings revenue guidance beats estimates",
            "2026-07-24T18:00:00Z",
        )
        upsert_articles(
            [before_close, after_cutoff, company_only],
            self.db_path,
            seen_at="2026-07-24T19:30:00Z",
        )
        upsert_articles(
            [after_close],
            self.db_path,
            seen_at="2026-07-24T22:05:00Z",
        )
        upsert_articles(
            [late_seen],
            self.db_path,
            seen_at="2026-07-25T00:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:10:00Z",
            db_path=self.db_path,
        )
        ids = {item["article_id"] for item in result["direct_evidence"]}

        self.assertIn("valid", ids)
        self.assertIn("future-published", ids)
        self.assertNotIn("after-cutoff", ids)
        self.assertNotIn("future-seen", ids)
        self.assertNotIn("company-only", ids)
        self.assertEqual(result["market_close_cutoff"], "2026-07-24T20:00:00Z")
        self.assertEqual(result["news_cutoff"], "2026-07-24T22:00:00Z")
        phases = {
            item["article_id"]: item["session_phase"]
            for item in result["direct_evidence"]
        }
        self.assertEqual(phases["valid"], "regular_session")
        self.assertEqual(phases["future-published"], "post_close")

    def test_direct_retrieval_reserves_three_pre_close_sources(self):
        regular = [
            article(
                f"regular-{index}",
                f"Stock market update {index}",
                f"2026-07-24T18:0{index}:00Z",
                source=f"Regular Source {index}",
            )
            for index in range(1, 4)
        ]
        post_close = [
            article(
                f"post-{index}",
                (
                    "S&P 500 stock market reacts to Federal Reserve "
                    f"inflation and Treasury outlook {index}"
                ),
                f"2026-07-24T20:3{index}:00Z",
                source=f"Post Source {index}",
            )
            for index in range(1, 4)
        ]
        upsert_articles(
            [*regular, *post_close],
            self.db_path,
            seen_at="2026-07-24T21:30:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:00:00Z",
            db_path=self.db_path,
        )

        phases = [
            item["session_phase"]
            for item in result["direct_evidence"]
        ]
        self.assertEqual(len(phases), 5)
        self.assertEqual(phases.count("regular_session"), 3)
        self.assertEqual(phases.count("post_close"), 2)

    def test_retrieval_does_not_use_articles_published_after_retrieval_time(self):
        upsert_articles(
            [
                article(
                    "future-at-retrieval",
                    "Federal Reserve outlook moves stocks",
                    "2026-07-24T19:30:00Z",
                )
            ],
            self.db_path,
            seen_at="2026-07-24T18:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T19:00:00Z",
            db_path=self.db_path,
        )

        self.assertNotIn(
            "future-at-retrieval",
            {item["article_id"] for item in result["direct_evidence"]},
        )

    def test_monday_window_uses_kst_workflow_cutoff_across_dst(self):
        upsert_articles(
            [
                article(
                    "friday",
                    "Friday stock market reacts to Federal Reserve policy",
                    "2026-07-24T15:00:00Z",
                ),
                article(
                    "after-monday-close",
                    "Stocks move after Monday close",
                    "2026-07-27T20:01:00Z",
                ),
                article(
                    "after-workflow-cutoff",
                    "Stocks move late in the evening",
                    "2026-07-27T22:01:00Z",
                ),
            ],
            self.db_path,
            seen_at="2026-07-27T21:00:00Z",
        )

        summer = retrieve_market_context(
            "2026-07-27",
            retrieval_as_of="2026-07-27T22:00:00Z",
            db_path=self.db_path,
        )
        winter = retrieve_market_context(
            "2026-01-05",
            retrieval_as_of="2026-01-05T22:00:00Z",
            db_path=self.db_path,
        )
        ids = {item["article_id"] for item in summer["direct_evidence"]}

        self.assertIn("friday", ids)
        self.assertIn("after-monday-close", ids)
        self.assertNotIn("after-workflow-cutoff", ids)
        self.assertEqual(summer["market_close_cutoff"], "2026-07-27T20:00:00Z")
        self.assertEqual(winter["market_close_cutoff"], "2026-01-05T21:00:00Z")
        self.assertEqual(summer["news_cutoff"], "2026-07-27T22:00:00Z")
        self.assertEqual(winter["news_cutoff"], "2026-01-05T22:00:00Z")

    def test_retrieval_splits_direct_and_matching_historical_context(self):
        items = [
            article(
                "direct-fed",
                "Federal Reserve interest rate decision moves stock market",
                "2026-07-24T18:00:00Z",
                source="Reuters",
            ),
            article(
                "old-fed",
                "Federal Reserve discusses interest rate path",
                "2026-07-14T15:00:00Z",
                source="Bloomberg",
            ),
            article(
                "old-tech",
                "Semiconductor and software shares lead technology sector",
                "2026-07-10T15:00:00Z",
                source="CNBC",
            ),
            article(
                "old-company-earnings",
                "Nvidia semiconductor earnings guidance beats estimates",
                "2026-07-11T15:00:00Z",
                source="MarketWatch",
            ),
            article(
                "unrelated-health",
                "Biotech trial update draws health care attention",
                "2026-07-12T15:00:00Z",
                source="AP",
            ),
        ]
        upsert_articles(items, self.db_path, seen_at="2026-07-24T19:00:00Z")

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:00:00Z",
            sector_names=["Information Technology"],
            db_path=self.db_path,
        )
        direct_ids = {item["article_id"] for item in result["direct_evidence"]}
        historical_ids = {
            item["article_id"] for item in result["historical_context"]
        }

        self.assertEqual(result["rag_status"], "ok")
        self.assertIn("direct-fed", direct_ids)
        self.assertEqual(historical_ids, {"old-fed", "old-tech"})
        self.assertTrue(direct_ids.isdisjoint(historical_ids))
        for item in result["direct_evidence"]:
            self.assertEqual(
                {
                    "evidence_id",
                    "article_id",
                    "published_at",
                    "published_date",
                    "headline",
                    "summary",
                    "source",
                    "url",
                    "tags",
                    "session_phase",
                },
                set(item),
            )
        for item in result["historical_context"]:
            self.assertEqual(
                {
                    "evidence_id",
                    "article_id",
                    "published_at",
                    "published_date",
                    "headline",
                    "summary",
                    "source",
                    "url",
                    "tags",
                },
                set(item),
            )

    def test_direct_retrieval_enforces_per_source_diversity_cap(self):
        items = [
            article(
                f"reuters-{index}",
                f"Federal Reserve interest rate moves stock market {index}",
                f"2026-07-24T{12 + index:02d}:00:00Z",
                source="Reuters",
            )
            for index in range(4)
        ] + [
            article(
                f"ap-{index}",
                f"Treasury bond yield affects stocks {index}",
                f"2026-07-24T{16 + index:02d}:00:00Z",
                source="AP",
            )
            for index in range(2)
        ]
        upsert_articles(
            items,
            self.db_path,
            seen_at="2026-07-24T19:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:00:00Z",
            db_path=self.db_path,
        )
        counts = {}
        for item in result["direct_evidence"]:
            counts[item["source"]] = counts.get(item["source"], 0) + 1

        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(len(result["direct_evidence"]), 4)

    def test_snapshot_supplies_top_and_bottom_sector_names(self):
        upsert_articles(
            [
                article(
                    "direct",
                    "Stocks rise after Federal Reserve decision",
                    "2026-07-24T18:00:00Z",
                ),
                article(
                    "old-tech",
                    "Semiconductor shares lift technology sector",
                    "2026-07-10T15:00:00Z",
                    source="CNBC",
                ),
            ],
            self.db_path,
            seen_at="2026-07-24T19:00:00Z",
        )
        record_market_snapshot(
            "2026-07-24",
            300,
            200,
            {
                "Information Technology": 2.0,
                "Utilities": -1.5,
                "Financials": 0.1,
            },
            self.db_path,
            recorded_at="2026-07-24T19:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:00:00Z",
            db_path=self.db_path,
        )

        self.assertIn(
            "old-tech",
            {item["article_id"] for item in result["historical_context"]},
        )
        self.assertEqual(database_status(self.db_path)["snapshot_count"], 1)

    def test_snapshot_recorded_after_as_of_is_not_used_for_sector_context(self):
        upsert_articles(
            [
                article(
                    "direct",
                    "Stocks rise after Federal Reserve decision",
                    "2026-07-24T18:00:00Z",
                ),
                article(
                    "old-tech",
                    "Semiconductor shares lift technology sector",
                    "2026-07-10T15:00:00Z",
                    source="CNBC",
                ),
            ],
            self.db_path,
            seen_at="2026-07-24T19:00:00Z",
        )
        record_market_snapshot(
            "2026-07-24",
            300,
            200,
            {"Information Technology": 2.0, "Utilities": -1.5},
            self.db_path,
            recorded_at="2026-07-24T23:00:00Z",
        )

        result = retrieve_market_context(
            "2026-07-24",
            retrieval_as_of="2026-07-24T22:00:00Z",
            db_path=self.db_path,
        )

        self.assertNotIn(
            "old-tech",
            {item["article_id"] for item in result["historical_context"]},
        )

    def test_retention_prunes_articles_older_than_90_days(self):
        result = upsert_articles(
            [
                article(
                    "old",
                    "Old stock market story",
                    "2026-04-01T12:00:00Z",
                ),
                article(
                    "current",
                    "Current stock market story",
                    "2026-07-24T12:00:00Z",
                ),
            ],
            self.db_path,
            seen_at="2026-07-24T19:00:00Z",
        )

        self.assertEqual(result["pruned"], 1)
        self.assertEqual(result["article_count"], 1)

    def test_missing_and_corrupt_database_have_clear_recovery_behavior(self):
        missing = database_status(self.db_path)
        self.assertEqual(missing["status"], "missing")
        self.assertFalse(quick_check(self.db_path)["ok"])

        self.db_path.write_bytes(b"not a sqlite database")
        self.assertEqual(database_status(self.db_path)["status"], "corrupt")
        with self.assertRaises(MarketRagError):
            init_db(self.db_path)

        recovered = init_db(self.db_path, recover_corrupt=True)
        self.assertTrue(recovered["ok"])
        self.assertTrue(Path(recovered["recovered_from"]).exists())

    def test_quick_check_rejects_valid_sqlite_without_rag_schema(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.close()

        result = quick_check(self.db_path)

        self.assertFalse(result["ok"])
        self.assertFalse(result["schema_ready"])
        self.assertEqual(result["result"], "ok")

    def test_missing_finnhub_key_returns_structured_status_and_cli_exit_two(self):
        with patch.dict(os.environ, {}, clear=True):
            result = ingest_finnhub_news(self.db_path)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["ingest", "--db", str(self.db_path)])

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_finnhub_api_key")
        self.assertEqual(exit_code, 2)
        self.assertEqual(
            json.loads(output.getvalue())["error_code"],
            "missing_finnhub_api_key",
        )

    @patch("market_rag.requests.get")
    def test_finnhub_ingest_normalises_and_persists_general_news(self, mock_get):
        response = mock_get.return_value
        response.ok = True
        response.json.return_value = [
            {
                "id": 123,
                "datetime": 1784916000,
                "headline": "Federal Reserve decision moves stocks",
                "summary": "Wall Street considered the interest rate path.",
                "source": "Reuters",
                "url": "https://example.com/fed",
            }
        ]

        with patch.dict(os.environ, {"FINNHUB_API_KEY": "test-key"}):
            result = ingest_finnhub_news(
                self.db_path,
                now=datetime(2026, 7, 24, 19, tzinfo=UTC),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["fetched"], 1)
        self.assertEqual(result["inserted"], 1)
        self.assertEqual(database_status(self.db_path)["article_count"], 1)
        self.assertEqual(
            mock_get.call_args.kwargs["params"]["category"],
            "general",
        )

    @patch("market_rag.requests.get")
    def test_finnhub_ingest_request_error_redacts_api_key(self, mock_get):
        mock_get.side_effect = requests.ConnectionError(
            "failed https://finnhub.io/api/v1/news?token=secret-key"
        )

        with patch.dict(os.environ, {"FINNHUB_API_KEY": "secret-key"}):
            result = ingest_finnhub_news(self.db_path)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "finnhub_request_failed")
        self.assertNotIn("secret-key", result["error"])
        self.assertNotIn("token=", result["error"])

    @patch("market_rag.requests.get")
    def test_finnhub_ingest_recovers_corrupt_cached_database(self, mock_get):
        self.db_path.write_bytes(b"not a sqlite database")
        response = mock_get.return_value
        response.ok = True
        response.json.return_value = [
            {
                "id": 124,
                "datetime": 1784916000,
                "headline": "Federal Reserve decision moves stocks",
                "summary": "Wall Street considered the interest rate path.",
                "source": "Reuters",
                "url": "https://example.com/fed-recovered",
            }
        ]

        with patch.dict(os.environ, {"FINNHUB_API_KEY": "test-key"}):
            result = ingest_finnhub_news(
                self.db_path,
                now=datetime(2026, 7, 24, 19, tzinfo=UTC),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(database_status(self.db_path)["article_count"], 1)
        self.assertEqual(
            len(list(self.db_path.parent.glob("market_rag.sqlite3.corrupt-*"))),
            1,
        )


if __name__ == "__main__":
    unittest.main()
