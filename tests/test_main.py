import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import main
from fetcher import UpstreamNotReady
from main import _load_published_data_date


class MainReportDateTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.report_dir = Path(self.temp_dir.name)
        self.summary_path = self.report_dir / "summary.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_published_report(self, data_date: str = "2026-07-24"):
        self.summary_path.write_text(
            json.dumps({"data_date": data_date}),
            encoding="utf-8",
        )
        html_path = self.report_dir / f"SPX_daily_{data_date.replace('-', '')}.html"
        html_path.write_text("published dashboard", encoding="utf-8")
        return html_path

    @staticmethod
    def _provenance():
        return {
            "git_sha": "test",
            "python_version": "test",
            "pandas_version": "test",
            "yfinance_version": "test",
            "calendar_version": "test",
            "retriever_version": "test",
        }

    def test_load_published_data_date_requires_matching_html(self):
        self.summary_path.write_text(
            json.dumps({"data_date": "2026-07-24"}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "일치하는 유효한 HTML"):
            _load_published_data_date(self.summary_path, self.report_dir)

    def test_load_published_data_date_rejects_malformed_summary(self):
        self.summary_path.write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "data_date를 검증"):
            _load_published_data_date(self.summary_path, self.report_dir)

    def test_load_published_data_date_rejects_empty_matching_html(self):
        html_path = self._write_published_report()
        html_path.write_bytes(b"")

        with self.assertRaisesRegex(RuntimeError, "유효한 HTML"):
            _load_published_data_date(self.summary_path, self.report_dir)

    def test_already_current_is_noop_before_yahoo_or_output(self):
        html_path = self._write_published_report()
        summary_before = self.summary_path.read_bytes()
        html_before = html_path.read_bytes()
        status_path = self.report_dir / "run-status.json"

        with (
            patch.object(main, "OUTPUT_DIR", self.report_dir),
            patch.object(main, "setup_logging"),
            patch.object(
                main,
                "_runtime_provenance",
                return_value=self._provenance(),
            ),
            patch.object(
                main,
                "get_expected_market_date",
            ) as mock_expected_date,
            patch.object(
                main,
                "fetch_all_data",
            ) as mock_fetch,
            patch.object(main, "enrich_with_ai") as mock_ai,
            patch.object(main, "generate_html") as mock_generate_html,
        ):
            exit_code = main.run(
                status_file=status_path,
                expected_date_override=date(2026, 7, 24),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(self.summary_path.read_bytes(), summary_before)
        self.assertEqual(html_path.read_bytes(), html_before)
        mock_fetch.assert_not_called()
        mock_expected_date.assert_not_called()
        mock_ai.assert_not_called()
        mock_generate_html.assert_not_called()
        self.assertEqual(
            json.loads(status_path.read_text(encoding="utf-8")),
            {
                "status": "already_current",
                "reason": "published_date_matches_expected_session",
                "expected_date": "2026-07-24",
                "published_date": "2026-07-24",
            },
        )

    def test_force_same_date_reports_source_not_ready_without_output_changes(self):
        html_path = self._write_published_report()
        summary_before = self.summary_path.read_bytes()
        html_before = html_path.read_bytes()
        status_path = self.report_dir / "run-status.json"
        source_error = UpstreamNotReady(
            "benchmark_close_missing",
            date(2026, 7, 24),
            raw_latest_date=date(2026, 7, 24),
            observed_date=date(2026, 7, 23),
            coverage=0.0,
            aligned=0,
            total=1,
            stale_count=1,
        )

        with (
            patch.object(main, "OUTPUT_DIR", self.report_dir),
            patch.object(main, "setup_logging"),
            patch.object(
                main,
                "_runtime_provenance",
                return_value=self._provenance(),
            ),
            patch.object(
                main,
                "get_expected_market_date",
                return_value=date(2026, 7, 24),
            ),
            patch.object(
                main,
                "fetch_all_data",
                side_effect=source_error,
            ) as mock_fetch,
            patch.object(main, "enrich_with_ai") as mock_ai,
            patch.object(main, "generate_html") as mock_generate_html,
        ):
            exit_code = main.run(
                status_file=status_path,
                force_same_date=True,
            )

        self.assertEqual(exit_code, 75)
        mock_fetch.assert_called_once_with(date(2026, 7, 24))
        mock_ai.assert_not_called()
        mock_generate_html.assert_not_called()
        self.assertEqual(self.summary_path.read_bytes(), summary_before)
        self.assertEqual(html_path.read_bytes(), html_before)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "source_not_ready")
        self.assertEqual(status["reason"], "benchmark_close_missing")
        self.assertEqual(status["expected_date"], "2026-07-24")
        self.assertEqual(status["observed_date"], "2026-07-23")
        self.assertEqual(status["published_date"], "2026-07-24")

    def test_published_ahead_of_calendar_fails_before_yahoo(self):
        self._write_published_report()
        status_path = self.report_dir / "run-status.json"

        with (
            patch.object(main, "OUTPUT_DIR", self.report_dir),
            patch.object(main, "setup_logging"),
            patch.object(
                main,
                "_runtime_provenance",
                return_value=self._provenance(),
            ),
            patch.object(
                main,
                "get_expected_market_date",
                return_value=date(2026, 7, 23),
            ),
            patch.object(main, "fetch_all_data") as mock_fetch,
        ):
            with self.assertRaises(SystemExit) as captured:
                main.run(status_file=status_path)

        self.assertEqual(captured.exception.code, 1)
        mock_fetch.assert_not_called()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "published_ahead_of_calendar")
        self.assertEqual(status["expected_date"], "2026-07-23")
        self.assertEqual(status["published_date"], "2026-07-24")


if __name__ == "__main__":
    unittest.main()
