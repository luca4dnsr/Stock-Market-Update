import unittest
from datetime import date, datetime, timezone

from market_clock import (
    market_close_utc,
    report_session_phase,
    target_korea_session_date,
    workflow_news_cutoff,
)


UTC = timezone.utc


class MarketClockTest(unittest.TestCase):
    def test_target_korea_session_uses_today_before_close_and_next_after_close(self):
        self.assertEqual(
            target_korea_session_date(
                datetime(2026, 7, 30, 1, tzinfo=timezone.utc)
            ),
            date(2026, 7, 30),
        )
        self.assertEqual(
            target_korea_session_date(
                datetime(2026, 7, 30, 7, tzinfo=timezone.utc)
            ),
            date(2026, 7, 31),
        )

    def test_target_korea_session_skips_weekend(self):
        self.assertEqual(
            target_korea_session_date(
                datetime(2026, 8, 1, 0, tzinfo=timezone.utc)
            ),
            date(2026, 8, 3),
        )

    def test_xnys_close_uses_regular_and_early_close_schedule(self):
        self.assertEqual(
            market_close_utc("2026-07-24"),
            datetime(2026, 7, 24, 20, tzinfo=UTC),
        )
        self.assertEqual(
            market_close_utc("2026-11-27"),
            datetime(2026, 11, 27, 18, tzinfo=UTC),
        )

    def test_exact_close_and_after_close_are_post_close(self):
        self.assertEqual(
            report_session_phase(
                datetime(2026, 11, 27, 17, 59, tzinfo=UTC),
                "2026-11-27",
            ),
            "regular_session",
        )
        self.assertEqual(
            report_session_phase(
                datetime(2026, 11, 27, 18, tzinfo=UTC),
                "2026-11-27",
            ),
            "post_close",
        )
        self.assertEqual(
            report_session_phase(
                datetime(2026, 11, 27, 19, tzinfo=UTC),
                "2026-11-27",
            ),
            "post_close",
        )

    def test_workflow_cutoff_remains_kst_0700_on_early_close(self):
        self.assertEqual(
            workflow_news_cutoff(
                "2026-11-27",
                datetime(2026, 11, 27, 23, tzinfo=UTC),
            ),
            datetime(2026, 11, 27, 22, tzinfo=UTC),
        )


if __name__ == "__main__":
    unittest.main()
