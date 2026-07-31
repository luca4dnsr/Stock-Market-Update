"""Exchange-aware timestamps shared by the report and news pipelines."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from config import (
    WORKFLOW_NEWS_CUTOFF_HOUR_KST,
    WORKFLOW_NEWS_CUTOFF_MINUTE_KST,
)

UTC = timezone.utc
SEOUL = ZoneInfo("Asia/Seoul")


def _session_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(str(value))


@lru_cache(maxsize=1)
def _xnys():
    return xcals.get_calendar("XNYS")


@lru_cache(maxsize=1)
def _xkrx():
    return xcals.get_calendar("XKRX")


def market_close_utc(value: str | date) -> datetime:
    """Return the actual XNYS close, including holidays and early closes."""
    session = _session_date(value)
    if not _xnys().is_session(session):
        raise ValueError(f"XNYS 거래일이 아닙니다: {session}")
    return _xnys().session_close(session).to_pydatetime().astimezone(UTC)


def target_korea_session_date(as_of: datetime) -> date:
    """Return today's still-open XKRX session, or the next available session."""
    observed = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
    observed_utc = observed.astimezone(UTC)
    local_date = observed_utc.astimezone(SEOUL).date()
    calendar = _xkrx()
    if (
        calendar.is_session(local_date)
        and observed_utc < calendar.session_close(local_date).to_pydatetime()
    ):
        return local_date
    if calendar.is_session(local_date):
        return calendar.next_session(local_date).date()
    return calendar.date_to_session(local_date, direction="next").date()


def workflow_news_cutoff(
    value: str | date,
    as_of: datetime | None = None,
) -> datetime:
    """Return scheduled KST cutoff, capped at the actual retrieval time."""
    cutoff_date = _session_date(value) + timedelta(days=1)
    scheduled = datetime.combine(
        cutoff_date,
        time(
            WORKFLOW_NEWS_CUTOFF_HOUR_KST,
            WORKFLOW_NEWS_CUTOFF_MINUTE_KST,
        ),
        SEOUL,
    ).astimezone(UTC)
    if as_of is None:
        return scheduled
    observed = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
    return min(scheduled, observed.astimezone(UTC))


def report_session_phase(
    published_at: datetime,
    data_date: str | date,
) -> str:
    """Classify evidence relative to the report session's actual close."""
    published = (
        published_at
        if published_at.tzinfo is not None
        else published_at.replace(tzinfo=UTC)
    )
    return (
        "post_close"
        if published.astimezone(UTC) >= market_close_utc(data_date)
        else "regular_session"
    )
