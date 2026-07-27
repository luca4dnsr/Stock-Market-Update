"""Small, time-aware SQLite FTS5 store for market-news RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

from config import (
    MARKET_RAG_CONTEXT_MAX_AGE_DAYS,
    MARKET_RAG_CONTEXT_MAX_SOURCES,
    MARKET_RAG_DB_FILE,
    MARKET_RAG_DIRECT_MAX_SOURCES,
    MARKET_RAG_RETENTION_DAYS,
    MARKET_MIN_NEWS_SOURCES,
)
from market_clock import (
    market_close_utc,
    report_session_phase,
    workflow_news_cutoff,
)

DEFAULT_DB_PATH = MARKET_RAG_DB_FILE
SCHEMA_VERSION = "1"
RETRIEVER_VERSION = "v2-sqlite-fts5-workflow-cutoff"
RETENTION_DAYS = MARKET_RAG_RETENTION_DAYS
DIRECT_MAX_SOURCES = MARKET_RAG_DIRECT_MAX_SOURCES
CONTEXT_MAX_SOURCES = MARKET_RAG_CONTEXT_MAX_SOURCES
DIRECT_SESSIONS = 3
HISTORY_DAYS = MARKET_RAG_CONTEXT_MAX_AGE_DAYS
FINNHUB_URL = "https://finnhub.io/api/v1/news"
UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")

MACRO_TAG_KEYWORDS = {
    "market": ("s&p 500", "nasdaq", "dow jones", "wall street", "stock market", "stocks", "vix"),
    "monetary_policy": ("federal reserve", "fed", "fomc", "rate cut", "rate hike", "interest rate"),
    "inflation": ("inflation", "consumer price index", "cpi", "ppi", "pce"),
    "labor": ("payroll", "jobs report", "jobless claims", "unemployment", "labor market"),
    "growth": ("gdp", "economic growth", "recession", "retail sales", "consumer spending"),
    "rates_bonds": ("treasury", "bond yield", "yield curve", "credit spread"),
    "trade_energy": ("tariff", "trade war", "sanction", "crude oil", "opec", "natural gas"),
}
SECTOR_TAG_KEYWORDS = {
    "Communication Services": ("social media", "streaming", "telecom"),
    "Consumer Discretionary": ("consumer discretionary", "automaker", "e-commerce", "travel demand"),
    "Consumer Staples": ("consumer staples", "grocery", "beverage", "household products"),
    "Energy": ("energy sector", "oil producer", "oilfield", "refiner"),
    "Financials": ("financial sector", "bank stocks", "banks", "insurer"),
    "Health Care": ("health care", "healthcare", "biotech", "pharmaceutical"),
    "Industrials": ("industrial sector", "aerospace", "machinery", "transportation"),
    "Information Technology": ("technology sector", "semiconductor", "software", "cloud computing", "chipmaker"),
    "Materials": ("materials sector", "mining", "steelmaker", "chemicals", "copper"),
    "Real Estate": ("real estate", "reit", "commercial property", "homebuilder"),
    "Utilities": ("utilities sector", "electric utility", "power producer", "regulated utility"),
}
EVENT_TAG_KEYWORDS = {
    "economic_data": ("cpi", "ppi", "pce", "payroll", "jobless claims", "retail sales", "gdp"),
    "central_bank": ("federal reserve", "fed", "fomc", "central bank", "rate decision"),
    "earnings": ("earnings", "quarterly results", "revenue", "guidance", "outlook"),
    "regulation": ("regulation", "regulatory", "antitrust", "investigation"),
    "m_and_a": ("merger", "acquisition", "takeover", "buyout"),
}
SECTOR_ALIASES = {
    "communications services": "Communication Services",
    "financial": "Financials",
    "healthcare": "Health Care",
    "technology": "Information Technology",
    **{name.casefold(): name for name in SECTOR_TAG_KEYWORDS},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS articles(
 id INTEGER PRIMARY KEY,
 provider TEXT NOT NULL,
 provider_article_id TEXT NOT NULL DEFAULT '',
 url TEXT NOT NULL,
 url_hash TEXT NOT NULL,
 content_hash TEXT NOT NULL,
 headline TEXT NOT NULL,
 summary TEXT NOT NULL DEFAULT '',
 source TEXT NOT NULL DEFAULT '',
 published_at TEXT NOT NULL,
 first_seen_at TEXT NOT NULL,
 last_seen_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_article_provider
 ON articles(provider, provider_article_id) WHERE provider_article_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS uq_article_url ON articles(url_hash);
CREATE UNIQUE INDEX IF NOT EXISTS uq_article_content ON articles(content_hash);
CREATE INDEX IF NOT EXISTS ix_article_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS ix_article_seen ON articles(first_seen_at);
CREATE TABLE IF NOT EXISTS article_tags(
 article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('macro','sector','event')),
 value TEXT NOT NULL,
 PRIMARY KEY(article_id, kind, value)
);
CREATE INDEX IF NOT EXISTS ix_tags ON article_tags(kind, value, article_id);
CREATE TABLE IF NOT EXISTS market_snapshots(
 data_date TEXT PRIMARY KEY,
 advances INTEGER NOT NULL,
 declines INTEGER NOT NULL,
 sector_returns_json TEXT NOT NULL,
 market_close_cutoff TEXT NOT NULL,
 recorded_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
 headline, summary, content='articles', content_rowid='id',
 tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
 INSERT INTO articles_fts(rowid,headline,summary) VALUES(new.id,new.headline,new.summary);
END;
CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
 INSERT INTO articles_fts(articles_fts,rowid,headline,summary)
 VALUES('delete',old.id,old.headline,old.summary);
END;
CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
 INSERT INTO articles_fts(articles_fts,rowid,headline,summary)
 VALUES('delete',old.id,old.headline,old.summary);
 INSERT INTO articles_fts(rowid,headline,summary) VALUES(new.id,new.headline,new.summary);
END;
"""


class MarketRagError(RuntimeError):
    pass


def _path(value: str | os.PathLike[str] | None) -> Path:
    return Path(value) if value is not None else DEFAULT_DB_PATH


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.min, tzinfo=UTC)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = datetime.fromtimestamp(value, UTC)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        result = (
            datetime.fromtimestamp(float(text), UTC)
            if re.fullmatch(r"\d+(?:\.\d+)?", text)
            else datetime.fromisoformat(text.replace("Z", "+00:00"))
        )
    else:
        raise ValueError(f"invalid {name}")
    return (result.replace(tzinfo=UTC) if result.tzinfo is None else result).astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _market_close(value: str | date) -> datetime:
    return market_close_utc(value)


def _workflow_news_cutoff(
    value: str | date,
    as_of: datetime | None = None,
) -> datetime:
    return workflow_news_cutoff(value, as_of)


def _direct_start(value: str | date) -> datetime:
    """Approximate three trading sessions by skipping weekends."""
    session = _session_date(value)
    included = 1
    while included < DIRECT_SESSIONS:
        session -= timedelta(days=1)
        if session.weekday() < 5:
            included += 1
    return datetime.combine(session, time.min, NEW_YORK).astimezone(UTC)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _canonical_url(value: Any) -> str:
    parsed = urlsplit(_clean(value))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"fbclid", "gclid", "ref", "source"}
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(query), ""))


def _hash(*values: str) -> str:
    return hashlib.sha256("\n".join(values).casefold().encode()).hexdigest()


def _tagged(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", text) is not None


def classify_tags(headline: str, summary: str = "") -> dict[str, list[str]]:
    text = f"{headline} {summary}".casefold()
    groups = (
        ("macro", MACRO_TAG_KEYWORDS),
        ("sector", SECTOR_TAG_KEYWORDS),
        ("event", EVENT_TAG_KEYWORDS),
    )
    return {
        kind: sorted(
            tag for tag, words in definitions.items()
            if any(_tagged(text, word) for word in words)
        )
        for kind, definitions in groups
    }


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=10)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        return db
    except Exception:
        db.close()
        raise


def _check(db: sqlite3.Connection) -> str:
    row = db.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else "no_result"


def quick_check(db_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = _path(db_path)
    if not path.exists():
        return {
            "ok": False, "db_path": str(path), "result": "missing",
            "quick_check": "missing", "article_count": 0,
        }
    db = None
    try:
        db = _connect(path)
        result = _check(db)
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "meta",
            "articles",
            "article_tags",
            "market_snapshots",
            "articles_fts",
        }
        schema_ready = required <= tables
        version_row = (
            db.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if schema_ready
            else None
        )
        schema_version = str(version_row[0]) if version_row else None
        schema_ready = schema_ready and schema_version == SCHEMA_VERSION
        try:
            article_count = int(db.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
        except sqlite3.DatabaseError:
            article_count = 0
        return {
            "ok": result == "ok" and schema_ready,
            "db_path": str(path),
            "result": result,
            "quick_check": result,
            "schema_version": schema_version,
            "schema_ready": schema_ready,
            "article_count": article_count,
        }
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False, "db_path": str(path), "result": "database_error",
            "quick_check": "database_error", "article_count": 0, "error": str(exc),
        }
    finally:
        if db is not None:
            db.close()


def database_status(db_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = _path(db_path)
    if not path.exists():
        return {"ok": False, "status": "missing", "exists": False, "db_path": str(path), "article_count": 0}
    db = None
    try:
        db = _connect(path)
        required = {"meta", "articles", "article_tags", "market_snapshots", "articles_fts"}
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required <= tables:
            return {"ok": False, "status": "uninitialized", "exists": True, "db_path": str(path), "article_count": 0}
        check = _check(db)
        version_row = db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        schema_version = str(version_row[0]) if version_row else None
        schema_ready = schema_version == SCHEMA_VERSION
        row = db.execute(
            "SELECT COUNT(*),MIN(published_at),MAX(published_at) FROM articles"
        ).fetchone()
        return {
            "ok": check == "ok" and schema_ready,
            "status": (
                "ok"
                if check == "ok" and schema_ready
                else "schema_mismatch"
                if check == "ok"
                else "corrupt"
            ),
            "exists": True,
            "db_path": str(path),
            "quick_check": check,
            "schema_version": schema_version,
            "article_count": int(row[0]),
            "snapshot_count": int(db.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]),
            "oldest_published_at": row[1],
            "newest_published_at": row[2],
            "size_bytes": path.stat().st_size,
        }
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "status": "corrupt", "exists": True, "db_path": str(path), "article_count": 0, "error": str(exc)}
    finally:
        if db is not None:
            db.close()


status = database_status


def init_db(
    db_path: str | os.PathLike[str] | None = None, *, recover_corrupt: bool = False
) -> dict[str, Any]:
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    for attempt in range(2):
        db = None
        try:
            db = _connect(path)
            with db:
                db.executescript(SCHEMA)
                db.execute(
                    "INSERT INTO meta VALUES('schema_version',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (SCHEMA_VERSION,),
                )
                if _check(db) != "ok":
                    raise sqlite3.DatabaseError("quick_check failed")
            result = database_status(path)
            if backup:
                result["recovered_from"] = str(backup)
            return result
        except sqlite3.DatabaseError as exc:
            if db is not None:
                db.close()
                db = None
            if not recover_corrupt or attempt or not path.exists():
                raise MarketRagError(f"market RAG database error: {exc}") from exc
            backup = path.with_name(f"{path.name}.corrupt-{_now():%Y%m%dT%H%M%S%fZ}")
            path.replace(backup)
            for suffix in ("-journal", "-wal", "-shm"):
                sidecar = Path(f"{path}{suffix}")
                if sidecar.exists():
                    sidecar.replace(Path(f"{backup}{suffix}"))
        finally:
            if db is not None:
                db.close()
    raise MarketRagError("market RAG recovery failed")


def _normalise_article(raw: Mapping[str, Any], seen: datetime) -> dict[str, Any]:
    headline, summary = _clean(raw.get("headline")), _clean(raw.get("summary"))
    url = _canonical_url(raw.get("url"))
    if not headline or not url:
        raise ValueError("headline and valid URL are required")
    published = _as_utc(raw.get("published_at", raw.get("datetime")), "published_at")
    provider = _clean(raw.get("provider") or "finnhub").casefold()
    provider_id = _clean(raw.get("provider_article_id", raw.get("article_id", raw.get("id"))))
    return {
        "provider": provider,
        "provider_id": provider_id,
        "url": url,
        "url_hash": _hash(url),
        "content_hash": _hash(headline, summary),
        "headline": headline[:500],
        "summary": summary[:2000],
        "source": _clean(raw.get("source") or provider)[:200],
        "published": _iso(published),
        "seen": _iso(seen),
        "tags": classify_tags(headline, summary),
    }


def _matches(db: sqlite3.Connection, item: Mapping[str, Any]) -> list[sqlite3.Row]:
    sql = "url_hash=? OR content_hash=?"
    values: list[Any] = [item["url_hash"], item["content_hash"]]
    if item["provider_id"]:
        sql += " OR (provider=? AND provider_article_id=?)"
        values += [item["provider"], item["provider_id"]]
    return list(
        db.execute(
            f"SELECT * FROM articles WHERE {sql} ORDER BY first_seen_at,id",
            values,
        )
    )


def _set_tags(db: sqlite3.Connection, article_id: int, tags: Mapping[str, Sequence[str]]) -> None:
    db.execute("DELETE FROM article_tags WHERE article_id=?", (article_id,))
    db.executemany(
        "INSERT INTO article_tags VALUES(?,?,?)",
        [(article_id, kind, value) for kind in ("macro", "sector", "event") for value in tags[kind]],
    )


def upsert_articles(
    articles: Iterable[Mapping[str, Any]],
    db_path: str | os.PathLike[str] | None = None,
    *,
    seen_at: datetime | str | None = None,
    retention_days: int = RETENTION_DAYS,
) -> dict[str, Any]:
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    path, seen = _path(db_path), _as_utc(seen_at or _now(), "seen_at")
    init_db(path)
    inserted = updated = deduplicated = skipped = 0
    db = _connect(path)
    try:
        with db:
            for raw in articles:
                try:
                    item = _normalise_article(raw, seen)
                except (AttributeError, TypeError, ValueError, OSError, OverflowError):
                    skipped += 1
                    continue
                matches = _matches(db, item)
                if not matches:
                    cursor = db.execute(
                        "INSERT INTO articles(provider,provider_article_id,url,url_hash,"
                        "content_hash,headline,summary,source,published_at,first_seen_at,last_seen_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            item["provider"], item["provider_id"], item["url"], item["url_hash"],
                            item["content_hash"], item["headline"], item["summary"], item["source"],
                            item["published"], item["seen"], item["seen"],
                        ),
                    )
                    article_id, inserted = int(cursor.lastrowid), inserted + 1
                    replace_content = True
                else:
                    target, article_id = matches[0], int(matches[0]["id"])
                    for duplicate in matches[1:]:
                        db.execute("DELETE FROM articles WHERE id=?", (duplicate["id"],))
                        deduplicated += 1
                    replace_content = item["seen"] < target["first_seen_at"]
                    db.execute(
                        "UPDATE articles SET provider_article_id=?,content_hash=?,headline=?,"
                        "summary=?,source=?,published_at=?,first_seen_at=?,last_seen_at=? "
                        "WHERE id=?",
                        (
                            target["provider_article_id"] or item["provider_id"],
                            (
                                item["content_hash"]
                                if replace_content
                                else target["content_hash"]
                            ),
                            item["headline"] if replace_content else target["headline"],
                            item["summary"] if replace_content else target["summary"],
                            item["source"] if replace_content else target["source"],
                            min(target["published_at"], item["published"]),
                            min(target["first_seen_at"], item["seen"]),
                            max(target["last_seen_at"], item["seen"]),
                            article_id,
                        ),
                    )
                    updated, deduplicated = updated + 1, deduplicated + 1
                if replace_content:
                    _set_tags(db, article_id, item["tags"])
            threshold = _iso(seen - timedelta(days=retention_days))
            pruned = db.execute("DELETE FROM articles WHERE published_at<?", (threshold,)).rowcount
            db.execute("DELETE FROM market_snapshots WHERE market_close_cutoff<?", (threshold,))
            if _check(db) != "ok":
                raise sqlite3.DatabaseError("quick_check failed")
        total = int(db.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
    finally:
        db.close()
    return {
        "ok": True, "db_path": str(path), "inserted": inserted, "updated": updated,
        "deduplicated": deduplicated, "skipped": skipped, "pruned": int(pruned),
        "article_count": total, "quick_check": "ok",
    }


def _sector_name(value: Any) -> str:
    text = _clean(value)
    return SECTOR_ALIASES.get(text.casefold(), text)


def _sector_returns(value: Any) -> dict[str, float]:
    pairs = value.items() if isinstance(value, Mapping) else (
        (
            row.get("sector", row.get("name")),
            row.get("return_1d", row.get("return", row.get("value"))),
        )
        for row in value if isinstance(row, Mapping)
    )
    result = {}
    for name, number in pairs:
        try:
            result[_sector_name(name)] = float(number)
        except (TypeError, ValueError):
            continue
    return {name: number for name, number in result.items() if name}


def record_market_snapshot(
    data_date: str | date,
    advances: int,
    declines: int,
    sector_returns: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    db_path: str | os.PathLike[str] | None = None,
    *,
    recorded_at: datetime | str | None = None,
) -> None:
    path, session = _path(db_path), _session_date(data_date)
    recorded = _as_utc(recorded_at or _now(), "recorded_at")
    init_db(path)
    db = _connect(path)
    try:
        with db:
            db.execute(
                "INSERT INTO market_snapshots VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(data_date) DO UPDATE SET advances=excluded.advances,"
                "declines=excluded.declines,sector_returns_json=excluded.sector_returns_json,"
                "market_close_cutoff=excluded.market_close_cutoff,recorded_at=excluded.recorded_at",
                (
                    session.isoformat(), int(advances), int(declines),
                    json.dumps(_sector_returns(sector_returns), sort_keys=True),
                    _iso(_market_close(session)), _iso(recorded),
                ),
            )
            if _check(db) != "ok":
                raise sqlite3.DatabaseError("quick_check failed")
    finally:
        db.close()


def _snapshot_sectors(
    db: sqlite3.Connection,
    data_date: str,
    as_of: datetime,
) -> list[str]:
    row = db.execute(
        "SELECT sector_returns_json FROM market_snapshots "
        "WHERE data_date=? AND recorded_at<=?",
        (data_date, _iso(as_of)),
    ).fetchone()
    if not row:
        return []
    try:
        ranked = sorted(json.loads(row[0]).items(), key=lambda item: (float(item[1]), item[0]))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(dict.fromkeys([name for name, _ in ranked[:2] + ranked[-2:]]))


def _fts_query(words: Iterable[str]) -> str:
    values = [f'"{_clean(word).casefold().replace(chr(34), chr(34) * 2)}"' for word in words if _clean(word)]
    return " OR ".join(dict.fromkeys(values))


def _terms(themes: Iterable[str], sectors: Iterable[str]) -> list[str]:
    result = []
    for theme in themes:
        result += list(MACRO_TAG_KEYWORDS.get(theme, (theme,)))
    for sector in sectors:
        canonical = _sector_name(sector)
        result += list(SECTOR_TAG_KEYWORDS.get(canonical, (canonical,)))
    return result


def _is_market_scoped(row: Mapping[str, Any]) -> bool:
    tags = row["tags"]
    if tags["macro"]:
        return True
    text = f"{row['headline']} {row['summary']}".casefold()
    return bool(tags["sector"]) and any(
        _tagged(text, term)
        for term in ("sector", "industry", "stocks", "companies")
    )


def _candidates(
    db: sqlite3.Connection,
    query: str,
    start: datetime,
    end: datetime,
    as_of: datetime,
) -> list[dict[str, Any]]:
    if not query:
        return []
    rows = [
        dict(row) for row in db.execute(
            "SELECT a.*,bm25(articles_fts,5.0,1.0) AS rank FROM articles_fts "
            "JOIN articles a ON a.id=articles_fts.rowid "
            "WHERE articles_fts MATCH ? AND a.published_at>=? AND a.published_at<=? "
            "AND a.first_seen_at<=?",
            (query, _iso(start), _iso(end), _iso(as_of)),
        )
    ]
    if not rows:
        return rows
    ids = [row["id"] for row in rows]
    tag_map = {article_id: {"macro": [], "sector": [], "event": []} for article_id in ids}
    marks = ",".join("?" for _ in ids)
    for tag in db.execute(
        f"SELECT * FROM article_tags WHERE article_id IN ({marks}) ORDER BY kind,value", ids
    ):
        tag_map[tag["article_id"]][tag["kind"]].append(tag["value"])
    for row in rows:
        row["tags"] = tag_map[row["id"]]
    return rows


def _diverse(rows: list[dict[str, Any]], limit: int, per_source: int) -> list[dict[str, Any]]:
    selected, counts = [], {}
    for row in rows:
        source = _clean(row["source"]).casefold() or "unknown"
        if counts.get(source, 0) >= per_source:
            continue
        selected.append(row)
        counts[source] = counts.get(source, 0) + 1
        if len(selected) == limit:
            return selected
    return selected


def _diverse_direct(
    rows: list[dict[str, Any]],
    market_close: datetime,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    source_counts: dict[str, int] = {}

    def add_row(row: dict[str, Any]) -> bool:
        row_id = int(row["id"])
        if row_id in selected_ids:
            return False
        source = _clean(row["source"]).casefold() or "unknown"
        if source_counts.get(source, 0) >= 2:
            return False
        selected.append(row)
        selected_ids.add(row_id)
        source_counts[source] = source_counts.get(source, 0) + 1
        return True

    for row in rows:
        if _as_utc(row["published_at"], "published_at") >= market_close:
            continue
        add_row(row)
        if len(selected) == MARKET_MIN_NEWS_SOURCES:
            break

    for row in rows:
        add_row(row)
        if len(selected) == DIRECT_MAX_SOURCES:
            break
    return selected


def _public(
    row: Mapping[str, Any],
    *,
    report_data_date: date | None = None,
) -> dict[str, Any]:
    provider_id = str(row["provider_article_id"] or row["id"])
    published = _as_utc(row["published_at"], "published_at")
    result = {
        "evidence_id": f"{row['provider']}:{provider_id}",
        "article_id": provider_id,
        "published_at": _iso(published),
        "published_date": published.astimezone(NEW_YORK).date().isoformat(),
        "headline": row["headline"],
        "summary": row["summary"],
        "source": row["source"],
        "url": row["url"],
        "tags": {kind: list(row["tags"][kind]) for kind in ("macro", "sector", "event")},
    }
    if report_data_date is not None:
        result["session_phase"] = report_session_phase(
            published,
            report_data_date,
        )
    return result


def _empty(data_date: str, as_of: datetime, corpus: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "direct_evidence": [], "historical_context": [], "rag_status": "unavailable",
        "retriever_version": RETRIEVER_VERSION,
        "market_close_cutoff": _iso(_market_close(data_date)),
        "news_cutoff": _iso(_workflow_news_cutoff(data_date, as_of)),
        "retrieval_as_of": _iso(as_of), "corpus_status": dict(corpus),
    }


def retrieve_market_context(
    data_date: str | date,
    retrieval_as_of: datetime | str | None = None,
    sector_names: Sequence[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    session, as_of, path = _session_date(data_date), _as_utc(retrieval_as_of or _now(), "retrieval_as_of"), _path(db_path)
    corpus = database_status(path)
    if not corpus["ok"]:
        return _empty(session.isoformat(), as_of, corpus)
    close, direct_start = _market_close(session), _direct_start(session)
    news_cutoff = _workflow_news_cutoff(session, as_of)
    evidence_end = news_cutoff
    db = _connect(path)
    try:
        sectors = [_sector_name(value) for value in (sector_names or ()) if _sector_name(value)]
        sectors = sectors or _snapshot_sectors(db, session.isoformat(), as_of)
        all_words = [
            word
            for group in (MACRO_TAG_KEYWORDS, SECTOR_TAG_KEYWORDS)
            for words in group.values() for word in words
        ]
        direct_rows = [
            row
            for row in _candidates(
                db,
                _fts_query(all_words),
                direct_start,
                evidence_end,
                as_of,
            )
            if _is_market_scoped(row)
        ]
        direct_rows.sort(
            key=lambda row: (
                -sum(map(len, row["tags"].values())), float(row["rank"]),
                -_as_utc(row["published_at"], "published_at").timestamp(), row["id"],
            )
        )
        direct = _diverse_direct(direct_rows, close)
        direct_ids = {row["id"] for row in direct}
        themes = sorted({tag for row in direct for tag in row["tags"]["macro"]})
        context_rows = _candidates(
            db, _fts_query(_terms(themes, sectors)),
            close - timedelta(days=HISTORY_DAYS),
            direct_start - timedelta(seconds=1),
            as_of,
        )
        theme_set, sector_set = set(themes), set(sectors)
        ranked_context = []
        for row in context_rows:
            if not _is_market_scoped(row):
                continue
            matches = len(theme_set & set(row["tags"]["macro"])) + len(
                sector_set & set(row["tags"]["sector"])
            )
            if row["id"] not in direct_ids and matches:
                age = (close - _as_utc(row["published_at"], "published_at")).total_seconds()
                row["_key"] = (-matches, float(row["rank"]), age, row["id"])
                ranked_context.append(row)
        ranked_context.sort(key=lambda row: row["_key"])
        historical = _diverse(ranked_context, CONTEXT_MAX_SOURCES, 1)
    except sqlite3.DatabaseError as exc:
        corpus.update({"ok": False, "status": "query_error", "error": str(exc)})
        return _empty(session.isoformat(), as_of, corpus)
    finally:
        db.close()
    rag_status = "ok" if direct and historical else "limited" if direct or historical else "empty"
    return {
        "direct_evidence": [
            _public(row, report_data_date=session) for row in direct
        ],
        "historical_context": [_public(row) for row in historical],
        "rag_status": rag_status, "retriever_version": RETRIEVER_VERSION,
        "market_close_cutoff": _iso(close),
        "news_cutoff": _iso(news_cutoff),
        "retrieval_as_of": _iso(as_of),
        "corpus_status": corpus,
    }


retrieve = retrieve_market_context


def _error(path: Path, code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "ok": False, "command": "ingest", "db_path": str(path),
        "error_code": code, "error": message, **extra,
    }


def ingest_finnhub_news(
    db_path: str | os.PathLike[str] | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    path, seen = _path(db_path), _as_utc(now or _now(), "now")
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return _error(path, "missing_finnhub_api_key", "FINNHUB_API_KEY is not configured")
    try:
        response = requests.get(
            FINNHUB_URL,
            params={"category": "general", "token": api_key},
            headers={"Accept": "application/json"},
            timeout=(10, 25),
        )
        if not response.ok:
            return _error(
                path, "finnhub_http_error", f"Finnhub returned HTTP {response.status_code}",
                http_status=response.status_code,
            )
        payload = response.json()
        if not isinstance(payload, list):
            return _error(path, "finnhub_invalid_payload", "Finnhub response is not an array")
        init_db(path, recover_corrupt=True)
        result = upsert_articles(payload, path, seen_at=seen)
        return {"command": "ingest", "fetched": len(payload), **result}
    except requests.RequestException as exc:
        return _error(
            path,
            "finnhub_request_failed",
            f"Finnhub request failed ({type(exc).__name__})",
        )
    except (MarketRagError, sqlite3.DatabaseError, OSError, ValueError) as exc:
        return _error(path, "database_error", str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ingest.add_argument("--now")
    for command in ("status", "quick-check"):
        child = commands.add_parser(command)
        child.add_argument("--db", default=str(DEFAULT_DB_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        try:
            result = ingest_finnhub_news(args.db, args.now)
        except ValueError as exc:
            result = _error(Path(args.db), "invalid_now", str(exc))
        code = 0 if result["ok"] else 2 if result["error_code"] == "missing_finnhub_api_key" else 1
    elif args.command == "status":
        result, code = {"command": "status", **database_status(args.db)}, None
        code = 0 if result["ok"] else 1
    else:
        result, code = {"command": "quick-check", **quick_check(args.db)}, None
        code = 0 if result["ok"] else 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
