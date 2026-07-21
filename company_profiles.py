"""상·하위 표에 표시할 기업별 사업 요약을 Yahoo Finance에서 가져온다."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import (
    BUSINESS_PROFILE_CACHE_DAYS,
    BUSINESS_PROFILE_CACHE_FILE,
    PROFILE_FETCH_WORKERS,
)

logger = logging.getLogger(__name__)


def _load_cache() -> dict:
    try:
        return json.loads(BUSINESS_PROFILE_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _is_fresh(entry: dict) -> bool:
    try:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        return fetched_at >= datetime.now() - timedelta(days=BUSINESS_PROFILE_CACHE_DAYS)
    except (KeyError, TypeError, ValueError):
        return False


def _one_line(text: str) -> str:
    """긴 기업 설명을 첫 문장, 최대 180자로 정리한다."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return sentence[:177].rstrip() + ("..." if len(sentence) > 180 else "")


def _fallback(record: dict) -> str:
    industry = record.get("sub_sector") or record.get("sector") or "해당 산업"
    return f"{industry} 분야의 미국 상장 기업"


def _fetch_one(ticker: str) -> tuple[str, str]:
    try:
        info = yf.Ticker(ticker).get_info()
        return ticker, _one_line(info.get("longBusinessSummary", ""))
    except Exception as exc:
        logger.warning("사업 요약 조회 실패 (%s): %s", ticker, exc)
        return ticker, ""


def add_business_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 ``business_summary`` 열을 추가한다.

    일간 보고서에 표시되는 종목만 조회하고, 30일 캐시를 우선 사용한다.
    외부 설명을 받지 못하면 GICS 세부 산업을 이용한 안전한 대체 문구를 쓴다.
    """
    result = df.copy()
    records = {str(r["ticker"]): r.to_dict() for _, r in result.iterrows()}
    cache = _load_cache()
    summaries: dict[str, str] = {}
    missing: list[str] = []

    for ticker, record in records.items():
        entry = cache.get(ticker, {})
        if _is_fresh(entry) and entry.get("summary"):
            summaries[ticker] = entry["summary"]
        else:
            missing.append(ticker)

    if missing:
        logger.info("기업 사업 요약 조회: %d개 (캐시 %d개)", len(missing), len(summaries))
        with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, ticker): ticker for ticker in missing}
            for future in as_completed(futures):
                ticker, summary = future.result()
                summaries[ticker] = summary or _fallback(records[ticker])
                cache[ticker] = {
                    "summary": summaries[ticker],
                    "fetched_at": datetime.now().isoformat(timespec="seconds"),
                }
        BUSINESS_PROFILE_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    result["business_summary"] = [
        summaries.get(str(ticker), _fallback(records[str(ticker)]))
        for ticker in result["ticker"]
    ]
    return result
