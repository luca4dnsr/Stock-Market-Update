"""Yahoo Finance 수치와 Finnhub 뉴스로 한국어 시장 인사이트를 만든다.

주가·시가총액·섹터·영문 사업 설명은 기존 Yahoo Finance 파이프라인을 유지한다.
뉴스 근거는 Finnhub만 사용하며 Gemini에는 검색 도구를 제공하지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    AI_INSIGHTS_CACHE_FILE,
    AI_INSIGHTS_CACHE_VERSION,
    FINNHUB_API_BASE_URL,
    FINNHUB_MARKET_NEWS_CATEGORY,
    FINNHUB_MARKET_NEWS_MAX_INPUT,
    FINNHUB_NEWS_MAX_PER_TICKER,
    FINNHUB_NEWS_REQUEST_DELAY_SEC,
    FINNHUB_REQUEST_TIMEOUT_SEC,
    GEMINI_API_URL,
    GEMINI_INSIGHTS_BATCH_SIZE,
    GEMINI_INSIGHTS_MAX_TOKENS,
    GEMINI_INSIGHTS_TIMEOUT_SEC,
    GEMINI_MODEL,
    MARKET_MAX_NEWS_SOURCES,
    MARKET_MIN_NEWS_SOURCES,
    NEWS_WINDOW_DAYS_AFTER,
    NEWS_WINDOW_DAYS_BEFORE,
    NIM_API_URL,
    NIM_CONNECT_TIMEOUT_SEC,
    NIM_GPT_OSS_MODEL,
    NIM_INSIGHTS_MAX_TOKENS,
    NIM_READ_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

LIMITED_REASON = "당일 전후의 종목 직접 관련 뉴스·공시 근거를 충분히 확인하지 못했습니다."
LIMITED_MARKET_INTERPRETATION = (
    "당일 전후 시황 기사 3건 이상을 검증하지 못했습니다. "
    "아래 해석은 가격·시장 폭·섹터 수익률에 한정됩니다."
)

SYSTEM_PROMPT = """당신은 미국 주식 리서치 보조자입니다.
모든 결과를 한국어로 작성하십시오. 제공된 Yahoo Finance 수치·기업 설명과 Finnhub 기사만 사용하십시오.
웹 검색·외부 지식·기사에 없는 사실을 사용하지 마십시오. 종목 등락 이유는 Finnhub 기사 제목 또는
요약에 직접 촉매가 명시되고 실제 article_id를 근거로 제시할 수 있을 때만 작성하십시오.
시장 시황 해석도 제공된 시장 기사 3건 이상에서만 작성하십시오. JSON 외 텍스트를 반환하지 마십시오."""

STOCK_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {"type": "STRING"},
                    "business_ko": {"type": "STRING"},
                    "move_reason_ko": {"type": "STRING"},
                    "evidence_status": {
                        "type": "STRING",
                        "enum": ["verified", "limited"],
                    },
                    "source_article_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": [
                    "ticker",
                    "business_ko",
                    "move_reason_ko",
                    "evidence_status",
                    "source_article_ids",
                ],
            },
        },
    },
    "required": ["items"],
}

MARKET_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "observation": {"type": "STRING"},
        "interpretation": {"type": "STRING"},
        "source_article_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["headline", "observation", "interpretation", "source_article_ids"],
}


def _load_cache() -> dict:
    try:
        return json.loads(AI_INSIGHTS_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    AI_INSIGHTS_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parse_json(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    cleaned = str(content).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(cleaned[start : end + 1])


def _chunked(items: list[dict], chunk_size: int) -> list[list[dict]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _news_window(data_date: str) -> tuple[date, date]:
    session_date = datetime.strptime(data_date, "%Y-%m-%d").date()
    return (
        session_date - timedelta(days=NEWS_WINDOW_DAYS_BEFORE),
        session_date + timedelta(days=NEWS_WINDOW_DAYS_AFTER),
    )


def _canonical_ticker(value: str) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _finnhub_symbol(ticker: str) -> str:
    return {"BRK-B": "BRK.B", "BF-B": "BF.B"}.get(ticker, ticker)


def _finnhub_get(path: str, params: dict) -> list[dict] | dict:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY가 설정되지 않았습니다.")
    response = requests.get(
        f"{FINNHUB_API_BASE_URL}/{path.lstrip('/')}",
        params={**params, "token": api_key},
        headers={"Accept": "application/json"},
        timeout=FINNHUB_REQUEST_TIMEOUT_SEC,
    )
    if not response.ok:
        raise RuntimeError(f"Finnhub HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    if not isinstance(payload, (list, dict)):
        raise ValueError("Finnhub 응답 형식이 올바르지 않습니다.")
    return payload


def _published_in_new_york(raw_timestamp) -> tuple[datetime, date] | None:
    try:
        published_at = datetime.fromtimestamp(
            int(raw_timestamp), tz=ZoneInfo("America/New_York")
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return published_at, published_at.date()


def _normalise_company_articles(
    ticker: str, raw_items: list[dict], start: date, end: date
) -> list[dict]:
    expected_ticker = _canonical_ticker(ticker)
    articles: list[dict] = []
    seen_ids: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        article_id = str(raw.get("id") or "").strip()
        headline = str(raw.get("headline") or "").strip()
        url = str(raw.get("url") or "").strip()
        published = _published_in_new_york(raw.get("datetime"))
        related = [
            _canonical_ticker(item)
            for item in str(raw.get("related") or "").split(",")
            if str(item).strip()
        ]
        if (
            not article_id
            or article_id in seen_ids
            or not headline
            or not url.startswith(("http://", "https://"))
            or not published
            or expected_ticker not in related
        ):
            continue
        published_at, published_date = published
        if not start <= published_date <= end:
            continue
        seen_ids.add(article_id)
        articles.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "published_at": published_at.isoformat(),
                "published_date": published_date.isoformat(),
                "headline": headline[:300],
                "summary": str(raw.get("summary") or "").strip()[:900],
                "source": str(raw.get("source") or "").strip()[:100],
                "url": url,
                "related": related,
            }
        )
    articles.sort(key=lambda item: item["published_at"], reverse=True)
    return articles[:FINNHUB_NEWS_MAX_PER_TICKER]


def _collect_company_news(tickers: list[str], start: date, end: date) -> dict[str, list[dict]]:
    """무료 API의 예측 가능한 요청량을 위해 종목 뉴스를 순차 수집한다."""
    result: dict[str, list[dict]] = {}
    for index, ticker in enumerate(tickers):
        if index:
            time.sleep(FINNHUB_NEWS_REQUEST_DELAY_SEC)
        try:
            raw_items = _finnhub_get(
                "company-news",
                {
                    "symbol": _finnhub_symbol(ticker),
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                },
            )
            result[ticker] = _normalise_company_articles(
                ticker, raw_items if isinstance(raw_items, list) else [], start, end
            )
        except Exception as exc:
            logger.warning("Finnhub 종목 뉴스 조회 실패 (%s): %s", ticker, exc)
            result[ticker] = []
    return result


def _collect_market_news(start: date, end: date) -> list[dict]:
    raw_items = _finnhub_get("news", {"category": FINNHUB_MARKET_NEWS_CATEGORY})
    articles: list[dict] = []
    seen_ids: set[str] = set()
    for raw in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(raw, dict):
            continue
        article_id = str(raw.get("id") or "").strip()
        headline = str(raw.get("headline") or "").strip()
        url = str(raw.get("url") or "").strip()
        published = _published_in_new_york(raw.get("datetime"))
        if (
            not article_id
            or article_id in seen_ids
            or not headline
            or not url.startswith(("http://", "https://"))
            or not published
        ):
            continue
        published_at, published_date = published
        if not start <= published_date <= end:
            continue
        seen_ids.add(article_id)
        articles.append(
            {
                "article_id": article_id,
                "published_at": published_at.isoformat(),
                "published_date": published_date.isoformat(),
                "headline": headline[:300],
                "summary": str(raw.get("summary") or "").strip()[:900],
                "source": str(raw.get("source") or "").strip()[:100],
                "url": url,
                "related": [
                    _canonical_ticker(item)
                    for item in str(raw.get("related") or "").split(",")
                    if str(item).strip()
                ],
            }
        )
    articles.sort(key=lambda item: item["published_at"], reverse=True)
    return articles[:FINNHUB_MARKET_NEWS_MAX_INPUT]


def _request_gemini_json(prompt: str, response_schema: dict) -> dict:
    """검색 도구 없이, 제공된 Finnhub 데이터만 해석하도록 Gemini를 호출한다."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_INSIGHTS_MAX_TOKENS,
            "responseFormat": {
                "text": {"mimeType": "APPLICATION_JSON", "schema": response_schema}
            },
        },
    }
    response = requests.post(
        GEMINI_API_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=(15, GEMINI_INSIGHTS_TIMEOUT_SEC),
    )
    if not response.ok:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")
    candidates = response.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini 응답에 후보가 없습니다.")
    content = "".join(
        str(part.get("text", ""))
        for part in candidates[0].get("content", {}).get("parts", [])
    )
    try:
        return _parse_json(content)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Gemini JSON 파싱 실패 원문 일부: %r", content[:500])
        raise ValueError(f"Gemini JSON 파싱 실패: {exc}") from exc


def _request_nim_json(model: str, system_prompt: str, prompt: str) -> dict:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY가 설정되지 않았습니다.")
    response = requests.post(
        NIM_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": NIM_INSIGHTS_MAX_TOKENS,
            "stream": False,
        },
        timeout=(NIM_CONNECT_TIMEOUT_SEC, NIM_READ_TIMEOUT_SEC),
    )
    if not response.ok:
        raise RuntimeError(f"NIM HTTP {response.status_code}: {response.text[:500]}")
    try:
        return _parse_json(response.json()["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"NIM JSON 파싱 실패: {exc}") from exc


def _business_fallback(record: dict) -> str:
    sector_labels = {
        "Communication Services": "통신·미디어·인터넷 서비스",
        "Consumer Discretionary": "소비재·자동차·레저",
        "Consumer Staples": "필수소비재",
        "Energy": "에너지",
        "Financials": "금융",
        "Health Care": "헬스케어·바이오",
        "Industrials": "산업재·운송·방산",
        "Information Technology": "정보기술",
        "Materials": "소재",
        "Real Estate": "부동산",
        "Utilities": "유틸리티",
    }
    sector = sector_labels.get(str(record.get("sector", "")), "해당 산업")
    return f"{sector} 분야를 영위하는 미국 상장 기업"


def _stock_prompt(items: list[dict], data_date: str, start: date, end: date) -> str:
    payload = {"data_date": data_date, "stocks": items}
    return f"""미국 거래일은 {data_date}입니다.
아래 Yahoo Finance 기업 정보와 Finnhub 기사만 사용하십시오. 웹 검색은 하지 마십시오.

허용 기사 발행일은 {start.isoformat()}~{end.isoformat()}입니다. 각 종목의 finnhub_articles는
코드가 이미 이 날짜 범위·원문 URL·관련 티커 조건을 확인한 기사입니다.

business_ko는 business_source_en을 바탕으로 70자 이내 한국어 한 문장으로 작성하십시오.
move_reason_ko는 기사 제목 또는 요약에 해당 종목의 직접 촉매(실적, 전망, 계약, 규제, M&A,
제품, 가이던스 등)가 명시된 경우에만 140자 이내로 작성하십시오. 이때 evidence_status는
verified로, source_article_ids에는 실제 사용한 해당 종목의 article_id를 1개 이상 넣으십시오.
그 외에는 evidence_status를 limited로 하고 move_reason_ko에는 빈 문자열, source_article_ids에는
빈 배열을 넣으십시오. 가격 변동이나 일반 시장 분위기를 원인으로 추정하지 마십시오.

입력 데이터:
{json.dumps(payload, ensure_ascii=False)}"""


def _market_prompt(
    base_market_summary: dict, articles: list[dict], data_date: str, start: date, end: date
) -> str:
    payload = {"market_data": base_market_summary, "finnhub_market_articles": articles}
    return f"""미국 거래일은 {data_date}입니다.
아래 시장 수치와 Finnhub 일반 시장 기사만 사용하십시오. 웹 검색이나 외부 지식은 사용하지 마십시오.

기사는 {start.isoformat()}~{end.isoformat()}에 발행된 것만 입력되어 있습니다. 미국 증시 전체 또는
주요 섹터 움직임과 직접 관련된 서로 다른 기사 3~5건을 골라 headline, observation, interpretation을
작성하십시오. source_article_ids에는 실제 사용한 article_id를 3~5개 넣으십시오.
기사 3건 이상으로 뒷받침할 수 없으면 headline·observation에는 시장 수치 관측만 쓰고,
interpretation에는 정확히 '{LIMITED_MARKET_INTERPRETATION}'을 넣으며 source_article_ids는 빈 배열로 하십시오.
기사에 없는 인과관계, 투자 조언, 단순 가격 변동의 원인 추정은 금지합니다.

입력 데이터:
{json.dumps(payload, ensure_ascii=False)}"""


def _fallback_stock_prompt(items: list[dict]) -> str:
    return f"""아래 기업의 영문 사업 설명만 바탕으로 business_ko를 한국어 한 문장(70자 이내)으로 작성하십시오.
뉴스나 시장 상황을 추정하지 말고, 반드시 {{"items":[{{"ticker":"...","business_ko":"..."}}]}} JSON만 반환하십시오.

{json.dumps(items, ensure_ascii=False)}"""


def _fallback_market_prompt(base_market_summary: dict) -> str:
    return f"""아래 수치 데이터만 바탕으로 한국어 headline과 observation을 작성하십시오.
뉴스·실적·정책 등의 원인을 추정하지 말고, 반드시 {{"headline":"...","observation":"..."}} JSON만 반환하십시오.

{json.dumps(base_market_summary, ensure_ascii=False)}"""


def _validated_sources(
    source_ids, articles: list[dict], start: date, end: date, ticker: str | None = None
) -> list[dict]:
    """LLM이 참조한 ID를 실제 Finnhub 기사·거래일 창과 다시 대조한다."""
    lookup = {str(article["article_id"]): article for article in articles}
    verified: list[dict] = []
    seen: set[str] = set()
    for source_id in source_ids if isinstance(source_ids, list) else []:
        article = lookup.get(str(source_id))
        if not article or article["article_id"] in seen:
            continue
        try:
            published_date = datetime.strptime(article["published_date"], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if not start <= published_date <= end:
            continue
        if ticker and _canonical_ticker(ticker) not in article.get("related", []):
            continue
        seen.add(article["article_id"])
        verified.append(article)
    return verified


def _normalise_stock_batch(
    generated: dict, expected_items: list[dict], start: date, end: date
) -> dict[str, dict]:
    expected = {str(item["ticker"]): item for item in expected_items}
    received = {
        str(item.get("ticker", "")).strip(): item
        for item in generated.get("items", [])
        if isinstance(item, dict) and str(item.get("ticker", "")).strip() in expected
    }
    entries: dict[str, dict] = {}
    for ticker, source_item in expected.items():
        raw = received.get(ticker, {})
        business = str(raw.get("business_ko", "")).strip() or _business_fallback(source_item)
        sources = _validated_sources(
            raw.get("source_article_ids"), source_item.get("finnhub_articles", []), start, end, ticker
        )
        is_verified = (
            raw.get("evidence_status") == "verified"
            and bool(sources)
            and bool(str(raw.get("move_reason_ko", "")).strip())
        )
        entries[ticker] = {
            "business_summary": business[:140],
            "move_reason": str(raw.get("move_reason_ko", "")).strip()[:280]
            if is_verified
            else LIMITED_REASON,
            "source_urls": [source["url"] for source in sources] if is_verified else [],
            "source_titles": [source["headline"] for source in sources] if is_verified else [],
            "provider": "Gemini + Finnhub" if is_verified else "Gemini (Finnhub 근거 부족)",
        }
    return entries


def _fallback_stock_entries(items: list[dict]) -> dict[str, dict]:
    provider_name = "NVIDIA NIM GPT-OSS 120B"
    try:
        generated = _request_nim_json(
            NIM_GPT_OSS_MODEL,
            "제공된 영문 사업 설명만 번역·요약하고 뉴스 원인을 추정하지 마십시오. JSON만 답하십시오.",
            _fallback_stock_prompt(items),
        )
        expected = {str(item["ticker"]): item for item in items}
        received = {
            str(item.get("ticker", "")).strip(): item
            for item in generated.get("items", [])
            if isinstance(item, dict)
            and str(item.get("ticker", "")).strip() in expected
            and str(item.get("business_ko", "")).strip()
        }
        if set(received) != set(expected):
            raise ValueError("GPT-OSS 응답에 일부 종목 사업 설명이 없습니다.")
        return {
            ticker: {
                "business_summary": str(received[ticker]["business_ko"]).strip()[:140],
                "move_reason": LIMITED_REASON,
                "source_urls": [],
                "source_titles": [],
                "provider": provider_name,
            }
            for ticker in expected
        }
    except Exception as exc:
        logger.warning("%s fallback 실패, 규칙 기반 제한 문구를 사용합니다: %s", provider_name, exc)
        return {}


def _limited_market_summary(base_market_summary: dict) -> dict:
    return {
        "headline": str(base_market_summary.get("headline", "당일 시장 흐름")),
        "observation": str(base_market_summary.get("observation", "")),
        "interpretation": LIMITED_MARKET_INTERPRETATION,
        "disclaimer": "뉴스 근거 확인이 제한된 자동 요약이며 투자 조언이 아닙니다.",
        "source_urls": [],
        "source_titles": [],
    }


def _fallback_market_summary(base_market_summary: dict) -> dict:
    provider_name = "NVIDIA NIM GPT-OSS 120B"
    try:
        generated = _request_nim_json(
            NIM_GPT_OSS_MODEL,
            "제공된 수치만 사용하고 뉴스·정책·실적 원인을 추정하지 마십시오. JSON만 답하십시오.",
            _fallback_market_prompt(base_market_summary),
        )
        headline = str(generated.get("headline", "")).strip()
        observation = str(generated.get("observation", "")).strip()
        if not headline or not observation:
            raise ValueError("GPT-OSS 시황 응답에 필수 문구가 없습니다.")
        return {
            "headline": headline[:300],
            "observation": observation[:600],
            "interpretation": LIMITED_MARKET_INTERPRETATION,
            "disclaimer": (
                f"{provider_name}가 가격·시장 폭·섹터 수익률만 정리한 자동 요약이며 "
                "뉴스 근거 기반 해석이나 투자 조언이 아닙니다."
            ),
            "source_urls": [],
            "source_titles": [],
        }
    except Exception as exc:
        logger.warning("%s 시황 fallback 실패, 규칙 기반 제한 문구를 사용합니다: %s", provider_name, exc)
        return _limited_market_summary(base_market_summary)


def _research_market_summary(
    base_market_summary: dict, articles: list[dict], data_date: str, start: date, end: date
) -> dict:
    try:
        generated = _request_gemini_json(
            _market_prompt(base_market_summary, articles, data_date, start, end),
            MARKET_RESPONSE_SCHEMA,
        )
        sources = _validated_sources(generated.get("source_article_ids"), articles, start, end)
        fields = ("headline", "observation", "interpretation")
        if not all(str(generated.get(field, "")).strip() for field in fields):
            raise ValueError("Gemini 시황 응답에 필수 문구가 없습니다.")
        if len(sources) < MARKET_MIN_NEWS_SOURCES:
            raise ValueError(f"검증된 Finnhub 시황 기사 수 부족: {len(sources)}건")
        sources = sources[:MARKET_MAX_NEWS_SOURCES]
        return {
            field: str(generated[field]).strip()[:600] for field in fields
        } | {
            "disclaimer": (
                f"Finnhub에서 확인된 거래일 전후 기사 {len(sources)}건을 바탕으로 한 자동 요약이며 "
                "투자 조언이 아닙니다."
            ),
            "source_urls": [source["url"] for source in sources],
            "source_titles": [source["headline"] for source in sources],
        }
    except Exception as exc:
        logger.warning("Gemini + Finnhub 시황 조사 실패, NIM fallback을 시도합니다: %s", exc)
        return _fallback_market_summary(base_market_summary)


def enrich_with_ai(
    stocks_df: pd.DataFrame,
    data_date: str,
    base_market_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    """상·하위 각 20종목을 Finnhub 뉴스로 검증하고, 없으면 제한 문구를 남긴다."""
    result = stocks_df.copy()
    cache = _load_cache()
    tickers = result["ticker"].astype(str).tolist()
    cache_keys = {
        ticker: f"{AI_INSIGHTS_CACHE_VERSION}:{data_date}:{ticker}" for ticker in tickers
    }
    missing = [ticker for ticker in tickers if cache_keys[ticker] not in cache]
    start, end = _news_window(data_date)
    records = result.set_index(result["ticker"].astype(str)).to_dict("index")

    if missing:
        try:
            news_map = _collect_company_news(missing, start, end)
        except Exception as exc:
            logger.warning("Finnhub 종목 뉴스 수집을 시작하지 못했습니다: %s", exc)
            news_map = {ticker: [] for ticker in missing}

        items = [
            {
                "ticker": ticker,
                "name": records[ticker].get("name", ""),
                "sector": records[ticker].get("sector", ""),
                "return_1d": records[ticker].get("return_1d"),
                "business_source_en": records[ticker].get("business_summary", ""),
                "finnhub_articles": news_map.get(ticker, []),
            }
            for ticker in missing
        ]
        for batch_index, batch in enumerate(
            _chunked(items, GEMINI_INSIGHTS_BATCH_SIZE), start=1
        ):
            try:
                generated = _request_gemini_json(
                    _stock_prompt(batch, data_date, start, end), STOCK_RESPONSE_SCHEMA
                )
                entries = _normalise_stock_batch(generated, batch, start, end)
                for ticker, entry in entries.items():
                    cache[cache_keys[ticker]] = entry
                _save_cache(cache)
                verified_count = sum(
                    entry["provider"] == "Gemini + Finnhub" for entry in entries.values()
                )
                logger.info(
                    "Finnhub 종목 뉴스 해석 완료: 묶음 %d (%d/%d건 근거 확인)",
                    batch_index,
                    verified_count,
                    len(batch),
                )
            except Exception as exc:
                logger.warning(
                    "Gemini 종목 해석 실패 (묶음 %d), GPT-OSS fallback을 시도합니다: %s",
                    batch_index,
                    exc,
                )
                entries = _fallback_stock_entries(batch)
                for ticker, entry in entries.items():
                    cache[cache_keys[ticker]] = entry
                if entries:
                    _save_cache(cache)

    result["business_summary"] = [
        cache.get(cache_keys[ticker], {}).get("business_summary")
        or _business_fallback(records[ticker])
        for ticker in tickers
    ]
    result["move_reason"] = [
        cache.get(cache_keys[ticker], {}).get("move_reason") or LIMITED_REASON
        for ticker in tickers
    ]

    market_key = f"{AI_INSIGHTS_CACHE_VERSION}:market:{data_date}"
    market_summary = cache.get(market_key)
    if not market_summary:
        try:
            market_articles = _collect_market_news(start, end)
        except Exception as exc:
            logger.warning("Finnhub 시장 뉴스 조회 실패: %s", exc)
            market_articles = []
        market_summary = _research_market_summary(
            base_market_summary, market_articles, data_date, start, end
        )
        if market_summary.get("source_urls"):
            cache[market_key] = market_summary
            _save_cache(cache)

    return result, dict(market_summary)
