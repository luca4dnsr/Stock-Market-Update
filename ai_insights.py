"""기업 설명과 당일 뉴스 근거를 한국어로 정리한다.

가격·기업 기본정보·Yahoo Finance 헤드라인은 기존 파이프라인에서 그대로
가져온다. 등락 이유와 장 전체 해석은 Gemini Google Search Grounding으로
별도 검색한 기사에 근거가 있을 때만 작성한다.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import yfinance as yf

from config import (
    AI_INSIGHTS_CACHE_FILE,
    AI_INSIGHTS_CACHE_VERSION,
    GEMINI_API_URL,
    GEMINI_GROUNDING_TIMEOUT_SEC,
    GEMINI_INSIGHTS_MAX_TOKENS,
    GEMINI_MODEL,
    GEMINI_SEARCH_BATCH_SIZE,
    MARKET_MAX_GROUNDED_SOURCES,
    MARKET_MIN_GROUNDED_SOURCES,
    NIM_API_URL,
    NIM_CONNECT_TIMEOUT_SEC,
    NIM_GPT_OSS_MODEL,
    NIM_INSIGHTS_MAX_TOKENS,
    NIM_LLAMA_MODEL,
    NIM_READ_TIMEOUT_SEC,
    NEWS_WINDOW_DAYS_AFTER,
    NEWS_WINDOW_DAYS_BEFORE,
    PROFILE_FETCH_WORKERS,
)

logger = logging.getLogger(__name__)

LIMITED_REASON = "당일 전후의 종목 직접 관련 뉴스·공시 근거를 충분히 확인하지 못했습니다."
LIMITED_MARKET_INTERPRETATION = (
    "당일 전후 시황 기사 3건 이상을 검증하지 못했습니다. "
    "아래 해석은 가격·시장 폭·섹터 수익률에 한정됩니다."
)

SYSTEM_PROMPT = """당신은 미국 주식 리서치 보조자입니다.
모든 결과는 한국어로 작성하고, 제공된 Yahoo Finance 데이터는 보조 단서로만 사용하십시오.
등락 원인과 시장 해석에는 반드시 Google Search로 직접 찾은 공개 기사·공시만 사용하십시오.
가격 변동 자체를 원인으로 추정하거나, 종목과 직접 관계 없는 업종·거시 기사를 개별 종목의 원인으로 쓰면 안 됩니다.
확인하지 못한 사실을 채우지 말고, 요구한 JSON 이외의 텍스트를 쓰지 마십시오."""

STOCK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "business_ko": {"type": "string"},
                    "move_reason_ko": {"type": "string"},
                    "evidence_status": {
                        "type": "string",
                        "enum": ["grounded", "limited"],
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "published_date": {"type": "string"},
                            },
                            "required": ["url", "title", "published_date"],
                        },
                    },
                },
                "required": [
                    "ticker",
                    "business_ko",
                    "move_reason_ko",
                    "evidence_status",
                    "sources",
                ],
            },
        },
    },
    "required": ["items"],
}

MARKET_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "observation": {"type": "string"},
        "interpretation": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "published_date": {"type": "string"},
                },
                "required": ["url", "title", "published_date"],
            },
        },
    },
    "required": ["headline", "observation", "interpretation", "sources"],
}

FALLBACK_STOCK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "business_ko": {"type": "string"},
                },
                "required": ["ticker", "business_ko"],
            },
        },
    },
    "required": ["items"],
}

FALLBACK_MARKET_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "observation": {"type": "string"},
    },
    "required": ["headline", "observation"],
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


def _unwrap_url(value) -> str:
    if isinstance(value, dict):
        return str(value.get("url", ""))
    return str(value or "")


def _get_ticker_news(ticker: str) -> tuple[str, list[dict]]:
    """기존 Yahoo Finance 뉴스 피드를 보조 단서로 유지한다."""
    try:
        raw_items = yf.Ticker(ticker).get_news(count=3)
    except Exception as exc:
        logger.warning("Yahoo Finance 뉴스 조회 실패 (%s): %s", ticker, exc)
        return ticker, []

    news = []
    for item in raw_items:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = str(content.get("title") or item.get("title") or "").strip()
        if not title:
            continue
        provider = content.get("provider", item.get("publisher", ""))
        if isinstance(provider, dict):
            provider = provider.get("displayName", "")
        url = _unwrap_url(
            content.get("canonicalUrl")
            or content.get("clickThroughUrl")
            or item.get("link")
        )
        news.append(
            {"title": title[:300], "publisher": str(provider or "")[:80], "url": url}
        )
    return ticker, news


def _collect_news(tickers: list[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as executor:
        futures = {executor.submit(_get_ticker_news, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker, news = future.result()
            result[ticker] = news
    return result


def _parse_json(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.IGNORECASE
    )
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(cleaned[start : end + 1])


def _canonical_url(value: str) -> str:
    """모델이 추적 파라미터를 생략해도 Grounding 원문과 비교할 수 있게 한다."""
    try:
        parts = urlsplit(str(value).strip())
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def _grounding_metadata(candidate: dict) -> tuple[dict[str, dict], list[str]]:
    """Gemini의 실제 Google Search 결과 URL만 추출한다."""
    metadata = candidate.get("groundingMetadata") or candidate.get("grounding_metadata") or {}
    grounded: dict[str, dict] = {}
    for chunk in metadata.get("groundingChunks") or metadata.get("grounding_chunks") or []:
        web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
        url = str(web.get("uri") or web.get("url") or "").strip()
        canonical = _canonical_url(url)
        if canonical and canonical not in grounded:
            grounded[canonical] = {"url": url, "title": str(web.get("title") or "").strip()}

    queries = metadata.get("webSearchQueries") or metadata.get("web_search_queries") or []
    return grounded, [str(query).strip() for query in queries if str(query).strip()]


def _request_gemini_grounded(prompt: str, response_schema: dict) -> tuple[dict, dict[str, dict], list[str]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        # Gemini 3 계열은 Google Search Grounding과 Structured Output을 함께 지원한다.
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_INSIGHTS_MAX_TOKENS,
            "responseFormat": {
                "text": {"mimeType": "application/json", "schema": response_schema}
            },
        },
    }
    response = requests.post(
        GEMINI_API_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=(15, GEMINI_GROUNDING_TIMEOUT_SEC),
    )
    if not response.ok:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")

    candidates = response.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini 응답에 후보가 없습니다.")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts)
    grounded, queries = _grounding_metadata(candidate)
    try:
        return _parse_json(content), grounded, queries
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Gemini JSON 파싱 실패 원문 일부: %r", content[:500])
        raise ValueError(f"Gemini JSON 파싱 실패: {exc}") from exc


def _request_nim_json(model: str, system_prompt: str, prompt: str) -> dict:
    """NVIDIA NIM의 OpenAI 호환 Chat Completions로 JSON만 요청한다."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY가 설정되지 않았습니다.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": NIM_INSIGHTS_MAX_TOKENS,
        "stream": False,
    }
    response = requests.post(
        NIM_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(NIM_CONNECT_TIMEOUT_SEC, NIM_READ_TIMEOUT_SEC),
    )
    if not response.ok:
        raise RuntimeError(f"NIM HTTP {response.status_code}: {response.text[:500]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_json(content)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("NIM JSON 파싱 실패 원문 일부: %r", str(locals().get("content", ""))[:500])
        raise ValueError(f"NIM JSON 파싱 실패: {exc}") from exc


def _chunked(items: list[dict], chunk_size: int) -> list[list[dict]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _news_window(data_date: str) -> tuple[date, date]:
    session_date = datetime.strptime(data_date, "%Y-%m-%d").date()
    return (
        session_date - timedelta(days=NEWS_WINDOW_DAYS_BEFORE),
        session_date + timedelta(days=NEWS_WINDOW_DAYS_AFTER),
    )


def _parse_source_date(value) -> date | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y-%m-%d").date()
    except ValueError:
        return None


def _verified_sources(
    reported_sources, grounded_sources: dict[str, dict], start: date, end: date
) -> list[dict]:
    """모델 응답을 실제 Grounding URL·거래일 전후 기사일자로 이중 확인한다."""
    verified = []
    seen = set()
    for source in reported_sources if isinstance(reported_sources, list) else []:
        if not isinstance(source, dict):
            continue
        canonical = _canonical_url(str(source.get("url", "")))
        published = _parse_source_date(source.get("published_date"))
        if (
            not canonical
            or canonical in seen
            or canonical not in grounded_sources
            or not published
            or not start <= published <= end
        ):
            continue
        seen.add(canonical)
        ground = grounded_sources[canonical]
        verified.append(
            {
                "url": ground["url"],
                "title": str(source.get("title") or ground["title"] or "기사").strip()[:140],
                "published_date": published.isoformat(),
            }
        )
    return verified


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
    return f"""미국장 거래일은 {data_date}입니다. Google Search를 사용해 아래 각 종목을 독립적으로 검색하십시오.

허용 기사일자: {start.isoformat()}~{end.isoformat()}(거래일 전후). 각 종목의 기사·공시가 종목명을 직접 언급하고, 당일 등락과 관련된 구체적 사건(실적, 전망, 계약, 규제, M&A, 제품, 애널리스트 조정 등)을 설명할 때만 grounded를 쓰십시오.
Yahoo Finance 헤드라인은 검색어를 정하는 보조 단서이며, 그것만으로 결론을 내리면 안 됩니다.

각 종목의 sources에는 Google Search로 실제 확인한 기사만 URL·제목·발행일(YYYY-MM-DD)과 함께 넣으십시오. grounded에는 위 기간의 검증 가능한 직접 관련 기사 최소 1건이 필수입니다. 그렇지 않으면 evidence_status를 limited로, move_reason_ko를 정확히 '{LIMITED_REASON}'로 쓰십시오.
business_ko는 제공된 영문 사업 설명을 바탕으로 한 한국어 한 문장(70자 이내)입니다. grounded 사유는 1~2문장(140자 이내)입니다.

입력 데이터:\n{json.dumps(payload, ensure_ascii=False)}"""


def _market_prompt(base_market_summary: dict, data_date: str, start: date, end: date) -> str:
    return f"""미국장 거래일은 {data_date}입니다. Google Search로 그날 미국 증시를 움직인 핵심 뉴스를 직접 검색하십시오.

{start.isoformat()}~{end.isoformat()}에 발행됐고 미국 증시 전체 또는 주요 섹터 흐름과 직접 관련 있는 서로 다른 신뢰 가능한 기사·공시를 3~5건 확인해야 합니다. 거시지표, 중앙은행, 국채금리, 무역·정책, 실적, 지정학 등 실제 촉매를 가격·섹터 데이터와 분리해서 서술하십시오.
sources에는 실제 Google Search 결과만 URL·제목·발행일(YYYY-MM-DD)로 3~5건 넣으십시오. 기사 근거가 3건 미만이면 추정하지 말고 observation에는 가격 데이터만, interpretation에는 '{LIMITED_MARKET_INTERPRETATION}'를 쓰십시오.

아래는 로컬에서 계산한 시장 데이터입니다. 수치는 그대로 활용하되, 기사에 없는 인과관계는 만들지 마십시오.
{json.dumps(base_market_summary, ensure_ascii=False)}"""


def _fallback_stock_prompt(items: list[dict]) -> str:
    return f"""아래 기업의 제공된 영문 사업 설명만 바탕으로 business_ko를 한국어 한 문장(70자 이내)으로 작성하십시오.
뉴스를 검색하거나 제공된 Yahoo Finance 헤드라인만으로 등락 원인을 추정하지 마십시오.
반드시 {{"items":[{{"ticker":"...","business_ko":"..."}}]}} JSON만 반환하십시오.

{json.dumps(items, ensure_ascii=False)}"""


def _fallback_market_prompt(base_market_summary: dict) -> str:
    return f"""아래 수치 데이터만 바탕으로 한국어 headline과 observation을 작성하십시오.
뉴스·실적·정책 등 원인을 추정하지 말고, 반드시 {{"headline":"...","observation":"..."}} JSON만 반환하십시오.

{json.dumps(base_market_summary, ensure_ascii=False)}"""


def _normalise_stock_batch(
    generated: dict,
    expected_items: list[dict],
    grounded_sources: dict[str, dict],
    queries: list[str],
    start: date,
    end: date,
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
        sources = _verified_sources(raw.get("sources"), grounded_sources, start, end)
        is_grounded = (
            raw.get("evidence_status") == "grounded"
            and bool(queries)
            and bool(sources)
            and bool(str(raw.get("move_reason_ko", "")).strip())
        )
        entries[ticker] = {
            "business_summary": business[:140],
            "move_reason": str(raw.get("move_reason_ko", "")).strip()[:280]
            if is_grounded
            else LIMITED_REASON,
            "source_urls": [source["url"] for source in sources] if is_grounded else [],
            "source_titles": [source["title"] for source in sources] if is_grounded else [],
            "provider": "Gemini Google Search" if is_grounded else "Gemini Google Search (limited)",
        }
    return entries


def _fallback_stock_entries(items: list[dict]) -> dict[str, dict]:
    """웹 검색이 불가한 NIM 모델은 사업 설명만 보완하고 등락 이유는 제한 처리한다."""
    system_prompt = (
        "당신은 한국어 금융 데이터 정리 보조자입니다. 제공된 영문 사업 설명만 번역·요약하십시오. "
        "뉴스 원인이나 시장 상황을 추정하지 말고 JSON만 답하십시오."
    )
    expected = {str(item["ticker"]): item for item in items}
    providers = [
        ("NVIDIA NIM Llama 3.3 70B", NIM_LLAMA_MODEL),
        ("NVIDIA NIM GPT-OSS 120B", NIM_GPT_OSS_MODEL),
    ]
    for provider_name, model in providers:
        try:
            generated = _request_nim_json(
                model, system_prompt, _fallback_stock_prompt(items)
            )
            received = {
                str(item.get("ticker", "")).strip(): item
                for item in generated.get("items", [])
                if isinstance(item, dict)
                and str(item.get("ticker", "")).strip() in expected
                and str(item.get("business_ko", "")).strip()
            }
            if set(received) != set(expected):
                raise ValueError("NIM 응답에 일부 종목 사업 설명이 없습니다.")
            logger.info("Gemini 실패 묶음의 사업 설명 보완 완료: %s", provider_name)
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
            logger.warning("%s fallback 실패, 다음 모델을 시도합니다: %s", provider_name, exc)
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
    """NIM fallback은 수치 관측만 다듬으며 뉴스 인과 해석은 허용하지 않는다."""
    system_prompt = (
        "당신은 한국어 금융 데이터 정리 보조자입니다. 제공된 수치만 사용하고 뉴스·정책·실적의 "
        "원인을 추정하지 마십시오. JSON만 답하십시오."
    )
    providers = [
        ("NVIDIA NIM Llama 3.3 70B", NIM_LLAMA_MODEL),
        ("NVIDIA NIM GPT-OSS 120B", NIM_GPT_OSS_MODEL),
    ]
    for provider_name, model in providers:
        try:
            generated = _request_nim_json(
                model, system_prompt, _fallback_market_prompt(base_market_summary)
            )
            headline = str(generated.get("headline", "")).strip()
            observation = str(generated.get("observation", "")).strip()
            if not headline or not observation:
                raise ValueError("NIM 시황 응답에 필수 문구가 없습니다.")
            logger.info("Gemini 실패 시황의 수치 관측 보완 완료: %s", provider_name)
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
            logger.warning("%s 시황 fallback 실패, 다음 모델을 시도합니다: %s", provider_name, exc)
    return _limited_market_summary(base_market_summary)


def _research_market_summary(
    base_market_summary: dict, data_date: str, start: date, end: date
) -> dict:
    try:
        generated, grounded, queries = _request_gemini_grounded(
            _market_prompt(base_market_summary, data_date, start, end), MARKET_RESPONSE_SCHEMA
        )
        sources = _verified_sources(generated.get("sources"), grounded, start, end)
        fields = ("headline", "observation", "interpretation")
        if not queries or not all(str(generated.get(field, "")).strip() for field in fields):
            raise ValueError("시황 검색 또는 필수 문구가 누락되었습니다.")
        if len(sources) < MARKET_MIN_GROUNDED_SOURCES:
            raise ValueError(f"검증된 시황 기사 수 부족: {len(sources)}건")

        sources = sources[:MARKET_MAX_GROUNDED_SOURCES]
        return {
            field: str(generated[field]).strip()[:600] for field in fields
        } | {
            "disclaimer": (
                f"Google Search로 확인한 당일 전후 기사 {len(sources)}건을 바탕으로 한 자동 요약이며 "
                "투자 조언이 아닙니다."
            ),
            "source_urls": [source["url"] for source in sources],
            "source_titles": [source["title"] for source in sources],
        }
    except Exception as exc:
        logger.warning("Gemini Google Search 시황 조사 실패, NIM fallback을 시도합니다: %s", exc)
        return _fallback_market_summary(base_market_summary)


def enrich_with_ai(
    stocks_df: pd.DataFrame,
    data_date: str,
    base_market_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    """상·하위 각 20개 종목을 직접 검색하고, 근거가 없으면 제한 문구를 남긴다."""
    result = stocks_df.copy()
    cache = _load_cache()
    tickers = result["ticker"].astype(str).tolist()
    cache_keys = {
        ticker: f"{AI_INSIGHTS_CACHE_VERSION}:{data_date}:{ticker}" for ticker in tickers
    }
    missing = [ticker for ticker in tickers if cache_keys[ticker] not in cache]
    start, end = _news_window(data_date)

    if missing:
        news_map = _collect_news(missing)
        records = result.set_index(result["ticker"].astype(str)).to_dict("index")
        items = [
            {
                "ticker": ticker,
                "name": records[ticker].get("name", ""),
                "sector": records[ticker].get("sector", ""),
                "return_1d": records[ticker].get("return_1d"),
                "business_source_en": records[ticker].get("business_summary", ""),
                "yahoo_finance_headlines": news_map.get(ticker, []),
            }
            for ticker in missing
        ]

        for batch_index, batch in enumerate(_chunked(items, GEMINI_SEARCH_BATCH_SIZE), start=1):
            try:
                generated, grounded, queries = _request_gemini_grounded(
                    _stock_prompt(batch, data_date, start, end), STOCK_RESPONSE_SCHEMA
                )
                entries = _normalise_stock_batch(
                    generated, batch, grounded, queries, start, end
                )
                for ticker, entry in entries.items():
                    cache[cache_keys[ticker]] = entry
                _save_cache(cache)
                grounded_count = sum(
                    entry["provider"] == "Gemini Google Search" for entry in entries.values()
                )
                logger.info(
                    "Google Search 종목 조사 완료: 묶음 %d (%d/%d건 근거 확인)",
                    batch_index,
                    grounded_count,
                    len(batch),
                )
            except Exception as exc:
                logger.warning(
                    "Gemini Google Search 종목 조사 실패 (묶음 %d), NIM fallback을 시도합니다: %s",
                    batch_index,
                    exc,
                )
                entries = _fallback_stock_entries(batch)
                for ticker, entry in entries.items():
                    cache[cache_keys[ticker]] = entry
                if entries:
                    _save_cache(cache)

    businesses, reasons = [], []
    records = result.set_index(result["ticker"].astype(str)).to_dict("index")
    for ticker in tickers:
        entry = cache.get(cache_keys[ticker], {})
        businesses.append(entry.get("business_summary") or _business_fallback(records[ticker]))
        reasons.append(entry.get("move_reason") or LIMITED_REASON)
    result["business_summary"] = businesses
    result["move_reason"] = reasons

    market_key = f"{AI_INSIGHTS_CACHE_VERSION}:market:{data_date}"
    market_summary = cache.get(market_key)
    if not market_summary:
        market_summary = _research_market_summary(base_market_summary, data_date, start, end)
        # 제한 문구는 일시적 API 오류 뒤 재시도할 수 있도록 캐시하지 않는다.
        if market_summary.get("source_urls"):
            cache[market_key] = market_summary
            _save_cache(cache)

    return result, dict(market_summary)
