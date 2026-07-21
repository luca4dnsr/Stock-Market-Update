"""기업·시황 인사이트를 여러 AI 공급자로 생성한다.

우선순위는 Gemini 3.5 Flash, NVIDIA NIM Mistral Medium 3.5,
NVIDIA NIM GPT-OSS 120B이다. 어느 호출도 보고서 생성을 중단시키지 않으며,
모두 실패하면 수집된 시장 데이터만 사용하는 보수적 문구로 대체한다.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

from config import (
    AI_INSIGHTS_CACHE_FILE,
    AI_INSIGHTS_CACHE_VERSION,
    AI_INSIGHTS_MAX_TOKENS,
    GEMINI_API_URL,
    GEMINI_MODEL,
    NIM_API_URL,
    NIM_GPT_OSS_MODEL,
    NIM_MISTRAL_MODEL,
    PROFILE_FETCH_WORKERS,
)

logger = logging.getLogger(__name__)

FALLBACK_REASON = "당일 뉴스·공시 근거를 확인하지 못했습니다."
DISCLAIMER = (
    "AI 요약은 제공된 Yahoo Finance 뉴스 헤드라인과 시장 데이터만 근거로 하며, "
    "투자 조언이 아닙니다."
)

SYSTEM_PROMPT = """당신은 미국 주식 리서치 보조자입니다. 반드시 제공된 데이터만 사용해 한국어로 답하십시오.
기업의 당일 등락 원인은 제공된 뉴스 헤드라인에 직접 뒷받침될 때만 서술하십시오. 뉴스 근거가 없거나 제목만으로 인과관계를 판단할 수 없으면 move_reason_ko에 정확히 '당일 뉴스·공시 근거를 확인하지 못했습니다.'라고 쓰십시오. 가격 변동을 뉴스의 원인으로 추정하지 마십시오.
business_ko는 제공된 영문 사업 설명을 바탕으로 한 한국어 한 문장(70자 이내)입니다. move_reason_ko는 한국어 1~2문장(140자 이내)입니다. source_urls에는 입력으로 받은 URL만 포함하십시오.
market_summary는 입력의 수치와 종목·섹터 데이터만 근거로 작성하십시오. 출력은 지정한 JSON 스키마를 따라야 합니다."""

RESPONSE_SCHEMA = {
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
                    "source_urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ticker", "business_ko", "move_reason_ko", "source_urls"],
            },
        },
        "market_summary": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "observation": {"type": "string"},
                "interpretation": {"type": "string"},
            },
            "required": ["headline", "observation", "interpretation"],
        },
    },
    "required": ["items", "market_summary"],
}


def _load_cache() -> dict:
    try:
        return json.loads(AI_INSIGHTS_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _unwrap_url(value) -> str:
    if isinstance(value, dict):
        return str(value.get("url", ""))
    return str(value or "")


def _get_ticker_news(ticker: str) -> tuple[str, list[dict]]:
    """Yahoo Finance에서 모델에 제공할 검증 가능한 뉴스만 읽는다."""
    try:
        raw_items = yf.Ticker(ticker).get_news(count=3)
    except Exception as exc:
        logger.warning("뉴스 조회 실패 (%s): %s", ticker, exc)
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
        news.append({"title": title[:300], "publisher": str(provider or "")[:80], "url": url})
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
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(cleaned[start : end + 1])


def _request_gemini(items: list[dict], market_context: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps({"market_context": market_context, "stocks": items}, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.7,
            "maxOutputTokens": AI_INSIGHTS_MAX_TOKENS,
            "responseFormat": {
                "text": {"mimeType": "application/json", "schema": RESPONSE_SCHEMA}
            },
        },
    }
    response = requests.post(
        GEMINI_API_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=(15, 180),
    )
    if not response.ok:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")
    candidates = response.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini 응답에 후보가 없습니다.")
    parts = candidates[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts)
    return _parse_json(content)


def _request_nim(model: str, items: list[dict], market_context: dict) -> dict:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY가 설정되지 않았습니다.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"market_context": market_context, "stocks": items}, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": AI_INSIGHTS_MAX_TOKENS,
        "reasoning_effort": "low",
        "stream": False,
    }
    response = requests.post(
        NIM_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json=payload,
        timeout=(15, 180),
    )
    if not response.ok:
        raise RuntimeError(f"NIM HTTP {response.status_code}: {response.text[:500]}")
    content = response.json()["choices"][0]["message"]["content"]
    return _parse_json(content)


def _normalise_response(
    generated: dict,
    expected_tickers: list[str],
    allowed_urls: set[str],
) -> tuple[dict[str, dict], dict]:
    """완전한 종목 결과만 수용해 부분·환각 응답은 다음 제공자로 넘긴다."""
    received = {}
    for item in generated.get("items", []):
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip()
        if ticker not in expected_tickers or ticker in received:
            continue
        business = str(item.get("business_ko", "")).strip()
        reason = str(item.get("move_reason_ko", "")).strip()
        if not business or not reason:
            continue
        source_urls = [
            url for url in item.get("source_urls", [])
            if isinstance(url, str) and url in allowed_urls
        ]
        received[ticker] = {
            "business_summary": business,
            "move_reason": reason,
            "source_urls": source_urls,
        }

    missing = set(expected_tickers) - set(received)
    if missing:
        raise ValueError(f"AI 응답에 필요한 종목 결과가 없습니다: {', '.join(sorted(missing))}")

    market = generated.get("market_summary", {})
    if not isinstance(market, dict) or not all(str(market.get(k, "")).strip() for k in ("headline", "observation", "interpretation")):
        raise ValueError("AI 응답에 유효한 시황 요약이 없습니다.")
    clean_market = {key: str(market[key]).strip() for key in ("headline", "observation", "interpretation")}
    return received, clean_market


def _provider_chain() -> list[tuple[str, Callable[[list[dict], dict], dict]]]:
    chain = []
    if os.getenv("GEMINI_API_KEY"):
        chain.append((f"Gemini ({GEMINI_MODEL})", _request_gemini))
    else:
        logger.warning("GEMINI_API_KEY 미설정: Gemini를 건너뜁니다.")

    if os.getenv("NVIDIA_API_KEY"):
        chain.extend([
            (f"NVIDIA NIM Mistral ({NIM_MISTRAL_MODEL})", lambda items, context: _request_nim(NIM_MISTRAL_MODEL, items, context)),
            (f"NVIDIA NIM GPT-OSS ({NIM_GPT_OSS_MODEL})", lambda items, context: _request_nim(NIM_GPT_OSS_MODEL, items, context)),
        ])
    else:
        logger.warning("NVIDIA_API_KEY 미설정: Mistral·GPT-OSS를 건너뜁니다.")
    return chain


def enrich_with_ai(
    stocks_df: pd.DataFrame,
    data_date: str,
    base_market_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    """사업 설명, 뉴스 근거 기반 등락 이유, 시황을 우선순위대로 생성한다."""
    result = stocks_df.copy()
    cache = _load_cache()
    tickers = result["ticker"].astype(str).tolist()
    cache_keys = {
        ticker: f"{AI_INSIGHTS_CACHE_VERSION}:{data_date}:{ticker}"
        for ticker in tickers
    }
    missing = [ticker for ticker in tickers if cache_keys[ticker] not in cache]
    market_key = f"{AI_INSIGHTS_CACHE_VERSION}:market:{data_date}"
    target_tickers = missing if missing else (tickers if market_key not in cache else [])

    if target_tickers:
        news_map = _collect_news(target_tickers)
        records = result.set_index(result["ticker"].astype(str)).to_dict("index")
        items = [
            {
                "ticker": ticker,
                "name": records[ticker].get("name", ""),
                "sector": records[ticker].get("sector", ""),
                "return_1d": records[ticker].get("return_1d"),
                "business_source_en": records[ticker].get("business_summary", ""),
                "news": news_map.get(ticker, []),
            }
            for ticker in target_tickers
        ]
        allowed_urls = {
            str(news_item.get("url", ""))
            for news in news_map.values()
            for news_item in news
            if news_item.get("url")
        }

        for provider_name, request_fn in _provider_chain():
            try:
                generated = request_fn(items, base_market_summary)
                entries, market = _normalise_response(generated, target_tickers, allowed_urls)
                for ticker, entry in entries.items():
                    cache[cache_keys[ticker]] = entry
                cache[market_key] = market
                AI_INSIGHTS_CACHE_FILE.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info("AI 인사이트 생성 완료: %s", provider_name)
                break
            except Exception as exc:
                logger.warning("%s 인사이트 생성 실패, 다음 우선순위를 시도합니다: %s", provider_name, exc)
        else:
            logger.warning("모든 AI 공급자 호출에 실패했습니다. 근거 기반 대체 문구를 사용합니다.")

    businesses, reasons = [], []
    for _, row in result.iterrows():
        entry = cache.get(cache_keys[str(row["ticker"])], {})
        korean_fallback = (
            f"{row.get('sub_sector') or row.get('sector') or '해당 산업'} 분야의 미국 상장 기업"
        )
        businesses.append(entry.get("business_summary") or korean_fallback)
        reasons.append(entry.get("move_reason") or FALLBACK_REASON)
    result["business_summary"] = businesses
    result["move_reason"] = reasons

    final_market_summary = dict(base_market_summary)
    final_market_summary.update(cache.get(market_key, {}))
    final_market_summary["disclaimer"] = DISCLAIMER
    return result, final_market_summary
