"""NVIDIA NIM의 Kimi 모델로 한글 기업·시황 인사이트를 생성한다."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf

from config import NIM_API_URL, NIM_CACHE_FILE, NIM_MAX_TOKENS, NIM_MODEL, PROFILE_FETCH_WORKERS

logger = logging.getLogger(__name__)

FALLBACK_REASON = "당일 뉴스·공시 근거를 확인하지 못했습니다."
DISCLAIMER = "AI 요약은 제공된 Yahoo Finance 기업 설명·뉴스 헤드라인과 시장 데이터만 근거로 하며 투자 의견이 아닙니다."


def _load_cache() -> dict:
    try:
        return json.loads(NIM_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _unwrap(value) -> str:
    if isinstance(value, dict):
        return str(value.get("url", ""))
    return str(value or "")


def _get_ticker_news(ticker: str) -> tuple[str, list[dict]]:
    """Yahoo Finance 뉴스에서 모델에 제공할 최소한의 검증 가능한 단서를 뽑는다."""
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
        link = _unwrap(content.get("canonicalUrl") or content.get("clickThroughUrl") or item.get("link"))
        news.append({"title": title[:300], "publisher": str(provider or "")[:80], "url": link})
    return ticker, news


def _collect_news(tickers: list[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as executor:
        futures = {executor.submit(_get_ticker_news, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker, news = future.result()
            result[ticker] = news
    return result


def _parse_json(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("NIM 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(cleaned[start : end + 1])


def _nim_request(items: list[dict], market_context: dict) -> dict:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY가 설정되지 않았습니다.")

    system = """당신은 미국 주식 리서치 보조자입니다. 반드시 제공된 데이터만 사용해 한국어로 답하십시오.
특히 등락 원인은 제공된 뉴스 헤드라인이 직접 뒷받침할 때만 서술하십시오. 뒷받침할 뉴스가 없거나 제목만으로 인과관계를 판단할 수 없으면 정확히 '당일 뉴스·공시 근거를 확인하지 못했습니다.'라고 쓰십시오. 가격 변동을 뉴스의 원인으로 단정하지 마십시오.
출력은 설명·마크다운 없이 JSON 객체 하나여야 합니다.
스키마: {"items":[{"ticker":"문자열","business_ko":"한국어 한 문장, 70자 이내","move_reason_ko":"한국어 1~2문장, 140자 이내","source_urls":["제공된 URL만 사용"]}],"market_summary":{"headline":"한 문장","observation":"수치 기반 관측 1~2문장","interpretation":"해석 1문장"}}"""
    user = json.dumps({"market_context": market_context, "stocks": items}, ensure_ascii=False)
    payload = {
        "model": NIM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": NIM_MAX_TOKENS,
        "stream": False,
    }
    response = requests.post(
        NIM_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json=payload,
        timeout=(15, 180),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _parse_json(content)


def enrich_with_nim(
    stocks_df: pd.DataFrame,
    data_date: str,
    base_market_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    """사업 문구를 한글화하고, 뉴스 근거 기반 등락 이유와 시황 요약을 생성한다.

    API 키·응답·뉴스가 없으면 보고서 생성을 멈추지 않고 명시적 대체 문구를 사용한다.
    """
    result = stocks_df.copy()
    cache = _load_cache()
    tickers = result["ticker"].astype(str).tolist()
    cache_keys = {ticker: f"{data_date}:{ticker}" for ticker in tickers}
    missing = [ticker for ticker in tickers if cache_keys[ticker] not in cache]
    market_key = f"market:{data_date}"

    if missing and os.getenv("NVIDIA_API_KEY"):
        news_map = _collect_news(missing)
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
            for ticker in missing
        ]
        try:
            generated = _nim_request(items, base_market_summary)
            allowed = set(missing)
            for item in generated.get("items", []):
                ticker = str(item.get("ticker", ""))
                if ticker not in allowed:
                    continue
                cache[cache_keys[ticker]] = {
                    "business_summary": str(item.get("business_ko", "")).strip(),
                    "move_reason": str(item.get("move_reason_ko", "")).strip(),
                    "source_urls": [u for u in item.get("source_urls", []) if isinstance(u, str)],
                }
            market = generated.get("market_summary", {})
            if all(str(market.get(k, "")).strip() for k in ("headline", "observation", "interpretation")):
                cache[market_key] = {k: str(market[k]).strip() for k in ("headline", "observation", "interpretation")}
            NIM_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("NIM 인사이트 생성 실패. 대체 문구를 사용합니다: %s", exc)
    elif missing:
        logger.warning("NVIDIA_API_KEY 미설정. NIM 인사이트 없이 보고서를 생성합니다.")

    businesses, reasons = [], []
    for _, row in result.iterrows():
        entry = cache.get(cache_keys[str(row["ticker"])], {})
        businesses.append(entry.get("business_summary") or row.get("business_summary") or "사업 설명을 확인하지 못했습니다.")
        reasons.append(entry.get("move_reason") or FALLBACK_REASON)
    result["business_summary"] = businesses
    result["move_reason"] = reasons

    final_market_summary = dict(base_market_summary)
    final_market_summary.update(cache.get(market_key, {}))
    final_market_summary["disclaimer"] = DISCLAIMER
    return result, final_market_summary
