"""
fetcher.py — S&P 500 구성종목 + 주가 + 시가총액 수집
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO

import certifi
import pandas as pd
import requests
import yfinance as yf

from config import (
    BATCH_SIZE,
    MC_FETCH_WORKERS,
    MC_TIMEOUT_SEC,
    MIN_LATEST_DATE_COVERAGE,
    MIN_MARKET_CAP_COVERAGE,
    MIN_PRICE_COVERAGE,
    REQUEST_DELAY_SEC,
    SP500_CACHE_DAYS,
    SP500_CACHE_FILE,
    SP500_WIKI_URL,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# S&P 500 구성종목
# ──────────────────────────────────────────────────────────

def get_sp500_components() -> pd.DataFrame:
    """
    Wikipedia에서 S&P 500 구성종목 목록을 가져온다.
    로컬 캐시가 유효하면 캐시를 사용한다.

    Returns
    -------
    DataFrame: columns=['ticker', 'name', 'sector', 'sub_sector']
    """
    # ── 캐시 확인 ──
    if SP500_CACHE_FILE.exists():
        age_days = (
            datetime.now()
            - datetime.fromtimestamp(SP500_CACHE_FILE.stat().st_mtime)
        ).days
        if age_days < SP500_CACHE_DAYS:
            logger.info("S&P 500 구성종목: 캐시 로드 (%d일 전 갱신)", age_days)
            return pd.read_csv(SP500_CACHE_FILE)

    # ── Wikipedia 파싱 (requests + certifi로 macOS SSL 문제 우회) ──
    logger.info("S&P 500 구성종목: Wikipedia 조회 중...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (SPX-Daily-Bot/1.0)"}
        resp = requests.get(
            SP500_WIKI_URL,
            headers=headers,
            verify=certifi.where(),   # macOS Python SSL 인증서 문제 해결
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
        df.columns = ["ticker", "name", "sector", "sub_sector"]
        # BRK.B / BF.B → BRK-B / BF-B (yfinance 형식)
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df.to_csv(SP500_CACHE_FILE, index=False)
        logger.info("S&P 500 구성종목 %d개 캐시 저장", len(df))
        return df
    except Exception as exc:
        logger.error("구성종목 조회 실패: %s", exc)
        raise


# ──────────────────────────────────────────────────────────
# 주가 데이터 (배치 다운로드)
# ──────────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str]) -> dict[str, pd.Series]:
    """
    yfinance.download으로 3개월치 일봉 종가를 배치 수집한다.

    Returns
    -------
    dict: {ticker: pd.Series(Close, index=DatetimeIndex)}
    """
    all_prices: dict[str, pd.Series] = {}
    n_batches = (len(tickers) - 1) // BATCH_SIZE + 1

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        logger.info("주가 배치 %d/%d (%d종목) 다운로드 중...", batch_no, n_batches, len(batch))

        try:
            raw = yf.download(
                batch,
                period="3mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                logger.warning("배치 %d: 빈 데이터 반환", batch_no)
                continue

            # 단일 티커 vs 다중 티커 컬럼 구조 통일
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]]
                close.columns = batch

            for ticker in batch:
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if len(series) >= 2:
                        all_prices[ticker] = series

        except Exception as exc:
            logger.warning("배치 %d 실패: %s", batch_no, exc)

        time.sleep(REQUEST_DELAY_SEC)

    logger.info("주가 수집 완료: %d/%d 종목", len(all_prices), len(tickers))
    return all_prices


# ──────────────────────────────────────────────────────────
# 시가총액 (병렬 조회)
# ──────────────────────────────────────────────────────────

def _get_single_market_cap(ticker: str) -> tuple[str, float | None]:
    """단일 티커 시가총액 조회 (ThreadPoolExecutor 워커)."""
    try:
        mc = yf.Ticker(ticker).fast_info.market_cap
        return ticker, float(mc) if mc else None
    except Exception:
        return ticker, None


def fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    """
    ThreadPoolExecutor로 시가총액을 병렬 조회한다.

    Returns
    -------
    dict: {ticker: market_cap_in_usd}
    """
    logger.info("시가총액 조회 중 (%d종목, %d스레드)...", len(tickers), MC_FETCH_WORKERS)
    market_caps: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=MC_FETCH_WORKERS) as executor:
        futures = {executor.submit(_get_single_market_cap, t): t for t in tickers}
        done = 0
        for future in as_completed(futures, timeout=MC_TIMEOUT_SEC):
            ticker, mc = future.result()
            if mc:
                market_caps[ticker] = mc
            done += 1
            if done % 50 == 0:
                logger.info("  시가총액 진행: %d/%d", done, len(tickers))

    logger.info("시가총액 조회 완료: %d/%d 종목", len(market_caps), len(tickers))
    return market_caps


# ──────────────────────────────────────────────────────────
# 통합 수집 함수
# ──────────────────────────────────────────────────────────

def fetch_all_data() -> tuple[pd.DataFrame, dict, dict]:
    """
    S&P 500 구성종목 + 주가 + 시가총액을 모두 수집한다.

    Returns
    -------
    (components_df, price_data_dict, market_cap_dict)
    """
    components = get_sp500_components()
    tickers = components["ticker"].tolist()

    price_data = fetch_price_data(tickers)
    price_coverage = len(price_data) / max(len(tickers), 1)
    if price_coverage < MIN_PRICE_COVERAGE:
        raise RuntimeError(
            "주가 수집 커버리지가 기준 미달입니다: "
            f"{len(price_data)}/{len(tickers)} ({price_coverage:.1%}), "
            f"최소 {MIN_PRICE_COVERAGE:.0%} 필요"
        )

    # 모든 종목의 최신 거래일이 거의 같은지 확인한다. 일부 종목의 stale price가
    # 당일 등락률·시장 폭 계산에 섞이는 것을 방지한다.
    latest_dates = [series.index[-1].date() for series in price_data.values() if not series.empty]
    latest_date = max(latest_dates)
    latest_count = sum(date == latest_date for date in latest_dates)
    latest_coverage = latest_count / max(len(price_data), 1)
    if latest_coverage < MIN_LATEST_DATE_COVERAGE:
        raise RuntimeError(
            "최신 거래일 정합성 기준 미달입니다: "
            f"{latest_date} 기준 {latest_count}/{len(price_data)} ({latest_coverage:.1%}), "
            f"최소 {MIN_LATEST_DATE_COVERAGE:.0%} 필요"
        )

    valid_tickers = list(price_data.keys())
    market_caps = fetch_market_caps(valid_tickers)

    market_cap_coverage = len(market_caps) / max(len(valid_tickers), 1)
    if market_cap_coverage < MIN_MARKET_CAP_COVERAGE:
        raise RuntimeError(
            "시가총액 수집 커버리지가 기준 미달입니다: "
            f"{len(market_caps)}/{len(valid_tickers)} ({market_cap_coverage:.1%}), "
            f"최소 {MIN_MARKET_CAP_COVERAGE:.0%} 필요"
        )

    logger.info(
        "데이터 품질 검증 통과 — 주가 %.1f%% | 최신일 %s %.1f%% | 시가총액 %.1f%%",
        price_coverage * 100,
        latest_date,
        latest_coverage * 100,
        market_cap_coverage * 100,
    )

    return components, price_data, market_caps
