"""
fetcher.py — S&P 500 구성종목 + 주가 + 시가총액 수집
"""

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from io import StringIO

import certifi
import exchange_calendars as xcals
import pandas as pd
import requests
import yfinance as yf

from config import (
    BATCH_SIZE,
    DAYS_1M,
    DAYS_1W,
    DAYS_3M,
    MC_FETCH_WORKERS,
    MC_TIMEOUT_SEC,
    MARKET_CLOSE_GRACE_MINUTES,
    MIN_LATEST_DATE_COVERAGE,
    MIN_MARKET_CAP_COVERAGE,
    MIN_PRICE_COVERAGE,
    MIN_RETURN_HISTORY_COVERAGE,
    PRICE_HISTORY_PERIOD,
    REQUEST_DELAY_SEC,
    SP500_CACHE_DAYS,
    SP500_CACHE_FILE,
    SP500_WIKI_URL,
)

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "^GSPC"
BENCHMARK_HISTORY_PERIOD = "1mo"
LATEST_DATE_SAMPLE_SIZE = 10
MAX_TARGETED_RETRY_RATIO = 0.05


class UpstreamNotReady(RuntimeError):
    """Yahoo의 정상 응답에 예상 거래일 가격이 아직 준비되지 않은 상태."""

    def __init__(
        self,
        reason: str,
        expected_date: date,
        *,
        raw_latest_date: date | None = None,
        observed_date: date | None = None,
        coverage: float | None = None,
        aligned: int | None = None,
        total: int | None = None,
        stale_count: int = 0,
        missing_count: int = 0,
        future_count: int = 0,
    ):
        self.reason = reason
        self.expected_date = expected_date
        self.raw_latest_date = raw_latest_date
        self.observed_date = observed_date
        self.coverage = coverage
        self.aligned = aligned
        self.total = total
        self.stale_count = stale_count
        self.missing_count = missing_count
        self.future_count = future_count
        super().__init__(
            "Yahoo 데이터가 예상 거래일에 아직 준비되지 않았습니다: "
            f"reason={reason}, expected={expected_date}, "
            f"observed={observed_date}, coverage={coverage}"
        )

    def status_details(self) -> dict:
        return {
            "reason": self.reason,
            "expected_date": self.expected_date.isoformat(),
            "raw_latest_date": (
                self.raw_latest_date.isoformat()
                if self.raw_latest_date is not None
                else None
            ),
            "observed_date": (
                self.observed_date.isoformat()
                if self.observed_date is not None
                else None
            ),
            "coverage": self.coverage,
            "aligned": self.aligned,
            "total": self.total,
            "stale_count": self.stale_count,
            "missing_count": self.missing_count,
            "future_count": self.future_count,
        }


def get_expected_market_date(as_of: datetime | None = None) -> date:
    """XNYS 일정에서 데이터 준비 유예시간이 지난 마지막 완료 세션을 구한다."""
    observed_at = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if observed_at.tz is None:
        raise ValueError("as_of는 timezone-aware datetime이어야 합니다.")
    observed_at = observed_at.tz_convert("UTC")
    ready_cutoff = observed_at - pd.Timedelta(
        minutes=MARKET_CLOSE_GRACE_MINUTES
    )
    calendar = xcals.get_calendar("XNYS")
    eligible_closes = calendar.closes[calendar.closes <= ready_cutoff]
    if eligible_closes.empty:
        raise RuntimeError("완료된 XNYS 거래일을 확인할 수 없습니다.")

    expected_date = eligible_closes.index[-1].date()
    logger.info(
        "예상 완료 거래일: %s | XNYS close=%s | as_of=%s | grace=%d분",
        expected_date,
        eligible_closes.iloc[-1],
        observed_at,
        MARKET_CLOSE_GRACE_MINUTES,
    )
    return expected_date


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

def _normalise_price_series(series: pd.Series) -> tuple[pd.Series, int, int]:
    """가격 Series의 인덱스를 정렬하고 NaT·중복을 결정론적으로 제거한다."""
    converted_index = pd.to_datetime(series.index, errors="coerce")
    normalised = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(converted_index),
        name=series.name,
    )
    nat_count = int(normalised.index.isna().sum())
    normalised = normalised[~normalised.index.isna()].dropna()
    normalised = normalised.sort_index(kind="mergesort")
    duplicate_count = int(normalised.index.duplicated(keep="last").sum())
    normalised = normalised[~normalised.index.duplicated(keep="last")]
    return normalised, nat_count, duplicate_count


def _latest_price_date(series: pd.Series) -> date | None:
    """선택 가격의 결측치를 제외한 실제 최대 거래일을 반환한다."""
    values = series.dropna()
    if values.empty:
        return None
    index = pd.DatetimeIndex(pd.to_datetime(values.index, errors="coerce"))
    index = index[~index.isna()]
    if index.empty:
        return None
    return index.max().date()


def _price_field_frame(raw: pd.DataFrame, field: str, batch: list[str]) -> pd.DataFrame | None:
    """yfinance 단일·다중 티커 컬럼 구조에서 가격 필드 DataFrame을 꺼낸다."""
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if field not in raw.columns.get_level_values(level):
                continue
            frame = raw.xs(field, axis=1, level=level, drop_level=True)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = [
                    str(column[-1] if isinstance(column, tuple) else column)
                    for column in frame.columns
                ]
            else:
                frame.columns = [str(column) for column in frame.columns]
            return frame
        return None

    if field not in raw.columns:
        return None
    frame = raw[[field]].copy()
    if len(batch) == 1:
        frame.columns = batch
    return frame


def _extract_batch_prices(
    raw: pd.DataFrame,
    batch: list[str],
    diagnostic_label: str,
    *,
    price_field: str = "Adj Close",
) -> dict[str, pd.Series]:
    """지정 가격 필드만 선택하고 배치 가격 형식을 진단한다."""
    adjusted = _price_field_frame(raw, "Adj Close", batch)
    close = _price_field_frame(raw, "Close", batch)
    selected = {"Adj Close": adjusted, "Close": close}.get(price_field)
    available = [
        field
        for field, frame in (("Adj Close", adjusted), ("Close", close))
        if frame is not None
    ]

    if selected is None:
        logger.warning(
            "%s 필수 가격 필드 없음: required=%s | available=%s",
            diagnostic_label,
            price_field,
            ",".join(available) or "none",
        )
        return {}

    raw_index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    nat_count = int(raw_index.isna().sum())
    valid_raw_index = raw_index[~raw_index.isna()]
    raw_latest = valid_raw_index.max() if not valid_raw_index.empty else None
    duplicate_count = int(valid_raw_index.duplicated(keep="last").sum())
    latest_gap_tickers: list[str] = []
    prices: dict[str, pd.Series] = {}

    for ticker in batch:
        if ticker not in selected.columns:
            continue
        raw_series = selected[ticker]
        if isinstance(raw_series, pd.DataFrame):
            raw_series = raw_series.iloc[:, -1]
        if raw_latest is not None:
            latest_values = raw_series[
                pd.DatetimeIndex(pd.to_datetime(raw_series.index, errors="coerce"))
                == raw_latest
            ]
            if latest_values.dropna().empty:
                latest_gap_tickers.append(ticker)

        series, _, _ = _normalise_price_series(raw_series)
        if len(series) >= 2:
            prices[ticker] = series

    logger.info(
        "%s 가격 진단: available=%s | selected=%s | rows=%d | NaT=%d | "
        "duplicate_index=%d | latest_gap=%d%s",
        diagnostic_label,
        ",".join(available) or "none",
        price_field,
        len(raw),
        nat_count,
        duplicate_count,
        len(latest_gap_tickers),
        (
            f" [{', '.join(latest_gap_tickers[:LATEST_DATE_SAMPLE_SIZE])}]"
            if latest_gap_tickers
            else ""
        ),
    )
    if price_field == "Adj Close" and latest_gap_tickers:
        logger.warning(
            "%s Adj Close 최신값 결측 %d건: 비조정 Close로 대체하지 않고 기준일 검증 대상으로 기록합니다.",
            diagnostic_label,
            len(latest_gap_tickers),
        )
    return prices


def fetch_price_data(tickers: list[str]) -> dict[str, pd.Series]:
    """
    yfinance.download으로 6개월치 일봉 조정 종가를 배치 수집한다.

    Returns
    -------
    dict: {ticker: pd.Series(adjusted Close, index=DatetimeIndex)}
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
                period=PRICE_HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if raw.empty:
                logger.warning("배치 %d: 빈 데이터 반환", batch_no)
                continue

            all_prices.update(
                _extract_batch_prices(raw, batch, f"주가 배치 {batch_no}/{n_batches}")
            )

        except Exception as exc:
            logger.warning("배치 %d 실패: %s", batch_no, exc)

        time.sleep(REQUEST_DELAY_SEC)

    logger.info("주가 수집 완료: %d/%d 종목", len(all_prices), len(tickers))
    return all_prices


def _latest_raw_index_date(raw: pd.DataFrame) -> date | None:
    index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    index = index[~index.isna()]
    return index.max().date() if not index.empty else None


def _series_has_date(series: pd.Series, expected_date: date) -> bool:
    index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    return bool((index.date == expected_date).any())


def fetch_benchmark_latest_date(expected_date: date) -> date:
    """예상 거래일의 Yahoo S&P 500 지수 Close 존재 여부를 검증한다."""
    raw = yf.download(
        BENCHMARK_TICKER,
        period=BENCHMARK_HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise UpstreamNotReady(
            "benchmark_empty",
            expected_date,
            coverage=0.0,
            aligned=0,
            total=1,
            missing_count=1,
        )
    close_frame = _price_field_frame(raw, "Close", [BENCHMARK_TICKER])
    if close_frame is None or BENCHMARK_TICKER not in close_frame.columns:
        raise RuntimeError("Yahoo 기준지수(^GSPC) 응답에 Close 필드가 없습니다.")

    raw_close = close_frame[BENCHMARK_TICKER]
    if isinstance(raw_close, pd.DataFrame):
        raw_close = raw_close.iloc[:, -1]
    series, _, _ = _normalise_price_series(raw_close)
    observed_date = _latest_price_date(series)
    raw_latest_date = _latest_raw_index_date(raw)
    if not _series_has_date(series, expected_date):
        raise UpstreamNotReady(
            "benchmark_close_missing",
            expected_date,
            raw_latest_date=raw_latest_date,
            observed_date=observed_date,
            coverage=0.0,
            aligned=0,
            total=1,
            stale_count=int(
                observed_date is not None and observed_date < expected_date
            ),
            missing_count=int(observed_date is None),
            future_count=int(
                observed_date is not None and observed_date > expected_date
            ),
        )

    logger.info(
        "Yahoo 기준지수 검증 통과: expected=%s | raw_latest=%s | observed=%s",
        expected_date,
        raw_latest_date,
        observed_date,
    )
    return expected_date


def _validate_return_history_coverage(
    price_data: dict[str, pd.Series], total_tickers: int
) -> None:
    """각 수익률 기간을 계산할 수 있는 종목 비율을 검증한다."""
    requirements = (
        ("1일", 2),
        ("1주", DAYS_1W + 1),
        ("1개월", DAYS_1M + 1),
        ("3개월", DAYS_3M + 1),
    )
    denominator = max(total_tickers, 1)
    diagnostics = []
    failures = []

    for label, minimum_rows in requirements:
        valid_count = sum(
            len(series) >= minimum_rows
            and pd.notna(series.iloc[-1])
            and pd.notna(series.iloc[-minimum_rows])
            and series.iloc[-minimum_rows] != 0
            for series in price_data.values()
        )
        coverage = valid_count / denominator
        diagnostics.append(f"{label} {valid_count}/{total_tickers} ({coverage:.1%})")
        if coverage < MIN_RETURN_HISTORY_COVERAGE:
            failures.append(f"{label} {valid_count}/{total_tickers} ({coverage:.1%})")

    logger.info("수익률 이력 커버리지: %s", " | ".join(diagnostics))
    if failures:
        raise RuntimeError(
            "수익률 이력 커버리지가 기준 미달입니다: "
            f"{' | '.join(failures)}, 최소 {MIN_RETURN_HISTORY_COVERAGE:.0%} 필요"
        )


def _classify_price_dates(
    price_data: dict[str, pd.Series],
    tickers: list[str],
    expected_date: date,
) -> tuple[list[str], list[str], list[str]]:
    """예상 거래일 대비 누락·stale·future-only 티커를 분류한다."""
    missing: list[str] = []
    stale: list[str] = []
    future: list[str] = []
    for ticker in tickers:
        series = price_data.get(ticker)
        if series is None:
            missing.append(ticker)
            continue

        original_latest = _latest_price_date(series)
        trimmed = _trim_price_series_to_date(series, expected_date)
        latest_date = _latest_price_date(trimmed)
        if latest_date is None:
            if original_latest is not None and original_latest > expected_date:
                future.append(ticker)
            else:
                missing.append(ticker)
        elif latest_date < expected_date:
            stale.append(ticker)
        elif latest_date > expected_date:
            # _trim_price_series_to_date 계약상 도달할 수 없는 방어 분기다.
            future.append(ticker)
    return missing, stale, future


def _trim_price_series_to_date(
    series: pd.Series,
    expected_date: date,
) -> pd.Series:
    """장중 partial row를 배제하도록 예상 거래일 이후 가격을 자른다."""
    index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    boundary = pd.Timestamp(expected_date)
    if index.tz is not None:
        boundary = boundary.tz_localize(index.tz)
    mask = ~index.isna() & (index.normalize() <= boundary)
    return series[mask].copy()


def _retry_inconsistent_prices(
    price_data: dict[str, pd.Series],
    tickers: list[str],
    expected_date: date,
) -> dict[str, pd.Series]:
    """예상일 커버리지가 이미 기준 이상일 때 소수 불일치만 재조회한다."""
    missing, stale, future = _classify_price_dates(
        price_data, tickers, expected_date
    )
    retry_set = set([*missing, *stale, *future])
    retry_tickers = [ticker for ticker in tickers if ticker in retry_set]
    if not retry_tickers:
        return price_data

    retry_limit = int(len(tickers) * MAX_TARGETED_RETRY_RATIO)
    if len(retry_tickers) > retry_limit:
        logger.warning(
            "예상일 불일치 %d/%d (%.1f%%)로 대규모 표적 재시도를 생략합니다.",
            len(retry_tickers),
            len(tickers),
            len(retry_tickers) / max(len(tickers), 1) * 100,
        )
        return price_data

    logger.warning(
        "주가 표적 재시도 1회: 누락=%d | stale=%d | future-only=%d | sample=%s",
        len(missing),
        len(stale),
        len(future),
        ", ".join(retry_tickers[:LATEST_DATE_SAMPLE_SIZE]),
    )
    retried = fetch_price_data(retry_tickers)
    merged = dict(price_data)
    for ticker, series in retried.items():
        merged[ticker] = series
    return merged


def _validate_latest_date_coverage(
    price_data: dict[str, pd.Series],
    tickers: list[str],
    expected_date: date,
) -> tuple[dict[str, pd.Series], float]:
    """예상 거래일 커버리지를 검증하고 그 날짜까지 자른 가격만 반환한다."""
    trimmed_prices = {
        ticker: _trim_price_series_to_date(series, expected_date)
        for ticker, series in price_data.items()
    }
    date_by_ticker = {
        ticker: (
            _latest_price_date(trimmed_prices[ticker])
            if ticker in trimmed_prices
            else None
        )
        for ticker in tickers
    }
    observed_by_ticker = {
        ticker: (
            _latest_price_date(price_data[ticker])
            if ticker in price_data
            else None
        )
        for ticker in tickers
    }
    histogram = Counter(
        str(latest_date) if latest_date is not None else "missing/invalid"
        for latest_date in observed_by_ticker.values()
    )
    histogram_text = " | ".join(
        f"{key}={count}" for key, count in sorted(histogram.items())
    )
    nat_count = sum(
        int(pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce")).isna().sum())
        for series in price_data.values()
    )
    stale = [
        (ticker, latest_date)
        for ticker, latest_date in date_by_ticker.items()
        if latest_date is not None and latest_date < expected_date
    ]
    future = [
        (ticker, observed_by_ticker[ticker])
        for ticker, latest_date in date_by_ticker.items()
        if observed_by_ticker[ticker] is not None
        and observed_by_ticker[ticker] > expected_date
        and latest_date is None
    ]
    missing = [
        ticker
        for ticker, latest_date in date_by_ticker.items()
        if latest_date is None
        and not (
            observed_by_ticker[ticker] is not None
            and observed_by_ticker[ticker] > expected_date
        )
    ]
    latest_tickers = [
        ticker for ticker, latest_date in date_by_ticker.items()
        if latest_date == expected_date
    ]
    coverage = len(latest_tickers) / max(len(tickers), 1)

    logger.info(
        "가격 거래일 분포: expected=%s | observed=%s | NaT=%d | "
        "stale=%d [%s] | future-only=%d [%s] | missing=%d [%s]",
        expected_date,
        histogram_text,
        nat_count,
        len(stale),
        ", ".join(f"{ticker}:{latest}" for ticker, latest in stale[:LATEST_DATE_SAMPLE_SIZE]),
        len(future),
        ", ".join(f"{ticker}:{latest}" for ticker, latest in future[:LATEST_DATE_SAMPLE_SIZE]),
        len(missing),
        ", ".join(missing[:LATEST_DATE_SAMPLE_SIZE]),
    )
    if coverage < MIN_LATEST_DATE_COVERAGE:
        mismatch_count = len(tickers) - len(latest_tickers)
        observed_counts = Counter(
            latest_date
            for latest_date in date_by_ticker.values()
            if latest_date is not None
        )
        observed_date = (
            observed_counts.most_common(1)[0][0]
            if observed_counts
            else None
        )
        raw_dates = [
            latest_date
            for latest_date in observed_by_ticker.values()
            if latest_date is not None
        ]
        raw_latest_date = max(raw_dates) if raw_dates else None
        stale_dominant = (
            len(stale) > 0
            and len(stale) * 5 >= mismatch_count * 4
        )
        if stale_dominant:
            raise UpstreamNotReady(
                "constituent_adj_close_lag",
                expected_date,
                raw_latest_date=raw_latest_date,
                observed_date=observed_date,
                coverage=coverage,
                aligned=len(latest_tickers),
                total=len(tickers),
                stale_count=len(stale),
                missing_count=len(missing),
                future_count=len(future),
            )
        raise RuntimeError(
            "예상 거래일 정합성 기준 미달입니다: "
            f"XNYS {expected_date} 기준 "
            f"{len(latest_tickers)}/{len(tickers)} ({coverage:.1%}), "
            f"최소 {MIN_LATEST_DATE_COVERAGE:.0%} 필요"
        )

    canonical_prices = {
        ticker: trimmed_prices[ticker]
        for ticker in latest_tickers
    }
    if len(canonical_prices) != len(tickers):
        logger.warning(
            "예상일 불일치 종목 %d개를 downstream 계산에서 제외합니다.",
            len(tickers) - len(canonical_prices),
        )
    return canonical_prices, coverage


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

def fetch_all_data(
    expected_date: date | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    S&P 500 구성종목 + 주가 + 시가총액을 모두 수집한다.

    Returns
    -------
    (components_df, price_data_dict, market_cap_dict)
    """
    expected_date = expected_date or get_expected_market_date()
    fetch_benchmark_latest_date(expected_date)

    components = get_sp500_components()
    tickers = components["ticker"].tolist()

    price_data = fetch_price_data(tickers)
    collected_price_count = len(price_data)
    price_coverage = collected_price_count / max(len(tickers), 1)
    if price_coverage < MIN_PRICE_COVERAGE:
        raise RuntimeError(
            "주가 수집 커버리지가 기준 미달입니다: "
            f"{collected_price_count}/{len(tickers)} ({price_coverage:.1%}), "
            f"최소 {MIN_PRICE_COVERAGE:.0%} 필요"
        )

    price_data = _retry_inconsistent_prices(
        price_data, tickers, expected_date
    )
    price_data, latest_coverage = _validate_latest_date_coverage(
        price_data, tickers, expected_date
    )
    _validate_return_history_coverage(price_data, len(tickers))

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
        expected_date,
        latest_coverage * 100,
        market_cap_coverage * 100,
    )

    return components, price_data, market_caps
