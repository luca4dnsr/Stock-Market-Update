"""
calculator.py — 수익률 계산, 섹터 집계, 시장 폭 계산
"""

import logging

import numpy as np
import pandas as pd

from config import DAYS_1M, DAYS_1W, DAYS_3M

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 단일 수익률 계산
# ──────────────────────────────────────────────────────────

def _calc_return(series: pd.Series, n_days: int) -> float | None:
    """
    n 거래일 전 대비 현재 수익률(%)을 계산한다.
    데이터 부족 시 None 반환.
    """
    if len(series) < n_days + 1:
        return None
    current = series.iloc[-1]
    past    = series.iloc[-(n_days + 1)]
    if past == 0 or pd.isna(past):
        return None
    return round((current - past) / past * 100, 1)


# ──────────────────────────────────────────────────────────
# 전 종목 수익률 DataFrame 생성
# ──────────────────────────────────────────────────────────

def calculate_returns(price_data: dict[str, pd.Series]) -> pd.DataFrame:
    """
    price_data 딕셔너리를 받아 각 티커의 수익률 DataFrame을 반환한다.

    Returns
    -------
    DataFrame: columns=['ticker', 'return_1d', 'return_1w', 'return_1m', 'return_3m', 'latest_date']
    """
    records = []
    for ticker, series in price_data.items():
        records.append({
            "ticker":      ticker,
            "return_1d":   _calc_return(series, 1),
            "return_1w":   _calc_return(series, DAYS_1W),
            "return_1m":   _calc_return(series, DAYS_1M),
            "return_3m":   _calc_return(series, DAYS_3M),
            "latest_date": series.index[-1].date() if len(series) > 0 else None,
        })
    df = pd.DataFrame(records)
    valid = df["return_1d"].notna().sum()
    logger.info("수익률 계산 완료: %d/%d 종목 유효", valid, len(df))
    return df


# ──────────────────────────────────────────────────────────
# 마스터 DataFrame 구성
# ──────────────────────────────────────────────────────────

def build_master_df(
    components: pd.DataFrame,
    returns_df: pd.DataFrame,
    market_caps: dict[str, float],
) -> pd.DataFrame:
    """
    구성종목, 수익률, 시가총액을 병합하고
    시총 순위(mc_rank)를 부여한다.
    """
    df = components.merge(returns_df, on="ticker", how="inner")
    df["market_cap"]   = df["ticker"].map(market_caps)
    df["market_cap_b"] = (df["market_cap"] / 1e9).round(1)   # 십억 달러

    # 시총 순위 (내림차순)
    df = df.sort_values("market_cap", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df["mc_rank"] = df.index + 1

    logger.info("마스터 DataFrame 구성 완료: %d 종목", len(df))
    return df


# ──────────────────────────────────────────────────────────
# 섹터 집계
# ──────────────────────────────────────────────────────────

def calculate_sector_returns(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    시총 가중 평균으로 섹터별 1일 수익률을 계산한다.

    Returns
    -------
    DataFrame: columns=['sector', 'return_1d']  (내림차순 정렬)
    """
    valid = master_df.dropna(subset=["return_1d", "market_cap"]).copy()

    def _weighted_mean(group: pd.DataFrame) -> float:
        w = group["market_cap"]
        total = w.sum()
        if total == 0:
            return np.nan
        return round((group["return_1d"] * w).sum() / total, 1)

    sector_df = (
        valid.groupby("sector")
        .apply(_weighted_mean)
        .reset_index()
    )
    sector_df.columns = ["sector", "return_1d"]
    sector_df = sector_df.sort_values("return_1d", ascending=False).reset_index(drop=True)
    return sector_df


# ──────────────────────────────────────────────────────────
# 시장 폭 (상승/하락 종목 수)
# ──────────────────────────────────────────────────────────

def count_advances_declines(master_df: pd.DataFrame) -> tuple[int, int]:
    """1일 수익률 기준 상승/하락 종목 수를 반환한다."""
    valid = master_df["return_1d"].dropna()
    advances = int((valid > 0).sum())
    declines = int((valid < 0).sum())
    logger.info("시장 폭 — 상승: %d, 하락: %d", advances, declines)
    return advances, declines


# ──────────────────────────────────────────────────────────
# 데이터 기준일 감지
# ──────────────────────────────────────────────────────────

def get_data_date(master_df: pd.DataFrame) -> str:
    """
    수집된 주가의 최신 거래일을 YYYY-MM-DD 형식으로 반환한다.
    """
    if "latest_date" in master_df.columns:
        latest = master_df["latest_date"].dropna().max()
        if latest:
            return str(latest)
    return ""
