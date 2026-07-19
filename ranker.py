"""
ranker.py — 1일 수익률 기준 상위/하위 정렬
"""

import pandas as pd

from config import BOTTOM_N, TOP_N


def get_top_bottom(
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1일 수익률 기준으로 상위 TOP_N, 하위 BOTTOM_N 종목을 반환한다.
    수익률이 null인 종목은 제외.

    Returns
    -------
    (top_df, bottom_df) — 각각 해당 방향으로 정렬된 DataFrame
    """
    valid = master_df.dropna(subset=["return_1d"]).copy()
    sorted_df = valid.sort_values("return_1d", ascending=False).reset_index(drop=True)

    # 1일 수익률 기준 순위 (1=최고 수익)
    sorted_df["day_rank"] = sorted_df.index + 1

    top_df    = sorted_df.head(TOP_N).copy()
    bottom_df = sorted_df.tail(BOTTOM_N).sort_values("return_1d").copy()

    return top_df, bottom_df
