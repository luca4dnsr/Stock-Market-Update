"""수치 기반의 일간 시황 관측·해석 문구 생성기."""

import pandas as pd


def _signed(value) -> str:
    return f"{float(value):+.1f}%"


def build_market_summary(
    sector_df: pd.DataFrame,
    advances: int,
    declines: int,
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
) -> dict[str, str]:
    """시장 폭, 섹터, 대표 종목을 근거로 한글 시황 요약을 만든다.

    원인(뉴스·실적·정책)은 이 파이프라인이 직접 수집하지 않으므로 단정하지 않는다.
    """
    total = max(advances + declines, 1)
    advance_pct = round(advances / total * 100)
    leaders = sector_df.head(2).to_dict("records")
    laggards = sector_df.tail(2).sort_values("return_1d").to_dict("records")
    top = top_df.iloc[0]
    bottom = bottom_df.iloc[0]

    if advance_pct >= 65:
        breadth = "상승 종목이 시장 전반으로 확산된 강한 위험선호 흐름"
        interpretation = "시장 폭과 섹터 흐름이 함께 개선돼, 일부 종목만의 반등보다 폭넓은 매수 우위로 해석됩니다."
    elif advance_pct >= 55:
        breadth = "상승 종목이 우세한 완만한 위험선호 흐름"
        interpretation = "상승 우위이지만 하락 종목도 적지 않아, 다음 거래일에도 섹터별 흐름의 지속 여부를 확인할 필요가 있습니다."
    elif advance_pct > 45:
        breadth = "상승·하락 종목이 엇갈린 혼조 흐름"
        interpretation = "지수 내부의 방향성이 분산된 장세로, 전체 지수보다 주도 섹터와 개별 종목의 차별화가 두드러졌습니다."
    elif advance_pct > 35:
        breadth = "하락 종목이 다소 우세한 약한 위험회피 흐름"
        interpretation = "매도 압력이 우세했지만 시장 전반의 일방적 하락으로 보기는 어려워, 방어 섹터의 상대 강도를 함께 볼 필요가 있습니다."
    else:
        breadth = "하락 종목이 시장 전반으로 확산된 강한 위험회피 흐름"
        interpretation = "시장 폭이 약해 단기적으로 방어적 포지셔닝과 변동성 확대 가능성에 유의할 구간입니다."

    leader_text = ", ".join(f"{r['sector']} {_signed(r['return_1d'])}" for r in leaders)
    laggard_text = ", ".join(f"{r['sector']} {_signed(r['return_1d'])}" for r in laggards)
    observation = (
        f"상승 {advances}개·하락 {declines}개로 상승 비율은 {advance_pct}%였습니다. "
        f"강세 섹터는 {leader_text}, 약세 섹터는 {laggard_text}였습니다. "
        f"개별 종목 중 {top['ticker']}({_signed(top['return_1d'])})가 가장 강했고, "
        f"{bottom['ticker']}({_signed(bottom['return_1d'])})가 가장 약했습니다."
    )
    return {
        "headline": breadth,
        "observation": observation,
        "interpretation": interpretation,
        "disclaimer": "본 요약은 가격·시장 폭·섹터 수익률에 근거한 관측이며, 뉴스·실적 등 원인을 직접 분석한 투자 의견은 아닙니다.",
    }
