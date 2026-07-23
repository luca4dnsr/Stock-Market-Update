"""
dashboard.py — 프리미엄 다크모드 HTML 대시보드 생성
"""

import json
import logging
from html import escape
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────

def _safe(val, fmt=".1f") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:{fmt}}"


def _signed(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:+.1f}"


def _mcap_str(val_b) -> str:
    """십억달러를 T/B 약자로 표현."""
    if val_b is None or (isinstance(val_b, float) and pd.isna(val_b)):
        return "—"
    if val_b >= 1000:
        return f"${val_b/1000:.2f}T"
    return f"${val_b:.1f}B"


def _color_class(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "flat"
    if val > 0:
        return "pos"
    if val < 0:
        return "neg"
    return "flat"


def _sector_color(val) -> str:
    """수익률 크기에 따라 섹터 타일 배경색 반환 (rgba)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "rgba(60,70,90,0.5)"
    if val >= 2:
        return "rgba(200,40,40,0.75)"
    if val >= 0.5:
        return "rgba(180,60,60,0.5)"
    if val > 0:
        return "rgba(150,80,80,0.3)"
    if val > -0.5:
        return "rgba(60,100,180,0.3)"
    if val > -2:
        return "rgba(50,90,200,0.5)"
    return "rgba(30,70,220,0.75)"


def _build_stock_rows(df: pd.DataFrame, section_class: str) -> str:
    rows = []
    for _, r in df.iterrows():
        r1d = r.get("return_1d")
        ticker = str(r.get("ticker", ""))
        name   = str(r.get("name", ""))[:32]
        mc_b   = r.get("market_cap_b")
        mc_rank = r.get("mc_rank", "")
        day_rank = r.get("day_rank", "")
        sector = str(r.get("sector", ""))
        business = str(r.get("business_summary", ""))
        move_reason = str(r.get("move_reason", ""))

        pct_cells = ""
        for val in [r.get("return_1d"), r.get("return_1w"),
                    r.get("return_1m"), r.get("return_3m")]:
            cc = _color_class(val)
            pct_cells += f'<td class="pct {cc}">{_signed(val)}%</td>'

        rows.append(f"""
        <tr class="{section_class}">
          <td class="rank-cell">{day_rank}</td>
          <td class="ticker-cell">{escape(ticker)}</td>
          <td class="name-cell" title="{escape(name)}">{escape(name)}</td>
          <td class="num-cell">{_mcap_str(mc_b)}</td>
          {pct_cells}
          <td class="num-cell">{mc_rank}</td>
          <td class="sector-cell">{escape(sector)}</td>
          <td class="business-cell">{escape(business)}</td>
          <td class="reason-cell">{escape(move_reason)}</td>
        </tr>""")
    return "\n".join(rows)


def _build_sector_tiles(sector_df: pd.DataFrame) -> str:
    tiles = []
    for _, r in sector_df.iterrows():
        sec  = str(r["sector"])
        val  = r["return_1d"]
        bg   = _sector_color(val)
        cc   = _color_class(val)
        sign = _signed(val)
        tiles.append(f"""
        <div class="sector-tile" style="background:{bg};">
          <div class="sector-name">{sec}</div>
          <div class="sector-val {cc}">{sign}%</div>
        </div>""")
    return "\n".join(tiles)


def _build_market_summary_html(market_summary: dict) -> str:
    """검증된 시황 요약과 Finnhub 기사 근거를 안전한 HTML로 만든다."""
    links = []
    for title, url in zip(
        market_summary.get("source_titles", []), market_summary.get("source_urls", [])
    ):
        safe_url = str(url).strip()
        if safe_url.startswith(("https://", "http://")):
            links.append(
                f'<a href="{escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                f"{escape(str(title) or '기사')}</a>"
            )
    sources_html = (
        f'<p class="market-summary-sources"><strong>근거</strong>{" · ".join(links)}</p>'
        if links
        else ""
    )
    return f"""
  <section class="market-summary-card">
    <div class="market-summary-label">MARKET TAKEAWAY</div>
    <h2>📈 {escape(market_summary['headline'])}</h2>
    <p><strong>관측</strong>{escape(market_summary['observation'])}</p>
    <p><strong>해석</strong>{escape(market_summary['interpretation'])}</p>
    {sources_html}
    <p class="market-summary-note">{escape(market_summary['disclaimer'])}</p>
  </section>"""


# ──────────────────────────────────────────────────────────
# HTML 템플릿
# ──────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SPX Daily Monitor — {data_date}</title>
  <meta name="description" content="S&P 500 일간 등락률 대시보드 {data_date}"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:       #080c14;
      --surface:  #0f1626;
      --card:     rgba(18, 28, 48, 0.85);
      --border:   rgba(255,255,255,0.08);
      --text:     #dde6f4;
      --muted:    #6b7fa0;
      --pos:      #ff4d4d;
      --neg:      #4d94ff;
      --gold:     #f0a500;
      --radius:   14px;
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ── Animated background ── */
    body::before {{
      content: '';
      position: fixed; inset: 0; z-index: -1;
      background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(255,77,77,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(77,148,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 100% 80% at 50% 50%, rgba(8,12,20,1) 0%, rgba(4,8,18,1) 100%);
      animation: bgShift 12s ease-in-out infinite alternate;
    }}
    @keyframes bgShift {{
      from {{ background-position: 0% 0%; }}
      to   {{ background-position: 100% 100%; }}
    }}

    /* ── Layout ── */
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 20px; }}

    /* ── Header ── */
    header {{
      display: flex; align-items: center;
      justify-content: space-between; flex-wrap: wrap; gap: 16px;
      padding: 32px 0 24px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 28px;
    }}
    .header-left h1 {{
      font-size: clamp(24px, 4vw, 36px);
      font-weight: 900;
      letter-spacing: -0.04em;
      background: linear-gradient(135deg, #fff 30%, var(--gold));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .header-left .subtitle {{
      color: var(--muted); font-size: 13px; margin-top: 4px;
      font-weight: 400; letter-spacing: 0.02em;
    }}
    .header-date {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px; color: var(--muted);
      background: var(--card);
      border: 1px solid var(--border);
      padding: 8px 16px; border-radius: 8px;
    }}

    /* ── Market Breadth ── */
    .breadth-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 28px;
      margin-bottom: 28px;
      backdrop-filter: blur(12px);
    }}
    .breadth-label {{
      font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 12px;
    }}
    .breadth-row {{
      display: flex; align-items: center; gap: 16px;
    }}
    .breadth-num {{ font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; min-width: 56px; }}
    .breadth-num.pos {{ color: var(--pos); }}
    .breadth-num.neg {{ color: var(--neg); }}
    .breadth-bar-wrap {{
      flex: 1; height: 12px; border-radius: 6px;
      background: rgba(255,255,255,0.07);
      overflow: hidden;
      position: relative;
    }}
    .breadth-bar-fill {{
      height: 100%; border-radius: 6px;
      background: linear-gradient(90deg, var(--pos) 0%, rgba(255,77,77,0.5) 100%);
      transition: width 1.2s cubic-bezier(.4,0,.2,1);
    }}
    .breadth-desc {{ font-size: 12px; color: var(--muted); }}

    /* ── Sector Heatmap ── */
    .section-title {{
      font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
      text-transform: uppercase; color: var(--muted);
      margin-bottom: 14px;
    }}
    .sector-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 32px;
    }}
    .sector-tile {{
      border-radius: 10px;
      padding: 14px 16px;
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(8px);
      transition: transform 0.2s, box-shadow 0.2s;
      cursor: default;
    }}
    .sector-tile:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }}
    .sector-name {{ font-size: 11px; color: rgba(220,230,248,0.75); margin-bottom: 6px; line-height: 1.3; }}
    .sector-val   {{ font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; }}

    /* ── Daily market summary ── */
    .market-summary-card {{
      margin-top: 28px; padding: 24px 28px;
      background: linear-gradient(135deg, rgba(27, 57, 93, 0.92), var(--card));
      border: 1px solid rgba(128, 179, 255, 0.28);
      border-radius: var(--radius); backdrop-filter: blur(12px);
    }}
    .market-summary-label {{
      color: #8bbdff; font-size: 10px; font-weight: 700;
      letter-spacing: .14em; margin-bottom: 8px;
    }}
    .market-summary-card h2 {{ font-size: 17px; margin-bottom: 16px; }}
    .market-summary-card p {{ color: rgba(220,230,248,.86); font-size: 13px; line-height: 1.75; margin: 8px 0; }}
    .market-summary-card strong {{ color: #fff; display: inline-block; width: 42px; }}
    .market-summary-sources a {{ color: #9bc8ff; text-decoration: underline; }}
    .market-summary-note {{ color: var(--muted) !important; font-size: 11px !important; margin-top: 14px !important; }}

    /* ── Tables ── */
    .tables-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }}
    .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      backdrop-filter: blur(12px);
    }}
    .table-header {{
      padding: 16px 20px 12px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 10px;
    }}
    .table-header h2 {{ font-size: 14px; font-weight: 700; }}
    .table-wrapper {{ overflow-x: auto; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    thead th {{
      padding: 10px 12px;
      text-align: right;
      font-size: 10.5px; font-weight: 600;
      letter-spacing: 0.06em; text-transform: uppercase;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      cursor: pointer; user-select: none;
      white-space: nowrap;
      background: rgba(255,255,255,0.02);
    }}
    thead th:hover {{ color: var(--text); }}
    thead th.left-align {{ text-align: left; }}

    tbody tr {{
      border-bottom: 1px solid rgba(255,255,255,0.04);
      transition: background 0.15s;
    }}
    tbody tr:hover {{ background: rgba(255,255,255,0.04); }}

    td {{ padding: 9px 12px; vertical-align: middle; white-space: nowrap; }}

    .rank-cell   {{ color: var(--muted); font-size: 11px; text-align: center; width: 36px; }}
    .ticker-cell {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 12px; }}
    .name-cell   {{ max-width: 220px; overflow: hidden; text-overflow: ellipsis; color: rgba(220,230,248,0.8); }}
    .num-cell    {{ text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); }}
    .sector-cell {{ font-size: 11px; color: var(--muted); max-width: 180px; overflow: hidden; text-overflow: ellipsis; }}
    .business-cell, .reason-cell {{
      white-space: normal; min-width: 210px; max-width: 300px;
      color: rgba(220,230,248,.84); font-size: 11px; line-height: 1.55;
    }}
    .reason-cell {{ color: #9ed5ff; min-width: 270px; }}

    .pct {{
      text-align: right;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      font-size: 12.5px;
    }}
    .pct.pos  {{ color: var(--pos); }}
    .pct.neg  {{ color: var(--neg); }}
    .pct.flat {{ color: var(--muted); }}

    /* 1D 수익률 하이라이트 */
    tr.top-row td.pct:first-of-type  {{ background: rgba(255,77,77,0.07); border-radius: 4px; }}
    tr.bot-row td.pct:first-of-type  {{ background: rgba(77,148,255,0.07); border-radius: 4px; }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      padding: 40px 0 24px;
      color: var(--muted);
      font-size: 11px;
    }}

    /* ── Color utils ── */
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}

    /* ── Responsive ── */
    @media (min-width: 1100px) {{
      .tables-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 600px) {{
      .sector-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    /* ── Sort indicator ── */
    th[data-sorted="asc"]::after  {{ content: " ▲"; font-size: 9px; }}
    th[data-sorted="desc"]::after {{ content: " ▼"; font-size: 9px; }}
  </style>
</head>
<body>
<div class="container">

  <!-- ── Header ── -->
  <header>
    <div class="header-left">
      <h1>SPX Daily Monitor</h1>
      <div class="subtitle">S&amp;P 500 — 일간 등락률 자동 대시보드</div>
    </div>
    <div class="header-date">📅 {data_date}</div>
  </header>

  <!-- ── Market Breadth ── -->
  <div class="breadth-card">
    <div class="breadth-label">Market Breadth — 상승 vs 하락 종목 수</div>
    <div class="breadth-row">
      <span class="breadth-num pos">↑ {advances}</span>
      <div class="breadth-bar-wrap">
        <div class="breadth-bar-fill" id="breadthFill" style="width:0%"></div>
      </div>
      <span class="breadth-num neg">↓ {declines}</span>
      <span class="breadth-desc">{total} 종목 중</span>
    </div>
  </div>

  <!-- ── Sector Heatmap ── -->
  <div class="section-title">📊 Sector Performance (시총 가중 평균)</div>
  <div class="sector-grid">
    {sector_tiles}
  </div>

  <!-- ── Top / Bottom Tables ── -->
  <div class="tables-grid">

    <!-- Top -->
    <div class="table-card">
      <div class="table-header">
        <h2>🚀 상위 종목 TOP {top_n}</h2>
      </div>
      <div class="table-wrapper">
        <table id="topTable">
          <thead>
            <tr>
              <th>#</th>
              <th class="left-align">티커</th>
              <th class="left-align">종목명</th>
              <th>시총</th>
              <th>1일%</th>
              <th>1주%</th>
              <th>1개월%</th>
              <th>3개월%</th>
              <th>시총순위</th>
              <th class="left-align">섹터</th>
              <th class="left-align">사업</th>
              <th class="left-align">등락 이유</th>
            </tr>
          </thead>
          <tbody>
            {top_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Bottom -->
    <div class="table-card">
      <div class="table-header">
        <h2>📉 하위 종목 BOTTOM {bottom_n}</h2>
      </div>
      <div class="table-wrapper">
        <table id="botTable">
          <thead>
            <tr>
              <th>#</th>
              <th class="left-align">티커</th>
              <th class="left-align">종목명</th>
              <th>시총</th>
              <th>1일%</th>
              <th>1주%</th>
              <th>1개월%</th>
              <th>3개월%</th>
              <th>시총순위</th>
              <th class="left-align">섹터</th>
              <th class="left-align">사업</th>
              <th class="left-align">등락 이유</th>
            </tr>
          </thead>
          <tbody>
            {bottom_rows}
          </tbody>
        </table>
      </div>
    </div>

  </div><!-- /tables-grid -->

  {market_summary_html}

  <footer>
    데이터 출처: Yahoo Finance (yfinance) &nbsp;|&nbsp;
    자동 생성: {generated_at} &nbsp;|&nbsp;
    <a href="https://github.com" style="color:var(--muted);">GitHub Actions</a>
  </footer>

</div><!-- /container -->

<script>
  // ── Breadth bar animation ──
  window.addEventListener('load', () => {{
    const fill = document.getElementById('breadthFill');
    const pct = {adv_pct};
    setTimeout(() => {{ fill.style.width = pct + '%'; }}, 200);
  }});

  // ── Table sort ──
  function makeSortable(tableId) {{
    const table = document.getElementById(tableId);
    if (!table) return;
    const headers = table.querySelectorAll('thead th');
    headers.forEach((th, colIdx) => {{
      th.addEventListener('click', () => {{
        const tbody = table.querySelector('tbody');
        const rows  = Array.from(tbody.querySelectorAll('tr'));
        const asc   = th.dataset.sorted !== 'asc';

        rows.sort((a, b) => {{
          const aVal = a.cells[colIdx]?.innerText.trim().replace(/[%$,↑↓+]/g, '') || '';
          const bVal = b.cells[colIdx]?.innerText.trim().replace(/[%$,↑↓+]/g, '') || '';
          const aNum = parseFloat(aVal);
          const bNum = parseFloat(bVal);
          if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
          return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        }});

        headers.forEach(h => delete h.dataset.sorted);
        th.dataset.sorted = asc ? 'asc' : 'desc';
        rows.forEach(r => tbody.appendChild(r));
      }});
    }});
  }}

  makeSortable('topTable');
  makeSortable('botTable');
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────
# 메인 함수
# ──────────────────────────────────────────────────────────

def generate_html(
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    master_df: pd.DataFrame,
    advances: int,
    declines: int,
    output_path: Path,
    data_date: str = "",
    generated_at: str = "",
    market_summary: dict | None = None,
):
    """
    프리미엄 다크모드 HTML 대시보드를 생성한다.
    """
    total = len(master_df)
    adv_pct = round(advances / max(advances + declines, 1) * 100, 1)
    html = HTML_TEMPLATE.format(
        data_date     = data_date,
        advances      = advances,
        declines      = declines,
        total         = total,
        adv_pct       = adv_pct,
        sector_tiles  = _build_sector_tiles(sector_df),
        top_rows      = _build_stock_rows(top_df, "top-row"),
        bottom_rows   = _build_stock_rows(bottom_df, "bot-row"),
        market_summary_html = _build_market_summary_html(market_summary) if market_summary else "",
        top_n         = len(top_df),
        bottom_n      = len(bottom_df),
        generated_at  = generated_at,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML 저장 완료: %s", output_path)
