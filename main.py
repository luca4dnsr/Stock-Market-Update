"""
main.py — SPX 일간 등락률 자동화 파이프라인 진입점
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import LOGS_DIR, OUTPUT_DIR
from fetcher import fetch_all_data
from calculator import (
    build_master_df,
    calculate_returns,
    calculate_sector_returns,
    count_advances_declines,
    get_data_date,
)
from ranker import get_top_bottom
from company_profiles import add_business_summaries
from market_summary import build_market_summary
from excel_writer import write_excel
from dashboard import generate_html


# ──────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    log_file = LOGS_DIR / f"spx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ──────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────

def run(dry_run: bool = False, verbose: bool = False):
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 55)
    logger.info("  SPX Daily Automation — 시작")
    logger.info("=" * 55)
    t0 = datetime.now()

    try:
        # Step 1: 데이터 수집
        logger.info("[1/5] 데이터 수집 중...")
        components, price_data, market_caps = fetch_all_data()

        # Step 2: 수익률 계산
        logger.info("[2/5] 수익률 계산 중...")
        returns_df = calculate_returns(price_data)

        # Step 3: 마스터 DataFrame 구성
        logger.info("[3/5] 데이터 통합 중...")
        master_df = build_master_df(components, returns_df, market_caps)

        # Step 4: 집계 지표
        logger.info("[4/5] 집계 지표 계산 중...")
        sector_df = calculate_sector_returns(master_df)
        advances, declines = count_advances_declines(master_df)
        data_date = get_data_date(master_df)

        # 전일 날짜 추정 (표시용)
        try:
            dt = datetime.strptime(data_date, "%Y-%m-%d")
            prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            prev_date = ""

        # Step 5: 정렬
        logger.info("[5/5] 종목 정렬 중...")
        top_df, bottom_df = get_top_bottom(master_df)

        logger.info(
            "결과: 데이터 기준일=%s | 상승=%d | 하락=%d | 상위=%d | 하위=%d",
            data_date, advances, declines, len(top_df), len(bottom_df),
        )

        if dry_run:
            logger.info("[DRY RUN] 파일 출력 생략")
            return

        # 표시에 필요한 기업 프로필과 시황 문구는 파일 생성 시에만 만든다.
        top_df = add_business_summaries(top_df)
        bottom_df = add_business_summaries(bottom_df)
        market_summary = build_market_summary(
            sector_df, advances, declines, top_df, bottom_df
        )

        # ── 파일 출력 ──
        date_tag = data_date.replace("-", "") if data_date else datetime.now().strftime("%Y%m%d")
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        excel_path = OUTPUT_DIR / f"SPX_daily_{date_tag}.xlsx"
        write_excel(
            top_df, bottom_df, sector_df,
            advances, declines,
            excel_path,
            data_date=data_date,
            prev_date=prev_date,
            market_summary=market_summary,
        )

        html_path = OUTPUT_DIR / f"SPX_daily_{date_tag}.html"
        generate_html(
            top_df, bottom_df, sector_df, master_df,
            advances, declines,
            html_path,
            data_date=data_date,
            generated_at=generated_at,
            market_summary=market_summary,
        )

        # ── 이메일용 요약 JSON 저장 ──
        import json
        summary = {
            "data_date":  data_date,
            "advances":   advances,
            "declines":   declines,
            "total":      len(master_df),
            "top3":  top_df.head(3)[["ticker", "name", "return_1d"]].to_dict("records"),
            "bot3":  bottom_df.head(3)[["ticker", "name", "return_1d"]].to_dict("records"),
            "excel_name": excel_path.name,
            "market_summary": market_summary,
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("summary.json 저장 완료")

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("=" * 55)
        logger.info("  완료 in %.1f초", elapsed)
        logger.info("  📊 Excel : %s", excel_path)
        logger.info("  🌐 HTML  : %s", html_path)
        logger.info("=" * 55)

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as exc:
        logger.exception("치명적 오류: %s", exc)
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPX Daily Automation")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="데이터 수집/계산만 수행, 파일 출력 없음",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="상세 로그 출력",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run, verbose=args.verbose)
