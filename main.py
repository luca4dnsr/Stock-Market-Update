"""
main.py — SPX 일간 등락률 자동화 파이프라인 진입점
"""

import argparse
import importlib.metadata
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import LOGS_DIR, MARKET_RAG_DB_FILE, OUTPUT_DIR
from fetcher import (
    UpstreamNotReady,
    fetch_all_data,
    get_expected_market_date,
)
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
from ai_insights import enrich_with_ai
from dashboard import generate_html


def _load_published_data_date(
    summary_path: Path,
    report_dir: Path,
) -> date | None:
    """검증 가능한 기존 보고서의 데이터 기준일을 읽는다."""
    if not summary_path.is_file():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        published_date = date.fromisoformat(summary["data_date"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"기존 summary.json의 data_date를 검증할 수 없습니다: {summary_path}"
        ) from exc

    expected_html = report_dir / f"SPX_daily_{published_date:%Y%m%d}.html"
    try:
        html_is_valid = (
            expected_html.is_file()
            and expected_html.stat().st_size > 0
        )
    except OSError as exc:
        raise RuntimeError(
            f"기존 HTML 보고서를 검증할 수 없습니다: {expected_html}"
        ) from exc
    if not html_is_valid:
        raise RuntimeError(
            "기존 summary.json과 일치하는 유효한 HTML 보고서가 없습니다: "
            f"{expected_html}"
        )
    return published_date


def _write_run_status(
    status_file: Path | None,
    status: str,
    **details,
) -> None:
    """워크플로가 생성·no-op을 구분하도록 임시 실행 상태를 기록한다."""
    if status_file is None:
        return
    status_file.write_text(
        json.dumps({"status": status, **details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    # yfinance는 DEBUG 로거가 켜지면 내부 멀티스레딩을 끈다. 가격 진단은
    # fetcher의 구조 로그로 남기고 외부 라이브러리의 대용량 DEBUG는 억제한다.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def _runtime_provenance() -> dict:
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    try:
        from market_rag import RETRIEVER_VERSION
    except Exception:
        RETRIEVER_VERSION = "unavailable"

    return {
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "python_version": sys.version.split()[0],
        "pandas_version": package_version("pandas"),
        "yfinance_version": package_version("yfinance"),
        "calendar_version": package_version("exchange-calendars"),
        "retriever_version": RETRIEVER_VERSION,
    }


# ──────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────

def run(
    dry_run: bool = False,
    verbose: bool = False,
    status_file: Path | None = None,
    force_same_date: bool = False,
    expected_date_override: date | None = None,
):
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    run_started_at = datetime.now(timezone.utc)
    provenance = _runtime_provenance()

    logger.info("=" * 55)
    logger.info("  SPX Daily Automation — 시작")
    logger.info(
        "  실행 provenance — SHA=%s | Python=%s | pandas=%s | yfinance=%s | "
        "calendar=%s | retriever=%s",
        provenance["git_sha"],
        provenance["python_version"],
        provenance["pandas_version"],
        provenance["yfinance_version"],
        provenance["calendar_version"],
        provenance["retriever_version"],
    )
    logger.info("=" * 55)
    t0 = datetime.now()
    published_date: date | None = None

    try:
        expected_date = (
            expected_date_override
            or get_expected_market_date(run_started_at)
        )
        if not dry_run:
            published_date = _load_published_data_date(
                OUTPUT_DIR / "summary.json",
                OUTPUT_DIR,
            )
            if (
                published_date is not None
                and published_date > expected_date
            ):
                _write_run_status(
                    status_file,
                    "published_ahead_of_calendar",
                    reason="published_date_after_expected_session",
                    expected_date=expected_date.isoformat(),
                    published_date=published_date.isoformat(),
                )
                raise RuntimeError(
                    "게시된 보고서 날짜가 XNYS 예상 완료 거래일보다 미래입니다: "
                    f"published={published_date}, expected={expected_date}"
                )
            if (
                published_date == expected_date
                and not force_same_date
            ):
                logger.info(
                    "기존 보고서가 예상 완료 거래일(%s)과 동일합니다. "
                    "Yahoo 조회와 재생성을 생략합니다.",
                    expected_date,
                )
                _write_run_status(
                    status_file,
                    "already_current",
                    reason="published_date_matches_expected_session",
                    expected_date=expected_date.isoformat(),
                    published_date=published_date.isoformat(),
                )
                return 0

        # Step 1: 데이터 수집
        logger.info("[1/5] 데이터 수집 중...")
        components, price_data, market_caps = fetch_all_data(expected_date)

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
        if date.fromisoformat(data_date) != expected_date:
            raise RuntimeError(
                "계산 결과 기준일이 XNYS 예상 완료 거래일과 다릅니다: "
                f"data={data_date}, expected={expected_date}"
            )

        # Step 5: 정렬
        logger.info("[5/5] 종목 정렬 중...")
        top_df, bottom_df = get_top_bottom(master_df)

        logger.info(
            "결과: 데이터 기준일=%s | 상승=%d | 하락=%d | 상위=%d | 하위=%d",
            data_date, advances, declines, len(top_df), len(bottom_df),
        )

        if dry_run:
            logger.info("[DRY RUN] 파일 출력 생략")
            _write_run_status(
                status_file,
                "dry_run",
                expected_date=expected_date.isoformat(),
            )
            return 0

        # 표시에 필요한 기업 프로필과 시황 문구는 파일 생성 시에만 만든다.
        top_df = add_business_summaries(top_df)
        bottom_df = add_business_summaries(bottom_df)
        base_market_summary = build_market_summary(
            sector_df, advances, declines, top_df, bottom_df
        )
        sector_returns = sector_df[["sector", "return_1d"]].to_dict("records")
        base_market_summary["breadth"] = {
            "advances": advances,
            "declines": declines,
            "total": len(master_df),
        }
        base_market_summary["sector_returns"] = sector_returns
        try:
            from market_rag import record_market_snapshot

            record_market_snapshot(
                data_date=data_date,
                advances=advances,
                declines=declines,
                sector_returns=sector_returns,
                db_path=MARKET_RAG_DB_FILE,
            )
        except Exception as exc:
            logger.warning("시장 RAG 스냅샷 저장 실패, 보고서는 계속 진행합니다: %s", exc)

        combined_df = pd.concat([top_df, bottom_df], ignore_index=True)
        combined_df, market_summary = enrich_with_ai(
            combined_df,
            data_date,
            base_market_summary,
            retrieval_as_of=datetime.now(timezone.utc),
        )
        top_df = combined_df.iloc[:len(top_df)].copy()
        bottom_df = combined_df.iloc[len(top_df):].copy()

        # ── 파일 출력 ──
        date_tag = data_date.replace("-", "") if data_date else datetime.now().strftime("%Y%m%d")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html_path = OUTPUT_DIR / f"SPX_daily_{date_tag}.html"
        generate_html(
            top_df, bottom_df, sector_df, master_df,
            advances, declines,
            html_path,
            data_date=data_date,
            generated_at=generated_at,
            market_summary=market_summary,
        )

        # ── 대시보드·이메일용 요약 JSON 저장 ──
        summary = {
            "data_date":  data_date,
            "advances":   advances,
            "declines":   declines,
            "total":      len(master_df),
            "top3":  top_df.head(3)[["ticker", "name", "return_1d"]].to_dict("records"),
            "bot3":  bottom_df.head(3)[["ticker", "name", "return_1d"]].to_dict("records"),
            "market_summary": market_summary,
            "market_snapshot": {
                "breadth": base_market_summary["breadth"],
                "sector_returns": sector_returns,
            },
            "pipeline_status": {
                "terminal_stage": "report_success",
                "rag_attempted": bool(market_summary.get("rag_attempted", False)),
                "rag_status": str(market_summary.get("rag_status", "legacy")),
                "fallback_stage": str(market_summary.get("fallback_stage", "unknown")),
            },
            "build": provenance,
            "run_started_at": run_started_at.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("summary.json 저장 완료")

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("=" * 55)
        logger.info("  완료 in %.1f초", elapsed)
        logger.info("  🌐 HTML  : %s", html_path)
        logger.info("=" * 55)
        _write_run_status(
            status_file,
            "generated",
            expected_date=expected_date.isoformat(),
            published_date=(
                published_date.isoformat()
                if published_date is not None
                else None
            ),
            html_path=str(html_path),
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        sys.exit(0)
    except UpstreamNotReady as exc:
        logger.warning("Yahoo upstream 미준비: %s", exc)
        _write_run_status(
            status_file,
            "source_not_ready",
            published_date=(
                published_date.isoformat()
                if published_date is not None
                else None
            ),
            **exc.status_details(),
        )
        return 75
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
    parser.add_argument(
        "--status-file", type=Path,
        help="GitHub Actions용 임시 실행 상태 JSON 경로",
    )
    parser.add_argument(
        "--force-same-date", action="store_true",
        help="기존 게시일과 예상 거래일이 같아도 보고서를 다시 생성",
    )
    parser.add_argument(
        "--expected-date",
        type=date.fromisoformat,
        help="재시도 간 고정할 예상 거래일 YYYY-MM-DD (워크플로 내부용)",
    )
    args = parser.parse_args()
    sys.exit(
        run(
            dry_run=args.dry_run,
            verbose=args.verbose,
            status_file=args.status_file,
            force_same_date=args.force_same_date,
            expected_date_override=args.expected_date,
        )
    )
