"""
config.py — 전역 설정 및 디렉토리 초기화
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# 디렉토리 자동 생성
for _d in [CACHE_DIR, OUTPUT_DIR, LOGS_DIR]:
    _d.mkdir(exist_ok=True)

# ── 표시 설정 ──────────────────────────────────────────
TOP_N    = 20   # 상위 N개 종목 표시
BOTTOM_N = 20   # 하위 N개 종목 표시

# ── 수익률 계산 기준 (거래일) ──────────────────────────
DAYS_1W = 5
DAYS_1M = 21
DAYS_3M = 63

# ── 데이터 수집 설정 ───────────────────────────────────
BATCH_SIZE         = 50    # yfinance 배치당 티커 수
REQUEST_DELAY_SEC  = 0.5   # 배치 간 대기(초)
MC_FETCH_WORKERS   = 20    # 시가총액 조회 동시 스레드 수
MC_TIMEOUT_SEC     = 120   # 시가총액 조회 전체 타임아웃(초)

# ── 캐시 설정 ──────────────────────────────────────────
SP500_CACHE_DAYS   = 7     # S&P 500 구성종목 캐시 유효기간(일)
SP500_CACHE_FILE   = CACHE_DIR / "sp500_components.csv"

# ── S&P 500 구성종목 소스 ──────────────────────────────
SP500_WIKI_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
