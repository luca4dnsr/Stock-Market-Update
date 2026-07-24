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

# ── 데이터 품질 기준 ────────────────────────────────────
# 지수 구성 변경·상장폐지 등을 고려해 절대 종목 수 대신 비율로 검증한다.
# 기준을 충족하지 못하면 불완전한 일간 리포트를 만들지 않고 실행을 실패시킨다.
MIN_PRICE_COVERAGE       = 0.98
MIN_MARKET_CAP_COVERAGE  = 0.95
MIN_LATEST_DATE_COVERAGE = 0.98

# ── 기업 사업 요약 ──────────────────────────────────────
BUSINESS_PROFILE_CACHE_FILE = CACHE_DIR / "business_profiles.json"
BUSINESS_PROFILE_CACHE_DAYS = 30
PROFILE_FETCH_WORKERS       = 6

# ── AI 인사이트 공급자 우선순위 ───────────────────────────
# 1) Gemini (Finnhub 기사 해석)  2) NVIDIA NIM GPT-OSS  3) 규칙 기반 제한 문구
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-3.6-flash"

# Yahoo Finance는 주가·기업 기본정보를 유지하고, 뉴스 근거는 Finnhub에서만 받는다.
FINNHUB_API_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_REQUEST_TIMEOUT_SEC = 25
# 무료 키의 호출량을 보수적으로 관리한다. 상·하위 40종목 조회는 약 40초가 걸린다.
FINNHUB_NEWS_REQUEST_DELAY_SEC = 1.0
FINNHUB_NEWS_MAX_PER_TICKER = 3
FINNHUB_MARKET_NEWS_CATEGORY = "general"
# /news는 최신 기사만 반환하므로, 매일 받은 기사를 캐시에 누적해 한 달 창을 만든다.
FINNHUB_MARKET_NEWS_POOL_FILE = CACHE_DIR / "finnhub_market_news.json"
FINNHUB_MARKET_NEWS_POOL_MAX_ITEMS = 300

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_GPT_OSS_MODEL = "openai/gpt-oss-120b"

AI_INSIGHTS_CACHE_FILE = CACHE_DIR / "ai_daily_insights.json"
AI_INSIGHTS_CACHE_VERSION = "v9-deterministic-finnhub-sources"
GEMINI_INSIGHTS_MAX_TOKENS = 3000
GEMINI_INSIGHTS_BATCH_SIZE = 4
GEMINI_INSIGHTS_TIMEOUT_SEC = 120
# 거래일 30일 전부터 다음 날까지를 확인한다. 시장 전체 뉴스는 롤링 캐시로 누적한다.
NEWS_WINDOW_DAYS_BEFORE = 30
NEWS_WINDOW_DAYS_AFTER = 1
MARKET_MIN_NEWS_SOURCES = 3
MARKET_MAX_NEWS_SOURCES = 5
# 사업 설명·뉴스 해석을 함께 생성할 수 있도록 여유를 둔다.
NIM_INSIGHTS_MAX_TOKENS = 1200
NIM_CONNECT_TIMEOUT_SEC = 10
NIM_READ_TIMEOUT_SEC = 60

# ── 캐시 설정 ──────────────────────────────────────────
SP500_CACHE_DAYS   = 7     # S&P 500 구성종목 캐시 유효기간(일)
SP500_CACHE_FILE   = CACHE_DIR / "sp500_components.csv"

# ── S&P 500 구성종목 소스 ──────────────────────────────
SP500_WIKI_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
