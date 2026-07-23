# 📈 SPX Daily Monitor

S&P 500 일간 등락률 **자동 대시보드** — 매일 아침 자동으로 Excel + HTML 보고서를 생성합니다.

## 기능

| 기능 | 설명 |
|------|------|
| 📊 **Excel 자동 생성** | 원본 레이아웃(섹터표 + SPX 타이틀 + 상/하위 종목표) 재현 |
| 🌐 **HTML 대시보드** | 프리미엄 다크모드, 섹터 히트맵, 정렬 가능한 테이블 |
| ⏰ **GitHub Actions 자동화** | 평일 미국 장 마감 후 자동 실행 (KST 기준 다음날 오전 7시) |
| 🔗 **GitHub Pages** | 최신 HTML이 자동으로 웹에 게시됨 |
| ✅ **데이터 품질 검증** | 주가·시가총액 커버리지와 최신 거래일 정합성 기준 미달 시 실패 처리 |
| 🤖 **한글 AI 인사이트** | Yahoo Finance + Finnhub 뉴스 → Gemini 3.6 Flash → GPT-OSS → 규칙 기반 문구 |

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
```

### 2. Python 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 로컬 실행

```bash
# 정상 실행 (output/ 에 Excel + HTML 생성)
python main.py

# 데이터 수집만 테스트 (파일 미생성)
python main.py --dry-run

# 상세 로그 출력
python main.py --verbose
```

### 4. GitHub 자동화 설정

#### ① GitHub 저장소 생성 후 Push

```bash
git init
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git add .
git commit -m "🚀 Initial commit"
git push -u origin main
```

#### ② GitHub Pages 활성화

`Settings → Pages → Source: Deploy from branch → Branch: main / docs/`

이후 `https://<YOUR_USERNAME>.github.io/<YOUR_REPO>/` 에서 최신 대시보드를 확인할 수 있습니다.

#### ④ AI 인사이트 API 설정 (선택)

`Settings → Secrets and variables → Actions`에서 아래 Repository Secret을 등록합니다.

- `GEMINI_API_KEY` — Google AI Studio의 `gemini-3.6-flash`에 사용
- `FINNHUB_API_KEY` — Finnhub의 종목별·시장 전체 뉴스에 사용
- `NVIDIA_API_KEY` — NVIDIA NIM의 `openai/gpt-oss-120b`에 사용

주가·시가총액·섹터·기업 기본 설명과 상승·하락 Top 20 선정은 **Yahoo Finance**를 유지합니다. 뉴스는 **Finnhub**만 사용합니다. 상승 20개·하락 20개 종목마다 거래일 전 2일~후 1일의 `company-news`를 받고, 시장 전체는 `general news`를 받습니다. Gemini는 웹 검색 없이 이 기사 제목·요약·발행 시각·URL만 한국어로 해석합니다. Finnhub 기사 ID·발행일·관련 티커가 코드 검증을 통과한 경우에만 종목 등락 이유를 작성하며, 근거가 부족하면 `당일 전후의 종목 직접 관련 뉴스·공시 근거를 충분히 확인하지 못했습니다.`라고 표시합니다.

실행 순서는 **Gemini 3.6 Flash → NVIDIA NIM GPT-OSS 120B → 규칙 기반 제한 문구**입니다. GPT-OSS는 Gemini가 실패했을 때 사업 설명과 가격·섹터 기반 관측만 보완합니다. GPT-OSS도 실패하면 규칙 기반 문구를 표시하며, 어느 경우에도 검증되지 않은 뉴스성 등락 이유나 시황 인과관계를 만들지 않습니다.

시황 요약은 Finnhub 일반 시장 기사 중 거래일 창을 통과한 서로 다른 기사 3~5건이 있어야 해석을 채택하며, 대시보드와 Excel에 근거 기사 링크를 함께 남깁니다. 기사 근거가 부족하면 가격·시장 폭·섹터 수익률에 한정된 관측과 제한 문구를 표시합니다.

#### ③ 자동 실행 확인

`Actions` 탭에서 **SPX Daily Update** 워크플로우가 등록된 것을 확인합니다.
수동 실행을 원할 경우 `Run workflow` 버튼을 클릭하세요.

---

## 파일 구조

```
📁 Stock Market Update/
├── main.py            # 진입점 — 전체 파이프라인 조율
├── config.py          # 전역 설정 (표시 종목 수, 배치 크기 등)
├── fetcher.py         # S&P 500 구성종목 + 주가 + 시가총액 수집
├── calculator.py      # 수익률 / 섹터 집계 / 시장 폭 계산
├── ranker.py          # 상위/하위 종목 정렬
├── excel_writer.py    # Excel 파일 생성 (openpyxl)
├── dashboard.py       # HTML 대시보드 생성
├── ai_insights.py     # Yahoo Finance + Finnhub → Gemini → GPT-OSS → 규칙 기반 검증
├── requirements.txt   # Python 의존성
├── .gitignore
├── cache/             # S&P 500 구성종목 캐시 (자동 생성, Actions cache로 복원)
├── output/            # 생성된 Excel + HTML (Actions이 커밋)
├── docs/              # GitHub Pages용 HTML (Actions이 관리)
├── logs/              # 실행 로그 (gitignored)
└── .github/
    └── workflows/
        └── daily_spx.yml   # GitHub Actions 워크플로우
```

---

## 설정 변경

`config.py` 에서 주요 파라미터를 수정할 수 있습니다:

```python
TOP_N    = 20    # 상위 표시 종목 수
BOTTOM_N = 20    # 하위 표시 종목 수
BATCH_SIZE = 50  # yfinance 배치당 티커 수 (속도 vs 안정성 조절)
MIN_PRICE_COVERAGE = 0.98  # 최소 주가 수집 비율
```

---

## 자동 실행 스케줄

| 이벤트 | 시각 |
|--------|------|
| GitHub Actions 실행 | UTC 22:00 (월~금) |
| KST 기준 | 다음날 07:00 (화~토) |
| 데이터 기준 | 미국 장 마감 (전일 종가) |

---

## 데이터 소스

- **주가 데이터**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance 비공식 API)
- **S&P 500 구성종목**: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- **섹터 분류**: GICS (Global Industry Classification Standard)
- **등락 이유·시황 뉴스**: Finnhub 뉴스 API + Gemini 해석 (대시보드·Excel에 검증된 기사 URL 표시)

> ⚠️ yfinance는 비공식 API로, 대량 요청 시 일시적으로 제한될 수 있습니다.
