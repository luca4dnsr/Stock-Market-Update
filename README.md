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

> ⚠️ yfinance는 비공식 API로, 대량 요청 시 일시적으로 제한될 수 있습니다.
