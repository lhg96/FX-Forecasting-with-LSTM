# 한국은행 API 환율 정보 수집 도구

한국은행 ECOS(Economic Statistics System) API를 활용하여 주요 통화의 환율 정보를 수집하는 Python 도구입니다.

## 📁 프로젝트 구조

```
bok-exchange-rate/
├── README.md                      # 프로젝트 문서
├── README_exchange_rate.md        # 상세 API 문서
├── exchange_rate_fetcher.py       # 메인 모듈
├── test_exchange_rate.py          # 테스트 코드
├── example_usage.py               # 사용 예제
├── requirements_exchange_rate.txt # 의존성 패키지
├── data/                          # CSV 데이터 저장 폴더
└── .gitignore                     # Git 제외 파일
```

## 🚀 빠른 시작

### 1. 패키지 설치

```bash
cd bok-exchange-rate
pip install -r requirements_exchange_rate.txt
```

### 2. API 키 발급 및 설정

1. [한국은행 ECOS](https://ecos.bok.or.kr/) 가입
2. **API 인증키 신청** (홈페이지 > 인증키 신청)
3. 환경변수 설정:

```bash
# 임시 설정 (현재 터미널에만 적용)
export BOK_API_KEY='your_api_key_here'

# 영구 설정 (bash)
echo 'export BOK_API_KEY="your_api_key_here"' >> ~/.bash_profile
source ~/.bash_profile

# 영구 설정 (zsh)
echo 'export BOK_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. 테스트 실행

```bash
# Mock 테스트 (API 키 불필요)
python test_exchange_rate.py

# 또는 특정 테스트만 실행
python -m unittest test_exchange_rate.TestExchangeRateFetcher
```

### 4. 예제 실행

```bash
# 모든 예제 실행
python example_usage.py

# 직접 사용
python exchange_rate_fetcher.py
```

## 💡 주요 기능

- ✅ **실시간 환율 조회** - USD, JPY, CNY 최신 환율
- ✅ **과거 데이터 수집** - 지정 기간의 환율 데이터
- ✅ **여러 통화 동시 조회** - 한 번에 여러 통화 데이터 수집
- ✅ **통계 분석** - 최저/최고/평균/변동성 자동 계산
- ✅ **CSV 내보내기** - 데이터 저장 및 공유
- ✅ **완전한 단위 테스트** - Mock 및 통합 테스트 포함

## 📊 지원 통화

| 코드 | 통화명 | 단위 | 설명 |
|------|--------|------|------|
| USD  | 미국 달러 | 1 USD | 달러당 원화 |
| JPY  | 일본 엔화 | 100 JPY | 100엔당 원화 |
| CNY  | 중국 위안 | 1 CNY | 위안당 원화 |

## 💻 코드 예제

### 기본 사용법

```python
from exchange_rate_fetcher import ExchangeRateFetcher
import os

# API 키로 초기화
api_key = os.getenv('BOK_API_KEY')
fetcher = ExchangeRateFetcher(api_key)

# 최신 환율 조회
latest = fetcher.get_latest_rate('USD')
print(f"현재 달러 환율: {latest['rate']:,.2f}원")
print(f"조회 날짜: {latest['date']}")
```

### 과거 데이터 분석

```python
# 2024년 전체 데이터 조회
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')

# 통계 확인
print(f"최저: {df['DATA_VALUE'].min():,.2f}원")
print(f"최고: {df['DATA_VALUE'].max():,.2f}원")
print(f"평균: {df['DATA_VALUE'].mean():,.2f}원")
print(f"표준편차: {df['DATA_VALUE'].std():,.2f}원")
```

### 여러 통화 비교

```python
# 여러 통화 동시 조회
rates = fetcher.fetch_multiple_rates(
    currencies=['USD', 'JPY', 'CNY'],
    start_date='20240101',
    end_date='20241231'
)

# 각 통화별 데이터 확인
for currency, df in rates.items():
    print(f"\n{currency}:")
    print(f"  데이터 건수: {len(df)}")
    print(f"  평균 환율: {df['DATA_VALUE'].mean():,.2f}원")
```

### CSV 파일 저장

```python
# 데이터 조회 및 저장
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')
df.to_csv('data/usd_2024.csv', index=False, encoding='utf-8-sig')
print("저장 완료!")
```

## 🧪 테스트

### Mock 테스트 (API 키 불필요)

```bash
# 전체 Mock 테스트
python -m unittest test_exchange_rate.TestExchangeRateFetcher -v

# 특정 테스트만
python -m unittest test_exchange_rate.TestExchangeRateFetcher.test_init -v
```

### 통합 테스트 (API 키 필요)

```bash
# 환경변수 설정 후
export BOK_API_KEY='your_key'
python -m unittest test_exchange_rate.TestExchangeRateIntegration -v
```

## 📈 데이터 형식

반환되는 DataFrame 구조:

```python
    datetime  DATA_VALUE currency UNIT_NAME      TIME
0 2024-01-02     1280.50      USD    원/달러  20240102
1 2024-01-03     1285.30      USD    원/달러  20240103
2 2024-01-04     1282.70      USD    원/달러  20240104
```

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| datetime | datetime64 | 날짜 (Python datetime) |
| DATA_VALUE | float64 | 환율 값 |
| currency | str | 통화 코드 (USD/JPY/CNY) |
| UNIT_NAME | str | 단위명 (원/달러 등) |
| TIME | str | 원본 날짜 (YYYYMMDD) |

## ⚠️ 주의사항

1. **API 제한**: 30분당 최대 300건 요청
2. **데이터 가용성**: 공휴일/주말 데이터 없음
3. **API 키 보안**: 환경변수 사용 권장, 코드에 직접 입력 금지
4. **대량 조회**: 자동 페이징으로 처리되므로 시간이 걸릴 수 있음

## 🔗 참고 자료

- [한국은행 ECOS API 홈페이지](https://ecos.bok.or.kr/)
- [API 사용 가이드](https://ecos.bok.or.kr/api/)
- [통계표 코드 안내](https://ecos.bok.or.kr/api/#/StatisticSearch)
- [참고 블로그](https://yenpa.tistory.com/106)


---

**문의사항이나 버그 리포트는 이슈로 등록해주세요!**
