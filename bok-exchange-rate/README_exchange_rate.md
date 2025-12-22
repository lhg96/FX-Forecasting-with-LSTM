# 한국은행 API 환율 정보 수집 도구

한국은행 ECOS(Economic Statistics System) API를 활용하여 주요 통화의 환율 정보를 수집하는 Python 도구입니다.

## 주요 기능

- ✅ 실시간 환율 정보 조회 (USD, JPY, CNY)
- ✅ 과거 환율 데이터 수집
- ✅ 여러 통화 동시 조회
- ✅ 환율 통계 분석 (최저/최고/평균/변동성)
- ✅ CSV 파일 내보내기
- ✅ 단위 테스트 포함

## 사용법

### 1. 설치

```bash
pip install -r requirements_exchange_rate.txt
```

### 2. API 키 발급

1. [한국은행 ECOS](https://ecos.bok.or.kr/) 가입
2. API 인증키 신청
3. 환경변수 설정:

```bash
export BOK_API_KEY='your_api_key_here'
```

### 3. 기본 사용법

```python
from exchange_rate_fetcher import ExchangeRateFetcher

# API 키로 초기화
fetcher = ExchangeRateFetcher(api_key='your_api_key')

# 최신 환율 조회
latest = fetcher.get_latest_rate('USD')
print(f"현재 환율: {latest['rate']}원 ({latest['date']})")

# 과거 데이터 조회
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')
print(df.head())

# 여러 통화 동시 조회
rates = fetcher.fetch_multiple_rates(['USD', 'JPY', 'CNY'], '20240101', '20241231')
```

### 4. 예제 실행

```bash
# 기본 예제
python example_usage.py

# 테스트 실행
python test_exchange_rate.py
```

## 지원 통화

| 코드 | 통화명 | 설명 |
|------|--------|------|
| USD  | 미국 달러 | 1 USD = ?원 |
| JPY  | 일본 엔화 | 100엔 = ?원 |
| CNY  | 중국 위안 | 1 CNY = ?원 |

## API 사양

### ExchangeRateFetcher 클래스

#### `__init__(api_key: str)`
- API 키로 객체 초기화

#### `fetch_exchange_rate(currency, start_date, end_date, cycle='D')`
- 특정 통화의 환율 데이터 조회
- **Parameters:**
  - `currency`: 'USD', 'JPY', 'CNY'
  - `start_date`: 시작일 (YYYYMMDD)
  - `end_date`: 종료일 (YYYYMMDD)
  - `cycle`: 주기 ('D'=일별, 'M'=월별, 'Y'=년별)
- **Returns:** pandas DataFrame

#### `fetch_multiple_rates(currencies, start_date, end_date)`
- 여러 통화의 환율 데이터 동시 조회
- **Returns:** Dict[str, DataFrame]

#### `get_latest_rate(currency)`
- 최신 환율 정보 조회
- **Returns:** Dict (currency, rate, date, unit)

## 데이터 형식

반환되는 DataFrame 구조:

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| datetime | datetime64 | 날짜 |
| DATA_VALUE | float64 | 환율 값 |
| currency | str | 통화 코드 |
| UNIT_NAME | str | 단위명 |
| TIME | str | 원본 날짜 (YYYYMMDD) |

## 테스트

```bash
# Mock 테스트 (API 키 불필요)
python -m unittest test_exchange_rate.TestExchangeRateFetcher

# 통합 테스트 (API 키 필요)
export BOK_API_KEY='your_key'
python -m unittest test_exchange_rate.TestExchangeRateIntegration
```

## 예제 코드

### 최신 환율 조회

```python
fetcher = ExchangeRateFetcher(api_key)

for currency in ['USD', 'JPY', 'CNY']:
    latest = fetcher.get_latest_rate(currency)
    print(f"{currency}: {latest['rate']:,.2f}원")
```

### 환율 통계 분석

```python
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')

print(f"최저: {df['DATA_VALUE'].min():,.2f}원")
print(f"최고: {df['DATA_VALUE'].max():,.2f}원")
print(f"평균: {df['DATA_VALUE'].mean():,.2f}원")
```

### CSV 저장

```python
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')
df.to_csv('usd_rate_2024.csv', index=False, encoding='utf-8-sig')
```

## 참고 자료

- [한국은행 ECOS API](https://ecos.bok.or.kr/)
- [API 가이드](https://ecos.bok.or.kr/api/)
- [통계 코드표](https://ecos.bok.or.kr/api/#/StatisticSearch)

## 주의사항

- API 요청 제한: 30분당 300건
- 대량 데이터 조회 시 자동으로 페이징 처리
- 공휴일/주말 데이터는 없을 수 있음
- API 키는 외부에 노출되지 않도록 주의

## 라이센스

MIT License

## 참고 블로그

이 도구는 다음 블로그 글을 참고하여 작성되었습니다:
- [한국은행 API: 위안, 엔, 달러 환율과 KOSPI](https://yenpa.tistory.com/106)
