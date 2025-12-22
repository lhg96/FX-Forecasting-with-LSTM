"""
한국은행 API를 활용한 환율 정보 수집 모듈
"""
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv


class ExchangeRateFetcher:
    """한국은행 ECOS API를 사용하여 환율 정보를 가져오는 클래스"""
    
    BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
    STAT_CODE = "731Y001"  # 주요국 통화의 대원화환율
    
    # 통화 코드
    CURRENCY_CODES = {
        "USD": "0000001",  # 달러
        "JPY": "0000002",  # 엔화 (100엔당)
        "CNY": "0000053",  # 위안
    }
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: 한국은행 ECOS API 인증키
        """
        self.api_key = api_key
        
    def fetch_exchange_rate(
        self, 
        currency: str, 
        start_date: str, 
        end_date: str,
        cycle: str = "D"
    ) -> pd.DataFrame:
        """
        특정 통화의 환율 정보를 가져옵니다.
        
        Args:
            currency: 통화 코드 (USD, JPY, CNY)
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            cycle: 주기 (D: 일별, M: 월별, Y: 년별)
            
        Returns:
            환율 정보가 담긴 DataFrame
        """
        if currency not in self.CURRENCY_CODES:
            raise ValueError(f"지원하지 않는 통화입니다: {currency}")
            
        currency_code = self.CURRENCY_CODES[currency]
        
        # 첫 요청으로 전체 데이터 개수 확인
        url = f"{self.BASE_URL}/{self.api_key}/json/kr/1/100/{self.STAT_CODE}/{cycle}/{start_date}/{end_date}/{currency_code}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            
            if 'StatisticSearch' not in result:
                raise ValueError(f"API 응답 오류: {result}")
            
            list_total_count = int(result['StatisticSearch']['list_total_count'])
            list_count = (list_total_count // 100) + 1
            
            # 모든 데이터 수집
            all_rows = []
            for i in range(list_count):
                start_idx = str(i * 100 + 1)
                end_idx = str((i + 1) * 100)
                
                url = f"{self.BASE_URL}/{self.api_key}/json/kr/{start_idx}/{end_idx}/{self.STAT_CODE}/{cycle}/{start_date}/{end_date}/{currency_code}"
                response = requests.get(url)
                response.raise_for_status()
                result = response.json()
                
                if 'StatisticSearch' in result and 'row' in result['StatisticSearch']:
                    all_rows.extend(result['StatisticSearch']['row'])
            
            # DataFrame 생성 및 데이터 변환
            df = pd.DataFrame(all_rows)
            df['datetime'] = pd.to_datetime(
                df['TIME'].str[:4] + '-' + 
                df['TIME'].str[4:6] + '-' + 
                df['TIME'].str[6:8]
            )
            df['DATA_VALUE'] = df['DATA_VALUE'].astype(float)
            df['currency'] = currency
            
            return df[['datetime', 'DATA_VALUE', 'currency', 'UNIT_NAME', 'TIME']]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {e}")
        except (KeyError, ValueError) as e:
            raise Exception(f"데이터 처리 실패: {e}")
    
    def fetch_multiple_rates(
        self, 
        currencies: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 통화의 환율 정보를 한번에 가져옵니다.
        
        Args:
            currencies: 통화 코드 리스트 (예: ['USD', 'JPY', 'CNY'])
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            
        Returns:
            통화별 환율 정보 딕셔너리
        """
        results = {}
        for currency in currencies:
            print(f"{currency} 환율 데이터 수집 중...")
            df = self.fetch_exchange_rate(currency, start_date, end_date)
            results[currency] = df
            print(f"{currency} 환율: {len(df)}건 수집 완료")
        
        return results
    
    def get_latest_rate(self, currency: str) -> Optional[Dict]:
        """
        최신 환율 정보를 가져옵니다.
        
        Args:
            currency: 통화 코드 (USD, JPY, CNY)
            
        Returns:
            최신 환율 정보
        """
        end_date = datetime.now().strftime('%Y%m%d')
        # 최근 30일 데이터 조회
        start_date = (datetime.now().replace(day=1)).strftime('%Y%m%d')
        
        df = self.fetch_exchange_rate(currency, start_date, end_date)
        
        if len(df) > 0:
            latest = df.iloc[-1]
            return {
                'currency': currency,
                'rate': latest['DATA_VALUE'],
                'date': latest['datetime'].strftime('%Y-%m-%d'),
                'unit': latest['UNIT_NAME']
            }
        return None


if __name__ == "__main__":
    # 테스트 코드
    import os
    
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    print("=" * 50)
    print("한국은행 API 환율 정보 수집 테스트")
    print("=" * 50)
    
    # API 키를 환경변수에서 가져오기
    api_key = os.getenv('BOK_API_KEY')
    
    if not api_key:
        print("\n⚠️  API 키를 설정해주세요!")
        print("1. 한국은행 ECOS에서 API 키 발급: https://ecos.bok.or.kr/")
        print("2. .env 파일 생성 후 BOK_API_KEY 설정")
        print("   또는 환경변수 설정: export BOK_API_KEY='your_key_here'")
    else:
        fetcher = ExchangeRateFetcher(api_key)
        
        # 최신 환율 조회
        print("\n[최신 환율 정보]")
        for currency in ['USD', 'JPY', 'CNY']:
            try:
                latest = fetcher.get_latest_rate(currency)
                if latest:
                    print(f"{currency}: {latest['rate']:,.2f}원 ({latest['date']}) - {latest['unit']}")
            except Exception as e:
                print(f"{currency} 조회 실패: {e}")
        
        # 기간별 환율 조회 예시
        print("\n[2024년 환율 데이터 수집]")
        try:
            rates = fetcher.fetch_multiple_rates(
                currencies=['USD', 'JPY', 'CNY'],
                start_date='20240101',
                end_date='20241231'
            )
            
            print("\n수집된 데이터 요약:")
            for currency, df in rates.items():
                print(f"\n{currency}:")
                print(f"  - 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
                print(f"  - 최저: {df['DATA_VALUE'].min():,.2f}원")
                print(f"  - 최고: {df['DATA_VALUE'].max():,.2f}원")
                print(f"  - 평균: {df['DATA_VALUE'].mean():,.2f}원")
                
        except Exception as e:
            print(f"데이터 수집 실패: {e}")
