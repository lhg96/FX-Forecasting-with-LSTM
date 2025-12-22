"""
환율 정보 수집 사용 예제
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from exchange_rate_fetcher import ExchangeRateFetcher

# .env 파일에서 환경변수 로드
load_dotenv()


def example_latest_rates():
    """최신 환율 조회 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 최신 환율 조회")
    print("=" * 60)
    
    api_key = os.getenv('BOK_API_KEY')
    if not api_key:
        print("⚠️  BOK_API_KEY를 .env 파일에 설정해주세요!")
        return
    
    fetcher = ExchangeRateFetcher(api_key)
    
    currencies = {
        'USD': '미국 달러',
        'JPY': '일본 엔화',
        'CNY': '중국 위안'
    }
    
    print("\n📊 현재 환율 정보:")
    print("-" * 60)
    
    for code, name in currencies.items():
        try:
            latest = fetcher.get_latest_rate(code)
            if latest:
                print(f"{name:10s} ({code}): {latest['rate']:>10,.2f}원 ({latest['date']})")
        except Exception as e:
            print(f"{name:10s} ({code}): 조회 실패 - {e}")


def example_historical_data():
    """과거 환율 데이터 조회 및 분석 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 과거 환율 데이터 조회 및 분석")
    print("=" * 60)
    
    api_key = os.getenv('BOK_API_KEY')
    if not api_key:
        print("⚠️  BOK_API_KEY를 .env 파일에 설정해주세요!")
        return
    
    fetcher = ExchangeRateFetcher(api_key)
    
    # 최근 3개월 데이터 조회
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    print(f"\n조회 기간: {start_date.date()} ~ {end_date.date()}")
    
    try:
        # 달러 환율 조회
        df_usd = fetcher.fetch_exchange_rate('USD', start_str, end_str)
        
        print(f"\n📈 USD/KRW 환율 통계:")
        print("-" * 60)
        print(f"데이터 건수: {len(df_usd):>10,}건")
        print(f"최저 환율:   {df_usd['DATA_VALUE'].min():>10,.2f}원")
        print(f"최고 환율:   {df_usd['DATA_VALUE'].max():>10,.2f}원")
        print(f"평균 환율:   {df_usd['DATA_VALUE'].mean():>10,.2f}원")
        print(f"표준 편차:   {df_usd['DATA_VALUE'].std():>10,.2f}원")
        
        # 변동성 분석
        df_usd['change'] = df_usd['DATA_VALUE'].diff()
        print(f"\n📊 변동성 분석:")
        print("-" * 60)
        print(f"최대 상승:   {df_usd['change'].max():>10,.2f}원")
        print(f"최대 하락:   {df_usd['change'].min():>10,.2f}원")
        print(f"평균 변동:   {df_usd['change'].abs().mean():>10,.2f}원")
        
    except Exception as e:
        print(f"데이터 조회 실패: {e}")


def example_compare_currencies():
    """여러 통화 비교 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 여러 통화 환율 비교")
    print("=" * 60)
    
    api_key = os.getenv('BOK_API_KEY')
    if not api_key:
        print("⚠️  BOK_API_KEY를 .env 파일에 설정해주세요!")
        return
    
    fetcher = ExchangeRateFetcher(api_key)
    
    # 최근 6개월 데이터 조회
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    print(f"\n조회 기간: {start_date.date()} ~ {end_date.date()}")
    
    try:
        # 여러 통화 데이터 조회
        rates = fetcher.fetch_multiple_rates(
            currencies=['USD', 'JPY', 'CNY'],
            start_date=start_str,
            end_date=end_str
        )
        
        print("\n" + "=" * 60)
        print("통화별 환율 비교")
        print("=" * 60)
        print(f"{'통화':<10} {'최저':<12} {'최고':<12} {'평균':<12} {'변동폭':<12}")
        print("-" * 60)
        
        for currency, df in rates.items():
            min_rate = df['DATA_VALUE'].min()
            max_rate = df['DATA_VALUE'].max()
            avg_rate = df['DATA_VALUE'].mean()
            range_rate = max_rate - min_rate
            
            print(f"{currency:<10} {min_rate:>10,.2f}원 {max_rate:>10,.2f}원 {avg_rate:>10,.2f}원 {range_rate:>10,.2f}원")
        
        # 상관관계 분석
        print("\n📊 통화간 상관관계:")
        print("-" * 60)
        
        # 데이터 병합
        merged_df = rates['USD'][['datetime', 'DATA_VALUE']].copy()
        merged_df.columns = ['datetime', 'USD']
        
        for currency in ['JPY', 'CNY']:
            temp_df = rates[currency][['datetime', 'DATA_VALUE']].copy()
            temp_df.columns = ['datetime', currency]
            merged_df = pd.merge(merged_df, temp_df, on='datetime', how='inner')
        
        # 상관계수 계산
        correlation = merged_df[['USD', 'JPY', 'CNY']].corr()
        print(correlation)
        
    except Exception as e:
        print(f"데이터 조회 실패: {e}")


def example_export_to_csv():
    """CSV 파일로 내보내기 예제"""
    print("\n" + "=" * 60)
    print("예제 4: CSV 파일로 내보내기")
    print("=" * 60)
    
    api_key = os.getenv('BOK_API_KEY')
    if not api_key:
        print("⚠️  BOK_API_KEY를 .env 파일에 설정해주세요!")
        return
    
    fetcher = ExchangeRateFetcher(api_key)
    
    # 2024년 전체 데이터 조회
    start_str = '20240101'
    end_str = '20241231'
    
    print(f"\n조회 기간: 2024년 전체")
    
    try:
        rates = fetcher.fetch_multiple_rates(
            currencies=['USD', 'JPY', 'CNY'],
            start_date=start_str,
            end_date=end_str
        )
        
        # 각 통화별로 CSV 저장
        output_dir = '/Users/hyun/workspace/finance/data'
        os.makedirs(output_dir, exist_ok=True)
        
        for currency, df in rates.items():
            filename = f"{output_dir}/exchange_rate_{currency}_2024.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ {currency} 데이터 저장 완료: {filename}")
        
        # 통합 데이터 저장
        merged_df = rates['USD'][['datetime', 'DATA_VALUE']].copy()
        merged_df.columns = ['date', 'USD']
        
        for currency in ['JPY', 'CNY']:
            temp_df = rates[currency][['datetime', 'DATA_VALUE']].copy()
            temp_df.columns = ['date', currency]
            merged_df = pd.merge(merged_df, temp_df, on='date', how='outer')
        
        merged_df = merged_df.sort_values('date')
        merged_filename = f"{output_dir}/exchange_rates_all_2024.csv"
        merged_df.to_csv(merged_filename, index=False, encoding='utf-8-sig')
        print(f"✅ 통합 데이터 저장 완료: {merged_filename}")
        
    except Exception as e:
        print(f"데이터 처리 실패: {e}")


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("한국은행 API 환율 정보 수집 예제")
    print("=" * 60)
    
    # API 키 확인
    api_key = os.getenv('BOK_API_KEY')
    if not api_key:
        print("\n⚠️  API 키 설정이 필요합니다!")
        print("\n설정 방법:")
        print("1. 한국은행 ECOS 가입: https://ecos.bok.or.kr/")
        print("2. API 인증키 신청")
        print("3. .env 파일 생성:")
        print("   BOK_API_KEY=your_api_key_here")
        print("\n또는 환경변수 설정:")
        print("   export BOK_API_KEY='your_api_key_here'")
        return
    
    # 예제 실행
    try:
        example_latest_rates()
        example_historical_data()
        example_compare_currencies()
        example_export_to_csv()
        
        print("\n" + "=" * 60)
        print("✅ 모든 예제 실행 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
