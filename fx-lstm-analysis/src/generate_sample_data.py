"""
샘플 환율 데이터 생성 스크립트
실제 크롤링이 작동하지 않을 때 테스트용 데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_forex_data(start_date='1998-03-23', end_date='2024-12-18'):
    """샘플 환율 데이터 생성"""
    
    print("샘플 환율 데이터 생성 중...")
    
    # 날짜 범위 생성
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # 기본 환율 설정 (실제와 유사한 값)
    base_usd_krw = 1200
    base_dollar_index = 100
    base_crb = 250
    
    # 랜덤 워크로 환율 생성 (트렌드 포함)
    n_days = len(dates)
    
    # USD/KRW: 장기 상승 트렌드 + 랜덤 변동
    trend = np.linspace(0, 200, n_days)  # 장기 상승 트렌드
    noise = np.random.randn(n_days).cumsum() * 10
    usd_krw = base_usd_krw + trend + noise
    usd_krw = np.clip(usd_krw, 800, 1600)  # 현실적인 범위로 제한
    
    # 달러 인덱스: 주기적 변동
    cycle = 20 * np.sin(np.linspace(0, 8*np.pi, n_days))
    noise_di = np.random.randn(n_days).cumsum() * 2
    dollar_index = base_dollar_index + cycle + noise_di
    dollar_index = np.clip(dollar_index, 80, 120)
    
    # CRB: 변동성 높은 패턴
    noise_crb = np.random.randn(n_days).cumsum() * 5
    trend_crb = np.linspace(0, 50, n_days)
    crb = base_crb + trend_crb + noise_crb
    crb = np.clip(crb, 150, 400)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'USD/KRW': usd_krw,
        '달러지수': dollar_index,
        'CRB': crb
    })
    
    # 주말 제거 (실제 거래일만)
    df = df[df['date'].dt.dayofweek < 5].reset_index(drop=True)
    
    print(f"샘플 데이터 생성 완료: {len(df)}개 행")
    print(f"기간: {df['date'].min()} ~ {df['date'].max()}")
    
    return df


if __name__ == "__main__":
    # 샘플 데이터 생성
    sample_data = generate_sample_forex_data()
    
    # 저장
    output_path = '../data/forex_data.xlsx'
    sample_data.to_excel(output_path, index=False)
    print(f"\n샘플 데이터 저장 완료: {output_path}")
    
    # 미리보기
    print("\n데이터 미리보기:")
    print(sample_data.head(10))
    print("\n최근 데이터:")
    print(sample_data.tail(10))
    
    # 통계
    print("\n데이터 통계:")
    print(sample_data.describe())
