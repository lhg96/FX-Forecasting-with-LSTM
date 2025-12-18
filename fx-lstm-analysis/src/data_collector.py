"""
환율 데이터 수집 모듈
- USD/KRW 환율
- 달러 인덱스
- CRB 지수
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
from datetime import datetime
import time
from functools import reduce


class ForexDataCollector:
    """환율 및 관련 지표 데이터 수집 클래스"""
    
    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_usd_krw(self):
        """USD/KRW 환율 데이터 수집"""
        print("USD/KRW 데이터 수집 중...")
        try:
            url = Request(
                'https://kr.investing.com/currencies/usd-krw-historical-data',
                headers=self.headers
            )
            
            res = urlopen(url)
            bs = BeautifulSoup(res, 'html.parser')
            
            table = bs.select_one('table')
            
            if not table:
                print("테이블을 찾을 수 없습니다.")
                return None
            
            rows = table.select('tbody > tr')
            
            date_list = []
            price_list = []
            
            for row in rows:
                try:
                    date_elem = row.find('time')
                    if date_elem:
                        date = date_elem.text.replace(' ', '')
                        date_list.append(date)
                    
                    price_elem = row.find('td', attrs={'dir': 'ltr'})
                    if price_elem:
                        price = price_elem.text.replace(',', '')
                        price_list.append(price)
                except Exception as e:
                    continue
            
            data = {'date': date_list, 'price': price_list}
            df = pd.DataFrame(data)
            df = df.sort_values(by='date')
            
            print(f"USD/KRW 데이터 수집 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            print(f"USD/KRW 데이터 수집 실패: {str(e)}")
            return None
    
    def collect_dollar_index(self):
        """달러 인덱스 데이터 수집"""
        print("달러 인덱스 데이터 수집 중...")
        try:
            url = Request(
                'https://kr.investing.com/currencies/us-dollar-index-historical-data',
                headers=self.headers
            )
            
            res = urlopen(url)
            bs = BeautifulSoup(res, 'html.parser')
            
            table = bs.find('table')
            
            if not table:
                print("테이블을 찾을 수 없습니다.")
                return None
            
            data = []
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 2:
                    cols = [col.text.strip() for col in cols[:2]]
                    data.append(cols)
            
            columns = ['date', 'price']
            df = pd.DataFrame(data[1:], columns=columns)
            
            df['date'] = pd.to_datetime(df['date'], format='%Y년 %m월 %d일', errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values(by='date')
            df['date'] = df['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
            
            print(f"달러 인덱스 데이터 수집 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            print(f"달러 인덱스 데이터 수집 실패: {str(e)}")
            return None
    
    def collect_crb_index(self):
        """CRB 지수 데이터 수집"""
        print("CRB 지수 데이터 수집 중...")
        try:
            url = Request(
                'https://kr.investing.com/indices/thomson-reuters---jefferies-crb-historical-data',
                headers=self.headers
            )
            
            res = urlopen(url)
            bs = BeautifulSoup(res, 'html.parser')
            
            table = bs.select_one('table')
            
            if not table:
                print("테이블을 찾을 수 없습니다.")
                return None
            
            rows = table.select('tbody > tr')
            
            date_list = []
            price_list = []
            
            for row in rows:
                try:
                    date_elem = row.find('time')
                    if date_elem:
                        date = date_elem.text.replace(' ', '')
                        date_list.append(date)
                    
                    price_elem = row.find('td', attrs={'dir': 'ltr'})
                    if price_elem:
                        price = price_elem.text
                        price_list.append(price)
                except Exception as e:
                    continue
            
            data = {'date': date_list, 'price': price_list}
            df = pd.DataFrame(data)
            df = df.sort_values(by='date')
            
            print(f"CRB 지수 데이터 수집 완료: {len(df)}개 행")
            return df
            
        except Exception as e:
            print(f"CRB 지수 데이터 수집 실패: {str(e)}")
            return None
    
    def merge_data(self, df_list, column_names):
        """데이터프레임 병합"""
        print("\n데이터 병합 중...")
        
        # 유효한 데이터프레임만 필터링
        valid_dfs = [df for df in df_list if df is not None and not df.empty]
        
        if len(valid_dfs) == 0:
            print("병합할 유효한 데이터가 없습니다.")
            return None
        
        # 병합
        merged = reduce(lambda x, y: pd.merge(x, y, on='date', how='outer'), valid_dfs)
        
        # 컬럼명 변경 (병합 후 실제 컬럼 개수에 맞춰서)
        actual_columns = len(merged.columns)
        expected_columns = 1 + len(column_names)  # date + 데이터 컬럼들
        
        if actual_columns != expected_columns:
            # price 컬럼이 중복되었을 경우 처리
            if actual_columns > expected_columns:
                # 중복 컬럼 제거 (첫 번째 price만 유지)
                price_cols = [col for col in merged.columns if col == 'price' or col.startswith('price')]
                if len(price_cols) > len(column_names):
                    # 각 price 컬럼에 임시 이름 부여
                    new_cols = ['date']
                    price_idx = 0
                    for col in merged.columns[1:]:
                        if col == 'price' or col.startswith('price'):
                            if price_idx < len(column_names):
                                new_cols.append(column_names[price_idx])
                                price_idx += 1
                        else:
                            new_cols.append(col)
                    merged.columns = new_cols
                else:
                    merged.columns = ['date'] + column_names
            else:
                merged.columns = ['date'] + column_names[:actual_columns-1]
        else:
            merged.columns = ['date'] + column_names
        
        # 데이터 타입 변환
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        
        for col in column_names:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
        
        # 날짜로 정렬
        merged = merged.sort_values('date').reset_index(drop=True)
        
        print(f"병합 완료: {len(merged)}개 행, {len(merged.columns)}개 컬럼")
        return merged
    
    def collect_all(self):
        """모든 데이터 수집 및 병합"""
        print("=" * 50)
        print("환율 데이터 수집 시작")
        print("=" * 50)
        
        # 각 데이터 수집
        df_usd_krw = self.collect_usd_krw()
        time.sleep(1)  # 요청 간격 조절
        
        df_dollar_index = self.collect_dollar_index()
        time.sleep(1)
        
        df_crb = self.collect_crb_index()
        
        # 병합
        merged_data = self.merge_data(
            [df_usd_krw, df_dollar_index, df_crb],
            ['USD/KRW', '달러지수', 'CRB']
        )
        
        print("=" * 50)
        print("데이터 수집 완료")
        print("=" * 50)
        
        return merged_data
    
    def save_data(self, df, filepath):
        """데이터 저장"""
        if df is not None and not df.empty:
            df.to_excel(filepath, index=False)
            print(f"\n데이터 저장 완료: {filepath}")
            return True
        else:
            print("\n저장할 데이터가 없습니다.")
            return False
    
    def load_and_update(self, existing_file, new_data):
        """기존 데이터 로드 후 새 데이터 병합"""
        try:
            existing_data = pd.read_excel(existing_file)
            print(f"기존 데이터 로드 완료: {len(existing_data)}개 행")
            
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data.drop_duplicates(subset=['date'], keep='first')
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            
            print(f"업데이트된 데이터: {len(combined_data)}개 행")
            return combined_data
            
        except FileNotFoundError:
            print("기존 파일이 없습니다. 새로운 데이터를 저장합니다.")
            return new_data


if __name__ == "__main__":
    collector = ForexDataCollector()
    data = collector.collect_all()
    
    if data is not None:
        # 데이터 미리보기
        print("\n수집된 데이터 미리보기:")
        print(data.head(10))
        print("\n데이터 정보:")
        print(data.info())
        print("\n결측치:")
        print(data.isnull().sum())
        
        # 저장
        output_path = '../data/forex_data.xlsx'
        collector.save_data(data, output_path)
