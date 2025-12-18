"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np


class ForexDataPreprocessor:
    """환율 데이터 전처리 클래스"""
    
    def __init__(self, data_path=None, start_date='1998-03-23'):
        self.data_path = data_path
        self.start_date = start_date
        self.data = None
        self.data_mean = None
        self.data_std = None
        
    def load_data(self):
        """데이터 로드"""
        if self.data_path is None:
            raise ValueError("데이터 경로가 지정되지 않았습니다.")
        
        print(f"데이터 로드 중: {self.data_path}")
        self.data = pd.read_excel(self.data_path)
        self.data = self.data.set_index('date')
        
        # 결측치 전방 채우기
        self.data = self.data.fillna(method='ffill')
        
        # 시작 날짜 이후 데이터만 사용
        if self.start_date:
            self.data = self.data[self.start_date:]
        
        print(f"데이터 로드 완료: {len(self.data)}개 행")
        print("\n데이터 정보:")
        print(self.data.info())
        
        return self.data
    
    def check_data_quality(self):
        """데이터 품질 확인"""
        if self.data is None:
            raise ValueError("먼저 데이터를 로드해야 합니다.")
        
        print("\n" + "=" * 50)
        print("데이터 품질 확인")
        print("=" * 50)
        
        print(f"\n총 데이터 개수: {len(self.data)}")
        print("\n결측치 개수:")
        print(self.data.isnull().sum())
        
        print("\n기초 통계량:")
        print(self.data.describe())
        
        print("\n데이터 미리보기:")
        print(self.data.head())
        
        return self.data.isnull().sum().sum() == 0
    
    def normalize_data(self):
        """데이터 표준화 (정규화)"""
        if self.data is None:
            raise ValueError("먼저 데이터를 로드해야 합니다.")
        
        print("\n데이터 표준화 중...")
        
        dataset = self.data.values
        self.data_mean = dataset.mean(axis=0)
        self.data_std = dataset.std(axis=0)
        
        normalized_data = (dataset - self.data_mean) / self.data_std
        
        print(f"표준화 완료: Shape = {normalized_data.shape}")
        print(f"Mean: {self.data_mean}")
        print(f"Std: {self.data_std}")
        
        return normalized_data
    
    def split_data(self, dataset, target_column_idx=0, start_index=0, 
                   end_index=None, history_size=10, target_size=1, step=1):
        """학습 및 검증 데이터 분리"""
        data = []
        labels = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size + 1
        
        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])
            
            # target_column_idx 컬럼의 값을 레이블로 사용
            labels.append(dataset[i:i + target_size, target_column_idx])
        
        return np.array(data), np.array(labels)
    
    def create_train_val_split(self, normalized_data, train_split_ratio=0.9,
                               past_history=10, future_target=1, step=1):
        """학습/검증 데이터 생성"""
        train_split = int(len(self.data) * train_split_ratio)
        
        print(f"\n학습/검증 데이터 분리:")
        print(f"학습 데이터: 0 ~ {train_split}")
        print(f"검증 데이터: {train_split} ~ {len(self.data)}")
        
        # 학습 데이터
        x_train, y_train = self.split_data(
            normalized_data, 
            target_column_idx=0,  # USD/KRW 열
            start_index=0,
            end_index=train_split,
            history_size=past_history,
            target_size=future_target,
            step=step
        )
        
        # 검증 데이터
        x_val, y_val = self.split_data(
            normalized_data,
            target_column_idx=0,
            start_index=train_split,
            end_index=None,
            history_size=past_history,
            target_size=future_target,
            step=step
        )
        
        print(f"\n학습 데이터 shape: X={x_train.shape}, Y={y_train.shape}")
        print(f"검증 데이터 shape: X={x_val.shape}, Y={y_val.shape}")
        
        return x_train, y_train, x_val, y_val, train_split
    
    def denormalize(self, normalized_values, column_idx=0):
        """표준화된 값을 원래 스케일로 복원"""
        if self.data_mean is None or self.data_std is None:
            raise ValueError("먼저 normalize_data()를 실행해야 합니다.")
        
        return normalized_values * self.data_std[column_idx] + self.data_mean[column_idx]
    
    def prepare_prediction_data(self, normalized_data, past_history=10):
        """다음 날 예측을 위한 데이터 준비"""
        pred_data = normalized_data[-past_history:]
        pred_data = pred_data.reshape(-1, past_history, normalized_data.shape[-1])
        
        return pred_data


if __name__ == "__main__":
    # 테스트 코드
    preprocessor = ForexDataPreprocessor(
        data_path='../data/forex_data.xlsx',
        start_date='1998-03-23'
    )
    
    # 데이터 로드
    data = preprocessor.load_data()
    
    # 데이터 품질 확인
    is_clean = preprocessor.check_data_quality()
    
    # 데이터 표준화
    normalized = preprocessor.normalize_data()
    
    # 학습/검증 데이터 생성
    x_train, y_train, x_val, y_val, train_split = preprocessor.create_train_val_split(
        normalized,
        train_split_ratio=0.9,
        past_history=10,
        future_target=1,
        step=1
    )
    
    print("\n전처리 완료!")
