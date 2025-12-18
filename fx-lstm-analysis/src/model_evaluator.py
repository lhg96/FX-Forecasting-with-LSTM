"""
모델 평가 및 예측 모듈
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, preprocessor):
        """
        Args:
            preprocessor: ForexDataPreprocessor 인스턴스
        """
        self.preprocessor = preprocessor
    
    def evaluate(self, model, x_val, y_val, verbose=True):
        """모델 평가"""
        # 예측
        predictions = model.predict(x_val)
        
        # 역정규화 (원래 스케일로 복원)
        predictions_denorm = self.preprocessor.denormalize(predictions, column_idx=0)
        y_val_denorm = self.preprocessor.denormalize(y_val, column_idx=0)
        
        # 평가 지표 계산
        mae = mean_absolute_error(y_val_denorm, predictions_denorm)
        r2 = r2_score(y_val_denorm, predictions_denorm)
        
        # 오차 계산
        errors = np.abs(y_val_denorm - predictions_denorm)
        max_error = np.max(errors)
        min_error = np.min(errors)
        mean_error = np.mean(errors)
        
        if verbose:
            print("\n" + "=" * 50)
            print("모델 평가 결과")
            print("=" * 50)
            print(f"MAE (평균 절대 오차): {mae:.5f}원")
            print(f"R² Score: {r2:.5f}")
            print(f"최대 오차: {max_error:.5f}원")
            print(f"최소 오차: {min_error:.5f}원")
            print(f"평균 오차: {mean_error:.5f}원")
            print("=" * 50)
        
        return {
            'mae': mae,
            'r2': r2,
            'max_error': max_error,
            'min_error': min_error,
            'mean_error': mean_error,
            'predictions': predictions_denorm,
            'actual': y_val_denorm,
            'errors': errors
        }
    
    def predict_next_day(self, model, data, past_history=10):
        """다음 날 환율 예측"""
        # 최근 past_history일 데이터 추출
        recent_data = data[-past_history:].values
        
        # 정규화
        recent_mean = recent_data.mean(axis=0)
        recent_std = recent_data.std(axis=0)
        recent_normalized = (recent_data - recent_mean) / recent_std
        
        # reshape for prediction
        pred_input = recent_normalized.reshape(-1, past_history, recent_data.shape[-1])
        
        # 예측
        prediction_normalized = model.predict(pred_input, verbose=0)
        
        # 역정규화
        prediction = self.preprocessor.denormalize(prediction_normalized, column_idx=0)
        
        # 다음 날짜
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            next_date = last_date + pd.Timedelta(days=1)
        else:
            next_date = datetime.strptime(str(last_date), '%Y-%m-%d') + timedelta(days=1)
        
        return {
            'date': next_date,
            'predicted_value': float(prediction[0][0]),
            'last_actual_value': float(data.iloc[-1, 0])
        }
    
    def create_prediction_dataframe(self, model, x_val, y_val, data_index):
        """예측 결과를 DataFrame으로 생성"""
        # 예측
        predictions = model.predict(x_val)
        
        # 역정규화
        predictions_denorm = self.preprocessor.denormalize(predictions, column_idx=0)
        y_val_denorm = self.preprocessor.denormalize(y_val, column_idx=0)
        
        # DataFrame 생성
        pred_df = pd.DataFrame({
            'actual': y_val_denorm.flatten(),
            'predicted': predictions_denorm.flatten(),
            'error': (y_val_denorm - predictions_denorm).flatten(),
            'abs_error': np.abs(y_val_denorm - predictions_denorm).flatten()
        }, index=data_index[-len(y_val):])
        
        return pred_df
    
    def analyze_prediction_trends(self, pred_df, window=5):
        """예측 트렌드 분석"""
        print("\n" + "=" * 50)
        print("예측 트렌드 분석")
        print("=" * 50)
        
        # 이동평균 계산
        pred_df['actual_ma'] = pred_df['actual'].rolling(window=window).mean()
        pred_df['predicted_ma'] = pred_df['predicted'].rolling(window=window).mean()
        
        # 최근 데이터
        recent = pred_df.tail(10)
        
        print(f"\n최근 {len(recent)}일 예측 결과:")
        print(recent[['actual', 'predicted', 'error']])
        
        # 정확도 분석
        accuracy_within_50 = (pred_df['abs_error'] < 50).sum() / len(pred_df) * 100
        accuracy_within_100 = (pred_df['abs_error'] < 100).sum() / len(pred_df) * 100
        
        print(f"\n예측 정확도:")
        print(f"- 오차 50원 이내: {accuracy_within_50:.2f}%")
        print(f"- 오차 100원 이내: {accuracy_within_100:.2f}%")
        
        return pred_df


class MovingAverageAnalyzer:
    """이동평균 기반 추세 분석 클래스"""
    
    def __init__(self, data, prediction, window=2):
        """
        Args:
            data: 실제 데이터 (DataFrame)
            prediction: 예측 값
            window: 이동평균 윈도우 크기
        """
        self.window = window
        self.ma_df = pd.DataFrame(index=data.index)
        self.ma_df['real'] = data.iloc[:, 0]  # USD/KRW
        
        # 예측값 추가 (다음날)
        last_date = data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # 새 행 추가
        new_row = pd.DataFrame({'real': [prediction]}, index=[next_date])
        self.ma_df = pd.concat([self.ma_df, new_row])
        
        # 이동평균 계산
        self.ma_df['ma2'] = self.ma_df['real'].rolling(window=2).mean()
        self.ma_df['ma3'] = self.ma_df['real'].rolling(window=3).mean()
        self.ma_df['ma5'] = self.ma_df['real'].rolling(window=5).mean()
        
        # 차이 계산
        self.ma_df['real-ma2'] = self.ma_df['real'] - self.ma_df['ma2']
    
    def predict_trend(self):
        """이동평균 기반 추세 예측"""
        print("\n" + "=" * 50)
        print("이동평균 기반 추세 분석")
        print("=" * 50)
        
        # 최근 데이터 확인
        print("\n최근 5일 데이터:")
        print(self.ma_df[['real', 'ma2', 'real-ma2']].tail(5))
        
        # 추세 판단 로직
        if pd.isna(self.ma_df.iloc[-2]['real-ma2']):
            return "데이터 부족으로 추세 판단 불가"
        
        # 실제값과 MA2 비교
        prev_diff = self.ma_df.iloc[-2]['real-ma2']
        curr_diff = self.ma_df.iloc[-1]['real-ma2']
        
        if pd.isna(curr_diff):
            return "현재 데이터로 추세 판단 불가"
        
        # 추세 판단
        if prev_diff > 0:  # 이전에 실제값이 MA2보다 컸음
            if prev_diff * curr_diff < 0:  # 부호가 바뀜
                trend = "하락 가능성 높음"
            else:
                trend = "추세 불확실"
        elif prev_diff < 0:  # 이전에 실제값이 MA2보다 작았음
            if prev_diff * curr_diff < 0:  # 부호가 바뀜
                trend = "상승 가능성 높음"
            else:
                trend = "추세 불확실"
        else:
            trend = "추세 불확실"
        
        print(f"\n예측 추세: {trend}")
        
        return trend
    
    def get_ma_dataframe(self):
        """이동평균 DataFrame 반환"""
        return self.ma_df


if __name__ == "__main__":
    print("모델 평가 및 예측 모듈")
    print("이 모듈은 다른 스크립트에서 import하여 사용합니다.")
