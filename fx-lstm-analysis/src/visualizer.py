"""
시각화 모듈
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# 한글 폰트 설정
matplotlib.rc('font', family='AppleGothic')  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


class ForexVisualizer:
    """환율 데이터 시각화 클래스"""
    
    def __init__(self, figsize=(14, 7)):
        self.figsize = figsize
    
    def plot_training_history(self, history, save_path=None):
        """학습 히스토리 시각화"""
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        plt.figure(figsize=self.figsize)
        plt.plot(epochs, loss, 'b-', label='학습 손실', linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label='검증 손실', linewidth=2)
        plt.title('모델 학습 손실 (Loss)', fontsize=16, fontweight='bold')
        plt.xlabel('에포크 (Epoch)', fontsize=12)
        plt.ylabel('손실 (MAE)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"학습 히스토리 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, actual, predicted, index=None, 
                        n_samples=None, save_path=None):
        """실제값 vs 예측값 시각화"""
        if n_samples:
            actual = actual[-n_samples:]
            predicted = predicted[-n_samples:]
            if index is not None:
                index = index[-n_samples:]
        
        plt.figure(figsize=self.figsize)
        
        if index is not None:
            plt.plot(index, actual, 'b-', label='실제 환율', linewidth=2, alpha=0.7)
            plt.plot(index, predicted, 'r--', label='예측 환율', linewidth=2, alpha=0.7)
        else:
            plt.plot(actual, 'b-', label='실제 환율', linewidth=2, alpha=0.7)
            plt.plot(predicted, 'r--', label='예측 환율', linewidth=2, alpha=0.7)
        
        plt.title('USD/KRW 환율 예측 결과', fontsize=16, fontweight='bold')
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('환율 (원)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"예측 결과 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_errors(self, errors, save_path=None):
        """오차 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]//1.5))
        
        # 오차 시계열
        axes[0].plot(errors, 'o-', color='red', alpha=0.5, markersize=4)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0].set_title('예측 오차 시계열', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('샘플 인덱스', fontsize=11)
        axes[0].set_ylabel('절대 오차 (원)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 오차 히스토그램
        axes[1].hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=np.mean(errors), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {np.mean(errors):.2f}')
        axes[1].set_title('오차 분포', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('절대 오차 (원)', fontsize=11)
        axes[1].set_ylabel('빈도', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"오차 분포 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_time_series_with_history(self, history_data, true_value, 
                                     predicted_value, save_path=None):
        """히스토리와 예측 시각화"""
        plt.figure(figsize=(12, 6))
        
        num_in = list(range(-len(history_data), 0))
        
        plt.plot(num_in, history_data[:, 0], 'b-', label='과거 환율', linewidth=2)
        plt.plot([0], [true_value[0]], 'go', label='실제 환율', markersize=10)
        
        if predicted_value is not None:
            plt.plot([0], [predicted_value[0]], 'ro', label='예측 환율', markersize=10)
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('환율 예측 (과거 데이터 기반)', fontsize=14, fontweight='bold')
        plt.xlabel('시간 (일)', fontsize=11)
        plt.ylabel('환율 (원)', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_full_data_with_predictions(self, data, predictions_df, 
                                       train_split_idx=None, save_path=None):
        """전체 데이터와 예측 결과 시각화"""
        plt.figure(figsize=(16, 8))
        
        # 전체 실제 데이터
        plt.plot(data.index, data.iloc[:, 0], 'b-', label='실제 환율', 
                linewidth=1.5, alpha=0.7)
        
        # 예측 데이터
        plt.plot(predictions_df.index, predictions_df['predicted'], 'r:', 
                label='예측 환율', linewidth=2)
        
        # 학습/검증 구분선
        if train_split_idx is not None:
            split_date = data.index[train_split_idx]
            plt.axvline(x=split_date, color='green', linestyle='--', 
                       linewidth=2, label='학습/검증 구분', alpha=0.7)
        
        plt.title('USD/KRW 환율: 전체 데이터 및 예측', fontsize=16, fontweight='bold')
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('환율 (원)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"전체 데이터 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_moving_averages(self, ma_df, n_days=5, save_path=None):
        """이동평균 시각화"""
        plt.figure(figsize=(12, 6))
        
        recent = ma_df.tail(n_days)
        
        plt.plot(recent.index, recent['real'], 'b-o', label='실제 환율', 
                linewidth=2, markersize=8)
        plt.plot(recent.index, recent['ma2'], 'r--s', label='이동평균(2일)', 
                linewidth=2, markersize=6)
        plt.plot(recent.index, recent['ma3'], 'g--^', label='이동평균(3일)', 
                linewidth=2, markersize=6, alpha=0.7)
        plt.plot(recent.index, recent['ma5'], 'y--d', label='이동평균(5일)', 
                linewidth=2, markersize=6, alpha=0.7)
        
        plt.title(f'최근 {n_days}일 환율 및 이동평균', fontsize=14, fontweight='bold')
        plt.xlabel('날짜', fontsize=11)
        plt.ylabel('환율 (원)', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"이동평균 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_recent_predictions(self, predictions_df, n_days=10, save_path=None):
        """최근 예측 결과 상세 시각화"""
        recent = predictions_df.tail(n_days)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 실제 vs 예측
        axes[0].plot(recent.index, recent['actual'], 'b-o', 
                    label='실제 환율', linewidth=2, markersize=8)
        axes[0].plot(recent.index, recent['predicted'], 'r--s', 
                    label='예측 환율', linewidth=2, markersize=8)
        axes[0].set_title(f'최근 {n_days}일 예측 비교', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('날짜', fontsize=11)
        axes[0].set_ylabel('환율 (원)', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 오차
        axes[1].bar(range(len(recent)), recent['error'], color='red', alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_title('예측 오차', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('날짜 인덱스', fontsize=11)
        axes[1].set_ylabel('오차 (원)', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"최근 예측 그래프 저장: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("시각화 모듈")
    print("이 모듈은 다른 스크립트에서 import하여 사용합니다.")
