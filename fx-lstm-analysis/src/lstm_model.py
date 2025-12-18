"""
LSTM 모델 학습 모듈
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os


class ForexLSTMModel:
    """환율 예측 LSTM 모델 클래스"""
    
    def __init__(self, input_shape, lstm_units=200, dropout_rate=0.1):
        """
        Args:
            input_shape: (timesteps, features) - 입력 데이터의 shape
            lstm_units: LSTM 레이어의 유닛 수
            dropout_rate: 드롭아웃 비율
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        # 재현성을 위한 시드 설정
        tf.random.set_seed(13)
        np.random.seed(13)
    
    def build_model(self):
        """LSTM 모델 구축"""
        print("\nLSTM 모델 구축 중...")
        
        self.model = Sequential([
            LSTM(self.lstm_units, 
                 input_shape=self.input_shape, 
                 activation='tanh'),
            Dense(1)  # 1개의 값 예측 (다음날 환율)
        ])
        
        # 모델 컴파일
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            loss='mae'
        )
        
        print("모델 구축 완료!")
        self.model.summary()
        
        return self.model
    
    def create_tf_dataset(self, x_data, y_data, batch_size=32, buffer_size=64, 
                         is_training=True):
        """TensorFlow Dataset 생성"""
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size).repeat()
        
        return dataset
    
    def train(self, x_train, y_train, x_val, y_val, 
              batch_size=32, epochs=100, patience=40,
              model_save_path='../models/best_model.ckpt'):
        """모델 학습"""
        if self.model is None:
            raise ValueError("먼저 build_model()을 호출해야 합니다.")
        
        print("\n" + "=" * 50)
        print("모델 학습 시작")
        print("=" * 50)
        
        # 학습 파라미터
        train_steps = len(x_train) // batch_size
        val_steps = len(x_val) // batch_size
        
        print(f"\n학습 파라미터:")
        print(f"- Batch size: {batch_size}")
        print(f"- Epochs: {epochs}")
        print(f"- Training steps per epoch: {train_steps}")
        print(f"- Validation steps: {val_steps}")
        print(f"- Early stopping patience: {patience}")
        
        # TensorFlow Dataset 생성
        train_data = self.create_tf_dataset(x_train, y_train, batch_size, is_training=True)
        val_data = self.create_tf_dataset(x_val, y_val, batch_size, is_training=False)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min',
                verbose=1,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                model_save_path,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
        ]
        
        # 모델 디렉토리 생성
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # 학습
        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_data,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n학습 완료!")
        
        return self.history
    
    def load_weights(self, model_path):
        """학습된 가중치 로드"""
        if self.model is None:
            raise ValueError("먼저 build_model()을 호출해야 합니다.")
        
        print(f"\n가중치 로드 중: {model_path}")
        self.model.load_weights(model_path)
        print("가중치 로드 완료!")
    
    def predict(self, x_data):
        """예측 수행"""
        if self.model is None:
            raise ValueError("먼저 build_model()을 호출해야 합니다.")
        
        predictions = self.model.predict(x_data)
        return predictions
    
    def get_training_history(self):
        """학습 히스토리 반환"""
        if self.history is None:
            raise ValueError("먼저 train()을 실행해야 합니다.")
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss']
        }
    
    def save_model(self, save_path):
        """전체 모델 저장"""
        if self.model is None:
            raise ValueError("먼저 build_model()을 호출해야 합니다.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"\n모델 저장 완료: {save_path}")
    
    def load_model(self, model_path):
        """전체 모델 로드"""
        print(f"\n모델 로드 중: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("모델 로드 완료!")
        return self.model


class ModelConfig:
    """모델 설정 클래스"""
    
    # 데이터 파라미터
    BATCH_SIZE = 32
    TRAIN_SPLIT_RATIO = 0.9
    BUFFER_SIZE = 64
    
    # 시계열 파라미터
    PAST_HISTORY = 10  # 과거 며칠 데이터를 사용할지
    FUTURE_TARGET = 1  # 몇 일 후를 예측할지
    STEP = 1
    
    # 모델 파라미터
    LSTM_UNITS = 200
    DROPOUT_RATE = 0.1
    
    # 학습 파라미터
    EPOCHS = 100
    PATIENCE = 40
    
    # 경로
    MODEL_SAVE_PATH = '../models/best_model.weights.h5'
    FULL_MODEL_PATH = '../models/forex_lstm_model.h5'
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("\n" + "=" * 50)
        print("모델 설정")
        print("=" * 50)
        print(f"배치 크기: {cls.BATCH_SIZE}")
        print(f"학습 데이터 비율: {cls.TRAIN_SPLIT_RATIO}")
        print(f"과거 데이터 길이: {cls.PAST_HISTORY}일")
        print(f"예측 목표: {cls.FUTURE_TARGET}일 후")
        print(f"LSTM 유닛: {cls.LSTM_UNITS}")
        print(f"드롭아웃 비율: {cls.DROPOUT_RATE}")
        print(f"에포크: {cls.EPOCHS}")
        print(f"조기종료 인내: {cls.PATIENCE}")
        print("=" * 50)


if __name__ == "__main__":
    # 테스트 코드
    print("LSTM 모델 모듈 테스트")
    
    # 설정 출력
    ModelConfig.print_config()
    
    # 더미 데이터로 모델 테스트
    dummy_x = np.random.randn(1000, 10, 3)  # (samples, timesteps, features)
    dummy_y = np.random.randn(1000, 1)
    
    # 모델 생성
    model = ForexLSTMModel(input_shape=(10, 3), lstm_units=200)
    model.build_model()
    
    print("\n모델 테스트 완료!")
