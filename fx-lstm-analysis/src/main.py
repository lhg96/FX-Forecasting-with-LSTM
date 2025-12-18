"""
메인 실행 스크립트 - 환율 예측 LSTM 모델
전체 파이프라인 실행
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from data_collector import ForexDataCollector
from data_preprocessor import ForexDataPreprocessor
from lstm_model import ForexLSTMModel, ModelConfig
from model_evaluator import ModelEvaluator, MovingAverageAnalyzer
from visualizer import ForexVisualizer


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 70)
    print("환율 예측 LSTM 모델 - 전체 파이프라인")
    print("=" * 70)
    
    # 경로 설정
    data_dir = '../data'
    models_dir = '../models'
    output_dir = '../output'
    
    # 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(data_dir, 'forex_data.xlsx')
    
    # ========================================
    # 1. 데이터 수집
    # ========================================
    print("\n[1/7] 데이터 수집")
    print("-" * 70)
    
    # 자동 실행 모드: 샘플 데이터 사용
    import sys
    if not sys.stdin.isatty():
        collect_new_data = 'n'
        print("n (자동 모드: 기존 데이터 사용)")
    else:
        try:
            collect_new_data = input("새로운 데이터를 수집하시겠습니까? (y/n, 권장: n): ").lower()
        except EOFError:
            collect_new_data = 'n'
            print("n")
    
    if collect_new_data == 'y':
        collector = ForexDataCollector()
        new_data = collector.collect_all()
        
        if new_data is not None:
            # 기존 데이터가 있으면 병합
            if os.path.exists(data_path):
                updated_data = collector.load_and_update(data_path, new_data)
                collector.save_data(updated_data, data_path)
            else:
                collector.save_data(new_data, data_path)
    
    # 데이터 파일 확인
    if not os.path.exists(data_path):
        print(f"\n오류: 데이터 파일이 없습니다: {data_path}")
        print("먼저 데이터를 수집해주세요.")
        return
    
    # ========================================
    # 2. 데이터 전처리
    # ========================================
    print("\n[2/7] 데이터 전처리")
    print("-" * 70)
    
    preprocessor = ForexDataPreprocessor(
        data_path=data_path,
        start_date='1998-03-23'
    )
    
    # 데이터 로드
    data = preprocessor.load_data()
    
    # 데이터 품질 확인
    preprocessor.check_data_quality()
    
    # 데이터 정규화
    normalized_data = preprocessor.normalize_data()
    
    # 학습/검증 데이터 분리
    x_train, y_train, x_val, y_val, train_split = preprocessor.create_train_val_split(
        normalized_data,
        train_split_ratio=ModelConfig.TRAIN_SPLIT_RATIO,
        past_history=ModelConfig.PAST_HISTORY,
        future_target=ModelConfig.FUTURE_TARGET,
        step=ModelConfig.STEP
    )
    
    # ========================================
    # 3. 모델 구축
    # ========================================
    print("\n[3/7] 모델 구축")
    print("-" * 70)
    
    ModelConfig.print_config()
    
    lstm_model = ForexLSTMModel(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        lstm_units=ModelConfig.LSTM_UNITS,
        dropout_rate=ModelConfig.DROPOUT_RATE
    )
    
    model = lstm_model.build_model()
    
    # ========================================
    # 4. 모델 학습
    # ========================================
    print("\n[4/7] 모델 학습")
    print("-" * 70)
    
    # 자동 실행 모드
    if not sys.stdin.isatty():
        train_model = 'y'
        print("y (자동 모드: 모델 학습 시작)")
    else:
        try:
            train_model = input("모델을 학습하시겠습니까? (y/n): ").lower()
        except EOFError:
            train_model = 'y'
            print("y")
    
    if train_model == 'y':
        history = lstm_model.train(
            x_train, y_train, x_val, y_val,
            batch_size=ModelConfig.BATCH_SIZE,
            epochs=ModelConfig.EPOCHS,
            patience=ModelConfig.PATIENCE,
            model_save_path=ModelConfig.MODEL_SAVE_PATH
        )
        
        # 학습 히스토리 시각화
        visualizer = ForexVisualizer()
        history_dict = lstm_model.get_training_history()
        visualizer.plot_training_history(
            history_dict,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    else:
        # 기존 모델 로드
        if os.path.exists(ModelConfig.MODEL_SAVE_PATH + '.index'):
            lstm_model.load_weights(ModelConfig.MODEL_SAVE_PATH)
        else:
            print("\n오류: 학습된 모델이 없습니다.")
            print("먼저 모델을 학습해주세요.")
            return
    
    # ========================================
    # 5. 모델 평가
    # ========================================
    print("\n[5/7] 모델 평가")
    print("-" * 70)
    
    evaluator = ModelEvaluator(preprocessor)
    eval_results = evaluator.evaluate(model, x_val, y_val)
    
    # 예측 결과 DataFrame 생성
    pred_df = evaluator.create_prediction_dataframe(
        model, x_val, y_val, data.index
    )
    
    # 저장
    pred_df.to_excel(os.path.join(output_dir, 'predictions.xlsx'))
    print(f"\n예측 결과 저장: {os.path.join(output_dir, 'predictions.xlsx')}")
    
    # 추세 분석
    pred_df = evaluator.analyze_prediction_trends(pred_df, window=5)
    
    # ========================================
    # 6. 시각화
    # ========================================
    print("\n[6/7] 결과 시각화")
    print("-" * 70)
    
    visualizer = ForexVisualizer()
    
    # 예측 결과 시각화
    visualizer.plot_predictions(
        eval_results['actual'],
        eval_results['predictions'],
        index=data.index[-len(eval_results['actual']):],
        save_path=os.path.join(output_dir, 'predictions_full.png')
    )
    
    # 최근 예측 결과
    visualizer.plot_recent_predictions(
        pred_df,
        n_days=20,
        save_path=os.path.join(output_dir, 'recent_predictions.png')
    )
    
    # 오차 분석
    visualizer.plot_errors(
        eval_results['errors'],
        save_path=os.path.join(output_dir, 'error_analysis.png')
    )
    
    # 전체 데이터와 예측
    visualizer.plot_full_data_with_predictions(
        data,
        pred_df,
        train_split_idx=train_split,
        save_path=os.path.join(output_dir, 'full_data_predictions.png')
    )
    
    # ========================================
    # 7. 다음 날 예측
    # ========================================
    print("\n[7/7] 다음 날 환율 예측")
    print("-" * 70)
    
    next_day_pred = evaluator.predict_next_day(
        model, data, past_history=ModelConfig.PAST_HISTORY
    )
    
    print(f"\n예측 날짜: {next_day_pred['date']}")
    print(f"예측 환율: {next_day_pred['predicted_value']:.2f}원")
    print(f"현재 환율: {next_day_pred['last_actual_value']:.2f}원")
    
    change = next_day_pred['predicted_value'] - next_day_pred['last_actual_value']
    change_pct = (change / next_day_pred['last_actual_value']) * 100
    
    print(f"예상 변동: {change:+.2f}원 ({change_pct:+.2f}%)")
    
    # 이동평균 기반 추세 분석
    ma_analyzer = MovingAverageAnalyzer(
        data,
        next_day_pred['predicted_value'],
        window=2
    )
    
    trend = ma_analyzer.predict_trend()
    ma_df = ma_analyzer.get_ma_dataframe()
    
    # 이동평균 시각화
    visualizer.plot_moving_averages(
        ma_df,
        n_days=10,
        save_path=os.path.join(output_dir, 'moving_averages.png')
    )
    
    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 70)
    print("모든 과정 완료!")
    print("=" * 70)
    print(f"\n결과 파일 위치: {output_dir}")
    print("- predictions.xlsx: 예측 결과 데이터")
    print("- training_history.png: 학습 손실 그래프")
    print("- predictions_full.png: 전체 예측 결과")
    print("- recent_predictions.png: 최근 예측 상세")
    print("- error_analysis.png: 오차 분석")
    print("- full_data_predictions.png: 전체 데이터 시각화")
    print("- moving_averages.png: 이동평균 분석")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
