import os, joblib
import sys
sys.path.append('/kaggle/input/weights')
import pandas as pd
import polars as pl
import kaggle_evaluation.cmi_inference_server
import warnings
warnings.filterwarnings('ignore')
from collections import Counter


# 예측 모델 로드
voting_clf = joblib.load('/kaggle/input/weights/voting_250604.pkl')

# 스케일러 로드
scaler = joblib.load('/kaggle/input/weights/scaler.pkl')

# 레이블 인코더 딕셔너리 로드
le_map = joblib.load('/kaggle/input/weights/label_encoder.pkl')


def tof_summary(sequence):
    tof_cols = [col for col in sequence.columns if 'tof' in col]

    sequence['tof_mean'] = sequence[tof_cols].mean(axis=1)
    sequence['tof_median'] = sequence[tof_cols].median(axis=1)
    sequence['tof_min'] = sequence[tof_cols].min(axis=1)
    sequence['tof_max'] = sequence[tof_cols].max(axis=1)
    sequence['tof_std'] = sequence[tof_cols].std(axis=1)
    sequence['tof_range'] = sequence['tof_max'] - sequence['tof_min']
    sequence['tof_quantile10'] = sequence[tof_cols].quantile(0.1, axis=1)
    sequence['tof_quantile25'] = sequence[tof_cols].quantile(0.25, axis=1)
    sequence['tof_quantile75'] = sequence[tof_cols].quantile(0.75, axis=1)
    sequence['tof_quantile90'] = sequence[tof_cols].quantile(0.9, axis=1)
    sequence['tof_no_signal'] = sequence[tof_cols].eq(-1).sum(axis=1) # -1은 측정된 거리가 없어 no signal

    # tof 픽셀 열 삭제
    sequence = sequence.drop(tof_cols, axis=1)
    return sequence


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    '''
    input : 한 시퀀스 데이터프레임
    output : 시퀀스의 예측 제스쳐
    '''

    sequence = sequence.to_pandas()

    # null 값 0으로 대체
    sequence = sequence.fillna(0)

    # 불필요 컬럼 제거
    sequence = sequence.drop(['sequence_id','row_id','sequence_counter','subject'], axis=1)
    if 'column_0' in sequence.columns:
        sequence = sequence.drop('column_0', axis=1)
    
    # 특징 요약
    sequence = tof_summary(sequence)
    
    # 스케일링
    
    sequence_scaled = scaler.transform(sequence.to_numpy())
    
    # 예측
    y_pred = voting_clf.predict(sequence_scaled).astype(int)
    
    # 예측 배열 중 최빈값을 선택
    y_pred = Counter(y_pred).most_common(1)[0][0]
    
    # 연속형 -> 범주형 레이블 변환
    return le_map[y_pred]


# 채점 시뮬레이션 제출용
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/weights/test.csv',
            '/kaggle/input/weights/test_demographics.csv',
        )
    )

