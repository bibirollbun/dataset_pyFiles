import torch
import sys
sys.path.append('/kaggle/input/weights')
import os, joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import polars as pl
import kaggle_evaluation.cmi_inference_server
from cnn_AutoEncoder import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


model = CNNAutoEncoderWithGAP(latent_channels=32, dropout_p=0.0)
model.load_state_dict(torch.load('/kaggle/input/weights/cnn_AE_250608.pth', map_location='cpu'))

le = joblib.load('/kaggle/input/weights/label_encoder.pkl')

scaler = joblib.load('/kaggle/input/weights/scaler_all_features_improved.pkl')
scaler_imu = joblib.load('/kaggle/input/weights/scaler_imu_optimized.pkl')

best_model = joblib.load('/kaggle/input/weights/voting_ensemble_best_all_features_improved.pkl')
imu_model = joblib.load('/kaggle/input/weights/voting_ensemble_best_imu_optimized_250613_1352.pkl')


def feature_engineering(df: pd.DataFrame):
    tqdm.pandas(desc="Sequence Features")
    imu_cols = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z','acc_mag']
    vector_cols = [f'tof_v{i}' for i in range(32)]
    
    # 집계 대상 수치형 센서 컬럼 (id, 대상, 인구통계 제외)
    stat_cols = [
        c for c in df.columns
        if c not in ['sequence_counter', "sequence_id", 'subject']+vector_cols
    ]
    
    def extract_features(group):
        feature = {}
        # 시퀀스 길이
        seq_len = group["sequence_counter"].max()
        feature["seq_len"] = seq_len
        
        # 전역 구간에 대해 기본적인 통계 특징 추출
        for col in stat_cols:
            if col in imu_cols or 'thm' in col:
                x = group[col]
                feature[f"{col}_mean"] = x.mean()
                feature[f"{col}_std"] = x.std()
                feature[f"{col}_min"] = x.min()
                feature[f"{col}_max"] = x.max()
                feature[f"{col}_q25"] = x.quantile(0.25)
                feature[f"{col}_q50"] = x.median()
                feature[f"{col}_q75"] = x.quantile(0.75)
                
                diff = group[col].diff().fillna(0)
                feature[f'{col}_diff_mean'] = diff.mean()
                feature[f'{col}_diff_std'] = diff.std()

        # 초기, 중간, 후기 phase로 나누어 조건부 집계
        conditions = {
            'first_30pct': group['sequence_counter'] < seq_len * 0.3,
            'middle_40pct': (group['sequence_counter'] >= seq_len * 0.3) & (group['sequence_counter'] < seq_len * 0.7),
            'last_10pct': group['sequence_counter'] >= seq_len * 0.9,
            'last_20pct': group['sequence_counter'] >= seq_len * 0.8,
            'last_30pct': group['sequence_counter'] >= seq_len * 0.7,
            'last_40pct': group['sequence_counter'] >= seq_len * 0.6,
            'window_10~20pct': (group['sequence_counter'] >= seq_len * 0.8) & (group['sequence_counter'] < seq_len * 0.9),
            'window_20~30pct': (group['sequence_counter'] >= seq_len * 0.7) & (group['sequence_counter'] < seq_len * 0.8),
            'window_30~40pct': (group['sequence_counter'] >= seq_len * 0.6) & (group['sequence_counter'] < seq_len * 0.7),
        }

        # phase에 대해 더 다양한 특징 추출
        for part_name, mask in conditions.items():
            part = group[mask]
            for col in imu_cols:
                x = part[col]
                feature[f'{col}_mean_{part_name}'] = x.mean()
                feature[f'{col}_std_{part_name}'] = x.std()
                feature[f'{col}_median_{part_name}'] = x.median()
                feature[f"{col}_min_{part_name}"] = x.min()
                feature[f"{col}_max_{part_name}"] = x.max()
                feature[f"{col}_q25_{part_name}"] = x.quantile(0.25)
                feature[f"{col}_q75_{part_name}"] = x.quantile(0.75)
                
                diff = group[col].diff().fillna(0)
                feature[f'{col}_diff_mean'] = diff.mean()
                feature[f'{col}_diff_std'] = diff.std()
        
        # tof vector 통계
        x = group[vector_cols].values.flatten() # (seq_len * 32)
        feature["tof_vec_mean"] = x.mean()
        feature["tof_vec_std"] = x.std()
        feature["tof_vec_q25"] = np.quantile(x, 0.25)
        feature["tof_vec_q50"] = np.median(x)
        feature["tof_vec_q75"] = np.quantile(x,0.75)

        return pd.Series(feature)

    # 시퀀스별 통계 집계
    cleaned_df = df.groupby("sequence_id").progress_apply(extract_features).reset_index()

    # 결측치 0으로 채우기
    cleaned_df = cleaned_df.fillna(0)
    return cleaned_df


def feature_engineering2(df: pd.DataFrame): # IMU Only Feature Extractor
    tqdm.pandas(desc="Sequence Features(IMU Only)") 
    imu_cols = [col for col in df.columns if 'acc' in col or 'rot' in col]
    
    def extract_features(group):
        feature = {}
        
        # 시퀀스 길이
        seq_len = group["sequence_counter"].max()
        feature["seq_len"] = seq_len
        
        # 전역 구간에 대해 기본적인 통계 특징 추출
        for col in imu_cols:
            x = group[col]
            feature[f"{col}_mean"] = x.mean()
            feature[f"{col}_std"] = x.std()
            feature[f"{col}_min"] = x.min()
            feature[f"{col}_max"] = x.max()
            feature[f"{col}_q25"] = x.quantile(0.25)
            feature[f"{col}_q50"] = x.median()
            feature[f"{col}_q75"] = x.quantile(0.75)


        # 후반 중점 세부 phase로 나누어 조건부 집계
        conditions = {
            'first_30pct': group['sequence_counter'] < seq_len * 0.3,
            'middle_40pct': (group['sequence_counter'] >= seq_len * 0.3) & (group['sequence_counter'] < seq_len * 0.7),
            'last_10pct': group['sequence_counter'] >= seq_len * 0.9,
            'last_20pct': group['sequence_counter'] >= seq_len * 0.8,
            'last_30pct': group['sequence_counter'] >= seq_len * 0.7,
            'last_40pct': group['sequence_counter'] >= seq_len * 0.6,
            'window_10~20pct': (group['sequence_counter'] >= seq_len * 0.8) & (group['sequence_counter'] < seq_len * 0.9),
            'window_20~30pct': (group['sequence_counter'] >= seq_len * 0.7) & (group['sequence_counter'] < seq_len * 0.8),
            'window_30~40pct': (group['sequence_counter'] >= seq_len * 0.6) & (group['sequence_counter'] < seq_len * 0.7),
        }
        
        # phase에 대해 더 다양한 특징 추출
        for part_name, mask in conditions.items():
            part = group[mask]
            for col in imu_cols:
                x = part[col]
                feature[f'{col}_mean_{part_name}'] = x.mean()
                feature[f'{col}_std_{part_name}'] = x.std()
                feature[f'{col}_median_{part_name}'] = x.median()
                feature[f"{col}_min_{part_name}"] = x.min()
                feature[f"{col}_max_{part_name}"] = x.max()
                feature[f"{col}_q25_{part_name}"] = x.quantile(0.25)
                feature[f"{col}_q75_{part_name}"] = x.quantile(0.75)
                feature[f"{col}_delta"] = x.iloc[-1] - x.iloc[0]
                feature[f"{col}_n_changes"] = (x.diff().abs() > 0).sum()
                feature[f"{col}_corr_time"] = group["sequence_counter"].corr(x) # 특성과 시간과의 상관계수 추가(추세 요약)
                peak, _ = find_peaks(x.values)
                feature[f"{col}_peak_count"] = len(peak)
                
                diff = group[col].diff().fillna(0)
                feature[f'{col}_diff_mean'] = diff.mean()
                feature[f'{col}_diff_std'] = diff.std()

        return pd.Series(feature)

    # 시퀀스별 집계
    cleaned_df = df.groupby("sequence_id").progress_apply(extract_features).reset_index()

    # 결측치 0으로 채우기
    cleaned_df = cleaned_df.fillna(0)
    return cleaned_df


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df = sequence.to_pandas()
    df = df.drop('row_id', axis=1) # 불필요 열 제거

    # 결측치 처리
    # tof NaN 값은 거리 센서로부터 신호를 받지 않은 것이므로 -1로 변경
    tof_cols = [col for col in df.columns if 'tof' in col]
    tof_missing_index = df[df[tof_cols].isnull().any(axis=1)].index
    
    df.loc[tof_missing_index, tof_cols] = df.loc[tof_missing_index, tof_cols].fillna(-1)
        
    # 나머지 결측치 0으로 채우기
    not_tof_cols = [col for col in df.columns if col not in tof_cols]
    for col in not_tof_cols:
        df[col] = df[col].fillna(0)

    # 픽셀 정보 -> 잠재 벡터
    tof_data = torch.tensor(df[tof_cols].values.reshape(-1, 5,8,8), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        _, z = model(tof_data)
    z = np.array(z)
    
    # vector to DataFrame
    vector_cols = [f'tof_v{i}' for i in range(32)]
    tof_vec = pd.DataFrame(z, columns=vector_cols)

    # tof 채널당 평균 추가
    for i in range(1, 6):
        cols = [col for col in df.columns if f'tof_{i}_v' in col]
        
        # -1을 NaN으로 바꾼 후 계산
        masked = df[cols].replace(-1, np.nan)
        df[f'tof_{i}_mean'] = masked.mean(axis=1)
    
    df = df.drop(tof_cols, axis=1)
    df = pd.concat([df, tof_vec], axis=1)

    # 가속도 크기 열 추가
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # 테스트 데이터가 IMU only인지 all features인지 확인하여 모델을 따로 적용
    non_valid = ['thm_1','thm_2','thm_3','thm_4','thm_5']
    IMU_ONLY = True if (df[non_valid].values==0).all() else False

    if IMU_ONLY:
        # 통계
        cleaned_data = feature_engineering2(df)
        
        valid_feature = [col for col in cleaned_data.columns if 'acc' in col or 'rot' in col] +['sequence_id','seq_len']
        invalid_feature = [col for col in cleaned_data.columns if col not in valid_feature]
        imu_only_data = cleaned_data.drop(invalid_feature, axis=1)

        test_X = imu_only_data.drop('sequence_id', axis=1)
        test_id = imu_only_data['sequence_id'].values
        X_test_scaled = scaler_imu.transform(test_X)
        y_pred = imu_model.predict(X_test_scaled).astype(int)
        
    else:
        # 통계 수행
        cleaned_data = feature_engineering(df)
        
        test_X = cleaned_data.drop('sequence_id', axis=1)
        test_id = cleaned_data['sequence_id'].values
        
        # 스케일링
        X_test_scaled = scaler.transform(test_X)
    
        # 한 시퀀스에 대해 예측
        y_pred = best_model.predict(X_test_scaled).astype(int)
        
    out = le.inverse_transform(y_pred)[0]
    return out


import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

