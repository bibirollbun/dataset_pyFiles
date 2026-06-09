import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import pickle
import warnings
warnings.filterwarnings('ignore')

# 環境判定
IS_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running on Kaggle: {IS_KAGGLE}")

if IS_KAGGLE:
    V12_MODEL_PATH = '/kaggle/input/cmi-v12-real-data-models'
else:
    V12_MODEL_PATH = 'models_v12'


# モデルをロード
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

print("Loading V12 models...")
lgb_model = lgb.Booster(model_file=os.path.join(V12_MODEL_PATH, 'lgb_model_v12_final.txt'))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(V12_MODEL_PATH, 'xgb_model_v12_final.json'))
cat_model = CatBoostClassifier()
cat_model.load_model(os.path.join(V12_MODEL_PATH, 'cat_model_v12_final.cbm'))

# メタデータ
with open(os.path.join(V12_MODEL_PATH, 'scaler_v12.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(V12_MODEL_PATH, 'feature_names_v12.pkl'), 'rb') as f:
    feature_names = pickle.load(f)
with open(os.path.join(V12_MODEL_PATH, 'gesture_mapping_v12.pkl'), 'rb') as f:
    gesture_mapping = pickle.load(f)

print(f"Models loaded. Features: {len(feature_names)}")


def extract_features(df):
    """特徴量抽出"""
    features = {}
    
    # 基本統計量
    for col in df.columns:
        if col not in ['timestamp', 'sequence_id']:
            features[f'{col}_mean'] = np.mean(df[col])
            features[f'{col}_std'] = np.std(df[col])
            features[f'{col}_min'] = np.min(df[col])
            features[f'{col}_max'] = np.max(df[col])
            features[f'{col}_median'] = np.median(df[col])
            features[f'{col}_q25'] = np.percentile(df[col], 25)
            features[f'{col}_q75'] = np.percentile(df[col], 75)
            features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
            features[f'{col}_skew'] = df[col].skew() if len(df[col]) > 2 else 0
            features[f'{col}_kurtosis'] = df[col].kurtosis() if len(df[col]) > 3 else 0
    
    # 重力除去線形加速度
    if 'acc_x' in df.columns:
        gravity_x = np.mean(df['acc_x'])
        gravity_y = np.mean(df['acc_y'])
        gravity_z = np.mean(df['acc_z'])
        
        linear_acc_x = df['acc_x'] - gravity_x
        linear_acc_y = df['acc_y'] - gravity_y
        linear_acc_z = df['acc_z'] - gravity_z
        
        features['linear_acc_x_mean'] = np.mean(linear_acc_x)
        features['linear_acc_y_mean'] = np.mean(linear_acc_y)
        features['linear_acc_z_mean'] = np.mean(linear_acc_z)
        features['linear_acc_magnitude'] = np.mean(np.sqrt(linear_acc_x**2 + linear_acc_y**2 + linear_acc_z**2))
    
    # クォータニオンからオイラー角
    if all(col in df.columns for col in ['quat_w', 'quat_x', 'quat_y', 'quat_z']):
        qw = df['quat_w'].values
        qx = df['quat_x'].values
        qy = df['quat_y'].values
        qz = df['quat_z'].values
        
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        features['euler_roll_mean'] = np.mean(roll)
        features['euler_pitch_mean'] = np.mean(pitch)
        features['euler_yaw_mean'] = np.mean(yaw)
        features['euler_roll_std'] = np.std(roll)
        features['euler_pitch_std'] = np.std(pitch)
        features['euler_yaw_std'] = np.std(yaw)
        
        # 角速度
        if len(df) > 1 and 'timestamp' in df.columns:
            dt = np.diff(df['timestamp'].values) / 1000.0
            dt[dt == 0] = 1e-6  # ゼロ除算を防ぐ
            dqw = np.diff(qw) / dt
            dqx = np.diff(qx) / dt
            dqy = np.diff(qy) / dt
            dqz = np.diff(qz) / dt
            
            features['angular_velocity_mean'] = np.mean(np.sqrt(dqx**2 + dqy**2 + dqz**2))
            features['angular_velocity_std'] = np.std(np.sqrt(dqx**2 + dqy**2 + dqz**2))
        else:
            features['angular_velocity_mean'] = 0
            features['angular_velocity_std'] = 0
    
    # ジャーク
    if 'acc_x' in df.columns and len(df) > 1 and 'timestamp' in df.columns:
        dt = np.diff(df['timestamp'].values) / 1000.0
        dt[dt == 0] = 1e-6
        jerk_x = np.diff(df['acc_x']) / dt
        jerk_y = np.diff(df['acc_y']) / dt
        jerk_z = np.diff(df['acc_z']) / dt
        
        features['jerk_x_mean'] = np.mean(jerk_x)
        features['jerk_y_mean'] = np.mean(jerk_y)
        features['jerk_z_mean'] = np.mean(jerk_z)
        features['jerk_magnitude'] = np.mean(np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2))
    else:
        features['jerk_x_mean'] = 0
        features['jerk_y_mean'] = 0
        features['jerk_z_mean'] = 0
        features['jerk_magnitude'] = 0
    
    # ToFセンサー
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        tof_values = df[tof_cols].values.flatten()
        features['tof_global_mean'] = np.mean(tof_values)
        features['tof_global_std'] = np.std(tof_values)
        features['tof_global_min'] = np.min(tof_values)
        features['tof_global_max'] = np.max(tof_values)
    
    # シーケンス長
    features['sequence_length'] = len(df)
    if 'timestamp' in df.columns:
        features['duration_ms'] = df['timestamp'].max() - df['timestamp'].min()
    else:
        features['duration_ms'] = 0
    
    return features


# CMI推論サーバー
if IS_KAGGLE:
    from kaggle_evaluation.cmi_inference_server import CMIInferenceServer
    
    def predict(test_data: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
        """V12モデルによる予測"""
        print(f"Starting prediction for {len(metadata)} sequences")
        
        # pandas変換
        test_df = test_data.to_pandas()
        metadata_df = metadata.to_pandas()
        
        predictions = []
        
        # ジェスチャー名リスト
        gesture_names = [
            'Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
            'Eyebrow - pull hair', 'Eyelash - pull hair', 'Feel around in tray and pull out an object',
            'Forehead - pull hairline', 'Forehead - scratch', 'Glasses on/off',
            'Neck - pinch skin', 'Neck - scratch', 'Pinch knee/leg skin',
            'Pull air toward your face', 'Scratch knee/leg skin', 'Text on phone',
            'Wave hello', 'Write name in air', 'Write name on leg'
        ]
        
        for idx, row in metadata_df.iterrows():
            seq_id = row['sequence_id']
            seq_data = test_df[test_df['sequence_id'] == seq_id].copy()
            
            if len(seq_data) == 0:
                predictions.append('Wave hello')
                continue
            
            # 特徴量抽出
            features = extract_features(seq_data)
            X = pd.DataFrame([features])
            
            # 特徴量を整形
            X_aligned = X[feature_names]
            X_scaled = scaler.transform(X_aligned)
            
            # 予測（アンサンブル）
            pred_lgb = lgb_model.predict(X_scaled, num_iteration=lgb_model.best_iteration)[0]
            
            dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
            pred_xgb = xgb_model.predict(dmatrix)[0]
            
            pred_cat = cat_model.predict_proba(X_scaled)[0]
            
            # アンサンブル（LightGBM 40% + XGBoost 30% + CatBoost 30%）
            ensemble_probs = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat
            
            # 最も確率の高いクラス
            pred_class = np.argmax(ensemble_probs)
            
            # ジェスチャー名
            gesture_name = gesture_names[pred_class] if pred_class < 18 else 'Wave hello'
            predictions.append(gesture_name)
        
        # 結果を返す
        result = pl.DataFrame({
            'sequence_id': metadata['sequence_id'],
            'gesture': predictions
        })
        
        print(f"Predictions completed. Shape: {result.shape}")
        return result
    
    # サーバー起動
    print("Starting CMI Inference Server...")
    inference_server = CMIInferenceServer(predict)
    inference_server.serve()
    print("Inference completed")
    
else:
    print("Local test mode - creating dummy submission")
    
    gesture_names = [
        'Above ear - pull hair', 'Cheek - pinch skin', 'Drink from bottle/cup',
        'Eyebrow - pull hair', 'Eyelash - pull hair', 'Feel around in tray and pull out an object',
        'Forehead - pull hairline', 'Forehead - scratch', 'Glasses on/off',
        'Neck - pinch skin', 'Neck - scratch', 'Pinch knee/leg skin',
        'Pull air toward your face', 'Scratch knee/leg skin', 'Text on phone',
        'Wave hello', 'Write name in air', 'Write name on leg'
    ]
    
    submission = pl.DataFrame({
        'sequence_id': list(range(1, 101)),
        'gesture': [gesture_names[i % 18] for i in range(100)]
    })
    submission.write_parquet('submission.parquet')
    print(f"Dummy submission created: {submission.shape}")

