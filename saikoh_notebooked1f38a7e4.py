# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import pandas as pd
import numpy as np
import joblib

# モデルの読み込み
print("Loading model...")
model_data = joblib.load('/kaggle/input/cmi-optimized-model/optimized_full_model.pkl')
models = model_data['models']
le = model_data['label_encoder']
feature_names = model_data['feature_names']
print("Model loaded successfully")

def create_features(seq_data):
  """特徴量作成"""
  features = {}

  # 加速度特徴量
  for axis in ['x', 'y', 'z']:
      col = f'acc_{axis}'
      data = seq_data[col].values

      features[f'{col}_mean'] = np.mean(data)
      features[f'{col}_std'] = np.std(data)
      features[f'{col}_max'] = np.max(data)
      features[f'{col}_min'] = np.min(data)
      features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
      features[f'{col}_p25'] = np.percentile(data, 25)
      features[f'{col}_p75'] = np.percentile(data, 75)
      features[f'{col}_iqr'] = features[f'{col}_p75'] - features[f'{col}_p25']

      diff1 = np.diff(data)
      if len(diff1) > 0:
          features[f'{col}_diff_mean'] = np.mean(diff1)
          features[f'{col}_diff_std'] = np.std(diff1)

  # 加速度の大きさ
  acc_mag = np.sqrt(seq_data['acc_x']**2 + seq_data['acc_y']**2 + seq_data['acc_z']**2)
  features['acc_mag_mean'] = np.mean(acc_mag)
  features['acc_mag_std'] = np.std(acc_mag)
  features['acc_mag_max'] = np.max(acc_mag)
  features['acc_mag_min'] = np.min(acc_mag)

  # 角度特徴
  features['acc_xy_angle_mean'] = np.mean(np.arctan2(seq_data['acc_y'], seq_data['acc_x']))
  features['acc_xy_angle_std'] = np.std(np.arctan2(seq_data['acc_y'], seq_data['acc_x']))

  # エネルギー
  features['acc_energy'] = np.sum(acc_mag**2) / len(acc_mag)
  features['acc_log_energy'] = np.log1p(features['acc_energy'])

  # ジャイロ特徴
  for axis in ['w', 'x', 'y', 'z']:
      col = f'rot_{axis}'
      features[f'{col}_mean'] = seq_data[col].mean()
      features[f'{col}_std'] = seq_data[col].std()
      features[f'{col}_max'] = seq_data[col].max()
      features[f'{col}_min'] = seq_data[col].min()

  # 回転の大きさ
  rot_mag = np.sqrt(seq_data['rot_x']**2 + seq_data['rot_y']**2 + seq_data['rot_z']**2)
  features['rot_mag_mean'] = np.mean(rot_mag)
  features['rot_mag_std'] = np.std(rot_mag)
  features['rot_energy'] = np.sum(rot_mag**2) / len(rot_mag)

  # サーモパイル
  for i in range(1, 6):
      col = f'thm_{i}'
      if col in seq_data.columns:
          valid_data = seq_data[seq_data[col] != -1][col]
          if len(valid_data) > 0:
              features[f'{col}_mean'] = valid_data.mean()
              features[f'{col}_std'] = valid_data.std()
          else:
              features[f'{col}_mean'] = 0
              features[f'{col}_std'] = 0
      else:
          features[f'{col}_mean'] = 0
          features[f'{col}_std'] = 0

  # ToF
  for sensor_id in range(1, 6):
      features[f'tof_{sensor_id}_mean'] = 0
      features[f'tof_{sensor_id}_std'] = 0
      features[f'tof_{sensor_id}_has_data'] = 0

  features['sequence_length'] = len(seq_data)

  mid = len(seq_data) // 2
  if mid > 0:
      features['acc_mag_trend'] = np.mean(acc_mag[mid:]) - np.mean(acc_mag[:mid])
  else:
      features['acc_mag_trend'] = 0

  return features

def predict(test_batch):
  """Kaggleから呼ばれる予測関数"""
  sequence, demographics = test_batch

  # polarsの場合はpandasに変換
  if hasattr(sequence, 'to_pandas'):
      sequence = sequence.to_pandas()

  # Gestureフェーズのデータを取得
  if 'phase' in sequence.columns:
      gesture_data = sequence[sequence['phase'] == 'Gesture']
  else:
      gesture_data = sequence

  if len(gesture_data) > 0:
      try:
          # 特徴量作成
          features = create_features(gesture_data)

          # 特徴量の存在を確認
          X_test = pd.DataFrame([features])
          missing_cols = set(feature_names) - set(X_test.columns)
          for col in missing_cols:
              X_test[col] = 0
          X_test = X_test[feature_names]

          # 予測
          all_probs = []
          for model_info in models:
              lgb_model = model_info['lgb']
              xgb_model = model_info['xgb']

              lgb_proba = lgb_model.predict_proba(X_test)[0]
              xgb_proba = xgb_model.predict_proba(X_test)[0]

              ensemble_proba = 0.7 * lgb_proba + 0.3 * xgb_proba
              all_probs.append(ensemble_proba)

          final_proba = np.mean(all_probs, axis=0)
          pred_class = np.argmax(final_proba)
          gesture = le.inverse_transform([pred_class])[0]

          return gesture

      except Exception as e:
          print(f"Error in prediction: {e}")
          import traceback
          traceback.print_exc()
          return 'Text on phone'
  else:
      return 'Text on phone'

print("Prediction function ready")



