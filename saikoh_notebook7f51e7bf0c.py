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
from tqdm import tqdm

# モデルロード
print("モデルを読み込み中...")
model_data = joblib.load('/kaggle/input/cmi-optimized-model/optimized_full_model.pkl')
models = model_data['models']
le = model_data['label_encoder']
feature_names = model_data['feature_names']

# テストデータ読み込み
print("テストデータを読み込み中...")
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
print(f"テストデータ: {len(test_df):,}行")

# BFRBジェスチャー定義
BFRB_GESTURES = [
  'Above ear - pull hair', 'Cheek - pinch skin', 'Eyebrow - pull hair',
  'Eyelash - pull hair', 'Forehead - pull hairline', 'Neck - pinch skin',
  'Pinch knee/leg skin', 'Neck - scratch', 'Forehead - scratch',
  'Scratch knee/leg skin'
]

def create_optimized_features(seq_data):
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

  # ToF
  if 'tof_1_v0' in seq_data.columns:
      for sensor_id in range(1, 6):
          tof_cols = [f'tof_{sensor_id}_v{i}' for i in range(64)]
          tof_data = seq_data[tof_cols]
          valid_mask = (tof_data != -1).any(axis=1)

          if valid_mask.any():
              valid_tof = tof_data[valid_mask].replace(-1, np.nan)
              features[f'tof_{sensor_id}_mean'] = np.nanmean(valid_tof.values)
              features[f'tof_{sensor_id}_std'] = np.nanstd(valid_tof.values)
              features[f'tof_{sensor_id}_has_data'] = 1
          else:
              features[f'tof_{sensor_id}_mean'] = 0
              features[f'tof_{sensor_id}_std'] = 0
              features[f'tof_{sensor_id}_has_data'] = 0

  features['sequence_length'] = len(seq_data)

  mid = len(seq_data) // 2
  if mid > 0:
      features['acc_mag_trend'] = np.mean(acc_mag[mid:]) - np.mean(acc_mag[:mid])

  return features

# 予測処理
print("予測を実行中...")
predictions = {}

for seq_id in tqdm(test_df['sequence_id'].unique()):
  seq_data = test_df[test_df['sequence_id'] == seq_id]

  if 'phase' in seq_data.columns:
      gesture_data = seq_data[seq_data['phase'] == 'Gesture']
  else:
      gesture_data = seq_data

  if len(gesture_data) > 0:
      try:
          features = create_optimized_features(gesture_data)
          X_test = pd.DataFrame([features])[feature_names]

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

          for row_id in seq_data['row_id']:
              predictions[row_id] = gesture

      except Exception as e:
          print(f"エラー: {seq_id} - {e}")
          for row_id in seq_data['row_id']:
              predictions[row_id] = 'Text on phone'

# 提出ファイル作成
print("提出ファイルを作成中...")
submission = pd.DataFrame([
  {'row_id': row_id, 'gesture': gesture}
  for row_id, gesture in predictions.items()
])

# 欠損チェック
all_test_rows = test_df['row_id'].unique()
missing_rows = set(all_test_rows) - set(submission['row_id'])

if missing_rows:
  print(f"{len(missing_rows)}個の欠損行を補完中...")
  missing_df = pd.DataFrame({
      'row_id': list(missing_rows),
      'gesture': 'Text on phone'
  })
  submission = pd.concat([submission, missing_df], ignore_index=True)

submission = submission.sort_values('row_id').reset_index(drop=True)

# 重要: Parquet形式で保存！
submission.to_parquet('submission.parquet', index=False)
print(f"\n✅ 完了！submission.parquetを作成しました")
print(f"予測数: {len(submission)}")
print(submission.head())

# 確認用にCSVも保存
submission.to_csv('submission.csv', index=False)



# CSVの代わりにParquet形式で保存
submission.to_parquet('submission.parquet', index=False)



