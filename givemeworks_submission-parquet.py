import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- 1. データ読み込み ---
print("Loading data...")
# Colab環境では、APIでダウンロードしたファイルが直下に展開される
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

# 属性情報をマージ
train = pd.merge(train, train_demo, on='subject', how='left')
test = pd.merge(test, test_demo, on='subject', how='left')

print("Data loaded successfully.")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# --- 2. 特徴量エンジニアリング ---
def create_features(df):
    """
    シーケンスデータから統計的特徴量を生成する関数
    """
    sensor_cols = [col for col in df.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    df[sensor_cols] = df[sensor_cols].fillna(0)
    aggregations = {
        col: ['mean', 'std', 'min', 'max', 'median'] for col in sensor_cols
    }
    agg_df = df.groupby('sequence_id').agg(aggregations)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    demo_cols = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    demo_df = df.groupby('sequence_id')[demo_cols].first()
    final_df = pd.concat([agg_df, demo_df], axis=1)
    return final_df

print("Creating features for training data...")
X_train_agg = create_features(train)

# --- ラベル（目的変数）の準備 ---
labels_map = train[['sequence_id', 'gesture']].drop_duplicates().set_index('sequence_id')['gesture']
y_train_series = X_train_agg.index.map(labels_map)

le = LabelEncoder()
y_train = le.fit_transform(y_train_series)
label_names = le.classes_

X_train = X_train_agg.reset_index(drop=True)
X_train = X_train.fillna(0)

print("Features created. Training data shape:", X_train.shape)

# --- 3. モデル学習 ---
print("Training LightGBM model...")

params = {
    'objective': 'multiclass',
    'num_class': len(label_names),
    'metric': 'multi_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train)],
          eval_metric='multi_logloss',
          callbacks=[lgb.early_stopping(100, verbose=False)])

print("Model training completed.")

# --- 4. 予測 ---
print("Generating predictions...")
predictions = []
test_sequence_ids = test['sequence_id'].unique()

for seq_id in test_sequence_ids:
    seq_df = test[test['sequence_id'] == seq_id]
    X_test_seq = create_features(seq_df)
    
    X_test_seq = X_test_seq.reset_index(drop=True)
    X_test_seq = X_test_seq.fillna(0)
    
    missing_cols = set(X_train.columns) - set(X_test_seq.columns)
    for c in missing_cols:
        X_test_seq[c] = 0
    X_test_seq = X_test_seq[X_train.columns]
    
    pred_label_encoded = model.predict(X_test_seq)
    pred_label = le.inverse_transform(pred_label_encoded)[0]
    
    predictions.append({'sequence_id': seq_id, 'gesture': pred_label})

# --- 5. 提出ファイルの作成 ---
submission_df = pd.DataFrame(predictions)
submission_df.to_parquet('/kaggle/working/submission.parquet', index=False)

print("Submission file created successfully: submission.parquet")
print(submission_df.head())

