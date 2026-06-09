# Kaggle Notebook 用: CMI - Detect Behavior with Sensor Data

import pandas as pd
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import tqdm

# --------------------
# 特徴量エンジニアリング
# --------------------
def feature_engineering(df):
    # 加速度の大きさ
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # 回転角
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    # ジャーク・角速度
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    return df

# --------------------
# モデルとスケーラー（ダミー）
# --------------------
NUMERICAL_FEATURES = ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel']
FEATURE_NAMES = NUMERICAL_FEATURES

# 通常は学習済みモデルとスケーラーをロードしますが、ここではダミーで初期化
feature_scaler = StandardScaler()
predictor = RandomForestClassifier()

# --------------------
# ダミー学習（本来は別Notebookで行って保存）
# --------------------
# 学習用ダミーデータを生成して学習（Notebook上で提出用に必要）
X_dummy = np.random.rand(100, len(NUMERICAL_FEATURES))
y_dummy = np.random.randint(0, 3, 100)
feature_scaler.fit(X_dummy)
predictor.fit(X_dummy.tolist(), y_dummy)

# --------------------
# 推論関数
# --------------------
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> int:
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()

    sequence = pd.merge(sequence, demographics, on='subject', how='left')
    sequence = feature_engineering(sequence)

    sequence[NUMERICAL_FEATURES] = feature_scaler.transform(sequence[NUMERICAL_FEATURES])

    # 特徴量をまとめる（シンプルに平均）
    features_agg = sequence[NUMERICAL_FEATURES].mean().values.reshape(1, -1)

    predicted_label = predictor.predict(features_agg)[0]
    return int(predicted_label)


if __name__ == "__main__":
    # --------------------
    # データ読み込み
    # --------------------
    test_seq = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
    test_demo = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
    
    # --------------------
    # 推論ループ
    # --------------------
    results = []
    for seq_id in tqdm(test_seq.select("sequence_id").unique().to_series().to_list()):
        seq_part = test_seq.filter(pl.col("sequence_id") == seq_id)
        pred = predict(seq_part, test_demo)
        results.append((seq_id, pred))
    
    # --------------------
    # 提出ファイル作成
    # --------------------
    submission = pd.DataFrame(results, columns=["sequence_id", "gesture"])
    submission.to_parquet("/kaggle/working/submission.parquet", index=False)
    
    # 提出用ファイルを確認
    import os
    print("存在確認:", os.path.exists("/kaggle/working/submission.parquet"))  # TrueならOK


