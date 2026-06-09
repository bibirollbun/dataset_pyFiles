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


import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"
TRAIN_PATH = DATA_DIR + "train.csv"
TRAIN_DEMO_PATH = DATA_DIR + "train_demographics.csv"

train_df = pd.read_csv(TRAIN_PATH)
dem_df = pd.read_csv(TRAIN_DEMO_PATH)


print("Train shape:", train_df.shape)
train_df.head(5)


print("Demographics shape:", dem_df.shape)
dem_df.head(5)


# シーケンスごとに一意のgestureを取得して、その分布を集計
gesture_counts = train_df[['sequence_id', 'gesture']].drop_duplicates().gesture.value_counts()
print("ユニークジェスチャー数:", gesture_counts.shape[0])
print(gesture_counts)


# 振る舞いごとに一意のgestureを取得して、その分布を集計
behavior_counts = train_df[['sequence_id','behavior']].drop_duplicates().behavior.value_counts()
print("\nユニークな振る舞いの数:", behavior_counts.shape[0])
print(behavior_counts)


# ターゲット(BFRB)と非ターゲットの比率
seq_types = train_df[['sequence_id', 'sequence_type']].drop_duplicates().sequence_type.value_counts()
print("\nTarget/Non-Targetシーケンス数:\n", seq_types)
print("ターゲット(%) = {:.1f}%".format(100 * seq_types["Target"] / seq_types.sum()))


# 棒グラフで可視化（ターゲット vs 非ターゲット）
plt.figure(figsize=(4,4))
sns.barplot(x=seq_types.index, y=seq_types.values)
plt.title("sequence: Target vs Non-Target")
plt.ylabel("sequence count")
plt.show()


# シーケンスの長さ（各sequence_idに対応する行数）を集計
seq_lengths = train_df.groupby('sequence_id')['sequence_counter'].max() + 1  # 0始まりなので+1
print(seq_lengths.describe())
plt.figure(figsize=(6,4))
sns.histplot(seq_lengths, bins=20, kde=False)
plt.title("sequence length")
plt.xlabel("time step of sequence")
plt.ylabel("sequence count")
plt.show()



# 例として特定のBFRBジェスチャーと非BFRBジェスチャーのシーケンスを各1つ抽出
example_bfrb = train_df[train_df['gesture'] == 'Above ear - pull hair']  # 髪を引っぱるの最初の200サンプル
example_non = train_df[train_df['gesture'] == 'Glasses on/off']  # 眼鏡を直すの最初の200サンプル

# 加速度の時系列をプロット
plt.figure(figsize=(10,4))
plt.plot(example_bfrb['sequence_counter'], example_bfrb['acc_x'], label='Hair pulling - acc_x')
plt.plot(example_bfrb['sequence_counter'], example_bfrb['acc_y'], label='Hair pulling - acc_y')
plt.plot(example_bfrb['sequence_counter'], example_bfrb['acc_z'], label='Hair pulling - acc_z')
plt.title("example: acc in behavior of pulling hair")
plt.xlabel("time step")
plt.ylabel("acc (m/s^2)")
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(example_non['sequence_counter'], example_non['acc_x'], label='Adjust glasses - acc_x')
plt.plot(example_non['sequence_counter'], example_non['acc_y'], label='Adjust glasses - acc_y')
plt.plot(example_non['sequence_counter'], example_non['acc_z'], label='Adjust glasses - acc_z')
plt.title("example: acc in behavior of glass on/off")
plt.xlabel("time step")
plt.ylabel("acc (m/s^2)")
plt.legend()
plt.show()



# 欠損率の計算
nan_ratio = train_df.isna().mean()  # 各列のNaN比率
nan_cols = nan_ratio[nan_ratio > 0]
print("欠損値の存在する列数:", nan_cols.shape[0])
nan_cols.sort_values(ascending=False).head(20)


# 成人/小児の人数と年齢分布
print(dem_df['adult_child'].value_counts())
print("年齢の最小,中央値,最大:", dem_df['age'].min(), dem_df['age'].median(), dem_df['age'].max())

# 身長や腕の長さの基本統計量
print(dem_df[['height_cm','shoulder_to_wrist_cm','elbow_to_wrist_cm']].describe())



# 前処理：NaNを補間する関数を定義
def fill_na_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # サーモパイルNaNを0で補完
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    df[thm_cols] = df[thm_cols].fillna(0.0)
    # ToF NaNを-1で補完
    tof_cols = [c for c in df.columns if c.startswith('tof_')]
    df[tof_cols] = df[tof_cols].fillna(-1)
    return df

# 1. Gestureフェーズに絞る
gesture_df = train_df[train_df['behavior'] == 'Performs gesture'].copy()
gesture_df = fill_na_values(gesture_df)

# 2. シーケンスごとに集計 -> 特徴量データフレーム作成
features_list = []  # ここに各シーケンスの特徴量辞書を追加
for seq_id, seq_data in gesture_df.groupby('sequence_id'):
    seq_feat = {"sequence_id": seq_id}
    # IMU特徴量: accとrotの平均と標準偏差
    for axis in ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']:
        seq_feat[f"{axis}_mean"] = seq_data[axis].mean()
        seq_feat[f"{axis}_std"]  = seq_data[axis].std()
    # サーモパイル特徴量: 平均と最大
    for i in range(1,6):
        col = f"thm_{i}"
        seq_feat[f"{col}_mean"] = seq_data[col].mean()
        seq_feat[f"{col}_max"]  = seq_data[col].max()
    # ToF特徴量: 最小距離と検出率
    for i in range(1,6):
        # 該当センサーの64列を抽出
        tof_cols = [f"tof_{i}_v{j}" for j in range(64)]
        tof_vals = seq_data[tof_cols].values  # (時間長さ x 64)のnumpy配列
        # 検出ありの値のみ考慮するため、-1をnp.nanに置換
        tof_vals_flat = tof_vals.flatten()
        tof_vals_flat = np.where(tof_vals_flat == -1, np.nan, tof_vals_flat)
        if np.all(np.isnan(tof_vals_flat)):
            # 一度も検出がない場合
            min_dist = 255  # 検出なしを255とする（255は通常範囲外の大きな値として仮定）
            detect_rate = 0.0
        else:
            min_dist = np.nanmin(tof_vals_flat)
            # フレームごとに少なくとも1つ検出があったかを判定
            frame_detect = np.any(np.isfinite(tof_vals), axis=1)  # True/False per time
            detect_rate = frame_detect.mean()
        seq_feat[f"tof_{i}_min_dist"] = min_dist
        seq_feat[f"tof_{i}_detect_rate"] = detect_rate
    # デモグラ情報付加
    subj = seq_data['subject'].iloc[0]
    dem_row = dem_df[dem_df['subject'] == subj].iloc[0]
    seq_feat["age"] = dem_row["age"]
    seq_feat["sex"] = dem_row["sex"]
    seq_feat["handedness"] = dem_row["handedness"]
    seq_feat["height_cm"] = dem_row["height_cm"]
    seq_feat["shoulder_to_wrist_cm"] = dem_row["shoulder_to_wrist_cm"]
    seq_feat["elbow_to_wrist_cm"] = dem_row["elbow_to_wrist_cm"]
    features_list.append(seq_feat)

train_features_df = pd.DataFrame(features_list)
print("特徴量データフレームのサイズ:", train_features_df.shape)
train_features_df.head(5)



# ラベル列を付加（gesture名とsequence_type）
seq_labels = train_df[['sequence_id','gesture','sequence_type']].drop_duplicates(subset='sequence_id')
train_features_df = train_features_df.merge(seq_labels, on='sequence_id', how='left')
print("特徴量データにラベルを結合したサイズ:", train_features_df.shape)
train_features_df[['sequence_id','gesture','sequence_type']].head(5)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# 便利のため、Target/Non-Target判定用の辞書を作成
gesture_to_type = {row['gesture']: row['sequence_type'] for _, row in seq_labels.iterrows()}
target_gestures = [g for g,t in gesture_to_type.items() if t=="Target"]

# 特徴量データから学習用データ構築
X = train_features_df.drop(columns=['sequence_id','gesture','sequence_type'])
y = train_features_df['gesture']  # 多クラスラベル（18クラス）

# 被験者が混ざらないよう、subjectでグループ分割
subjects = train_df[['sequence_id','subject']].drop_duplicates()['subject']
print(f"X: {X.shape}, y: {y.shape}, subjects: {subjects.shape}")


X = X.fillna(0.0)
X.head(3)


X.columns


y.head(3)


# 最後の１行だけドロップする
subjects = subjects.iloc[:-1]
subjects.head(3)


X_train, X_val, y_train, y_val, sub_train, sub_val = train_test_split(
    X, y, subjects, test_size=0.2, stratify=y, random_state=42
)
print("学習データサイズ:", X_train.shape, "検証データサイズ:", X_val.shape)

# ベースモデル：IMU特徴量のみを使用
imu_features = [c for c in X.columns if c.startswith('acc_') or c.startswith('rot_')]
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=0)
clf_baseline.fit(X_train[imu_features], y_train)

# 検証データで予測
y_pred_base = clf_baseline.predict(X_val[imu_features])
print("Baselineモデルの分類レポート:")
print(classification_report(y_val, y_pred_base, zero_division=0))


# Binary F1とMacro F1の計算
# 実際のsequence_type（Target/Non-Target）に基づき、予測もTarget/Non-Targetにマップする
y_val_type = [gesture_to_type[g] for g in y_val] 
y_pred_type = [gesture_to_type[g] for g in y_pred_base]
binary_f1 = f1_score(y_val_type, y_pred_type, pos_label="Target")
# Macro F1: 8個のターゲットクラス＋Non-Target1クラス の計9クラスで計算
y_val_macro = [g if gesture_to_type[g]=="Target" else "NonTarget" for g in y_val]
y_pred_macro = [g if gesture_to_type[g]=="Target" else "NonTarget" for g in y_pred_base]
labels = [g for g,t in gesture_to_type.items() if t=="Target"] + ["NonTarget"]
macro_f1 = f1_score(y_val_macro, y_pred_macro, labels=labels, average='macro')
print(f"Baselineモデル -> Binary F1 = {binary_f1:.3f}, Macro F1 = {macro_f1:.3f}, 平均 = {((binary_f1+macro_f1)/2):.3f}")



# 改善モデル：全特徴量を使用（IMU+サーモ+ToF+デモグラ）
clf_improved = RandomForestClassifier(n_estimators=100, random_state=0)
clf_improved.fit(X_train, y_train)
y_pred_imp = clf_improved.predict(X_val)

# 性能評価
print("Improvedモデルの分類レポート:")
print(classification_report(y_val, y_pred_imp, zero_division=0))
y_val_type = [gesture_to_type[g] for g in y_val]
y_pred_type = [gesture_to_type[g] for g in y_pred_imp]
binary_f1_imp = f1_score(y_val_type, y_pred_type, pos_label="Target")
y_val_macro = [g if gesture_to_type[g]=="Target" else "NonTarget" for g in y_val]
y_pred_macro = [g if gesture_to_type[g]=="Target" else "NonTarget" for g in y_pred_imp]
macro_f1_imp = f1_score(y_val_macro, y_pred_macro, labels=labels, average='macro')
print(f"Improvedモデル -> Binary F1 = {binary_f1_imp:.3f}, Macro F1 = {macro_f1_imp:.3f}, 平均 = {((binary_f1_imp+macro_f1_imp)/2):.3f}")


# 特徴量重要度の取得と上位表示
importances = clf_improved.feature_importances_
feature_names = X.columns
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print("上位10特徴量:\n", imp_df.head(10))

# 棒グラフで可視化
plt.figure(figsize=(8,5))
sns.barplot(x=imp_df.head(10)['importance'], y=imp_df.head(10)['feature'])
plt.title("importance of feature top 10")
plt.xlabel("importance")
plt.ylabel("feature amount")
plt.show()


# 推論用にグローバルでモデルとデモグラを準備
model = clf_improved  # 学習済みモデルを使用
dem_df_indexed = dem_df.set_index('subject')  # 被験者IDで検索しやすく

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    seq_pd = sequence.to_pandas()

    # フェーズ情報があればGesture部分を抽出、なければ全行を使用
    if 'behavior' in seq_pd.columns:
        seq_pd = seq_pd[seq_pd['behavior'] == 'Performs gesture'].copy()
    elif 'phase' in seq_pd.columns:
        seq_pd = seq_pd[seq_pd['phase'] == 'Gesture'].copy()
    else:
        seq_pd = seq_pd.copy()

    if seq_pd.empty:
        raise ValueError("No usable rows found in the input sequence.")

    # NaN補間
    seq_pd = fill_na_values(seq_pd)

    feat = {}
    # IMU特徴量
    for axis in ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']:
        feat[f"{axis}_mean"] = seq_pd[axis].mean()
        feat[f"{axis}_std"]  = seq_pd[axis].std()

    # サーモパイル特徴量
    for i in range(1, 6):
        col = f"thm_{i}"
        if seq_pd[col].isna().all():
            feat[f"{col}_mean"] = 26.0
            feat[f"{col}_max"] = 26.0
            continue
        feat[f"{col}_mean"] = seq_pd[col].mean()
        feat[f"{col}_max"] = seq_pd[col].max()

    # ToF特徴量（flattenせず、画素ごとに処理）
    for i in range(1, 6):
        tof_cols = [
            f"tof_{i}_v{j}" for j in range(64) 
            if f"tof_{i}_v{j}" in seq_pd.columns
        ]
        if not tof_cols:
            feat[f"tof_{i}_mean_dist"] = 255.0
            feat[f"tof_{i}_min_dist"] = 255.0
            continue
        # tof_colsのカラムがNaNの場合
        if seq_pd[tof_cols].isna().all().all():
            feat[f"tof_{i}_mean_dist"] = 255.0
            feat[f"tof_{i}_min_dist"] = 255.0
            continue
        # -1を NaN に置換
        vals = seq_pd[tof_cols].replace(-1, np.nan)
        feat[f"tof_{i}_mean_dist"] = vals.mean(skipna=True).mean()
        feat[f"tof_{i}_min_dist"] = vals.min(skipna=True).min()

    # デモグラ特徴
    subj = seq_pd['subject'].iloc[0]
    if subj in dem_df_indexed.index:
        dem_row = dem_df_indexed.loc[subj]
        feat["age"] = dem_row["age"]
        feat["sex"] = dem_row["sex"]
        feat["handedness"] = dem_row["handedness"]
        feat["height_cm"] = dem_row["height_cm"]
        feat["shoulder_to_wrist_cm"] = dem_row["shoulder_to_wrist_cm"]
        feat["elbow_to_wrist_cm"] = dem_row["elbow_to_wrist_cm"]
    else:
        feat["age"] = dem_df["age"].mean()
        feat["sex"] = 1
        feat["handedness"] = 1
        feat["height_cm"] = dem_df["height_cm"].mean()
        feat["shoulder_to_wrist_cm"] = dem_df["shoulder_to_wrist_cm"].mean()
        feat["elbow_to_wrist_cm"] = dem_df["elbow_to_wrist_cm"].mean()

    # 特徴量をモデルに合わせて並び替え
    X_seq = pd.DataFrame([feat])
    # 特徴量を学習時の順序に揃え、不足している列は0.0で埋める
    X_seq = X_seq.reindex(model.feature_names_in_, axis=1, fill_value=0.0)
    X_seq = X_seq.fillna(0.0)

    pred_gesture = model.predict(X_seq)[0]
    return pred_gesture



import kaggle_evaluation.cmi_inference_server

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


# test_df = pd.read_csv(DATA_DIR + "test.csv")
# test_demo_df = pd.read_csv(DATA_DIR + "test_demographics.csv")


# print(test_df.shape)
# display(test_df.head(10))
# display(test_demo_df.head(5))


# import polars as pl
# from tqdm import tqdm

# # 推論対象のユニークな sequence_id を取得
# sequence_ids = test_df['sequence_id'].unique()

# # 結果保存用
# results = []

# demo_pl = pl.from_pandas(test_demo_df)

# # 1シーケンスずつ推論
# for seq_id in tqdm(sequence_ids):
#     # 対応するシーケンスのデータを Polars に変換
#     seq_df = test_df[test_df["sequence_id"] == seq_id]
#     seq_pl = pl.from_pandas(seq_df)
#
#     try:
#         gesture = predict(seq_pl, demo_pl)
#         print(gesture)
#         results.append((seq_id, gesture))
#     except Exception as e:
#         print(f"Error for {seq_id}: {e}")
#         results.append((seq_id, "ERROR"))


# pred_df = pd.DataFrame(results, columns=["sequence_id", "gesture"])
# pred_df.head()


# res = pd.read_parquet("/kaggle/working/submission.parquet")
# res.head(10)


# test_seq_id = test_df["sequence_id"].iloc[0]

# # 2. そのシーケンスを抽出してコピー
# seq_masked = test_df[test_df["sequence_id"] == test_seq_id].copy()

# # 3. Thermopile / ToF 列を NaN マスク
# non_imu_cols = [c for c in seq_masked.columns
#                 if c.startswith("thm_") or c.startswith("tof_")]
# seq_masked[non_imu_cols] = np.nan

# # 4. Polars へ変換（predict は pl.DataFrame 受け取り）
# seq_pl  = pl.from_pandas(seq_masked)
# demo_pl = pl.from_pandas(test_demo_df)

# display(seq_pl.head(3))
# display(demo_pl.head(3))

# # 5. 推論呼び出し
# pred_gesture = predict(seq_pl, demo_pl)
# print(f"sequence_id: {test_seq_id}  →  predicted gesture = {pred_gesture}")

