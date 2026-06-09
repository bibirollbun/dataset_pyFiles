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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgb


# ファイルパス
TRAIN_PATH = '/kaggle/input/playground-series-s5e3/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e3/test.csv'
SUBMISSION_PATH = '/kaggle/input/playground-series-s5e3/sample_submission.csv'


# データを読み込み (id列をインデックスに設定)
train_df = pd.read_csv(TRAIN_PATH, index_col='id')
test_df = pd.read_csv(TEST_PATH, index_col='id')
submission_df = pd.read_csv(SUBMISSION_PATH)


print("データの読み込み完了")
print(f"学習データの形状: {train_df.shape}")
print(f"テストデータの形状: {test_df.shape}")
print(f"submissionデータの形状: {submission_df.shape}")


# --- 2. 新しい特徴量の作成 ---
def add_new_features(df):
    """データフレームに3つの新しい特徴量を追加する関数"""
    # 最低気温と露点の差
    df['min_temp_dew_diff'] = df['mintemp'] - df['dewpoint']
    # 最高気温と最低気温の差（日較差）
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    # 気温と露点の差
    df['max_temp_dew_diff'] = df['temparature'] - df['dewpoint']
    return df


# 訓練データとテストデータの両方に特徴量を追加
train_df = add_new_features(train_df)
test_df = add_new_features(test_df)


train_df.head()


train_df.describe()


# # --- 2. 相関行列の計算 ---
# # 'id', 'day' は分析に不要なため除外
# features_to_analyze = [col for col in train_df.columns if col not in ['id', 'day']]
# correlation_matrix = train_df[features_to_analyze].corr()

# # --- 3. `rainfall`との相関を確認 ---
# # `rainfall`との相関係数を抽出して降順に表示
# rainfall_correlation = correlation_matrix['rainfall'].sort_values(ascending=False)

# print("--- 各特徴量とrainfallの相関係数 ---")
# print(rainfall_correlation)

# # --- 4. 相関行列の可視化 (ヒートマップ) ---
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
# plt.title('特徴量間の相関ヒートマップ', fontsize=16)
# plt.show()


# 特徴量とターゲット（目的変数）を分離
X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']


# テストデータも同じように準備 (ターゲット列は存在しない)
X_test = test_df


# ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint','humidity', 'cloud', 'sunshine', 'winddirection','windspeed']
# # 予測モデルの汎化性能を落とす可能性があるものを削除
# X = X.drop(['day', 'winddirection', 'windspeed'], axis=1)
# X_test = X_test.drop(['day','winddirection', 'windspeed'], axis=1)

# # 予測モデルの汎化性能を落とす可能性があるものを削除
# X = X.drop(['day', 'winddirection','windspeed', 'pressure', 'maxtemp', 'mintemp', 'dewpoint', 'temparature'], axis=1)
# X_test = X_test.drop(['day', 'winddirection','windspeed', 'pressure', 'maxtemp', 'mintemp', 'dewpoint', 'temparature'], axis=1)

# 予測モデルの汎化性能を落とす可能性があるものを削除
X = X.drop(['day'], axis=1)
X_test = X_test.drop(['day'], axis=1)


preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # 欠損値を中央値で補完
    ('scaler', StandardScaler())                    # データを標準化
])
# preprocessor = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')) # 欠損値を中央値で補完
# ])


# # 前処理と分類器を結合した最終的なパイプラインを作成します
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=15))
# ])

# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=3,
#     random_state=42
#     ))
# ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1
    ))
])


## 2. 学習データと検証データに分割
# stratify=y を指定することで、元のデータのExitedの比率を保ったまま分割する
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


print(f"学習データ数: {len(X_train)}")
print(f"検証データ数: {len(X_val)}")


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 1. 目標とする比率を設定
# SAMPLING_RATIO = 1.0  # -> 1:1 (元のコードと同じ)
# SAMPLING_RATIO = 2.0  # -> 多数派:少数派 = 2:1
SAMPLING_RATIO = 1.0  # -> 多数派:少数派 = 3:1
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

print(f"元のデータ数:\n{y_train.value_counts()}\n")

# 2. 0と1のインデックスを取得
class_0_indices = y_train[y_train == 0].index
class_1_indices = y_train[y_train == 1].index

# 3. 多数派と少数派を判断
if len(class_0_indices) > len(class_1_indices):
    majority_indices = class_0_indices
    minority_indices = class_1_indices
else:
    majority_indices = class_1_indices
    minority_indices = class_0_indices

# 4. 多数派からサンプリングする新しいサイズを計算
new_majority_size = int(len(minority_indices) * SAMPLING_RATIO)

# 安全装置：計算後のサイズが元の多数派のサンプル数を超えないようにする
new_majority_size = min(new_majority_size, len(majority_indices))

print(f"目標比率: {SAMPLING_RATIO}:1")
print(f"少数派の数: {len(minority_indices)}")
print(f"サンプリング後の多数派の数: {new_majority_size}\n")

# 5. 多数派のインデックスから、計算したサイズでランダムにサンプリング
np.random.seed(42) # 再現性のためのseed
downsampled_majority_indices = np.random.choice(
    majority_indices,
    size=new_majority_size, # <- ここを新しいサイズに変更
    replace=False
)

# 6. 少数派のインデックスと、間引いた多数派のインデックスを結合
resampled_indices = np.concatenate([downsampled_majority_indices, minority_indices])

# 7. 新しいインデックスを使って、X_trainとy_trainから新しいデータを作成
X_train = X_train.loc[resampled_indices]
y_train = y_train.loc[resampled_indices]


## 4. モデルの学習
print("\nモデルの学習を開始します...")
# 分割した学習データ(X_train, y_train)で学習
model_pipeline.fit(X_train, y_train)
print("モデルの学習が完了しました。")


## 5. モデル性能の評価

# --- 学習データでの性能評価 ---
print("\n--- 学習データでの性能 ---")
# 予測（確率とクラス）
y_train_pred_proba = model_pipeline.predict_proba(X_train)[:, 1]
y_train_pred = model_pipeline.predict(X_train)
# 評価指標を計算
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_pred_proba)
# 結果を表示
print(f"Accuracy (正解率): {train_accuracy:.4f}")
print(f"AUC スコア: {train_auc:.4f}")
print("\n分類レポート (学習データ):")
print(classification_report(y_train, y_train_pred))


# --- 検証データでの性能評価 ---
print("\n--- 検証データでの性能 ---")
# 予測（確率とクラス）
y_val_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
y_val_pred = model_pipeline.predict(X_val)
# 評価指標を計算
val_accuracy = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_pred_proba)
# 結果を表示
print(f"Accuracy (正解率): {val_accuracy:.4f}")
print(f"AUC スコア: {val_auc:.4f}")
print("\n分類レポート (検証データ):")
print(classification_report(y_val, y_val_pred))


test_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
test_pred = model_pipeline.predict(X_test)

submission_df['rainfall'] = test_pred_proba


submission_df.head(10)


submission_df.to_csv('submission.csv', index=False)
print('ok')




