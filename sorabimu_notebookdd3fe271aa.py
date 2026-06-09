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
# from imblearn.under_sampling import RandomUnderSampler # アンダーサンプリングのライブラリをインポート


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


train_df.isnull().sum()


test_df.isnull().sum()


train_df.head(10)


test_df.head(10)


submission_df.head()


# 特徴量とターゲット（目的変数）を分離
X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']


# テストデータも同じように準備 (ターゲット列は存在しない)
X_test = test_df


print(X)


print(X_test)


# 予測モデルの汎化性能を落とす可能性があるものを削除
X = X.drop(['day'], axis=1)
X_test = X_test.drop(['day'], axis=1)


sns.set(style = "darkgrid")

sns.countplot(data = train_df, x = "rainfall", palette = "mako")
plt.tight_layout()
plt.show()


# 処理内容　「欠損値補完」→「標準化」
# preprocessor = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),  # 欠損値を中央値で補完
#     ('scaler', StandardScaler())                    # データを標準化
# ])
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) # 欠損値を中央値で補完
])


# 前処理と分類器を結合した最終的なパイプラインを作成します
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=15))
])


## 2. 学習データと検証データに分割
# stratify=y を指定することで、元のデータのExitedの比率を保ったまま分割する
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


print(f"学習データ数: {len(X_train)}")
print(f"検証データ数: {len(X_val)}")


# --- train_test_splitでデータを分割した後の学習データ (X_train, y_train) を使います ---
# X_train, y_train が既にあると仮定します。

# 1. y_trainを元に、0と1のインデックスを取得
class_0_indices = y_train[y_train == 0].index
class_1_indices = y_train[y_train == 1].index

# 2. どちらが多数派で、どちらが少数派かを判断
if len(class_0_indices) > len(class_1_indices):
    majority_indices = class_0_indices
    minority_indices = class_1_indices
else:
    majority_indices = class_1_indices
    minority_indices = class_0_indices

# 3. 多数派のインデックスから、少数派の数だけランダムにサンプリング
# random_stateを固定することで、毎回同じ結果になります
np.random.seed(42) # 再現性のためのseed
downsampled_majority_indices = np.random.choice(
    majority_indices, 
    size=len(minority_indices), 
    replace=False
)

# 4. 少数派のインデックスと、間引いた多数派のインデックスを結合
resampled_indices = np.concatenate([downsampled_majority_indices, minority_indices])

# 5. 新しいインデックスを使って、X_trainとy_trainから新しいデータを作成
X_train_resampled = X_train.loc[resampled_indices]
y_train_resampled = y_train.loc[resampled_indices]


len(X_train_resampled)


## 4. モデルの学習
# print("\nモデルの学習を開始します...")
# # 分割した学習データ(X_train, y_train)で学習
# model_pipeline.fit(X_train, y_train)
# print("モデルの学習が完了しました。")

print("\nモデルの学習を開始します...")
# 分割した学習データ(X_train, y_train)で学習
model_pipeline.fit(X_train_resampled, y_train_resampled)
print("モデルの学習が完了しました。")


# ## 5. モデル性能の評価

# # --- 学習データでの性能評価 ---
# print("\n--- 学習データでの性能 ---")
# # 予測（確率とクラス）
# y_train_pred_proba = model_pipeline.predict_proba(X_train)[:, 1]
# y_train_pred = model_pipeline.predict(X_train)
# # 評価指標を計算
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_auc = roc_auc_score(y_train, y_train_pred_proba)
# # 結果を表示
# print(f"Accuracy (正解率): {train_accuracy:.4f}")
# print(f"AUC スコア: {train_auc:.4f}")
# print("\n分類レポート (学習データ):")
# print(classification_report(y_train, y_train_pred))

## 5. モデル性能の評価

# --- 学習データでの性能評価 ---
print("\n--- 学習データでの性能 ---")
# 予測（確率とクラス）
y_train_pred_proba = model_pipeline.predict_proba(X_train_resampled)[:, 1]
y_train_pred = model_pipeline.predict(X_train_resampled)
# 評価指標を計算
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
train_auc = roc_auc_score(y_train_resampled, y_train_pred_proba)
# 結果を表示
print(f"Accuracy (正解率): {train_accuracy:.4f}")
print(f"AUC スコア: {train_auc:.4f}")
print("\n分類レポート (学習データ):")
print(classification_report(y_train_resampled, y_train_pred))


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

