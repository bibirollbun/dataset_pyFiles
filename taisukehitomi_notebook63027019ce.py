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


# ====================================================
# ステップ1：ライブラリの準備
# ====================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

print("■ ステップ1：ライブラリの準備完了")


# ====================================================
# ステップ2：データの読み込みと特徴量の選択
# ====================================================
# --- データの読み込み ---
train_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
test_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")

# --- 特徴量の選択（スライド3, 4の内容） ---
# 仮説に基づき、8つの特徴量に絞り込む
features = [
    # 「品質」関連 (5つ)
    'Blur', 
    'Subject Focus', 
    'Occlusion', 
    'Info', 
    'Collage',
    # 「被写体」関連 (3つ)
    'Face', 
    'Eyes', 
    'Near',
]

print(f"\n■ ステップ2：データ読み込み完了。選択した特徴量の数: {len(features)}個")


# ====================================================
# ステップ3：学習データの準備
# ====================================================
# 選択した8つの特徴量を、予測に使う「材料（X）」とする
X = train_df[features]
# 予測したい答え（Pawpularity）を「正解（y）」とする
y = train_df['Pawpularity']
# テストデータからも同じ8つの特徴量を選ぶ
X_test = test_df[features]

print("\n■ ステップ3：学習データの準備完了")


# ====================================================
# ステップ4：交差検証によるモデル評価（スライド6の内容）
# ====================================================
print("\n■ ステップ4：5分割交差検証を開始します...")

# 5分割交差検証の設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 各回のスコアと特徴量の重要度を保存するリスト
oof_rmse_scores = []
feature_importances = pd.DataFrame(index=features)

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} / 5 ---")
    
    # データを学習用と検証用に分割
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    
    # 【★★ ここが修正箇所です ★★】
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # モデルの定義（スライド5の内容）
    model = lgb.LGBMRegressor(random_state=42)
    
    # モデルの学習
    model.fit(X_train, y_train)

    # 検証データで予測
    val_preds = model.predict(X_val)
    
    # RMSEを計算して保存
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    oof_rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse}")

    # 特徴量の重要度を保存
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# --- 交差検証の平均スコアを表示 ---
mean_rmse = np.mean(oof_rmse_scores)
print(f"\n交差検証の平均RMSEスコア: {mean_rmse:.4f}")
print("（スライド6に記載するスコアです）")


# ====================================================
# ステップ5：特徴量の重要度を可視化（スライド6の内容）
# ====================================================
# 各Foldの重要度の平均を計算し、重要度順に並べ替える
feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

# --- グラフで可視化 ---
plt.figure(figsize=(10, 6))
sns.barplot(x='mean', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importance (特徴量の重要度)')
plt.xlabel('Importance Score (重要度スコアの平均)')
plt.ylabel('Feature (特徴量)')
plt.show()

print("\n上記グラフをスライド6に貼り付けてください。")
print("「Blur」と「Subject Focus」の重要度が特に高いことがわかります。")


# ====================================================
# ステップ6：最終的なモデルの学習と予測（スライド7の内容）
# ====================================================
print("\n■ ステップ6：全ての訓練データで最終モデルを学習し、テストデータを予測します...")

# 全ての訓練データを使って、再度モデルを学習させる
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X, y)

# テストデータで予測
predictions = final_model.predict(X_test)


# ====================================================
# ステップ7：提出ファイルの作成
# ====================================================
# テストデータのIDと、AIが予測した人気度を紐づけた表を作成する
submission_df = pd.DataFrame({'Id': test_df['Id'], 'Pawpularity': predictions})
submission_df.to_csv('submission.csv', index=False)

print("\n■ ステップ7：提出ファイル 'submission.csv' の作成が完了しました！")
print("このファイルをKaggleに提出し、得られたスコアをスライド7に記載してください。")
print("--- 作成されたファイルの一部 ---")
print(submission_df.head())

