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
# ステップ1：必要な道具（ライブラリ）を準備する
# ====================================================
import pandas as pd
import lightgbm as lgb # LightGBMという、性能の良いAIモデルのライブラリ

print("ステップ1：ライブラリの準備完了")


# ====================================================
# ステップ2：データを読み込む
# ====================================================
# 学習用のデータ（答えが書いてあるデータ）を読み込む
train_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
# テスト用のデータ（これから予測したいデータ）を読み込む
test_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")

print("ステップ2：データの読み込み完了")
print("--- 学習データの一部 ---")
print(train_df.head())


# ====================================================
# ステップ3：AIに学習させる「材料」と「答え」を決める
# ====================================================
# これら12個のメタデータを、予測に使う「材料（特徴量）」とする
features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

# 学習用の材料 (X_train)
X_train = train_df[features]
# 学習用の答え (y_train)
y_train = train_df['Pawpularity']

# テスト用の材料 (X_test)
X_test = test_df[features]

print("\nステップ3：材料と答えの準備完了")
print("--- 学習に使う材料（X_train）の一部 ---")
print(X_train.head())


# ====================================================
# ステップ4：AIモデルを学習させる
# ====================================================
# LightGBMモデル（AI）の器を用意する
# random_state=42 は、毎回同じ結果にするためのおまじない
model = lgb.LGBMRegressor(random_state=42)

# AIに材料(X_train)と答え(y_train)のセットを見せて、パターンを学習させる
print("\nステップ4：AIの学習を開始します...")
model.fit(X_train, y_train)
print("AIの学習が完了しました！")


# ====================================================
# ステップ5：学習したAIで「テストデータ」を予測する
# ====================================================
# 賢くなったAIに、答えのわからないテスト用の材料(X_test)を見せて、人気度を予測させる
print("\nステップ5：テストデータの人気度を予測します...")
predictions = model.predict(X_test)
print("予測が完了しました！")


# ====================================================
# ステップ6：Kaggleに提出するファイルを作成する
# ====================================================
# テストデータのIDと、AIが予測した人気度を紐づけた表を作成する
submission_df = pd.DataFrame({'Id': test_df['Id'], 'Pawpularity': predictions})

# 'submission.csv' という名前でファイルに保存する
# index=False は、余計な行番号を保存しないための設定
submission_df.to_csv('submission.csv', index=False)

print("\nステップ6：提出ファイル 'submission.csv' の作成が完了しました！")
print("画面右側の 'Output' セクションからダウンロードまたは提出できます。")
print("--- 作成されたファイルの一部 ---")
print(submission_df.head())


# ====================================================
# ステップ7：モデルがどの特徴量を重要視したかを知る
# ====================================================
# グラフ表示に使うライブラリを準備する
import matplotlib.pyplot as plt
import seaborn as sns

print("ステップ7：特徴量の重要度を計算・表示します...")

# 学習済みモデルから、特徴量の重要度スコアを取得する
feature_importances = model.feature_importances_

# 特徴量の名前とスコアをセットにした表を作成する
importance_df = pd.DataFrame({
    'feature': X_train.columns,  # 特徴量の名前
    'importance': feature_importances  # 重要度のスコア
})

# 重要度スコアが高い順に並べ替える
importance_df = importance_df.sort_values(by='importance', ascending=False)


# --- 結果をグラフで可視化する ---
plt.figure(figsize=(10, 8)) # グラフのサイズを指定
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance (特徴量の重要度)')
plt.xlabel('Importance Score (重要度スコア)')
plt.ylabel('Feature (特徴量)')
plt.show()


# --- 結果を表で表示する ---
print("\n--- 特徴量の重要度（スコアが高いほど重要）---")
print(importance_df)


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


# ====================================================
# ライブラリとデータの準備
# ====================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# train.csvを読み込む
try:
    train_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
except FileNotFoundError:
    print("エラー: /kaggle/input/petfinder-my-pawpularity-contest/train.csv が見つかりません。")
    print("ノートブックの右側にある「+ Add Data」から、コンペのデータを追加してください。")
    # この後の処理を続行しないように、ここで処理を中断します。
    # raise SystemExit

# 分析対象の12個の特徴量リスト
features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

print("■ 全12特徴量について、有無によるPawpularity平均スコアを比較します。\n")


# ====================================================
# 各特徴量でループして、結果を計算・可視化
# ====================================================
for feature in features:
    
    # --- 1. 平均スコアを計算 ---
    # 特徴量の有無(0 or 1)でグループ分けし、Pawpularityの平均値を計算
    average_scores = train_df.groupby(feature)['Pawpularity'].mean().round(2) # 小数点第2位で四捨五入

    # --- 2. 計算結果を表形式で表示 ---
    print(f"--- 特徴量: {feature} ---")
    print("  なし(0) の平均スコア:", average_scores.get(0, "N/A")) # .get(0)でキーが存在しない場合のエラーを防ぐ
    print("  あり(1) の平均スコア:", average_scores.get(1, "N/A"))
    
    # 差を計算して表示
    if 0 in average_scores and 1 in average_scores:
        score_diff = abs(average_scores[1] - average_scores[0])
        print(f"  スコアの差: {score_diff:.2f}")
    
    # --- 3. 結果を棒グラフで可視化 ---
    plt.figure(figsize=(6, 4)) # グラフのサイズを少し小さめに設定
    sns.barplot(x=average_scores.index, y=average_scores.values)
    plt.title(f'Pawpularity by "{feature}"') # グラフのタイトル
    plt.ylabel('Average Pawpularity Score') # Y軸のラベル
    plt.xlabel(f'"{feature}" (0: なし, 1: あり)') # X軸のラベル
    plt.xticks([0, 1]) # X軸の目盛りを0と1に固定
    plt.ylim(0, 50) # Y軸の範囲を0から50に固定して比較しやすくする
    plt.show() # グラフを表示
    
    print("-" * 40) # 区切り線

