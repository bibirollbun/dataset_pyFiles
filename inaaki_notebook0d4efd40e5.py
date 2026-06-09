# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install japanize-matplotlib

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import japanize_matplotlib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#csvファイルの読み込み
train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



train.head()


train.info()


test.isnull().sum()


# 欠損値の確認
train.isnull().sum()


#ランダムフォレストのインポート
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = train.corr()


# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Title
plt.title("Feature Correlation Heatmap")

# Show plot
plt.show()


# rainfallと相関が高い特徴量のリスト
features_to_plot = ['cloud', 'sunshine', 'humidity']
# 各特徴量とrainfallの関係をグラフで表示
for feature in features_to_plot:
    plt.figure(figsize=(8, 6)) # グラフのサイズを設定
    
    # 箱ひげ図（Box Plot）を描画
    sns.boxplot(x='rainfall', y=feature, data=train)
    
    # グラフのタイトルとラベルを設定
    plt.title(f'降雨の有無と {feature} の関係', fontsize=16)
    plt.xlabel('降雨の有無 (0: No, 1: Yes)', fontsize=12)
    plt.ylabel(f'{feature} の値', fontsize=12)
    
    # グリッドを表示
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # グラフを表示
    plt.show()


#winddirectionの欠損値を中央値で補完
# 学習データの中央値を計算
median_winddirection = train['winddirection'].median()
# テストデータの欠損値を学習データの中央値で埋めます
test['winddirection'] = test['winddirection'].fillna(median_winddirection)



print("\n--- 補完後の欠損値 ---")
print("Test IsNull (after fillna):\n", test.isnull().sum())


#学習に 'winddirection' を含める
# 'id', 'day', 'maxtemp', 'mintemp' は相関や冗長性を考慮して削除
train_df = train.drop(['id', 'day','maxtemp', 'rainfall'], axis=1)
test_df = test.drop(['id', 'day','maxtemp'], axis=1)


# 目的変数を設定
y = train['rainfall']


print("\n--- 学習に使用する特徴量 ---")
print(train_df.columns)



#選別後の特徴量の確認
test_df.info()


X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=10)

# モデルの選択(ランダムフォレスト)
model = RandomForestClassifier()

# 学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)


test_pred=model.predict(test_df)


test_pred


# 「雨が降る(1)」と予測される確率を取得
y_pred_proba = model.predict_proba(X_test)[:, 1]

# AUCスコアを計算
auc_score = roc_auc_score(y_test, y_pred_proba)

# ROC曲線用のデータを計算
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# ROC曲線の描画
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate (偽陽性率)')
plt.ylabel('True Positive Rate (真陽性率)')
plt.title('ROC Curve (ROC曲線)')
plt.legend()
plt.grid()
plt.show()

print(f"AUCスコア: {auc_score:.4f}")


# テストデータに対して「雨が降る確率」を予測
predicted_probabilities = model.predict_proba(test_df)[:, 1]

# 提出用のDataFrameを作成
submission_df = pd.DataFrame({
    'id': test['id'],
    'rainfall': predicted_probabilities
})

# CSVファイルとして書き出し (index=Falseを忘れずに)
submission_df.to_csv('submission.csv', index=False)

print("提出用ファイル 'submission.csv' が作成されました。")
print("ファイルの中身 (先頭5行):")
print(submission_df.head())

