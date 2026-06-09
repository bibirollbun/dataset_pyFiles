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

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# データの確認(頭5行)
train.head()


# データの要約
train.info()


# 欠損値の確認
train.isnull().sum()


correlation_matrix = train.corr()

# ヒートマップの描画
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Feature Correlation Heatmap(特徴量相関ヒートマップ)")

plt.show()



# 目的変数（雨が降ったかどうか）
y = train['rainfall']

# 使用する特徴量を明示的に指定（train用）
features = ['sunshine','humidity','cloud','maxtemp', 'temparature', 'mintemp', 'pressure', 'dewpoint']
train_df = train[features]

# testデータにも同じ特徴量を適用
test_df = test[features]



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


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
print(len(train))
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(len(test))




