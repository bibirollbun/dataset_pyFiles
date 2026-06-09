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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
  for filename in filenames:
    print(os.path.join(dirname, filename))


#csvファイルの読み込み
train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train.info()


test.isnull().sum()


#ランダムフォレストのインポート
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = train.corr()


# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)


# Show plot
plt.show()


# --- 温度差を追加 ---
train['temp_diff'] = train['maxtemp'] - train['mintemp']
test['temp_diff'] = test['maxtemp'] - test['mintemp']


#訓練データの特徴量選別(いらない特徴量を消す)
train_df=train.drop(['id','day','winddirection','rainfall'],axis=1) #id,day,rainfallはけしておｋ,#他は自由


y=train['rainfall']#予測値


#テストデータにも訓練と同様の処理をする
test_df=test.drop(['id','day','winddirection'],axis=1)#trainと合わせる,rainfallは消さない


#選別後の特徴量の確認
test_df.info()


#訓練データと検証データで分割(訓練データ80%, 検証データ20%)
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=12)


#モデルの初期化
model = RandomForestClassifier()


#モデルを訓練データで学習
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

# 検証データ（X_test）を使って予測を行う
y_pred = model.predict(X_test)


# 正解率を計算
accuracy = accuracy_score(y_test, y_pred)


# 計算した正解率を表示
print("\n--- モデルの性能評価 ---")
print(f"検証データに対する正解率 (Accuracy): {accuracy:.4f}")


#テストデータで予測
test_pred=model.predict(test_df)
test_pred=model.predict(test_df)


#予測結果
test_pred


#提出用のデータ保存
pred = pd.DataFrame({'id':test['id'],'rainfall': test_pred})

pred.to_csv('submit1.csv', index=False)

pred

