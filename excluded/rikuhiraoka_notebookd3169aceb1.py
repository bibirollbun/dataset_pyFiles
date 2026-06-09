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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
print(df.head())


print(f"{df.shape[0]}行, {df.shape[1]}列")


print(df.info())


print(df.isnull().sum())


print(df.describe(include='object'))


!pip install japanize_matplotlib
import japanize_matplotlib
# かさの直径（cap-diameter）と毒性の関係を可視化
sns.histplot(data=df, x='cap-diameter', hue='class', kde=True, bins=50)
plt.title('かさの直径と毒性の分布')
plt.show()


# 生息地と毒性の関係
sns.countplot(data=df, x='habitat', hue='class')


df['habitat'].value_counts()


df['habitat'].unique()


drop = ['veil-type','spore-print-color','stem-root','veil-color','stem-surface','gill-spacing']
df = df.drop(drop, axis=1)


print(df.columns)


print(df.isnull().sum())


df_backup = df.copy()
print(df_backup.shape)


valid_habitats = [
    'd', 'g', 'l', 'm', 'h', 'w', 'p', 'u', 'e', 's', 
    'n', 't', 'r', 'y', 'a', 'k', 'c', 'b', 'o', 'f', 
    'i', 'x', 'z'
]
df = df[df['habitat'].isin(valid_habitats)]


df['habitat'].value_counts()


df['cap-surface'].value_counts()


# 文字列で、かつ長さが1より大きい行を特定
# 文字列にすべて変更して1文字
condition = df['cap-surface'].astype(str).str.len() > 1
df = df[~condition]
df['cap-surface'].value_counts()


print(df.isnull().sum())


df['cap-color'].value_counts()


# 文字列で、かつ長さが1より大きい行を特定
# 文字列にすべて変更して1文字
condition = df['cap-color'].astype(str).str.len() > 1
df = df[~condition]
df['cap-color'].value_counts()


print(df.isnull().sum())


df['cap-shape'].value_counts()


# 文字列で、かつ長さが1より大きい行を特定
# 文字列にすべて変更して1文字
condition = df['cap-shape'].astype(str).str.len() > 1
df = df[~condition]
df['cap-shape'].value_counts()


print(df.isnull().sum())


df['does-bruise-or-bleed'].value_counts()


condition = df['does-bruise-or-bleed'].astype(str).str.len() > 1
df = df[~condition]
df['does-bruise-or-bleed'].value_counts()


print(df.isnull().sum())


df['gill-attachment'].value_counts()


condition = df['gill-attachment'].astype(str).str.len() > 1
df = df[~condition]
df['gill-attachment'].value_counts()


print(df.isnull().sum())


df['gill-color'].value_counts()


condition = df['gill-color'].astype(str).str.len() > 1
df = df[~condition]
df['gill-color'].value_counts()


values_to_delete = ['4', '5']
condition = df['gill-color'].isin(values_to_delete)
df = df[~condition]
df['gill-color'].value_counts()


print(df.isnull().sum())


df['stem-color'].value_counts()


condition = df['stem-color'].astype(str).str.len() > 1
df = df[~condition]
df['stem-color'].value_counts()


print(df.isnull().sum())


df['has-ring'].value_counts()


df['ring-type'].value_counts()


values_to_delete = ['4', '1']
condition = df['ring-type'].isin(values_to_delete)
df = df[~condition]
df['ring-type'].value_counts()


print(df.isnull().sum())


df['season'].value_counts()


print(df.info())


# 表示するグラフの特徴量まとめ
features_to_plot = [
    'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 
    'does-bruise-or-bleed', 'gill-attachment', 'gill-color', 
    'stem-height', 'stem-width', 'stem-color', 'has-ring', 
    'ring-type', 'habitat', 'season'
]

# forループで各特徴量のグラフを順番に作成
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    
    if df[feature].dtype == 'object':
        sns.countplot(data=df, x=feature, hue='class', palette='viridis')
        plt.title(f'{feature} と毒性の関係')
    else:
        # 数値データの場合: ヒストグラム
        sns.histplot(data=df, x=feature, hue='class', kde=True, palette='plasma')
        plt.title(f'{feature} と毒性の関係')
        
    plt.show() # グラフを1つずつ表示


print(df.isnull().sum())


print(df.describe())


print(df.describe(include='object'))


import pandas as pd

# テストデータを読み込む
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')

# 提出に必要なidを別の変数に保存しておく
test_ids = test_df['id']

# 予測に使う特徴量（id列以外）を準備
X_to_submit = test_df.drop('id', axis=1)


df = df.drop('id', axis=1, errors='ignore') 


X = df.drop('class', axis=1)
y = df['class']


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 数値データと文字データの列名をリストアップ
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 3. 前処理パイプラインを作成
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. ColumnTransformerで上記2つの処理を統合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef

# LightGBMをインポート
import lightgbm as lgb

# モデル部分を差し替える
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42)) # ここを変更！
])

# 再学習と評価
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"LightGBMのMCCスコア: {mcc:.4f}")


# 学習済みパイプラインを使って予測を実行
submission_predictions = model_pipeline.predict(X_to_submit)


# 提出用のデータフレームを作成
submission_df = pd.DataFrame({
    'id': test_ids,
    'class': submission_predictions
})

# CSVファイルとして保存
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print(submission_df.head())

