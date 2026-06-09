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


# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')

# 最初の5行を表示
print(df.head())


from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# データコピー
df_encoded = df.copy()

# 1. 文字列カテゴリを 'missing' で補完し、数値に変換
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].fillna('missing')
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

# 2. 数値型の欠損を平均で補完
for col in df_encoded.columns:
    if df_encoded[col].dtype in ['float64', 'float32', 'int64']:
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())

# 3. 'class'と'id'を除いた特徴量と目的変数に分ける
X = df_encoded.drop(columns=['class', 'id'], errors='ignore')  # errors='ignore' でidがなくてもOK
y = df_encoded['class']

# 4. 相互情報量を計算
mi_scores = mutual_info_classif(X, y, discrete_features=True)

# 5. 結果の表示
mi_df = pd.DataFrame({'特徴量': X.columns, '相互情報量': mi_scores})
mi_df = mi_df.sort_values(by='相互情報量', ascending=False)
print(mi_df)


print(df_encoded.isnull().sum())


# 1. ユニーク値が10個以上の列をリストアップ
high_unique_cols = [col for col in df_encoded.columns if df_encoded[col].nunique() > 2]

print("ユニーク値が非常に多い列:", high_unique_cols)

if high_unique_cols:
    mask = df_encoded[high_unique_cols].apply(lambda col: col.isin(col.unique()))
    rows_to_drop = mask.any(axis=1)
    
    # 3. 削除
    df_encoded = df_encoded[~rows_to_drop].reset_index(drop=True)
    
    print(f"削除後のデータ件数: {len(df_encoded)}")
else:
    print("ユニーク値10個超の列はありません。")



# 1. 削除対象の列をリストアップ（ユニーク値が15個超）
high_unique_cols = [col for col in df_encoded.columns if df_encoded[col].nunique() > 15]

# 2. 削除対象を表示
print("削除対象の列（ユニーク値 > 15）:")
print(high_unique_cols)


# データセットの行数と列数を確認
print(f"{df.shape[0]}行, {df.shape[1]}列")
# データセットの行数と列数を確認
print(f"{df.shape[0]}行, {df.shape[1]}列")

# データ全体の情報確認
print(df.info())


# 毒性のあるものないものの表記(ほかの特徴量も見る必要あり)
sns.countplot(x='class', hue='class', data=df)
plt.title('Mushroom Class by class')
plt.show()


valid_habitats = [
'd', 'g', 'l', 'm', 'h', 'w', 'p', 'u', 'e', 's', 
'n', 't', 'r', 'y', 'a', 'k', 'c', 'b', 'o', 'f', 
'i', 'x', 'z'
]
# 'habitat'列がvalid_habitatsリスト内の値である行だけを保持
df_cleaned = df[df['habitat'].isin(valid_habitats)]

# 処理後のデータ件数を確認
print(f"元のデータ数: {len(df)}")
print(f"クリーンなデータ数: {len(df_cleaned)}")

# クリーンになったか、再度value_counts()で確認
print("\nクリーンなデータのhabitat列:")
print(df_cleaned['habitat'].value_counts())


# 元のデータフレーム df を使います
suspicious_columns = []

# 文字列(object)型の列だけをループでチェック
for column in df.select_dtypes(include=['object']).columns:
    if column == 'class':  # 'class'列はチェック対象外
        continue
    
    # 欠損値を除いたユニークな値を取得
    unique_values = df[column].dropna().unique()
    
    # ユニークな値の中に、2文字以上の文字列があるかチェック
    for value in unique_values:
        if isinstance(value, str) and len(value) > 1:
            suspicious_columns.append(column)
            break  # 怪しい列を見つけたら、次の列のチェックに移る

print("1文字ではない値が混入している可能性のある列:")
print(suspicious_columns)


df = df.drop(columns=['habitat', 'has-ring','veil-color'], errors='ignore')
print(df.columns)


df=print(df.isnull().sum())


import numpy as np

def fillna_with_mean(df):
    for col in df.columns:
        # float型かどうかを厳密に判定
        if np.issubdtype(df[col].dtype, np.floating):
            # 文字列のNaNなどをnp.nanに変換しておく（念のため）
            df[col] = df[col].replace(['NaN', 'nan', '', None], np.nan)
            
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
    return df


print(df.head(10))
print(f"{df.shape[0]}行, {df.shape[1]}列")
print(df.info())


import pandas as pd
def replace_nan_high_missing(df, threshold=0, replacement='nn'):
    # 欠損率0%以上(すべて)の列を抽出
    missing_ratio = df.isnull().mean()
    cols_to_replace = missing_ratio[missing_ratio >= threshold].index
    # その列のNaNを'replacement'で置換
    df.loc[:, cols_to_replace] = df.loc[:, cols_to_replace].fillna(replacement)

    return df

# 使用例
df = replace_nan_high_missing(df, threshold=0, replacement='nn')


print(df.head(10))
print(f"{df.shape[0]}行, {df.shape[1]}列")
print(df.info())


from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# コピーして処理用に使う
df_encoded = df.copy()

# 1. class をカテゴリ → 数値変換（e/p はそのままラベルとして扱う）
df_encoded['class'] = df_encoded['class'].astype('category').cat.codes
# ※ e → 0, p → 1（内部的にはこの順）

# 2. 他のカテゴリ列も必要に応じて数値変換（ここでは文字列列のみ対象）
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

# 3. 特徴量と目的変数に分ける
X = df.drop(columns=['class', 'id'])  # idを除外
y = df_encoded['class']                 # 目的変数：class（0=e, 1=p）

# 4. 相互情報量を計算
mi_scores = mutual_info_classif(X, y, discrete_features=True)

# 5. 結果の整形と表示
mi_df = pd.DataFrame({
    '特徴量': X.columns,
    '相互情報量': mi_scores
}).sort_values(by='相互情報量', ascending=False)

print(mi_df)


## def encode_class_column(df):
    df['class'] = df['class'].map({'e': 1, 'p': 0})
    return df.head()
print(encode_class_column(df))


# 前処理(機械が学習できる形にする)
# 'class'列を目的変数yに、それ以外の列を特徴量Xに分割 
X = df.drop('class', axis=1)
y = df['class']

# yの'p'を1に、'e'を0に変換
y = y.map({'p': 1, 'e': 0})
print(y.head())

# 特徴量XをOne-Hot Encoding
# この一行だけで全ての文字の列自動でOne-Hot Encodingしてくれる
X_encoded = pd.get_dummies(X)
print(f"元の列数: {X.shape[1]} 変換後の列数: {X_encoded.shape[1]}")

print(X_encoded.head())

