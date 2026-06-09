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


#データを解凍
!apt-get install p7zip
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/sample_submission.csv.7z
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip


#最初の
from sklearn.linear_model import Ridge , LogisticRegression,Lasso
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error  


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

import warnings
warnings.filterwarnings(action="ignore")


#現在のディレクトりを確認
import os
print("現在の作業ディレクトリ:", os.getcwd())
print("ファイル一覧:")
for file in os.listdir('.'):
    if file.endswith(('.tsv', '.csv')):
        print(f"  {file}")


train_df = pd.read_csv('/kaggle/working/train.tsv', sep='\t')
test_df = pd.read_csv('/kaggle/working/test_stg2.tsv', sep='\t')
sample_df = pd.read_csv("/kaggle/working/sample_submission.csv")
    
print("✅ 'train.csv', 'test.csv', 'sample_submission.csv' の読み込みに成功しました。\n")


train_df.info(memory_usage="deep")


X = train_df[["category_name", "brand_name"]].fillna({'category_name': 'No_Category', 'brand_name': 'No_Brand'})
# ここで対数を取る
y = np.log1p(train_df["price"])
X_test = test_df[["category_name", "brand_name"]].fillna({'category_name': 'No_Category', 'brand_name': 'No_Brand'})


X


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)


from sklearn.linear_model import LinearRegression


# before vectorization (Count & TF-IDF) and encoding (ONE-HOT), memory need to be cleared
#import gc
#gc.collect()
# ===== 元のコード =====
# cnt_vec = CountVectorizer()
# X_name = cnt_vec.fit_transform(mercari_df['name'])

# ===== 新しいコード =====
cnt_vec = CountVectorizer(
    max_features=50000,  # 最大5万次元に制限
    min_df=10            # 10回未満出現の単語は無視
)
X_name = cnt_vec.fit_transform(train_df['name'])


print(f'✅ name vectorization shape: {X_name.shape}')
print(f'   (元: 207,968次元 → 新: {X_name.shape[1]}次元)')


#20分ぐらいかかる　メモリ20Gぐらい必要
tfidf_descp = TfidfVectorizer(
    max_features=10000,      # 50000 → 20000 (3-gram削除で十分)
    ngram_range=(1, 2),      # (1,3) → (1,2) に変更
    stop_words='english',
    min_df=3,                # 3回未満の単語は無視
    max_df=0.8               # 50%以上の文書に出現する単語は無視
)

# NaN値を空の文字列で埋めてからfit_transformを適用
X_descp = tfidf_descp.fit_transform(train_df['item_description'].fillna(''))

print('name vectorization shape:',X_name.shape)
print('item_description vectorization shape:',X_descp.shape)


# train_test_split によって分割されたインデックスを取得
train_indices = X_train.index
valid_indices = X_valid.index

# 全体から作成されたX_nameとX_descpをトレーニングセットと検証セットに分割
from scipy.sparse import hstack

X_name_train = X_name[train_indices]
X_name_valid = X_name[valid_indices]

X_descp_train = X_descp[train_indices]
X_descp_valid = X_descp[valid_indices]

print(f"X_name_train shape: {X_name_train.shape}")
print(f"X_descp_train shape: {X_descp_train.shape}")


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical features
categorical_features = ['category_name','brand_name']

# Create a column transformer to apply one-hot encoding to categorical features
# handle_unknown='ignore' will ignore categories not seen during training
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Create a pipeline that first preprocesses the data and then applies linear regression
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

# Fit the model
model.fit(X_train, y_train)


# カテゴリカル特徴量のワンホットエンコーディング
# preprocessor はすでに定義されており、OneHotEncoderを含んでいます
X_cat_train_encoded = preprocessor.fit_transform(X_train)
X_cat_valid_encoded = preprocessor.transform(X_valid) # 検証セットはtransformのみ

print(f"X_cat_train_encoded shape: {X_cat_train_encoded.shape}")


# すべての疎な特徴量を結合
X_train_combined = hstack([X_cat_train_encoded, X_name_train, X_descp_train]).tocsr()
X_valid_combined = hstack([X_cat_valid_encoded, X_name_valid, X_descp_valid]).tocsr()

print(f"X_train_combined shape: {X_train_combined.shape}")
print(f"X_valid_combined shape: {X_valid_combined.shape}")


# Ridge回帰モデルの学習
from sklearn.linear_model import Ridge

# 以下の設定を試してみてください。ただし、アップグレードがベストです。
# tolを非常に小さく設定すると、cg()の内部処理が変わる可能性があります。
ridge_model = Ridge(
    solver='lsqr',
    alpha=1.0,
    random_state=0,
)
ridge_model.fit(X_train_combined, y_train)



# 予測
pred_train_combined = ridge_model.predict(X_train_combined)
pred_valid_combined = ridge_model.predict(X_valid_combined)

# 評価 (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_train = np.sqrt(mean_squared_error(y_train, pred_train_combined))
rmse_valid = np.sqrt(mean_squared_error(y_valid, pred_valid_combined))

print(f"Combined features - Train RMSE: {rmse_train}")
print(f"Combined features - Valid RMSE: {rmse_valid}")


# テストデータに対する予測の準備

# X_test はすでに定義済み

# カテゴリカル特徴量のワンホットエンコーディング (fit_transformではなくtransformを使用)
X_cat_test_encoded = preprocessor.transform(X_test)

# nameとitem_descriptionのベクトル化 (fit_transformではなくtransformを使用)
X_name_test = cnt_vec.transform(test_df['name'])
X_descp_test = tfidf_descp.transform(test_df['item_description'].fillna(''))

# すべての疎な特徴量を結合
X_test_combined = hstack([X_cat_test_encoded, X_name_test, X_descp_test]).tocsr()

print(f"X_test_combined shape: {X_test_combined.shape}")

# テストセットでの予測
pred_test_combined = ridge_model.predict(X_test_combined)

# 対数変換を元に戻す (np.expm1 を使用)
final_predictions = np.expm1(pred_test_combined)

# 負の値にならないようにクリップ
final_predictions = np.maximum(0, final_predictions)


# サブミッションファイルの作成
sub_df_combined = pd.DataFrame({
    "test_id": test_df["test_id"],
    "price": final_predictions
})

sub_df_combined.to_csv("submission.csv", index=False)

display(sub_df_combined.head())

