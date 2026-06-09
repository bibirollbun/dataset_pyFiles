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


# 1. 念のためカレントディレクトリ（/kaggle/working）に移動
import os
os.chdir('/kaggle/working')

# 2. 強制的に上書きモード(-y)で、現在の場所に解凍
print("解凍を開始します...")
!7z x -y /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!7z x -y /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z
!7z x -y /kaggle/input/mercari-price-suggestion-challenge/sample_submission.csv.7z

# 3. 展開されたファイルがあるか確認
print("\n--- 現在のフォルダのファイル一覧 ---")
!ls -lh


import os
os.environ['OMP_NUM_THREADS'] = '4' # CPUを効率的に使う設定

import time
import pandas as pd
import numpy as np
from typing import List, Dict
from contextlib import contextmanager
from functools import partial
from operator import itemgetter

# 学習モデルを Ridge（線形回帰）に変更
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def on_field(f, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    
    with timer('fit_predict (Ridge)'):
        # solver="lsqr" を明示的に指定します
        # これにより、問題の cg(tol=...) を呼び出すのを防ぎ、エラーを回避します
        model = Ridge(alpha=3.0, solver="lsqr", max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        # 返り値が2次元の場合があるので、1次元に平坦化します
        if len(preds.shape) > 1:
            preds = preds.ravel()
        return preds



def preprocess_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # カテゴリ分割 (上位陣のテクニック)
    def split_cat(text):
        try: return text.split("/")
        except: return ("No Label", "No Label", "No Label")
    
    df['subcat_0'], df['subcat_1'], df['subcat_2'] = \
        zip(*df['category_name'].fillna('No/No/No').apply(lambda x: split_cat(x)))
    
    # テキストの結合（Ridge用）
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + 
                  df['name'] + ' ' + 
                  df['category_name'].fillna(''))
    
    return df


with timer('process train'):
    # パスから /kaggle/working/ を外して直接ファイル名にする
    train = pd.read_table('train.tsv') 
    train = train[train['price'] > 0].reset_index(drop=True)
    
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train_data, valid_data = train.iloc[train_ids], train.iloc[valid_ids]
    
    del train
    import gc
    gc.collect()

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(np.log1p(train_data['price'].values.reshape(-1, 1)))
    print("処理完了！")


# Ridge用のベクトル化設定
vectorizer = make_union(
    on_field('name', Tfidf(ngram_range=(1, 2), max_features=50000)),
    on_field('text', Tfidf(ngram_range=(1, 2), max_features=100000, token_pattern=r'\w+')),
    on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()),
)


vectorizer = make_union(
    on_field('name', Tfidf(max_features=100000, token_pattern=r'\w+')),
    on_field('text', Tfidf(max_features=100000, token_pattern=r'\w+', ngram_range=(1, 2))),
    on_field(['shipping', 'item_condition_id'],
             FunctionTransformer(to_records, validate=False), DictVectorizer()),
    n_jobs=4)

with timer('vectorizing train'):
    X_train = vectorizer.fit_transform(preprocess(train_data)).astype(np.float32)
    Xb_train = X_train.astype(bool).astype(np.float32) # これが必要です！
    
with timer('vectorizing valid'):
    X_valid = vectorizer.transform(preprocess(valid_data)).astype(np.float32)

print(f'X_train shape: {X_train.shape}')


Xb_train = X_train.astype(bool).astype(np.float32)
Xb_valid = X_valid.astype(bool).astype(np.float32)

# Ridgeは非常に速いので、これだけで十分です
y_pred_list = []
print("Model 1 (Binary) 学習中...")
y_pred_list.append(fit_predict([Xb_train, Xb_valid], y_train))

print("Model 2 (TF-IDF) 学習中...")
y_pred_list.append(fit_predict([X_train, X_valid], y_train))

y_pred = np.mean(y_pred_list, axis=0)
y_pred_out = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid_data['price'], y_pred_out))))


import gc
# 学習に使った大きな行列を削除
if 'X_train' in locals(): del X_train
if 'Xb_train' in locals(): del Xb_train
if 'train_data' in locals(): del train_data
gc.collect()


# Stage 2 の大きなデータを解凍
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip


import gc
import pandas as pd
import numpy as np

# --- STEP 1: 学習データの準備 (train_data を確実に用意する) ---
with timer('Full Process: Prepare y_train'):
    # y_train または train_data がメモリにない場合は読み込み直す
    if ('y_train' not in locals()) or ('train_data' not in locals()):
        print("学習データを読み込み中...")
        train = pd.read_table('train.tsv')
        train = train[train['price'] > 0].reset_index(drop=True)
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, _ = next(cv.split(train))
        train_data = train.iloc[train_ids]
        
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(np.log1p(train_data['price'].values.reshape(-1, 1)))
        del train
        gc.collect()

# --- STEP 2: 学習データのベクトル化 ---
with timer('Full Process: Vectorizing Train'):
    # X_train がない場合はベクトル化を実行
    if 'X_train' not in locals():
        print("学習データをベクトル化中...")
        X_train = vectorizer.fit_transform(preprocess(train_data)).astype(np.float32)
        Xb_train = X_train.astype(bool).astype(np.float32)
        # ベクトル化が終わったら train_data は消してOK
        del train_data
        gc.collect()

# --- STEP 3: 本番データ(Stage 2)の読み込みとベクトル化 ---
with timer('Full Process: Vectorizing Stage 2 Test'):
    print("本番データ(Stage 2)を読み込み中...")
    test = pd.read_table('test_stg2.tsv')
    test_ids = test['test_id']
    
    print(f'Stage 2 Test Rows: {len(test)}') # 3460725 と出るはず
    
    X_test = vectorizer.transform(preprocess(test)).astype(np.float32)
    Xb_test = X_test.astype(bool).astype(np.float32)
    
    del test
    gc.collect()



# --- STEP 4: 学習と予測 (Ridge + NN) ---
with timer('Full Process: Predict'):
    # --- 4-1: Ridge予測 ---
    y_test_pred_list = []
    print("Ridge Model 1 (Binary) 予測中...")
    y_test_pred_list.append(fit_predict([Xb_train, Xb_test], y_train))
    
    print("Ridge Model 2 (TF-IDF) 予測中...")
    y_test_pred_list.append(fit_predict([X_train, X_test], y_train))
    
    y_ridge_test_pred = np.mean(y_test_pred_list, axis=0)
    y_ridge_out = np.expm1(y_scaler.inverse_transform(y_ridge_test_pred.reshape(-1, 1))[:, 0])

    # --- 4-2: 軽量NN (MLP) の学習と予測 ---
    # ここで X_train が消える前に NN の学習を行います
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.preprocessing import MaxAbsScaler

    print("NN用のスケーリング中...")
    scaler = MaxAbsScaler()
    # NNはデータのスケールに敏感なためスケーリングが必須です
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.transform(X_test)

    print(f"NNモデル学習開始 (Input dim: {X_train_nn.shape[1]})...")
    model_nn = Sequential([
        Dense(256, input_dim=X_train_nn.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model_nn.compile(loss='mse', optimizer='adam')
    
    # メモリと時間の節約のため、大きなバッチサイズで2エポック学習
    model_nn.fit(X_train_nn, y_train, epochs=2, batch_size=2048, verbose=1)
    
    print("NN予測中...")
    nn_preds_log = model_nn.predict(X_test_nn, batch_size=2048)
    nn_preds = np.expm1(y_scaler.inverse_transform(nn_preds_log.reshape(-1, 1))[:, 0])

# --- STEP 4-3: 最終アンサンブル (ここが上位陣の肝) ---
with timer('Full Process: Ensemble'):
    # NN(70%) と Ridge(30%) をブレンド
    final_price = (nn_preds * 0.7) + (y_ridge_out * 0.3)

# --- STEP 5: CSV保存 ---
submission = pd.DataFrame({
    "test_id": test_ids,
    "price": final_price
})
submission.to_csv("submission.csv", index=False)
print("✅ アンサンブル完了！ submission.csv を保存しました。")

