from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


DATA_PATH = Path('/kaggle/input/2025-stellar-temperature-challenge')

df_train = pd.read_csv(DATA_PATH / 'train.csv') # train.csvをpandasのデータフレームとして読み込む
df_test = pd.read_csv(DATA_PATH / 'test.csv') # test.csvをpandasのデータフレームとして読み込む
df_sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv') # 提出ファイルの例


df_train.head()


df_test.head()


df_sample_submission


FEATURES = ['kepmag', 'logg', 'mass']
target = 'teff'


X = df_train[FEATURES] # 列名がFEATURESの列のみを抽出
y = df_train[target] # 列名がtargetの列のみを抽出


X_test = df_test[FEATURES] # テストデータにはtargetの列は存在しない


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=29)


print('X_trainの行数と列数', X_train.shape)
print('X_validの行数と列数', X_valid.shape)


## LightGBM用にデータを変形
lgb_train = lgb.Dataset(X_train, y_train)

## パラメータを設定
params = {
    'objective': 'regression', # 回帰問題として指定
    'metric': 'mse', # 評価
    'seed' : 29
}

## モデルを訓練
model = lgb.train(
    params,
    lgb_train,
)


valid_pred = model.predict(X_valid) # 検証データに対する予測を作成


validation_score = mean_squared_error(y_valid, valid_pred)
print('検証データのスコア:',validation_score)


y_pred = model.predict(X_test)


test_ids = df_sample_submission['id']

df_submission = pd.DataFrame({
    "id": test_ids,
    "teff": y_pred
})


df_submission.to_csv("submission.csv", index=False)
df_submission.head()


df_valid = pd.DataFrame({
    'y_valid_pred': valid_pred,
    'y_valid_true': y_valid.to_numpy(),
})


df_valid

