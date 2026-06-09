# train.csvをdf_trainのデータフレームとして読み込み

from pathlib import Path
import pandas as pd

DATA_PATH = Path('/kaggle/input/2025-stellar-temperature-challenge')

df_train = pd.read_csv(DATA_PATH / 'train.csv')
df_test = pd.read_csv(DATA_PATH / 'test.csv')

print(df_train.shape)


# チュートリアル1と同様に列を限定する

FEATURES = ['kepmag', 'logg', 'mass']
target = 'teff'
X = df_train[FEATURES] # 列名がFEATURESの列のみを抽出
y = df_train[target] # 列名がtargetの列のみを抽出

# テストデータに対しても列の抽出を行う
X_test = df_test[FEATURES] # テストデータにはtargetの列は存在しない


X


y


from sklearn.model_selection import train_test_split

# _trainが訓練データ、_validが検証データになる
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=29)


import lightgbm as lgb

## LightGBM用にデータを変形
lgb_train = lgb.Dataset(X_train, y_train)

## パラメータを設定
params = {
    'objective': 'regression', # 回帰問題として指定
    'metric': 'mse', # 評価
    'seed' : 29
}

## モデルを学習させる
model = lgb.train(
    params,
    lgb_train,
)


valid_pred = model.predict(X_valid)

print(valid_pred)


from sklearn.metrics import mean_squared_error

validation_score = mean_squared_error(y_valid, valid_pred)
print('検証データのスコア:',validation_score)


# X_testは上のセルで得られています
# テストデータにモデルを適用し、予測値y_predを作成

y_pred = model.predict(X_test)

print(y_pred)


df_sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv')


df_sample_submission


# df_sample_submissionからid列のみをtest_idsとして取ってくる
test_ids = df_sample_submission['id']

# df_submisisonを新たに作成 id列はtest_ids teff列は予測値y_pred を代入
df_submission = pd.DataFrame({
    "id": test_ids,
    "teff": y_pred
})


df_submission


# CSVファイルとして書き出す
df_submission.to_csv("submission.csv", index=False)




