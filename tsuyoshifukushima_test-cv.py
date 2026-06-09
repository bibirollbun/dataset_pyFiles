# 使用モジュールをインポート
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from sklearn.linear_model import Ridge
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, mean_squared_error, mean_squared_log_error


# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


############################################
# 各自がアップロードしたデータ名を代入：
data_name = 'playground-series-s4e4'
############################################
train_path = f'/kaggle/input/{data_name}/train.csv'
test_path = f'/kaggle/input/{data_name}/test.csv'


# --- モデルの定義 ---
# xgboost
class Model1Xgb:
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, num_round):
        params = {
            'objective': 'reg:squarederror', 
            'verbosity': 1, 
            'random_state': 71, 
            'eval_metric': 'rmse',
            'max_depth' : 10
        }
        num_round = num_round
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(
            params,
            dtrain,
            num_round,
            evals=watchlist,
            early_stopping_rounds=20
        )

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# LightGBM
class Model1Lgb:
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, num_round):
        params = {
            'objective': 'regression',
            'seed': 71, 
            'verbose': 0,
            'metrics': 'rmse',
            'max_depth' : 30
        }
        num_round = num_round

        # 特徴量と目的変数をlightgbmのデータ構造に変換する
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        self.model = lgb.train(params, lgb_train, num_boost_round=num_round,
                        valid_names=['train', 'valid'], 
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=30, verbose=True), # early_stopping用コールバック関数
                            lgb.log_evaluation(1)
                        ] # コマンドライン出力用コールバック関数
                    )

    def predict(self, x):
        pred = self.model.predict(x)
        return pred


# Catboost
class Model1Cat:
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, num_round):
        # --- ハイパーパラメータ ---
        params = {
            "loss_function": "RMSE",        # 回帰なので RMSE
            "eval_metric":  "RMSE",
            "random_seed":  71,
            "verbose":      True,          # False→学習中は静か / True→1iter毎にログ
            "iterations":   num_round,     # LightGBM の num_round に相当
            "early_stopping_rounds": 30,    # 10 iter 悪化で打ち切り
            'max_depth' : 10
        }

        # --- モデルをインスタンス化して学習 ---
        # CatBoost では fit() に eval_set を渡すだけで early‑stopping 可
        self.model = cb.CatBoostRegressor(**params)
        self.model.fit(
            tr_x, tr_y,
            eval_set=(va_x, va_y),          # 検証データ
            use_best_model=True,            # 検証スコア最良のモデルを保持
            verbose=False                   # True にすると 1 iter ごとにログ
        )

    def predict(self, x):
        pred = self.model.predict(x)
        return pred


# ---------------------------------
# データ等の準備
# ----------------------------------
# train_xは学習データ、train_yは目的変数、test_xはテストデータ
# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print(train)
print(test)


# 列の特定
ID_COL = 'id'
OBJECT_COL = 'Rings'


# --- 前処理 ---
# 1. カテゴリ変数の分離
all_data = pd.concat([train.drop(OBJECT_COL, axis=1), test], axis=0)
cat_cols = all_data.select_dtypes(include='object').columns

# 2. 欠損補完（カテゴリ変数を 'NA'）
for col in cat_cols:
    all_data[col] = all_data[col].fillna('NA')

# 3. Label Encoding（train+test 全体で fit）
for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# 4. データを train/test に分割しなおす
train_x = all_data.iloc[:len(train), :].copy()
train_x = train_x.drop("id", axis=1)
train_y = train[OBJECT_COL]

# 目的変数に1を足してlogを取る(log1p変換)
train_y_log = np.log1p(train_y)

test_x = all_data.iloc[len(train):, :].copy()
test_x = test_x.drop("id", axis=1)


print(train_x)
#print(train)


# -----------------------------------
# clipping
# -----------------------------------
# 列ごとに学習データの1％点、99％点を計算
p01 = train_x.quantile(0.01)
p99 = train_x.quantile(0.99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x = train_x.clip(p01, p99, axis=1)
test_x = test_x.clip(p01, p99, axis=1)

# 学習データにおける各特徴量の要約統計量
# ⇒ 件数，平均，標準偏差，最小値，4分位数，最大値
print(train_x.describe())

print(train)
print(train_x)


# 2乗した列だけをまとめて作る
sq_cols = train_x.drop('Sex', axis=1)                       # 数値列だけ抽出
train_sq = sq_cols ** 2                                     # 全列を一括 2 乗
train_sq.columns = [f"{c}_squared" for c in train_sq.columns]

# logした列だけをまとめて作る
log_cols = train_x.drop('Sex', axis=1)                        # 数値列だけ抽出
train_log = np.log(log_cols)                                  # 全列を一括 log
train_log.columns = [f"{c}_log" for c in train_log.columns]

# sqrtした列だけをまとめて作る
sqrt_cols = train_x.drop('Sex', axis=1)                       # 数値列だけ抽出
train_sqrt = np.sqrt(sqrt_cols)                               # 全列を一括 sqrt
train_sqrt.columns = [f"{c}_sqrt" for c in train_sqrt.columns]

# 元データと横結合して完成
train_x = pd.concat([train_x, train_sq, train_log, train_sqrt], axis=1)
print(train_x)


# 2乗した列だけをまとめて作る
sq_cols = test_x.drop('Sex', axis=1)                       # 数値列だけ抽出
test_sq = sq_cols ** 2                                     # 全列を一括 2 乗
test_sq.columns = [f"{c}_squared" for c in test_sq.columns]

# logした列だけをまとめて作る
log_cols = test_x.drop('Sex', axis=1)                        # 数値列だけ抽出
test_log = np.log(log_cols)                                  # 全列を一括 log
test_log.columns = [f"{c}_log" for c in test_log.columns]

# sqrtした列だけをまとめて作る
sqrt_cols = test_x.drop('Sex', axis=1)                       # 数値列だけ抽出
test_sqrt = np.sqrt(sqrt_cols)                               # 全列を一括 sqrt
test_sqrt.columns = [f"{c}_sqrt" for c in test_sqrt.columns]

# 元データと横結合して完成
test_x = pd.concat([test_x, test_sq, test_log, test_sqrt], axis=1)
print(test_x)
print(test_x.describe())


# 特徴量の作成 (train)
train_x['Combined_Whole_Weight'] = train_x['Whole weight'] + train_x['Whole weight.1'] + train_x['Whole weight.2']
train_x['Shell_Volume'] = (4/3) * 3.14 * (train_x['Diameter'] / 2) ** 2 * train_x['Height']
train_x['Shell_Thickness'] = train_x['Height'] - train_x['Diameter']
train_x['Diameter_Length_Product'] = train_x['Diameter'] * train_x['Length']
train_x['Shell_Surface_Area'] = 4 * 3.14 * (train_x['Diameter'] / 2) ** 2
train_x['Shell_Volume_2'] = train_x['Length'] * train_x['Height'] * train_x['Diameter']
train_x['Weight_remains'] = train_x['Whole weight'] - train_x['Whole weight.1'] - train_x['Whole weight.2'] - train['Shell weight']


# 特徴量の作成 (test)
test_x['Combined_Whole_Weight'] = test_x['Whole weight'] + test_x['Whole weight.1'] + test_x['Whole weight.2']
test_x['Shell_Volume'] = (4/3) * 3.14 * (test_x['Diameter'] / 2) ** 2 * test_x['Height']
test_x['Shell_Thickness'] = test_x['Height'] - test_x['Diameter']
test_x['Diameter_Length_Product'] = test_x['Diameter'] * test_x['Length']
test_x['Shell_Surface_Area'] = 4 * 3.14 * (test_x['Diameter'] / 2) ** 2
test_x['Shell_Volume_2'] = test_x['Length'] * test_x['Height'] * test_x['Diameter']
test_x['Weight_remains'] = test_x['Whole weight'] - test_x['Whole weight.1'] - test_x['Whole weight.2'] - test_x['Shell weight']

print(train_x)
print(test_x)


# ---------------------------------
# スタッキング
# ----------------------------------
# models.pyにModel1Xgb, Model1NN, Model2Linearを定義しているものとする
# 各クラスは、fitで学習し、predictで予測値の確率を出力する

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, train_x, train_y, test_x, num_round):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=6, shuffle=True, random_state=71)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y, num_round)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test


# モデルの学習
num_round = 500 # 学習回数

# 1. Lgb
model_Lgb = Model1Lgb()
pred_train_Lgb, pred_test_Lgb = predict_cv(model_Lgb, train_x, train_y_log, test_x, num_round)

# 予測値から1を引いてexpを取って元のスケールに戻す(expm1変換)
pred_train_Lgb_exp = np.expm1(pred_train_Lgb)
pred_test_Lgb_exp  = np.expm1(pred_test_Lgb)

# 評価
rmse = np.sqrt(mean_squared_log_error(train_y, pred_train_Lgb_exp))
print(f'Lgb_RMSLE : {rmse:.6f}')



# 2.Xgb
model_Xgb = Model1Xgb()
pred_train_Xgb, pred_test_Xgb = predict_cv(model_Xgb, train_x, train_y_log, test_x, num_round)

# 予測値から1を引いてexpを取って元のスケールに戻す(expm1変換)
pred_train_Xgb_exp = np.expm1(pred_train_Xgb)
pred_test_Xgb_exp  = np.expm1(pred_test_Xgb)

# 評価
rmse = np.sqrt(mean_squared_log_error(train_y, pred_train_Xgb_exp))
print(f'Xgb_RMSLE : {rmse:.6f}')


# 3.Cat
model_Cat = Model1Cat()
pred_train_Cat, pred_test_Cat = predict_cv(model_Cat, train_x, train_y_log, test_x, num_round)

# 予測値から1を引いてexpを取って元のスケールに戻す(expm1変換)
pred_train_Cat_exp = np.expm1(pred_train_Cat)
pred_test_Cat_exp  = np.expm1(pred_test_Cat)

# 評価
rmse = np.sqrt(mean_squared_log_error(train_y, pred_train_Cat_exp))
print(f'Cat_RMSLE : {rmse:.6f}')



# 3つのモデルの出力を平均化
pred_test = (pred_test_Lgb_exp + pred_test_Xgb_exp + pred_test_Cat_exp)/3


# 提出用データ作成
submission = pd.DataFrame({ID_COL: test[ID_COL], OBJECT_COL: pred_test})
submission.to_csv("submission.csv", index=False)

