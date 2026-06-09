# 必要なライブラリのimport
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings('ignore')

import datetime
import datetime as dt


# 訓練データの読み込み
# load train_df.csv
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')

# カラム"datetime"を、datetime型に変換
train_df['datetime'] = pd.to_datetime(train_df['datetime'])

# 確認
print(f'train_df.shape = {train_df.shape}')
train_df


# 予測対象データの読み込み
# load test_df.csv
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')

# カラム"datetime"を、datetime型に変換
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# 確認
print(f'test_df.shape = {test_df.shape}')
test_df


# Submissionファイルの読み込み
# load submission.csv
submission = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/submission.csv')

# 確認
print(f'submission.shape = {submission.shape}')
submission





# ーーーー 特徴量エンジニアリング ーーーー
# 以下、必要に応じて、特徴量生成等を行う。


# 例えば、特徴量生成
# datetimeから、時系列に関する変数を生成したり・・・・
# 数値変数から、例えば、1/変数　等の新たな変数を生成したり・・・


# datetimeから、year, month, day, hour, day_of_year, day_of_weekの情報を生成
# 訓練データ
train_df['year'] = train_df['datetime'].dt.year
train_df['month'] = train_df['datetime'].dt.month
train_df['day'] = train_df['datetime'].dt.day
train_df['hour'] = train_df['datetime'].dt.hour
train_df['day_of_year'] = train_df['datetime'].dt.dayofyear
train_df['day_of_year'] = train_df['datetime'].dt.dayofweek


# 予測対象データ
test_df['year'] = test_df['datetime'].dt.year
test_df['month'] = test_df['datetime'].dt.month
test_df['day'] = test_df['datetime'].dt.day
test_df['hour'] = test_df['datetime'].dt.hour
test_df['day_of_year'] = test_df['datetime'].dt.dayofyear
test_df['day_of_year'] = test_df['datetime'].dt.dayofweek



# 表示
display(train_df)
display(test_df)





# ーーーー　時系列考慮のHold-Out ーーーー





# モデル訓練用データ
# 2024年6月30日までのデータをモデル訓練用として分割
train_hold_df = train_df[(train_df['datetime'] <= dt.datetime(2024,6,30))]

# 確認
print(f'train_hold_df.shape = {train_hold_df.shape}')
train_hold_df


# モデル検証用データ
# 2024年7月1日から、8月31日までのデータをモデル検証用として分割
valid_hold_df = train_df[(train_df['datetime'] >= dt.datetime(2024,7,1)) & (train_df['datetime'] <= dt.datetime(2024,8,31))]

# 確認
print(f'valid_hold_df.shape = {valid_hold_df.shape}')
valid_hold_df


# 学習用データの目的変数を設定
y_train_hold = train_hold_df[['e_users']]

# 学習用データの説明変数を設定
x_train_hold = train_hold_df.drop(['datetime', 'e_users' ], axis=1)


# 検証データの目的変数を設定
y_valid_hold = valid_hold_df[['e_users']]

# 検証データの説明変数を設定
x_valid_hold = valid_hold_df.drop(['datetime', 'e_users' ], axis=1)

# 確認
display(x_train_hold)
display(x_valid_hold)





# LightGBM ハイパーパラメーターの設定
lgbm_params = {
    'boosting_type': 'gbdt',        # Gradient Boosting Decision Tree
    'objective': 'regression',      # 回帰タスク
    'metric': 'rmse',               # 評価指標
    
    'learning_rate': 0.02,          # デフォルトは0.1ですが、やや大きいと感じるので、数万程度のデータ数の場合は、0.02〜0.05に設定することが多いです。
    'n_estimators': 100000,         # Early-Stoppingを使いますので、大きめに設定
    'importance_type': 'gain',      # 特徴量の重要度取得用。 gainを選択した方が良いと言われています。
    
    # -- モデルの学習促進 --           大きくすると、学習は進みますが、過学習しやすくなります。
    'num_leaves': 32,               # 8〜256
    
    
    # -- モデルの過学習抑制 --         # 大きくすることで、過学習を抑制できます。
    'min_data_in_leaf': 20,         # 5〜200
    'min_sum_hessian_in_leaf': 20,  # 5〜200
    'lambda_l1': 0.0,               # 0.01〜100
    'lambda_l2': 0.0,               # 0.01〜100
    
    'bagging_fraction': 0.9,        # 小さくするほど、過学習を抑え、学習速度もあがる。
    'bagging_freq': 1,              # 1にすると、決定木作成ごとに、毎回、サンプリングが実行される。（デフォルトは0）
    'feature_fraction': 0.9,        # 小さくするほど、過学習を抑え、学習速度もあがる。

    'random_seed': 123              # 乱数設定。 数値はご自由に。
}


# LightGBMREgressor()インスタンスの生成
model = lgb.LGBMRegressor(**lgbm_params)

# verbose_evalの設定
verbose_eval = 1

# モデルのfitting
model.fit(
        x_train_hold,
        y_train_hold, 
        eval_set = [(x_train_hold, y_train_hold), (x_valid_hold, y_valid_hold)], 
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),lgb.log_evaluation(verbose_eval)], 
)

# 学習曲線の描画
lgb.plot_metric(model);





# 学習済みモデルが重視した変数の確認

# 空のDataFrameを用意
imp_df = pd.DataFrame()

# 説明変数の重要度をDataFrame化
temp_imp = pd.DataFrame({
    'col': x_train_hold.columns, 
    'imp': model.feature_importances_
})
    
# 重要度のDataFrameに結合
imp_df = pd.concat(
    [imp_df, temp_imp], 
    axis=0, 
    ignore_index=True
)

# 説明変数の重要度を整形
imp_group_df = imp_df.groupby('col')['imp'].agg(['mean', ])

# カラム名の変更
imp_group_df.columns = ['imp_mean', ]

# indexをリセット
imp_group_df = imp_group_df.reset_index(drop=False)

# imp_meanでソート
imp_group_df.sort_values('imp_mean', ascending=False, ignore_index=True).loc[0:50, :]


# 学習用データによるpredict
y_pred_train_hold = model.predict(x_train_hold, num_iteration=model.best_iteration_) 

# 検証用データによるpredict
y_pred_valid_hold = model.predict(x_valid_hold, num_iteration=model.best_iteration_) 

# RMSEを算出
# 学習用データ
temp_rmse_train = np.sqrt(mean_squared_error(y_train_hold, y_pred_train_hold))
    
# 検証用データ
temp_rmse_valid = np.sqrt(mean_squared_error(y_valid_hold, y_pred_valid_hold))
    
# RMSEの表示
print(f'\nRMSE(train_data) = {temp_rmse_train:.4f}')
print(f'RMSE(valid_data) = {temp_rmse_valid:.4f}\n')





# 学習済みモデルに投入する予測対象データの説明変数を設定
x_test = test_df.drop(['datetime'], axis=1)

# 確認
display(x_test)


# 予測対象用データで、実際に予測

# predict
preds_test = model.predict(x_test, num_iteration=model.best_iteration_)

# preds_testリストをNumPy配列に変換
preds_test_np = np.array(preds_test)

# 予測値から、submissionファイルを作成
submission['e_users'] = preds_test_np

# 確認
display(submission)


# submit submission.csv
submission.to_csv('./submission.csv', index=False)




