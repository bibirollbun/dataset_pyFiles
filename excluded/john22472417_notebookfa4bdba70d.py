# !pip install optuna
# !pip install optuna-integration[lightgbm]
# !pip install scikit-learn==1.5.2
# !pip install numpy==1.26.4
# !pip install pandas==2.2.3
# !pip show scikit-learn
# !pip freeze


import lightgbm as lgb
import optuna.integration.lightgbm as otp_lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor
# from xgboost import XGBRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Competition variables.
base_path = "/kaggle/input/prediction-interval-competition-ii-house-price/"

train_df = pd.read_csv(base_path + "dataset.csv") # parse_dates=["sale_date"]
test_df = pd.read_csv(base_path + "test.csv")
test_ID = test_df['id']

# alpha = 0.1 は信頼区間 90% に相当（予測区間：[5%, 95%] を生成するための設定）
alpha = 0.1  # the specified competition alpha (i.e., 90% coverage)
quantiles = [alpha / 2, 1 - alpha / 2]


print(train_df.shape)
print(test_df.shape)


ID_COL = 'id'
TARGET_COL = 'sale_price'
train_df_id_target_col = train_df[[ID_COL, TARGET_COL]]
test_df_id = test_df[ID_COL]

df = pd.concat([train_df.drop(columns=[TARGET_COL]), test_df], axis=0).reset_index(drop=True)
print(df.shape)
df.head()


list(set(df.dtypes.tolist()))


# df_num = df.select_dtypes(include=['int64', 'float64'])
# df_num.hist(figsize=(16, 20), bins=50)


# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# # 箱ひげ図をまとめて描画（5列×n行のグリッド形式）
# cols = 5
# rows = (len(numeric_cols) + cols - 1) // cols

# plt.figure(figsize=(cols * 4, rows * 3))
# for i, col in enumerate(numeric_cols):
#     plt.subplot(rows, cols, i + 1)
#     sns.boxplot(x=df[col], color='skyblue')
#     plt.title(col)
#     plt.tight_layout()

# plt.show()


# df
missing_df = df.isnull().sum()
missing_df = missing_df[missing_df > 0].sort_values(ascending=False)
print(missing_df)


# 外れ値を除外する関数
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean


binary_features = ['wfnt', 'golf', 'greenbelt', 'view_rainier', 'view_olympics', 
                    'view_cascades', 'view_territorial', 'view_skyline', 'view_sound', 
                    'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other']

# 前処理
def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['sale_warning'] = df['sale_warning'].str.replace(' ', '')
    df['sale_warning'] = df['sale_warning'].str.replace('　', '')
    # 欠損値処理(カテゴリ変数)
    df = df.fillna('NA')
    # 欠損値処理(数値型)
    df["sale_nbr"] = df["sale_nbr"].fillna(0)

    # 特徴量作成
    df['has_sale_warning'] = df['sale_warning'].apply(lambda x: 0 if x == '' else 1)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = pd.to_datetime(df['sale_date']).dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_dayofweek'] = df['sale_date'].dt.dayofweek
    first_sale_month = df['sale_date'].dt.to_period('M').min()
    df['months_since_first_sale'] = (df['sale_date'].dt.to_period('M') - first_sale_month).apply(lambda x: x.n)
    df['house_age'] = df['sale_year'] - df['year_built']
    df['is_renovated'] = (df['year_reno'] > 0).astype(int)
    # df['years_since_reno'] = np.where(df['year_reno'] > 0, df['sale_year'] - df['year_reno'], np.nan)
    df['total_sqft'] = df['sqft'] + df['sqft_fbsmt'] + df['garb_sqft'] + df['gara_sqft']
    df['total_bath'] = df['bath_full'] + 0.75 * df['bath_3qtr'] + 0.5 * df['bath_half']
    df['total_views'] = df[binary_features].sum(axis=1)
    df['has_view'] = (df['total_views'] > 0).astype(int)
    df['log_sqft'] = np.log1p(df['sqft'])
    df['log_sqft_lot'] = np.log1p(df['sqft_lot'])

    # 余計な特徴量削除
    df = df.drop(columns=[ID_COL, "sale_date", "subdivision", "sale_warning"])
    
    # 歪度を調整
    skewed_feats = df.select_dtypes(include=['int64', 'float64']).apply(lambda x: x.skew()).sort_values(ascending=False)
    
    skewed_cols = skewed_feats[skewed_feats > 0.5].index
    skewed_cols = skewed_cols.drop(["longitude", "house_age"])
    df[skewed_cols] = np.log1p(df[skewed_cols])
    
    # 標準化
    num_cat_list = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cat_list] = (df[num_cat_list] - df[num_cat_list].mean()) / df[num_cat_list].std()
    
    df_clean = remove_outliers_iqr(df, num_cat_list)
    
    # カテゴリ変数を数値化
    # df = pd.get_dummies(df, drop_first=True)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    
    return df


df_preprocess = preprocess(df)
df_preprocess.shape


# trainとtestに分割
x_data_train = df_preprocess.loc[:train_df_id_target_col.shape[0] - 1, :]
y_data_train = train_df_id_target_col[TARGET_COL]
x_data_test  = df_preprocess.loc[train_df_id_target_col.shape[0]:, :]
print(x_data_train.shape)
print(x_data_test.shape)


# ターゲットのSalePriceを取り出し
y = np.log1p(y_data_train) #ターゲット
X = x_data_train


# LightGBM ハイパーパラメータ
params_base = {
    'objective': 'quantile',
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
}


# # ランダムシード値
# RANDOM_STATE = 10

# # 学習データと評価データの割合
# TEST_SIZE = 0.2

# # trainのデータセットの3割をモデル学習時のバリデーションデータとして利用する
# x_train, x_valid, y_train, y_valid = train_test_split(X,
#                               y,
#                               test_size=TEST_SIZE,
#                               random_state=RANDOM_STATE)

# # LightGBMを利用するのに必要なフォーマットに変換
# lgb_train = otp_lgb.Dataset(x_train, y_train)
# lgb_eval = otp_lgb.Dataset(x_valid, y_valid, reference=lgb_train)

# params_gbm_lower = params_base.copy()
# params_gbm_lower['alpha'] = quantiles[0]
# params_gbm_upper = params_base.copy()
# params_gbm_upper['alpha'] = quantiles[1]

# # LightGBM学習
# gbm_lower = otp_lgb.train(params_gbm_lower,
#                 lgb_train,
#                 num_boost_round=500,
#                 valid_sets=[lgb_train, lgb_eval],
#                 callbacks=[otp_lgb.early_stopping(stopping_rounds=10,
#                                 verbose=True), # early_stopping用コールバック関数
#                 lgb.log_evaluation(0)] # コマンドライン出力用コールバック関数
#                )

# gbm_upper = otp_lgb.train(params_gbm_upper,
#                 lgb_train,
#                 num_boost_round=500,
#                 valid_sets=[lgb_train, lgb_eval],
#                 callbacks=[otp_lgb.early_stopping(stopping_rounds=10,
#                                 verbose=True), # early_stopping用コールバック関数
#                 lgb.log_evaluation(0)] # コマンドライン出力用コールバック関数
#                )

# print("▼ Lower Model Parameters:")
# print(gbm_lower.params)

# print("\n▼ Upper Model Parameters:")
# print(gbm_upper.params)


# # 目的関数の定義
# def quantile_loss(q):
#     def loss(preds, dtrain):
#         labels = dtrain.get_label()
#         errors = labels - preds
#         grad = np.where(errors > 0, -q, 1 - q)
#         hess = np.ones_like(preds)
#         return grad, hess
#     return loss

# def objective_xgb(trial, quantile):
#     param = {
#         "eta": trial.suggest_loguniform("eta", 1e-8, 1.0),
#         "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "min_child_weight": trial.suggest_loguniform("min_child_weight", 1, 40),
#         "max_delta_step": trial.suggest_loguniform("max_delta_step", 1e-8, 1.0),
#         "subsample": trial.suggest_uniform("subsample", 0.0, 1.0),
#         "lambda": trial.suggest_uniform("reg_lambda", 0.0, 1000.0),
#         "alpha": trial.suggest_uniform("reg_alpha", 0.0, 1000.0),
#     }

#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dvalid = xgb.DMatrix(x_valid, label=y_valid)

#     bst = xgb.train(
#         param,
#         dtrain,
#         num_boost_round=300,
#         obj=quantile_loss(quantile),
#         verbose_eval=False
#     )

#     preds = bst.predict(dvalid)
#     rmse = root_mean_squared_error(y_valid, preds)

#     return rmse

# # 下限（0.05）の探索
# study_xgb_lower = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
# study_xgb_lower.optimize(lambda trial: objective_xgb(trial, quantiles[0]), n_trials=50)
# best_params_xgb_lower = study_xgb_lower.best_params
# best_params_xgb_lower["random_state"] = RANDOM_STATE

# # 上限（0.95）の探索
# study_xgb_upper = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
# study_xgb_upper.optimize(lambda trial: objective_xgb(trial, quantiles[1]), n_trials=50)
# best_params_xgb_upper = study_xgb_upper.best_params
# best_params_xgb_upper["random_state"] = RANDOM_STATE


# # 目的関数の定義
# def objective_cat(trial, quantile):

#     param = {
#         "iterations": trial.suggest_int("iterations", 50, 300),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
#         "random_strength": trial.suggest_int("random_strength", 0, 100),
#         "bagging_temperature": trial.suggest_loguniform(
#             "bagging_temperature", 0.01, 100.00
#         ),
#         "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
#         "od_wait": trial.suggest_int("od_wait", 10, 50),
#         "loss_function": f"Quantile:alpha={quantile}",
#         "verbose": False
#     }

#     model = CatBoostRegressor(**param)

#     model.fit(
#         x_train,
#         y_train,
#         eval_set=[(x_valid, y_valid)],
#         verbose=False,
#     )

#     preds = model.predict(x_valid)
#     rmse = root_mean_squared_error(y_valid, preds)

#     return rmse

# # 下限
# study_cat_lower = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
# study_cat_lower.optimize(lambda trial: objective_cat(trial, quantiles[0]), n_trials=50)

# # 上限
# study_cat_upper = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
# study_cat_upper.optimize(lambda trial: objective_cat(trial, quantiles[1]), n_trials=50)


# # 目的関数の定義
# def objective_quantile_ridge(trial, quantile):
#     alpha = trial.suggest_float('alpha', 0.001, 10.0)
#     model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")  # solverは"highs"が推奨
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_valid)
#     return root_mean_squared_error(y_valid, y_pred)

# # 下限
# study_qr_lower = optuna.create_study(direction='minimize')
# study_qr_lower.optimize(lambda trial: objective_quantile_ridge(trial, quantiles[0]), n_trials=50)

# # 上限
# study_qr_upper = optuna.create_study(direction='minimize')
# study_qr_upper.optimize(lambda trial: objective_quantile_ridge(trial, quantiles[1]), n_trials=50)
# # 結果の確認
# print(study_qr_lower.best_value)
# print(study_qr_lower.best_params)


# best_params_ridge = {
#     'alpha': 0.008812014056503567,
#     'random_state': 42
# }
best_params_lgb_lower = {
    'objective': 'quantile',
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'alpha': 0.05,
    'feature_pre_filter': False,
    'lambda_l1': 7.532749962844209,
    'lambda_l2': 2.032983797605824e-08,
    'num_leaves': 35,
    'feature_fraction': 0.6479999999999999,
    'bagging_fraction': 0.9953013028257032,
    'bagging_freq': 6,
    'min_child_samples': 50,
    'num_iterations': 500
}

best_params_lgb_upper = {
    'objective': 'quantile',
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'alpha': 0.95,
    'feature_pre_filter': False,
    'lambda_l1': 9.23116849927469,
    'lambda_l2': 8.846454813037917,
    'num_leaves': 31,
    'feature_fraction': 0.7,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 100,
    'num_iterations': 500
}

best_params_xgb_lower = {
    'eta': 0.6157871448750519,
    'gamma': 0.00021163157157489846,
    'learning_rate': 0.09901212861731738,
    'max_depth': 4,
    'min_child_weight': 6.856847742680047,
    'max_delta_step': 0.21729447577514674,
    'subsample': 0.580817500978563,
    'reg_lambda': 182.08804399008284,
    'reg_alpha': 261.2765574470825,
    'random_state': 10
}

best_params_xgb_upper = {
    'eta': 0.0005433251850404714,
    'gamma': 9.193071524598263e-08,
    'learning_rate': 0.04975549968720229,
    'max_depth': 8,
    'min_child_weight': 23.1947853160839,
    'max_delta_step': 0.9561281690503357,
    'subsample': 0.5132128874027321,
    'reg_lambda': 134.55719655144125,
    'reg_alpha': 36.43478770073472,
    'random_state': 10
}

best_params_cat_lower = {
    'iterations': 290,
    'depth': 8,
    'learning_rate': 0.16664019397790825,
    'random_strength': 22,
    'bagging_temperature': 16.72249590600205,
    'od_type': 'Iter',
    'od_wait': 27
}

best_params_cat_upper = {
    'iterations': 272,
    'depth': 10,
    'learning_rate': 0.10370429617939152,
    'random_strength': 6,
    'bagging_temperature': 42.50322505247505,
    'od_type': 'IncToDec',
    'od_wait': 15
}


# 評価指標の元
def calculate_score(row):
    range_width = row['pi_upper'] - row['pi_lower']
    if row['true'] < row['pi_lower']:
        return range_width + 2 / alpha * (row['pi_lower'] - row['true'])
    elif row['true'] > row['pi_upper']:
        return range_width + 2 / alpha * (row['true'] - row['pi_upper'])
    else:
        return range_width


def get_score(ids, lower_predict, upper_predict, y_true):
    pred_df = pd.DataFrame()
    pred_df[ID_COL] =ids
    pred_df['pi_lower'] = np.expm1(lower_predict)
    pred_df['pi_upper'] = np.expm1(upper_predict)
    pred_df['true'] = y_true
    pred_df['score'] = pred_df.apply(calculate_score, axis=1)
    return pred_df['score'].mean()


# モデル訓練
# ridge.fit(X, y)
# xgb_model.fit(X, y)

#Cross Validation（KFold）でモデル作成
y_data_preds_lgb_lower = []
y_data_preds_lgb_upper = []
kf = KFold(n_splits=5, shuffle=True, random_state=0)

#y_data_trainをもとに分割するため、splitの引数にy_data_trainを追加
for fold_id, (train_index, valid_index) in enumerate(kf.split(X)):
    # データ分割
    X_tr, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[valid_index]
    
    # モデル作成
    lgb_model_lower = lgb.LGBMRegressor(**best_params_lgb_lower)
    lgb_model_upper = lgb.LGBMRegressor(**best_params_lgb_upper)
    lgb_model_lower.fit(X_tr, y_tr)
    lgb_model_upper.fit(X_tr, y_tr)
    
    # 訓練データで確認
    y_tr_pred_lgb_lower = lgb_model_lower.predict(X_tr)
    y_tr_pred_lgb_upper = lgb_model_upper.predict(X_tr)
    loss_tr_w = get_score(train_df_id_target_col[ID_COL].iloc[train_index], y_tr_pred_lgb_lower, y_tr_pred_lgb_upper, train_df_id_target_col[TARGET_COL].iloc[train_index])
    print(f"lgb_tr ,fold_id = {fold_id}, score = {loss_tr_w:.5f}")
    
    # 検証データで確認
    y_val_pred_lgb_lower = lgb_model_lower.predict(X_val)
    y_val_pred_lgb_upper = lgb_model_upper.predict(X_val)
    loss_val_w = get_score(train_df_id_target_col[ID_COL].iloc[valid_index], y_val_pred_lgb_lower, y_val_pred_lgb_upper, train_df_id_target_col[TARGET_COL].iloc[valid_index])
    print(f"lgb_val,fold_id = {fold_id}, score = {loss_val_w:.5f}")
    
    # 提出用データ格納
    y_data_pred_lgb_lower = lgb_model_lower.predict(x_data_test)
    y_data_pred_lgb_upper = lgb_model_upper.predict(x_data_test)
    y_data_preds_lgb_lower.append(y_data_pred_lgb_lower)
    y_data_preds_lgb_upper.append(y_data_pred_lgb_upper)


y_data_preds_lgb_lower_array = np.array(y_data_preds_lgb_lower)  # shape: (5, N)
y_data_preds_lgb_lower_final = np.mean(y_data_preds_lgb_lower_array, axis=0)  # shape: (N)

y_data_preds_lgb_upper_array = np.array(y_data_preds_lgb_upper)  # shape: (5, N)
y_data_preds_lgb_upper_final = np.mean(y_data_preds_lgb_upper_array, axis=0)  # shape: (N)


# # モデル定義
# ridge = Ridge(**best_params_ridge)
# xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.7)
# # xgb_model = xgb.XGBRegressor(**best_params_xgb)

# 5%分位点のモデル（下限）
# lgb_model_lower = lgb.LGBMRegressor(**best_params_gbm_lowwer)

# 95%分位点のモデル（上限）
# lgb_model_upper = lgb.LGBMRegressor(**best_params_gbm_upper)
# # cat_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=3, verbose=False)
# cat_model = CatBoostRegressor(**best_params_cat)


# # モデル訓練
# ridge.fit(X, y)
# xgb_model.fit(X, y)
# lgb_model.fit(X, y)
# lgb_model_lower.fit(X, y)
# lgb_model_upper.fit(X, y)
# cat_model.fit(X, y)


# # 訓練データの精度検証
# pred_ridge = ridge.predict(X)
# pred_xgb = xgb_model.predict(X)
# pred_lgb = lgb_model.predict(X)
# pred_cat = cat_model.predict(X)

# rmse_ridge = root_mean_squared_error(y_true=y, y_pred=pred_ridge)
# print(f"RMSE_rid: {rmse_ridge:.5f}")
# rmse_xgb = root_mean_squared_error(y_true=y, y_pred=pred_xgb)
# print(f"RMSE_xgb: {rmse_xgb:.5f}")
# rmse_lgb = root_mean_squared_error(y_true=y, y_pred=pred_lgb)
# print(f"RMSE_lgb: {rmse_lgb:.5f}")
# pred_lgb_lower = lgb_model_lower.predict(x_data_test)
# pred_lgb_upper = lgb_model_upper.predict(x_data_test)
# rmse_cat = root_mean_squared_error(y_true=y, y_pred=pred_cat)
# print(f"RMSE_cat: {rmse_cat:.5f}")


# final_preds = (pred_ridge + pred_xgb + pred_lgb + pred_cat) / 4
# 今は一つのモデルしか使用していないためfinalにpred_lgbを代入しているだけ
final_preds_lower = y_data_preds_lgb_lower_final
final_preds_upper = y_data_preds_lgb_upper_final

final_preds_lower = np.expm1(final_preds_lower)
final_preds_upper = np.expm1(final_preds_upper)


sub = pd.DataFrame()
sub[ID_COL] = test_df_id
sub['pi_lower'] = final_preds_lower
sub['pi_upper'] = final_preds_upper
sub.head(5)


# sub.to_csv("submission.csv", index=False)

