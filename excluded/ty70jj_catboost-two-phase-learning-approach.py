import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, CatBoostRegressor

import optuna

from warnings import filterwarnings
filterwarnings("ignore")
%matplotlib inline

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# データセットの読み込み
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv', parse_dates=['datetime'])
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv', parse_dates=['datetime'])


# datetimeをインデックスに設定
train_df = train_df.set_index('datetime')
test_df = test_df.set_index('datetime')


# train_dfとtest_dfの結合
df = pd.concat([train_df, test_df], axis=0)


# 短縮形に変換
df = df.rename(columns = {
    'promotion_1': '1',
    'promotion_2': '2',
    'promotion_3': '3'
})


# 特徴量を作成
pro_cols = ['1', '2', '3']
for col in pro_cols:
    df['e'+col] = np.exp(df[col])
    df['l'+col] = np.log1p(df[col])


# 拡張する特徴量のリスト
expand_cols = ['1','2','3','e1','e2','e3','l1','l2','l3']


# 再帰を用いて特徴量の定義
def Recursion(df, depth):
    if depth == 1:
        return df
    cols = [col for col in df.columns if col not in ['e_users']]
    ncols = len(cols)
    print(f'depth: {depth}, ncols: {ncols}')
    for i, col, in enumerate(cols):
        for j in range(ncols):
            if i == j:
                continue
            name1 = cols[i]
            name2 = cols[j]
            # 足し算は人気がないので削除
            # add = '(' + expand_cols[i] + '_a_' + expand_cols[j] + ')'
            # if add in expand_cols:
            #     continue
            # expand_cols.append(add)
            # print(add)
            # df[add] = df[name1] + df[name2]
            sub = '(' + expand_cols[i] + '_s_' + expand_cols[j] + ')'
            if sub in expand_cols:
                continue
            expand_cols.append(sub)
            # print(sub)
            df[sub] = df[name1] - df[name2]
            div = '(' + expand_cols[i] + '_d_' + expand_cols[j] + ')'
            if div in expand_cols:
                continue
            expand_cols.append(div)
            # print(div)
            df[div] = df[name1] / df[name2]
            if (i < 3) and (j < 3) and (i > j):
                continue 
            mul = '(' + expand_cols[i] + '_m_' + expand_cols[j] + ')'
            if mul in expand_cols:
                continue
            expand_cols.append(mul)
            # print(mul)
            df[mul] = df[name1] * df[name2]

            Recursion(df,depth+1)
    return df


# 再帰を用いて特徴量の生成
df2 = Recursion(df, 0)


# df2のサイズ確認
print(f'df2.shape: {df2.shape}')


# 特徴量の選択
selected_features = ['e_users', '1',  '2',  '3',  'e1',  'e2',  'e3',  '(1_m_2)',  '(1_d_e1)',  '(1_d_e3)',  '(1_d_l2)',  '(2_s_3)',  '(3_m_e2)',  '(e1_m_e3)',  '(e1_m_l2)',  '(e3_d_e1)',  '(e3_d_l2)',  '(e3_m_l2)',  '(l2_d_e1)',  '(l2_d_e3)',  '(l2_m_e3)']


df2 = df2[selected_features]


# 日付由来の特徴量を定義
def deal_date(_df):
    df = _df.copy()

    df['hour'] = df.index.hour
    # hour をラベルエンコーディング化
    encoder = LabelEncoder()
    df['hour'] = encoder.fit_transform(df['hour'])
    df['hour'] = df['hour'].astype('category')    
    # del df['hour'] # hourは削除しない

    df['dayofweek'] = df.index.dayofweek
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    del df['dayofweek']

    df['week'] = df.index.isocalendar().week
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52).astype(float)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52).astype(float)
    del df['week']

    df['day'] = df.index.day
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 30)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 30)
    del df['day']

    df['month'] = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    del df['month']

    df['is_off_week'] = np.where(df.index.dayofweek.isin([2,3,4]), 1, 0)
    df['is_off_week'] = df['is_off_week'].astype('category')
 
    # 繫閑分類モデル用
    df['busy_flag'] = (df['e_users'] > df['e_users'].median()).astype(int)

    return df


# 日付由来の特徴量を作成
df3 = deal_date(df2)


for col in df3.columns:
    print(f"'{col}',",end=' ')
print()


# busy_flag を目的変数とした、分類モデルの特徴量
features1 = [col for col in df3.columns if col not in ['e_users', 'busy_flag']]


# lagとrollingに使用する特徴量
used_cols = ['1',  '2',  '3',  'e1',  'e2',  'e3',  '(1_m_2)',  '(1_d_e1)',  '(1_d_e3)',  '(1_d_l2)',  '(2_s_3)',  '(3_m_e2)',  '(e1_m_e3)',  '(e1_m_l2)',  '(e3_d_e1)',  '(e3_d_l2)',  '(e3_m_l2)',  '(l2_d_e1)',  '(l2_d_e3)',  '(l2_m_e3)']


# e_users予測（回帰）の為の
# lagとrollingの特徴量の定義
def gene_feature(X):
    df = X.copy()

    # 基本lagは1週間前のみ
    for lag in [168]:
        for col in used_cols:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # 大きな周期（1週間以上）でrolling統計を取る
    # long_periods = [24*7, 24*15, 24*30]  # 7日, 15日, 30日

    # 大きな周期は30日のみ
    long_periods = [24*30]
    for period in long_periods:
        for col in used_cols:
            df[f'{col}_mean_{period}'] = df[col].rolling(period, min_periods=1).mean()
            df[f'{col}_std_{period}'] = df[col].rolling(period, min_periods=1).std()
            df[f'{col}_max_{period}'] = df[col].rolling(period, min_periods=1).max()
            df[f'{col}_min_{period}'] = df[col].rolling(period, min_periods=1).min()
            df[f'{col}_median_{period}'] = df[col].rolling(period, min_periods=1).median()
    
    return df


# 特徴量の生成
df4 = gene_feature(df3)


# df4のサイズ確認
print(f'df4 shape: {df4.shape}')


# train, test の分離
train = df4[:len(train_df)]
test  = df4[len(train_df):]


# CatBoostClassifierで分類モデル
clf = CatBoostClassifier(cat_features=['hour', 'is_off_week'])


# 学習
clf.fit(train[features1], train['busy_flag'])


# 予測の結果（確率）
train['busy_prob'] = clf.predict_proba(train[features1])[:,1]
test['busy_prob'] = clf.predict_proba(test[features1])[:,1]


# e_usersが目的変数である、回帰モデルの特徴量
features2 = [col for col in train.columns if col not in ['e_users', 'busy_flag']]


# 一定のハイパーパラメータ
constant_param_cb = {
        'cat_features': ['hour', 'is_off_week'], #['is_off_week'],
        'random_seed': 0,
        'early_stopping_rounds': 100,
        'eval_metric': 'RMSE',
        # 'task_type': 'GPU',
        'verbose': False,
}


# 時系列データの分割
tscv = TimeSeriesSplit(n_splits=5)


# X, y, X_testの生成
X = train[features2]
y = train['e_users']
X_test = test[features2]


# Optunaによるハイパーパラメータの最適化
# def objective(trial):
#     scores = []
#     params = {
#             'iterations':               trial.suggest_int('iterations', 100, 1000),
#             'learning_rate':            trial.suggest_float('learning_rate', 0.01, 0.3),
#             'depth':                    trial.suggest_int('depth', 4, 6),
#             'bagging_temperature':      trial.suggest_float('bagging_temperature', 0, 1),
#     }
#     params = {**constant_param_cb, **params}
#     model = CatBoostRegressor(**params)
#     score = cross_val_score(estimator=model, X=X, y=y.values, cv=tscv, scoring='neg_root_mean_squared_error').mean()
#     scores.append(score)

#     return -np.mean(scores)


# optunaの最適化
# n_trials = 3
# best_params = {}
# best_values = float('inf')
# for i in range(5):
#     print(f'{i+1} trial:')
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

#     # 最適化されたハイパーパラメータの表示
#     params = study.best_params
#     values = study.best_value

#     print(f'\nRMSE: {values:.4f}')
#     if best_values > values:
#         best_values = values
#         best_params = params


# 最適化されたハイパーパラメータ
best_params = {
    'iterations': 694,
    'learning_rate': 0.20465813054570742,
    'depth': 6,
    'bagging_temperature': 0.5974863814008128,    
}
best_params |= constant_param_cb


# 学習
reg = CatBoostRegressor(**best_params)
rmse = cross_val_score(estimator=reg, X=X, y=y.values, cv=tscv, scoring='neg_root_mean_squared_error').mean()
print(f"rmse: {-rmse:.4f}")
reg.fit(X, y)


# 特徴量の上位重要度の表示
feature_importance = reg.feature_importances_
feature_names = X.columns
# Feature importance analysis
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_idx][:20]
sorted_feature_names = [feature_names[i] for i in sorted_idx][:20]

plt.figure(figsize=(10,5))
plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance)
plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()


# 予測
y_pred_train = reg.predict(X)
rmse_train = mean_squared_error(y.values, y_pred_train, squared=False)
print(f'Train RMSE: {rmse_train}')
test_pred = reg.predict(X_test)


# 提出準備
submission = pd.DataFrame({
    'datetime': test_df.index,
    'e_users': test_pred
}
)


# 提出
# submission.to_csv('submit.csv', index = False)




