# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

import optuna

from warnings import filterwarnings
filterwarnings("ignore")
%matplotlib inline


# データセットの読み込み
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv', parse_dates=['datetime'])
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv', parse_dates=['datetime'])


# targetの設定, train_dfとtest_dfの結合
target = 'e_users'
df = pd.concat([train_df, test_df], axis=0)


# 特徴量エンジニアリング
def gene_feature(X):
    df = X.copy()

    # df['p2divp1'] = df['promotion_2'] / df['promotion_1']
    df['p3divp1'] = df['promotion_3'] / df['promotion_1']
    df['p1divp2'] = df['promotion_1'] / df['promotion_2']
    df['p3divp2'] = df['promotion_3'] / df['promotion_2']
    # df['p1divp3'] = df['promotion_1'] / df['promotion_3']
    df['p2divp3'] = df['promotion_2'] / df['promotion_3']

    df['p1mulp2'] = df['promotion_1'] * df['promotion_2']
    df['p1mulp3'] = df['promotion_1'] * df['promotion_3']
    df['p2mulp3'] = df['promotion_2'] * df['promotion_3']

    feature_cols = [col for col in df.columns if col not in ['datetime', 'e_users']]

    short_term = [1,2,3,6,8,12,18,24,36,48,60,72]
    long_term = [24*7, 24*15, 24*30, 24*45, 24*60, 24*90] #　, 24*120, 24*180]

    for period in short_term + long_term:
        for col in feature_cols:
            df[f's{period}_{col}'] = df[col].shift(period)
            df[f's{period}_{col}_diff'] = df[col].shift(period).diff()

    for period in [6,12,24,48,72] + long_term:
        for col in feature_cols:
            df[f'{col}_mean_{period}'] = df[col].rolling(period).mean()
            df[f'{col}_std_{period}'] = df[col].rolling(period).std()
            df[f'{col}_max_{period}'] = df[col].rolling(period).max()
            df[f'{col}_min_{period}'] = df[col].rolling(period).min()
            df[f'{col}_median_{period}'] = df[col].rolling(period).median()
                  
    df['month'] = df['datetime'].dt.month
    df['sin_moth'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_moth'] = np.cos(2 * np.pi * df['month'] / 12)
    del df['month'] 

    df['day'] = df['datetime'].dt.day
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 30)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 30)
    del df['day']
    
    df['days'] = df['datetime'].dt.dayofyear
    df['sin_days'] = np.sin(2 * np.pi * df['days'] / 365)
    df['cos_days'] = np.cos(2 * np.pi * df['days'] / 365)
    del df['days']

    df['week'] = df['datetime'].dt.isocalendar().week
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    del df['week']

    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)   
    del df['dayofweek']

    df['hour'] = df['datetime'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24) 
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    del df['hour']       

    df['on_season'] = np.where(df['datetime'].dt.month.isin([11, 12, 1, 2, 3, 4]), 1, 0)
    df['on_season'] = df['on_season'].astype('category')

    df['is_off_week'] = np.where(df['datetime'].dt.dayofweek.isin([2,3,4]), 1, 0)
    df['is_off_week'] = df['is_off_week'].astype('category')

    del df['datetime']

    return df


# 特徴量の作成
total_df = gene_feature(df) 


# total_dfのサイズの確認
print('total_df shapes: ',total_df.shape)


# トレーニングデータとテストデータの分割
X = total_df[:len(train_df)].drop(columns=target, axis=1)
y = total_df[:len(train_df)][target]
X_test = total_df[len(train_df):].drop(columns=target, axis=1)


# 時系列データの分割
tscv = TimeSeriesSplit(n_splits=5)


# 一定のハイパーパラメータ
constant_param_cb = {
        'cat_features': ['on_season', 'is_off_week'],
        'random_seed': 0,
        'early_stopping_rounds': 100,
        'eval_metric': 'RMSE',
#        'task_type': 'GPU',
        'verbose': False,
}


# Optunaによるハイパーパラメータの最適化(Notebook環境に負担を掛けるので、自分の環境で試して下さい)
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
#     score = cross_val_score(estimator=model, X=X, y=y, cv=tscv, scoring='neg_root_mean_squared_error').mean()
#     scores.append(score)

#     return -np.mean(scores)


# optunaの最適化(Notebook環境に負担を掛けるので、自分の環境で試して下さい)
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=5, show_progress_bar=True)

# 最適化されたハイパーパラメータの表示
# best_params = study.best_params
# best_values = study.best_value

# print(f'\nBest RMSE: {best_values:.4f}')
# print("\nBest hyperparameters:")
# for param, value in best_params.items():
#     print(f'\'{param}\': {value},')


# optunaによって最適化されたハイパーパラメータ
best_params = {
    'iterations': 464,
    'learning_rate': 0.07751798351628163,
    'depth': 5,
    'bagging_temperature': 0.10611974400830415,    
}


# 最適化されたハイパーパラメータの組み込み
best_params |= constant_param_cb


# 学習
model = CatBoostRegressor(**best_params)
model.fit(X,y)


# 特徴量の上位重要度の表示
feature_importance = model.feature_importances_
feature_names = X.columns

sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_idx][:20]
sorted_feature_names = [feature_names[i] for i in sorted_idx][:20]

plt.figure(figsize=(10,5))
plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance)
plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()


for name, importance in zip(feature_names, feature_importance):
    if name == 'on_season':
        print(f'name: {name} importance: {importance}')


# 予測
train_pred = model.predict(X)
test_pred = model.predict(X_test)


# 予測の結果の可視化
plt.figure(figsize=(12,6))
sns.lineplot(x=train_df['datetime'], y=train_df['e_users'], label='Train actual', color='blue')
sns.lineplot(x=train_df['datetime'], y=train_pred, label='Train predicted', color='green', alpha=0.65)
sns.lineplot(x=test_df['datetime'], y=test_pred, label='Test predicted', color = 'red', alpha=0.65)
# plt.xlim(pd.Timestamp('2022-05-06'), pd.Timestamp('2022-05-11'))
# plt.ylim(30000, 51000)
plt.grid()
plt.legend()
plt.show()


# 提出データの作成
submission = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': test_pred
}
)


# 提出
# submission.to_csv('submit.csv', index = False)




