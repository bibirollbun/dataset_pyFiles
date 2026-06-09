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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats

from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
n_splits = 8
n_estimators=5000
early_stopping_rounds = 100


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

train_data.head()


# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 在训练数据上拟合向量化器并转换训练数据
tfidf_train = vectorizer.fit_transform(train_data['Podcast_Name'])

# 获取特征名称（词汇表中的单词）
feature_names = vectorizer.get_feature_names_out()

# 将稀疏矩阵转换为DataFrame，方便查看
tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=feature_names)

# 使用相同的向量化器转换测试数据（不重新拟合）
tfidf_test = vectorizer.transform(test_data['Podcast_Name'])

# 将测试数据的稀疏矩阵转换为DataFrame
tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=feature_names)


missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

print(missing_values_train)
print(missing_values_test)
print(unique_values)
print(feature_types)


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")


train_data.describe().T


train_data['episode_num'] = [int(x.split()[-1]) for x in train_data['Episode_Title'].values]
test_data['episode_num'] = [int(x.split()[-1]) for x in test_data['Episode_Title'].values]

genre_mapping = {
    'Music': 0,         # 46.58 min (más escuchado)
    'True Crime': 1,    # 46.04 min
    'Health': 2,        # 45.74 min
    'Education': 3,     # 45.74 min
    'Technology': 4,    # 45.63 min
    'Business': 5,      # 45.54 min
    'Lifestyle': 6,     # 45.52 min
    'Sports': 7,        # 44.94 min
    'Comedy': 8,        # 44.43 min
    'News': 9           # 44.41 min (menos escuchado)
}

# Mapeo para Publication_Day basado en tiempo promedio de escucha
day_mapping = {
    'Tuesday': 0,     # 46.13 min (más escuchado)
    'Monday': 1,      # 45.97 min
    'Wednesday': 2,   # 45.81 min
    'Saturday': 3,    # 45.33 min
    'Friday': 4,      # 45.21 min
    'Thursday': 5,    # 44.87 min
    'Sunday': 6       # 44.82 min (menos escuchado)
}

# Mapeo para Publication_Time basado en tiempo promedio de escucha
time_mapping = {
    'Night': 0,      # 46.46 min (más escuchado)
    'Afternoon': 1,  # 45.53 min
    'Morning': 2,    # 44.96 min
    'Evening': 3     # 44.76 min (menos escuchado)
}

def data_process(df):
    df['Episode_Title_num'] = df['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=3)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace({'Neutral': 0, 'Positive': 1, 'Negative': -1})

    df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
    df['Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Popularity_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Host_Popularity_squared'] = df['Host_Popularity_percentage'] ** 2
    df['Popularity_Average'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'])/2
    
    df['Genre_Num'] = df['Genre'].map(genre_mapping)
    df['Publication_Day_Num'] = df['Publication_Day'].map(day_mapping)
    df['Publication_Time_Num'] = df['Publication_Time'].map(time_mapping)

    return df

train_data=data_process(train_data)
test_data=data_process(test_data)


cat_cols = train_data.select_dtypes(exclude=['number']).columns.tolist()
#train_data[cat_cols] = train_data[cat_cols].fillna(train_data[cat_cols].mode().iloc[0])
#test_data[cat_cols] = test_data[cat_cols].fillna(test_data[cat_cols].mode().iloc[0])
train_data[cat_cols] = train_data[cat_cols].fillna("Missing")
test_data[cat_cols] = test_data[cat_cols].fillna("Missing")
print(cat_cols)
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col]) 


train_data = pd.concat([train_data, tfidf_train_df], axis=1)
test_data = pd.concat([test_data, tfidf_test_df], axis=1)

X = train_data.drop(['id','Listening_Time_minutes'],axis=1)
y = train_data['Listening_Time_minutes']
test = test_data.drop(['id'],axis=1)

print('true y min:',y.min())
print('true y max:',y.max())
print('true y mean:',y.mean())
print('true y median:',y.median())


kf = KFold(n_splits, shuffle=True, random_state=42)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

lgbm_params1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    'max_depth': 15,
    'num_leaves': 64, 
    'reg_alpha' : 1,
    'reg_lambda' : 8,
    'colsample_bytree' : 0.7,
    'subsample' : 1.0,
    'subsample_freq' : 6,
    'seed': SEED,
    'verbose': -1,
    'device' : 'cpu' 
}

lgb_preds = []
for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds),lgb.log_evaluation(period=1000)]
    model = lgb.LGBMRegressor(**lgbm_params1)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric='rmse', categorical_feature=cat_cols, callbacks=callbacks)
    
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration_)
    lgb_preds += val_pred.tolist()
    score = mean_squared_error(y_val_fold, val_pred,squared=False)
    scores1.append(score)
    
    test_pred = np.maximum(model.predict(test, num_iteration=model.best_iteration_),0)
    test_preds1.append(test_pred)

    
    print(f'LightGBM Fold {i + 1} rmse: {score}')
print(f'LightGBM rmse: {np.mean(scores1):.5f};');

y_preds_lgb = np.mean(test_preds1, axis=0)
y_preds_lgb = y_preds_lgb
print('predict mean :',y_preds_lgb.mean())
print('predict median :',np.median(y_preds_lgb))
y_preds_lgb = np.clip(y_preds_lgb,0,119.97)
print('predict mean final:',y_preds_lgb.mean())
print('predict median final:',np.median(y_preds_lgb))


kf = KFold(n_splits, shuffle=True, random_state=43)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

xgb_params1 = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    'max_depth': 15,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'seed': SEED,
    'tree_method': 'hist',  # 使用 hist 方法加速训练
    'device': 'gpu'
}

xgb_preds = []

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params1)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=early_stopping_rounds, verbose=1000)

    val_pred = model.predict(X_val_fold)
    xgb_preds += val_pred.tolist()
    score = mean_squared_error(y_val_fold, val_pred, squared=False)
    scores1.append(score)

    test_pred = np.maximum(model.predict(test), 0)
    test_preds1.append(test_pred)

    print(f'XGBoost Fold {i + 1} rmse: {score}')
print(f'XGBoost rmse: {np.mean(scores1):.5f};')

y_preds_xgb = np.mean(test_preds1, axis=0)
y_preds_xgb = y_preds_xgb
print('predict mean :',y_preds_xgb.mean())
print('predict median :',np.median(y_preds_xgb))
y_preds_xgb = np.clip(y_preds_xgb,0,119.97)
print('predict mean final:',y_preds_xgb.mean())
print('predict median final:',np.median(y_preds_xgb))


kf = KFold(n_splits, shuffle=True, random_state=45)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

random_forest_params = {
    'n_estimators': 100,  # 树的数量
    'max_depth': 15,  # 树的最大深度
    'min_samples_split': 2,  # 内部节点再划分所需最小样本数
    'min_samples_leaf': 1,  # 叶节点最少样本数
    'max_features': 'auto',  # 寻找最佳分割时要考虑的特征数量
    'bootstrap': True,
    'random_state': SEED,
    'n_jobs': -1,  # 使用所有可用的处理器
}

rf_preds = []

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(**random_forest_params)
    model.fit(X_train_fold, y_train_fold)

    val_pred = model.predict(X_val_fold)
    rf_preds += val_pred.tolist()
    score = mean_squared_error(y_val_fold, val_pred, squared=False)
    scores1.append(score)

    test_pred = np.maximum(model.predict(test), 0)
    test_preds1.append(test_pred)

    print(f'Random Forest Fold {i + 1} rmse: {score}')

print(f'Random Forest rmse: {np.mean(scores1):.5f};')


y_preds_rf = np.mean(test_preds1, axis=0)
y_preds_rf = y_preds_rf
print('predict mean :',y_preds_rf.mean())
print('predict median :',np.median(y_preds_rf))
y_preds_rf = np.clip(y_preds_rf,0,119.97)
print('predict mean final:',y_preds_rf.mean())
print('predict median final:',np.median(y_preds_rf))


import optuna
import logging

# 确保预测值为 NumPy 数组
lgb_preds = np.array(lgb_preds)
xgb_preds = np.array(xgb_preds)
rf_preds = np.array(rf_preds)

# 获取 Optuna 的日志记录器
optuna_logger = logging.getLogger('optuna')
# 设置日志级别为 ERROR，这样只会输出错误级别的日志
optuna_logger.setLevel(logging.ERROR)

def weighted_ensemble_optuna(trial: optuna.Trial):
    weights = {
        'lgb': trial.suggest_float('lgb', 0.001, 1.0, log=True),
        'xgb': trial.suggest_float('xgb', 0.001, 1.0, log=True),
        'rf': trial.suggest_float('rf', 0.001, 1.0, log=True),
    }
    preds = (lgb_preds * weights['lgb'] + xgb_preds * weights['xgb'] + rf_preds * weights['rf']) / sum(weights.values())
    return mean_squared_error(train_data['Listening_Time_minutes'], preds)


study_weights = optuna.create_study(direction='minimize')
study_weights.optimize(weighted_ensemble_optuna, n_trials=2000)
weights = study_weights.best_params
print("Weighted Ensemble CV Score:", study_weights.best_value)


# y_preds = y_preds_lgb*0.2 + y_preds_xgb*0.7 + y_preds_rf*0.1
y_preds = (y_preds_lgb*weights['lgb'] + y_preds_xgb*weights['xgb'] + y_preds_rf*weights['rf']) / sum(weights.values())
# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'Listening_Time_minutes': y_preds})
submission.to_csv('submission.csv', index=False)
print(submission.head())

