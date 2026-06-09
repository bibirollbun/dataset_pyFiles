import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')


# load data
df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv')

df_train.head()


# remove id column
df_train.drop('id', axis = 1, inplace = True)
df_test.drop('id', axis = 1, inplace = True)

# check num of duplicated rows
print(f'Num of duplicated rows: {len(df_train[df_train.duplicated()])}')


# remove duplicates
df_train.drop_duplicates(inplace = True)


# check num of nulls
pd.DataFrame([df_train.isnull().sum(), df_test.isnull().sum()]).T\
    .rename(columns = {0: 'num of nulls in df_train', 1: 'num of nulls in df_test'})



# distribution of columns

plt.figure(figsize = (15, 15))
for i, col in enumerate([col for col in df_train.columns if col != 'id'], 1):
    plt.subplot(5, 3, i)
    sbn.histplot(df_train[col], color = 'green')
    plt.grid(axis = 'y')
plt.tight_layout()
plt.show()


# correlation
fig, axs = plt.subplots(1, 1, figsize = (9, 5))
sbn.heatmap(df_train.corr(numeric_only = True), annot = True, cmap = 'coolwarm', fmt = '.2f')


df = pd.concat([df_train, df_test])


# feature engineering (worse results - exclude from features)

# df['curvature_squared'] = df['curvature'] ** 2
# df['curvature_cubed'] = df['curvature'] ** 3
# df['speed_squared'] = df['speed_limit'] ** 2

# df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
# df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])

# df['speed_curvature'] = df['speed_limit'] * df['curvature']
# df['lanes_curvature'] = df['num_lanes'] * df['curvature']
# df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
# df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
# df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']

# df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int).astype('category')
# df['weather_lighting_risk'] = (
#     ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) &
#     ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
# ).astype(int).astype('category')

# df['is_night'] = (df['lighting'] == 'night').astype(int).astype('category')
# df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int).astype('category')
# df['is_highway'] = (df['road_type'] == 'highway').astype(int).astype('category')
# df['is_urban'] = (df['road_type'] == 'urban').astype(int).astype('category')

# df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int).astype('category')
# df['is_weekend'] = df['holiday'].astype(int).astype('category')

# df['safety_score'] = (
#     df['road_signs_present'].astype(int) * 2 +
#     (df['lighting'] == 'daylight').astype(int) +
#     (df['weather'] == 'clear').astype(int)
# ).astype('category')

# df['danger_score'] = (
#     (df['curvature'] > 0.6).astype(int) +
#     (df['speed_limit'] >= 60).astype(int) +
#     df['is_bad_weather'].astype(int) +
#     df['is_night'].astype(int) +
#     (df['num_reported_accidents'] >= 2).astype(int)
# ).astype('category')

# df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
# df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50


# categorical features
cat_cols = [col for col in df.select_dtypes([object, bool, 'category']).columns]


# Change type into 'category'

for col in cat_cols:
    df[col] = df[col].astype('category')
    
df.head()


# define features and target
X = df[df['accident_risk'].notnull()].drop('accident_risk', axis = 1)
y = df[df['accident_risk'].notnull()]['accident_risk']

# stratification of target
y_bin = pd.qcut(y, q = 10, labels = False, duplicates = 'drop')


# config optuna
# pruner = optuna.pruners.HyperbandPruner()


############### LGBM: optimize parameters by optuna #################

# def objective(trial):
#     param = {
#                 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'rf']),
#                 'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1, log = True),
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0, log = True),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, log = True)
#             }
        
#     model = lgb.LGBMRegressor(**param, random_state = 51, n_jobs = -1, early_stopping_rounds = 100, n_estimators = 2000, device = 'gpu')
    
#     scores = []
#     cv = StratifiedKFold(n_splits = 5, shuffle = True)
    
#     for train_idx, test_idx in cv.split(X, y_bin):
#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
#         model.fit(X_train, y_train, eval_metric = 'rmse', categorical_feature = cat_cols,
#                   eval_set = [(X_test, y_test)])
#         preds = model.predict(X_test)
        
#         score = np.sqrt(mean_squared_error(y_test, preds))
#         if score > 0.0564:            # We expect score about 0.56
#             scores.append(1)          # so we can break optimization for some set of parameters
#             break                     # if score will be very bad for some fold
#         else:                         # This way we will speed up the search for optimal parameters
#             scores.append(score)
    
#     return np.mean(scores)


# study = optuna.create_study(direction = 'minimize', pruner = pruner, load_if_exists = True)
# study.optimize(objective, timeout = 28800, n_jobs = -1, show_progress_bar = True)


############### XGB: optimize parameters by optuna #################

# def objective(trial):
#     param = {
#                 'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1, log = True),
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0, log = True),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, log = True)
#             }
        
#     model = xgb.XGBRegressor(**param, random_state = 51, n_jobs = -1, enable_categorical = True, tree_method = 'hist', device = 'cuda', 
#                              n_estimators = 5000, early_stopping_rounds = 100, eval_metric = 'rmse')
    
#     scores = []
#     cv = StratifiedKFold(n_splits = 5, shuffle = True)
    
#     for train_idx, test_idx in cv.split(X, y_bin):
#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
#         model.fit(X_train, y_train, verbose = 200,
#                   eval_set = [(X_test, y_test)])
#         preds = model.predict(X_test)
        
#         score = np.sqrt(mean_squared_error(y_test, preds))
#         if score > 0.0564:            # We expect score about 0.56
#             scores.append(1)          # so we can break optimization for some set of parameters
#             break                     # if score will be very bad for some fold
#         else:                         # This way we will speed up the search for optimal parameters
#             scores.append(score)
    
#     return np.mean(scores)


# study = optuna.create_study(direction = 'minimize', pruner = pruner, load_if_exists = True)
# study.optimize(objective, timeout = 28800, n_jobs = -1, show_progress_bar = True)


############### CatBoost: optimize parameters by optuna #################

# def objective(trial):
#     param = {
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0, log = True),
#                 'learning_rate': trial.suggest_uniform('learning_rate', 0.06, 0.12)
#                 # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0, log = True)
#             }
    
#     model = catb.CatBoostRegressor(**param, random_state = 51, cat_features = cat_cols, task_type = 'GPU',
#                              iterations = 5000, early_stopping_rounds = 100, eval_metric = 'RMSE', bootstrap_type = 'Bernoulli')
    
#     scores = []
#     cv = StratifiedKFold(n_splits = 5, shuffle = True)
    
#     for train_idx, test_idx in cv.split(X, y_bin):
#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
#         model.fit(X_train, y_train, verbose = 200, cat_features = cat_cols,
#                   eval_set = [(X_test, y_test)])
#         preds = model.predict(X_test)
        
#         score = np.sqrt(mean_squared_error(y_test, preds))
#         if score > 0.0564:            # We expect score about 0.56
#             scores.append(1)          # so we can break optimization for some set of parameters
#             break                     # if score will be very bad for some fold
#         else:                         # This way we will speed up the search for optimal parameters
#             scores.append(score)
    
#     return np.mean(scores)


# study = optuna.create_study(direction = 'minimize', pruner = pruner, load_if_exists = True)
# study.optimize(objective, n_trials = 200, timeout = 14400, show_progress_bar = True)


# models with the best parameters optimized by optuna

# LGBM
best_params = {
    'boosting_type': 'gbdt', 
    'learning_rate': 0.0360269510015689, 
    'subsample': 0.8059018900516028, 
    'colsample_bytree': 0.9625693024050926
}

best_model_lgbm = lgb.LGBMRegressor(
    **best_params, 
    random_state = 51, 
    n_jobs = -1, 
    n_estimators = 2000, 
    early_stopping_rounds = 100, 
    device = 'gpu'
)

# XGB
best_params = {
    'learning_rate': 0.018095111403323844, 
    'subsample': 0.8849524851971824, 
    'colsample_bytree': 0.9645096790114126
}
best_model_xgb = xgb.XGBRegressor(
    **best_params, 
    random_state = 51, 
    n_jobs = -1, 
    enable_categorical = True, 
    n_estimators = 5000, 
    early_stopping_rounds = 100, 
    eval_metric = 'rmse', 
    tree_method = 'hist', 
    device = 'cuda'
)

# CatBoost
best_params = {
    'subsample': 0.931753361976819, 
    'learning_rate': 0.07951639588772055
}
best_model_catb = catb.CatBoostRegressor(
    **best_params, 
    random_state = 51, 
    cat_features = cat_cols, 
    iterations = 5000, 
    early_stopping_rounds = 100, 
    eval_metric = 'RMSE', 
    task_type = 'GPU',
    bootstrap_type = 'Bernoulli'
)


models = [best_model_lgbm, best_model_xgb, best_model_catb]


# Prepare test dataset
df_test = df[df['accident_risk'].isnull()].drop('accident_risk', axis = 1)


# cross validation on train dataset and prediction

cv = StratifiedKFold(n_splits = 5, shuffle = True)
preds_model, scores_model = [], []

for model in models:

    preds, scores = [], []
    for train_idx, test_idx in cv.split(X, y_bin):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if type(model).__name__ == 'LGBMRegressor':
            model.fit(X_train, y_train, eval_metric = 'rmse', categorical_feature = cat_cols,
                  eval_set = [(X_test, y_test)])
        elif type(model).__name__ == 'XGBRegressor':
            model.fit(X_train, y_train, verbose = 200,
                  eval_set = [(X_test, y_test)])
        else:
            model.fit(X_train, y_train, verbose = 200, cat_features = cat_cols,
                  eval_set = [(X_test, y_test)])
            
        pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, pred))
        scores.append(score)

        pred = model.predict(df_test)
        preds.append(pred)

    scores_model.append(scores)
    preds_model.append(preds)


# sum up results

summary = pd.DataFrame([i for i in scores_model], index = ['lgbm', 'xgb', 'catb'])
for i in range(1, 6):
    summary.rename(columns = {i-1: f'Fold {i}'}, inplace = True)
summary['mean score'] = np.mean(summary, axis = 1)
summary['std score'] = np.std(summary.drop('mean score', axis = 1), axis = 1)
summary.sort_values(by = ['mean score', 'std score'])


def ensemble(preds, scores):
    inverse = []
    for model in scores:
        inverse.append([1 / a for a in model])

    sum_of_inverse = []
    for model in inverse:
        sum_of_inverse.append(np.sum(model))

    weights = []
    for i, model in enumerate(inverse):
        weights.append([a / sum_of_inverse[i] for a in model])

    weighted_preds = []
    for model in range(len(weights)):
        temp = []
        for j in range(len(weights[i])):
            temp.append(weights[model][j] * np.array(preds[model][j]))
        weighted_preds.append(temp)

    final_preds = []
    for i in weighted_preds:
        final_preds.append(list(map(sum, zip(*i))))
    
    return [final_preds]


ensemble_by_model = ensemble(preds_model, scores_model)
mean_score_by_model = [[np.mean(i) for i in scores_model]]
final_preds = ensemble(ensemble_by_model, mean_score_by_model)


sub = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv', usecols = ['id'])
sub['accident_risk'] = final_preds[0][0]
sub.head()


sub.to_csv(r'/kaggle/working/submission.csv', index = False)

