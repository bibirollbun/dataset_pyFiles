import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import optuna
import math
import warnings

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
seed = 42


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print("train: ", train.shape)
print("test: ", test.shape)
print("sample_submission: ", sample.shape)


train.isnull().sum().sort_values(ascending=False)


test.isnull().sum().sort_values(ascending=False)


def fill_missing(df):
    imputer = SimpleImputer(strategy='most_frequent')

    grouped = df.groupby('Podcast_Name')
    for title, group in grouped:
        group['Episode_Length_minutes'] = imputer.fit_transform(group[['Episode_Length_minutes']])
        df.loc[group.index, 'Episode_Length_minutes'] = group['Episode_Length_minutes']
        group['Guest_Popularity_percentage'] = imputer.fit_transform(group[['Guest_Popularity_percentage']])
        df.loc[group.index, 'Guest_Popularity_percentage'] = group['Guest_Popularity_percentage']

    df['Number_of_Ads'] = imputer.fit_transform(df[['Number_of_Ads']])

    return df

train = fill_missing(train)
test = fill_missing(test)


X = train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
test = test.drop('id', axis=1)


numeric_features = test.select_dtypes(include=['number']).columns.tolist()
print("numeric_features", numeric_features)

categorical_features = test.select_dtypes(include=['object']).columns.tolist()
print("categorical_features", categorical_features)


def handle_numeric_features(df):
    num_features_len = len(numeric_features)
    new_columns = {}
    
    for first_feature in range(num_features_len):
        f1 = numeric_features[first_feature]
        new_columns.update({
            f'{f1}_sin': np.sin(df[f1]),
            f'{f1}_cos': np.cos(df[f1]),
        })
        
        for second_feature in range(first_feature + 1, num_features_len):
            f2 = numeric_features[second_feature]

            new_columns.update({
                f'{f1}_plus_{f2}': df[f1] + df[f2],
                f'{f1}_minus_{f2}': df[f1] - df[f2],
                f'{f1}_times_{f2}': df[f1] * df[f2],
                f'{f1}_log_{f2}': np.log1p(np.abs(df[f1])) - np.log1p(np.abs(df[f2])),
                f'{f1}_sqrt_{f2}': np.sqrt(np.abs(df[f1])) - np.sqrt(np.abs(df[f2])),
                f'{f1}_abs_diff_{f2}': np.abs(df[f1] - df[f2]),
            })

    return pd.concat([df, pd.DataFrame(new_columns)], axis=1)

X = handle_numeric_features(X)
test = handle_numeric_features(test)


def handle_categorical_features(df):
    df['Episode_Title_Length'] = df.apply(lambda row: len(row['Episode_Title']), axis=1)
    df['Has_Ads'] = df.apply(lambda row: 1 if row['Number_of_Ads'] > 0 else 0, axis=1)
    df['Popularity'] = df.apply(lambda row: 1 if (row['Host_Popularity_percentage'] > 60 and row['Guest_Popularity_percentage'] > 60) else 0, axis=1)
    
    df['Podcast_Name'] = df['Podcast_Name'].factorize()[0]
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '').astype(int)
    df['Genre'] = df['Genre'].factorize()[0]
    df['Publication_Day'] = df['Publication_Day'].map({
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    })
    df['Publication_Time'] = df['Publication_Time'].map({
        "Morning": 0,
        "Afternoon": 1,
        "Night": 2,
        "Evening": 3,
    })
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map({
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    })

    df['Publication_Day_Weekend'] = (df['Publication_Day'] > 4).astype(int)
    df['Publication_Day_Monday'] = (df['Publication_Day'] == 0).astype(int)
    df['Publication_Day_Friday'] = (df['Publication_Day'] == 4).astype(int)
    df['Publication_Day_Sunday'] = (df['Publication_Day'] == 6).astype(int)
    df['Publication_Time_Late'] = (df['Publication_Time'] > 1).astype(int)

    df['Publication_Day_Work_Early'] = df.apply(lambda row: 1 if row['Publication_Day_Weekend'] == 0 and row['Publication_Time_Late'] == 0 else 0, axis=1)
    df['Publication_Day_Work_Late'] = df.apply(lambda row: 1 if row['Publication_Day_Weekend'] == 0 and row['Publication_Time_Late'] == 1 else 0, axis=1)
    df['Publication_Day_Weekend_Early'] = df.apply(lambda row: 1 if row['Publication_Day_Weekend'] == 1 and row['Publication_Time_Late'] == 0 else 0, axis=1)
    df['Publication_Day_Weekend_Late'] = df.apply(lambda row: 1 if row['Publication_Day_Weekend'] == 1 and row['Publication_Time_Late'] == 1 else 0, axis=1)
    df['Publication_Day_Monday_Early'] = df.apply(lambda row: 1 if row['Publication_Day_Monday'] == 1 and row['Publication_Time_Late'] == 0 else 0, axis=1)
    df['Publication_Day_Monday_Late'] = df.apply(lambda row: 1 if row['Publication_Day_Monday'] == 1 and row['Publication_Time_Late'] == 1 else 0, axis=1)
    df['Publication_Day_Friday_Early'] = df.apply(lambda row: 1 if row['Publication_Day_Friday'] == 1 and row['Publication_Time_Late'] == 0 else 0, axis=1)
    df['Publication_Day_Friday_Late'] = df.apply(lambda row: 1 if row['Publication_Day_Friday'] == 1 and row['Publication_Time_Late'] == 1 else 0, axis=1)
    df['Publication_Day_Sunday_Early'] = df.apply(lambda row: 1 if row['Publication_Day_Sunday'] == 1 and row['Publication_Time_Late'] == 0 else 0, axis=1)
    df['Publication_Day_Sunday_Late'] = df.apply(lambda row: 1 if row['Publication_Day_Sunday'] == 1 and row['Publication_Time_Late'] == 1 else 0, axis=1)

    columns_to_combine = [
        'Publication_Day_Weekend', 'Publication_Day_Monday', 'Publication_Day_Friday',
        'Publication_Day_Sunday', 'Publication_Time_Late', 'Publication_Day_Work_Early',
        'Publication_Day_Work_Late', 'Publication_Day_Weekend_Early', 'Publication_Day_Weekend_Late',
        'Publication_Day_Monday_Early', 'Publication_Day_Monday_Late', 'Publication_Day_Friday_Early',
        'Publication_Day_Friday_Late', 'Publication_Day_Sunday_Early', 'Publication_Day_Sunday_Late'
    ]

    for col in columns_to_combine:
        for sentiment_value in df['Episode_Sentiment'].unique():
            df[f'{col}_{sentiment_value}'] = df[col] * (df['Episode_Sentiment'] == sentiment_value)

        df[f'{col}_Host_Popularity_percentage'] = df[col] * (df['Host_Popularity_percentage'] > 60)
        df[f'{col}_Guest_Popularity_percentage'] = df[col] * (df['Guest_Popularity_percentage'] > 60)
        df[f'{col}_Popularity'] = df[col] * df['Popularity']

    return df

X = handle_categorical_features(X)
test = handle_categorical_features(test)


X = X.sort_index(axis=1)
test = test.sort_index(axis=1)

print("X: ", X.shape)
print("test: ", test.shape)


models = []

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state=seed)
temp_train_X, temp_valid_X, temp_train_y, temp_valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=seed)


# def objective(trial):
#     params = {
#         'n_estimators': 300,
#         'max_depth': trial.suggest_int('max_depth', 8, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 30, 100),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 80),
#         'max_features': trial.suggest_float('max_features', 0.5, 1.0),
#         'random_state': seed,
#         'n_jobs': -1
#     }

#     model = RandomForestRegressor(**params)
    
#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)


# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True,
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"RandomForestRegressor best rmse: {study.best_value:.4f}")
# print("RandomForestRegressor best parmas:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# models.append(RandomForestRegressor(**study.best_params))


# RandomForestRegressor best mse: 13.0819
# RandomForestRegressor best parmas:
# max_depth: 20
# min_samples_split: 35
# min_samples_leaf: 20
# max_features: 0.7808497396306461

models.append(RandomForestRegressor(**{
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 35,
    'min_samples_leaf': 20,
    'max_features': 0.7808497396306461,
    'random_state': seed,
    'n_jobs': -1
}))


# def objective(trial):
#     params = {
#         'n_estimators': 300,
#         'max_depth': trial.suggest_int('max_depth', 8, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 30, 100),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 80),
#         'max_features': trial.suggest_float('max_features', 0.5, 1.0),
#         'random_state': seed,
#         'n_jobs': -1
#     }

#     model = ExtraTreesRegressor(**params)

#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)

# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"ExtraTreesRegressor best rmse: {study.best_value:.4f}")
# print("ExtraTreesRegressor best parmas:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# models.append(ExtraTreesRegressor(**study.best_params))


# ExtraTreesRegressor best rmse: 13.4099
# ExtraTreesRegressor best parmas:
# max_depth: 20
# min_samples_split: 89
# min_samples_leaf: 47
# max_features: 0.9205246940134206

models.append(ExtraTreesRegressor(**{
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 89,
    'min_samples_leaf': 47,
    'max_features': 0.9205246940134206,
    'random_state': seed,
    'n_jobs': -1
}))


# def objective(trial):
#     params = {
#         'random_state': seed,
#         'n_estimators': 300,
#         "verbosity": 0,
#         'eval_metric': 'rmse',
#         'max_depth': trial.suggest_int('max_depth', 8, 20),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
#         'gamma': trial.suggest_float('gamma', 0.0, 5.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
#     }

#     model = XGBRegressor(**params)

#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)

# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"XGBRegressor best rmse: {study.best_value:.4f}")
# print("XGBRegressor best parmas:")
# for key, value in study.best_params.items():
#   print(f"{key}: {value}")

# models.append(XGBRegressor(**study.best_params))


# XGBRegressor best rmse: 13.0536
# XGBRegressor best parmas:
# max_depth: 13
# learning_rate: 0.08063097625242895
# colsample_bytree: 0.5253044198221457
# subsample: 0.8463050954881912
# min_child_weight: 1
# gamma: 4.803678838939221
# reg_alpha: 0.8820751130482305
# reg_lambda: 0.18498257048067107

models.append(XGBRegressor(**{
    'random_state': seed,
    'n_estimators': 300,
    "verbosity": 0,
    'eval_metric': 'rmse',
    'max_depth': 13,
    'learning_rate': 0.08063097625242895,
    'colsample_bytree': 0.5253044198221457,
    'subsample': 0.8463050954881912,
    'min_child_weight': 1,
    'gamma': 4.803678838939221,
    'reg_alpha': 0.8820751130482305,
    'reg_lambda': 0.18498257048067107
}))


# def objective(trial):
#     params = {
#         'random_state': seed,
#         'max_iter': 300,
#         'verbose': 0,
#         'max_depth': trial.suggest_int('max_depth', 8, 20),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
#     }

#     model = HistGradientBoostingRegressor(**params)
    
#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)

# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"HistGradientBoostingRegressor best rmse: {study.best_value:.4f}")
# print("HistGradientBoostingRegressor best parmas:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# models.append(HistGradientBoostingRegressor(**study.best_params))


# HistGradientBoostingRegressor best rmse: 13.2436
# HistGradientBoostingRegressor best parmas:
# max_depth: 14
# learning_rate: 0.21719443482331074

models.append(HistGradientBoostingRegressor(**{
    'random_state': seed,
    'max_iter': 300,
    'verbose': 0,
    'max_depth': 14,
    'learning_rate': 0.21719443482331074
}))


# import optuna
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     params = {
#         'random_state': seed,
#         'n_estimators': 300,
#         'verbose': 0,
#         "eval_metric": "RMSE",
#         'depth': trial.suggest_int('depth', 8, 16),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.6, 1.0),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50),
#     }

#     model = CatBoostRegressor(**params)
    
#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)

# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"CatBoostRegressor best rmse: {study.best_value:.4f}")
# print("CatBoostRegressor best parmas:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# models.append(CatBoostRegressor(**study.best_params))


# CatBoostRegressor best rmse: 13.2362
# CatBoostRegressor best parmas:
# depth: 9
# learning_rate: 0.40946591516773606
# l2_leaf_reg: 4.557171356193724
# bagging_temperature: 0.6488887154665768
# min_data_in_leaf: 30

models.append(CatBoostRegressor(**{
    'random_state': seed,
    'n_estimators': 300,
    'verbose': 0,
    "eval_metric": "RMSE",
    'depth': 9,
    'learning_rate': 0.40946591516773606,
    'l2_leaf_reg': 4.557171356193724,
    'bagging_temperature': 0.6488887154665768,
    'min_data_in_leaf': 30
}))


# import optuna
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     params = {
#         'random_state': seed,
#         'n_estimators': 300,
#         'verbose': -1,
#         'num_leaves': trial.suggest_int('num_leaves', 50, 200),
#         'max_depth': trial.suggest_int('max_depth', 8, 20),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5),
#         'min_child_samples': trial.suggest_int('min_child_samples', 50, 150),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
#         'eval_metric': 'rmse',
#     }

#     model = LGBMRegressor(**params)
    
#     # too slow
#     # mse = -(cross_val_score(
#     #     model,
#     #     X=train_X,
#     #     y=train_y,
#     #     cv=5,
#     #     scoring='neg_mean_squared_error'
#     # ).mean())

#     model.fit(temp_train_X, temp_train_y)
#     y_pred = model.predict(temp_valid_X)
#     mse = mean_squared_error(temp_valid_y, y_pred)

#     return math.sqrt(mse)

# sampler = optuna.samplers.TPESampler(
#     n_startup_trials=5,
#     multivariate=True
# )

# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=10)

# print(f"LGBMRegressor best rmse: {study.best_value:.4f}")
# print("LGBMRegressor best parmas:")
# for key, value in study.best_params.items():
#     print(f"{key}: {value}")

# models.append(LGBMRegressor(**study.best_params))


# LGBMRegressor best rmse: 13.1067
# LGBMRegressor best parmas:
# num_leaves: 165
# max_depth: 14
# learning_rate: 0.13807208049154535
# min_child_samples: 71
# subsample: 0.6240517655588945
# colsample_bytree: 0.8787979863752625
# reg_alpha: 0.4592471476344985
# reg_lambda: 0.7342798064934071

models.append(LGBMRegressor(**{
    'random_state': seed,
    'n_estimators': 300,
    'verbose': -1,
    'num_leaves': 165,
    'max_depth': 14,
    'learning_rate': 0.13807208049154535,
    'min_child_samples': 71,
    'subsample': 0.6240517655588945,
    'colsample_bytree': 0.8787979863752625,
    'reg_alpha': 0.4592471476344985,
    'reg_lambda': 0.7342798064934071,
    'eval_metric': 'rmse',
}))


preds = []

for index, model in enumerate(models):
    model.fit(train_X, train_y)
    pred = model.predict(valid_X)
    preds.append(pred)
    print(f'model {index}: {math.sqrt(mean_squared_error(valid_y, pred))}')


# initial_weights = np.ones(len(models)) / len(models)

# def objective(weights):
#     weighted_pred = np.sum([w * pred for w, pred in zip(weights, preds)], axis=0)
#     mse = mean_squared_error(valid_y, weighted_pred)
#     return mse

# constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
# bounds = [(0, 1) for _ in range(len(models))]

# result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100})
# optimal_weights = result.x

# print(f'optimal_weights: {optimal_weights}')

optimal_weights = [3.12515955e-01, 8.46363458e-18, 5.74366524e-01, 2.21914540e-16, 2.02150276e-15, 1.13117521e-01]


optimal_pred = np.sum([w * pred for w, pred in zip(optimal_weights, preds)], axis=0)
print(f'optimal_pred: {math.sqrt(mean_squared_error(valid_y, optimal_pred))}')


final_pred = np.sum([w * model.predict(test) for w, model in zip(optimal_weights, models)], axis=0)


submission = pd.DataFrame({'id': sample['id'], 'Listening_Time_minutes': final_pred})
print(submission.head())
submission.to_csv('submission.csv', index=False)

