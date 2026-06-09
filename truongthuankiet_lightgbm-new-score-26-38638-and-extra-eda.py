import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
# import train_test_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')  
test_idx = test['id']
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
train.head()


train.info()


train.describe()



train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


# Distribution of features
num_features = train.select_dtypes(include=[np.number]).columns.tolist()

train[num_features].hist(bins=15, figsize=(15, 10), layout=(4, 4))
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.show()



# Correlation heatmap
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()



# Check outliers using box plot
train[num_features].plot(kind='box', subplots=True, layout=(4,4), figsize=(15,10), sharex=False, sharey=False)


# Import OLS
import statsmodels.api as sm
X_tmp = train.select_dtypes(include=[np.number]).drop(columns=['BeatsPerMinute'])
X_tmp['AudioLoudness'] = abs(X_tmp['AudioLoudness'])
X_tmp = sm.add_constant(X_tmp)
y_tmp = train['BeatsPerMinute']

print(sm.OLS(y_tmp, X_tmp).fit().summary())



# Check linearity between energy and BeatsPerMinute
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train['Energy'], y=train['BeatsPerMinute'])
plt.xlabel('Energy')
plt.ylabel('BeatsPerMinute')
plt.title('Scatter plot of Energy vs BeatsPerMinute')
plt.show()

# Calculate and display correlation coefficient
corr = train['Energy'].corr(train['BeatsPerMinute'])
print(f'Correlation coefficient between Energy and BeatsPerMinute: {corr:.4f}')


df_tmp = train.copy()
# Define custom BPM bins and labels based on genres
bpm_bins = [0, 70, 110, 128, 136, 145, 160, 200, np.inf]
bpm_labels = [
    'Below Trap (<70)',
    'Trap (70-110)',
    'Deep/Tech House (110-128)',
    'Breakbeat (128-136)',
    'Prog/Electro House (136-145)',
    'Liquid DnB/Dubstep (145-160)',
    'Hard DnB/Jungle (160-200)',
    'Very Fast (>200)'
]
df_tmp['BeatsPerMinute_genre_bin'] = pd.cut(df_tmp['BeatsPerMinute'], bins=bpm_bins, labels=bpm_labels, right=False)
# Custom color palette for each genre bin (8 colors for 8 bins)
custom_palette = [
# Below Trap (<70)
    '#8dd3c7',
# Trap (70-110)
    '#ffffb3',
# Deep/Tech House (110-128)
    '#bebada',
# Breakbeat (128-136)
    '#fb8072',
# Prog/Electro House (136-145)
    '#80b1d3',
# Liquid DnB/Dubstep (145-160)
    '#fdb462',
# Hard DnB/Jungle (160-200)
    '#b3de69',
# Very Fast (>200)
    '#fccde5'
]
sns.countplot(data=df_tmp, x='BeatsPerMinute_genre_bin', hue='BeatsPerMinute_genre_bin', palette=custom_palette)
plt.title('Count plot of Energy by Genre BPM bins')
plt.xticks(rotation=30, ha='right')


df_tmp['BeatsPerMinute_genre_bin'].value_counts()


plt.Figure(figsize=(20, 15))
# Bin acousticQuality
df_tmp['AcousticQuality_bin'] = pd.cut(df_tmp['AcousticQuality'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
sns.boxplot(y='AcousticQuality_bin', x='LivePerformanceLikelihood', hue='BeatsPerMinute_genre_bin', data=df_tmp, palette=custom_palette)
plt.title('Box plot of Acoustic Quality vs Live Performance Likelihood by Genre BPM bins')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.tight_layout()
plt.show();


# selected_cols = ['RhythmScore', 'VocalContent', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'AudioLoudness']
from sklearn.preprocessing import StandardScaler
def feature_engineer(df):
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Loudness_norm'] = df['AudioLoudness'] + 30
    df['LivePerformanceMood'] = df['LivePerformanceLikelihood'] * df['MoodScore']
    df['Rhythm_sqr'] = df['RhythmScore'] ** 2
    df['Vocal_sqr'] = df['VocalContent'] ** 2
    df['Mood_sqr'] = df['MoodScore'] ** 2
    df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']
    return df
# outlier_bins = ['Below Trap (<70)', 'Very Fast (>200)']
# train = df_tmp.copy()
# train = train[~train['BeatsPerMinute_genre_bin'].isin(outlier_bins)]

# train = train[selected_cols + ['BeatsPerMinute']]
scaler = StandardScaler()
X = feature_engineer(train.drop(columns=['BeatsPerMinute']))
X = scaler.fit_transform(X)
y = train['BeatsPerMinute']
# test = test[selected_cols]
test = feature_engineer(test)
# Standardize
test = scaler.transform(test)

scaler_for_y = StandardScaler()
y = scaler_for_y.fit_transform(y.values.reshape(-1, 1))


# def objective(trial):
#         param = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'boosting_type': 'gbdt',
#             'verbosity': -1,
#             'seed': 42,
#             'feature_pre_filter': False,
#             'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#             'max_depth': trial.suggest_int('max_depth', 4, 16),
#             'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
#             'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
#             'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#             'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
#             'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#             'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#             'n_jobs': -1
#         }
#         kf = KFold(n_splits=3, shuffle=True, random_state=42)
#         oof_preds = np.zeros(len(X))
#         for train_idx, val_idx in kf.split(X):
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
#             train_data = lgb.Dataset(X_train, label=y_train)
#             val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
#             model = lgb.train(
#                 param,
#                 train_data,
#                 valid_sets=[val_data],
#                 num_boost_round=10000,
#                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
#             )
#             val_pred = model.predict(X_val, num_iteration=model.best_iteration)
#             oof_preds[val_idx] = val_pred
#         score = np.sqrt(mean_squared_error(y, oof_preds))
#         return score

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=15)
# print("Best trial:", study.best_trial.params)
best_params = {'learning_rate': 0.0016633000546178811, 'num_leaves': 22, 'max_depth': 11, 'feature_fraction': 0.8508821665691166, 'bagging_fraction': 0.9234262384167368, 'bagging_freq': 1, 'min_child_samples': 88, 'lambda_l1': 0.09367488668412283, 'lambda_l2': 0.032104017717388585}
def train_optuna_lightgbm(X, y, X_test):
    # best_params = study.best_trial.params
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        # X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        # y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, X_val = X[train_idx], X[val_idx]   
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(0)
            ]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5

        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(fold_rmse)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    cv_score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Optuna LightGBM CV RMSE: {cv_score:.5f}")
    print(f"Fold Scores Std: {np.std(fold_scores):.5f}")

    return oof_preds, test_preds, cv_score


print("=== TUNED LIGHTGBM ===")
tuned_lgb_oof, tuned_lgb_test, tuned_lgb_score = train_optuna_lightgbm(X, y, test)
# Inverse transform the predictions
tuned_lgb_test = scaler_for_y.inverse_transform(tuned_lgb_test.reshape(-1, 1)).flatten()


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
    score = np.sqrt(mean_squared_error(y, oof_preds))
    return score

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_xgb, n_trials=15)

# print('Best XGB params:', study.best_trial.params)

# def train_xgb_optuna(X, y, X_test, best_params):
#     params = best_params.copy()
#     params.update({'n_estimators': 1000, 'random_state': 42, 'tree_method': 'hist', 'n_jobs': -1})
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X))
#     test_preds = np.zeros(len(X_test))
#     fold_scores = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"Fold {fold + 1}")
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = XGBRegressor(**params)
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=0)

#         val_pred = model.predict(X_val)
#         test_pred = model.predict(X_test)

#         oof_preds[val_idx] = val_pred
#         test_preds += test_pred / 5

#         fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#         fold_scores.append(fold_rmse)
#         print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

#     cv_score = np.sqrt(mean_squared_error(y, oof_preds))
#     print(f"Optuna XGBoost CV RMSE: {cv_score:.5f}")
#     print(f"Fold Scores Std: {np.std(fold_scores):.5f}")

#     return oof_preds, test_preds, cv_score

# print("=== OPTUNA XGBOOST ===")
# xgb_oof, xgb_test, xgb_score = train_xgb_optuna(X, y, test, study.best_trial.params)


def train_xgb(X, y, X_test):
    params = {'n_estimators': 181, 'learning_rate': 0.02789545798446095, 'max_depth': 3, 'subsample': 0.753412491579697, 'colsample_bytree': 0.9451121607929804, 'gamma': 0.2126285839424732, 'reg_alpha': 2.2638741043025927, 'reg_lambda': 7.950652578123739}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5

        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(fold_rmse)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    cv_score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"XGBoost CV RMSE: {cv_score:.5f}")
    print(f"Fold Scores Std: {np.std(fold_scores):.5f}")

    return oof_preds, test_preds, cv_score

print("=== XGBOOST ===")
xgb_oof, xgb_test, xgb_score = train_xgb(X, y, test)
xgb_test = scaler_for_y.inverse_transform(xgb_test.reshape(-1, 1)).flatten()


import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# def objective_catboost(trial):
#     params = {
#         'iterations': 1000,
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
#         'depth': trial.suggest_int('depth', 3, 12),
#         'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
#         'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
#         'random_strength': trial.suggest_uniform('random_strength', 0, 1),
#         'random_seed': 42,
#         'loss_function': 'RMSE',
#         'eval_metric': 'RMSE',
#         'verbose': 0,
#         'task_type': 'CPU'
#     }
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X))
#     for train_idx, val_idx in kf.split(X):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#         model = CatBoostRegressor(**params)
#         model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
#         val_pred = model.predict(X_val)
#         oof_preds[val_idx] = val_pred
#     score = np.sqrt(mean_squared_error(y, oof_preds))
#     return score

# study = optuna.create_study(direction='minimize')
# study.optimize(objective_catboost, n_trials=15)

# print('Best CatBoost params:', study.best_trial.params)

# def train_catboost_optuna(X, y, X_test, best_params):
#     params = best_params.copy()
#     params.update({'iterations': 1000, 'random_seed': 42, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'verbose': 0, 'task_type': 'CPU'})
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X))
#     test_preds = np.zeros(len(X_test))
#     fold_scores = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"Fold {fold + 1}")
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = CatBoostRegressor(**params)
#         model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

#         val_pred = model.predict(X_val)
#         test_pred = model.predict(X_test)

#         oof_preds[val_idx] = val_pred
#         test_preds += test_pred / 5

#         fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#         fold_scores.append(fold_rmse)
#         print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

#     cv_score = np.sqrt(mean_squared_error(y, oof_preds))
#     print(f"Optuna CatBoost CV RMSE: {cv_score:.5f}")
#     print(f"Fold Scores Std: {np.std(fold_scores):.5f}")

#     return oof_preds, test_preds, cv_score

# print("=== OPTUNA CATBOOST ===")
# cat_oof, cat_test, cat_score = train_catboost_optuna(X, y, test, study.best_trial.params)


# def train_catboost_optuna(X, y, X_test):
#     params = {'learning_rate': 0.008107431983794362, 
#               'depth': 6, 
#               'l2_leaf_reg': 4.384321216169613e-06, 
#               'bagging_temperature': 0.45885341010502595, 
#               'random_strength': 0.23314032786675742}
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X))
#     test_preds = np.zeros(len(X_test))

#     fold_scores = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"Fold {fold + 1}")
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         model = CatBoostRegressor(**params)
#         model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

#         val_pred = model.predict(X_val)
#         test_pred = model.predict(X_test)

#         oof_preds[val_idx] = val_pred
#         test_preds += test_pred / 5

#         fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#         fold_scores.append(fold_rmse)
#         print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

#     cv_score = np.sqrt(mean_squared_error(y, oof_preds))
#     print(f"Optuna CatBoost CV RMSE: {cv_score:.5f}")
#     print(f"Fold Scores Std: {np.std(fold_scores):.5f}")

#     return oof_preds, test_preds, cv_score

# print("=== OPTUNA CATBOOST ===")
# cat_oof, cat_test, cat_score = train_catboost_optuna(X, y, test)


# # Submission
# submission = pd.DataFrame({
#     'id': test_idx,
#     'BeatsPerMinute': xgb_test
# })
# submission.to_csv('xgb_submission_added_all_features_scaled.csv', index=False)


# Submission
submission = pd.DataFrame({
    'id': test_idx,
    'BeatsPerMinute': tuned_lgb_test
})
submission.to_csv('submission_lgb_all_features_new_features_scaled_mood^2.csv', index=False)


# submission = pd.DataFrame({
#     'id': test_idx,
#     'BeatsPerMinute': cat_test
# })
# submission.to_csv('cat_submission.csv', index=False)

