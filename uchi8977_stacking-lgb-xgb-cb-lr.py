import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

input_dir = '../input/playground-series-s5e9'
train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
test = pd.read_csv(os.path.join(input_dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))


def create_features(input_df):
    output_df = input_df.copy()
    output_df.drop('id', axis=1, inplace=True)
    
    # https://www.kaggle.com/code/shrutisaxena/playground-s5e9-predicting-the-beats
    output_df['TrackDurationMin'] = output_df['TrackDurationMs'] / 60000
    output_df['Energy_Acoustic_Ratio'] = output_df['Energy'] / (output_df['AcousticQuality'] + 1e-5)
    output_df['Vocal_Instrument_Balance'] = output_df['VocalContent'] / (output_df['InstrumentalScore'] + 1e-5)
    output_df['MoodRhythm'] = output_df['MoodScore'] * output_df['RhythmScore']
    output_df['PerformanceIntensity'] = output_df['LivePerformanceLikelihood'] * output_df['AudioLoudness']
    output_df['RhythmEnergy'] = output_df['RhythmScore'] * output_df['Energy']
    output_df['MoodAcoustic'] = output_df['MoodScore'] * output_df['AcousticQuality']

    return output_df


train_x = create_features(train.drop('BeatsPerMinute', axis=1))
train_y = train['BeatsPerMinute']
test_x = create_features(test)


class Results:
    def __init__(self, train_pred_oof, train_pred_oof_rmse, test_pred, models):
        self.train_pred_oof = train_pred_oof
        self.train_pred_oof_rmse = train_pred_oof_rmse
        self.test_pred = test_pred
        self.models = models


def run_fold(train_x, train_y, test_x, params, model_type, n_folds=5):
    fold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_pred_oof = np.zeros(len(train_x))
    test_pred = np.zeros((n_folds, len(test_x)))
    models = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold.split(train_x)):
        if model_type == 'lgb':
            tr_ds = lgb.Dataset(train_x.iloc[tr_idx], train_y[tr_idx])
            va_ds = lgb.Dataset(train_x.iloc[va_idx], train_y[va_idx])

            model = lgb.train(
                params,
                tr_ds,
                num_boost_round=10000,
                valid_sets=[va_ds],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            va_pred = model.predict(train_x.iloc[va_idx], num_iteration=model.best_iteration)
            train_pred_oof[va_idx] = va_pred

            test_pred[fold_idx, :] = model.predict(test_x, num_iteration=model.best_iteration)
            models.append(model)
        
        elif model_type == 'xgb':
            d_tr = xgb.DMatrix(train_x.iloc[tr_idx], label=train_y[tr_idx])
            d_va = xgb.DMatrix(train_x.iloc[va_idx],  label=train_y[va_idx])

            model = xgb.train(
                params=params,
                dtrain=d_tr,
                num_boost_round=10000,
                evals=[(d_va, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            va_pred = model.predict(xgb.DMatrix(train_x.iloc[va_idx]), iteration_range=(0, model.best_iteration + 1))
            train_pred_oof[va_idx] = va_pred

            test_pred[fold_idx, :] = model.predict(xgb.DMatrix(test_x), iteration_range=(0, model.best_iteration + 1))
            models.append(model)
        
        elif model_type == 'cat':
            x_tr, y_tr = train_x.iloc[tr_idx], train_y[tr_idx]
            x_va, y_va = train_x.iloc[va_idx], train_y[va_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                x_tr, y_tr,
                eval_set=(x_va, y_va),
                verbose=False,
                use_best_model=True,
                early_stopping_rounds=50,
            )

            va_pred = model.predict(x_va)
            train_pred_oof[va_idx] = va_pred

            test_pred[fold_idx, :] = model.predict(test_x)
            models.append(model)

    train_pred_oof_rmse = mean_squared_error(train_y, train_pred_oof, squared=False)
    mean_test_pred = test_pred.mean(axis=0)

    return Results(train_pred_oof, train_pred_oof_rmse, mean_test_pred, models)


def run_fold_lin(lgb_results, xgb_results, cat_results, train_y, n_folds=5):
    train_x = np.vstack([
        lgb_results.train_pred_oof,
        xgb_results.train_pred_oof,
        cat_results.train_pred_oof,
    ]).T
    test_x = np.vstack([
        lgb_results.test_pred,
        xgb_results.test_pred,
        cat_results.test_pred,
    ]).T

    fold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_pred_oof = np.zeros(len(train_x))
    test_pred = np.zeros((n_folds, len(test_x)))
    models = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold.split(train_x)):
        x_tr, y_tr = train_x[tr_idx], train_y[tr_idx]
        x_va, y_va = train_x[va_idx], train_y[va_idx]

        model = LinearRegression()
        model.fit(x_tr, y_tr)

        va_pred = model.predict(x_va)
        train_pred_oof[va_idx] = va_pred

        test_pred[fold_idx, :] = model.predict(test_x)
        models.append(model)

    train_pred_oof_rmse = mean_squared_error(train_y, train_pred_oof, squared=False)
    mean_test_pred = test_pred.mean(axis=0)

    return Results(train_pred_oof, train_pred_oof_rmse, mean_test_pred, models)


def get_best_params(train_x, train_y, test_x, model_type, n_trials=30):
    if model_type == 'lgb':
        fixed_params = {
            'metric': 'rmse',
            'verbose': -1,
            'seed': 42,
            'bagging_seed': 42,
            'feature_fraction_seed': 42,
            'objective': 'regression',
        }

        def objective(trial):
            params = {
                **fixed_params,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, .3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'feature_fraction': trial.suggest_float('feature_fraction', .4, 1.),
            }
            results = run_fold(train_x, train_y, test_x, params, 'lgb')
            return results.train_pred_oof_rmse

    elif model_type == 'xgb':
        fixed_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'random_state': 42,
        }

        def objective(trial):
            params = {
                **fixed_params,
                'eta': trial.suggest_float('eta', 1e-3, .3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'max_leaves': trial.suggest_int('max_leaves', 31, 255),
                'colsample_bytree': trial.suggest_float('colsample_bytree', .4, 1.),
            }
            results = run_fold(train_x, train_y, test_x, params, 'xgb')
            return results.train_pred_oof_rmse

    elif model_type == 'cat':
        fixed_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'iterations': 10000,
            'random_seed': 42,
        }

        def objective(trial):
            params = {
                **fixed_params,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, .3, log=True),
                'depth': trial.suggest_int('depth', 3, 12),
                'rsm': trial.suggest_float('rsm', .4, 1.0),
            }
            results = run_fold(train_x, train_y, test_x, params, 'cat')
            return results.train_pred_oof_rmse

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {**fixed_params, **study.best_params}


params_lgb = get_best_params(train_x, train_y, test_x, 'lgb')
params_xgb = get_best_params(train_x, train_y, test_x, 'xgb')
params_cat = get_best_params(train_x, train_y, test_x, 'cat')


results_lgb = run_fold(train_x, train_y, test_x, params_lgb, 'lgb')
print('lightgbm oof rmse:', results_lgb.train_pred_oof_rmse)

results_xgb = run_fold(train_x, train_y, test_x, params_xgb, 'xgb')
print('xgboost oof rmse:', results_xgb.train_pred_oof_rmse)

results_cat = run_fold(train_x, train_y, test_x, params_cat, 'cat')
print('catboost oof rmse:', results_cat.train_pred_oof_rmse)

results_lin = run_fold_lin(results_lgb, results_xgb, results_cat, train_y)
print('[final] linear-regression oof rmse:', results_lin.train_pred_oof_rmse)


def visualize_importance(models, x_train, ax, model_type):
    feature_importances = pd.DataFrame()
    for i, model in enumerate(models):
        tmp = pd.DataFrame()
        if model_type == 'lgb':
            tmp['feature_importance'] = model.feature_importance(importance_type='gain')
        elif model_type == 'xgb':
            tmp['feature_importance'] = model.get_score(importance_type='gain')
        elif model_type == 'cat':
            tmp['feature_importance'] = model.get_feature_importance(type='PredictionValuesChange')

        tmp['column'] = x_train.columns
        tmp['fold'] = i + 1
        feature_importances = pd.concat([feature_importances, tmp], axis=0, ignore_index=True)

    order = feature_importances.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    abbr_to_official = {'lgb': 'LightGBM', 'xgb': 'XGBoost', 'cat': 'CatBoost'}
    ax.set_title(abbr_to_official[model_type])
    sns.boxenplot(data=feature_importances, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    return ax


fig, axes = plt.subplots(1, 3, figsize=(15, 8))
visualize_importance(results_lgb.models, train_x, axes[0], 'lgb')
visualize_importance(results_xgb.models, train_x, axes[1], 'xgb')
visualize_importance(results_cat.models, train_x, axes[2], 'cat')
fig.tight_layout()


fig, ax = plt.subplots(figsize=(8, 8))
sns.histplot(results_lin.test_pred, binwidth=0.2, alpha=0.3, kde=True, label='Test Predict')
sns.histplot(results_lin.train_pred_oof, binwidth=0.2, alpha=0.3, kde=True, label='Out Of Fold')
ax.set_xlim([116, 122])
ax.legend()
ax.grid()


sample_submission['BeatsPerMinute'] = results_lin.test_pred
sample_submission.to_csv('submission_xxx.csv', index=False)

