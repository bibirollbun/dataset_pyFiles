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


%%time

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy import optimize as opt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/chydv-hackathon-2025/train.csv')
test = pd.read_csv('/kaggle/input/chydv-hackathon-2025/test.csv')
sample = pd.read_csv("/kaggle/input/chydv-hackathon-2025/sample_submission.csv")

# Preprocessing
train['quality'] = train['quality'].astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
train = train.drop_duplicates()

# Feature Engineering
def feature_engineering(df):
    df['feature_sum'] = df.sum(axis=1)
    df['feature_mean'] = df.mean(axis=1)
    df['feature_std'] = df.std(axis=1)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Quadratic Kappa Metric
def quadratic_kappa(actuals, preds, N=6):
    O = confusion_matrix(actuals, preds)
    w = np.zeros((N, N))
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float((i - j) ** 2 / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)
    O = O / O.sum()
    E = E / E.sum()

    num, den = 0, 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return 1 - (num / den)

# Punctuate Function
def punctuate(threshold, pred_prob):
    pred_exp = pred_prob * range(6)
    pred_srial = pred_exp.sum(axis=1)
    pred_thresholded = np.zeros([len(pred_srial)])

    t0, t1, t2, t3, t4 = threshold
    for i, p in enumerate(pred_srial):
        if p <= t0:
            pred_thresholded[i] = 0
        elif p <= t1:
            pred_thresholded[i] = 1
        elif p <= t2:
            pred_thresholded[i] = 2
        elif p <= t3:
            pred_thresholded[i] = 3
        elif p <= t4:
            pred_thresholded[i] = 4
        else:
            pred_thresholded[i] = 5
    return pred_thresholded.astype(int)

# Ensemble Model Training
def TRAIN_ENSEMBLE(train, test, SEED=0):
    x = train.drop(['quality'], axis=1)
    y = train['quality']
    y_shift = y - 3
    test_x = test

    preds_tests_xgb, preds_tests_lgb = [], []
    oof_preds_xgb, oof_preds_lgb = np.zeros(len(train)), np.zeros(len(train))
    validation_scores_xgb, validation_scores_lgb = [], []

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=SEED)

    for fold, (ids_train, ids_valid) in enumerate(skf.split(x, y_shift)):
        tr_x, va_x = x.iloc[ids_train, :], x.iloc[ids_valid, :]
        tr_y, va_y = y_shift.iloc[ids_train], y_shift.iloc[ids_valid]

        # XGBoost
        dtrain_xgb = xgb.DMatrix(tr_x, label=tr_y)
        dvalid_xgb = xgb.DMatrix(va_x, label=va_y)
        dvax_xgb = xgb.DMatrix(va_x)
        dtest_xgb = xgb.DMatrix(test_x)

        params_xgb = {
            'objective': 'multi:softprob',
            'eval_metric': "mlogloss",
            'num_class': 6,
            'eta': 0.1,
            'gamma': 0.0,
            'alpha': 0.0,
            'lamda': 1.0,
            'min_child_weight': 8,
            'max_depth': 3,
            'subsample': 0.9,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 0.4,
            'random_state': SEED,
        }
        model_xgb = xgb.train(params_xgb, dtrain_xgb, num_boost_round=20000, evals=[(dtrain_xgb, "train"), (dvalid_xgb, "eval")], verbose_eval=0, early_stopping_rounds=50)
        va_x_pred_prob_xgb = model_xgb.predict(dvax_xgb, iteration_range=(0, model_xgb.best_iteration))

        # LightGBM
        dtrain_lgb = lgb.Dataset(tr_x, label=tr_y)
        dvalid_lgb = lgb.Dataset(va_x, label=va_y)

        params_lgb = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 3,
            'min_child_weight': 8,
            'subsample': 0.9,
            'colsample_bytree': 1.0,
            'random_state': SEED,
            'verbose': -1
        }
        model_lgb = lgb.train(params_lgb, dtrain_lgb, num_boost_round=20000, valid_sets=[dtrain_lgb, dvalid_lgb], valid_names=['train', 'valid'], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        va_x_pred_prob_lgb = model_lgb.predict(va_x, num_iteration=model_lgb.best_iteration)

        # Threshold Optimization
        def func(threshold):
            actuals = np.array(va_y)
            preds_xgb = punctuate(threshold, va_x_pred_prob_xgb)
            preds_lgb = punctuate(threshold, va_x_pred_prob_lgb)
            preds_ensemble = (preds_xgb + preds_lgb) / 2  # Weighted Average
            preds_ensemble = np.round(preds_ensemble).astype(int)  # Ensure discrete classes
            q_kappa = quadratic_kappa(actuals, preds_ensemble, N=6)
            return 1 / q_kappa

        bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        t_initial = [0.3, 1.5, 2.5, 3.5, 4.3]
        result = opt.minimize(func, t_initial, method="Nelder-Mead", bounds=bounds)
        threshold = result.x

        # Validation Predictions
        preds_xgb = punctuate(threshold, va_x_pred_prob_xgb)
        preds_lgb = punctuate(threshold, va_x_pred_prob_lgb)
        preds_ensemble = (preds_xgb + preds_lgb) / 2
        preds_ensemble = np.round(preds_ensemble).astype(int)  # Ensure discrete classes

        oof_preds_xgb[ids_valid] = preds_xgb
        oof_preds_lgb[ids_valid] = preds_lgb

        score_xgb = quadratic_kappa(np.array(va_y), preds_xgb, N=6)
        score_lgb = quadratic_kappa(np.array(va_y), preds_lgb, N=6)
        score_ensemble = quadratic_kappa(np.array(va_y), preds_ensemble, N=6)

        validation_scores_xgb.append(score_xgb)
        validation_scores_lgb.append(score_lgb)
        print(f"Fold {fold + 1} - XGB Kappa: {score_xgb:.5f}, LGB Kappa: {score_lgb:.5f}, Ensemble Kappa: {score_ensemble:.5f}")

        # Test Predictions
        test_pred_prob_xgb = model_xgb.predict(dtest_xgb, iteration_range=(0, model_xgb.best_iteration))
        test_pred_prob_lgb = model_lgb.predict(test_x, num_iteration=model_lgb.best_iteration)

        preds_test_xgb = punctuate(threshold, test_pred_prob_xgb)
        preds_test_lgb = punctuate(threshold, test_pred_prob_lgb)
        preds_tests_xgb.append(preds_test_xgb)
        preds_tests_lgb.append(preds_test_lgb)

    # Mean Validation Scores
    mean_score_xgb = np.mean(validation_scores_xgb)
    mean_score_lgb = np.mean(validation_scores_lgb)
    print(f"\nMean XGB Kappa: {mean_score_xgb:.5f}, Mean LGB Kappa: {mean_score_lgb:.5f}")

    # Ensemble Test Predictions
    test_pred_xgb, _ = stats.mode(preds_tests_xgb, axis=0)
    test_pred_lgb, _ = stats.mode(preds_tests_lgb, axis=0)
    test_pred_ensemble = (test_pred_xgb + test_pred_lgb) / 2
    test_pred_ensemble = np.round(test_pred_ensemble).astype(int)

    test_pred_quality = test_pred_ensemble + 3
    test_pred_flat = test_pred_quality.flatten()

    f_test_preds = pd.DataFrame(test_pred_flat, columns=["quality"])
    return f_test_preds

# Train Ensemble Model
pred_ensemble = TRAIN_ENSEMBLE(train, test, SEED=0)

# Save Submission
final_sub = pd.concat([sample['id'], pred_ensemble], axis=1)
final_sub.to_csv('submission.csv', index=False)
print(final_sub['quality'].value_counts())

