%%time

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
from scipy import optimize as opt
from sklearn.model_selection import RepeatedStratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

from zmq.sugar.socket import T
import scipy.optimize as opt

import statistics
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')


%%time

train = pd.read_csv('/kaggle/input/chydv-hackathon-2025/train.csv')
test = pd.read_csv('/kaggle/input/chydv-hackathon-2025/test.csv')
sample = pd.read_csv("/kaggle/input/chydv-hackathon-2025/sample_submission.csv")

train['quality'] = train['quality'].astype(int)
train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

train = train.drop_duplicates()


%%time

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
    q_kappa = 1 - (num / den)
    return q_kappa


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


%%time

def TRAIN_XGB(train, test, SEED=0):
    x = train.drop(['quality'], axis=1)
    y = train['quality']
    y_shift = y - 3
    test_x = test

    from sklearn.model_selection import RepeatedStratifiedKFold 
    import xgboost as xgb
    import numpy as np
    import scipy.optimize as opt
    from sklearn.metrics import confusion_matrix

    preds_tests = []
    oof_preds = np.zeros(len(train))
    validation_scores = []

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=SEED)

    for fold, (ids_train, ids_valid) in enumerate(skf.split(x, y_shift)):
        tr_x = x.iloc[ids_train, :]
        va_x = x.iloc[ids_valid, :]
        tr_y = y_shift.iloc[ids_train]
        va_y = y_shift.iloc[ids_valid]

        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        dvax = xgb.DMatrix(va_x)
        dtest = xgb.DMatrix(test_x)

        params = {
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
        num_round = 20000

        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        model = xgb.train(params, dtrain, num_round, evals=watchlist, verbose_eval=0, early_stopping_rounds=50)

        iteration_range = (0, model.best_iteration) if hasattr(model, 'best_iteration') else (0, num_round)
        va_x_pred_prob = model.predict(dvax, iteration_range=iteration_range)

        def func(threshold):
            actuals = np.array(va_y)
            preds = punctuate(threshold, va_x_pred_prob)
            q_kappa = quadratic_kappa(actuals, preds, N=6)
            return 1 / q_kappa

        bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        t_initial = [0.3, 1.5, 2.5, 3.5, 4.3]

        result = opt.minimize(func, t_initial, method="Nelder-Mead", bounds=bounds)
        threshold = result.x

        preds = punctuate(threshold, va_x_pred_prob)
        oof_preds[ids_valid] = preds

        score = quadratic_kappa(np.array(va_y), preds, N=6)
        validation_scores.append(score)
        print(f"Fold {fold + 1} Validation Kappa Score: {score:.5f}")

        test_pred_prob = model.predict(dtest, iteration_range=iteration_range)
        preds_test = punctuate(threshold, test_pred_prob)
        preds_tests.append(preds_test)

    mean_score = np.mean(validation_scores)
    print(f"\nMean Validation Kappa Score: {mean_score:.5f}")

    test_pred, test_pred_count = stats.mode(preds_tests, axis=0) 

    test_pred_quality = np.array(test_pred) + 3 
    test_pred_flat = test_pred_quality.flatten()
    
    f_test_preds = pd.DataFrame(test_pred_flat,columns=["quality"]) 

    return oof_preds, f_test_preds


%%time

oof_xgb, pred_xgb = TRAIN_XGB(train, test, SEED=0)


%%time

def TRAIN_LGB(train, test, target_col='quality', seed=0):

    x = train.drop([target_col], axis=1)
    y = train[target_col]
    y_shift = y - 3
    test_x = test

    preds_tests = []
    validation_scores = []
    oof_preds = np.zeros(len(train))
    
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

    for fold, (ids_train, ids_valid) in enumerate(skf.split(x, y_shift)):
        tr_x, va_x = x.iloc[ids_train, :], x.iloc[ids_valid, :]
        tr_y, va_y = y_shift.iloc[ids_train], y_shift.iloc[ids_valid]

        dtrain = lgb.Dataset(tr_x, label=tr_y)
        dvalid = lgb.Dataset(va_x, label=va_y)

        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 3,
            'min_child_weight': 8,
            'subsample': 0.9,
            'colsample_bytree': 1.0,
            'random_state': seed,
            'verbose': -1
        }

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=20000,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        def func(threshold):
            actuals = np.array(va_y)
            va_x_pred_prob = model.predict(va_x, num_iteration=model.best_iteration)
            preds = punctuate(threshold, va_x_pred_prob)
            q_kappa = quadratic_kappa(actuals, preds, N=6)
            return 1 / q_kappa

        bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        t_initial = [0.3, 1.5, 2.5, 3.5, 4.3]
        result = opt.minimize(func, t_initial, method="Nelder-Mead", bounds=bounds)
        threshold = result.x

        va_x_pred_prob = model.predict(va_x, num_iteration=model.best_iteration)
        preds = punctuate(threshold, va_x_pred_prob)
        actuals = np.array(va_y)
        
        oof_preds[ids_valid] = preds
        score = quadratic_kappa(actuals, preds, N=6)
        validation_scores.append(score)
        print(f"Fold {fold + 1} Validation Kappa Score: {score:.5f}")

        test_pred_prob = model.predict(test_x, num_iteration=model.best_iteration)
        preds_test = punctuate(threshold, test_pred_prob)
        preds_tests.append(preds_test)

    mean_score = np.mean(validation_scores)
    print(f"\nMean Validation Kappa Score: {mean_score:.5f}")

    test_pred, test_pred_count = stats.mode(preds_tests, axis=0) 

    test_pred_quality = np.array(test_pred) + 3 
    test_pred_flat = test_pred_quality.flatten()
    
    f_test_preds = pd.DataFrame(test_pred_flat,columns=["quality"]) 

    return oof_preds, f_test_preds


oof_lgb , pred_lgb = TRAIN_LGB(train, test, target_col='quality', seed=0)


%%time

xgb_sub = pd.concat([sample['id'],pred_xgb],axis=1)
lgb_sub = pd.concat([sample['id'],pred_lgb],axis=1)

merged_sub = xgb_sub.merge(lgb_sub, on='id', suffixes=('_xgb', '_lgb'))

merged_sub['quality'] = merged_sub.apply(
    lambda row: row['quality_xgb'] if row['quality_xgb'] == row['quality_lgb'] else row[['quality_xgb', 'quality_lgb']].mode()[0],
    axis=1
)

final_sub = merged_sub[['id', 'quality']]
final_sub.to_csv('submission.csv',index=False)
print(final_sub['quality'].value_counts())

