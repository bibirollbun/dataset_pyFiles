n_splits, seed = 3, 1984
path = "/kaggle/input"

import torch
cuda = torch.cuda.is_available()
print(f"Is GPU/CUDA available : {cuda}")


try:
    from lifelines.utils import concordance_index
except ModuleNotFoundError:
    print('Installing lifelines...')
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
    !pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
set_config(transform_output="pandas")

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor


train = pd.read_csv(f"{path}/equity-post-HCT-survival-predictions/train.csv", index_col = ["ID"])
test = pd.read_csv(f"{path}/equity-post-HCT-survival-predictions/test.csv", index_col = ["ID"])
print(f"Train shape : {train.shape} | Test shape : {test.shape}")
race_groups = list(train["race_group"].unique())


features = list(test.columns)
CAT_FEATURES = []
for c in test.columns:
    if train[c].dtype=="object":
        CAT_FEATURES.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        
combined = pd.concat([train, test], axis = 0)#, ignore_index = True)
print("The CATEGORICAL FEATURES: ",end="")
for c in test.columns:
    if c in CAT_FEATURES:
        print(f"{c}, ", end="")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].copy()[features]


# Kaplan-Meier target
res = np.zeros(train.shape[0])
for trn_idx, val_idx in folds.split(train, train["race_group"]):

    X_trn, X_val = train.iloc[trn_idx], train.iloc[val_idx]
    kmf = KaplanMeierFitter()
    kmf.fit(durations = X_trn['efs_time'], event_observed = X_trn['efs'])
    res[val_idx] = kmf.survival_function_at_times(X_val['efs_time']).values

train["Kaplan-Meier-target"] = res
train.loc[train['efs'] == 0, 'Kaplan-Meier-target'] -= 0.2

sns.histplot(data=train, x = 'Kaplan-Meier-target', hue = 'efs', element = 'step', common_norm = False)
plt.legend(title = 'efs')
plt.title('Distribution of Target by EFS')
plt.xlabel('Target')
plt.ylabel('Density')
plt.show()


metric_columns = ["race_group", "efs", "efs_time"]

def score_(y_true, y_pred, return_details = False):
    metric_list = []
    for race in race_groups:
        mask = y_true.race_group.values == race
        metric_list.append(concordance_index(y_true.efs_time[mask], - y_pred[mask], y_true.efs[mask]))
        
    if return_details:
        return np.mean(metric_list) - np.std(metric_list), np.mean(metric_list), np.std(metric_list)
    else:
        return np.mean(metric_list) - np.std(metric_list)


# Index trick : instead of true values, I will pass index of true values to our metric
def score_xgb(y_true_idx, y_pred):
    return -score_(train[metric_columns].iloc[y_true_idx], y_pred)


%%time

target = "Kaplan-Meier-target"
val_scores = []
features = [f for f in list(test.columns) if f != "race_group"]

oof, pred = pd.Series(0.0, name = "prediction", index = train.index), pd.Series(0.0, name = "prediction", index = test.index)
for fold, (trn_idx, val_idx) in enumerate(folds.split(train, train["race_group"])):

    X_trn, y_trn, y_trn_ = train[features].iloc[trn_idx], train[metric_columns].iloc[trn_idx], train[target].iloc[trn_idx]
    X_val, y_val = train[features].iloc[val_idx], train[metric_columns].iloc[val_idx]
    
    my_model = XGBRegressor(
        tree_method="hist",
        objective='reg:squarederror',
        device = "cuda" if cuda else "cpu",
        n_estimators = 2000, early_stopping_rounds=100,
        eval_metric = score_xgb, # the custom metric
        disable_default_eval_metric = True, # only show the custom metric
        enable_categorical = True,
        max_depth=3,
        colsample_bytree=0.5, 
        subsample=0.8, 
        learning_rate=.1,#0.03,
        min_child_weight=5,
    )
    my_model.fit(X_trn, y_trn_, eval_set = [(X_val, val_idx)], verbose = 250)
    # Yes, val_idx and not y_val: it's a trick for this competition

    y_val_pred = my_model.predict(X_val)
    val_tot_s, val_mean_s, val_std_s = score_(y_val, y_val_pred, True)
    print(f"    Fold {fold + 1:2} : Val. score  : {val_tot_s:.3f} ({val_mean_s:.3f} ±{val_std_s:.3f})")
    val_scores.append(val_tot_s)
    oof.iloc[val_idx] += y_val_pred
    pred += my_model.predict(test[features])
    
oof_tot_s, oof_mean_s, oof_std_s = score_(train, oof, True)
print(f"OOF score : {oof_tot_s:.3f} ({oof_mean_s:.3f} ±{oof_std_s:.3f})")


from numba import jit

@jit(nopython=True)
def init_BTree(values):
    times_to_compare = np.empty_like(values)
    last_full_row = int(np.log2(len(values) + 1) - 1)
    len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
    if len_ragged_row > 0:
#        bottom_row_ix = np.s_[: 2 * len_ragged_row : 2]
        bottom_row_ix = slice(None, 2 * len_ragged_row, 2)
        times_to_compare[-len_ragged_row:] = values[bottom_row_ix]
        values = np.delete(values, bottom_row_ix)
    values_start = 0
    values_space = 2
    values_len = 2 ** last_full_row
    while values_start < len(values):
        times_to_compare[values_len - 1 : 2 * values_len - 1] = values[values_start::values_space]
        values_start += int(values_space / 2)
        values_space *= 2
        values_len = int(values_len / 2)
    return times_to_compare

@jit(nopython=True)
def insert(counts, pred, times_to_compare):
    i = 0
    n = len(times_to_compare)
    while (i < n):
        cur = times_to_compare[i]
        counts[i] += 1
        if pred < cur:
            i = 2 * i + 1
        elif pred > cur:
            i = 2 * i + 2
        else:
            return counts
    #raise ValueError("Value %s not contained in tree." "Also, the counts are now messed up." % times_to_compare)

@jit(nopython=True)
def fn_rank(pred, times_to_compare, counts):
    i = 0
    n = len(times_to_compare)
    rank = 0
    count = 0
    while (i < n):
        cur = times_to_compare[i]
        if pred < cur:
            i = 2 * i + 1
            continue
        elif pred > cur:
            rank += counts[i]
            # subtract off the right tree if exists
            nexti = 2 * i + 2
            if nexti < n:
                rank -= counts[nexti]
                i = nexti
                continue
            else:
                return rank, count
        else:  # value == cur
            count = counts[i]
            lefti = 2 * i + 1
            if lefti < n:
                nleft = counts[lefti]
                count -= nleft
                rank += nleft
                righti = lefti + 1
                if righti < n:
                    count -= counts[righti]
            return rank, count
    return rank, count


@jit(nopython=True)
def handle_pairs(truth, pred, first_ix, times_to_compare, counts):
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = counts[0] * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
#        rank, count = times_to_compare.rank(censored_pred[i])
        rank, count = fn_rank(pred[i], times_to_compare, counts)
        correct += rank
        tied += count
    return (pairs, correct, tied, next_ix)


@jit(nopython=True)
def fast_concordance_index(event_times, predicted_event_times, event_observed):
    
    died_mask = event_observed==1#.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    
    times_to_compare = init_BTree(np.unique(died_pred))
#    counts = np.zeros_like(times_to_compare, dtype=int)
    counts = np.full(len(times_to_compare), 0)
    
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare, counts)
            censored_ix = next_ix
            
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(died_truth, died_pred, died_ix, times_to_compare, counts)

            for pred in died_pred[died_ix:next_ix]:
                insert(counts, pred, times_to_compare)
                                
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied
        
#    print(num_pairs, num_correct, num_tied)

    return (num_correct + num_tied / 2) / num_pairs


metric_columns = ["race_group", "efs", "efs_time"]

def score_f(y_true, y_pred, return_details = False):
    metric_list = []
    for race in race_groups:
        mask = y_true.race_group.values == race
        if isinstance(y_pred, pd.Series):
            c_index_race = fast_concordance_index(y_true.efs_time[mask].values, - y_pred[mask].values.ravel(), y_true.efs[mask].values)
        else:
            c_index_race = fast_concordance_index(y_true.efs_time[mask].values, - y_pred[mask].ravel(), y_true.efs[mask].values)
        metric_list.append(c_index_race)
        
    if return_details:
        return np.mean(metric_list) - np.std(metric_list), np.mean(metric_list), np.std(metric_list)
    else:
        return np.mean(metric_list) - np.std(metric_list)

def score_xgb(y_true_idx, y_pred):
    return -score_f(train[metric_columns].iloc[y_true_idx], y_pred)


oof_tot_s, oof_mean_s, oof_std_s = score_f(train, oof, True)
print(f"OOF score : {oof_tot_s:.3f} ({oof_mean_s:.3f} ±{oof_std_s:.3f})")


%%time
oof_tot_s, oof_mean_s, oof_std_s = score_f(train, oof, True)
print(f"OOF score : {oof_tot_s:.3f} ({oof_mean_s:.3f} ±{oof_std_s:.3f})")


%timeit -n 5 -r 10 score_(train, oof)
%timeit -n 5 -r 10 score_f(train, oof)


%%time
%timeit -n 3 -r 100 score_(train, oof)


%%time
%timeit -n 3 -r 100 score_f(train, oof)


for r in train["race_group"].unique():
    mask = train["race_group"]==r
    print(f"{r:45} Original (without numba) : {concordance_index(train.efs_time[mask], -oof[mask], train.efs[mask]):.5f} | ", end='')
    print(f"Faster (with numba): {fast_concordance_index(train.efs_time[mask].values, -oof[mask].values, train.efs[mask].values):.5f}")


%%time

target = "Kaplan-Meier-target"
val_scores = []
features = [f for f in list(test.columns) if f != "race_group"]

oof2, pred = pd.Series(0.0, name = "prediction", index = train.index), pd.Series(0.0, name = "prediction", index = test.index)
for fold, (trn_idx, val_idx) in enumerate(folds.split(train, train["race_group"])):

    X_trn, y_trn, y_trn_ = train[features].iloc[trn_idx], train[metric_columns].iloc[trn_idx], train[target].iloc[trn_idx]
    X_val, y_val = train[features].iloc[val_idx], train[metric_columns].iloc[val_idx]
    
    my_model = XGBRegressor(
        tree_method="hist",
        objective='reg:squarederror',
        device = "cuda" if cuda else "cpu",
        n_estimators = 2000, early_stopping_rounds=100,
        eval_metric = score_xgb, # the custom metric
        disable_default_eval_metric = True, # only show the custom metric
        enable_categorical = True,
        max_depth=3,
        colsample_bytree=0.5, 
        subsample=0.8, 
        learning_rate=.1,#0.03,
        min_child_weight=5,
    )
    my_model.fit(X_trn, y_trn_, eval_set = [(X_val, val_idx)], verbose = 250)

    y_val_pred = my_model.predict(X_val)
    val_tot_s, val_mean_s, val_std_s = score_(y_val, y_val_pred, True)
    print(f"    Fold {fold + 1:2} : Val. score  : {val_tot_s:.3f} ({val_mean_s:.3f} ±{val_std_s:.3f})")
    val_scores.append(val_tot_s)
    oof2.iloc[val_idx] += y_val_pred
    pred += my_model.predict(test[features])
    
oof_tot_s, oof_mean_s, oof_std_s = score_(train, oof2, True)
print(f"OOF score : {oof_tot_s:.3f} ({oof_mean_s:.3f} ±{oof_std_s:.3f})")




