# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')


import warnings, gc, os

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


orig = pd.read_csv("/kaggle/input/bank-data/bank-full.csv",delimiter=";")
orig['y'] = orig.y.map({'yes':1,'no':0})
orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')
print("Original data shape", orig.shape )
orig.head()


test_id = test['id']
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)



train.head()


unknown_values = (train == 'unknown').sum()
unknown_values
unknown_values_T = (test == 'unknown').sum()
unknown_values_T


null_count = train.isnull().sum()
features_dtype = train.dtypes
df_ = pd.DataFrame({'null count':null_count, 'dtype': features_dtype})
df_


cat_columns = train.select_dtypes('object').columns.tolist()


for i in cat_columns:
    print(f'\ncolumn: {i}')
    print(train[i].unique())


mapping_yes_no = {'no':0, 'yes':1}

for c in ['default','housing','loan']:
    train[c] = train[c].map(mapping_yes_no)
    test[c]  = test[c].map(mapping_yes_no)
    if c in orig.columns:
        orig[c] = orig[c].map(mapping_yes_no)



label_columns = ['job','marital','education','contact','month','poutcome']



def labelencode_(train_df, test_df, orig_df, columns):
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    orig_enc = orig_df.copy()
    for col in columns:
        tr_col = train_enc[col].astype(str)
        te_col = test_enc[col].astype(str)
        og_col = orig_enc[col].astype(str)
        comb = pd.concat([tr_col, te_col, og_col], axis=0)
        le = LabelEncoder()
        le.fit(comb)
        train_enc[col] = le.transform(tr_col)
        test_enc[col] = le.transform(te_col)
        orig_enc[col] = le.transform(og_col)
    return train_enc, test_enc, orig_enc


train, test, orig = labelencode_(train, test, orig, label_columns)


train.info()


train['duration_sin'] = np.sin(2*np.pi * train['duration'] / 540)
train['duration_cos'] = np.cos(2*np.pi * train['duration'] / 540)

test['duration_sin'] = np.sin(2*np.pi * test['duration'] / 540)
test['duration_cos'] = np.cos(2*np.pi * test['duration'] / 540)

orig['duration_sin'] = np.sin(2*np.pi * orig['duration'] / 540)
orig['duration_cos'] = np.cos(2*np.pi * orig['duration'] / 540)


train.info()


def load_and_prepare_data(df):
    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 540)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 540)
    
    df['balance_age_ratio'] = df['balance'] / (df['age'] + 1)
    df['campaign_previous_ratio'] = df['campaign'] / (df['previous'] + 1)
    df['duration_campaign_interaction'] = df['duration'] * df['campaign']
    
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['young', 'middle_young', 'middle', 'middle_old', 'old']
    )
    age_group_encoded = pd.get_dummies(df['age_group'], prefix='age_group', dtype=np.int8)
    df = pd.concat([df.drop(columns=['age_group']), age_group_encoded], axis=1)

    df['balance_category'] = pd.cut(
        df['balance'],
        bins=[-np.inf, 0, 1000, 5000, np.inf],
        labels=['negative', 'low', 'medium', 'high']
    )
    balance_cat_encoded = pd.get_dummies(df['balance_category'], prefix='balance_cat', dtype=np.int8)
    df = pd.concat([df.drop(columns=['balance_category']), balance_cat_encoded], axis=1)

    df['duration_category'] = pd.cut(
        df['duration'],
        bins=[0, 100, 300, 600, np.inf],
        labels=['very_short', 'short', 'medium', 'long']
    )
    duration_cat_encoded = pd.get_dummies(df['duration_category'], prefix='duration_cat', dtype=np.int8)
    df = pd.concat([df.drop(columns=['duration_category']), duration_cat_encoded], axis=1)

    df['age_squared'] = df['age'] ** 2
    df['balance_squared'] = df['balance'] ** 2
    df['duration_squared'] = df['duration'] ** 2
    
    df['log_duration'] = np.log1p(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])

    df['age_balance_interaction'] = df['age'] * df['balance']
    df['job_education_interaction'] = df['job'] * df['education']

    return df



train = load_and_prepare_data(train)
test = load_and_prepare_data(test)
orig = load_and_prepare_data(orig)


# train = pd.concat([train, orig], ignore_index=True)


train.info()


def clean_dataframe(train, test, target='y'):
    X = train.drop(target, axis=1)
    y = train[target].astype('int8')

    all_cols = X.columns.union(test.columns)
    X = X.reindex(columns=all_cols, fill_value=0)
    test = test.reindex(columns=all_cols, fill_value=0)
    for df in [X, test]:
        for c in df.columns:
            if df[c].dtype == 'int64':
                df[c] = df[c].astype('int32')
            elif df[c].dtype == 'float64':
                df[c] = df[c].astype('float32')

    return X, y, test


X, y, test = clean_dataframe(train, test, target='y')
X_base, T_base = X, test
RANDOM_STATE = 42


def feature_selection(X, y, n_features=50):
    selector_stats = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    X_stats = selector_stats.fit_transform(X, y)
    stats_features = selector_stats.get_support()
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_top_features = np.argsort(rf_importance)[-n_features:]
    rf_features = np.zeros(X.shape[1], dtype=bool)
    rf_features[rf_top_features] = True
    
    xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_model.fit(X, y)
    xgb_importance = xgb_model.feature_importances_
    xgb_top_features = np.argsort(xgb_importance)[-n_features:]
    xgb_features = np.zeros(X.shape[1], dtype=bool)
    xgb_features[xgb_top_features] = True
    
    combined_features = stats_features | rf_features | xgb_features
    
    print(f"Selected {combined_features.sum()} features out of {X.shape[1]}")
    
    return combined_features, {
        'rf_importance': rf_importance,
        'xgb_importance': xgb_importance,
        'stats_scores': selector_stats.scores_
    }


selected_features, importances = feature_selection(X, y, n_features=50)



def plot_feature_importances(importances, feature_names, top_n=42):
    rf_importance = importances['rf_importance']
    xgb_importance = importances['xgb_importance']
    stats_scores = importances['stats_scores']

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "RandomForest": rf_importance,
        "XGBoost": xgb_importance,
        "Stats": stats_scores
    })
    df_importance["MeanImportance"] = df_importance[["RandomForest", "XGBoost", "Stats"]].mean(axis=1)
    df_top = df_importance.sort_values("MeanImportance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_top["Feature"], df_top["MeanImportance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.show()

plot_feature_importances(importances, X.columns, top_n=42)


def get_feature_importance_df(importances, feature_names, top_n=42):
    rf_importance = importances['rf_importance']
    xgb_importance = importances['xgb_importance']
    stats_scores = importances['stats_scores']

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "RandomForest": rf_importance,
        "XGBoost": xgb_importance,
        "Stats": stats_scores
    })
    df_importance["MeanImportance"] = df_importance[["RandomForest", "XGBoost", "Stats"]].mean(axis=1)
    df_top = df_importance.sort_values("MeanImportance", ascending=False).head(top_n).reset_index(drop=True)
    return df_top


df_top_features = get_feature_importance_df(importances, X.columns, top_n=42)
df_top_features


def _te_fit_map(df_fit, cols, target='y', smooth=20.0):
    gr = df_fit.groupby(list(cols), observed=True)[target].agg(['mean','count']).reset_index()
    global_mean = float(df_fit[target].mean())
    gr['te_val'] = (gr['mean']*gr['count'] + global_mean*smooth) / (gr['count']+smooth)
    gr = gr.drop(columns=['mean','count'])
    return gr, global_mean

def _te_apply(df_target, cols, mapping, global_mean, new_name):
    m = df_target.merge(mapping, how='left', on=list(cols))
    df_target[new_name] = m['te_val'].fillna(global_mean).astype('float32').values
    return df_target

def get_global_te_stats(orig_df_with_y, te_single, te_pairs, target='y', smooth=50.0):
    global_stats = {}
    for col in te_single:
        mapping, gmean = _te_fit_map(orig_df_with_y, [col], target=target, smooth=smooth)
        global_stats[col] = {'mapping': mapping, 'global_mean': gmean}
    for c1, c2 in te_pairs:
        mapping, gmean = _te_fit_map(orig_df_with_y, [c1, c2], target=target, smooth=smooth)
        global_stats[f"{c1}_{c2}"] = {'mapping': mapping, 'global_mean': gmean}
    return global_stats

def _te_apply_with_global(df_target, cols, local_map, local_mean, global_stats, key, new_name):
    m = df_target.merge(local_map, how='left', on=list(cols))
    te = m['te_val']

    if key in global_stats:
        gm = global_stats[key]['mapping']
        miss = te.isna()
        if miss.any():
            m2 = df_target.loc[miss, list(cols)].merge(gm, how='left', on=list(cols))
            te.loc[miss] = m2['te_val'].values
        te = te.fillna(global_stats[key]['global_mean'])
    else:
        te = te.fillna(local_mean)

    df_target[new_name] = te.astype('float32').values
    return df_target



def add_te_features_fold(x_train, x_valid, x_test,
                         train_cats, test_cats,
                         y_train_fold, train_idx, valid_idx,
                         te_single, te_pairs, global_stats,
                         smooth_local=20.0):
    tr_cat = train_cats.loc[train_idx].copy()
    va_cat = train_cats.loc[valid_idx].copy()
    te_cat = test_cats.copy()

    tr_df = tr_cat.copy()
    tr_df['y'] = y_train_fold.values

    for col in te_single:
        local_map, local_mean = _te_fit_map(tr_df, [col], target='y', smooth=smooth_local)
        key = col
        tr_cat = _te_apply_with_global(tr_cat, [col], local_map, local_mean, global_stats, key, f'te_{col}')
        va_cat = _te_apply_with_global(va_cat, [col], local_map, local_mean, global_stats, key, f'te_{col}')
        te_cat = _te_apply_with_global(te_cat, [col], local_map, local_mean, global_stats, key, f'te_{col}')
        x_train[f'te_{col}'] = tr_cat[f'te_{col}'].values
        x_valid[f'te_{col}'] = va_cat[f'te_{col}'].values
        x_test[f'te_{col}']  = te_cat[f'te_{col}'].values

    for c1, c2 in te_pairs:
        local_map, local_mean = _te_fit_map(tr_df, [c1, c2], target='y', smooth=smooth_local)
        key = f"{c1}_{c2}"
        tr_cat = _te_apply_with_global(tr_cat, [c1, c2], local_map, local_mean, global_stats, key, f'te_{c1}_{c2}')
        va_cat = _te_apply_with_global(va_cat, [c1, c2], local_map, local_mean, global_stats, key, f'te_{c1}_{c2}')
        te_cat = _te_apply_with_global(te_cat, [c1, c2], local_map, local_mean, global_stats, key, f'te_{c1}_{c2}')
        x_train[f'te_{c1}_{c2}'] = tr_cat[f'te_{c1}_{c2}'].values
        x_valid[f'te_{c1}_{c2}'] = va_cat[f'te_{c1}_{c2}'].values
        x_test[f'te_{c1}_{c2}']  = te_cat[f'te_{c1}_{c2}'].values

    return x_train, x_valid, x_test



from catboost import CatBoostClassifier

oof_catb = np.zeros(len(train))
pred_catb = np.zeros(len(test))

for i, (train_index, valid_index) in enumerate(skf.split(train[FEATURES], target_train)):
    print("#"*25)
    print(f"### CatBoost Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_trn   = target_train.loc[train_index].copy()
    x_valid = train.loc[valid_index,  FEATURES].copy()
    y_val   = target_train.loc[valid_index].copy()
    x_test  = test[FEATURES].copy()

    x_train, x_valid, x_test = add_te_features_fold(
        x_train, x_valid, x_test,
        train_cats_for_te, test_cats_for_te,
        y_trn, train_index, valid_index,
        TE_SINGLE, TE_PAIRS, global_te_stats,
        smooth_local=20.0
    )

    for c in NUM_FEATURES:
        m = x_train[c].mean(); s = x_train[c].std()
        x_train[c] = WGT[c] * (x_train[c]-m)/s
        x_valid[c] = WGT[c] * (x_valid[c]-m)/s
        x_test[c]  = WGT[c] * (x_test[c]-m)/s
        x_train[c] = x_train[c].fillna(0); x_valid[c] = x_valid[c].fillna(0); x_test[c] = x_test[c].fillna(0)

    model = CatBoostClassifier(
        iterations=10_000,
        depth=5,
        learning_rate=0.0377,
        l2_leaf_reg=2.7384616507314177,
        eval_metric="AUC",
        random_seed=777,
        bootstrap_type='Bernoulli',
        subsample=0.8111694121602908,
        task_type="GPU",     # GPU varsa
        early_stopping_rounds=100,
        verbose=100
    )
    model.fit(x_train, y_trn, eval_set=[(x_valid, y_val)])

    oof_catb[valid_index] = model.predict_proba(x_valid.values)[:,1]
    pred_catb += model.predict_proba(x_test.values)[:,1]
    del model, x_train, x_valid, x_test
    gc.collect()

pred_catb /= FOLDS
print(f"CatBoost CV AUC: {roc_auc_score(target_train.values, oof_catb):.6f}")




from lightgbm import LGBMClassifier
import lightgbm as lgb

oof_lgbm = np.zeros(len(train))
pred_lgbm = np.zeros(len(test))

for i, (train_index, valid_index) in enumerate(skf.split(train[FEATURES], target_train)):
    print("#"*25)
    print(f"### LightGBM Fold {i+1}")
    print("#"*25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_trn   = target_train.loc[train_index].copy()
    x_valid = train.loc[valid_index,  FEATURES].copy()
    y_val   = target_train.loc[valid_index].copy()
    x_test  = test[FEATURES].copy()

    # TE (fold içi)
    x_train, x_valid, x_test = add_te_features_fold(
        x_train, x_valid, x_test,
        train_cats_for_te, test_cats_for_te,
        y_trn, train_index, valid_index,
        TE_SINGLE, TE_PAIRS, global_te_stats,
        smooth_local=20.0
    )

    # Standardizasyon
    for c in NUM_FEATURES:
        m = x_train[c].mean(); s = x_train[c].std()
        x_train[c] = WGT[c] * (x_train[c] - m) / s
        x_valid[c] = WGT[c] * (x_valid[c] - m) / s
        x_test[c]  = WGT[c] * (x_test[c]  - m) / s
        x_train[c] = x_train[c].fillna(0); x_valid[c] = x_valid[c].fillna(0); x_test[c] = x_test[c].fillna(0)

    model = LGBMClassifier(
        max_depth=6,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=10_000,
        learning_rate=0.1,
        reg_alpha=1,
        objective="binary",
        metric="auc",
        random_state=777,
        n_jobs=-1
    )

    model.fit(
        x_train, y_trn,
        eval_set=[(x_valid, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )

    oof_lgbm[valid_index] = model.predict_proba(x_valid)[:, 1]
    pred_lgbm += model.predict_proba(x_test)[:, 1]
    del model, x_train, x_valid, x_test
    gc.collect()

pred_lgbm /= FOLDS
print(f"LightGBM CV AUC: {roc_auc_score(target_train.values, oof_lgbm):.6f}")




te_single = ['job','marital','education','contact','month','poutcome']
te_pairs  = [('job','education'), ('contact','month'), ('poutcome','marital'), ('housing','loan')]
global_te_stats = get_global_te_stats(orig, te_single, te_pairs)
N_FOLDS = 5
N_TRIALS = 1


pos = int(y.sum())
neg = int((y == 0).sum())
spw = neg / max(1, pos)


def cv_score_with_te(params, X_data, y_data, test_data, global_stats, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = []
    
    for tr_idx, va_idx in skf.split(X_data, y_data):
        X_tr, X_va = X_data.iloc[tr_idx].copy(), X_data.iloc[va_idx].copy()
        y_tr, y_va = y_data.iloc[tr_idx].copy(), y_data.iloc[va_idx].copy()
        
        tr_df = pd.concat([X_tr, y_tr], axis=1)
        
        for col in te_single:
            mapping, gmean = _te_fit_map(tr_df, [col], target='y', smooth=20.0)
            
            X_tr = _te_apply(X_tr, [col], mapping, gmean, f'te_{col}')
            X_va = _te_apply(X_va, [col], mapping, gmean, f'te_{col}')
            
        for col_pair in te_pairs:
            mapping, gmean = _te_fit_map(tr_df, list(col_pair), target='y', smooth=20.0)
            
            X_tr = _te_apply(X_tr, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')
            X_va = _te_apply(X_va, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')

        model = xgb.XGBClassifier(
            **params,
            # n_estimators=1000,  
            tree_method='hist',
            eval_metric='auc',
            random_state=seed
        )
        # use_gpu = int(xgb.__version__.split('.')[0]) >= 2
        # tree_method = 'hist' if use_gpu else 'gpu_hist'
        # extra = {'device': 'cuda'} if use_gpu else {}

        # model = xgb.XGBClassifier(
        #     **params,
        #     tree_method=tree_method,
        #     eval_metric='auc',
        #     random_state=seed,
        #     **extra
        # )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=True)]
        )
        pred_va = model.predict_proba(X_va)[:,1]
        scores.append(roc_auc_score(y_va, pred_va))
        del model, X_tr, X_va, y_tr, y_va, tr_df
        gc.collect()

    return float(np.mean(scores))



# def objective(trial):
#     params = {
#         'n_estimators':trial.suggest_int('n_estimators',300,5000),
#         'max_depth': trial.suggest_int('max_depth', 4, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#         'subsample': trial.suggest_float('subsample', 0.7, 0.95),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-1, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
#         'scale_pos_weight': spw,
#         'n_jobs': -1
#     }
    
#     score = cv_score_with_te(params, X, y, test, global_te_stats, seed=RANDOM_STATE)
#     return score


def make_xgb_params(idx=0):
    parameters_xgboost = {
        'n_estimators': 15000,
        'max_leaves': 127,
        'min_child_weight': 1.5,
        'max_depth': 0,
        'grow_policy': 'lossguide',
        'learning_rate': 0.008,
        'tree_method': 'gpu_hist',        
        'subsample': 0.85,
        'colsample_bylevel': 0.7,
        'colsample_bytree': 0.75,
        'colsample_bynode': 0.85,
        'sampling_method': 'gradient_based',
        'reg_alpha': 2.5,
        'reg_lambda': 0.8,
        'enable_categorical': True,
        'max_cat_to_onehot': 1,
        'device': 'cuda',                 
        'n_jobs': -1,
        'random_state': 42 + idx,
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': spw,
    }
    return parameters_xgboost



true = target_train.values
oof_ens = 0.5 * oof_lgbm + 0.25 * oof_xgb + 0.25 * oof_catb
print(f"Ensemble OOF AUC: {roc_auc_score(true, oof_ens):.6f}")

best_public_df = pd.read_csv("/kaggle/input/ps-s5e8-binary-classification-v-blend/submission.csv")
try:
    from IPython.display import display
    display(best_public_df.head())
except:
    print(best_public_df.head())

best_public = best_public_df.y.values

from scipy.stats import rankdata
sub = pd.read_csv("/kaggle/input/ps-s5e8-binary-classification-v-blend/submission.csv")
sub.y = -0.1 * (0.25 * rankdata(pred_lgbm) + 0.25 * rankdata(pred_xgb) + 0.5 * rankdata(pred_catb)) + 1.1 * rankdata(best_public)
sub.y = rankdata(sub.y) / len(sub)
print(sub.shape)
sub.to_csv("submission.csv", index=False)
print(sub.head())



# sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
# pruner  = optuna.pruners.MedianPruner(n_startup_trials=5)

# study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
# study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True)

# # print("BEST params:", study.best_params)#bunu yazdirmaya gerek yok
# print("BEST ROC-AUC (CV 5-fold):", study.best_value)

# best_params = {
#     **study.best_params,
#     'objective': 'binary:logistic',
#     'tree_method': 'hist',
#     'device': 'cuda',
#     'eval_metric': 'auc',
#     'n_jobs': -1,
#     'scale_pos_weight': spw
# }


# skf_final = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
# oof = np.zeros(len(X), dtype=float)
# pred = np.zeros(len(test), dtype=float)


# for fold, (tr_idx, va_idx) in enumerate(skf_final.split(X, y), 1):
#     print(f"Training fold {fold}...")
#     X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
#     y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()
#     te_df = test.copy()
#     tr_df = pd.concat([X_tr, y_tr], axis=1)

#     for col in te_single:
#         mapping, gmean = _te_fit_map(tr_df, [col], target='y', smooth=20.0)
#         X_tr = _te_apply(X_tr, [col], mapping, gmean, f'te_{col}')
#         X_va = _te_apply(X_va, [col], mapping, gmean, f'te_{col}')
#         te_df = _te_apply(te_df, [col], mapping, gmean, f'te_{col}')

#     for col_pair in te_pairs:
#         mapping, gmean = _te_fit_map(tr_df, list(col_pair), target='y', smooth=20.0)
#         X_tr = _te_apply(X_tr, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')
#         X_va = _te_apply(X_va, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')
#         te_df = _te_apply(te_df, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')

#     params_fold = make_xgb_params(idx=fold)
#     model = xgb.XGBClassifier(**params_fold)
#     model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_va, y_va)],
#         verbose=False,
#         callbacks=[xgb.callback.EarlyStopping(rounds=100, maximize=True)]
#     )

#     oof[va_idx] = model.predict_proba(X_va)[:, 1]
#     pred += model.predict_proba(te_df)[:, 1] / N_FOLDS
#     fold_auc = roc_auc_score(y_va, oof[va_idx])
#     print(f'Fold {fold} AUC: {fold_auc:.6f}')

#     del model, X_tr, X_va, y_tr, y_va, te_df, tr_df
#     gc.collect()

# cv_auc = roc_auc_score(y, oof)
# print(f'OOF AUC: {cv_auc:.6f}')


# model_final = xgb.XGBClassifier(**make_xgb_params(idx=0))
# X_temp = X.copy()


# for col in te_single:
#     mapping, gmean = _te_fit_map(pd.concat([X_temp, y], axis=1), [col], target='y', smooth=20.0)
#     X_temp = _te_apply(X_temp, [col], mapping, gmean, f'te_{col}')

# for col_pair in te_pairs:
#     mapping, gmean = _te_fit_map(pd.concat([X_temp, y], axis=1), list(col_pair), target='y', smooth=20.0)
#     X_temp = _te_apply(X_temp, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')

# model_final.fit(X_temp, y)



# for col in te_single:
#     mapping, gmean = _te_fit_map(pd.concat([X_temp, y], axis=1), [col], target='y', smooth=20.0)
#     X_temp = _te_apply(X_temp, [col], mapping, gmean, f'te_{col}')

# for col_pair in te_pairs:
#     mapping, gmean = _te_fit_map(pd.concat([X_temp, y], axis=1), list(col_pair), target='y', smooth=20.0)
#     X_temp = _te_apply(X_temp, list(col_pair), mapping, gmean, f'te_{col_pair[0]}_{col_pair[1]}')

# model_final.fit(X_temp, y)
# feature_importance = pd.DataFrame({
#     'feature': X_temp.columns,
#     'importance': model_final.feature_importances_
# }).sort_values('importance', ascending=False)


# submission = pd.DataFrame({"id": test_id, "y": pred})
# submission.to_csv("submission.csv", index=False)



# submission


# use_gpu = int(xgb.__version__.split('.')[0]) >= 2
# tree_method = 'hist' if use_gpu else 'gpu_hist'
# extra = {'device': 'cuda'} if use_gpu else {}

# model = xgb.XGBClassifier(
#     **params,
#     tree_method=tree_method,
#     eval_metric='auc',
#     random_state=seed,
#     **extra
# )




#columns grafik hangisi daha cok eki etmis ekle


# RANDOM_STATE = 42

# pos = int(y.sum())
# neg = int((y == 0).sum())
# spw = neg / max(1, pos)

# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)



# def feature_selection(X, y, n_features=50):
#     print("Starting feature selection...")
#     selector_stats = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
#     X_stats = selector_stats.fit_transform(X, y)
#     stats_features = selector_stats.get_support()
    
#     rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
#     rf.fit(X, y)
#     rf_importance = rf.feature_importances_
#     rf_top_features = np.argsort(rf_importance)[-n_features:]
#     rf_features = np.zeros(X.shape[1], dtype=bool)
#     rf_features[rf_top_features] = True
    
#     xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
#     xgb_model.fit(X, y)
#     xgb_importance = xgb_model.feature_importances_
#     xgb_top_features = np.argsort(xgb_importance)[-n_features:]
#     xgb_features = np.zeros(X.shape[1], dtype=bool)
#     xgb_features[xgb_top_features] = True
    
#     combined_features = stats_features | rf_features | xgb_features
    
#     print(f"Selected {combined_features.sum()} features out of {X.shape[1]}")
    
#     return combined_features, {
#         'rf_importance': rf_importance,
#         'xgb_importance': xgb_importance,
#         'stats_scores': selector_stats.scores_
#     }


# selected_features, importances = feature_selection(X, y, n_features=50)


# def plot_feature_importances(importances, feature_names, top_n=42):
#     rf_importance = importances['rf_importance']
#     xgb_importance = importances['xgb_importance']
#     stats_scores = importances['stats_scores']

#     df_importance = pd.DataFrame({
#         "Feature": feature_names,
#         "RandomForest": rf_importance,
#         "XGBoost": xgb_importance,
#         "Stats": stats_scores
#     })
#     df_importance["MeanImportance"] = df_importance[["RandomForest", "XGBoost", "Stats"]].mean(axis=1)
#     df_top = df_importance.sort_values("MeanImportance", ascending=False).head(top_n)

#     plt.figure(figsize=(10, 6))
#     plt.barh(df_top["Feature"], df_top["MeanImportance"], color="skyblue")
#     plt.gca().invert_yaxis()
#     plt.title(f"Top {top_n} Feature Importances")
#     plt.xlabel("Importance Score")
#     plt.show()

# plot_feature_importances(importances, X.columns, top_n=42)



# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators',300,2000),
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',          
#         'booster': 'gbtree',
#         'tree_method': 'hist',         
#         'random_state': RANDOM_STATE,
#         'n_jobs': -1,                            
#         'scale_pos_weight': spw,
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-2, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
#     }

#     fold_scores = []
#     for tr_idx, va_idx in cv.split(X, y):
#         X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#         model = xgb.XGBClassifier(**params)
#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_va, y_va)],
#             verbose=False,
#             callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=True)]
#         )
#         y_pred = model.predict_proba(X_va)[:, 1]
#         fold_scores.append(roc_auc_score(y_va, y_pred))

#     return float(np.mean(fold_scores))



# sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
# pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

# study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
# study.optimize(objective, n_trials=3, show_progress_bar=True, gc_after_trial=True)


# print("BEST params:", study.best_params)
# print("BEST ROC-AUC:", study.best_value)

# best_params = {
#     **study.best_params,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'booster': 'gbtree',
#     'tree_method': 'hist',
#     'random_state': RANDOM_STATE,
#     'n_jobs': -1,
#     'scale_pos_weight': spw
# }



# final_model = xgb.XGBClassifier(**best_params)

# X_tr, X_va, y_tr, y_va = train_test_split(
#     X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
# )

# final_model.fit(
#     X_tr, y_tr,
#     eval_set=[(X_va, y_va)],
#     verbose=False,
#     callbacks=[xgb.callback.EarlyStopping(rounds=200, maximize=True)]
# )
# best_iter = getattr(final_model, "best_iteration", None)
# best_iter = int(best_iter) + 1 if best_iter is not None else best_params.get('n_estimators', 1000)

# final_model.set_params(n_estimators=best_iter)
# final_model.fit(X, y, verbose=False)


# test_pred = final_model.predict_proba(test[X.columns])[:, 1]
# submission = pd.DataFrame({"id": test_id, "y": test_pred})
# submission.to_csv("submission.csv", index=False)



# ls


# submission


# ls

