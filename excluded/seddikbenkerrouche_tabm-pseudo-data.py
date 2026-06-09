!pip install --upgrade pytabkit==1.5.0



%load_ext cudf.pandas


import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from itertools import combinations
from cuml.preprocessing import TargetEncoder


print(f"XGBoost version {xgb.__version__}")


import pandas as pd
import numpy as np
import warnings
import zipfile
import gc
import os
from sklearn.model_selection import StratifiedKFold
from pandas.errors import PerformanceWarning
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import TabM_D_Classifier
from itertools import combinations
from tqdm import tqdm




PATH = "/kaggle/input/playground-series-s5e11/"
train = pd.read_csv(f"{PATH}train.csv").set_index('id')
print("Train shape", train.shape )
train.head()


test = pd.read_csv(f"{PATH}test.csv").set_index('id')
test['y'] = -1
print("Test shape", test.shape )
test.head()


orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')
print("Original data shape", orig.shape )
orig.head()


# credit score categories

def map_fico_tier(score):
    """Maps a credit score to its corresponding FICO tier."""
    if score >= 800:
        return 'Exceptional'
    elif score >= 740:
        return 'Very Good'
    elif score >= 670:
        return 'Good'
    elif score >= 580:
        return 'Fair'
    else: # Below 580
        return 'Poor'

def map_vantage_tier(score):
    """Maps a credit score to its corresponding VantageScore tier."""
    if score >= 781:
        return 'Excellent'
    elif score >= 661:
        return 'Good'
    elif score >= 601:
        return 'Fair'
    elif score >= 500:
        return 'Poor'
    else: # Below 500
        return 'Very Poor'

# Creates two new categorical features based on FICO and VantageScore ranges 
# using the existing 'credit_score' column in both train and test DataFrames.

for data in [train, test, orig]:
    data['credit_score_FICO_tier'] = data['credit_score'].apply(map_fico_tier)
    data['credit_score_Vantage_tier'] = data['credit_score'].apply(map_vantage_tier)


TARGET='loan_paid_back'


ROUND = []
rounding_levels = {'100s': -2, '10s': -1, '1s': 1}

for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        for df in [train, test, orig]:
            df[new_col_name] = df[col].round(level).astype(int)

print(f'{len(ROUND)} ROUND Features created.')


for data in [train, test, orig]:
    # split grade_subgrade into 2 columns 
    data['grade_letter'] = data['grade_subgrade'].str[0]
    data['grade_score'] = data['grade_subgrade'].str[1].astype(int)

    # data['financial_health'] = (data['credit_score'] / 850) * (1 - data['debt_to_income_ratio'])
    # data['loan_burden'] = data['loan_amount'] / (data['annual_income'] + 1)
    # data['monthly_burden'] = (data['loan_amount'] * data['interest_rate'] / 1200) / ((data['annual_income'] / 12) + 1)
    # data['credit_power'] = data['credit_score'] / (data['interest_rate'] + 0.1)
    # data['income_efficiency'] = data['annual_income'] * (1 - data['debt_to_income_ratio'])
    
    data['high_risk'] = ((data['debt_to_income_ratio'] > 0.4) & (data['credit_score'] < 650)).astype(int)
    data['low_risk'] = ((data['debt_to_income_ratio'] < 0.3) & (data['credit_score'] > 700)).astype(int)
    data['medium_risk'] = ((~data['high_risk'].astype(bool)) & (~data['low_risk'].astype(bool))).astype(int)


target = 'loan_paid_back'
common_columns = [col for col in train.columns if col not in ['id', target, 'annual_income', 'loan_amount']]

print("<------ Train ------>\n")
for col in common_columns:
    print(f"{col}: {train[col].nunique()} unique values")

print('\n<------ Test ------>\n')
for col in common_columns:
    print(f"{col}: {test[col].nunique()} unique values")

print('\n<------ Orig ------>\n')
for col in common_columns:
    print(f"{col}: {orig[col].nunique()} unique values")


test['loan_paid_back'] = -1

combine_cols = common_columns + [target]
combine = pd.concat([train[combine_cols], test[combine_cols], orig[combine_cols]],axis=0)
print("Combined data shape", combine.shape )


CATS = []
NUMS = []
for c in combine.columns[:-1]:
    t = "CAT"
    if combine[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combine[c].nunique()
    na = combine[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


CATS1 = []
SIZES = {}
for c in NUMS + CATS:
    n = c
    if c in NUMS: 
        n = f"{c}2"
        CATS1.append(n)
    combine[n],_ = combine[c].factorize()
    SIZES[n] = combine[n].max()+1

    combine[c] = combine[c].astype('int32')
    combine[n] = combine[n].astype('int32')

print("New CATS:", CATS1 )
print("Cardinality of all CATS:", SIZES )


pairs = combinations(CATS + CATS1, 2)
new_cols = {}
CATS2 = []

for c1, c2 in pairs:
    name = "_".join(sorted((c1, c2)))
    new_cols[name] = combine[c1] * SIZES[c2] + combine[c2]
    CATS2.append(name)
if new_cols:
    new_df = pd.DataFrame(new_cols)         
    combine = pd.concat([combine, new_df], axis=1) 

print(f"Created {len(CATS2)} new CAT columns")


CE_FEATS = CATS + CATS1 + CATS2
CE = []
new_cols = {}
for col in CE_FEATS:
    nm_col = f"CE_{col.upper()}"
    if nm_col not in combine.columns:
        new_cols[nm_col] = combine.groupby(col)["loan_paid_back"].transform("count").astype("int32")
        CE.append(nm_col)
tmp_df = pd.DataFrame(new_cols)
combine = pd.concat([combine, tmp_df], axis=1)


train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
del combine
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", orig.shape )


FEATURES = NUMS+CATS+CATS1+CATS2+CE
print(f"We have {len(FEATURES)} features.")

FOLDS = 5
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.01,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,          
    "alpha": 3.0,
}


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=128*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        #dt = cudf.DataFrame(self.df.iloc[a:b])
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1


# oof_preds = np.zeros(len(train))
# test_preds = np.zeros(len(test))

# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
#     print("#"*25)
#     print(f"### Fold {fold+1} ###")
#     print("#"*25)

#     Xy_train = train.iloc[train_idx][ FEATURES+['loan_paid_back'] ].copy()
#     Xy_more = orig[ FEATURES+['loan_paid_back'] ]
#     for k in range(1):
#         Xy_train = pd.concat([Xy_train,Xy_more],axis=0,ignore_index=True)
    
#     X_valid = train.iloc[val_idx][FEATURES].copy()
#     y_valid = train.iloc[val_idx]['loan_paid_back']
#     X_test = test[FEATURES].copy()

#     CC = CATS1+CATS2
#     print(f"Target encoding {len(CC)} features... ",end="")
#     for i,c in enumerate(CC):
#         if i%10==0: print(f"{i}, ",end="")
#         TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
#         Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['loan_paid_back']).astype('float32')
#         X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
#         X_test[c] = TE0.transform(X_test[c]).astype('float32')
#     print()

#     Xy_train[CATS] = Xy_train[CATS].astype('category')
#     X_valid[CATS] = X_valid[CATS].astype('category')
#     X_test[CATS] = X_test[CATS].astype('category')

#     Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'loan_paid_back')
#     dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
#     dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
#     dtest  = xgb.DMatrix(X_test, enable_categorical=True)

#     model = xgb.train(
#         params=params,
#         dtrain=dtrain,
#         num_boost_round=10_000,
#         evals=[(dtrain, "train"), (dval, "valid")],
#         early_stopping_rounds=200,
#         verbose_eval=200
#     )

#     oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
#     test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


# m = roc_auc_score(train.loan_paid_back, oof_preds)
# print(f"XGB with Original Datµa as rows CV = {m}")


# fig, ax = plt.subplots(figsize=(10, 5))
# xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
# plt.title("Top 20 Feature Importances (XGBoost)")
# plt.show()


TE_ORIG = []
CC = CATS+CATS1+CATS2

print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = orig.groupby(c).loan_paid_back.mean()
    tmp = tmp.astype('float32')
    tmp.name = f"TE_ORIG_{c}"
    TE_ORIG.append( f"TE_ORIG_{c}" )
    train = train.merge(tmp, on=c, how='left')
    test = test.merge(tmp, on=c, how='left')
print()


# FEATURES += TE_ORIG
# print(f"We have {len(FEATURES)} features.")

# FOLDS = 5
# SEED = 42

# params = {
#     "objective": "binary:logistic",  
#     "eval_metric": "auc",           
#     "learning_rate": 0.01,
#     "max_depth": 0,
#     "subsample": 0.8,
#     "colsample_bytree": 0.7,
#     "seed": SEED,
#     "device": "cuda",
#     "grow_policy": "lossguide", 
#     "max_leaves": 32,           
#     "alpha": 2.0,
# }


test_preds=pd.read_csv('/kaggle/input/ps-s5e11-blend/submission.csv')['loan_paid_back'].values


threshold_high = 0.96
threshold_low = 0.03

# الأقوى ثقة
high_confidence_idx =   (test_preds < threshold_low)

X_pseudo = test[high_confidence_idx]
y_pseudo = (test_preds[high_confidence_idx] > 0.5).astype(int)




y_=pd.concat([train["loan_paid_back"],pd.Series(y_pseudo,name="loan_paid_back")],axis=0).reset_index(drop=True)
train=pd.concat([train[FEATURES],X_pseudo[FEATURES]],axis=0).reset_index(drop=True)

train['loan_paid_back']=y_



train=train.fillna(0)
test=test.fillna(0)



oof_preds2 = np.zeros(len(train))
test_preds2 = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train)), train[TARGET])):
    X_train, X_val = train.loc[train_idx, FEATURES], train.loc[val_idx, FEATURES]
    y_train, y_val = train.loc[train_idx, TARGET], train.loc[val_idx, TARGET]
    X_test = test[FEATURES].copy()

    for col in tqdm(CATS1+CATS2):
        encoder = TargetEncoder(n_folds=10, smooth=0, seed=42, split_method='random', stat='mean')
        X_train[col] = encoder.fit_transform(X_train[col], y_train)
        X_val[col] = encoder.transform(X_val[col])
        X_test[col] = encoder.transform(X_test[col])

    param_grid = {
        'device': 'cuda',
        'val_metric_name': '1-auc_ovr',
        'random_state': 100,
        'verbosity': 2,
        'arch_type': 'tabm-mini',
        'tabm_k': 32,
        'num_emb_type': 'pwl',
        'd_embedding': 12,
        'batch_size': 128,
        'lr': 1e-3,
        'n_epochs': 10,
        'dropout': 0.1,
        'd_block': 256,
        'n_blocks': 3
    }

    model = TabM_D_Classifier(**param_grid)
    model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
    oof_preds2[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds2 += model.predict_proba(X_test)[:, 1]

    print(f'Fold {idx + 1}: {roc_auc_score(y_val, oof_preds2[val_idx])}')

    del model, X_train, X_val, y_train, y_val, X_test
    gc.collect()

test_preds2 /= 5
print(f'CV AUC: {roc_auc_score(train[TARGET], oof_preds2)}')


# m = roc_auc_score(train.loan_paid_back, oof_preds2)
# print(f"XGB with Original Data as columns CV = {m}")


# fig, ax = plt.subplots(figsize=(10, 5))
# xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
# plt.title("Top 20 Feature Importances (XGBoost)")
# plt.show()


# m = roc_auc_score(train.loan_paid_back, oof_preds+oof_preds2)
# print(f"Ensemble CV = {m}")


# SAVE OOF PREDS
# np.save('oof_xgb_with_orig_rows',oof_preds)
np.save('oof_xgb_with_orig_cols',oof_preds2)


# best_score = -1
# w1 = 0.5
# w2 = 0.5
# for i in np.arange(0, 1.01, 0.01):
#     score = roc_auc_score(train.loan_paid_back[:len(oof_preds)], (1-i) * oof_preds + i * oof_preds2[:len(oof_preds)])
#     if score > best_score:
#         best_score = score
#         w1 = 1 - i
#         w2  = i

# print(f"Best score {best_score:.6f} with weights = {w1, w2}")


sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub['loan_paid_back'] = test_preds2
sub.to_csv("submission_as_rows.csv",index=False)

# sub['loan_paid_back'] = test_preds
# sub.to_csv("submission_as_cols.csv",index=False)


# sub = pd.read_csv(f"{PATH}sample_submission.csv")
# sub['loan_paid_back'] = w1 * test_preds + w2 * test_preds2
# sub.to_csv("submission.csv",index=False)
# print('Submission shape',sub.shape)
# sub.head()


# plt.hist(sub.loan_paid_back,bins=100)
# plt.title('Test Preds')
# plt.ylim((0,50_000))
# plt.show()




