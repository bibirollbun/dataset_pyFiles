%load_ext cudf.pandas


import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from itertools import combinations
from cuml.preprocessing import TargetEncoder

import gc

import warnings
warnings.filterwarnings('ignore')
                        
print(f"XGBoost version {xgb.__version__}")


PATH = "/kaggle/input/playground-series-s5e11/"
train = pd.read_csv(f"{PATH}train.csv").set_index('id')
print("Train shape", train.shape )
train.head()


test = pd.read_csv(f"{PATH}test.csv").set_index('id')
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



ROUND = []
rounding_levels = {'1000s': -3, '100s': -2, '10s': -1, '1s': 1}

for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        for df in [train, test, orig]:
            df[new_col_name] = df[col].round(level).astype(int)

print(f'{len(ROUND)} ROUND Features created.')


first_cols = set(train.columns.to_list())

for data in [train, test, orig]:
    # split grade_subgrade into 2 columns 
    data['grade_letter'] = data['grade_subgrade'].str[0]
    data['grade_score'] = data['grade_subgrade'].str[1].astype(int)
    
    data['high_risk'] = ((data['debt_to_income_ratio'] > 0.4) & (data['credit_score'] < 650)).astype(int)
    data['low_risk'] = ((data['debt_to_income_ratio'] < 0.3) & (data['credit_score'] > 700)).astype(int)
    data['medium_risk'] = ((~data['high_risk'].astype(bool)) & (~data['low_risk'].astype(bool))).astype(int)

    data['loan_to_income'] = data['loan_amount'] / (data['annual_income'] + 1)
    
    data['total_debt'] = data['debt_to_income_ratio'] * data['annual_income']
    
    data['available_income'] = data['annual_income'] * (1 - data['debt_to_income_ratio'])
    
    data['affordability'] = data['available_income'] / (data['loan_amount'] + 1)
    
    data['monthly_payment'] = data['loan_amount'] * (1 + data['interest_rate']/100) / 12
    
    data['payment_to_income'] = data['monthly_payment'] / (data['annual_income']/12 + 1)
    
    data['risk_score'] = (data['debt_to_income_ratio'] * 40 + 
                           (1 - data['credit_score']/850) * 30 + data['interest_rate'] * 2)
    
    data['grade_number'] = data['grade_subgrade'].str[1].astype(int)
    data['grade'] = data['grade_subgrade'].str[0]
    
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    data['grade_rank'] = data['grade_letter'].map(grade_map)
    
    data['grade_combined'] = data['grade_rank'] * 10 + data['grade_number']
    
    data['credit_interest'] = data['credit_score'] * data['interest_rate'] / 100
    
    data['income_credit'] = np.log1p(data['annual_income']) * data['credit_score'] / 1000
    
    data['debt_loan'] = data['debt_to_income_ratio'] * np.log1p(data['loan_amount'])


# created_columns = list(set(train.columns.to_list()) ^ first_cols)
created_cols = ['affordability', 'available_income', 'credit_interest', 'debt_loan', 'grade', 'grade_combined', 'grade_rank',
                'income_credit', 'loan_to_income', 'monthly_payment', 'payment_to_income', 'risk_score', 'total_debt']


target = 'loan_paid_back'
common_columns = [col for col in train.columns if col not in ['id', target, 'annual_income', 'loan_amount']]

print("<------ Train ------>\n")
for col in common_columns:
    print(f"{col}: {train[col].nunique()} unique values")

# print('\n<------ Test ------>\n')
# for col in common_columns:
#     print(f"{col}: {test[col].nunique()} unique values")

# print('\n<------ Orig ------>\n')
# for col in common_columns:
#     print(f"{col}: {orig[col].nunique()} unique values")


test['loan_paid_back'] = -1



combine_cols = ['annual_income', 'loan_amount', 'debt_to_income_ratio', 'credit_score', 'grade_letter', 'grade_score',
                'high_risk','low_risk','medium_risk',
       'interest_rate', 'gender', 'marital_status', 'education_level', 'credit_score_FICO_tier', 'credit_score_Vantage_tier',
       'employment_status', 'loan_purpose', 'grade_subgrade'] + ROUND + created_cols +['loan_paid_back']


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


CE_FEATS = CATS + CATS1
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


FEATURES = NUMS+CATS+CATS1+CE
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

    "min_samples_split": 5,
    'lambda': 5.0, 
    'alpha': 2.5,
}


params_lgbm = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 128,
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'categorical_feature':CATS,
    'subsample': 0.7,
    'reg_alpha': 3,
    'reg_lambda': 1,
    'random_state': 42,
    'max_bin': 512,
    'n_jobs': -1,
    'metric': 'auc',
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'verbosity': -1,
}

cat_params = {
    'loss_function': 'Logloss',
    'bootstrap_type': 'Bernoulli',
    'eval_metric': 'AUC',     
    'iterations': 100000,      
    'learning_rate': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'early_stopping_rounds': 100,
    'random_seed': 42,        
    'thread_count': -1,       
    'verbose': 1000,           
    'task_type': 'GPU'
}


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
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


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

oof_preds_lgb = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))

oof_preds_cat = np.zeros(len(train))
test_preds_cat = np.zeros(len(test))


kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    Xy_train = train.iloc[train_idx][ FEATURES+['loan_paid_back'] ].copy()
    Xy_more = orig[ FEATURES+['loan_paid_back'] ]
    for k in range(1):
        Xy_train = pd.concat([Xy_train,Xy_more],axis=0,ignore_index=True)
    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['loan_paid_back']
    X_test = test[FEATURES].copy()

    CC = CATS1
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        TE0 = TargetEncoder(n_folds=10, smooth=1.0, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['loan_paid_back']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    X_train = Xy_train.drop(columns='loan_paid_back', axis=1)
    y_train= Xy_train[['loan_paid_back']]

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'loan_paid_back')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=300,
        verbose_eval=300
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS

    # clear memory 
    del dtrain, dval, dtest
    gc.collect()
    
    print('\n---------------------------------------------\n')
    
    model_lgb = lgb.LGBMClassifier(**params_lgbm)
    
    model_lgb.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(stopping_rounds=300),
                         lgb.log_evaluation(period=300)
                        ],
            )

    val_preds_lgb = model_lgb.predict_proba(X_valid)[:, 1]
    oof_preds_lgb[val_idx] = val_preds_lgb
    
    fold_score = roc_auc_score(y_valid, val_preds_lgb)
    print(f'LGB Fold {fold} AUC: {fold_score:.4f}\n')
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / FOLDS

    # clear memory
    del model_lgb
    gc.collect() 
    
    print('\n---------------------------------------------\n')
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),      
              cat_features=CATS 
             )

    val_preds = model_cat.predict_proba(X_valid)[:, 1]
    oof_preds_cat[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_valid, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / FOLDS

    # clear memory 
    del model_cat, Xy_train, X_train, y_train, X_valid, y_valid, X_test
    gc.collect()


for m, value in [('XGB', oof_preds), ('LGB', oof_preds_lgb), ('CAT', oof_preds_cat)]:
    roc_score = roc_auc_score(train.loan_paid_back, value)
    print(f"{m} with Original Data as rows CV = {roc_score}")


for m, preds in [('XGB', test_preds), ('LGB', test_preds_lgb), ('CAT', test_preds_cat)]:
    
    sub = pd.read_csv(f"{PATH}sample_submission.csv")
    sub['loan_paid_back'] = preds
    sub.to_csv(f"submission_as_rows_{m}.csv",index=False)


fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()


TE_ORIG = []
CC = CATS+CATS1

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


FEATURES += TE_ORIG
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

    "min_samples_split": 5,
    'lambda': 5.0, 
    'alpha': 3.0,
}


oof_preds2 = np.zeros(len(train))
test_preds2 = np.zeros(len(test))

oof_preds_lgb2 = np.zeros(len(train))
test_preds_lgb2 = np.zeros(len(test))

oof_preds_cat2 = np.zeros(len(train))
test_preds_cat2 = np.zeros(len(test))


kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    Xy_train = train.iloc[train_idx][ FEATURES+['loan_paid_back'] ].copy()    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['loan_paid_back']
    X_test = test[FEATURES].copy()

    CC = CATS1
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        TE0 = TargetEncoder(n_folds=10, smooth=1.0, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['loan_paid_back']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    X_train = Xy_train.drop(columns='loan_paid_back', axis=1)
    y_train= Xy_train[['loan_paid_back']]

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'loan_paid_back')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)


    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=300,
        verbose_eval=300
    )

    oof_preds2[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds2 += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS

    # clear memory 
    del dtrain, dval, dtest
    gc.collect()
    
    print('\n---------------------------------------------\n')
    
    model_lgb = lgb.LGBMClassifier(**params_lgbm)
    
    model_lgb.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(stopping_rounds=300),
                         lgb.log_evaluation(period=300)
                        ],
            )

    val_preds_lgb2 = model_lgb.predict_proba(X_valid)[:, 1]
    oof_preds_lgb2[val_idx] = val_preds_lgb2
    
    fold_score = roc_auc_score(y_valid, val_preds_lgb2)
    print(f'LGB Fold {fold} AUC: {fold_score:.4f}\n')
    test_preds_lgb2 += model_lgb.predict_proba(X_test)[:, 1] / FOLDS

    # clear memory
    del model_lgb
    gc.collect() 
    
    print('\n---------------------------------------------\n')
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),      
              cat_features=CATS 
             )

    val_preds = model_cat.predict_proba(X_valid)[:, 1]
    oof_preds_cat2[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_valid, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    
    test_preds_cat2 += model_cat.predict_proba(X_test)[:, 1] / FOLDS

    # clear memory 
    del model_cat, Xy_train, X_train, y_train, X_valid, y_valid, X_test
    gc.collect()


for m, value in [('XGB', oof_preds2), ('LGB', oof_preds_lgb2), ('CAT', oof_preds_cat2)]:
    roc_score = roc_auc_score(train.loan_paid_back, value)
    print(f"{m} with Original Data as rows CV = {roc_score}")


for m, preds in [('XGB', test_preds2), ('LGB', test_preds_lgb2), ('CAT', test_preds_cat2)]:
    
    sub = pd.read_csv(f"{PATH}sample_submission.csv")
    sub['loan_paid_back'] = preds
    sub.to_csv(f"submission_as_cols_{m}.csv",index=False)


fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()


m = roc_auc_score(train.loan_paid_back, oof_preds+oof_preds2)
print(f"Ensemble CV = {m}")


# SAVE OOF PREDS
np.save('oof_xgb_with_orig_rows',oof_preds)
np.save('oof_xgb_with_orig_cols',oof_preds2)


sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub['loan_paid_back'] = (test_preds_cat2 + test_preds_cat)/2.
sub.to_csv("submission_cats.csv",index=False)

sub['loan_paid_back'] = (test_preds_lgb2 + test_preds_lgb)/2.
sub.to_csv("submission_lgb.csv",index=False)

sub['loan_paid_back'] = (test_preds + test_preds2)/2.
sub.to_csv("submission_lgb.csv",index=False)


sub['loan_paid_back'] = (test_preds_cat + test_preds_cat2 + test_preds_lgb + test_preds_lgb2 + test_preds + test_preds2)/6.
sub.to_csv("submission.csv",index=False)

print('Submission shape',sub.shape)
sub.head()


plt.hist(sub.loan_paid_back,bins=100)
plt.title('Test Preds')
plt.ylim((0,50_000))
plt.show()




