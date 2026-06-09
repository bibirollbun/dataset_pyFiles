import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna, json

import lightgbm as lgb

from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from cuml import LogisticRegression
from cuml.preprocessing.TargetEncoder import TargetEncoder


import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


# downcast numerical columns

def downcasting(data: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:

    mem_before = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
            
    for col in data.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="float")

    mem_after = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%\n")

    
    return data

# train = downcasting(train)
# test = downcasting(test)
# orig = downcasting(orig)


train.head()


orig.head()


target = 'loan_paid_back'
cats = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# collect only common columns
common_columns = [col for col in train.columns if col not in ['id', target]]


# function to compare train and test dataframes
def compare_dataframes(train_df, test_df, cols):
    report_data = []
    
    all_cols = cols

    for col in all_cols:
        row_data = {'Column': col}
        
        if col in train_df.columns and col in test_df.columns:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                train_stats = train_df[col].describe()
                test_stats = test_df[col].describe()
                
                row_data.update({
                    'Data Type': 'Numeric',
                    'Train Mean': train_stats['mean'],
                    'Test Mean': test_stats['mean'],
                    'Train Median': train_stats['50%'],
                    'Test Median': test_stats['50%'],
                    'Train Std': train_stats['std'],
                    'Test Std': test_stats['std']
                })
            else:
                train_counts = train_df[col].value_counts(normalize=True)
                test_counts = test_df[col].value_counts(normalize=True)
                
                row_data.update({
                    'Data Type': 'Categorical',
                    'Train Unique': len(train_df[col].unique()),
                    'Test Unique': len(test_df[col].unique()),
                    'Train Mode': train_df[col].mode()[0],
                    'Test Mode': test_df[col].mode()[0]
                })

            if 'Train Mean' in row_data:
                mean_diff_pct = (abs(row_data['Train Mean'] - row_data['Test Mean']) / 
                                 np.mean([row_data['Train Mean'], row_data['Test Mean']])) * 100
                row_data['Mean Diff %'] = f"{mean_diff_pct:.2f}%"
            
            row_data['Train Count'] = train_df[col].count()
            row_data['Test Count'] = test_df[col].count()
        
        report_data.append(row_data)

    report_df = pd.DataFrame(report_data).apply(lambda x: round(x, 2), axis = 0)
    return report_df.set_index('Column')


report = compare_dataframes(train, test, common_columns)
report


compare_dataframes(train, orig, common_columns)


cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
num_features = len(cols)

fig, axs = plt.subplots(2, 5, figsize=(20, 10))

fig.suptitle('Feature Distribution Analysis', fontsize=18, y=1.02)

for i in range(num_features):
    sns.histplot(
        data=train,
        x=cols[i],
        bins=20,
        ax=axs[0, i],
        color='lightgreen',
        edgecolor='black',
        kde=True
    )
    axs[0, i].set_title(cols[i])


# try logged version of columns
train_copy = train.copy(deep=True)
for col in cols:
    train_copy[col] = np.log1p(train_copy[col])
    
for i in range(num_features):
    sns.histplot(
        data=train_copy,
        x=cols[i],
        bins=20,
        ax=axs[1, i],
        color='lightgreen',
        edgecolor='black',
        kde=True
    )
    axs[1, i].set_title(f"{cols[i]} - logged verion")

plt.tight_layout(rect=[0, 0, 1, 0.98]) 
plt.show()


# split grade_subgrade into 2 columns 
for data in [train, test, orig]:
    data['grade'] = data['grade_subgrade'].str[0]
    data['grade_num'] = data['grade_subgrade'].str[1].astype(int)


# add to common columns
common_columns.append('grade')
common_columns.append('grade_num')

print("<------ Train ------>\n")
for col in common_columns:
    print(f"{col}: {train[col].nunique()} unique values")

print('\n<------ Test ------>\n')
for col in common_columns:
    print(f"{col}: {test[col].nunique()} unique values")

print('\n<------ Orig ------>\n')
for col in common_columns:
    print(f"{col}: {orig[col].nunique()} unique values")


fig, ax = plt.subplots(figsize=(10,6))
corr = train.corr(numeric_only= True).round(3)

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap', fontsize = 15, pad=10)
plt.tight_layout()
plt.show()



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


test['loan_paid_back'] = -1

combine_cols = ['annual_income', 'loan_amount', 'debt_to_income_ratio', 'credit_score', 'grade', 'grade_num',
       'interest_rate', 'gender', 'marital_status', 'education_level', 'credit_score_FICO_tier', 'credit_score_Vantage_tier',
       'employment_status', 'loan_purpose', 'grade_subgrade'] + ROUND +['loan_paid_back']

combine = pd.concat([train[combine_cols], test[combine_cols], orig[combine_cols]],axis=0)


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



import xgboost as xgb
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
    "max_leaves": 128,          

    "min_samples_split": 5,
    'lambda': 5.0, 
    'alpha': 2.5,
}


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

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
        TE0 = TargetEncoder(n_folds=10, smooth=1, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['loan_paid_back']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

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


m = roc_auc_score(train.loan_paid_back, oof_preds)
print(f"XGB with Original Data as rows CV = {m}")


fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()


# save results
submission['loan_paid_back'] = test_preds
submission.to_csv('submission_orig_as_rows.csv', index=False) 
submission.head()


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


target = 'loan_paid_back'
cats = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# collect only common columns
common_columns = [col for col in train.columns if col not in ['id', target]]


def credit_features(train, test):

    first_cols = set(train.columns.to_list())
    
    train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
    test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)
    
    train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
    test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']
    
    train['available_income'] = train['annual_income'] * (1 - train['debt_to_income_ratio'])
    test['available_income'] = test['annual_income'] * (1 - test['debt_to_income_ratio'])
    
    train['affordability'] = train['available_income'] / (train['loan_amount'] + 1)
    test['affordability'] = test['available_income'] / (test['loan_amount'] + 1)
    
    train['monthly_payment'] = train['loan_amount'] * (1 + train['interest_rate']/100) / 12
    test['monthly_payment'] = test['loan_amount'] * (1 + test['interest_rate']/100) / 12
    
    train['payment_to_income'] = train['monthly_payment'] / (train['annual_income']/12 + 1)
    test['payment_to_income'] = test['monthly_payment'] / (test['annual_income']/12 + 1)
    
    train['risk_score'] = (train['debt_to_income_ratio'] * 40 + 
                           (1 - train['credit_score']/850) * 30 + train['interest_rate'] * 2)
    test['risk_score'] = (test['debt_to_income_ratio'] * 40 + 
                          (1 - test['credit_score']/850) * 30 + test['interest_rate'] * 2)
    
    train['grade_number'] = train['grade_subgrade'].str[1].astype(int)
    test['grade_number'] = test['grade_subgrade'].str[1].astype(int)

    train['grade'] = train['grade_subgrade'].str[0]
    test['grade'] = test['grade_subgrade'].str[0]
    
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    train['grade_rank'] = train['grade'].map(grade_map)
    test['grade_rank'] = test['grade'].map(grade_map)
    
    train['grade_combined'] = train['grade_rank'] * 10 + train['grade_number']
    test['grade_combined'] = test['grade_rank'] * 10 + test['grade_number']
    
    train['credit_interest'] = train['credit_score'] * train['interest_rate'] / 100
    test['credit_interest'] = test['credit_score'] * test['interest_rate'] / 100
    
    train['income_credit'] = np.log1p(train['annual_income']) * train['credit_score'] / 1000
    test['income_credit'] = np.log1p(test['annual_income']) * test['credit_score'] / 1000
    
    train['debt_loan'] = train['debt_to_income_ratio'] * np.log1p(train['loan_amount'])
    test['debt_loan'] = test['debt_to_income_ratio'] * np.log1p(test['loan_amount'])

    created_columns = list(set(train.columns.to_list()) ^ first_cols)
    
    print(f"{len(created_columns)} Features created")
    
    return train, test, created_columns

train, test, new_cols = credit_features(train, test)

cats.append('grade')


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
    data['credit_score_FICO_tier'] = data['credit_score'].apply(map_fico_tier).astype('category')
    data['credit_score_Vantage_tier'] = data['credit_score'].apply(map_vantage_tier).astype('category')


cats.append('credit_score_FICO_tier')
cats.append('credit_score_Vantage_tier')

common_columns.append('credit_score_FICO_tier')
common_columns.append('credit_score_Vantage_tier')


orig['grade'] = orig['grade_subgrade'].str[0]


INTER = []
inter_cols = [col for col in common_columns if col not in ['annual_income', 'loan_amount']] # remmove high cardinality columns

for col1, col2 in combinations(list(set(inter_cols + cats)), 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

print(f'{len(INTER)} Features.')



ORIG = []

for col in common_columns:
    # MEAN
    mean_map = orig.groupby(col)[target].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(len(ORIG), 'Orig Features Created!!')


FEATURES = common_columns + ORIG + INTER + new_cols
print(len(FEATURES), 'Features will be used')


fig, ax = plt.subplots(figsize=(40,20))

corr = train.corr(numeric_only= True).round(3)

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap', fontsize = 15, pad=10)
plt.tight_layout()
plt.show()




def find_high_correlation_features(df, threshold=0.9):
    corr_matrix = df.corr(numeric_only= True).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for row in range(len(upper.columns)):
        for col in range(row + 1, len(upper.columns)):
            if upper.iloc[row, col] > threshold:
                to_drop.add(upper.columns[col]) # Add the column name (which is the second feature in the pair) to the set of features to drop.
                to_drop.add(upper.columns[row])

    return list(to_drop)


features_to_remove = find_high_correlation_features(train, threshold=0.9)
features_to_remove


X = train[FEATURES].copy(deep = True)
y = train[target].copy(deep = True)

test = test.drop(columns=['id'], axis=1).copy(deep=True)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 300,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cuda',
    'enable_categorical': True,
    "grow_policy": "lossguide", 

    'scale_pos_weight': 0.8, # usefull for unbalanced data
    'lambda': 5.0, 
    'alpha': 3.0,
    'max_bin': 512
}

params_lgbm = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'num_leaves': 128,
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'categorical_feature':cats,
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
    'scale_pos_weight' : 0.79, # mainly for unbalanced binary data
}


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(test))

oof_preds_lgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    for col in INTER:
        TE = TargetEncoder(n_folds=10, smooth=1.5, split_method='random', stat='mean')
    
        X_train[col] = TE.fit_transform(X_train[[col]], y_train)
        X_val[col] = TE.transform(X_val[[col]])
        X_test[col] = TE.transform(X_test[[col]])

    X_train[cats] = X_train[cats].astype('category')
    X_val[cats] = X_val[cats].astype('category')
    X_test[cats] = X_test[cats].astype('category')

    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=500)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds_xgb[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'XGB Fold {fold} AUC: {fold_score:.4f}')
    test_preds_xgb += model.predict_proba(X_test)[:, 1] / N_SPLITS

    print('---------------------------------------------')
    
    model_lgb = LGBMClassifier(**params_lgbm)
    
    model_lgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=300)],
              # verbose=500
            )

    val_preds_lgb = model_lgb.predict_proba(X_val)[:, 1]
    oof_preds_lgb[val_idx] = val_preds_lgb
    
    fold_score = roc_auc_score(y_val, val_preds_lgb)
    print(f'LGB Fold {fold} AUC: {fold_score:.4f}\n')
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

overall_auc = roc_auc_score(y,oof_preds_xgb )
print(f'====================')
print(f'XGB Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')
print(f'XGB Overall OOF AUC: {roc_auc_score(y, oof_preds_lgb):.4f}')
print(f'====================')


feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': FEATURES, 
    'importance': feature_importances,
    'importance_lgb': model_lgb.feature_importances_
})


fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Feature Importance Comparison', fontsize=18, y=1.02)


sns.barplot(data=importance_df.sort_values('importance', ascending=False).head(20),
            x='importance',           
            y='feature',
            ax=axs[0],                
            edgecolor='black',
           )
axs[0].set_title("XGBoost Model Feature Importance") 

sns.barplot(data=importance_df.sort_values('importance_lgb', ascending=False).head(20),
            x='importance_lgb',       
            y='feature',
            ax=axs[1],                
            edgecolor='black',
           )
axs[1].set_title("LGBoost Model Feature Importance") 

plt.tight_layout()
plt.show()


feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': FEATURES, 
    'importance': feature_importances,
    'importance_lgb': model_lgb.feature_importances_
})


fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle('Feature Importance Comparison - least important features', fontsize=18, y=1.02)


sns.barplot(data=importance_df.sort_values('importance', ascending=False).tail(20),
            x='importance',           
            y='feature',
            ax=axs[0],                
            edgecolor='black',
           )
axs[0].set_title("XGBoost Model Feature Importance") 

sns.barplot(data=importance_df.sort_values('importance_lgb', ascending=False).tail(20),
            x='importance_lgb',       
            y='feature',
            ax=axs[1],                
            edgecolor='black',
           )
axs[1].set_title("LGBoost Model Feature Importance") 

plt.tight_layout()
plt.show()


for pred in [oof_preds_xgb, oof_preds_lgb, oof_preds]:
        print(f"{pred} auc score : {roc_auc_score(y, pred):.4f}")


# save results
submission['loan_paid_back'] = test_preds_xgb
submission.to_csv('submission_xgb.csv', index=False) 

# save results
submission['loan_paid_back'] = test_preds_lgb
submission.to_csv('submission_lgb.csv', index=False) 


# save final results
submission['loan_paid_back'] = (test_preds_xgb + test_preds_lgb + test_preds) / 3
submission.to_csv('submission.csv', index=False) 
submission.head()

