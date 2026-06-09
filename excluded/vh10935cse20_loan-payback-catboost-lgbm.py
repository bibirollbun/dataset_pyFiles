import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


def data(df):
    print(df.head(2))
    print('\n' + '-' * 50)
    print("\nDataType : ",df.dtypes)
    print("\nShape of the Dataset : ",df.shape)
    print('\n' + '*' * 50)
    print("\n",df.info())
    print('\n' + '-' * 50)
    print("\nTotal Null Values Present : \n",df.isna().sum())
    print("\nDescriptive : \n",df.describe())
    print('\n' + '*' * 50)


data(train)


data(test)


train['grade_subgrade'].unique()


train['education_level'].unique()


test_id=test['id']


import matplotlib.pyplot as plt
import seaborn as sns


ax = sns.barplot(
    x='employment_status',
    y='loan_paid_back',
    data=train,
    palette='viridis' 
)
plt.title('Loan Repayment Rate by Employment Status', fontsize=16)
plt.xlabel('Employment Status', fontsize=12)
plt.ylabel('Proportion of Loans Paid Back (Repayment Rate)', fontsize=12)


ax = sns.countplot(
    y='employment_status', 
    data=train, 
    order=train['employment_status'].value_counts().index, 
    palette='plasma'
)

ax.set_title('2. Count of Applicants by Employment Status', fontsize=14)
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Employment Status', fontsize=12)
plt.show()


x = sns.histplot(
    train['annual_income'] / 1000, 
    kde=True, 
    color='skyblue' 
)
ax.set_title('1. Distribution of Annual Income (in Thousands)', fontsize=14)
ax.set_xlabel('Annual Income (in $1,000s)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

plt.show()


def create_freq(train,test):
    freq_features_train = pd.DataFrame(index=train.index)
    freq_features_test = pd.DataFrame(index=test.index)
    bin_features_train = pd.DataFrame(index=train.index)
    bin_features_test = pd.DataFrame(index=test.index)
    for col in cols:
        # --- Frequency encoding ---
        freq = train[col].value_counts()
        train[f"{col}_freq"] = train[col].map(freq)
        freq_features_test[f"{col}_freq"] = test[col].map(freq).fillna(freq.mean())

        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0
    train = pd.concat([train, freq_features_train, bin_features_train], axis=1)
    test = pd.concat([test, freq_features_test, bin_features_test], axis=1)

    return train, test


def target_encoding(train, test, n_splits=10):
    """
    Add K-Fold target mean encoded features to train and predict datasets.
    
    Parameters:
    - train: training DataFrame
    - predict: prediction/test DataFrame
    - target: name of the target column
    - n_splits: number of folds for K-Fold encoding
    
    Returns:
    - train and predict DataFrames with new mean encoded features
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=test.index)

    for col in cols:
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply global mean mapping to prediction/test data ---
        global_mean = train.groupby(col)[target].mean()
        mean_features_test[f'mean_{col}'] = test[col].map(global_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    test = pd.concat([test, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    test = test.copy()
    return train,test


target='loan_paid_back'


# Rounding the values
for c in ['annual_income', 'loan_amount']:
    for s, l in {'1s': 0, '10s': -1}.items():
        for g in [train, test]:
            g[f'{c}_ROUND_{s}'] = g[c].round(l).astype(int)


# Specific feature engineering
for gf in [train, test]:
    gf['subgrade'] = gf['grade_subgrade'].str[1:].astype(int)
    gf['grade'] = gf['grade_subgrade'].str[0]
    gf['total_debt_burden'] = (gf['loan_amount'] * gf['interest_rate'] / 100) / (gf['annual_income'] + 1) 
cols = train.drop(columns=[target,"id"]).columns.tolist()
cat = [c for c in cols if train[c].dtype in ["object","category"]]
num = [c for c in cols if train[c].dtype not in ["object","category","bool"]]


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


# Creating new features based on the frequency of numerical features
train, test = target_encoding(train, test)
train, test =  create_freq(train, test)


train[cat], test[cat] = train[cat].astype("category"), test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq']


train, test = train.drop(columns = remove+["id"]), test.drop(columns = remove)

print(f"Number of columns {len(train.columns.tolist())}")
print(train.columns.tolist())


train.isnull().sum()[lambda x: x>0] # Null values count
X=train.drop(columns=['loan_paid_back'],axis=1)
y=train['loan_paid_back']


import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation,LGBMClassifier


lgb_train = lgb.Dataset(X, label=y, free_raw_data=True)


lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'max_depth': 6, 'num_leaves': 50, 'learning_rate': 0.01,
    'colsample_bytree': 0.8, 'subsample': 0.8,
    'subsample_freq': 1, 'min_child_samples': 20, 'reg_alpha': 0.05,
    'reg_lambda': 0.1, 'random_state': 42,
    'n_jobs': -1, 'device': 'gpu','verbose': -1,
    #'scale_pos_weight':scale_pos_weight
}


cv_results = lgb.cv(
    params=lgb_params,
    train_set=lgb_train,
    num_boost_round=20000,
    nfold=7,
    stratified=True,
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period = 300)],
    seed=42
)


#cv_df = pd.DataFrame(cv_results)


lgbm_best_round = len(cv_results['valid auc-mean'])
lgbm_best_auc = cv_results['valid auc-mean'][-1]
print(f"Best round: {lgbm_best_round}, Best CV AUC: {lgbm_best_auc:.7f}")


#lgb_params["n_estimators"] = best_round + 300
# Prepare training data
#X_train = train.drop(columns=target)
#y_train = train[target]


# Train LGBM model
#model = LGBMClassifier(**lgb_params)
#model.fit(X_train, y_train)


# Predict on test set
#final_1 = model.predict_proba(test.drop(columns = "id"))[:, 1]


from catboost import CatBoostClassifier
import catboost as cb


cat_train = cb.Pool(data=X, label=y, cat_features=cat)


cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'depth': 6,
    'learning_rate': 0.01,
    # 'colsample_bylevel': 0.8, <--- REMOVED
    #'subsample': 0.8, # Row subsampling is usually fine
    'l2_leaf_reg': 0.1,
    'min_data_in_leaf': 20,
    'task_type': 'GPU', # Stays on GPU
    'random_seed': 42,
    'verbose': 0,
    #'thread_count': -1
}



cv_results_cat = cb.cv(
    pool=cat_train,  # Use the CatBoost Pool object
    params=cat_params, # Use the CatBoost parameter dictionary
    # Equivalent to 'num_boost_round' (max number of trees)
    iterations=5000, 
    # Equivalent to 'nfold'
    fold_count=5,
    # CatBoost uses its own method for stratification (set to True for binary tasks)
    shuffle=True, # Randomly shuffle the data before splitting
    # Equivalent to early_stopping(stopping_rounds=100)
    early_stopping_rounds=100,
    # Equivalent to log_evaluation(period=300)
    verbose=300, 
    seed=42, )


#cv_df = pd.DataFrame(cv_results)


cat_best_round = len(cv_results['valid auc-mean'])
cat_best_auc = cv_results['valid auc-mean'][-1]
print(f"Best round: {cat_best_round}, Best CV AUC: {cat_best_auc:.7f}")


N_TRAIN = X.shape[0]
N_TEST = test.shape[0]

lgbm_oof = np.zeros(N_TRAIN)
cat_oof = np.zeros(N_TRAIN)
lgbm_test_preds = np.zeros(N_TEST)
cat_test_preds = np.zeros(N_TEST)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    
    # Split Data
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    # --- 1. Train and Predict LGBM ---
    lgbm_model_fold = LGBMClassifier(
        **lgb_params,
        n_estimators=lgbm_best_round + 300, # Use best round from LGBM CV
        #random_state=42
    )
    lgbm_model_fold.fit(X_train_fold, y_train_fold)
    
    # OOF Prediction (Train Set)
    lgbm_oof[val_index] = lgbm_model_fold.predict_proba(X_val_fold)[:, 1]
    
    # Test Prediction (for averaging)
    lgbm_test_preds += lgbm_model_fold.predict_proba(test.drop(columns="id"))[:, 1] / kf.n_splits
    
    # --- 2. Train and Predict CatBoost ---
    cat_model_fold = CatBoostClassifier(
        **cat_params,
        iterations=cat_best_round + 300,
        cat_features=cat,# Use best round from CatBoost CV
        #random_state=42,
        #verbose=0
    )
    cat_model_fold.fit(X_train_fold, y_train_fold)
    
    # OOF Prediction (Train Set)
    cat_oof[val_index] = cat_model_fold.predict_proba(X_val_fold)[:, 1]
    
    # Test Prediction (for averaging)
    cat_test_preds += cat_model_fold.predict_proba(test.drop(columns="id"))[:, 1] / kf.n_splits


print("\n--- OOF Generation Complete ---")
print(f"LGBM OOF AUC: {roc_auc_score(y, lgbm_oof):.7f}")
print(f"CatBoost OOF AUC: {roc_auc_score(y, cat_oof):.7f}")


from sklearn.linear_model import LogisticRegression


X_blender = pd.DataFrame({
    'lgbm_oof': lgbm_oof,
    'cat_oof': cat_oof
})


blender = LogisticRegression(solver='liblinear')
blender.fit(X_blender, y)


X_test_blender = pd.DataFrame({
    'lgbm_oof': lgbm_test_preds,
    'cat_oof': cat_test_preds
})


final_blended_preds = blender.predict_proba(X_test_blender)[:, 1]


print(f"Blender OOF AUC: {roc_auc_score(y, blender.predict_proba(X_blender)[:, 1]):.5f}")


submission=pd.DataFrame({'id':test_id,'loan_paid_back':final_blended_preds})
submission.to_csv('submission.csv',index=False)


submission.head(3)

