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


test_id = test['id']


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


train.columns


from sklearn.model_selection import KFold


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


# Creating new features based on the frequency of numerical features
train, test = target_encoding(train, test)
train, test =  create_freq(train, test)


train[cat], test[cat] = train[cat].astype("category"), test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq'
]


train, test = train.drop(columns = remove+["id"]), test.drop(columns = remove)

print(f"Number of columns {len(train.columns.tolist())}")
print(train.columns.tolist())


train.head(2)


train.isnull().sum()[lambda x: x>0] # Null values count
X=train.drop(columns=['loan_paid_back'],axis=1)
y=train['loan_paid_back']


#Class Imbalance
count_0, count_1 = train['loan_paid_back'].value_counts()
scale_pos_weight = count_1 / count_0


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


cv_df = pd.DataFrame(cv_results)
print(cv_df.tail())


best_round = len(cv_results['valid auc-mean'])
best_auc = cv_results['valid auc-mean'][-1]
print(f"Best round: {best_round}, Best CV AUC: {best_auc:.7f}")


lgb_params["n_estimators"] = best_round + 300
# Prepare training data
X_train = train.drop(columns=target)
y_train = train[target]


# Train LGBM model
model = LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)


# Predict on test set
final = model.predict_proba(test.drop(columns = "id"))[:, 1]


submission=pd.DataFrame({'id':test_id,'loan_paid_back':final})
submission.to_csv('submission.csv',index=False)


submission.head(3)

