import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col='id')

test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv",index_col='id')


train.head()


def new_feats(df):
    df = df.copy()
    df['balance_posi'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['long_duration'] = (df['duration'] >= 360).astype(int)
    df['campaign_multi'] = (df['campaign'] >= 2).astype(int)
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['high_campaign'] = (df['campaign'] >= 3).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 40, 55, 97], 
                             labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['cubed_duration'] = df['duration'] ** 3
    df['log_campaign']=np.log1p(df['campaign'])
    df['sqrt_age'] = df['age'] ** 2
    df['cubed_age'] = df['age'] ** 3
    df['log_age'] = np.log1p(df['age'])
    
    return df
train = new_feats(train)
test = new_feats(test)
month_to_quarter = {
        'jan': 1, 'feb': 1, 'mar': 1,
        'apr': 2, 'may': 2, 'jun': 2,
        'jul': 3, 'aug': 3, 'sep': 3,
        'oct': 4, 'nov': 4, 'dec': 4
    }
train['quarter'] = train['month'].map(month_to_quarter)
test['quarter'] = test['month'].map(month_to_quarter)


for col in ['job', 'marital', 'education', 'contact', 'poutcome']:
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[col + '_freq'] = train[col].map(freq_map)
    test[col + '_freq'] = test[col].map(freq_map)


train['job_duration_mean'] = train.groupby('job')['duration'].transform('mean')
test['job_duration_mean'] = test.groupby('job')['duration'].transform('mean')


def bin_pdays(df):
    df = df.copy()
    df['pdays_bin'] = -1  # default value for no previous contact
    mask = df['pdays'] > 0
    df.loc[mask, 'pdays_bin'] = pd.cut(
        df.loc[mask, 'pdays'],
        bins=[0, 5, 15, 50, 999],
        labels=False,
        include_lowest=True
    ).astype(float)
    return df

train = bin_pdays(train)
test = bin_pdays(test)


train["poutcome_grouped"] = train["poutcome"].apply(
    lambda x: "positive" if x == "success" else "negative"
)
train.drop(columns=["poutcome"], inplace=True)
test["poutcome_grouped"] = test["poutcome"].apply(
    lambda x: "positive" if x == "success" else "negative"
)
test.drop(columns=["poutcome"], inplace=True)


train['day_sin'] = np.sin(2 * np.pi * train['day'] / 31)
train['day_cos'] = np.cos(2 * np.pi * train['day'] / 31)

test['day_sin'] = np.sin(2 * np.pi * test['day'] / 31)
test['day_cos'] = np.cos(2 * np.pi * test['day'] / 31)


day_target_means = {
    1: 0.358098, 2: 0.175324, 3: 0.204208, 4: 0.185990, 5: 0.132650,
    6: 0.093481, 7: 0.098154, 8: 0.106369, 9: 0.104032, 10: 0.238264,
    11: 0.128927, 12: 0.177113, 13: 0.166694, 14: 0.112711, 15: 0.143117,
    16: 0.149409, 17: 0.098600, 18: 0.100237, 19: 0.075264, 20: 0.067663,
    21: 0.097294, 22: 0.161951, 23: 0.132059, 24: 0.139721, 25: 0.136525,
    26: 0.104896, 27: 0.109415, 28: 0.071508, 29: 0.068390, 30: 0.210000,
    31: 0.055202
}
train['day_te'] = train['day'].map(day_target_means)
test['day_te'] = test['day'].map(day_target_means)
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
train['month_ord'] = train['month'].map(month_map)
test['month_ord'] = test['month'].map(month_map)

mean_map = {
    'apr': 0.235654, 'aug': 0.112161, 'dec': 0.513291, 'feb': 0.206801,
    'jan': 0.124148, 'jul': 0.090847, 'jun': 0.103726, 'mar': 0.571355,
    'may': 0.071354, 'nov': 0.109806, 'oct': 0.490004, 'sep': 0.534755
}
train['month_te'] = train['month'].map(mean_map)
test['month_te'] = test['month'].map(mean_map)


job_target_means = {
    "admin.": 0.116453,
    "blue-collar": 0.067438,
    "entrepreneur": 0.081386,
    "housemaid": 0.084653,
    "management": 0.150392,
    "retired": 0.246241,
    "self-employed": 0.129443,
    "services": 0.082714,
    "student": 0.340784,
    "technician": 0.118321,
    "unemployed": 0.179823,
    "unknown": 0.120672
}
train['job_te'] = train['job'].map(job_target_means)
test['job_te'] = test['job'].map(job_target_means)

train['job_grouped'] = train['job'].replace({
    "student": "high",
    "retired": "medium",
    "unemployed": "medium",
    "management": "medium",
    "self-employed": "medium",
    "admin.": "low",
    "technician": "low",
    "entrepreneur": "low",
    "housemaid": "low",
    "services": "low",
    "blue-collar": "low",
    "unknown": "low"
})
test['job_grouped'] = test['job'].replace({
    "student": "high",
    "retired": "medium",
    "unemployed": "medium",
    "management": "medium",
    "self-employed": "medium",
    "admin.": "low",
    "technician": "low",
    "entrepreneur": "low",
    "housemaid": "low",
    "services": "low",
    "blue-collar": "low",
    "unknown": "low"
})


num_cols = train.select_dtypes(include=[np.number]).columns.tolist()[:-1]
object_cols = train.select_dtypes(include="object").columns


X = train.drop('y', axis=1)
y = train['y']

X_test = test


for col_name in object_cols:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name])
    X_test[col_name] = le.transform(X_test[col_name])


SEEDS = [42, 2023, 7]  # different seeds
N_SPLITS = 10

oof_preds = []
test_preds = []

for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Seed {seed}, Fold {fold + 1}/{N_SPLITS}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            random_state=seed,
            n_estimators=25000,
            learning_rate=0.05,
            num_leaves=100,
            max_depth=14,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.8,
            reg_lambda=3
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(period=500)]
        )
        
        # OOF for this fold
        oof_fold = model.predict_proba(X_val)[:, 1]
        # Test prediction for this fold
        test_fold = model.predict_proba(X_test)[:, 1]
        
        oof_preds.append(oof_fold)
        test_preds.append(test_fold)


# Initialize empty array for OOF predictions
oof_matrix = np.zeros((len(X), len(oof_preds)))  # shape: (n_samples, n_models)

model_idx = 0
for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # fill OOF for val_idx in correct position
        oof_matrix[val_idx, model_idx] = oof_preds[model_idx]
        model_idx += 1


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

meta_model = LogisticRegression(max_iter=1000)

meta_model.fit(oof_matrix, y)
print("Meta model trained!")


test_matrix = np.column_stack(test_preds)


final_preds = meta_model.predict_proba(test_matrix)[:, 1]


submission = pd.DataFrame({'id': test.index, 'prediction': final_preds})
submission.to_csv('final_submission.csv', index=False)




