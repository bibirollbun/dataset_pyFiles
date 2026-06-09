import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from lightgbm import LGBMClassifier
import lightgbm as lgb
from scipy.stats import uniform, randint
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


#Droping "id" column
train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


print(f"df Train shape {train.shape}")
print(f"df Test shape {test.shape}")


pd.set_option('display.max_columns', None)
train.head()


print("Number of null value in Train DF : ",train.isna().sum().sum())
print("Number of null value in Test DF : ",test.isna().sum().sum())



#Check Duplicate Rows
print("Number of Duplicated Row in Train DF : ", train.duplicated().sum())



num_cols = train.select_dtypes(exclude= 'object').columns

cat_cols = train.select_dtypes(include= 'object').columns


print(f"\n Numerical features + Target ({len(num_cols)}):")
for i, col in enumerate(num_cols, 1):
    print(f"   {i}. {col}")
print("="*50)
print(f"\n Categorical features ({len(cat_cols)}):")
for i, col in enumerate(cat_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Total  features: {len(num_cols) + len(cat_cols)}")


train[num_cols].describe().T


type(num_cols)


# numerical features correlation
plt.figure(figsize=(8, 6))
correlation_matrix = train[num_cols ].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            linewidths=1, cmap="Greens")
plt.show()


#Droping Target column
num_cols=num_cols.drop('loan_paid_back')


plt.figure(figsize=(10, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 2, 2*i - 1)
    sns.histplot(train[col], kde=True, bins=40, color="#8da0cb")
    plt.title(f'Distribution: {col}')

    plt.subplot(len(num_cols), 2, 2*i)
    sns.boxplot(x=train[col], color="#fc8d62")
    plt.title(f'Boxplot: {col}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(train[col], label='Train', fill=True, alpha=0.5)
    sns.kdeplot(test[col], label='Test', fill=True, alpha=0.3)
    plt.title(f'Distribution comparison: {col}')
    plt.legend()
plt.tight_layout()
plt.show()


# 1. Basic counts
target_counts = train['loan_paid_back'].value_counts()

# 2. Percentages
target_percent = train['loan_paid_back'].value_counts(normalize=True) * 100

target_counts = train['loan_paid_back'].value_counts()
target_percent = train['loan_paid_back'].value_counts(normalize=True) * 100

# 2. Plot
plt.figure(figsize=(5,4))
bars = plt.bar(target_counts.index.astype(str),
               target_counts.values,
               color=['#66c2a5','#fc8d62'])

# Add percentage labels on each bar
for bar in bars:
    height = bar.get_height()
    percent = (height / target_counts.sum()) * 100
    plt.text(bar.get_x() + bar.get_width()/2, height + 1000,  # adjust '1000' if scale differs
             f'{percent:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Distribution of Loan Paid Back')
plt.xlabel('Loan Paid Back (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


for col in cat_cols:
    print(f"\n=== {col.upper()} ===")
    
    # Frequency table
    freq = train[col].value_counts(dropna=False)
  
    
    # Repayment rate (mean of target per category)
    repayment_rate = train.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False)

    
    # Combine both 
    summary = pd.concat([freq, repayment_rate], axis=1)
    summary.columns = ['Count', 'Repayment_Rate']
    print("\nSummary:")
    print(summary)
    
    # --- Visualization ---
    plt.figure(figsize=(8,4))
    
    # Bar for repayment rate (target mean)
    sns.barplot(
        x=repayment_rate.index,
        y=repayment_rate.values,
        palette="viridis"
    )
    plt.title(f'Repayment Rate by {col}')
    plt.ylabel('Mean loan_paid_back (repayment rate)')
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


train.head()


from scipy.stats import skew

skew_values = train[num_cols].apply(lambda x: skew(x.dropna()))
print(skew_values.sort_values(ascending=False))


skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()
print("Highly skewed columns:", skewed_cols)

for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col]  = np.log1p(test[col])

from sklearn.preprocessing import PowerTransformer

# Initialize Yeo-Johnson transformer
pt = PowerTransformer(method='yeo-johnson')

# Apply transformation to skewed columns
# train[skewed_cols] = pt.fit_transform(train[skewed_cols])
# test[skewed_cols] = pt.transform(test[skewed_cols])


for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)


target = 'loan_paid_back'
cols = train.columns
cols = cols.drop('loan_paid_back')


def target_encoding(train, predict, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

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
        mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    predict = predict.copy()
    return train, predict

train , test = target_encoding(train, test)


def create_frequency_features(df, df_test):

    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num_cols:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test

train, test = create_frequency_features(train, test)


train.head()



# train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
# test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)

# train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
# test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']

# train['available_income'] = train['annual_income'] * (1 - train['debt_to_income_ratio'])
# test['available_income'] = test['annual_income'] * (1 - test['debt_to_income_ratio'])

# train['affordability'] = train['available_income'] / (train['loan_amount'] + 1)
# test['affordability'] = test['available_income'] / (test['loan_amount'] + 1)

# train['monthly_payment'] = train['loan_amount'] * (1 + train['interest_rate']/100) / 12
# test['monthly_payment'] = test['loan_amount'] * (1 + test['interest_rate']/100) / 12

# train['payment_to_income'] = train['monthly_payment'] / (train['annual_income']/12 + 1)
# test['payment_to_income'] = test['monthly_payment'] / (test['annual_income']/12 + 1)

# train['risk_score'] = (train['debt_to_income_ratio'] * 40 + 
#                        (1 - train['credit_score']/850) * 30 + train['interest_rate'] * 2)
# test['risk_score'] = (test['debt_to_income_ratio'] * 40 + 
#                       (1 - test['credit_score']/850) * 30 + test['interest_rate'] * 2)


# train['credit_interest'] = train['credit_score'] * train['interest_rate'] / 100
# test['credit_interest'] = test['credit_score'] * test['interest_rate'] / 100

# train['income_credit'] = np.log1p(train['annual_income']) * train['credit_score'] / 1000
# test['income_credit'] = np.log1p(test['annual_income']) * test['credit_score'] / 1000

# train['debt_loan'] = train['debt_to_income_ratio'] * np.log1p(train['loan_amount'])
# test['debt_loan'] = test['debt_to_income_ratio'] * np.log1p(test['loan_amount'])

# train['log_income'] = np.log1p(train['annual_income'])
# test['log_income'] = np.log1p(test['annual_income'])

# train['log_loan'] = np.log1p(train['loan_amount'])
# test['log_loan'] = np.log1p(test['loan_amount'])

# for df in [train, test]:
#     df['income_to_dti'] = df['annual_income'] / (1 + df['debt_to_income_ratio'])
#     df['interest_to_score'] = df['interest_rate'] / df['credit_score']
#     df['loan_per_score'] = df['loan_amount'] / df['credit_score']
#     df['loan_to_dti'] = df['loan_amount'] / (1 + df['debt_to_income_ratio'])


# # credit score categories

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

# for data in [train, test]:
#     data['credit_score_FICO_tier'] = data['credit_score'].apply(map_fico_tier)
#     data['credit_score_Vantage_tier'] = data['credit_score'].apply(map_vantage_tier)



from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

train['grade'] = train['grade_subgrade'].str[0]
train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)

test['grade'] = test['grade_subgrade'].str[0]
test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)

grade_order = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}
train['grade'] = train['grade'].map(grade_order)
test['grade'] = test['grade'].map(grade_order)

train = train.drop('grade_subgrade', axis =1)
test = test.drop('grade_subgrade', axis =1)


cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
onehot_cols = cat_cols

# onehot_cols = ['gender', 'marital_status', 'education_level', 
#                'employment_status', 'loan_purpose']



# One-hot encode
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_train = ohe.fit_transform(train[onehot_cols])
encoded_test = ohe.transform(test[onehot_cols])

# Convert to DataFrame
encoded_train_df = pd.DataFrame(encoded_train, 
                                columns=ohe.get_feature_names_out(onehot_cols),
                                index=train.index)
encoded_test_df = pd.DataFrame(encoded_test, 
                               columns=ohe.get_feature_names_out(onehot_cols),
                               index=test.index)

# Concatenate back
train = pd.concat([train.drop(columns=onehot_cols), encoded_train_df], axis=1)
test = pd.concat([test.drop(columns=onehot_cols), encoded_test_df], axis=1)




train['credit_rank'] = train['grade'] * 10 + train['subgrade']
test['credit_rank']  = test['grade'] * 10 + test['subgrade']



X = train.drop(columns='loan_paid_back',axis=1)
y = train['loan_paid_back']


X.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier

params = dict(
    n_estimators=1320,
    learning_rate=0.05,
    num_leaves=93,
    max_depth=5,
    colsample_bytree=0.975,
    subsample=0.743,
    reg_alpha=2.95,
    reg_lambda=0.0022,
    random_state=42,
    n_jobs=-1,
    metric='auc',
    objective='binary',
    boosting_type='gbdt',
    verbosity=-1,
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_curves, fold_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"--- Fold {fold}/{skf.n_splits} ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred

    test_preds += model.predict_proba(test)[:, 1] / skf.n_splits

    auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, val_pred)
    roc_curves.append((fpr, tpr, auc))

overall_auc = roc_auc_score(y, oof_preds)
print("Fold AUCs:", [round(s, 4) for s in fold_scores])
print(f"Overall OOF AUC: {overall_auc:.5f}")



final_model = LGBMClassifier(**params)
final_model.fit(X, y)


# lgb_params = dict(
#     n_estimators=1320,
#     learning_rate=0.05,
#     num_leaves=93,
#     max_depth=5,
#     colsample_bytree=0.975,
#     subsample=0.743,
#     reg_alpha=2.95,
#     reg_lambda=0.0022,
#     random_state=42,
#     n_jobs=-1,
#     metric='auc',
#     objective='binary',
#     boosting_type='gbdt',
#     verbosity=-1,
# )

# xgb_params = dict(
#     objective="binary:logistic",
#     eval_metric="auc",
#     tree_method="hist",           
#     max_depth=6,
#     learning_rate=0.0669438421783529,
#     n_estimators=732,
#     min_child_weight=8.368496274182363,
#     subsample=0.8638990746572127,
#     colsample_bytree=0.9262609574627299,
#     gamma=1.9880100566380507,
#     reg_alpha=0.010470012214699875,
#     reg_lambda=0.010061409517576274,
#     max_bin=504,                  
#     random_state=42,
#     n_jobs=-1,
#     verbosity=0   
# )

# lgb_model = LGBMClassifier(**lgb_params)

# xgb_model = xgb.XGBClassifier(**xgb_params)


# # oof_preds = np.zeros(len(X))
# # test_preds = np.zeros(len(test))
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# roc_curves, fold_scores = [], []

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
#     print(f"--- Fold {fold}/{skf.n_splits} ---")
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

#     lgb_model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         eval_metric='auc',
#     )
#     lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

#     xgb_model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         verbose=False
#     )
#     xgb_pred = xgb_model.predict_proba(X_val)[:, 1]


#     val_pred =  xgb_pred 
#     val_pred = 0.6 * xgb_pred + 0.4 * lgb_pred
    
#     auc = roc_auc_score(y_val, val_pred)
#     fold_scores.append(auc)
#     print(f"Fold {fold} AUC: {auc:.4f}")

#     fpr, tpr, _ = roc_curve(y_val, val_pred)
#     roc_curves.append((fpr, tpr, auc))

# print("Fold AUCs:", [round(s, 4) for s in fold_scores])
# simple_avg_score = np.mean(fold_scores)
# print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(fold_scores):.5f})")


lgb_pred = final_model.predict_proba(test)[:, 1]
# xgb_pred = xgb_model.predict_proba(test)[:, 1]

# ensemble_pred = 0.6 * xgb_pred + 0.4 * lgb_pred
ensemble_pred = lgb_pred




sub['loan_paid_back'] = ensemble_pred

sub.to_csv('submission.csv', index=False)

sub.head()



