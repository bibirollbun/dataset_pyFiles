import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings

from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)


print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")


pd.set_option('display.max_columns', None) # check this
train.head()


print(f"Number of Null records in train df: {train.isna().sum().sum()}")
print(f"Number of Null records in test df: {test.isna().sum().sum()}")


print(f"Number of dupliacte rows in train df: {train.duplicated().sum()}")


num_cols = train.select_dtypes(exclude='object').columns
print('Numerical columns:', num_cols, "\n\n Number of numerical columns", len(num_cols))

print("-"*50)

cat_cols = train.select_dtypes(include='object').columns
print('Categorical columns:', num_cols, "\n\n Number of categorical columns", len(cat_cols))


train[num_cols].describe().T


plt.figure(figsize=(8,6))
correlation_matrix = train[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='0.2f', linewidths=1, cmap='Greens')
plt.show()


num_cols = num_cols.drop(['loan_paid_back'])


plt.figure(figsize=(10,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 2, 2*i - 1)
    sns.histplot(train[col], kde=True, bins=40, color="#8da0cb")
    plt.title(f'Distribution: {col}')

    plt.subplot(len(num_cols), 2, 2*i)
    sns.boxplot(x=train[col], color="#fc8d62")
    plt.title(f'Boxplot: {col}')

plt.tight_layout()
plt.show()


from scipy.stats import skew

skew_values = train[num_cols].apply(lambda x: skew(x.dropna()))
print(skew_values.sort_values(ascending=False))


skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()
print("Highly skewed columns:", skewed_cols)

# compress large values to make the data more balanced
for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col]  = np.log1p(test[col])


# If a number is too small or too big, just keep it within the 1% to 99% safe zone
for col in num_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)


# Checking train and test datasets are similar in their feature distributions, ensuring a fair evaluation of model performance.

plt.figure(figsize=(14,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2,3,i)
    sns.kdeplot(train[col], label='Train', fill=True, alpha=0.5)
    sns.kdeplot(test[col], label='Test', fill=True, alpha=0.3)
    plt.title(f'Distribution comparision: {col}')
    plt.legend()

plt.tight_layout()
plt.show()


# 1. Basic counts
target_counts = train['loan_paid_back'].value_counts()

# 2. Percentages
target_percent = train['loan_paid_back'].value_counts(normalize=True)*100

# 3. Plot
plt.figure(figsize=(5,4))
bars = plt.bar(target_counts.index.astype(str), target_counts.values, color=['#66c2a5','#fc8d62'])

# Add percentage labels on each bar
for bar in bars:
    height=bar.get_height()
    percent = (height/target_counts.sum()) * 100
    plt.text(bar.get_x() + bar.get_width()/2, height+1000, f'{percent:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Distribution of Loan Paid Back')
plt.xlabel('Loan Paid Back (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


for col in cat_cols:
    print(f"\n=== {col.upper()}===")
    
    # Frequency table
    freq = train[col].value_counts(dropna=False)
    
    # Repayments rate (mean of taeget per category)
    repayment_rate = train.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False)

    # Combine both
    summary = pd.concat([freq, repayment_rate], axis=1)
    summary.columns = ['Count', 'Repayment_Rate']
    print("\nSummary:")
    print(summary)

    # Visualization
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






X = train.drop(['loan_paid_back'], axis=1)
y = train['loan_paid_back']


def add_frequency_and_bins(train, test, q_list=[5, 10, 15]):

    
    train = train.copy()
    test = test.copy()

    # Detect column types
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------------------- #
    # Frequency encoding for categorical columns
    # ---------------------- #
    for col in cat_cols:
        freq = train[col].value_counts(dropna=False)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(freq.mean())

    # ---------------------- #
    # Quantile binning for numeric columns
    # ---------------------- #
    for col in num_cols:
        for q in q_list:
            try:
                train_bins, bins = pd.qcut(
                    train[col],
                    q=q,
                    labels=False,
                    retbins=True,
                    duplicates="drop"
                )
                train[f"{col}_bin{q}"] = train_bins
                test[f"{col}_bin{q}"] = pd.cut(
                    test[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )
            except Exception:
                train[f"{col}_bin{q}"] = 0
                test[f"{col}_bin{q}"] = 0

    print(f"Added frequency features for {len(cat_cols)} categorical cols "
          f"and bin features for {len(num_cols)} numeric cols.")

    return train, test

X,test = add_frequency_and_bins(X, test, q_list=[5, 10, 15])


# Apply label encoding

cat_cols = train.select_dtypes(include=['object','category']).columns.tolist()

# for col in cat_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     test[col] = le.fit_transform(test[col])

for col in cat_cols:
    le = LabelEncoder()
    le.fit(X[col])             # learn mapping from train data
    X[col] = le.transform(X[col])   # convert train
    test[col]  = le.transform(test[col])    # convert test using same mapping




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


lgb_pred = final_model.predict_proba(test)[:, 1]


sub['loan_paid_back'] = lgb_pred

sub.to_csv('submission.csv', index=False)

sub.head()

