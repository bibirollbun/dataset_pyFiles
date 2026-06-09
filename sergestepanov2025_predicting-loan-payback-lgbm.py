from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import lightgbm as lgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, KFold
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier
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


train.columns


pd.set_option('display.max_columns', None)
train.head()


print("Number of null value in Train DF : ",train.isna().sum().sum())
print("Number of null value in Test DF : ",test.isna().sum().sum())



#Check Duplicate Rows
print("Number of Duplicated Row in Train DF : ", train.duplicated().sum())



num_cols = train.select_dtypes(exclude= 'object').columns
print('Numerical columns :' ,num_cols ,"\n\n number of numerical columns:" ,len(num_cols))

print("="*100)
cat_cols = train.select_dtypes(include= 'object').columns
print('Categorical columns :' ,cat_cols,"\n\n number of categorical columns:" ,len(cat_cols) )


train[num_cols].describe().T


# numerical features correlation
plt.figure(figsize=(8, 6))
correlation_matrix = train[num_cols ].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            linewidths=1, cmap="Greens")
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


X = train.drop(columns='loan_paid_back',axis=1)
y = train['loan_paid_back']


# Apply label encoding

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])



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



