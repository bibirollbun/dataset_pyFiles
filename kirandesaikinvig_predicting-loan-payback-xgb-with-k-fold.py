# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve, 
                            classification_report, accuracy_score,
mean_absolute_error)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


train.describe()


test.describe()


train.head()


print("\nMissing values in train:", train.isnull().sum().sum())
print("\nMissing values in test:", test.isnull().sum().sum())
print("\nTrain Columns:", train.columns.tolist())


numcols = ['id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
           'interest_rate']
catcols = [col for col in test.columns if col not in numcols] 
#Note catcols was done on test and so excludes the target

print("Numerical Columns:", numcols)
print("\nCategorical Columns:", catcols)



# cardinalities
for col in catcols:
    print(f"Cardinality of {col}: ", train[col].nunique() )
for col in catcols:
    print(f"\nOptions for {col}:" ,train[col].unique())


# bar charts of %1 by category
for c in catcols:
    p = train.groupby(c)['loan_paid_back'].mean().sort_values() * 100
    plt.figure(); plt.bar(p.index.astype(str), p.values)
    plt.title(f"{'loan_paid_back'}=1 by {c}"); plt.ylabel("% with 1")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

# correlation matrix
plt.figure(figsize=(10,8))
corr_matrix = train[numcols+['loan_paid_back']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center = 0)
plt.title('Correlation Matrix')
plt.show()


# numeric distributions
interesting = ['annual_income', 'loan_amount', 'credit_score']
for c in interesting:
    x = train[c].clip(lower=0)
    print(f"{c}: skew={x.skew():.2f}, log1p skew={np.log1p(x).skew():.2f}")
    plt.figure(); x.hist(bins=50); plt.title(c); plt.show()
    plt.figure(); np.log1p(x).hist(bins=50); plt.title("log1p "+c); plt.show()



BASE = [col for col in train.columns if col not in ['id', 'loan_paid_back', 'grade']]

for col in BASE:
    mean_map = orig.groupby(col)['loan_paid_back'].mean()
    train[f"orig_mean_{col}"] = train[col].map(mean_map)
    test[f"orig_mean_{col}"] = test[col].map(mean_map)
    
    count_map = orig.groupby(col).size()
    train[f"orig_count_{col}"] = train[col].map(count_map)
    test[f"orig_count_{col}"] = test[col].map(count_map)


#Log1p of income
train['log_income'] = np.log1p(train['annual_income'].clip(lower=0))
test['log_income'] = np.log1p(test['annual_income'].clip(lower=0))

#Log1p of loan amount

train['log_loan'] = np.log1p(train['loan_amount'].clip(lower=0))
test['log_loan']  = np.log1p(test['loan_amount'].clip(lower=0))

#The income to loan ratio

train['income_to_loan_ratio'] = train['annual_income'] / (train['loan_amount'] + 1)
test['income_to_loan_ratio'] = test['annual_income'] / (test['loan_amount'] + 1)

#Total value of the debt
train['debt_absolute'] = train['annual_income'] * train['debt_to_income_ratio']
test['debt_absolute'] = test['annual_income'] * test['debt_to_income_ratio']

#The available income

train['available_income'] = train['annual_income'] * (1 - train['debt_to_income_ratio'])
test['available_income'] = test['annual_income'] * (1 - test['debt_to_income_ratio'])

#The affordability ratio

train['affordability_ratio'] = train['available_income'] / (train['loan_amount'] + 1)
test['affordability_ratio'] = test['available_income'] / (test['loan_amount'] + 1)

#The monthly payment

train['monthly_payment'] = train['loan_amount'] * (1 + train['interest_rate']/100) / 12
test['monthly_payment'] = test['loan_amount'] * (1 + test['interest_rate']/100) / 12

#The payment-to-income ratio

train['payment_to_income'] = train['monthly_payment'] / (train['annual_income']/12 + 1)
test['payment_to_income'] = test['monthly_payment'] / (test['annual_income']/12 + 1)

#A measure of risk (credit to Ozan Mohurcu for the suggested formu)
train['risk_score'] = (train['debt_to_income_ratio'] * 40 + 
                       (1 - train['credit_score']/850) * 30 + train['interest_rate'] * 2)
test['risk_score'] = (test['debt_to_income_ratio'] * 40 + 
                      (1 - test['credit_score']/850) * 30 + test['interest_rate'] * 2)

#Credit interest
train['credit_interest'] = train['credit_score'] * train['interest_rate'] / 100
test['credit_interest'] = test['credit_score'] * test['interest_rate'] / 100

#Income credit
train['income_credit'] = np.log1p(train['annual_income']) * train['credit_score'] / 1000
test['income_credit'] = np.log1p(test['annual_income']) * test['credit_score'] / 1000

train["dti_x_rate"]  = train["debt_to_income_ratio"] * train["interest_rate"]
test["dti_x_rate"] = test["debt_to_income_ratio"] * test["interest_rate"]

train["amount_x_dti"] = train["loan_amount"] * train["debt_to_income_ratio"]
test["amount_x_dti"] = test["loan_amount"] * test["debt_to_income_ratio"]

train["rate_x_score"] = train["interest_rate"] * train["credit_score"]
test["rate_x_score"] = test["interest_rate"] * test["credit_score"]




#Dealing with the grade_subgrade feature

train['grade_letter'] = train['grade_subgrade'].str[0].astype(str)
test['grade_letter'] = test['grade_subgrade'].str[0].astype(str)
train['grade_number'] = train['grade_subgrade'].str[1].astype(int)
test['grade_number'] = test['grade_subgrade'].str[1].astype(int)


grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
train['grade_rank'] = train['grade_letter'].map(grade_map)
test['grade_rank'] = test['grade_letter'].map(grade_map)

train['grade_combined'] = train['grade_rank'] * 10 + train['grade_number']
test['grade_combined'] = test['grade_rank'] * 10 + test['grade_number']

#Dealing with the education_level feature

education_mapping = {'High School':1, 'Other':2, "Bachelor's":3, "Master's":4, 'PhD':5}
train['numericaleducation'] = train['education_level'].map(education_mapping)
test['numericaleducation'] = test['education_level'].map(education_mapping)


cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade', 'grade_letter', 
           'grade_number']

for cat in cat_cols:
    mean_map = train.groupby(cat)['loan_amount'].mean()
    train[f'{cat}_loan_mean'] = train[cat].map(mean_map)
    test[f'{cat}_loan_mean'] = test[cat].map(mean_map)
    
    mean_map = train.groupby(cat)['credit_score'].mean()
    train[f'{cat}_credit_mean'] = train[cat].map(mean_map)
    test[f'{cat}_credit_mean'] = test[cat].map(mean_map)


from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

risk_features = ["debt_to_income_ratio", "interest_rate", "credit_score"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train["risk_score_lr"] = cross_val_predict(
    pipe, train[risk_features], train["loan_paid_back"],
    method="predict_proba", cv=kf
)[:, 1]

pipe.fit(train[risk_features], train["loan_paid_back"])
test["risk_score_lr"] = pipe.predict_proba(test[risk_features])[:, 1]


features = [col for col in train.columns if col not in (['id', 'loan_paid_back']+cat_cols)]
X = train[features].copy()
y = train['loan_paid_back'].copy()
X_test = test[features].copy()


N_SPLITS = 7
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5,
    'learning_rate': 0.01, 'n_estimators': 10000, 'colsample_bytree': 0.8,
    'subsample': 0.85, 'min_child_weight': 3, 'gamma': 0.05,
    'reg_alpha': 0.05, 'reg_lambda': 1.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'device': 'cuda', 'tree_method': 'hist'
}


oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test)) if 'X_test' in globals() and X_test is not None else None
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=200,
        verbose=False
    )

    p_va = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = p_va
    auc = roc_auc_score(y_va, p_va)
    mae = mean_absolute_error(y_va, p_va)
    print(f"Fold {fold}: AUC={auc:.4f} | MAE={mae:.4f}")

    if test_pred is not None:
        test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

    models.append(model)

print(f"\nOOF AUC={roc_auc_score(y, oof):.4f} | OOF MAE={mean_absolute_error(y, oof):.4f}")


#Submitting work

submission_df = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_pred
})

submission_df.to_csv('/kaggle/working/submission.csv', index = False)

