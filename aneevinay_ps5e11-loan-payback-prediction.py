import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s5e11/train.csv'
test_path = '/kaggle/input/playground-series-s5e11/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def dataset_summary(datasets):
    summary = []

    for name, df, path in datasets:
        size_on_disk = os.path.getsize(path) / (1024 * 1024)  # MB
        size_in_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        rows, cols = df.shape

        summary.append({
            "Dataset": name,
            "Size on Disk (MB)": round(size_on_disk, 2),
            "Size in Memory (MB)": round(size_in_memory, 2),
            "# of Rows": rows,
            "# of Cols": cols
        })

    return pd.DataFrame(summary)


datasets = [
    ("train", train, train_path),
    ("test", test, test_path)
]

dataset_summary(datasets)


train.head()


test.head()


train["annual_income"] = np.log1p(train["annual_income"])
train["debt_to_income_ratio"] = np.log1p(train["debt_to_income_ratio"])

test["annual_income"] = np.log1p(test["annual_income"])
test["debt_to_income_ratio"] = np.log1p(test["debt_to_income_ratio"])


train.nunique()


train.info()


train['new_education_level'] = train['education_level'].astype('category').cat.codes
train['new_grade_subgrade'] = train['grade_subgrade'].astype('category').cat.codes

mean_target = train.groupby('loan_purpose')['loan_paid_back'].mean()
train['new_loan_purpose'] = train['loan_purpose'].map(mean_target)

train = pd.get_dummies(train, columns=['gender', 'marital_status', 'employment_status'], drop_first=False)

cols_to_drop = ['education_level', 'grade_subgrade', 'loan_purpose']
train.drop(columns=cols_to_drop, inplace=True)

print("✅ Encoding complete!")
print(train.head())


X=train.drop(columns=['id','loan_paid_back'],axis=1)
y=train['loan_paid_back']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    random_state=42,
    verbose=100
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", auc)


X_sample = X_train.sample(frac=0.20, random_state=42)
y_sample = y_train.loc[X_sample.index]

print("Sample size:", X_sample.shape)


X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(
    X_sample, y_sample,
    test_size=0.25,   
    random_state=42,
    stratify=y_sample
)


def objective(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
        "task_type": "GPU",     
    }

    model = CatBoostClassifier(**params)

    model.fit(
        X_train_opt,
        y_train_opt,
        eval_set=(X_test_opt, y_test_opt),
        verbose=False
    )

    preds = model.predict_proba(X_test_opt)[:, 1]
    auc = roc_auc_score(y_test_opt, preds)

    return auc


best_params = {
    "iterations": 1990,
    "depth": 4,
    "learning_rate": 0.12631748037338048,
    "l2_leaf_reg": 9.129473510577657,
    "random_strength":0.9650700748626257,
    "bagging_temperature": 0.5639052919751759 ,
    "border_count": 230,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "task_type": "CPU"  
}


model = CatBoostClassifier(**best_params)

model.fit(
    X, y,
    verbose=500
)


test_ids = test['id'].copy()

test['new_education_level'] = test['education_level'].astype('category').cat.codes
test['new_grade_subgrade'] = test['grade_subgrade'].astype('category').cat.codes

test['new_loan_purpose'] = test['loan_purpose'].map(mean_target)

test = pd.get_dummies(test, columns=['gender', 'marital_status', 'employment_status'], drop_first=False)

cols_to_drop = ['education_level', 'grade_subgrade', 'loan_purpose']
test.drop(columns=cols_to_drop, inplace=True)

test_model = test.drop(columns=['id'], errors='ignore') 
test_model = test_model.reindex(columns=X.columns, fill_value=0)
        
y_pred = model.predict_proba(test_model)[:, 1]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': y_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")

