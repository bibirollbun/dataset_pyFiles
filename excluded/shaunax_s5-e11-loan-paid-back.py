# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train_df.head()


train_df.info()


train_df.describe().T


train_df = train_df.drop(["id"], axis=1)
test_df = test_df.drop(["id"], axis=1)


print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")


train_df.duplicated().sum()


num_cols = train_df.select_dtypes(include = ["int", "float"]).columns
cat_cols = train_df.select_dtypes(include = ["object"]).columns
num_cols = num_cols.drop("loan_paid_back")


target_counts = train_df["loan_paid_back"].value_counts()
plt.figure(figsize = (8,4))
plt.bar(target_counts.index.astype(str), target_counts.values, color = ["green", "red"])
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.tight_layout()
plt.show()


for col in cat_cols:
    plt.figure(figsize = (10, 4))
    sns.countplot(data = train_df, x=col, hue = "loan_paid_back")
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()
print(f"Plotted all cat_cols")


def plot_hist(data, column_name):
    plt.figure(figsize = (10, 4))
    sns.histplot(data, kde=True, color="blue")
    plt.title(f"Distribution of {column_name}")
    plt.show()

for col in num_cols:
    plot_hist(train_df[col], col)
print(f"Plotted all num_cols for hist.")


for i in num_cols:
    
    plt.figure(figsize = (8,4))
    sns.boxplot(data = train_df, x = train_df[i])
    plt.show()


for col in num_cols:
    Q1 = train_df[col].quantile(0.15)
    Q3 = train_df[col].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df[col] = train_df[col].clip(lower = lower_bound, upper = upper_bound)
    test_df[col] = test_df[col].clip(lower = lower_bound, upper = upper_bound)


cols_to_corr = ["annual_income", "debt_to_income_ratio", "credit_score", 
                     "loan_amount", "interest_rate", "loan_paid_back"]

corr_df = train_df[cols_to_corr].corr()
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="viridis")
plt.show()


train_df.head()    


from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in cat_cols:
    if col in train_df.columns:  
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        label_encoders[col] = le
        print(f"\n{col} encoded - Classes: {le.classes_}")
    else:
        print(f"\nWarning: Column '{col}' not found in dataframe")


train_df.head()


label_encoders = {}

for col in cat_cols:
    if col in test_df.columns:  
        le = LabelEncoder()
        test_df[col] = le.fit_transform(test_df[col].astype(str))
        label_encoders[col] = le
        print(f"\n{col} encoded - Classes: {le.classes_}")
    else:
        print(f"\nWarning: Column '{col}' not found in dataframe")


test_df.head()


X = train_df.drop(columns = "loan_paid_back", axis=1)
y = train_df["loan_paid_back"]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
        'device': 'gpu'
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)  
        
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred)
        aucs.append(auc_score)
    
    return np.mean(aucs)


study = optuna.create_study(direction = "maximize", study_name = "best-lgb-params")
study.optimize(objective, n_trials = 10)


best_params = study.best_params
final_model = lgb.LGBMClassifier(**best_params, objective = "binary", metric = "auc")
final_model.fit(X, y)

predictions = final_model.predict_proba(test_df)[:, 1]


submission = pd.DataFrame({
    "id": sample_df["id"],
    "target": predictions
})

submission.to_csv('submission.csv', index=False)


plt.figure(figsize = (10, 8))

feature_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False).head(11)

sns.barplot(data = feature_imp, x = "importance", y = "feature", palette = "magma")
plt.title("Top 11 Important Features")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

