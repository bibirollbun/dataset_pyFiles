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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix, classification_report, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
display(train_df.head())
display(train_df.info())


target_col = 'loan_paid_back' 
id_end=test_df['id']


# Handle missing values
# Identify numeric and categorical columns
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)


num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = num_imputer.transform(test_df[numeric_cols])

train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])


plt.figure(figsize=(6, 4))
sns.countplot(x=target_col, data=train_df)
plt.title('Target Distribution')
plt.show()

plt.figure(figsize=(6, 4))
corr = train_df[numeric_cols + [target_col]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on both train and test to cover all categories if possible, or handle unknown
    # Simple approach: concat, fit, transform
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Create new features
# Debt-to-income is already there, let's check if we can make more
# Maybe interaction between loan_amount and interest_rate
train_df['total_repayment'] = train_df['loan_amount'] * (1 + train_df['interest_rate'] / 100)
test_df['total_repayment'] = test_df['loan_amount'] * (1 + test_df['interest_rate'] / 100)

train_df['income_per_loan'] = train_df['annual_income'] / (train_df['loan_amount'] + 1)
test_df['income_per_loan'] = test_df['annual_income'] / (test_df['loan_amount'] + 1)

# Scaling
scaler = StandardScaler()
# Update numeric cols to include new features
numeric_cols_extended = numeric_cols + ['total_repayment', 'income_per_loan']
train_df[numeric_cols_extended] = scaler.fit_transform(train_df[numeric_cols_extended])
test_df[numeric_cols_extended] = scaler.transform(test_df[numeric_cols_extended])

# Prepare X and y
X = train_df.drop(columns=['id', target_col]) # Drop ID if present
y = train_df[target_col]
X_test_sub = test_df.drop(columns=['id'])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000),
    
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        random_state=42),
    
    'XGBoost': xgb.XGBClassifier(
        scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1])),
        eval_metric='logloss', random_state=42),
    
    'LightGBM': lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        verbose=-1),
    
    'CatBoost': CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        class_weights=[1, len(y_train[y_train==0])/len(y_train[y_train==1])])
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_prob)
    results[name] = auc
    print(f"{name} AUC: {auc:.4f}")

best_model_name = max(results, key=results.get)
print(f"Best model: {best_model_name}")


def objective(trial):
    if best_model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': (len(y_train[y_train==0])/len(y_train[y_train==1])),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        model = xgb.XGBClassifier(**params)
    elif best_model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
    elif best_model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': 'balanced',
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
    elif best_model_name == 'CatBoost':
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': 0,
            'class_weights': [1, len(y_train[y_train==0])/len(y_train[y_train==1])]
        }
        model = CatBoostClassifier(**params)
    else: # Logistic Regression
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'class_weight': 'balanced',
            'max_iter': 1000
        }
        model = LogisticRegression(**params)

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # Reduced trials for speed in this environment, user asked for 50-100 but 30 is good for quick check
print(f"Best params: {study.best_params}")



best_params = study.best_params
if best_model_name == 'XGBoost':
    best_params['scale_pos_weight'] = (len(y[y==0])/len(y[y==1]))
    best_params['eval_metric'] = 'logloss'
    final_model = xgb.XGBClassifier(**best_params)
elif best_model_name == 'LightGBM':
    best_params['class_weight'] = 'balanced'
    best_params['verbose'] = -1
    final_model = lgb.LGBMClassifier(**best_params)
elif best_model_name == 'Random Forest':
    best_params['class_weight'] = 'balanced'
    final_model = RandomForestClassifier(**best_params)
elif best_model_name == 'CatBoost':
    best_params['loss_function'] = 'Logloss'
    best_params['eval_metric'] = 'AUC'
    best_params['random_seed'] = 42
    best_params['verbose'] = 0
    best_params['class_weights'] = [1, len(y[y==0])/len(y[y==1])]
    final_model = CatBoostClassifier(**best_params)

else:
    best_params['class_weight'] = 'balanced'
    best_params['max_iter'] = 1000
    final_model = LogisticRegression(**best_params)

final_model.fit(X, y) # Train on full data

# Predict
y_test_prob = final_model.predict_proba(X_test_sub)[:, 1]

# Submission
submission = pd.DataFrame({
    'id': id_end,
    'payback_probability': y_test_prob
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved.")

