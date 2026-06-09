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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler
import os

# Set random state
RANDOM_STATE = 42


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train.columns


print(train.shape)
print(test.shape)


def check_missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (100*df.isnull().sum()/df.isnull().count()).sort_values(ascending=False).round(1)
    all_missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data = all_missing_data[all_missing_data['Total'] > 0]
    print(f"Missing {missing_data.shape[0]} / {df.shape[1]} columns\n")
    return missing_data

missing_data = check_missing(train)
missing_columns = missing_data.index
print(missing_data)
print("="*80)
print(missing_columns)


train.head()


plt.figure(figsize=(6, 4))
sns.countplot(x='efs', data=train, color="pink")
plt.title('Distribution of Event-Free Survival (Target)')
plt.show()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


for column in train.columns:
    print(f'{column} {train[column].unique()}')


# Feature Engineering
FEATURES = [col for col in train if col not in ["ID", "efs", "efs_time"]]
num_features = train[FEATURES].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = train[FEATURES].select_dtypes(include=["object"]).columns.tolist()

print(len(num_features))
print(len(cat_features))
print(len(FEATURES))


for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))



# Numerical → median
train[num_features] = train[num_features].fillna(train[num_features].median())
test[num_features] = test[num_features].fillna(train[num_features].median())

# Categorical → mode
train[cat_features] = train[cat_features].fillna(train[cat_features].mode().iloc[0])
test[cat_features] = test[cat_features].fillna(train[cat_features].mode().iloc[0])


X = train[FEATURES].copy()
y = train['efs'].copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_valid_scaled = X_valid.copy()

X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_valid_scaled[num_features] = scaler.transform(X_valid[num_features])



def evaluate_model(model, X_val, y_val, model_name):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    return {'model_name': model_name, 'accuracy': acc, 'auc': auc, 'model': model}

results = []


def optimize_logistic_regression(X_train, y_train, X_val, y_val, n_trials=30):    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga',
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        }
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best AUC: {study.best_value:.4f}")
    return study.best_params

best_lr_params = optimize_logistic_regression(X_train_scaled, y_train, X_valid_scaled, y_valid)

lr_model = LogisticRegression(**best_lr_params, solver='saga', max_iter=1000, random_state=RANDOM_STATE)
lr_model.fit(X_train_scaled, y_train)

lr_result = evaluate_model(lr_model, X_valid_scaled, y_valid, "Logistic Regression")
results.append(lr_result)

acc_lr = lr_result['accuracy']
auc_lr = lr_result['auc']


def dt_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': RANDOM_STATE
    }
    
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    
    return auc

dt_study = optuna.create_study(direction='maximize')
dt_study.optimize(dt_objective, n_trials=30, show_progress_bar=True)

print(f"Best AUC: {dt_study.best_value:.4f}")
print(f"Best Params: {dt_study.best_params}")


best_dt_params = dt_study.best_params
dt_model = DecisionTreeClassifier(**best_dt_params, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_valid)
y_pred_proba_dt = dt_model.predict_proba(X_valid)[:, 1]

acc_dt = accuracy_score(y_valid, y_pred_dt)
auc_dt = roc_auc_score(y_valid, y_pred_proba_dt)

print(f"Accuracy: {acc_dt:.4f}")
print(f"AUC-ROC: {auc_dt:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_dt))


def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    
    return auc

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=20, show_progress_bar=True)

print(f"Best AUC: {rf_study.best_value:.4f}")
print(f"Best Params: {rf_study.best_params}")


best_rf_params = rf_study.best_params
rf_model = RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_valid)
y_pred_proba_rf = rf_model.predict_proba(X_valid)[:, 1]

acc_rf = accuracy_score(y_valid, y_pred_rf)
auc_rf = roc_auc_score(y_valid, y_pred_proba_rf)

print(f"Accuracy: {acc_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_rf))


def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    
    return auc

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=20, show_progress_bar=True)

print(f"Best AUC: {xgb_study.best_value:.4f}")
print(f"Best Params: {xgb_study.best_params}")


print("=" * 50)
print("XGBOOST")
print("=" * 50)

best_xgb_params = xgb_study.best_params
xgb_model = XGBClassifier(**best_xgb_params, random_state=RANDOM_STATE, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_valid)
y_pred_proba_xgb = xgb_model.predict_proba(X_valid)[:, 1]

acc_xgb = accuracy_score(y_valid, y_pred_xgb)
auc_xgb = roc_auc_score(y_valid, y_pred_proba_xgb)

print(f"Accuracy: {acc_xgb:.4f}")
print(f"AUC-ROC: {auc_xgb:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_xgb))


def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': RANDOM_STATE,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    
    return auc

lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)

print(f"Best AUC: {lgb_study.best_value:.4f}")
print(f"Best Params: {lgb_study.best_params}")


print("=" * 50)
print("LIGHTGBM")
print("=" * 50)

best_lgb_params = lgb_study.best_params
lgb_model = LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, verbose=-1)
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_valid)
y_pred_proba_lgb = lgb_model.predict_proba(X_valid)[:, 1]

acc_lgb = accuracy_score(y_valid, y_pred_lgb)
auc_lgb = roc_auc_score(y_valid, y_pred_proba_lgb)

print(f"Accuracy: {acc_lgb:.4f}")
print(f"AUC-ROC: {auc_lgb:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_lgb))


def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'random_state': RANDOM_STATE,
        'verbose': False,
        'allow_writing_files': False
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    
    return auc

cat_study = optuna.create_study(direction='maximize')
cat_study.optimize(cat_objective, n_trials=30, show_progress_bar=True)

print(f"Best AUC: {cat_study.best_value:.4f}")
print(f"Best Params: {cat_study.best_params}")


print("=" * 50)
print("CATBOOST")
print("=" * 50)

best_cat_params = cat_study.best_params
cat_model = CatBoostClassifier(**best_cat_params, random_state=RANDOM_STATE, verbose=False)
cat_model.fit(X_train, y_train)

y_pred_cat = cat_model.predict(X_valid)
y_pred_proba_cat = cat_model.predict_proba(X_valid)[:, 1]

acc_cat = accuracy_score(y_valid, y_pred_cat)
auc_cat = roc_auc_score(y_valid, y_pred_proba_cat)

print(f"Accuracy: {acc_cat:.4f}")
print(f"AUC-ROC: {auc_cat:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_cat))


model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost'],
    'Accuracy': [acc_lr, acc_dt, acc_rf, acc_xgb, acc_lgb, acc_cat],
    'AUC-ROC': [auc_lr, auc_dt, auc_rf, auc_xgb, auc_lgb, auc_cat]
})

model_comparison = model_comparison.sort_values('AUC-ROC', ascending=True) 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_comparison.plot(x='Model', y='Accuracy', kind='barh', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('')
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

model_comparison.plot(x='Model', y='AUC-ROC', kind='barh', ax=axes[1], color='lightcoral', edgecolor='black')
axes[1].set_title('Model Comparison: AUC-ROC', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(model_comparison.sort_values('AUC-ROC', ascending=False))


final_model = CatBoostClassifier(**best_cat_params, random_state=42, verbose=0)
final_model.fit(X, y)

test_features = test.drop(['ID'], axis=1)
test_preds_proba = final_model.predict_proba(test_features)[:, 1] 
submission = pd.DataFrame({
    'ID': test['ID'],
    'prediction': test_preds_proba
})

submission.to_csv('submission.csv', index=False)
print(submission.head())

