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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import optuna
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


print("Размеры данных:")
print(f"Обучающая выборка: {train.shape}")
print(f"Тестовая выборка: {test.shape}")


plt.figure(figsize=(8, 5))
sns.countplot(x='y', data=train)
plt.title('Распределение целевой переменной')
plt.show()


num_cols = train.select_dtypes(include=np.number).columns.drop(['id', 'y'])
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(train[col], bins=50, kde=True)
    plt.title(f'Распределение {col}')
plt.tight_layout()
plt.show()


cat_cols = train.select_dtypes(include='object').columns
plt.figure(figsize=(15, 15))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i+1)
    sns.countplot(y=col, data=train, order=train[col].value_counts().index)
    plt.title(f'Распределение {col}')
    plt.ylabel('')
plt.tight_layout()
plt.show()


def feature_engineering(df):
    df = df.copy()
    
    df['contacted_more_than_once'] = (df['previous'] > 0).astype(int)
    df['balance_to_age_ratio'] = df['balance'] / (df['age'] + 1e-6)
    df['interaction_loan_housing'] = (df['loan'] == 'yes') & (df['housing'] == 'yes')
    
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map)
    
    return df


train = feature_engineering(train)
test = feature_engineering(test)


X = train.drop(['id', 'y'], axis=1)
y = train['y']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0),
        'random_state': 42,
        'device': 'cuda'
    }
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))])
    
    score = cross_val_score(model, X_train, y_train, 
                          scoring='roc_auc', 
                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                          n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)


print(f"Лучшее ROC-AUC: {study.best_value:.4f}")
print("Лучшие параметры:", study.best_params)


best_params = study.best_params

best_params.update({
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'n_jobs': -1
})

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(**best_params))])

final_model.fit(X_train, y_train)


calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)


val_probs = calibrated_model.predict_proba(X_val)[:, 1]
val_preds = calibrated_model.predict(X_val)


print("\nОценка качества модели:")
print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
print("\nОтчет по классификации:")
print(classification_report(y_val, val_preds))


# ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_val, val_probs):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Confusion matrix
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


test_probs = calibrated_model.predict_proba(test.drop('id', axis=1))[:, 1]


submission = submission.copy()
submission['y'] = test_probs
submission.to_csv('submission.csv', index=False)




