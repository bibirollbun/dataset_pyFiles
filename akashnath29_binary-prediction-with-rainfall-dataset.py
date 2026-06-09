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


%pip install lightgbm optuna


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Summary of train data
print("\nTrain data info:")
print(train_data.info())
print("\nTrain data description:")
print(train_data.describe())

# Check for missing values
print("\nMissing values in train data:")
print(train_data.isnull().sum())
print("\nMissing values in test data:")
print(test_data.isnull().sum())

# Target distribution
print("\nTarget distribution:")
print(train_data['rainfall'].value_counts(normalize=True))


plt.figure(figsize=(12, 6))
sns.countplot(x='rainfall', data=train_data)
plt.title('Distribution of Rainfall')
plt.show()


plt.figure(figsize=(14, 10))
correlation = train_data.drop('id', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x='rainfall', y=feature, data=train_data)
    plt.title(f'{feature} vs Rainfall')
plt.tight_layout()
plt.show()


def clean_and_create_features(df):
    df_new = df.copy()
    
    print(f"Missing values before imputation: {df_new.isnull().sum().sum()}")
    
    for col in df_new.columns:
        if df_new[col].isnull().sum() > 0:
            if col not in ['id', 'rainfall']:
                median_val = df_new[col].median()
                df_new[col].fillna(median_val, inplace=True)
    
    print(f"Missing values after imputation: {df_new.isnull().sum().sum()}")
    
    df_new['temp_humidity_ratio'] = df_new['temparature'] / df_new['humidity'].replace(0, 0.1)
    df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
    df_new['pressure_change'] = df_new.groupby(['day'])['pressure'].diff().fillna(0)
    df_new['temp_dewpoint_diff'] = df_new['temparature'] - df_new['dewpoint']
    
    df_new['wind_x'] = np.cos(np.radians(df_new['winddirection']))
    df_new['wind_y'] = np.sin(np.radians(df_new['winddirection']))
    
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day'] / 365)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day'] / 365)
    
    df_new['sunshine_safe'] = df_new['sunshine'] + 0.1
    df_new['weather_index'] = df_new['humidity'] * df_new['cloud'] / df_new['sunshine_safe']
    
    if df_new.isnull().sum().sum() > 0:
        print("Remaining missing values by column:")
        print(df_new.isnull().sum())
        
        for col in df_new.columns:
            if df_new[col].isnull().sum() > 0:
                if col not in ['id', 'rainfall']:
                    if df_new[col].dtype in ['int64', 'float64']:
                        df_new[col].fillna(df_new[col].median(), inplace=True)
                    else:
                        df_new[col].fillna(df_new[col].mode()[0], inplace=True)
    
    return df_new

train_fe = clean_and_create_features(train_data)
test_fe = clean_and_create_features(test_data)

print("\nFinal check for missing values in train data:")
print(train_fe.isnull().sum().sum())
print("\nFinal check for missing values in test data:")
print(test_fe.isnull().sum().sum())


X = train_fe.drop(['id', 'rainfall'], axis=1)
y = train_fe['rainfall']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    model = LGBMClassifier(**params)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    return cv_scores.mean()



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.number}")
print(f"Best AUC: {study.best_trial.value:.4f}")
print("Best hyperparameters:", study.best_params)

best_params = study.best_params
final_model = LGBMClassifier(**best_params, random_state=42)
final_model.fit(X_train_scaled, y_train)

y_pred_proba = final_model.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation AUC: {val_auc:.4f}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5, 
    random_state=42,
)
rf_model.fit(X_train_scaled, y_train)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    missing=np.nan 
)
xgb_model.fit(X_train_scaled, y_train)

y_pred_lgbm = final_model.predict_proba(X_val_scaled)[:, 1]
y_pred_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
y_pred_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]

y_pred_ensemble = 0.5 * y_pred_lgbm + 0.25 * y_pred_rf + 0.25 * y_pred_xgb
ensemble_auc = roc_auc_score(y_val, y_pred_ensemble)
print(f"Ensemble Validation AUC: {ensemble_auc:.4f}")


X_test = test_fe.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X_test)

test_pred_lgbm = final_model.predict_proba(X_test_scaled)[:, 1]
test_pred_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
test_pred_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

test_pred_ensemble = 0.5 * test_pred_lgbm + 0.25 * test_pred_rf + 0.25 * test_pred_xgb

submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_pred_ensemble
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")




