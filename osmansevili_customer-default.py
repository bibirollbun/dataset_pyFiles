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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_excel('/kaggle/input/da514-default-prediction/train-kaggle.xlsx')
test_df = pd.read_excel('/kaggle/input/da514-default-prediction/test-kaggle.xlsx')
sample_sub = pd.read_csv('/kaggle/input/da514-default-prediction/sample_submission.csv')

macro_df = pd.read_excel('/kaggle/input/da514-default-prediction/macro_data.xlsx')
unemployment_df = pd.read_excel('/kaggle/input/da514-default-prediction/unemployment.xlsx')


print(f" Train shape: {train_df.shape}")
print(f" Test shape: {test_df.shape}")
print(f" Sample submission shape: {sample_sub.shape}")


train_df.head()


train_df.isnull().sum()


target_counts = train_df['target'].value_counts()
target_pct = train_df['target'].value_counts(normalize=True) * 100
print(f"0 (Ödedi): {target_counts[0]} (%{target_pct[0]:.2f})")
print(f"1 (Temerrüt): {target_counts[1]} (%{target_pct[1]:.2f})") 


train_df['loan_category'].value_counts()


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Target dağılımı
axes[0, 0].bar(['Ödedi (0)', 'Temerrüt (1)'], target_counts.values, color=['green', 'red'])
axes[0, 0].set_title('Target Dağılımı', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Sayı')
for i, v in enumerate(target_counts.values):
    axes[0, 0].text(i, v + 100, str(v), ha='center', fontweight='bold')

# Loan amount dağılımı
axes[0, 1].hist(train_df['loan_amt'], bins=50, color='skyblue', edgecolor='black')
axes[0, 1].set_title('Kredi Miktarı Dağılımı', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Kredi Miktarı')
axes[0, 1].set_ylabel('Frekans')

# Target'a göre loan amount
train_df.boxplot(column='loan_amt', by='target', ax=axes[1, 0])
axes[1, 0].set_title('Target\'a Göre Kredi Miktarı', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Target (0: Ödedi, 1: Temerrüt)')
axes[1, 0].set_ylabel('Kredi Miktarı')
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['Ödedi', 'Temerrüt'])

# Loan category dağılımı
category_counts = train_df['loan_category'].value_counts()
axes[1, 1].bar(category_counts.index, category_counts.values, color='coral')
axes[1, 1].set_title('Kredi Kategorisi Dağılımı', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Kategori')
axes[1, 1].set_ylabel('Sayı')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()


# Train ve test'i birleştir
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['target'] = -1


all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)


all_data.info()


all_data['loan_release_date'] = pd.to_datetime(all_data['loan_release_date'])
all_data['loan_due_date'] = pd.to_datetime(all_data['loan_due_date'])


all_data['release_year'] = all_data['loan_release_date'].dt.year
all_data['release_month'] = all_data['loan_release_date'].dt.month
all_data['release_day'] = all_data['loan_release_date'].dt.day
all_data['release_dayofweek'] = all_data['loan_release_date'].dt.dayofweek
all_data['release_quarter'] = all_data['loan_release_date'].dt.quarter

all_data['due_year'] = all_data['loan_due_date'].dt.year
all_data['due_month'] = all_data['loan_due_date'].dt.month
all_data['due_day'] = all_data['loan_due_date'].dt.day



all_data['loan_duration_days'] = (all_data['loan_due_date'] - all_data['loan_release_date']).dt.days


all_data['payback_ratio'] = all_data['payback_amt'] / (all_data['loan_amt'] + 1)
all_data['lender_payback_ratio'] = all_data['lender_payback_amt'] / (all_data['lender_loan_amt'] + 1)


all_data['loan_payback_diff'] = all_data['payback_amt'] - all_data['loan_amt']
all_data['lender_diff'] = all_data['lender_payback_amt'] - all_data['lender_loan_amt']


all_data['loan_vs_lender_amt'] = all_data['loan_amt'] - all_data['lender_loan_amt']
all_data['payback_vs_lender_payback'] = all_data['payback_amt'] - all_data['lender_payback_amt']


le = LabelEncoder()
all_data['loan_category_encoded'] = le.fit_transform(all_data['loan_category'])


all_data['log_loan_amt'] = np.log1p(all_data['loan_amt'])
all_data['log_payback_amt'] = np.log1p(all_data['payback_amt'])
all_data['log_lender_loan_amt'] = np.log1p(all_data['lender_loan_amt'])


customer_stats = all_data[all_data['is_train'] == 1].groupby('customerID').agg({
    'loan_amt': ['mean', 'sum', 'count', 'std'],
    'target': 'mean'
}).reset_index()
customer_stats.columns = ['customerID', 'customer_loan_mean', 'customer_loan_sum', 
                          'customer_loan_count', 'customer_loan_std', 'customer_default_rate']
customer_stats['customer_loan_std'] = customer_stats['customer_loan_std'].fillna(0)

all_data = all_data.merge(customer_stats, on='customerID', how='left')
all_data[['customer_loan_mean', 'customer_loan_sum', 'customer_loan_count', 
          'customer_loan_std', 'customer_default_rate']] = all_data[
    ['customer_loan_mean', 'customer_loan_sum', 'customer_loan_count', 
     'customer_loan_std', 'customer_default_rate']].fillna(0)


macro_df['Year'] = macro_df['Year'].astype(int)
macro_df['Month'] = macro_df['Month'].apply(lambda x: datetime.strptime(x, '%B').month)

all_data = all_data.merge(
    macro_df, 
    left_on=['release_year', 'release_month'], 
    right_on=['Year', 'Month'], 
    how='left'
)
all_data = all_data.drop(['Year', 'Month'], axis=1, errors='ignore')


unemployment_dict = unemployment_df.set_index('Year')['Unemployment'].to_dict()
all_data['unemployment_rate'] = all_data['release_year'].map(unemployment_dict)


# Train ve test'i ayır
train_final = all_data[all_data['is_train'] == 1].copy()
test_final = all_data[all_data['is_train'] == 0].copy()

# Gereksiz kolonları çıkar
drop_cols = ['processID', 'customerID', 'loanID', 'MLid', 'loan_category', 
             'loan_release_date', 'loan_due_date', 'is_train', 'target']


X = train_final.drop(drop_cols, axis=1)
y = train_final['target']

X_test_final = test_final.drop(drop_cols, axis=1)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)


# Class weight hesapla (imbalanced data için)
class_weights = {0: 1, 1: len(y[y==0]) / len(y[y==1])}
print(f"Class weights: {class_weights}")


models_results = {}


# Logistic Regression 
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=0.1
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_val_scaled)
lr_f1 = f1_score(y_val, lr_pred)
models_results['Logistic Regression'] = {
    'model': lr_model,
    'f1_score': lr_f1,
    'predictions': lr_pred
}
print(f" F1 Score: {lr_f1:.4f}")


# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_f1 = f1_score(y_val, rf_pred)
models_results['Random Forest'] = {
    'model': rf_model,
    'f1_score': rf_f1,
    'predictions': rf_pred
}
print(f" F1 Score: {rf_f1:.4f}")


# XGBoost
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_f1 = f1_score(y_val, xgb_pred)
models_results['XGBoost'] = {
    'model': xgb_model,
    'f1_score': xgb_f1,
    'predictions': xgb_pred
}
print(f" F1 Score: {xgb_f1:.4f}")


# Validation tahminleri
y_pred = xgb_model.predict(X_val)
y_proba = xgb_model.predict_proba(X_val)[:, 1]


# F1 Score
f1 = f1_score(y_val, y_pred)
print(f" Validation F1 Score: {f1:.4f}")



cm = confusion_matrix(y_val, y_pred)


cm


best_f1 = 0
best_threshold = 0.5


for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_threshold = (y_proba >= threshold).astype(int)
    f1_temp = f1_score(y_val, y_pred_threshold)
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_threshold = threshold

print(f" Optimal Threshold: {best_threshold:.3f}")
print(f" Optimized F1 Score: {best_f1:.4f}")
print(f" Improvement: +{(best_f1 - f1) * 100:.2f}%")
 


# Optimal threshold ile yeni tahmin
y_pred_optimized = (y_proba >= best_threshold).astype(int)
print(classification_report(y_val, y_pred_optimized, target_names=['Ödedi (0)', 'Temerrüt (1)']))


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")


test_proba = xgb_model.predict_proba(X_test_final)[:, 1]
test_predictions = (test_proba >= best_threshold).astype(int) 


unique, counts = np.unique(test_predictions, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:5d} (%{c/len(test_predictions)*100:.1f})")


submission = sample_sub.copy()
submission['TARGET'] = test_predictions

submission.to_csv('submission.csv', index=False)


submission



















