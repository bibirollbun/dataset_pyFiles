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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train_data


import seaborn as sns
import matplotlib.pyplot as plt


train_data.info()


train_data.describe()


numeric_df = train_data.select_dtypes(include=['int64', 'float64'])

if 'id' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['id'])

numeric_features = numeric_df.columns
for col in numeric_features:
    
    plt.figure(figsize=(10, 6))
    train_data[col].plot(kind='hist', bins=50, edgecolor='black')
    
    plot_title = f'Histogram of {col.replace("_", " ").title()}'
    plt.title(plot_title)
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Save the figure
    plt.tight_layout()




train_data["loan_paid_back"].value_counts()


categorical_features = train_data.select_dtypes(include=['object']).columns

CARDINALITY_LIMIT = 20 

for col in categorical_features:    
    crosstab_df = train_data.groupby([col, 'loan_paid_back']).size().unstack(fill_value=0)
    crosstab_df['total'] = crosstab_df.sum(axis=1)
    crosstab_df = crosstab_df.sort_values('total', ascending=False)
    is_truncated = False
    if len(crosstab_df) > CARDINALITY_LIMIT:
        data_to_plot = crosstab_df.head(CARDINALITY_LIMIT)
        is_truncated = True
    else:
        data_to_plot = crosstab_df
    data_to_plot = data_to_plot.drop(columns=['total'])
    
    plt.figure(figsize=(12, 7)) 
    
    data_to_plot.plot(kind='bar', edgecolor='black', stacked=True)
    
    col_title = col.replace("_", " ").title()
    if is_truncated:
        plot_title = f'Bar Chart of Top {CARDINALITY_LIMIT} {col_title} Categories'
    else:
        plot_title = f'Bar Chart of {col_title}'
    
    plt.title(plot_title)
    plt.xlabel(col_title)
    plt.ylabel('Count') 
    
    plt.xticks(rotation=45, ha='right') 
    
    plt.grid(axis='y', alpha=0.75)
    
    plt.tight_layout() 


numeric_df.corr()


encoded_train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
encoded_test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)

encoded_train_data.info()


from sklearn.model_selection import train_test_split
submission_test = encoded_test_data.drop('id', axis=1)
x = encoded_train_data.drop(['loan_paid_back', 'id'], axis=1)
y = encoded_train_data['loan_paid_back']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
import time
param_grid = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_leaf': 10
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **param_grid)
print(rf)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
y_pred_proba = rf.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc:.4f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(cm)

report = classification_report(y_test, y_pred)
print(report)


from xgboost import XGBClassifier

xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'verbosity': 3,
    'tree_method': 'hist'
}

xgboost_model = XGBClassifier(random_state=42, **xgb_params)

xgboost_model.fit(
    x_train, y_train,
)

y_pred_proba = xgboost_model.predict_proba(x_test)[:, 1]
y_pred = xgboost_model.predict(x_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc:.4f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(cm)

report = classification_report(y_test, y_pred)
print(report)


xgboost_model = XGBClassifier(random_state=42, **xgb_params)

xgboost_model.fit(
    x, y,
)

y_pred_proba = xgboost_model.predict_proba(submission_test)[:, 1]

final_df = pd.DataFrame({
    "id": encoded_test_data["id"],
    "loan_paid_back": y_pred_proba
})
final_df.to_csv("submission.csv", index=False)

