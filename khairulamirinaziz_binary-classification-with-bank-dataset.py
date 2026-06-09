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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


print('Train data missing val:\n', train_data.isnull().sum())
print('Test data missing val:\n', test_data.isnull().sum())


print('Train data duplicate:', train_data.duplicated().sum())
print('Test data duplicate:', test_data.duplicated().sum())

print('Train data shape:', train_data.shape)
print('Test data shape:', test_data.shape)


train_data.head(10)


train_data.info()


test_data.head(5)


#This library is for EDA

import matplotlib.pyplot as plt
import seaborn as sns


#Visualize the target variable distribution (y)

plt.figure(figsize=(6,5))
sns.countplot(x='y', data=train_data)
plt.title('Target variable distribution (y)')
plt.xlabel('y (0=No, 1=Yes)')
plt.show()

print(train_data['y'].value_counts(normalize=True).mul(100))


#Distribution of numerical columns

num_cols = train_data.select_dtypes(include=['int', 'float']).columns

#Checking outlier using boxplot
for col in num_cols:
    plt.figure(figsize=(10,3))
    sns.boxplot(x=col, data=train_data)
    plt.title(f'Outlier - {col}')
    plt.show()


#Visualize the numerical distribution

for col in num_cols:
    if col != 'y':
        plt.figure(figsize=(7,5))
        sns.histplot(data=train_data, x=col, hue='y', kde=True, stat='density', common_norm=False)
        plt.title(f'Distribution of {col} by y')
        plt.legend(title='y', labels=['No', 'Yes'])
        plt.show()


#Distribution of categorical columns

cat_cols = train_data.select_dtypes(include='object').columns

#Check using bar chart

for col in cat_cols:
    plt.figure(figsize=(10,3))
    ax = sns.countplot(x=col,hue='y', data=train_data)
    plt.title(f'Count of {col} by y')
    ax.tick_params(axis='x', rotation=30)
    plt.show()


#Check correlation for num features

corr_num = train_data[num_cols].corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr_num, annot=True, cmap='Oranges', fmt=".2f", linewidths=0.5)
plt.show()


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


for col in cat_cols:
    LE = LabelEncoder()
    train_data[col] = LE.fit_transform(train_data[col])
    test_data[col] = LE.transform(test_data[col])
    


train_data.head(5)


X = train_data.drop(columns='y')
y = train_data['y']


from sklearn.model_selection import StratifiedKFold

#Stratified = ensures each fold has roughly the same proportion of target classes.
#CV = splits the training data into multiple train/val partitions, so every row is used for validation once.
#ensures the proportion of each class in the target variable (y) is approximately the same in each fold as it is in the complete dataset.

#Define CV splitter
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)




from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


#Log Reg Baseline

#Scale for linear model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


logreg_scores=[]

for train_idx, val_idx in skf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    logreg_scores.append(auc)

print('Avg LogReg CV AUC:', sum(logreg_scores)/len(logreg_scores))


#Random Forest Baseline

from sklearn.ensemble import RandomForestClassifier

rf_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    rf_scores.append(auc)

print('RF CV AUC:', sum(rf_scores)/len(rf_scores))



import xgboost as xgb

xgb_scores = []
y_prob = np.zeros(len(test_data))

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42, 
                              use_label_encoder=False, eval_metric='auc')
 
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    y_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    xgb_scores.append(auc)
    y_prob += model.predict_proba(test_data)[:, 1]/skf.n_splits

print('XGB CV AUC:', sum(xgb_scores)/len(xgb_scores))


results = pd.DataFrame({
    "Model": ["LogReg", "RandomForest", "XGBoost"],
    "CV AUC": [
        sum(logreg_scores)/len(logreg_scores),
        sum(rf_scores)/len(rf_scores),
        sum(xgb_scores)/len(xgb_scores)
    ]
})
print(results)


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission['y'] = y_prob
submission.to_csv("submission_2.csv", index=False)
submission.head()

