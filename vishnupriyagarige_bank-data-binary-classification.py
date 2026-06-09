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


# Importing required libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e8/test.csv")
original_data = pd.read_csv(r"/kaggle/input/bank-full-data/bank-full.csv", sep=";")
sample_submission_data = pd.read_csv(r"/kaggle/input/playground-series-s5e8/sample_submission.csv")


print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("original_data :", original_data.shape)
print("sample_submission_data :", sample_submission_data.shape)


train_data.head()


original_data.head()


original_data['y'] = original_data['y'].map({"no": 0, "yes": 1})


train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)


# Combining train_data and original_data:
train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data.shape


train_data.head()


test_data.head()


train_data.isnull().sum().sort_values(ascending=False)


test_data.isnull().sum().sort_values(ascending=False)


train_data = train_data.dropna()
train_data = train_data.drop_duplicates()


train_data.info()


train_data.columns


train_data.describe()


num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['y']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col_name in train_data.columns:
    if train_data[col_name].dtypes=='object':
        train_data[col_name]=labelencoder.fit_transform(train_data[col_name])
        test_data[col_name]=labelencoder.transform(test_data[col_name])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])


plt.figure(figsize=(18,12))
sns.heatmap(train_data.corr(), annot=True,cmap="coolwarm")


X = train_data.drop(['y'], axis=1)
y = train_data['y']
test = test_data.copy()


#Best AUC ROC: 0.9683314721663688
#xgb_params = {'learning_rate': 0.03679726897488401, 'max_depth': 10, 'min_child_weight': 3, 'gamma': 0.9279624595163816, 'subsample': 0.705936847613209, 'colsample_bytree': 0.7936048487576377, 'n_estimators': 906}


#parameters = {'learning_rate': 0.013239509798508579, 'max_depth': 9, 'min_child_weight': 7, 'gamma': 1.0035966146495832, 'subsample': 0.8409430436540541, 'colsample_bytree': 0.7413099183642564, 'n_estimators': 3692}


#Best AUC ROC: 0.9669720089837235
#params = {'learning_rate': 0.05845878838640932, 'max_depth': 10, 'min_child_weight': 10, 'gamma': 1.2764186942221976, 'subsample': 0.7509857603947285, 'colsample_bytree': 0.7361907798652525, 'n_estimators': 804}


#parameters={'learning_rate': 0.009181007986686836, 'max_depth': 10, 'min_child_weight': 10, 'gamma': 0.13202271007039632, 'subsample': 0.6601926731740064, 'colsample_bytree': 0.776044994219286, 'n_estimators': 6381}
#value: 0.9671745276952729.
#Best AUC ROC: 0.9672268599858199
parameters = {'learning_rate': 0.010054105165313366, 'max_depth': 9, 'min_child_weight': 7, 'gamma': 0.33850531612442214, 'subsample': 0.6993830860066202, 'colsample_bytree': 0.7247793702265075, 'n_estimators': 3591}


from sklearn.model_selection import StratifiedKFold

#Define XGBClassifier
xgb_model = XGBClassifier(**parameters)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

auc_scores = []  # To store AUC scores
all_preds = []  # To store out-of-fold (OOF) predictions

# Perform Cross-Validation
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = xgb_model.fit(X_train, y_train)
    
    # Predict probabilities for AUC calculation
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilities for class 1
    proba = model.predict_proba(test)[:, 1]
    # Compute AUC-ROC score
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc)
    
    # Store Out-of-Fold (OOF) predictions
    all_preds.append(proba)
    
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy').mean()
# Print Results
print(f"\nAUC-ROC Scores per Fold: {auc_scores}")
print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
print(f"Mean Accuracy: {accuracy:.4f}")

# Store Predictions in DataFrame
preds = np.mean(all_preds, axis=0)
submission = pd.DataFrame({'id': sample_submission_data.id, 'y': preds})
print(submission.head())
submission.to_csv('submission_xgb.csv', index=False)

