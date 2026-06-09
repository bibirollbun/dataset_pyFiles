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


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e11/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e11/test.csv")
original_data = pd.read_csv(r"/kaggle/input/loan-data/loan_dataset_20000.csv")
sample_submission_data = pd.read_csv(r"/kaggle/input/playground-series-s5e11/sample_submission.csv")


print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("original_data :", original_data.shape)
print("sample_submission_data :", sample_submission_data.shape)


train_data.head()


original_data.head()


train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)


original = original_data[train_data.columns]
original = original.astype(train_data.dtypes.to_dict())
original.head()


#Combining train_data and original_data:
train_data = pd.concat([train_data, original], ignore_index=True)
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


num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['loan_paid_back']))
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


X = train_data.drop(['loan_paid_back'], axis=1)
y = train_data['loan_paid_back']
test = test_data.copy()


Params = {'depth': 3, 'learning_rate': 0.28742026777587476, 'l2_leaf_reg': 4.025073331171142, 'bagging_temperature': 0.4754208532198644, 'border_count': 255, 'random_strength': 8.624645251395403e-07}


parameters={'depth': 3, 'learning_rate': 0.2835383323520494, 'l2_leaf_reg': 6.926884198166292, 'bagging_temperature': 0.4514567482175748, 'border_count': 253, 'random_strength': 4.495324972363784e-07}
#value: 0.9225332679270997.


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Initialize model
cat_model = CatBoostClassifier(**parameters,
    eval_metric='AUC',
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

auc_scores = []   # To store AUC per fold
all_preds = []    # To store test predictions
oof_preds = np.zeros(len(y))  # Out-of-fold predictions

# ---- Cross-validation ----
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    #print(f"\nFold {fold}")

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    # Predict probabilities for validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute AUC
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc)
    #print(f"AUC: {auc:.4f}")
    
    # Store OOF predictions
    oof_preds[val_idx] = y_pred_proba
    
    # (Optional) If you have a test set, predict on it for averaging later
    test_proba = model.predict_proba(test)[:, 1]
    all_preds.append(test_proba)

# ---- After CV ----
#print("\nAUC-ROC Scores per Fold:", auc_scores)
print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")

# ---- Mean Accuracy (optional) ----
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(cat_model, X, y, cv=5, scoring='accuracy').mean()
print(f"Mean Accuracy: {accuracy:.4f}")

# ---- Prepare submission ----
preds = np.mean(all_preds, axis=0) if all_preds else oof_preds  # if you used test predictions
submission = pd.DataFrame({'id': sample_submission_data.id, 'loan_paid_back': preds})
print(submission.head())
submission.to_csv('submission_catboost.csv', index=False)


