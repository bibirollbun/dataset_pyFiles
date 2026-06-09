import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder




!pip install -Uq scikit-learn==1.3.0



from sklearn.metrics import roc_auc_score, f1_score, classification_report





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


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')



train_transaction.head()


train_identity.head()


# first, we will explore the data for el test
# missing values
missing_values_count = train_transaction.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_transaction.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


train = train_transaction.merge(train_identity, on='TransactionID', how='left')



missing_percent = train.isnull().sum() / train.shape[0]
columns_to_drop = missing_percent[missing_percent > 0.90].index
train.drop(columns=columns_to_drop, axis=1, inplace=True)



# Fill remaining missing values with median
train_transaction = train_transaction.fillna(train_transaction.median(numeric_only=True))


del missing_values_count, total_cells, total_missing



def coorelation_analysis(cols,title='Coorelation Analysis',size=(12,12)):
    cols = sorted(cols)
    fig,axes = plt.subplots(1,1,figsize=size)
    df_corr = train_transaction[cols].corr()
    sns.heatmap(df_corr,annot=True,cmap='RdBu_r')
    axes.title.set_text(title)
    plt.show()


cols = ['V3', 'V9', 'V5', 'V11', 'V10', 'V8', 'V7', 'V6', 'V4', 'V2', 'V1']
coorelation_analysis(cols,title='Coorelation Analysis: V1-V11')


 train = train.drop(['TransactionID', 'TransactionDT'], axis=1)  # Drop ID cols
train.fillna(-999, inplace=True)  # Fill missing values


# dealing with imbalanced data
# encoding
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))


X = train.drop('isFraud', axis=1)
y = train['isFraud']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#  Random Forest model with balanced class weights to fix el data
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds_prob = rf_model.predict_proba(X_val)[:, 1]
rf_preds = rf_model.predict(X_val)


print("Random Forest ROC AUC:", roc_auc_score(y_val, rf_preds_prob))
print("Random Forest F1 Score:", f1_score(y_val, rf_preds))


# we will use lightgbm to boost el model 
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds_prob = lgb_model.predict_proba(X_val)[:, 1]
lgb_preds = lgb_model.predict(X_val)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbosity': -1}


param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [5, 10],
    'min_child_samples': [20, 50]
}



print("LightGBM ROC AUC:", roc_auc_score(y_val, lgb_preds_prob))
print("LightGBM F1 Score:", f1_score(y_val, lgb_preds))


# next we will average the predicted prob. to reduce individual weaknesses and improve overall performance 
combined_preds_prob = (rf_preds_prob + lgb_preds_prob) / 2
combined_preds = (combined_preds_prob > 0.5).astype(int)


# These are arrays of predicted probabilities from each model
rf_preds_prob = rf_model.predict_proba(X_val)[:, 1]
lgb_preds_prob = lgb_model.predict_proba(X_val)[:, 1]



combined_preds_prob = (rf_preds_prob + lgb_preds_prob) / 2



# لو اكبر من ال 0.5 يبقى fraud لو اقل تبقى 0
combined_preds = (combined_preds_prob > 0.5).astype(int)


# evaluation finally
roc_auc_score(y_val, combined_preds_prob)
f1_score(y_val, combined_preds)


