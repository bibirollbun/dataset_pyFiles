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
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col = 'id')


train.head(3)


test.head(3)


# Shape of the dataset

rows, cols = train.shape
print(f"The Train dataset contains: Rows: {rows} & Columns: {cols}")

rows, cols = test.shape
print(f"\nThe Test dataset contains: Rows: {rows} & Columns: {cols}")


# data information

train.info()
print('\n\n') # > to add space between two outputs
test.info()


train.describe()



test.describe()


# Target Column distribution plot

plt.figure(figsize=(5,3))
sns.countplot(x='y', data=train)
plt.title('Target Column Distribution', fontsize=12)
plt.xlabel('Target (y)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.show()

# Target column value counts

train['y'].value_counts(normalize=True)


# Categorical features in Train Data

categorical_features = [data for data in train.columns if train[data].dtype == 'object']

for col in train.columns:
    if col in categorical_features:
        print(col, train[col].unique())
        print('-'* 50)


fig, axes = plt.subplots(3,3, figsize=(20,15))
axes = axes.flatten()

for col, feature in enumerate(categorical_features):
    sns.countplot(data=train, x =feature, ax=axes[col], palette='rocket')
    axes[col].set_title(feature, fontsize=15)
    axes[col].tick_params(axis='x', rotation=45)

plt.suptitle('Categorical Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()


numerical_features = [data for data in train.columns if train[data].dtype != 'object']

numerical_features


# Distribution Plots for Numerical Features

plt.figure(figsize=(15,10))

for col, feature in enumerate(numerical_features,1):
    plt.subplot(len(numerical_features)//3+1, 3, col)
    sns.histplot(data=train, x=feature, bins=30, kde=True, color='orange')
    plt.suptitle('Numerical Features Distribution', fontsize=16)
    plt.title(f"{feature}", fontsize=11)
    plt.xlabel("")
    plt.grid(True)
    
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,10))

for col, feature in enumerate(numerical_features,1):
    plt.subplot(len(numerical_features)//3+1, 3, col)
    sns.boxplot(x=train[feature])
    plt.suptitle('Numerical Features Distribution', fontsize=16)
    plt.title(f'Box plot of {feature}', fontsize=11)
    plt.xlabel("")

plt.show()
    


# Checking Multicollinearity in Numerical Dataset

sns.heatmap(train.select_dtypes('number').drop(columns='y').corr())  #"number" is a general alias that includes all numeric dtypes


from sklearn.preprocessing import LabelEncoder

def label_encoding(train, test):

    # Make copies so original data is untouched
    train_enc = train.copy()
    test_enc = test.copy()

    label_encoders = {}

    #  Encode Categorical features in training set
    for column in train_enc.columns:
        if train_enc[column].dtype == 'object':
            le = LabelEncoder()
            train_enc[column] = le.fit_transform(train_enc[column].astype(str))
            label_encoders[column] = le

    # Encode categorical features in test set using train encoders
    for column in test_enc.columns:
        if column in label_encoders:
            le = label_encoders[column]
            test_enc[column] = test_enc[column].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        elif test_enc[column].dtype == 'object':
            test_enc[column] = -1 # default encoding for useen categorical columns

    return train_enc, test_enc, label_encoders
        
        



train, test, label_encoders = label_encoding(train, test)


train.head()


test.head()


from sklearn.preprocessing import StandardScaler

def data_standardization(train, test, target_variable):

    # Store targett columns seperately from training data
    # Only othr features should be scaled.
    target_values = train[target_variable]
    train = train.drop(columns = [target_variable])

    # Ensure train and test have same set of feature columns
    common_columns = train.columns.intersection(test.columns)
    train = train[common_columns]
    test = test[common_columns]


    std_scaler = StandardScaler()

    # Fit_transform train data & transform test data
    train_scaled = pd.DataFrame(std_scaler.fit_transform(train), columns = common_columns)
    test_scaled = pd.DataFrame(std_scaler.transform(test), columns = common_columns)

    # Reattach the target column back to scaled trainning data
    train_scaled[target_variable] = target_values.reset_index(drop=True)


    return train_scaled, test_scaled


train_scaled, test_scaled = data_standardization(train, test, 'y')


train_scaled.head()


test_scaled.head()


X = train_scaled.drop(columns = ['y'])
Y = train_scaled['y']


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=42)


X.shape, X_train.shape, X_test.shape


Y.shape, Y_train.shape, Y_test.shape


from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


# Base Models
cat_model = CatBoostClassifier(n_estimators = 500, verbose = 0, random_state = 42)
lgb_model = LGBMClassifier(n_estimators = 500, random_state = 42, verbose = -1)
xgb_model = XGBClassifier(n_estimators = 500, random_state = 42, use_label_encoder = False, eval_metric = 'logloss')
rf_model = RandomForestClassifier(n_estimators = 300, random_state = 42)


models = {
    'XGBoost' : xgb_model,
    'LightGBM' : lgb_model,
    'CatBoost' : cat_model,
    'Random Forest': rf_model
}

for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print(f'ROC AUC: {roc_auc_score(Y_test, y_pred)}')
    print(f'F1 Score: {f1_score(Y_test, y_pred)}')
    print(classification_report(Y_test, y_pred))

    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:\n", cm)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

Y_probabilities = lgb_model.predict_proba(test_scaled)[:, 1]

submission_df = pd.DataFrame({
    'id' : sample_submission['id'],
    'y' : Y_probabilities
})

submission_df.to_csv('submission.csv', index=False)

print('submission.csv created successfully.')
submission_df.head()






















