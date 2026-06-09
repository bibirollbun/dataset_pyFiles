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
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import glob

# Define the path to the CSV files (use wildcards to match multiple files)
csv_files_path = "/kaggle/input/neo-bank-non-sub-churn-prediction/*.parquet"

# Use glob to get a list of all CSV file paths
csv_files = glob.glob(csv_files_path)

# Read each CSV file into a DataFrame and store them in a list
dataframes = [pd.read_parquet(file) for file in csv_files]



# Concatenate all DataFrames into a single DataFrame
train = pd.concat(dataframes, ignore_index=True)


train.shape


# train = pd.read_parquet("/kaggle/input/neo-bank-non-sub-churn-prediction/train_2023.parquet")
# train.shape


train.head()


# drop id, customer_id
train.drop(columns=['Id', 'customer_id'], axis=1, inplace=True)


# features
train.columns


# info
train.info()


## statistical analysis
train.describe()


# null check
train.isnull().sum().sum()


# create year feature
train['date'] = pd.to_datetime(train['date'])
train['date_of_birth'] = pd.to_datetime(train['date_of_birth'])



train[['date', 'date_of_birth']].dtypes


# create year and month of transaction form date feature 
train['transaction_year'] = train['date'].dt.year
train['transaction_month'] = train['date'].dt.month


# create year and month of transaction form date_of_birth feature 
train['date_of_birth_year'] = train['date_of_birth'].dt.year
train['date_of_birth_month'] = train['date_of_birth'].dt.month


# drop date and date_of_birth feature
train.drop(columns=["date", "date_of_birth"], axis=1, inplace=True)


categorical_features = [feature for feature in train.columns if train[feature].dtypes in [object]]
numercial_features = [feature for feature in train.columns if train[feature].dtypes not  in [object, bool]]
bool_features = [feature for feature in train.columns if train[feature].dtypes in [bool]]


categorical_features, numercial_features, bool_features


num_corr_matrix = train[numercial_features].corr()
num_corr_matrix


import seaborn as sns
import matplotlib.pyplot as plt


# heatmap of the correlation of numerical features
sns.heatmap(num_corr_matrix)
plt.show()


# dataset is imbalanced
train['model_predicted_fraud'].value_counts()


train_df = train[['interest_rate','transaction_year', 'atm_transfer_in','atm_transfer_out',
  'bank_transfer_in','bank_transfer_out','crypto_in','crypto_out',
  'bank_transfer_in_volume','bank_transfer_out_volume','crypto_in_volume',
  'crypto_out_volume','tenure','from_competitor',
  'churn_due_to_fraud', 'model_predicted_fraud']]


train_df['transaction_year'] = train_df['transaction_year'].astype(int)


corr_metrix = train_df.corr()
sns.heatmap(corr_metrix)
plt.show()


train_df = train[['interest_rate','transaction_year',
  'bank_transfer_in','bank_transfer_out','crypto_in','crypto_out',
  'bank_transfer_in_volume','bank_transfer_out_volume','crypto_in_volume',
  'crypto_out_volume','tenure','model_predicted_fraud']]


train_df['model_predicted_fraud'].value_counts()


corr_metrix = train_df.corr()
sns.heatmap(corr_metrix)
plt.show()


train_df


# spile the data for training and for testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns="model_predicted_fraud"), 
                                                    train_df['model_predicted_fraud'],
                                                    test_size=0.25,
                                                    random_state=42)

X_train.shape, X_test.shape


# # under sampling
# from imblearn.under_sampling import AllKNN
# allknn = AllKNN()
# X_train_smoted, y_train_smoted = allknn.fit_resample(X_train, y_train)
# X_test_smoted, y_test_smoted = allknn.fit_resample(X_test, y_test)


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_smoted, y_train_smoted = rus.fit_resample(X_train, y_train)
X_test_smoted, y_test_smoted = rus.fit_resample(X_test, y_test)


# now imbalencing is handled
pd.Series(y_train_smoted).value_counts()


X_train_smoted


X_train_smoted.isnull().sum().sum(),X_test_smoted.isnull().sum().sum(),y_train_smoted.isnull().sum().sum(),y_test_smoted.isnull().sum().sum() 


num_features = [feature for feature in X_train if X_train[feature].dtypes in [float, int]]
bool_features = [feature for feature in X_train if X_train[feature].dtypes in [bool]]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

# handle categorical features
transformer = ColumnTransformer([
    ("bool fearure encoding", OneHotEncoder(drop="first"), bool_features),
    ("num_scaling", StandardScaler(), num_features)
], remainder="passthrough")

y_hat_encoder = LabelEncoder()


import torch
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


X_train_trf = transformer.fit_transform(X_train_smoted)
X_test_trf = transformer.transform(X_test_smoted)
y_train_encoded = y_hat_encoder.fit_transform(y_train_smoted)
y_test_encoded = y_hat_encoder.fit_transform(y_test_smoted)


X_train_trf.shape


train_df.head()


from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            classification_report,
                            precision_score,
                            recall_score)


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_trf, y_train_encoded)

y_pred = svc.predict(X_test_trf)

print("Accuracy:", accuracy_score(y_pred, y_test_encoded))
print("\nPrecision score:\n", precision_score(y_pred, y_test_encoded))
print("\nRecall Score:\n", recall_score(y_pred, y_test_encoded))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred, y_test_encoded))
print("\n\nClassification Report:", classification_report(y_pred, y_test_encoded))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_trf, y_train_encoded)

y_pred = rf.predict(X_test_trf)

print("Accuracy:", accuracy_score(y_pred, y_test_encoded))
print("\nPrecision score:\n", precision_score(y_pred, y_test_encoded))
print("\nRecall Score:\n", recall_score(y_pred, y_test_encoded))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred, y_test_encoded))
print("\n\nClassification Report:", classification_report(y_pred, y_test_encoded))


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train_trf, y_train_encoded)

y_pred = xgb.predict(X_test_trf)

print("Accuracy:", accuracy_score(y_pred, y_test_encoded))
print("\nPrecision score:\n", precision_score(y_pred, y_test_encoded))
print("\nRecall Score:\n", recall_score(y_pred, y_test_encoded))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred, y_test_encoded))
print("\n\nClassification Report:", classification_report(y_pred, y_test_encoded))


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_trf, y_train_encoded)

y_pred = lr.predict(X_test_trf)

print("Accuracy:", accuracy_score(y_pred, y_test_encoded))
print("\nPrecision score:\n", precision_score(y_pred, y_test_encoded))
print("\nRecall Score:\n", recall_score(y_pred, y_test_encoded))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred, y_test_encoded))
print("\n\nClassification Report:", classification_report(y_pred, y_test_encoded))


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train_trf, y_train_encoded)

y_pred = gbc.predict(X_test_trf)

print("Accuracy:", accuracy_score(y_pred, y_test_encoded))
print("\nPrecision score:\n", precision_score(y_pred, y_test_encoded))
print("\nRecall Score:\n", recall_score(y_pred, y_test_encoded))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred, y_test_encoded))
print("\n\nClassification Report:", classification_report(y_pred, y_test_encoded))


test_data = pd.read_parquet("/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet")
test_data.shape


test_data.head()


test_data.isnull().sum().sum()


test_data.info()


# make data type of date feature in datetime
test_data['date'] = pd.to_datetime(test_data['date'])


# create transaction_year feature
test_data['transaction_year'] = test_data['date'].dt.year


test_df = test_data[list(train_df.columns)]
test_df.head()


# split the data for training and for testing
X = test_df.drop(columns='model_predicted_fraud', axis=1)
y = test_df['model_predicted_fraud']


X_test_trf = transformer.transform(X)


y_pred1 = gbc.predict(X_test_trf)


print("Accuracy:", accuracy_score(y_pred1, y))
print("\nPrecision score:\n", precision_score(y_pred1, y))
print("\nRecall Score:\n", recall_score(y_pred1, y))
print("\nConfusion matrix:\n\n", confusion_matrix(y_pred1, y))
print("\n\nClassification Report:", classification_report(y_pred1, y))


pd.DataFrame(X_train_trf1)


pd.read_csv("/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv")


submission_dict = {"Id": [], "churn": []}


for ids, pred in zip(test_data['Id'], y_pred1):
    submission_dict['Id'].append(ids)
    submission_dict['churn'].append(pred)


pd.DataFrame(submission_dict).to_csv("submission_file.csv")


import joblib

joblib.dump(rf, "neo_churn_prediction.pkl")




