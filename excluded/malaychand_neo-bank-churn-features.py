import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import sys
import os

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import random
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt



import pandas as pd
import glob

# Define paths
BASEPATH = '/kaggle/input/neo-bank-non-sub-churn-prediction/'
train_path = BASEPATH + 'train_*.parquet'
test_path = BASEPATH + 'test.parquet'

# Read all train parquet files matching the pattern
train_files = glob.glob(train_path)
train_df = pd.concat([pd.read_parquet(file) for file in train_files], ignore_index=True)

# Read the test parquet file
test_df = pd.read_parquet(test_path)


train_df.describe().T


len(test_df)


# no of unique customer in training data 
len(train_df['customer_id'].unique())


# no of unique customer in Testings data
len(test_df['customer_id'].unique())


# Any overlaping between ID
train_ids = train_df['Id'].unique()
test_ids = test_df['Id'].unique()
common_ids = np.intersect1d(train_ids, test_ids)
len(common_ids)


# Any overlaping between customer ID
train_Cids = train_df['customer_id'].unique()
test_Cids = test_df['customer_id'].unique()
common_ids = np.intersect1d(train_Cids, test_Cids)
len(common_ids)


# 
len(train_df[train_df['churn_due_to_fraud'] == True]['customer_id'].unique())


# 
len(train_df[train_df['churn_due_to_fraud'] == True]['customer_id'].unique())


len(train_df[train_df['model_predicted_fraud'] == True]['customer_id'].unique())


len(test_df[test_df['churn_due_to_fraud'] == True]['customer_id'].unique())


len(test_df[test_df['model_predicted_fraud'] == True]['customer_id'].unique())






train_churn_ids = train_df[train_df['churn_due_to_fraud'] == True]['customer_id'].unique()
test_churn_ids = test_df[test_df['churn_due_to_fraud'] == True]['customer_id'].unique()

# Find common churned customer IDs
common_churn_ids = np.intersect1d(train_churn_ids, test_churn_ids)
len(common_churn_ids)


# Usage is an extra columns in test data 
test_df['Usage'].value_counts()


# It not significant and not present in train data we can drop it to drop
test_df[(test_df['Usage'] == 'Public') & (test_df['churn_due_to_fraud']==True)]['Id'].unique().shape


test_df = test_df.drop(columns=['Usage'])


# As there is no overlap between the IDs in the test and train datasets, 
# It may combine them for feature engineering 
test_df['rowtype'] = 'test'
train_df['rowtype'] = 'train'


# add both the test and train datasets
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


len(df)


 df.groupby('rowtype').size()


df[df['churn_due_to_fraud']==True]['customer_id'].unique().shape


df.columns


df[df['churn_due_to_fraud']==True]['customer_id'].unique().shape


df.groupby('rowtype').size()





# created age feature that extrating customer score ,customer curent  age by date of birth and count event 
def pd_transform1(df):
    # creating diffrent columns for customer score on diffrent median
    csat_expanded = pd.json_normalize(df['csat_scores'])
    df = pd.concat([df, csat_expanded], axis=1)
    df[['appointment', 'email', 'phone', 'whatsapp']] = df[['appointment', 'email', 'phone', 'whatsapp']].fillna(0)
    df['appointment'] = df['appointment'].fillna(0)
    df['email'] = df['email'].fillna(0)
    df['phone'] = df['phone'].fillna(0)
    df['whatsapp'] = df['whatsapp'].fillna(0)
    df = df.drop(columns=['touchpoints','csat_scores'])
    # Finding current age using given date of birth
    now = pd.Timestamp('now')
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%y%m%d')
    df['date_of_birth'] = df['date_of_birth'].where(df['date_of_birth'] < now, df['date_of_birth'] - np.timedelta64(int(100 * 365.25), 'D'))
    df['age'] = ((now - df['date_of_birth']).dt.days / 365.25).astype(int)
    # Finding number of event of update
    df['event_count'] = df.groupby('customer_id')['customer_id'].transform('count')
    return df


def pd_transform2(df): 
    df['days_between'] = (df['date'] - df['date'].shift(1)).dt.days  
    # Adding previous window operations (like previous 10 days, 450 days, etc.)
    df['prior_mean_days_between'] = df.groupby('customer_id')['days_between'].transform('mean')
    df['prior_max_days_between'] = df.groupby('customer_id')['days_between'].transform('max')
    df['prior_mean_bank_transfer_in'] = df.groupby('customer_id')['bank_transfer_in'].transform('mean')
    df['prior_mean_bank_transfer_out'] = df.groupby('customer_id')['bank_transfer_out'].transform('mean')
    df['prior_mean_crypto_in'] = df.groupby('customer_id')['crypto_in'].transform('mean')
    df['prior_mean_crypto_out'] = df.groupby('customer_id')['crypto_out'].transform('mean')
    df['prior_mean_bank_transfer_in_volume'] = df.groupby('customer_id')['bank_transfer_in_volume'].transform('mean')
    df['prior_mean_bank_transfer_out_volume'] = df.groupby('customer_id')['bank_transfer_out_volume'].transform('mean')
    df['prior_mean_crypto_in_volume'] = df.groupby('customer_id')['crypto_in_volume'].transform('mean')
    df['prior_mean_crypto_out_volume'] = df.groupby('customer_id')['crypto_out_volume'].transform('mean')
    #  Last 10 days #(using rolling window with a period of 10 days)
    df['prior10_count'] = df.groupby('customer_id')['date'].transform(lambda x: x.rolling(window=10, min_periods=1).count())
    df['prior10_mean_days_between'] = df.groupby('customer_id')['days_between'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_max_days_between'] = df.groupby('customer_id')['days_between'].transform(lambda x: x.rolling(window=10, min_periods=1).max())
    df['prior10_mean_bank_transfer_in'] = df.groupby('customer_id')['bank_transfer_in'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_bank_transfer_out'] = df.groupby('customer_id')['bank_transfer_out'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_crypto_in'] = df.groupby('customer_id')['crypto_in'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_crypto_out'] = df.groupby('customer_id')['crypto_out'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_bank_transfer_in_volume'] = df.groupby('customer_id')['bank_transfer_in_volume'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_bank_transfer_out_volume'] = df.groupby('customer_id')['bank_transfer_out_volume'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_crypto_in_volume'] = df.groupby('customer_id')['crypto_in_volume'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['prior10_mean_crypto_out_volume'] = df.groupby('customer_id')['crypto_out_volume'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    # 3. Last 450 days (using a rolling window with a period of 450 days)
    df['prior450_count'] = df.groupby('customer_id')['date'].transform(lambda x: x.rolling(window=450, min_periods=1).count())
    df['prior450_mean_days_between'] = df.groupby('customer_id')['days_between'].transform(lambda x: x.rolling(window=450, min_periods=1).mean())
    df['prior450_max_days_between'] = df.groupby('customer_id')['days_between'].transform(lambda x: x.rolling(window=450, min_periods=1).max())
    df['prior450_mean_bank_transfer_in_volume'] = df.groupby('customer_id')['bank_transfer_in_volume'].transform(lambda x: x.rolling(window=450, min_periods=1).mean())
    df['prior450_mean_bank_transfer_out_volume'] = df.groupby('customer_id')['bank_transfer_out_volume'].transform(lambda x: x.rolling(window=450, min_periods=1).mean())
    df['prior450_mean_crypto_in_volume'] = df.groupby('customer_id')['crypto_in_volume'].transform(lambda x: x.rolling(window=450, min_periods=1).mean())
    df['prior450_mean_crypto_out_volume'] = df.groupby('customer_id')['crypto_out_volume'].transform(lambda x: x.rolling(window=450, min_periods=1).mean())
    return df        


# # 4. This week's volume (for the country, last 7 days rolling window)
#     df['this_week_volume'] = df.groupby('country')['bank_transfer_in_volume'].transform(lambda x: x.rolling(window=7, min_periods=1).sum() - df.groupby('country')['bank_transfer_out_volume'].transform(lambda x: x.rolling(window=7, min_periods=1).sum()) + df.groupby('country')['crypto_in_volume'].transform(lambda x: x.rolling(window=7, min_periods=1).sum()) - df.groupby('country')['crypto_out_volume'].transform(lambda x: x.rolling(window=7, min_periods=1).sum()))


def pd_transform3(df):
    df['date'] = pd.to_datetime(df['date'])
    df['days_from_today'] = (datetime.datetime.now() - df['date']).dt.days
    df['year'] = df['date'].dt.year
    # coverting into numerical 
    df['model_predicted_fraud'] = df['model_predicted_fraud'].astype(int)
    df['from_competitor'] = df['from_competitor'].astype(int)
    df['churn_due_to_fraud'] = df['churn_due_to_fraud'].astype(int)
    # Calculate the day gap (difference in days between consecutive events for each customer)
    df['day_gap'] = df.groupby('customer_id')['date'].diff().dt.days
    df['day_gap'] = df['day_gap'].fillna(0).astype(int)
    return df


combined_data = pd.concat([train_df, test_df], ignore_index=True)
combined_data = combined_data.sort_values(by='Id').reset_index(drop=True)
# train_df=pd_transform1(train_df)
# train_df=pd_transform2(train_df)
# train_df=pd_transform3(train_df)
# train_data = train_df


print (5)


combined_data=pd_transform1(combined_data)
combined_data=pd_transform2(combined_data)
combined_data=pd_transform3(combined_data)
combined = combined_data


# Define target and feature columns
TARGET_NAME = 'churn_due_to_fraud'
# 'customer_id', -as it provide good importent feature 
features =['Id','customer_id', 'interest_rate', 'atm_transfer_in', 'atm_transfer_out',
       'bank_transfer_in', 'bank_transfer_out', 'complaints', 'tenure',
       'from_competitor', 'model_predicted_fraud',
       'appointment', 'email', 'phone', 'whatsapp', 'age', 'event_count',
       'days_between', 'prior_mean_days_between', 'prior_max_days_between',
       'prior_mean_bank_transfer_in', 'prior_mean_bank_transfer_out',
       'prior10_count', 'prior10_mean_days_between',
       'prior10_max_days_between', 'prior10_mean_bank_transfer_in',
       'prior10_mean_bank_transfer_out', 'prior450_count',
       'prior450_max_days_between', 'prior450_mean_bank_transfer_in_volume',
       'days_from_today', 'day_gap']


# General imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, confusion_matrix, roc_curve



X_train, X_val, y_train, y_val = train_test_split(combined_data[features], combined_data[TARGET_NAME],test_size=0.15, random_state=42,stratify=combined_data[TARGET_NAME])
# Initialize and train CatBoostClassifier
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Logloss',
    random_seed=42,
    verbose=0
)


cat.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,  
    verbose=0 
)


# Retrieve all rows from combined_data where 'Id' is in test_df['Id']
filtered_data = combined_data[combined_data['Id'].isin(test_df['Id'])]


len(filtered_data)




# Extract test features and target from combined_data for evaluation
X_test = filtered_data[features]
y_test = filtered_data[TARGET_NAME]

# Predict and evaluate on test set
test_preds = cat.predict(X_test)
test_recall = recall_score(y_test, test_preds)
print(f"Validation Recall: {test_recall:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_preds))

# ROC Curve and Optimal Threshold
fpr, tpr, roc_thresholds = roc_curve(y_test, test_preds)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_roc = roc_thresholds[optimal_idx]
print(f"Optimal ROC Threshold: {optimal_threshold_roc:.4f}")

# Apply optimal threshold for predictions
y_pred = (test_preds >= optimal_threshold_roc).astype(int)

# Prepare submission DataFrame
submission = filtered_data[['Id']].copy()
submission['churn'] = y_pred 

# Save to CSV
submission.to_csv("CatBoost_submission.csv", index=False, header=True)

# Print the count of predicted 'churn' (1's)
print(len(submission[submission['churn'] == 1]))



# Get feature importance scores
feature_importance = cat.get_feature_importance()
feature_names = X_train.columns  # Get feature names

# Convert to DataFrame for better readability
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
print(feature_importance_df)


feature_importance




