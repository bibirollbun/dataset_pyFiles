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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

print("---------DATA HEAD------------")
print(train_df.head())

print("\n")

print("---------DATA INFO------------")
print(train_df.info())



# Summary statistics

print(train_df.describe())


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

#Visualisations of Numerical features
plt.figure(figsize=(10,6))
sns.histplot(train_df['age'], bins=30, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")

# Distribution of Balance
plt.figure(figsize=(10,6))
sns.histplot(train_df['balance'], bins=30, kde=True)
plt.title("Distribution of Balance")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.savefig("balance_distribution.png")

# plt.close('all')


# Target Variable Analysis
print("Distribution of Target variable 'Y' for analysis")
# Now summarising the counts 
print(train_df['y'].value_counts(normalize=True))

sns.set_style('whitegrid')

plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=train_df)
plt.title("Distribution of Target Variable (y)")
plt.xlabel("Subscribed to Term Deposit(0 = no, 1 = yes)")
plt.ylabel("Count")
plt.savefig("Target_Distribution.png")


# Visualising of Categorical features

sns.set_style('whitegrid')

# --- JOB DISTRIBUTION---
plt.figure(figsize=(6, 4))
sns.countplot(y = 'job', data=train_df, order = train_df['job'].value_counts().index)
plt.title("Distribution of Jobs")
plt.xlabel("Count")
plt.ylabel("Job")
plt.savefig('job_distribution.png')

# Education Distribution
plt.figure(figsize=(6, 4))
sns.countplot(y='education', data=train_df, order= train_df['education'].value_counts().index)
plt.title("Distribution of Education")
plt.xlabel('Count')
plt.ylabel('Subject')
plt.savefig("education_distribution.png")


# Numerical columns

numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = train_df[numerical_cols].corr()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Numerical Features')
plt.savefig('correlation_heatmap.png')


import pandas as pd

# Load the original training and test data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# --- IMPORTANT: Separate the labels FIRST ---
# This ensures the order is preserved before any other operations.
train_y = train_df['y']
train_features = train_df.drop('y', axis=1)

# Store the test IDs for the final submission file
test_ids = test_df['id']

# --- Combine features for consistent encoding ---
combined_df = pd.concat([train_features, test_df], ignore_index=True)

# Identify categorical columns
categorical_cols = combined_df.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False)

# --- Separate back into training and testing sets ---
# The order is naturally preserved from the concat operation.
train_processed = combined_df_encoded.iloc[:len(train_df)]
test_processed = combined_df_encoded.iloc[len(train_df):]

# --- Save the final, aligned files ---
train_processed.to_csv('train_processed.csv', index=False)
test_processed.to_csv('test_processed.csv', index=False)
train_y.to_csv('train_labels.csv', index=False)

print("Preprocessing complete! The following files have been created and are correctly aligned:")
print("- train_processed.csv")
print("- test_processed.csv")
print("- train_labels.csv")


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

X = pd.read_csv('train_processed.csv')
y = pd.read_csv('train_labels.csv').squeeze()
X_test = pd.read_csv('test_processed.csv')

# Dropping ID columns
X_train_ids = X['id']
X_test_ids = X_test['id']
X = X.drop('id', axis=1)
X_test = X_test.drop('id', axis=1)

# ----- Data Split--------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Handle Scale Imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Training the model

lgb_clf = lgb.LGBMClassifier(objective='binary',
                             metric='auc',
                             n_estimators=1000,
                             learning_rate=0.05,
                             num_leaves=31,
                             max_depth=-1,
                             random_state=42,
                             n_jobs=-1,
                             colsample_bytree=0.8,
                             subsample=0.8,
                             reg_alpha=0.1,
                             reg_lambda=0.1,
                             scale_pos_weight=scale_pos_weight
                            )
print(f"Training LightBGM model....")
lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=True)])
y_val_pred_proba = lgb_clf.predict_proba(X_val)[:, 1]

#Calculate AUC Score
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC Score: {auc_score:.5f}")

# we can also check the classification report for further details
y_val_pred_class = (y_val_pred_proba > 0.5).astype(int)
print("Classification Report (threshold=0.5):")
print(classification_report(y_val, y_val_pred_class))

# ---- Make Predictions----
test_predictions = lgb_clf.predict_proba(X_test)[:, 1]

# Create Submission CSV File
submission_df = pd.DataFrame({'id': X_test_ids, 'y' : test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Submission File Created successfully")
print(submission_df.head())


import pandas as pd

# Load your final processed data and labels
X_check = pd.read_csv('train_processed.csv')
y_check = pd.read_csv('train_labels.csv').squeeze()

# Combine them into one dataframe for checking
check_df = pd.concat([X_check, y_check], axis=1)

# Check the correlation between 'duration' and 'y'
correlation = check_df['duration'].corr(check_df['y'])

print(f"The correlation between 'duration' and 'y' is: {correlation:.4f}")

