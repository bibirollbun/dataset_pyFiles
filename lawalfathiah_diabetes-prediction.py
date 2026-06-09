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


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df


df.shape


df.info()


import matplotlib.pyplot as plt
import seaborn as sns

for col in df.columns:
    if df[col].dtype != 'object':
        sns.displot(df[col])


df.columns


df.groupby("diagnosed_diabetes")['gender'].value_counts(normalize = True)


df.describe()


df.corr(numeric_only = True)


plt.figure(figsize = (12,8))
sns.heatmap(df.corr(numeric_only = True))
plt.show


df.groupby('diagnosed_diabetes')["ethnicity"].value_counts()


df.columns


sns.scatterplot(df, x ='screen_time_hours_per_day', y = 'diagnosed_diabetes')


df.head()


df.groupby('income_level')['diagnosed_diabetes'].value_counts(normalize = True)


df.groupby('gender')['diagnosed_diabetes'].value_counts(normalize = True)


df


df.info()


df['diagnosed_diabetes'] = df['diagnosed_diabetes'].astype('int')


df['diagnosed_diabetes'].value_counts()


df[['gender', 'smoking_status']]


def drop_columns(temp_df):
    temp_df.drop(columns = ['ethnicity', 'education_level', 'income_level', 'employment_status', 'screen_time_hours_per_day','alcohol_consumption_per_week','sleep_hours_per_day','diastolic_bp', 'heart_rate', 'hypertension_history', 'cardiovascular_history', 'gender','smoking_status'], inplace = True)
    temp_df.set_index('id', inplace = True)
drop_columns(df)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# Create the Cholesterol Ratio: HDL Cholesterol divided by LDL Cholesterol
# We use .assign() to create a new column without modifying the original DataFrame in place

def new_column(temp_df):
    temp_df["cholesterol_ratio"] = temp_df['hdl_cholesterol'] / temp_df['ldl_cholesterol']

# Confirm the new column is created
new_column(df)
print(df[['hdl_cholesterol', 'ldl_cholesterol', 'cholesterol_ratio']].head())


# Calculate correlation with the target variable ('diagnosed_diabetes')
target_correlation = df[['hdl_cholesterol', 'ldl_cholesterol', 'cholesterol_ratio']].corrwith(df['diagnosed_diabetes'])

print("\nCorrelation with Diagnosed Diabetes:")
print(target_correlation.sort_values(key=abs, ascending=False))


df


X = df.drop(columns = 'diagnosed_diabetes')
y = df['diagnosed_diabetes']


scaler = StandardScaler()
X = scaler.fit_transform(X)


### Initially tried oversampling but decided to remove it

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)


from catboost import CatBoostClassifier, Pool
cb = CatBoostClassifier()
model = cb.fit(X_train, y_train)


model.score(X_train, y_train)


model.score(X_valid, y_valid)


from sklearn.metrics import f1_score, recall_score, confusion_matrix

# Get predictions using the best iteration
y_pred = model.predict(X_valid)

# Calculate metrics
f1 = f1_score(y_valid, y_pred, pos_label=0)  # Assuming 0 is the 'No' (minority) label
recall = recall_score(y_valid, y_pred, pos_label=0)

print(f"\nMinority Class (No) F1-Score: {f1:.4f}")
print(f"Minority Class (No) Recall: {recall:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred))
print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


test_df


drop_columns(test_df)
new_column(test_df)


X_test = test_df


y_test = model.predict(X_test)


# Get predictions using the best iteration
y_test = model.predict(X_test)


test_df['diagnosed_diabetes'] = y_test


test_df


test_df.reset_index(inplace = True)
test_df


output_df = test_df[['id', 'diagnosed_diabetes']]
output_df['diagnosed_diabetes'] = output_df['diagnosed_diabetes'].astype(int)

output_df.to_csv('submission_file.csv', index=False)


print("Successfully created 'submission_file.csv' with 'id' and 'diagnosed_diabetes'.")

























































































































































































































































































































































































































































































