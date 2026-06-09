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
import matplotlib.pyplot as plt
import seaborn as sns

!pip install -U autogluon --quiet
from autogluon.tabular import TabularPredictor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original = pd.read_csv("/kaggle/input/personality-data/personality_dataset.csv")
original_data = pd.read_csv("/kaggle/input/personality-dataset/personality_datasert.csv")


print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("original :", original.shape)
print("original_data :", original_data.shape)
print("sample_submission :", sample_submission.shape)


train_data.head()


train_data['Personality'].value_counts()


train_data.info()


# Calculate missing values
missing_values = train_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


# Calculate missing values
missing_values = original.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


test_data.isna().sum().sort_values(ascending=False)


# Calculate missing values
missing_values = test_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


# Categorical columns to plot
cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

# Set up 2x2 grid for subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cat_cols):
    train_data[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()


# Categorical columns to plot
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Set up 2x2 grid for subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 4))
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cat_cols):
    test_data[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()


train_data = train_data.drop("id", axis=1)
#train_data = pd.concat([train_data, original], ignore_index=True)
#train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data = train_data.drop_duplicates()
print("shape of the data :",train_data.shape)
train_data.head()


num_cols = list(train_data.select_dtypes(exclude=['object']).columns)
cat_cols = list(train_data.select_dtypes(include=['object']).columns.difference(['Personality']))

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


len(cat_cols_test),len(cat_cols)


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object', 'category']).columns] = test_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col_name in cat_cols:
    train_data[col_name]=labelencoder.fit_transform(train_data[col_name]).astype(int)
        
#for col_name in cat_cols_test:
    test_data[col_name]=labelencoder.transform(test_data[col_name]).astype(int)

target_le = LabelEncoder()
train_data['Personality'] = target_le.fit_transform(train_data['Personality'])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


# Define your target column
target = 'Personality'  # change if your target is named differently

# Use entire dataset for training
test_data = test_data.drop(['id'],axis=1)


predictor = TabularPredictor(label=target, eval_metric='accuracy').fit(train_data)



predictor.leaderboard(silent=True)


performance = predictor.evaluate(train_data)
print("Train accuracy:", performance['accuracy'])


# Predict target
preds = predictor.predict(test_data)
preds = target_le.inverse_transform(preds)

submission = pd.DataFrame({
    "id": sample_submission["id"],  # if your test data has an ID column
    "Personality": preds  # or final_preds if labels are already strings
})

submission.to_csv("submission.csv", index=False)
submission.head()

