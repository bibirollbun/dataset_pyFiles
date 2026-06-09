import pandas as pd
import numpy as np
import os

from scipy.stats import ks_2samp #nonparametric test

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


pd.set_option('display.max_columns', None)


train


train.isna().sum().sum()


test


test.isna().sum().sum()


submission


#del_cols = []

#for col in test:
#    stat, pv = ks_2samp(train[col], test[col])
#    if pv < 0.10:
#        del_cols.append(col)

#print(del_cols)

#train = train.drop(del_cols, axis = 1)
#test = test.drop(del_cols, axis = 1)

#train.shape, test.shape


#train


# Create the histogram
plt.hist(train['diagnosed_diabetes'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' Diagnosis of Diabetes')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# Identify and collect categorical columns with less than 10 unique values
# Initialize an empty list to store categorical columns
cat_cols = [] 

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'object':  
        # Add the column to the list
        cat_cols.append(col)     
cat_cols


# Create subplots in a 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through categorical columns and plot on each axis
for i, col in enumerate(cat_cols):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f"Count plot of {col}")

# Remove any unused axes (if cat_cols < 8)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


for col in test.columns:   # iterate over column names
    if test[col].dtype == object:   # check if column is object type
        print(col, test[col].unique())   # call .unique() on the Series


#replace 'other' in categorical columns
train["gender"] = train["gender"].replace({"Other": "other_gender"})
test["gender"] = test["gender"].replace({"Other": "other_gender"})

train["ethnicity"] = train["ethnicity"].replace({"Other": "other_ethnic"})
test["ethnicity"] = test["ethnicity"].replace({"Other": "other_ethnic"})


int_cols = []

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'int' and train[col].nunique() < 10:    
        # Add the column to the list
        int_cols.append(col)     
int_cols


# Create subplots in a 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through categorical columns and plot on each axis
for i, col in enumerate(int_cols):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f"Count plot of {col}")

# Remove any unused axes (if cat_cols < 8)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# Create subplots to display distribution plots for numeric columns in 'num_cols'
num_cols = []

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'float':  
        # Add the column to the list
        num_cols.append(col)     
num_cols


# Create subplots in a 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through categorical columns and plot on each axis
for i, col in enumerate(int_cols):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f"Count plot of {col}")

# Remove any unused axes (if cat_cols < 8)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


for col in cat_cols:
    temp = pd.get_dummies(train[col]).astype('int')
    train = pd.concat([train, temp], axis=1)

train.drop(cat_cols, axis=1, inplace=True)
train.shape


for col in cat_cols:
    temp = pd.get_dummies(test[col]).astype('int')
    test = pd.concat([test, temp], axis=1)

test.drop(cat_cols, axis=1, inplace=True)
test.shape


y = train.pop('diagnosed_diabetes')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.15)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape


# Initialize CatBoostClassifier
model = CatBoostClassifier(loss_function='CrossEntropy',iterations=1000, learning_rate=0.1, depth=6, verbose=100)

# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_val)
y_pred


# Create the histogram
plt.hist(y_pred, bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' y-pred')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")


pred = model.predict(X_test)
pred


pred_proba = model.predict_proba(X_test,
              ntree_start=0,
              ntree_end=0,
              thread_count=-1,
              verbose=None)
pred_proba = pred_proba[:,1]
pred_proba


# Create the histogram
plt.hist(pred_proba, bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' Probability of diabetes diagnosis')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


submission['diagnosed_diabetes'] = pred_proba
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

