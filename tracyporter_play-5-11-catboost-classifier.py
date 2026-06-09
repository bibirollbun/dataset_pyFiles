import numpy as np
import pandas as pd
import os

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submit = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


pd.set_option('display.max_columns', None)


train


train.isna().sum().sum()


train.info()


test


test.isna().sum().sum()


test.info()


submit


train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

train.shape, test.shape


# Create the histogram
plt.hist(train['loan_paid_back'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' Loan Paid Back')
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


# Create subplots to display count plots for categorical columns in 'cat_cols'
# Subplots are arranged in a 4x2 grid, and 
# each subplot shows the count distribution of a categorical column.
plt.subplots(figsize=(15, 15))
for i, col in enumerate(cat_cols):
    plt.subplot(4, 2, i+1)
    sns.countplot(data=train, x=col)
# Ensures proper spacing between subplots
plt.tight_layout() 
# Display the subplots
plt.show()


int_cols = []

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'int' and train[col].nunique() < 10:    
        # Add the column to the list
        int_cols.append(col)     
int_cols


#plt.subplots(figsize=(15, 15))
#for i, col in enumerate(int_cols):
#    plt.subplot(4, 2, i+1)
#    sns.countplot(data=train, x=col)
# Ensures proper spacing between subplots
#plt.tight_layout() 
# Display the subplots
#plt.show()


# Create subplots to display distribution plots for numeric columns in 'num_cols'
num_cols = []

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'float':  
        # Add the column to the list
        num_cols.append(col)     
num_cols


#plt.subplots(figsize=(10, 5))
##for i, col in enumerate(num_cols):
#    plt.subplot(1, 2, i+1)
#    sns.histplot(train[col],kde=True)
#plt.tight_layout()  
#plt.show()


train["gender"] = train["gender"].replace({"Other": "other_gender"})
test["gender"] = test["gender"].replace({"Other": "other_gender"})

train["education_level"] = train["education_level"].replace({"Other": "other_education"})
test["education_level"] = test["education_level"].replace({"Other": "other_education"})

train["loan_purpose"] = train["loan_purpose"].replace({"Other": "other_loan"})
test["loan_purpose"] = test["loan_purpose"].replace({"Other": "other_loan"})


for col in cat_cols:
    temp = pd.get_dummies(train[col]).astype('int')
    train = pd.concat([train, temp], axis=1)

train.drop(cat_cols, axis=1, inplace=True)
train.shape


train.columns


for col in cat_cols:
    temp = pd.get_dummies(test[col]).astype('int')
    test = pd.concat([test, temp], axis=1)

test.drop(cat_cols, axis=1, inplace=True)
test.shape



# Check if value exists
value_to_find = "C5"
exists = (train == value_to_find).any().any()
print(f"Value {value_to_find} exists: {exists}")

# Find positions of the value
positions = train[train == value_to_find].stack().index.tolist()
#print(f"Positions of {value_to_find}: {positions}")


y = train.pop('loan_paid_back')
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
plt.title(' Probability of loan repayment')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


submit['loan_paid_back'] = pred_proba
submit.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

