import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


pd.set_option('display.max_columns', None)  # Show all columns


train


train.info()


train.isna().sum().sum()


for col in train:
    if train[col].dtype == 'bool':
        train[col] = train[col].replace({True: 1, False: 0})
train.info()


test


test.info()


for col in test:
    if test[col].dtype == 'bool':
        test[col] = test[col].replace({True: 1, False: 0})
test.info()


test.isna().sum().sum()


submission


train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

test.shape, train.shape


# Create the histogram
plt.hist(train['accident_risk'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' Accident Risk')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# Identify and collect categorical columns with less than 10 unique values
# Initialize an empty list to store categorical columns
cat_cols = [] 

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'object' and train[col].nunique() < 10:  
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


plt.subplots(figsize=(15, 15))
for i, col in enumerate(int_cols):
    plt.subplot(4, 2, i+1)
    sns.countplot(data=train, x=col)
# Ensures proper spacing between subplots
plt.tight_layout() 
# Display the subplots
plt.show()


# Create subplots to display distribution plots for numeric columns in 'num_cols'
num_cols = []

# Iterate through DataFrame columns
for col in train.columns: 
    if train[col].dtype == 'float':  
        # Add the column to the list
        num_cols.append(col)     
num_cols


plt.subplots(figsize=(10, 5))
for i, col in enumerate(num_cols):
    plt.subplot(1, 2, i+1)
    sns.histplot(train[col],kde=True)
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


y = train.pop('accident_risk')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.15)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape


# Initialize the CatBoostRegressor with RMSE as the loss function
model = CatBoostRegressor(loss_function='RMSE')

# Fit the model on the training data with verbose logging every 100 iterations
model.fit(X_train, y_train, verbose=100)


# Generate predictions on the training and validation sets using the trained 'model'
y_pred = model.predict(X_val)

# Calculate and print the Root Mean Squared Error (RMSE) for training and validation sets
print("Validation RMSE: ", np.sqrt(mse(y_val, y_pred)))



# Plot the results
fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


pred = model.predict(X_test)
pred


# Create the histogram
plt.hist(pred, bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title(' Accident Risk - Predictions')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


submission['accident_risk'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

