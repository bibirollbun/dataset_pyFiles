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


# Load the training data
df_traindata = pd.read_csv('breast-cancer-classification-fall-2025/train.csv')
df_traindata.head()


# Split the data into features (X) and target variable (y)
# Hint: training model doesn't require label and id
X =
y =


X.head()


y.head()


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val =


from sklearn.preprocessing import StandardScaler

scaler =
X_train_scaled =
X_val_scaled =


from sklearn.linear_model import Perceptron

pct_model =


from sklearn.linear_model import LogisticRegression

log_model =


from sklearn.svm import SVC

svm_model =


from sklearn.tree import DecisionTreeClassifier

dt_model =


from sklearn.ensemble import RandomForestClassifier

rf_model =


# Fit the pct_model - Example: pct_model.fit()



# Fit the log_model



# Fit the svm_model



# Fit the dt_model



# Fit the rf_model



# Perceptron predictions
pct_pred =

# Logistic Regression predictions
log_pred =

# SVM predictions
svm_pred =

# Decision Tree predictions
dt_pred =

# Random Forest predictions
rf_pred =


from sklearn.metrics import accuracy_score

# Perceptron accuracy
pct_accuracy =

# Logistic Regression accuracy
log_accuracy =

# SVM accuracy
svm_accuracy =

# Decision Tree accuracy
dt_accuracy =

# Random Forest accuracy
rf_accuracy =


print(f"{'Perceptron':<20} Accuracy: {pct_accuracy * 100:>6.2f}%")
print(f"{'Logistic Regression':<20} Accuracy: {log_accuracy * 100:>6.2f}%")
print(f"{'SVM':<20} Accuracy: {svm_accuracy * 100:>6.2f}%")
print(f"{'Decision Tree':<20} Accuracy: {dt_accuracy * 100:>6.2f}%")
print(f"{'Random Forest':<20} Accuracy: {rf_accuracy * 100:>6.2f}%")


# Load the test data
df_test = pd.read_csv('breast-cancer-classification-fall-2025/test.csv')

# Preprocess the test data as test set
test_data =
test_data.head()


test_scaled =


# Example - test_pred = pct_model.predict()
test_pred =


df_submission = pd.DataFrame({'id': df_test['id'], 'label': test_pred})
df_submission.to_csv('bcc_submission.csv', index=False)
df_submission.head()

