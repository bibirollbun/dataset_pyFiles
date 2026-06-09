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


test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


test=test_df.ffill()
test.shape


 #Feature Engineering
test['temp_range'] = test['maxtemp'] -test['mintemp']
test['dewpoint_depression'] = test['temparature'] - test['dewpoint']
test['day_of_week'] = pd.to_datetime(test['day']).dt.dayofweek
test['month'] = pd.to_datetime(test['day']).dt.month



test.head(3)


train_df.shape






train= train_df.drop(train_df.index[730:])


train.head(2)


train.shape


# Feature Engineering
#test['temp_range'] = test['maxtemp'] -test['mintemp']
#test['dewpoint_depression'] = test['temparature'] - test['dewpoint']
#test['day_of_week'] = pd.to_datetime(test['day']).dt.dayofweek
#test['month'] = pd.to_datetime(test['day']).dt.month


#Feature Engineering
train['temp_range'] = train['maxtemp'] -train['mintemp']
train['dewpoint_depression'] = train['temparature'] - train['dewpoint']
train['day_of_week'] = pd.to_datetime(train['day']).dt.dayofweek
train['month'] = pd.to_datetime(train['day']).dt.month








train.head(3)


x=train.drop(['rainfall'],axis=1)
y=train.rainfall


print(x)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier




x_train, x_val, \
    y_train, y_val = train_test_split(x,
                                      y,
                                      test_size=0.2,
                                      stratify=y,
                                      random_state=2)



# Normalizing the features for stable and fast training.
scaler = StandardScaler()
x= scaler.fit_transform(x)
x_val = scaler.transform(x_val)


from sklearn.metrics import roc_auc_score

# List of models
models = [
    LogisticRegression(solver='liblinear', max_iter=1000),
    SVC(kernel='rbf', probability=True),
    CatBoostClassifier(random_state=42, iterations=200, learning_rate=0.1, depth=6, verbose=0)
]
# Variables to store the best model and its performance
best_model = None
best_roc_auc = -1

# Train and evaluate each model
for i in range(len(models)):
    models[i].fit(x, y)

    print(f'{models[i]} : ')

    # Training predictions
    train_preds = models[i].predict_proba(x)[:, 1]
    train_roc_auc = roc_auc_score(y, train_preds)
    print('ROC AUC Training Accuracy : ', train_roc_auc)
   # Validation predictions
    val_preds = models[i].predict_proba(x_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, val_preds)
    print('ROC AUC Validation Accuracy : ', val_roc_auc)
    print()

    # Check if this model is the best so far
    if val_roc_auc > best_roc_auc:
        best_roc_auc = val_roc_auc
        best_model = models[i]

# Print the best model
print(f'Best Model: {best_model}')
print(f'Best ROC AUC on Validation Set: {best_roc_auc}')

# Use the best model to make predictions on the test set
test_preds = best_model.predict_proba(test)[:, 1]

# Prepare the submission file (if required)
submission = pd.DataFrame({
    'id': test['id'],  # Replace with your actual test IDs
    'rainfall': test_preds
})

# Save the submission file to a CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")


submission.head()

