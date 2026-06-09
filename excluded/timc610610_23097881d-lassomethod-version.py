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


import kagglehub

# Download latest version
path = kagglehub.dataset_download("mdmub0587/older-dataset-for-dont-overfit-ii-challenge")

print("Path to dataset files:", path)


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score


# reading data and understanding data
train_data = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv')
x_train = train_data.drop (columns=['id', 'target'])
y_train = train_data['target']
x_test = test_data.drop(columns=['id'])
print (x_train.shape)


# scaling the data
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# Lasso method for a dataset with 300 features
lasso_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.05,
    random_state=42,
    max_iter=2000
)
#training:
lasso_model.fit(x_scaled, y_train)
# check numbers of selected features:
selected_features = np.sum(lasso_model.coef_[0] != 0)
print("number of selected features: "+ str(selected_features))


test_predictions = lasso_model.predict_proba(x_test_scaled)[:, 1]
submission = pd.DataFrame({
    'id': test_data['id'],
    'target': test_predictions
})

submission.to_csv('lasso_submission.csv', index=False)
print("Lasso submission file saved as 'lasso_submission.csv'")


print(f"Prediction statistics:")
print(f"Min probability: {test_predictions.min():.4f}")
print(f"Max probability: {test_predictions.max():.4f}")
print(f"Mean probability: {test_predictions.mean():.4f}")
print(f"Number of predictions > 0.5: {np.sum(test_predictions > 0.5)}")
cv_scores = cross_val_score(
    lasso_model, 
    x_scaled, 
    y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc'
)
#print(f"AUC score: {cv_scores}")
print(f"mean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

