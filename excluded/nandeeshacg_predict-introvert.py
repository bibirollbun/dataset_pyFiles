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


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import os


#load the required data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


#name the feature
X_train = df_train.drop(['id', 'Personality'], axis=1)
y_train = df_train['Personality']
X_test = df_test.drop(['id'], axis=1)

# Fill missing values as string
X_train = X_train.fillna("Missing")
X_test = X_test.fillna("Missing")

# Convert categorical-like columns to string (based on low unique count)
for col in X_train.columns:
    if X_train[col].nunique() < 20:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

# List of categorical features for CatBoost
categorical_features = X_train.columns[X_train.dtypes == 'object'].tolist()


#Hyperparameter Tunig
cbc = CatBoostClassifier(verbose=0)

params = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [300, 500]
}

grid = GridSearchCV(cbc, param_grid=params, scoring='accuracy', cv=3)
grid.fit(X_train, y_train, cat_features=categorical_features)

print(" Best Parameters Found:", grid.best_params_)

#Split for Validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

#Train Final Model with Best Parameters
model = CatBoostClassifier(
    **grid.best_params_,
    cat_features=categorical_features,
    verbose=100,
    random_seed=42
)

model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)



#Evaluate Accuracy
y_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)
print(f" Validation Accuracy: {val_acc:.4f}")
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

#Predict on Test Set & Save Submission
test_preds = model.predict(X_test)
submission['Personality'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(" Submission file created successfully!")

