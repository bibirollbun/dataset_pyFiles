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


## Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("Imported libraires successfully")


## Loading the data.

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

#print(train_df.sample(2))

print("Data Loading successful")


## Encoding the target feature.

le = LabelEncoder()
train_df['encoded_Personality'] = le.fit_transform(train_df['Personality'])

print(train_df.head(3))


## Preparing the features for training and testing data

X = train_df.drop(columns = ['id', 'Personality', 'encoded_Personality'])
y = train_df['encoded_Personality']
X_test = test_df.drop(columns=['id'])


print(X.columns[X.columns.duplicated()])
print(X_test.columns[X_test.columns.duplicated()])



## Encoding the input features 
## We will be using OrdinalEncoder here.

##Combining the training and testing data (X and X_test)

combined = pd.concat([X, X_test], axis = 0)
cat_cols = combined.select_dtypes(include = "object").columns.tolist()
oe = OrdinalEncoder()
combined[cat_cols] = oe.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop = True)
X_test = combined.iloc[len(X):].reset_index(drop = True)


print(X.columns)
print(X_test.columns)
print(y.sample(4))


# 6. Setting up XGBoost
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


# 7. Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=10, verbose_eval=False)
    
    oof_preds[val_idx] = model.predict(dval) > 0.5
    test_preds += model.predict(dtest) / skf.n_splits


# 8. Evaluating the model.
cv_acc = accuracy_score(y, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

# 9. Create submission
final_preds = (test_preds > 0.5).astype(int)
submission_df["Personality"] = le.inverse_transform(final_preds)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()




