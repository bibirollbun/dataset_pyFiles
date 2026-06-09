# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train.head(10)


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test.head()


submission= pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


sns.histplot(train['accident_risk'],kde = True,bins=30)
plt.title('Distribution of Accident Risk')
plt.show()


# remove id column as it is not useful for prediction
train =train.drop(columns=['id'])
test = test.drop(columns=['id'])


# Categorical data and boolean data
bool_cols=train.select_dtypes(include='bool').columns
cat_cols= train.select_dtypes(include='object').columns


# handle boolean data attributes
for col in bool_cols:
    train[col]= train[col].astype(int)
    test[col] = test[col].astype(int)


# handle categorical data
for col in cat_cols:
    le=LabelEncoder()
    train[col]=le.fit_transform(train[col])
    test[col]=le.transform(test[col])


train.head()


#split the data into features and target variable
X= train.drop(columns=['accident_risk'])
y= train['accident_risk']


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)


xgb =XGBRegressor(objective='reg:squarederror',n_jobs=-1,random_state=42)


param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5]
}


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,              # number of random combinations to try
    scoring="neg_root_mean_squared_error",
    cv=3,                   # 3-fold cross validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


print("Best Parameters:", random_search.best_params_)
print("Best CV RMSE:", -random_search.best_score_)


#validation
best_xgb = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_xgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.5f}")
r2 = r2_score(y_val, y_pred)
print(f"Validation R^2: {r2:.5f}")


#predict test data
y_test_pred = best_xgb.predict(test)
y_test_pred=np.clip(y_test_pred,0,1)


#submission file
submission_sample = submission.copy()
submission_sample['accident_risk'] = y_test_pred
submission_sample.to_csv('submission.csv',index=False)
print("Submission file created: submission.csv")




