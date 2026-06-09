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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


test.head()


train.info()


train.shape


test.shape


train.describe().T


train.isnull().sum()


test.isnull().sum()


train = train.drop(columns=['id'], axis=1)
test = test.drop(columns=['id'], axis=1)


## analyze target distribution
sns.histplot(x='Listening_Time_minutes', data=train)


num_cols = train.select_dtypes(include=['float64', 'int64']).columns
cat_cols = train.select_dtypes(include=['object']).columns


for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(train[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(8, 3))
    top_vals = train[col].value_counts().nlargest(10).index
    sns.countplot(data=train[train[col].isin(top_vals)], x=col, order=top_vals)
    plt.xticks(rotation=45)
    plt.title(f'Top 10 Categories in {col}')
    plt.tight_layout()
    plt.show()


cols_to_fill = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
train[cols_to_fill] = train[cols_to_fill].fillna(train[cols_to_fill].mean())
test[cols_to_fill] = test[cols_to_fill].fillna(test[cols_to_fill].mean())


train.isnull().sum()


test.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


train.head()


test.head()


corr = train.corr()
plt.figure(figsize =(14, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm')


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(columns = ['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']


def train(model, X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation (uses negative MSE, so we take sqrt of abs)
    cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    cv_rmse = np.mean(np.sqrt(np.abs(cv_scores)))  # Convert each fold's MSE to RMSE and average
    
    # Print results
    print("Results")
    print("Test RMSE:", rmse)
    print("CV RMSE:", cv_rmse)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Model Coefficients')


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')


from xgboost import XGBRegressor
xg_model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', random_state=42)
train(xg_model, X, y)

features = pd.Series(xg_model.feature_importances_, index=X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='XGBoost Feature Importance')
plt.show()


from lightgbm import LGBMRegressor
model = LGBMRegressor(device='gpu', random_state=42)
train(model, X, y)

features = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='LightGBM Feature Importance')
plt.show()


from catboost import CatBoostRegressor
model = CatBoostRegressor(task_type='GPU',devices='0',verbose=0, random_state=42)
train(model, X, y)

features = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='CatBoost Feature Importance')
plt.show()


y_pred = xg_model.predict(test)


submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = y_pred
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")

