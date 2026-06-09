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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.head(5)


test_df.head(5)


train_df.isnull().sum()


test_df.isnull().sum()


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.fit_transform(test_df['Sex'])


test_ids = test_df['id']
test_df = test_df.drop(['id'], axis=1)


train_df.head(5)


test_df.head(5)


X = train_df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = train_df['Calories']


import matplotlib.pyplot as plt
import seaborn as sns

for col in X.columns:
    plt.figure(figsize=(5, 3))
    sns.scatterplot(x=X[col], y=y)
    plt.title(f'{col} vs target')
    plt.show()



import pandas as pd

correlation = X.assign(target=y).corr()['target'].drop('target')
print(correlation.sort_values(ascending=False))


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
len(train_X), len(test_X), len(train_y), len(test_y)


from sklearn.metrics import mean_squared_log_error, r2_score
def pred_metrics(predictions, test_y):
    mse = mean_squared_log_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    print(f"msle: {mse:.4f}")
    print(f"r2: {r2:.4f}")


# Check if there are negative values in predictions 
# (no negative permit value)
def neg_test(pred):
    neg = 0
    for i in pred:
        if i < 0:
            neg = neg + 1
    return neg


from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1
)


model.fit(train_X, train_y)


predictions = model.predict(test_X)


neg_test(predictions)


pred_metrics(predictions, test_y)


from xgboost import XGBRegressor

xgb = XGBRegressor(
    max_depth=5, 
    n_estimators=1500, 
    learning_rate=0.05
)

xgb.fit(train_X, train_y)
predictions_xgb = xgb.predict(test_X)


neg_test(predictions_xgb)


pred_metrics(predictions_xgb, test_y)


from catboost import CatBoostRegressor

catboost = CatBoostRegressor(
    learning_rate=0.05,
    max_depth=10,
    n_estimators=1000
)

catboost.fit(train_X, train_y, verbose=False)
predictions_catboost = catboost.predict(test_X)


neg_test(predictions_catboost)


pred_metrics(predictions_catboost, test_y)


from sklearn.ensemble import StackingRegressor

base_models = [
    ('xgb', xgb),
    ('catboost', catboost)
]

meta_model = LinearRegression()

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False,
    verbose=0
)

stacked_model.fit(train_X, train_y)

stack_predictions = stacked_model.predict(test_X)


neg_test(stack_predictions)


pred_metrics(stack_predictions, test_y)


submission_predictions = stacked_model.predict(test_df)


neg_test(submission_predictions)


submission = pd.DataFrame({'id': test_ids.values,
                          'Calories': submission_predictions 
                          })
submission.head(5)


submission.to_csv('/kaggle/working/calories_submission.csv', index=False)

