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


# not using this anymore
# import polars as pl
import pandas as pd


# change this back to kaggle path
# df = pd.read_csv("../data/calories-burnt-data/train.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
X,y = df.drop(columns=['Calories','id']), df['Calories']


X_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
# X_test = pd.read_csv("../data/calories-burnt-data/test.csv")
ids = X_test['id']
X_test = X_test.drop(columns=['id'])


X.head()


# polars is way faster and used less memory but can't use it in both the algorithms I have selected as explained at the start of the notebook.
X.info()


from missingno import matrix
matrix(X)
# no missing values as we can see here


# converting Sex to category type
X['Sex'] = X['Sex'].astype('category')
X_test['Sex'] = X_test['Sex'].astype('category')

# rest of the columns are already in the right type


numerical_columns = X.select_dtypes(include=np.number).columns


# THIS CODE WORKS I HAVE JUST COMMENTED IT OUT TO DECREASE THE RUN TIME IN KAGGLE ENVIRONMENT
import seaborn as sns

# import matplotlib.pyplot as plt

# # Select numerical columns
# numerical_columns = X.select_dtypes(include=np.number).columns

# # Calculate the number of rows required for the grid
# num_cols = len(numerical_columns)
# num_rows = (num_cols + 1) // 2  # Two plots per row

# # Create a grid for plotting
# fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))
# axes = axes.flatten()

# # Plot each numerical column using seaborn
# for i, col in enumerate(numerical_columns):
#     sns.histplot(data=X, x=col, ax=axes[i], bins=30, kde=True, color='blue')
#     axes[i].set_title(f'Distribution of {col}')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('Frequency')

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].set_visible(False)

# # Adjust layout
# plt.tight_layout()
# plt.show()


# get the skewness of the numerical columns
# Calculate and print skewness for each numerical column
skewness = X[numerical_columns].skew()
for col, skew in skewness.items():
    print(f"Skewness of {col}: {skew:.4f}")


X['Sex'].value_counts().plot(kind='bar')


# The y values are uniformally distributed
y.skew()


# The CatBoostRegressor class is a gradient boosting algorithm that is particularly effective for categorical features.
# using log1p

import numpy as np

class RMSLEMetricLog1p(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))  # Prevent divide by zero

    def is_max_optimal(self):
        return False  # Lower RMSLE is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        approx = approxes[0]
        assert len(target) == len(approx)

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            y_pred = max(0.0, approx[i])
            y_true = max(0.0, target[i])

            log_pred = np.log1p(y_pred)
            log_true = np.log1p(y_true)
            error = (log_pred - log_true) ** 2

            w = 1.0 if weight is None else weight[i]
            error_sum += w * error
            weight_sum += w

        return error_sum, weight_sum



# using epsilon = 1e-5
class RMSLEMetric(object):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def get_final_error(self, error, weight):
        # Prevent divide by zero
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        # Lower RMSLE is better if we want to get the max we return True
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        approx = approxes[0]
        assert len(target) == len(approx)

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            y_pred = approx[i]
            y_true = target[i]

            # Soft floor
            y_pred = max(y_pred, self.epsilon)
            y_true = max(y_true, self.epsilon)

            log_pred = np.log(y_pred)
            log_true = np.log(y_true)
            error = (log_pred - log_true) ** 2

            w = 1.0 if weight is None else weight[i]
            error_sum += w * error
            weight_sum += w

        return error_sum, weight_sum


# we will use CatBoost to train the model and predict the values and submit.
# Everything is set to default and CatBoost is quite good at vanilla settings as well.

from catboost import CatBoostRegressor
reg = CatBoostRegressor(
    eval_metric=RMSLEMetricLog1p(),
    early_stopping_rounds=50,
)


# for better performance we can use the GPU and convert the train and test data to catboost pool
from catboost import Pool
train_pool = Pool(
    X,
    y,
    cat_features=['Sex']
)

test_pool = Pool(
    X_test,
    cat_features=['Sex']
)


# training the CatBoostRegressor
reg.fit(
    train_pool
)


y_preds = reg.predict(test_pool)


# its important here to check if the min value is greater than zero otherise we will get an error when submitting the predictions.
y_preds.min()


def create_submission(y_preds,ids,model):
    # Create submission DataFrame for CatBoost (Score: 0.05930) Baseline model
    pd.DataFrame({
        'id': ids,
        'Calories': y_preds
    }).to_csv(f'submission_{model.__class__.__name__}.csv', index=False)


X.head()


# custom method for RMSLE just like CatBoost
import numpy as np

def rmsle_eval_epsilon(preds, train_data, epsilon=1e-5):
    y_true = train_data.get_label()
    weight = train_data.get_weight()  # This returns None if no weights provided

    # Soft floor
    y_pred = np.maximum(preds, epsilon)
    y_true = np.maximum(y_true, epsilon)

    log_pred = np.log(y_pred)
    log_true = np.log(y_true)
    squared_log_error = (log_pred - log_true) ** 2

    if weight is not None:
        weighted_error = np.sum(weight * squared_log_error)
        weight_sum = np.sum(weight)
    else:
        weighted_error = np.sum(squared_log_error)
        weight_sum = len(y_true)

    rmsle = np.sqrt(weighted_error / (weight_sum + 1e-38))  # to avoid division by 0
    return 'rmsle', rmsle, False  # False = lower is better



import lightgbm as lgb
import numpy as np
# Create LightGBM dataset
train_data = lgb.Dataset(X, label=y)
test_data  = lgb.Dataset(X_test)

# Set parameters
params = {
    'objective': 'poisson',
    'metric':'None' # setting this to none, since we are using custom metric
}

# Train the model
model = lgb.train(
    params,
    train_data,
    feval=lambda preds, data: rmsle_eval_epsilon(preds, data, epsilon=1e-5),
    num_boost_round=1000,
)





# Predict
y_pred = model.predict(X_test)




# this saves the lightgbm model
# create_submission(y_pred,ids,model)
# The score is not that good for this one


# creating submission for the catboost model
create_submission(y_preds,ids,reg)

