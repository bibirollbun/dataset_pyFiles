import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
import time
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.DataFrame(data_train.drop(['id'], axis=1))
df.isnull().sum().sort_values(ascending=False)


df['Sex'] = df['Sex'].astype('category')


X = df.drop("Calories", axis=1)
y = df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "reg:squarederror", "tree_method": "hist", "eta":0.1, "max_depth":9,
          "min_child_weight": 3, "subsample":0.8, "updater":"grow_quantile_histmaker", "num_parallel_tree":4,
          "eval_metric":"rmsle"}
evals = [(dtrain_reg, "Train"), (dtest_reg, "Test")]

n = 2000
start_t = time.time()

model_t = xgb.train(params=params,
                    evals=evals,
                    dtrain=dtrain_reg,
                    num_boost_round=n,
                    verbose_eval=100,
                    early_stopping_rounds=50)

end_t = time.time()

preds = model_t.predict(dtest_reg)
rmsle = root_mean_squared_log_error(y_test, preds)

print(f"\nTime to fit XGB Regression Model: {end_t - start_t:.3f} seconds")
print(f"\nRMSLE of the base_t model: {rmsle:.5f}")


gain = xgb.plot_importance(model_t, importance_type='gain', show_values=False, max_num_features=10, xlabel='F Score', ylabel='Features', title='XGBoost Feature Importance(Gain)')
weight = xgb.plot_importance(model_t, importance_type='weight', show_values=False, max_num_features=10, xlabel='F Score', ylabel='Features', title='XGBoost Feature Importance(Weight)')
cover = xgb.plot_importance(model_t, importance_type='cover', show_values=False, max_num_features=10, xlabel='F Score', ylabel='Features', title='XGBoost Feature Importance(Cover)')

plt.show()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

submission_id = test_df['id']


test_df = pd.DataFrame(test_df.drop(['id'], axis=1))
test_df.isnull().sum().sort_values(ascending=False)

test_df['Sex'] = test_df['Sex'].astype('category')

test_df = xgb.DMatrix(test_df, enable_categorical=True)


submission_prediction = model_t.predict(test_df)

#-------------Look for negative values and change them to 0----------#
for i in range(len(submission_prediction)):
   if submission_prediction[i] < 0:
      submission_prediction[i] = 0


#------------Widdle down Calories to 3 decimal places---------------#
to_places = 3

def truncate(f, n):
   return math.floor(f * 10 ** n) / 10 ** n

#---------------Make a dataframe with the id and calories-----------#
output = pd.DataFrame({'id': submission_id,
                       'Calories': submission_prediction.squeeze()})

#-----------Use the truncate function for calories---------#
output['Calories'] = output['Calories'].apply(lambda number: truncate(number, to_places))

print(output.head)

print("\nUsing Trained Regression Model: \n", output['Calories'].sort_values(ascending=False))


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sample_sub['Calories'] = submission_prediction
sample_sub.to_csv('/kaggle/working/Submission.csv', index=False)

