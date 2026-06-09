import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# load datasets and drop the id column inthe train and test datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submit_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


# Take a look at the training dataset dtypes and for any missing rows
train_df.info()


# get a peek the first 5 lines of the data.
train_df.head()


# transform the true/false entries into 0/1
binary = train_df.select_dtypes(include=['bool']).columns
train_df[binary] = train_df[binary].astype(int)
test_df[binary] = test_df[binary].astype(int)


train_df = pd.get_dummies(train_df, dtype=int)
test_df = pd.get_dummies(test_df, dtype=int)
train_df.info()


corr = train_df.corr()
sorted_corr = corr['accident_risk'].sort_values(ascending=False)
sns.heatmap(sorted_corr.to_frame(), annot=True, vmax=1, vmin=-1, cmap='RdBu')
plt.title('Accident Risk Correlation')
plt.show()


X = train_df.drop(columns=['accident_risk'])
y = train_df['accident_risk']


xgb_params = {'verbosity':0,
              'max_depth':8,
              'learning_rate':0.004,
              'n_estimators':2000,
              'subsample':0.9,
              'colsample_bytree':1.0,
              'device':'cuda',         # cuda for GPU, cpu for CPU
              'eval_metric':'rmse',
              'objective':'reg:squarederror',
              'gamma':0.001
              }


# runtime 17 minutes on CPU
# runtime 84 seconds on TPU
folds = 5
train_rmse = []   # holder for each fold's rmse to be graphed later
test_pred = np.zeros(len(test_df))
kf = KFold(n_splits=folds, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(X)):

  X_train = X.iloc[train_index]
  X_test = X.iloc[test_index]
  y_train = y.iloc[train_index]
  y_test = y.iloc[test_index]

  model = XGBRegressor(**xgb_params)
  model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

  test_pred += model.predict(test_df)/folds

  y_pred = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  train_rmse.append(rmse)

  print(f'Fold {i+1} RMSE: {rmse:.6f}')


plt.plot(train_rmse, label='Training RMSE', marker='o', linestyle='-')
plt.hlines(y=np.mean(train_rmse), xmin=0, xmax=folds-1, linestyles='--',
           colors='red', label='Mean ' + str(np.round(np.mean(train_rmse),6)))
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.legend()
plt.show()


# feature importances
sorted_idx = model.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Importance")
plt.title('XGBoost Feature Importances')
plt.show()


submission = pd.DataFrame({
    'id': submit_df['id'],
    'accident_risk': test_pred
})
submission.to_csv('submission.csv', index=False)
submission[:5]

