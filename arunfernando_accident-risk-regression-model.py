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
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
data.head()



data.info()


data.describe()


sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


data.drop("id",inplace=True,axis=1)


fig= plt.figure(figsize=(10,3) )
fig.add_subplot(1,2,1)
a= data["accident_risk"].value_counts(normalize=True).plot.pie()
fig.add_subplot(1,2,2)
churnchart=sns.countplot(x=data["accident_risk"])
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10,5))
fig.add_subplot(1,3,1)
ar_6 = sns.boxplot(x = data["accident_risk"], y = data["lighting"])
fig.add_subplot(1,3,2)
ar_6 = sns.boxplot(x = data["accident_risk"], y = data["weather"])
fig.add_subplot(1,3,3)
ar_6 = sns.boxplot(x = data["accident_risk"], y = data["time_of_day"])
plt.tight_layout()
plt.show()




cols = data.select_dtypes(["object"]).columns
ds = pd.get_dummies(data[cols], drop_first = True)
ds


data = pd.concat([data, ds], axis =1)
data.drop(cols, axis =1, inplace = True)


data.info()


bcols = data.select_dtypes(["bool"]).columns
data[bcols] = data[bcols].astype(int)
data.info()


np.random.seed(123)
train, test = np.split(data.sample(frac=1, random_state=42), 
                       [int(.8*len(data))])

train.shape, test.shape


X_train = train.copy() 
#X_val = val.copy()
X_test = test.copy()

Y_train = X_train.pop('accident_risk')
#Y_val = X_val.pop('accident_risk')
Y_test = X_test.pop('accident_risk')

X_train.shape, Y_train.shape


##Random forest regressor for feature selection

from sklearn.ensemble import RandomForestRegressor

rregressor = RandomForestRegressor(n_estimators = 100, max_depth = 10)


from sklearn.feature_selection import RFE
n_features_to_select = 1
rfe = RFE(rregressor, n_features_to_select = n_features_to_select)
rfe.fit(X_train, Y_train)


from operator import itemgetter
features = X_train.columns.to_list()
top_features = list()
for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
    print(x, y)
    top_features.append(y)


top_features[0:9]


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
#X_val = scale.transform(X_val)
X_test_scaled = scale.transform(X_test)



Y_train.value_counts(normalize=True)


#from sklearn.linear_model import LogisticRegression

#lr_basemodel = LogisticRegression()
#lr_basemodel.fit(X_train, Y_train)


X_train_new = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
X_test_new= pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)

X_train_new.shape, X_test_new.shape, Y_train.shape, Y_test.shape


X_train_new


X_train_new_10feat = X_train_new[top_features[0:9]]
X_test_new_10feat = X_test_new[top_features[0:9]]


import xgboost as xgb
regressor = xgb.XGBRegressor(eval_metric = 'rmse')

from sklearn.model_selection import GridSearchCV


param_grid = {'max_depth':[4,5,6],
              'n_estimators':[500,600,700],
              'learning_rate':[0.01, 0.015]
             }


search = GridSearchCV(regressor, param_grid, cv = 5, scoring = 'neg_root_mean_squared_error').fit(X_train_new_10feat, Y_train)


search_results = pd.DataFrame(search.cv_results_)
columns = [column for column in search_results if column.startswith('param_')]
columns.append('mean_test_score')
columns.append('rank_test_score')

search_results["RMSE"] = np.sqrt(-search_results['mean_test_score'])
columns.append('RMSE')
search_results[columns].sort_values(by = 'mean_test_score',ascending = False)


print("The best hyperparameters are ",search.best_params_)


regressor = xgb.XGBRegressor(learning_rate = search.best_params_['learning_rate'],
                            n_estimators = search.best_params_['n_estimators'],
                             max_depth = search.best_params_['max_depth'],
                             eval_metric = 'rmse'
                            )

eval_set = [(X_train_new_10feat, Y_train), (X_test_new_10feat, Y_test)]

regressor.fit(X_train_new_10feat, Y_train, eval_set = eval_set, early_stopping_rounds = 100, verbose = True)


predictions = regressor.predict(X_test_new_10feat)


from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(Y_test, predictions))

RMSE


data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col =0)
cols = data.select_dtypes(["object"]).columns
ds = pd.get_dummies(data[cols], drop_first = True)
data = pd.concat([data, ds], axis =1)
data.drop(cols, axis =1, inplace = True)
bcols = data.select_dtypes(["bool"]).columns
data[bcols] = data[bcols].astype(int)
data_scaled = scale.transform(data)

data_new = pd.DataFrame(data_scaled, columns = data.columns, index = data.index)
data_new_10feat = data_new[top_features[0:9]]





predictions = regressor.predict(data_new_10feat)
predictions.shape



data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
data

id_df = data.pop('id')


output = pd.DataFrame({'id':id_df, 'accident_risk':predictions})
output.to_csv('submission.csv', index=False)


output_written = pd.read_csv('/kaggle/working/submission.csv')
output_written


from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(regressor, max_num_features=10, ax=ax)
plt.show();

