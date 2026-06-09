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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.isnull().sum()


train.head()


import seaborn as sns
sns.countplot(x = 'Color', data = train)


sns.countplot(x = 'Brand', data = train)


sns.histplot(x = 'Weight Capacity (kg)' ,data=train)


def fill_ave(c):
 missing_indices = train[train[c].isna()].index

 non_missing_colors = train[c].dropna()
 train.loc[missing_indices, c] = np.random.choice(non_missing_colors, size=len(missing_indices))

def fill_ave_t(c):
 missing_indices = test[test[c].isna()].index
 non_missing_colors = test[c].dropna()
 test.loc[missing_indices, c] = np.random.choice(non_missing_colors, size=len(missing_indices))


fill_ave("Material")
fill_ave('Size')
fill_ave('Waterproof')
fill_ave("Style")
fill_ave('Brand')
fill_ave('Laptop Compartment')
fill_ave('Color')

fill_ave_t("Material")
fill_ave_t('Size')
fill_ave_t('Waterproof')
fill_ave_t("Style")
fill_ave_t('Brand')
fill_ave_t('Laptop Compartment')
fill_ave_t('Color')


test.isnull().sum()


test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())
train = train.dropna(subset = ['Weight Capacity (kg)'])


train.isnull().sum()


train


dm1 = pd.get_dummies(train['Brand'])
dm2 = pd.get_dummies(train['Material'])
dm3 = pd.get_dummies(train['Style'])
dm4 = pd.get_dummies(train['Color'])


dm11 = pd.get_dummies(test['Brand'])
dm21 = pd.get_dummies(test['Material'])
dm31 = pd.get_dummies(test['Style'])
dm41 = pd.get_dummies(test['Color'])


test = pd.concat([test,dm11,dm21,dm31,dm41],axis = 1)


train = pd.concat([train,dm1,dm2,dm3,dm4],axis = 1)


train.drop(['id','Brand','Material','Size','Style','Color'],axis = 1,inplace = True)


test.drop(['Brand','Material','Size','Style','Color'],axis = 1,inplace = True)


train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes':1,'No':0})
train['Waterproof'] = train['Waterproof'].map({'Yes':1,'No':0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes':1,'No':0})
test['Waterproof'] = test['Waterproof'].map({'Yes':1,'No':0})


train['Weight Capacity (kg)'] = pd.to_numeric(train['Weight Capacity (kg)'], errors='coerce')
test['Weight Capacity (kg)'] = pd.to_numeric(test['Weight Capacity (kg)'], errors='coerce')


train.info()


test


test_n = test.drop('id', axis = 1)


y = train['Price']
x = train.drop('Price', axis = 1)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x, y, test_size = 0.2)


from xgboost import XGBRegressor



# from sklearn.model_selection import GridSearchCV

# # 参数网格
# param_grid = {
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.1, 0.2, 0.3],
#     'n_estimators': [100, 200, 300, 400],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# # 网格搜索
# grid_search = GridSearchCV(estimator=XGBRegressor(device = "cuda"), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3,verbose = 5)
# grid_search.fit(x_train, y_train)

# # 最佳参数
# print("最佳参数：", grid_search.best_params_)


params = {
    'max_depth': 4,
    'learning_rate': 0.3,
    'n_estimators': 400,
    'device':'cuda'
}


# model = XGBRegressor(**params)
# model.fit(x_train,y_train)
# y_pre = model.predict(x_val)


from catboost import CatBoostRegressor
cat_Params = {
    'learning_rate':0.1,
    'task_type':'GPU',
    'iterations':2000,
    'loss_function':'RMSE',
    'depth':6
}
cat_model = CatBoostRegressor(**cat_Params)
cat_model.fit(x_train,y_train,eval_set = (x_val,y_val))
y_pre = cat_model.predict(x_val)


from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
print(mean_absolute_percentage_error(y_val,y_pre))
mean_absolute_error(y_val,y_pre)


rmse = np.sqrt(mean_squared_error(y_val,y_pre))


rmse


# import xgboost as xgb
# xgb.plot_importance(model,importance_type = 'weight') #because the importance of Weight cap so I delete all culoum that lost vaule


test_pre = cat_model.predict(test_n)


submission = pd.DataFrame({'id':test['id'],'Price':test_pre})


submission.to_csv('submission.csv',index=False)

