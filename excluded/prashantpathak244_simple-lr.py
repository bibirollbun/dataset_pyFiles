import numpy as np
import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
train.head(2)


test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test.head(2)


test.isna().sum().sum()


#this suggest we have no missing value and the data type is of integers/floats.
train.info()


train.isna().sum().sum()


#checking for distribution of data
train.describe()


train.corr()


train.drop(['id'] , inplace = True , axis = 1)
test.drop(['id'] , inplace = True , axis = 1)


import seaborn as sns
sns.heatmap(train.corr() , cmap = 'coolwarm' , annot = True , fmt = '.2f')


from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
x_sc = sc.fit_transform(train[train.columns[:-1]])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb


train[train.columns[:-1]].head(2)


train['BeatsPerMinute'].shape


x_train ,x_val ,  y_train , y_val = train_test_split(x_sc , train['BeatsPerMinute'] , test_size = 0.25 , random_state = 42 , shuffle = True)



# x_train ,x_val ,  y_train , y_val = train_test_split(train[train.columns[:-1]] , train['BeatsPerMinute'] , test_size = 0.3 , random_state = 42 , shuffle = True)



# x_train.head(2)


y_train.shape


from lightgbm import LGBMRegressor


# #objective is telling xgb which loss function you are optimizing
# xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
lgbm = LGBMRegressor(n_estimators=1000,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    subsample=0.6, 
    reg_lambda=0.5, 
    reg_alpha=0.1,
    min_child_samples=50,  
    colsample_bytree=1.0 )


# param_grid = {
#     "n_estimators": [100, 300, 500],        # number of boosting rounds
#     "max_depth": [3, 5, 7],                 # tree depth
#     "learning_rate": [0.01, 0.05, 0.1]     # shrinkage
#     # "subsample": [0.8, 1.0],                # fraction of samples used per tree
#     # "colsample_bytree": [0.8, 1.0]          # fraction of features per tree
# }


# from sklearn.model_selection import RandomizedSearchCV
# # Grid search
# grid = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_grid,
#     scoring="neg_mean_squared_error", 
#     cv=5,
#     verbose=1,
#     n_jobs=-1, 
#     random_state=42
# )


# grid.fit(x_train, y_train)


# print(grid.best_params_)


# (-grid.best_score_)**0.5


# best_model = grid.best_estimator_


# xg = xgb.XGBRegressor()
# xg.fit(x_train , y_train)
lgbm.fit(x_train , y_train)


# lr = LinearRegression()
# lr.fit(x_train , y_train)


y_pred = lgbm.predict(x_val)
y_pred.shape


import sklearn
print(sklearn.__version__)



def root_mean_squared_error(y_tr , y_pr , loss = 0):
    y_tr = np.array(y_tr)
    y_pr = np.array(y_pr)

    mse = np.mean((y_pr - y_tr) ** 2)
    rmse = np.sqrt(mse)
    return rmse
    
    # for tr, pr in zip(y_tr.values , y_pr):
    #     loss += (tr - pr)**2
    # loss = loss/len(y_pr)
    # loss = loss ** 0.5
    # return loss
    


rm = root_mean_squared_error(y_val , y_pred)


print(rm)


test_sc = sc.transform(test)


out = lgbm.predict(test_sc)


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
sample_sub.head(2)


sample_sub['BeatsPerMinute'] = out


sample_sub.to_csv("submission.csv", index=False)







