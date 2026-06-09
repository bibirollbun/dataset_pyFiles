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


data=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
print(data)
test_data=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
test_data


from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor,BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt



data.fillna(data.mean(),inplace=True)


X = data[['f1','f2','f3','f4','f5','f6']]
y = data['target']

print(X)
print(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#PARAMETER TUNING RANDOM SEARCH XGB
# param_grid = {
#     'n_estimators': np.arange(100, 1000, 100),
#     'max_depth': np.arange(3, 12, 1),
#     'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'gamma': np.arange(0, 10, 1),
#     'reg_alpha': np.arange(0, 1.1, 0.1),
#     'reg_lambda': np.arange(0, 10, 1)
# }

# xgb = XGBRegressor()
# random_search = RandomizedSearchCV(xgb,param_distributions=param_grid, n_iter = 50,cv=5,scoring='r2',n_jobs=-1)

# random_search.fit(X_train,y_train)

# print("Best parameters:" , random_search.best_params_)




#PARAMETER TUNING RANDOM SEARCH RF
# param_grid_rf = {
#     'n_estimators': np.arange(100, 1000, 100),  
#     'max_depth': np.arange(3, 20, 1),  
#     'min_samples_split': [2, 5, 10],  
#     'min_samples_leaf': [1, 2, 4, 10],  
#     'max_features': ['auto', 'sqrt', 'log2'],  
#     'bootstrap': [True, False]  
# }

# rf = RandomForestRegressor()

# random_search_rf = RandomizedSearchCV(rf,param_distributions = param_grid_rf,n_iter = 50, cv=5 , scoring = 'r2', n_jobs= -1,verbose=2,random_state=42)
# random_search_rf.fit(X_train, y_train)

# print("Best Parameters:", random_search_rf.best_params_)



# et_model = ExtraTreesRegressor(random_state=42)

# param_dist = {
#     "n_estimators": [100, 200, 300, 500],  # Number of trees
#     "max_depth": [None, 10, 20, 30],  # Tree depth
#     "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
#     "min_samples_leaf": [1, 2, 5, 10],  # Minimum samples in a leaf
#     "max_features": ["sqrt", "log2", None],  # Number of features to consider for split
#     "bootstrap": [True, False],  # Bootstrap sampling
# }

# random_search = RandomizedSearchCV(
#     et_model,
#     param_distributions=param_dist,
#     n_iter=20,  # Number of different settings tested
#     cv=5,  # 5-fold cross-validation
#     scoring="r2",  # Optimize for R² score
#     n_jobs=-1,  # Use all CPU cores
#     verbose=2,
#     random_state=42
# )

# random_search.fit(X_train, y_train)
# print("Best Parameters:", random_search.best_params_)
# print(f"Best R² Score: {random_search.best_score_:.4f}")



rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
et = ExtraTreesRegressor(random_state=42)


rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}

xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

et_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}

rf_random = RandomizedSearchCV(rf, rf_params, n_iter=10, cv=5, scoring="r2", n_jobs=-1, verbose=2, random_state=42)
xgb_random = RandomizedSearchCV(xgb, xgb_params, n_iter=10, cv=5, scoring="r2", n_jobs=-1, verbose=2, random_state=42)
et_random = RandomizedSearchCV(et, et_params, n_iter=10, cv=5, scoring="r2", n_jobs=-1, verbose=2, random_state=42)

rf_random.fit(X_train, y_train)
xgb_random.fit(X_train, y_train)
et_random.fit(X_train, y_train)

best_rf_params = rf_random.best_params_
best_xgb_params = xgb_random.best_params_
best_et_params = et_random.best_params_

print("Best RF Params:", best_rf_params)
print("Best XGB Params:", best_xgb_params)
print("Best ET Params:", best_et_params)

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), { 
    "n_estimators": [best_rf_params["n_estimators"] - 50, best_rf_params["n_estimators"], best_rf_params["n_estimators"] + 50],
    "max_depth": [best_rf_params["max_depth"] - 5, best_rf_params["max_depth"], best_rf_params["max_depth"] + 5],
    "min_samples_split": [best_rf_params["min_samples_split"]],
    "min_samples_leaf": [best_rf_params["min_samples_leaf"]],
}, cv=5, scoring="r2", n_jobs=-1, verbose=2)

xgb_grid = GridSearchCV(XGBRegressor(random_state=42), {
    "n_estimators": [best_xgb_params["n_estimators"] - 50, best_xgb_params["n_estimators"], best_xgb_params["n_estimators"] + 50],
    "max_depth": [best_xgb_params["max_depth"] - 2, best_xgb_params["max_depth"], best_xgb_params["max_depth"] + 2],
    "learning_rate": [best_xgb_params["learning_rate"]],
    "subsample": [best_xgb_params["subsample"]],
    "colsample_bytree": [best_xgb_params["colsample_bytree"]],
}, cv=5, scoring="r2", n_jobs=-1, verbose=2)

et_grid = GridSearchCV(ExtraTreesRegressor(random_state=42), {
    "n_estimators": [best_et_params["n_estimators"] - 50, best_et_params["n_estimators"], best_et_params["n_estimators"] + 50],
    "max_depth": [best_et_params["max_depth"] - 5, best_et_params["max_depth"], best_et_params["max_depth"] + 5],
    "min_samples_split": [best_et_params["min_samples_split"]],
    "min_samples_leaf": [best_et_params["min_samples_leaf"]],
}, cv=5, scoring="r2", n_jobs=-1, verbose=2)

rf_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)
et_grid.fit(X_train, y_train)

final_rf = rf_grid.best_estimator_
final_xgb = xgb_grid.best_estimator_
final_et = et_grid.best_estimator_

print("final RF Params:", final_rf)
print("final XGB Params:", final_xgb)
print("final ET Params:", final_et)


#final_estimator = XGBRegressor(n_estimators = 50,learning_rate=0.05)



model=RandomForestRegressor(n_estimators = 800,oob_score=True,max_leaf_nodes = 300 ,ccp_alpha=0.01,min_samples_split= int(0.02 * len(X_train)),min_samples_leaf=int(0.01 * len(X_train)),max_features=None ,max_depth = None,bootstrap=True,random_state=42)


xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=10, colsample_bytree=0.7, subsample=0.9,reg_lambda = 2,reg_alpha=0.30000000000000004,gamma = 5)


et_model = ExtraTreesRegressor(n_estimators=200, max_depth=None,min_samples_split=5,min_samples_leaf=1,max_features = 'log2',bootstrap=False,random_state=42)



#lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=10,force_col_wise=True)

#ensemble = VotingRegressor(estimators=[('model', model), ('xgb_model', xgb_model),('lgb',lgb)],weights=[0.3,0.6,0.1])

ensemble = StackingRegressor(estimators= [('model',model),('xgb_model',final_xgb),('et_model',final_et)],final_estimator = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, colsample_bytree=0.8)
)

#ENSEMBLE MODEL FITTING
ensemble.fit(X_train, y_train)






#ENSEMBLE MODEL PREDICTION

y_pred = ensemble.predict(X_test)
y_pred




#ENSEMBLE SCORE
r2=r2_score(y_test,y_pred)
r2


#ALL MODELS FITTING
model.fit(X_train,y_train)
final_xgb.fit(X_train,y_train)
final_et.fit(X_train,y_train)


# rf_importance = model.feature_importances_
# xgb_importance = final_xgb.feature_importances_
# et_importance = final_et.feature_importances_

# # Plot feature importance
# plt.figure(figsize=(12, 5))
# plt.bar(np.arange(len(et_importance)), et_importance, label="ExtraTrees", alpha=0.7)
# plt.bar(np.arange(len(rf_importance)), rf_importance, label="Random Forest", alpha=0.5)
# #plt.bar(np.arange(len(xgb_importance)), xgb_importance, label="XGBoost", alpha=0.3)
# plt.legend()
# plt.title("Feature Importance Comparison")
# plt.show()


#ALL MODELS PREDICTION
y_pred_rf = model.predict(X_test)

y_pred_xgb = final_xgb.predict(X_test)

y_pred_et = final_et.predict(X_test)


#ALL MODEL SCORE
r2_rf = r2_score(y_test,y_pred_rf)
print("the rf has score: ", r2_rf)

r2_xgb = r2_score(y_test,y_pred_xgb)
print("the xgb has score: ", r2_xgb)

r2_et = r2_score(y_test,y_pred_et)
print("the et has score:", r2_et)


test_X = test_data[['f1','f2','f3','f4','f5','f6']]
test_X_scaled = scaler.transform(test_X)


test_pred = ensemble.predict(test_X_scaled)
print(test_pred)


#test_pred = xgb_model.predict(test_X_scaled)
#print(test_pred)


id_column = np.arange(1,len(test_pred)+1)


result = pd.DataFrame({
    'id':id_column,
    'target':test_pred
})

result


result.to_csv('submission.csv', index=False)




