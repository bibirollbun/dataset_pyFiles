import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import numpy as np
from sklearn.feature_selection import SelectKBest,f_regression


data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


data,target = data.drop(columns='BeatsPerMinute'),data['BeatsPerMinute']


def feature_eng(data):
    
    epsilon = 1e-8
    for i,feature_x in enumerate(data.columns[1:11]):
        #data[f'sqrt_{feature_x}'] = np.sqrt(np.maximum(data[feature_x], 0))
        data[f'square_{feature_x}'] = np.square(np.where(pd.isna(data[feature_x]), 0, data[feature_x]))
        for feature_y in data.columns[i+1:11]:
            if f'{feature_x}_x_{feature_y}' not in data.columns and feature_y!=feature_x:
                data[f'{feature_x}_x_{feature_y}'] = data[feature_y] * data[feature_x]
                data[f'{feature_x}_/_{feature_y}'] = data[feature_y] // (data[feature_x] + epsilon)
                data[f'{feature_x}_+_{feature_y}'] = data[feature_y] + data[feature_x]

    return data


cat_p = {'bootstrap_type': 'Bernoulli', 'grow_policy': 'Lossguide', 'iterations': 302, 'learning_rate': 0.007270568577829121, 'depth': 5, 'l2_leaf_reg': 0.4680070406997779, 'subsample': 0.7115155868128413, 'colsample_bylevel': 0.8619788903193408, 'min_data_in_leaf': 30, 'random_strength': 6.090644689831845e-05}


xgb_p={'n_estimators': 457, 'learning_rate': 0.010871673742809437, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.5054595720979277, 'colsample_bytree': 0.9967792973675427, 'gamma': 0.17096824878292366, 'reg_alpha': 2.4797738176673883e-07, 'reg_lambda': 2.66308155578018e-05}


 lgb_p = {'n_estimators': 586, 'learning_rate': 0.020066724377060474, 'num_leaves': 22, 'max_depth': 12, 'min_child_samples': 75, 'subsample': 0.5016747714119865, 'colsample_bytree': 0.6418465821066085, 'reg_alpha': 2.40074863943761e-08, 'reg_llambda': 3.735103058319593e-06, 'min_split_gain': 0.0006581741141247193}


class Model:
    def __init__(self,n_models):
        self.n_models = n_models
        self.weights = []
        self.models = []
        self.x_params = xgb_p
        self.l_params = lgb_p
        self.c_params = cat_p
    def fit(self,data,target):

        X_train,X_val,y_train,y_val = train_test_split(data ,target,train_size=0.95,random_state=68)
        for i in range(self.n_models):
            #model_1 = LGBMRegressor(**self.x_params)
            model_2 = XGBRegressor(**self.l_params)
            model_3 = CatBoostRegressor(**self.c_params)
            
            #model_1.fit(X_train,y_train)
            model_2.fit(X_train,y_train)
            model_3.fit(X_train,y_train)
            
            #y_1 = model_1.predict(X_val)
            y_2 = model_2.predict(X_val)
            y_3 = model_3.predict(X_val)
            
            #score_1 = mean_squared_error(y_val,y_1)
            score_2 = mean_squared_error(y_val,y_2)
            score_3 = mean_squared_error(y_val,y_3)
            
            #self.models.append(model_1)
            self.models.append(model_2)
            self.models.append(model_3)


            #weight_1 = 1 / (score_1 + 1e-10)
            weight_2 = 1 / (score_2 + 1e-10)
            weight_3 = 1 / (score_3 + 1e-10)

            
            #self.weights.append(weight_1)
            self.weights.append(weight_2)
            self.weights.append(weight_3)
            
            
        
    def predict(self,data):
        res = []
        for model in self.models:
            proba = model.predict(data)
            res.append(proba)
        return np.average(res,axis=0,weights=self.weights)


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


data = feature_eng(data)
test = feature_eng(test)


from sklearn.feature_selection import RFE

model = XGBRegressor(**xgb_p, random_state=42)
selector = RFE(estimator=model, n_features_to_select=50, step=5)
selector.fit(data, target)

names = data.columns[selector.get_support()]

test = test[names]
data = data[names]


model = Model(10)
model.fit(data,target)
output = model.predict(test)
submission = pd.DataFrame({
    'id': range(524164, 524164 + len(test)),
    'y': output
})
submission.to_csv('/kaggle/working/submission.csv', index=False)


