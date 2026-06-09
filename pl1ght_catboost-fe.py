from sklearn.preprocessing import OneHotEncoder
#from scipy.sparce import sparce
import pandas as pd
from math import sqrt
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score  
import optuna
from sklearn.model_selection import train_test_split


def preprocess_data(data:pd.DataFrame):
    data = data.drop(columns=['poutcome','default'],axis=1)
    enc = OneHotEncoder(sparse=True)
    x = enc.fit_transform(data[['job']])
    fe = enc.get_feature_names_out(['job'])

    fe_d = pd.DataFrame.sparse.from_spmatrix(
        x,
        columns=fe,
        index = data.index
    )

    data = pd.concat([data.drop(columns='job',axis=1),fe_d],axis=1)

    x = enc.fit_transform(data[['marital']])
    fe = enc.get_feature_names_out(['marital'])

    fe_d = pd.DataFrame.sparse.from_spmatrix(
        x,
        columns=fe,
        index = data.index
    )

    data = pd.concat([data.drop(columns='marital',axis=1),fe_d],axis=1)
    
    

    
    x = enc.fit_transform(data[['education']])
    fe = enc.get_feature_names_out(['education'])
    
    fe_d = pd.DataFrame.sparse.from_spmatrix(
        x,
        columns=fe,
        index = data.index
    )
    
    data = pd.concat([data.drop(columns='education',axis=1),fe_d],axis=1)
    
    data.loc[data['loan']=='no','loan'] = 0
    data.loc[data['loan']=='yes','loan'] = 1
    data['loan'] = data['loan'].astype(int)
    data.loc[data['housing']=='no','housing'] = 0
    data.loc[data['housing']=='yes','housing'] = 1
    data['housing'] = data['housing'].astype(int)
    
    data['contact'] = data['contact'].map({'unknown':0,'cellular':1,'telephone':2})
    #data = data[data['contact']!=2] 
    data['month'] = data['month'].map({
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    })
    return data


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


cat_params = {'bootstrap_type': 'Bernoulli',
 'grow_policy': 'Depthwise',
 'iterations': 1759,
 'learning_rate': 0.053937554532955165,
 'depth': 7,
 'l2_leaf_reg': 5.3318982303485205,
 'subsample': 0.8943963522764744,
 'colsample_bylevel': 0.7979590924313184,
 'min_data_in_leaf': 48,
 'random_strength': 0.001389347086954187,
 'auto_class_weights': 'SqrtBalanced'}


class Model:
    def __init__(self,n_models):
        self.n_models = n_models
        self.weights = []
        self.models = []
        self.params = cat_params
    def fit(self,data,target):

        X_train,X_val,y_train,y_val = train_test_split(data ,target,train_size=0.95)
        for i in range(self.n_models):
            model = CatBoostClassifier(**self.params)
            model.fit(X_train,y_train)
            y_b = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val,y_b)
            self.models.append(model)
            self.weights.append(score)
        self.weights = np.array(self.weights)/ len(self.weights)
    def predict(self,data):
        res = []
        for model in self.models:
            proba = model.predict_proba(data)[:,1]
            res.append(proba)
        return np.average(res, axis=0, weights=self.weights)
        
        
        


train= preprocess_data(pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv'))
test= preprocess_data(pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv'))
test = feature_eng(test)
train = feature_eng(train)
target = train['y']
data = train.drop(columns='y')


selector = SelectKBest(score_func=f_classif,k=50)
s_f = selector.fit(data,target)
names = data.columns[selector.get_support()]
test = test[names]
data = data[names]


model = Model(10)

model.fit(data, target)
output = model.predict(test)
submission = pd.DataFrame({
    'id': range(750000, 750000 + len(test)),
    'y': output
})
submission.to_csv('/kaggle/working/submission.csv', index=False)




