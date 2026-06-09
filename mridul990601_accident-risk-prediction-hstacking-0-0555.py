# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


id = df_test['id']


df_train.head()


df_train.drop('id',axis = 1,inplace = True)
df_test.drop('id',axis = 1,inplace = True)


df_train.info()


df_train.isnull().sum()


num_cols = df_train.select_dtypes(include = ['int','float']).drop('accident_risk',axis = 1).columns
cat_cols = df_train.select_dtypes(include = ['object']).columns
bool_cols = df_train.select_dtypes(include = ['bool']).columns


for col in bool_cols:
    df_train[col] = df_train[col].astype(int)


fig,axis = plt.subplots(2,2,figsize = (9,8))

axis = axis.flatten()

for ind,cols in enumerate(num_cols):
    axis[ind].boxplot(df_train[cols])
    axis[ind].set_title(cols)


fig,axis = plt.subplots(2,2,figsize = (9,8))

axis = axis.flatten()

for ind,cols in enumerate(num_cols):
    sns.histplot(df_train[cols],kde = True,ax = axis[ind])


df_train.corr(numeric_only = True)


# Feature Concstruction

def add(x):
    x['lanes*speed'] = x['num_lanes']*x['speed_limit']
    x['cur*speed'] = x['curvature']*x['speed_limit']
    x['cur*lanes'] = x['curvature']*x['num_lanes']
    x['acc_perlane'] = x['num_reported_accidents']/x['num_lanes']

    x['high_curvature'] = (x['curvature']>0.7).astype(int)
    x['high_speed'] = (x['speed_limit']>60).astype(int)
    x['acc_prone'] = (x['num_reported_accidents']>=2).astype(int)
    x['few_lanes'] = (x['num_lanes']<=2).astype(int)
    
    x['curvature_squared'] = x['curvature'] ** 2
    x['speed_squared'] = x['speed_limit'] ** 2

    x['speed_per_lane'] = x['speed_limit'] / (x['num_lanes'] + 1)
    x['curvature_speed_ratio'] = x['curvature'] / (x['speed_limit'] + 1)

    return x


num_cols = ['curvature','lanes*speed','cur*speed','cur*lanes','acc_perlane','curvature_squared','speed_squared','speed_per_lane',
           'curvature_speed_ratio']
cat_cols = ['road_type','lighting','weather','time_of_day']
ord_cols = ['speed_limit']


scaler = StandardScaler()
encoder = OneHotEncoder(drop = 'first',handle_unknown = 'ignore',sparse_output = False)
oe = OrdinalEncoder(handle_unknown = 'use_encoded_value',unknown_value = -1)


transformer = FunctionTransformer(func = add,validate = False)

preprocess = ColumnTransformer( 
    transformers = [
        ('scaler',scaler,num_cols),
        ('encoder',encoder,cat_cols),
        ('oe',oe,ord_cols)
    ]
    ,remainder = 'passthrough'
)


'''
def objective(trial):
    classifier_name = trial.suggest_categorical('model',['RandomForest','GradientBoosting'])

    if classifier_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators',50,150)
        max_depth = trial.suggest_int('max_depth',3,20)
        min_samples_split = trial.suggest_int('min_samples_split',5,10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',3,20)
        bootstrap = trial.suggest_categorical('bootstrap',[True,False])

        model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            bootstrap = bootstrap
        )

    elif classifier_name == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators',50,150)
        max_depth = trial.suggest_int('max_depth',3,20)
        learning_rate = trial.suggest_int('learning_rate',0.01,0.1)
        min_samples_split = trial.suggest_int('min_samples_split',5,10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',3,20)

        model = GradientBoostingRegressor(
            
        n_estimators = n_estimators,
        max_depth = max_depth,
        learning_rate = learning_rate,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split
           
        )
    pipeline = Pipeline([
        ('preprocessor',preprocess),
        ('model',model)
    ])

    score = cross_val_score(pipeline,x_train,y_train,cv = 5,scoring = 'neg_mean_absolute_error').mean()
    return score

    study = optuna.create_study(direction = 'maximize')
study.optimize(objective,n_trials = 50)

   ''' 


model1 = XGBRegressor( n_estimators=1500,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1)
model2 = RandomForestRegressor(n_estimators = 100,
                               max_depth = 15,
                               min_samples_leaf = 10,
                               min_samples_split = 5,
                               bootstrap = True)

base_models = [
    ('xgb',model1),
    ('rfg',model2)
]
meta_model = Ridge(alpha = 1.0)
stk_reg = StackingRegressor(estimators = base_models,final_estimator = meta_model,cv = 3)


pipeline = Pipeline([
    ('functr',transformer),
    ('preprocess',preprocess),
    ('stk_reg',stk_reg)
])


x_train = df_train.drop('accident_risk',axis = 1)
y_train = df_train['accident_risk']


pipeline.fit(x_train,y_train)
#score = cross_val_score(pipeline,x_train,y_train,cv = 5,scoring = 'r2').mean()
y_pred = pipeline.predict(df_test)


submission = pd.DataFrame({'id':id,'accident_risk':y_pred})
submission.to_csv('submission.csv',index = False)
print("Submit File Created Succesfully")







