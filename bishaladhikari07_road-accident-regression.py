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


df_train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_train.head()


df_train.shape


df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_test.head()


df_test.shape


df=pd.concat([df_train,df_test],ignore_index=True)
df.shape


df.isna().sum()


df_train.shape


df_train.isna().sum()


df_test.isna().sum()


df.head()


df['num_lanes'].value_counts()



df['holiday'].value_counts()


df.shape
# (517754, 14)
# 690339


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

binaryCol=['road_signs_present','public_road','holiday','school_season']

for col in binaryCol:
    df[col]=df[col].map({True:1,False:0})



df.head()


df_train=df[:517754]
df_test=df[517754:]


df_train_x=df_train.drop(['accident_risk'],axis=1)
# df_train_x.head()
df_train_y=df_train['accident_risk']
df_train_y.head()


df_test_x=df_test.drop(['accident_risk'],axis=1)
df_test_x.head()





# speed_limit,,road_type,lighting,weather,time_of_day --categorical


# curvature,num_reported_accidents,num_lanes --numerical


# road_signs_present,public_road,holiday,school_season -binary





numerical_col=['curvature','num_reported_accidents','num_lanes']

categorical_features=["speed_limit","road_type","lighting","weather","time_of_day"]

preprocessor=ColumnTransformer([
    ('cat',OneHotEncoder(),categorical_features),
    ('num',StandardScaler(),numerical_col)
],remainder='passthrough')



tree_models={
    'RFMODEL':Pipeline([
        ('preprocessor',preprocessor),
        ('model',RandomForestRegressor())
    ]),
    'XGBModel':Pipeline([
        ('preprocessor',preprocessor),
        ('model',XGBRegressor())
    ]),
    'DecisionTree':Pipeline([
        ('preprocessor',preprocessor),
        ('model', tree.DecisionTreeRegressor())
    ]),
        'KneighborModel':Pipeline([
        ('preprocessor',preprocessor),
        ('model', KNeighborsRegressor())
    ])
}





df_test.head()


df.info()


from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict


X_train,X_test,y_train,y_test=train_test_split(df_train_x,df_train_y,test_size=0.2,random_state=42)




# from sklearn.metrics import mean_squared_error, r2_score

# for name, model_pipeline in tree_models.items():
    
#     print(f"\n=== {name} ===")
    
#      # Fit the pipeline on training data
#     model_pipeline.fit(X_train, y_train)
    
# #     # Predictions
#     y_pred = model_pipeline.predict(X_test)
    

#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print("Test MSE:", mse)
#     print("Test R2:", r2)
    
    
#      # Optional: cross-validation score for more robust metric
#     scores = cross_val_score(model_pipeline, df_train_x, df_train_y, cv=5, scoring='r2')
#     print("5-fold CV Accuracy:", scores.mean())



from sklearn.linear_model import LinearRegression,Ridge,BayesianRidge

non_tree_models={
    'LRMODEL':Pipeline([
        ('preprocessor',preprocessor),
        ('model',LinearRegression())
    ]),
    'RIDGEModel':Pipeline([
        ('preprocessor',preprocessor),
        ('model',Ridge())
    ]),
    'BRModel':Pipeline([
        ('preprocessor',preprocessor),
        ('model',BayesianRidge())
    ])
 
}


# for name,model_pipeline in non_tree_models.items():
#     print(f'model:{name}')
#     NTModel=model_pipeline.fit(X_train,y_train)
#     y_preds=NTModel.predict(X_test)
#     print('noremal score',NTModel.score(X_test,y_test))
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print("Test MSE:", mse)
#     print("Test R2:", r2)
    
    
#      # Optional: cross-validation score for more robust metric
#     scores = cross_val_score(model_pipeline, df_train_x, df_train_y, cv=5, scoring='r2')
#     print("5-fold CV Accuracy:", scores.mean())


from sklearn.model_selection import GridSearchCV

params_grid={
    'model__n_estimators':[100,200,300],
    'model__max_depth':[3,5,7],
    'model__learning_rate':[0.01,0.1,0.3],
    'model__subsample':[0.8,1.0],
    'model__colsample_bytree':[0.8,1.0]
}

grid_search=GridSearchCV(tree_models['XGBModel'],params_grid,cv=3,scoring='r2',n_jobs=-1)

grid_search.fit(X_train,y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


finalModel=grid_search.best_estimator_
finalModel.fit(X_train,y_train)


finalModel.score(X_test,y_test)


y_pred=finalModel.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test MSE:", mse)
print("Test R2:", r2)


submission=pd.DataFrame({
    "id":df_test['id'],
    "accident_risk":finalModel.predict(df_test_x)
})
print(submission)
submission.to_csv('submission.csv',index=False)
submission.head(2)





