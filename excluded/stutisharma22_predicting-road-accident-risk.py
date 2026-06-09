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


#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


#reading data from the files
df_train= pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test= pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


#resetting the index
df_train= df_train.set_index('id')
df_test= df_test.set_index('id')


#viewing first five rows of the dataset
df_train.head()


#number of columns and data types in the train dataset
df_train.info()


df_train.shape, df_test.shape


#summary statistics of numerical columns in train dataset
df_train.describe()


#number of unique values in the columns
df_train.nunique()


#plotting histogram for all the input variables
df_train.hist(figsize=(16,10), bins=50, color='blue', edgecolor='black')
plt.suptitle("Histogram for all numerical variables")
plt.show()


numerical_variables= df_train[['curvature', 'accident_risk','num_reported_accidents']]
correlation_matrix= numerical_variables.corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


sns.boxplot(data= df_train,x='curvature' )
plt.show()


sns.boxplot(data= df_train,x='num_reported_accidents' )
plt.show()


df_train['log_curvature'] = np.log1p(df_train['curvature'])
df_test['log_curvature'] = np.log1p(df_test['curvature'])
sns.histplot(data=df_train, x='log_curvature')
plt.show()


#Creating feature interactions
df_train['speed_curvature']= df_train['speed_limit'] * df_train['curvature']
df_train['lanes_is_public']= df_train['num_lanes'] * df_train['public_road']==True
df_train['is_night_rainy']= (df_train['time_of_day'] == 'night') * (df_train['weather']=='rainy')
df_train['accidents_per_lane']= df_train['num_reported_accidents'] / df_train['num_lanes']
df_train['accidents_per_speed_limit']= df_train['num_reported_accidents'] / df_train['speed_limit']
df_train['road_weather'] = df_train['road_type'].astype('str')+ '_' + df_train['weather'].astype('str')
df_train['curvature_bin'] = pd.qcut(df_train['curvature'], q= 10, duplicates='drop')
df_train['speed_bin'] = pd.qcut(df_train['speed_limit'], q= 10, duplicates='drop')
df_train['curvature_squared']= df_train['curvature'] ** 2
df_train['speed_limit_squared']= df_train['speed_limit'] ** 2
df_train['curvature_abs']= np.abs(df_train['curvature'])
df_train['inverse_speed']= np.reciprocal(df_train['speed_limit'] )



df_test['speed_curvature']= df_test['speed_limit'] * df_test['curvature']
df_test['lanes_is_public']= df_test['num_lanes'] * df_test['public_road']==True
df_test['is_night_rainy']= (df_test['time_of_day'] == 'night') * (df_test['weather']=='rainy')
df_test['accidents_per_lane']= df_test['num_reported_accidents'] / df_test['num_lanes']
df_test['accidents_per_speed_limit']= df_test['num_reported_accidents'] / df_test['speed_limit']
df_test['road_weather'] = df_test['road_type'].astype('str')+ '_' + df_test['weather'].astype('str')
df_test['curvature_bin'] = pd.qcut(df_test['curvature'], q= 10, duplicates='drop')
df_test['speed_bin'] = pd.qcut(df_test['speed_limit'], q= 10, duplicates='drop')
df_test['curvature_squared']= df_test['curvature'] ** 2
df_test['speed_limit_squared']= df_test['speed_limit'] ** 2
df_test['curvature_abs']= np.abs(df_test['curvature'])
df_test['inverse_speed']= np.reciprocal(df_test['speed_limit'] )
df_test.head()


#categorical variables in the dataset
categorical_variables= df_train.select_dtypes(include=['object','category']).columns.tolist()
categorical_variables


#setting up the KFold
kf= KFold(n_splits= 5, shuffle= True, random_state= 42)  


#Encoding categorical variables using target encoding
def target_encode(train, test, column, target_col, n_splits= 5, smooth= 10):
    global_mean= train[target_col].mean()
    train_encoded= np.zeros(train.shape[0])
    for tr_idx, val_idx in kf.split(train):
        tr, val= train.iloc[tr_idx], train.iloc[val_idx]
        means= tr.groupby(column)[target_col].mean()
        counts= tr.groupby(column)[target_col].count()
        means_smooth= (means * counts + global_mean * smooth) / (counts + smooth)
        train_encoded[val_idx]= val[column].map(means_smooth)

        train_encoded= pd.Series(train_encoded).fillna(global_mean)
        means= train.groupby(column)[target_col].mean()
        counts= train.groupby(column)[target_col].count()
        means_smooth= (means * counts + global_mean * smooth) / (counts + smooth)
        test_encoded= test[column].map(means_smooth).fillna(global_mean)

        return train_encoded, test_encoded


for col in categorical_variables:
    df_train[col]= df_train[col].astype(str)
    df_test[col]= df_test[col].astype(str)
    df_train[f'{col}_te'], df_test[f'{col}_te'] = target_encode(df_train, df_test, column= col, target_col='accident_risk')


df_train=df_train.drop(['curvature_bin', 'speed_bin'], axis=1)
df_test=df_test.drop(['curvature_bin', 'speed_bin'], axis=1)


train_encoded= pd.get_dummies(df_train, columns= ['road_type', 'lighting', 'weather', 'time_of_day', 'road_weather'])
test_encoded= pd.get_dummies(df_test, columns= ['road_type', 'lighting', 'weather', 'time_of_day', 'road_weather'])
#replacing True and False with 1 and 0
train_encoded=train_encoded.replace({True: 1, False: 0})
test_encoded=test_encoded.replace({True: 1, False: 0})
train_encoded.head().T


X= train_encoded.drop('accident_risk', axis=1)
y= train_encoded['accident_risk']


#Initializing storage
oof_lgb= np.zeros(len(X))
oof_xgb= np.zeros(len(X))
oof_cat= np.zeros(len(X))

preds_lgb= np.zeros(len(test_encoded))
preds_xgb= np.zeros(len(test_encoded))
preds_cat= np.zeros(len(test_encoded))


#models with parameters
model1= lgb.LGBMRegressor(objective='regression', random_state=42, learning_rate=0.03, n_estimators=1000, num_leaves=64, max_depth=-1, 
                          subsample= 0.8,colsample_bytree=0.8 )
model2= xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, 
                         eval_metric='rmse', tree_method= 'hist', random_state=42)
model3= cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_seed= 42, verbose= False)



#training and predicting
for fold, (train_idx, valid_idx) in enumerate(kf.split(X,y)):
    print(f"\n--------Fold{fold+1}---------")
    X_train, X_valid= X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid= y.iloc[train_idx], y.iloc[valid_idx]

    model1.fit(X_train,y_train, eval_set=[(X_valid, y_valid)], eval_metric= 'rmse')
    oof_lgb[valid_idx]= model1.predict(X_valid)
    preds_lgb += model1.predict(test_encoded) / kf.n_splits

    model2.fit(X_train,y_train)
    oof_xgb[valid_idx]= model2.predict(X_valid)
    preds_xgb += model2.predict(test_encoded) / kf.n_splits

    model3.fit(X_train,y_train)
    oof_cat[valid_idx]= model3.predict(X_valid)
    preds_cat += model3.predict(test_encoded) / kf.n_splits



importance = model1.booster_.feature_importance(importance_type='gain')
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


X_stack= np.vstack([oof_lgb, oof_xgb, oof_cat]).T


def rmse_loss(weights, alpha= 0.01):
    blended= np.dot(X_stack, weights)
    return mean_squared_error(y, blended, squared= False) + alpha * np.sum(weights ** 2)


init_weights= [1/3, 1/3, 1/3]
constraints= {'type': 'eq', 'fun': lambda w: 1-sum(w)}
bounds= [(0,1) for _ in range(X_stack.shape[1])]
result= minimize(rmse_loss, init_weights, method= 'SLSQP', bounds= bounds, constraints= constraints)
best_weights= result.x
print('optimized weights: ', best_weights)


#calculating rmse for the optimized blend for oof stack
oof_blend_optimized= np.dot(X_stack, best_weights)
rmse_optimized= mean_squared_error(y, oof_blend_optimized, squared= False)
print("Optimized OOF RMSE: ", rmse_optimized)


#calculating rmse for the optimized blend for the test stack
preds_stack= np.vstack([preds_lgb, preds_xgb, preds_cat]).T
test_blend_optimized= np.dot(preds_stack, best_weights)


#saving the submission file
output= pd.DataFrame({'id':df_test.index, 'accident_risk': test_blend_optimized})
output.to_csv('submission.csv', index= False)

