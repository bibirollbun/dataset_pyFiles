


import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold



import sklearn
sklearn.metrics.get_scorer_names() 


def RMSLE (y_true,y_pred):
    return np.sqrt(rmse(y_true,y_pred))


data=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col=0)
data_submission=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col=0)
print(f'Columns: \n {data.columns.values}')




Y_set_for_training=np.log1p(data['Calories'])
data.drop(columns=['Calories'],inplace=True)

print(data['Sex'].value_counts())
data['Sex']=data['Sex'].map({'male': 0, 'female': 1})
data_submission['Sex']=data_submission['Sex'].map({'male': 0, 'female': 1})

print(data.head(5))


# for column in data.columns.values:
#     plt.figure()
#     plt.hist(data[column])
#     plt.show()
#     plt.suptitle(f'{column}')



data_indexes=data.index.values
concatenate_train_test=pd.concat([data,data_submission])
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(concatenate_train_test)
data_normalized = pd.DataFrame(x_scaled,columns=data.columns.values)




set_for_training=data_normalized.iloc[data_indexes,:]

set_for_submission=data_normalized.iloc[np.max(data_indexes)+1:,:]


x_train, x_test, y_train, y_test = train_test_split(
    set_for_training, Y_set_for_training, test_size=0.3, random_state=42)





print(f'Data normalized and splitted into train and test datasets. \n '
      f'Train dataset size: {x_train.shape[0]}, test dataset size:{x_test.shape[0]}')



sets=[x_train,x_test,y_train,y_test]
[print(len(i)) for i in sets]


for column in x_train.columns.values:
    plt.figure()
    plt.hist(x_train[column])
    plt.suptitle(f'{column}')
    plt.show()
    




# sns.pairplot(x_train)
# plt.show()



# from sklearn.metrics import root_mean_squared_log_error as rmsle
from sklearn.metrics import mean_squared_log_error as rmse

# model1 = XGBRegressor(objective='reg:absoluteerror')

import lightgbm as lgb
from catboost import CatBoostRegressor

model1 = CatBoostRegressor(loss_function='RMSE',verbose=0)




best_ratios=0

RMSLE_train_arr,RMSLE_test_arr=[],[]
kf = KFold(n_splits=10)

best_score=100
for i,(train_index, test_index) in enumerate(kf.split(set_for_training)):
    
    x_fold_train, x_fold_test = set_for_training.iloc[train_index],set_for_training.iloc[test_index]
    y_fold_train, y_fold_test = Y_set_for_training.iloc[train_index], Y_set_for_training.iloc[test_index]


    # distributions = {
    # # 'n_estimators': [10,50],
    # 'max_depth': [9,15,20,30] ,
    #                  # 'lambda':[0.1,0.5,1],
    #                        # 'alpha':[0.1,0.5,1],
    #                        #  'eta':[0.3,0.1,0.5,0.8],
    # 'reg_lambda': [0,0.1,0.3,0.5,0.7,1],
    # 'reg_alpha': [0,0.1,0.3,0.5,0.7,0.9,1],
    # 'learning_rate': [0.1, 0.15, 0.20,0.5],
    # 'min_child_weight': [1, 2, 4,10,20,30],
    # 'colsample_bytree':[0,0.1,0.3,0.5,0.7,1],
    # 'colsample_bylevel':[0,0.1,0.3,0.5,0.7,1],
    # 'colsample_bynode':[0,0.1,0.3,0.5,0.7,1],
    # }
    param_dist = {
    'iterations': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.2, 0.5, 0.8],
    'border_count': [32, 64, 128],
    }

    
    clf = RandomizedSearchCV(model1, param_dist,
         verbose=2,scoring='neg_root_mean_squared_error',cv=2)
    search = clf.fit(x_fold_train, y_fold_train)
    
    model=search.best_estimator_
   # model=model1.fit(x_fold_train, y_fold_train)
    print('Search is finished. Best parameters: \n')
    # print(search.best_params_)
    # print(best_params)
            
    y_pred = model.predict(x_fold_train)
    RMSLE_train = RMSLE(y_fold_train,y_pred)

    y_pred = model.predict(x_fold_test)
    RMSLE_test = RMSLE(y_fold_test,y_pred)

    RMSLE_train_arr.append(RMSLE_train)
    RMSLE_test_arr.append(RMSLE_test)

    print(f'\n FOLD-{i}. MAPE train: {RMSLE_train}, MAPE test: {RMSLE_test} \n')

    
    if RMSLE_test<best_score:
        best_score=RMSLE_test
        best_model=model
        print(f' FOLD-{i}. Model has best score. ')


model1=best_model






predict_test=model1.predict(x_test)
print(len(predict_test),len(y_test))
predict_test=[0 if element <0 else element for element in predict_test  ]
print(np.min(predict_test))
print(np.min(y_test))

rmse_test=RMSLE(y_test,predict_test)
print(f'RMSLE for Test data: {rmse_test}')



# import lightgbm as lgb
# from catboost import CatBoostRegressor

# model = CatBoostRegressor(loss_function='RMSE')

# # Fit the model on the training data with verbose logging every 100 iterations
# model.fit(x_train, y_train, verbose=100)




# print(y_test)
# result_catboost=model.predict(x_test)



# result_catboost=[0 if element <0 else element for element in result_catboost  ]
# print(np.min(result_catboost))
# print(np.min(y_test))

# rmse_test=RMSLE(y_test,result_catboost)
# print(f'RMSLE for Test data: {rmse_test}')






submission_answer=model1.predict(set_for_submission)
print(np.min(set_for_submission))
submission=pd.concat([pd.Series(set_for_submission.index.values,name='id'),pd.Series(np.expm1(submission_answer),name='Calories')],names=['id','Calories'],axis=1)
print(submission)
submission.to_csv('submission.csv',index=False)

