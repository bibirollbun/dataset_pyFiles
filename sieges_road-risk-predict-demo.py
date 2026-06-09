# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train.head(5)


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
#test.head(5)


#NAN handle

#print(len(train),train['road_type'].isna().sum())


#data processing
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 500)
def data_process(data,is_train= True):
    data = data.copy()
    #print(data.head(5),len(data))
    ##print(data['road_signs_present'].head(5),'prev')

    #data['road_type'] = data['road_type'].map({'urban':0,'rural':1,'highway':2})
    #data['lighting'] = data['lighting'].map({'rainy':0,'clear':1,'foggy':2})
    # 同时对 'Embarked' 和 'Sex' 进行独热编码
    data = pd.get_dummies(data, columns=['road_type', 'lighting','weather','time_of_day'], prefix=['road_type', 'lighting','weather','time_of_day'])
    data[['road_type_highway','road_type_rural','road_type_urban','lighting_daylight','lighting_dim','lighting_night','weather_clear','weather_foggy','weather_rainy','time_of_day_afternoon','time_of_day_evening','time_of_day_morning']] = data[['road_type_highway','road_type_rural','road_type_urban','lighting_daylight','lighting_dim','lighting_night',
          'weather_clear','weather_foggy','weather_rainy','time_of_day_afternoon','time_of_day_evening','time_of_day_morning']].astype(int)
    
    ##data['road_signs_present'] = data['road_signs_present'].map({'False':0,'True':1})
    ##print(data['road_signs_present'].head(5),'post')
    ##this is bool data,not str    
    data[['road_signs_present','public_road','holiday','school_season']] = data[['road_signs_present','public_road','holiday','school_season']].astype(int)
    #print(data.head(5),'post')
    #data['public_road'] = data['public_road'].astype(int)
    #data['holiday'] = data['holiday'].astype(int)
    #data['school_season'] = data['school_season'].astype(int)

    #post processed
    #print(data.head(5),len(data))
    #print(data.head())
    features = ['num_lanes','speed_limit','curvature','road_signs_present','public_road','holiday','school_season',
                'num_reported_accidents','road_type_highway','road_type_rural','road_type_urban',
                'lighting_daylight','lighting_dim','lighting_night','weather_clear','weather_foggy','weather_rainy',
                'time_of_day_afternoon','time_of_day_evening','time_of_day_morning']
    #'accident_risk',
    
    x = data[features]
    if is_train:
        y = data['accident_risk']
    else:
        y = data['id']
    
    return x,y,features
    
#data_process(train, True)




#train test split
from sklearn.model_selection import train_test_split

def train_test_split_d(x,y):
    x_train,x_val,y_train,y_val = train_test_split(
        x,
        y,
        test_size = 0.3,
        random_state = 1337,
        stratify = y
        
    )
    #print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)

    return x_train,x_val,y_train,y_val

#train_test_split_d(train,)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
#cross val score
from sklearn.model_selection import cross_val_score



#if __name__ == '__main__':
x, y, features = data_process(train, True)
#NaN check post preprocess
# for fea in features:
#     print(len(x),len(y),fea,x[fea].isna().sum())
#print(x.head(5))

x_train, x_val, y_train, y_val = train_test_split_d(x, y)

#y is continuous ,thus not use classifier but regressor
model = RandomForestRegressor(
    n_estimators = 100,
    max_depth = 15,
    min_samples_split = 5,
    max_features = 0.5,#'sqrt',
    random_state = 1337
)


# model = RandomForestRegressor(
#     n_estimators = 100,
#     max_depth = 20,
#     min_samples_split = 2,
#     max_features = 'sqrt',
#     random_state = 1337
# )
#overfit a bit
# prediction r2 score is: 0.8556424035428081
# train r2 score is: 0.9666721703558848
# prediction mean_squared_error score is: 0.0034930836313594262
# train mean_squared_error score is: 0.0008397243240027143
# prediction mean_absolute_error score is: 0.045906916454033456
# train mean_absolute_error score is: 0.02251496199325673
# cv=5 cross_val_score r2 is: [0.87320393 0.87540752 0.87396257 0.87470275 0.87533292]
# cv=5 cross_val_score r2 mean is: 0.8745219393220023
# cv=5 cross_val_score r2 std is: 0.0008396043040118188
# cv=5 cross_val_score mse is: [0.00351184 0.00344332 0.00350079 0.00346366 0.00345564]
# cv=5 cross_val_score mse mean is: 0.003475048889534828
# cv=5 cross_val_score mse std is: 2.6568736998260008e-05

model.fit(x_train.to_numpy(),y_train.to_numpy())
y_train_predict = model.predict(x_train.to_numpy())
y_val_predict = model.predict(x_val.to_numpy())

#claassification metric
# print(f'prediction accuracy score is: {accuracy_score(y_val_predict,y_val)}')
# print(f'train accuracy score is {accuracy_score(y_train_predict,y_train)}')
# print(f'detail report is {classification_report(y_val_predict,y_val)}')

#regressor metics
print(f'prediction r2 score is: {r2_score(y_val_predict,y_val):.4f}')
print(f'train r2 score is: {r2_score(y_train_predict,y_train):.4f}')
print(f'prediction mean_squared_error score is: {mean_squared_error(y_val_predict,y_val):.4f}')
print(f'train mean_squared_error score is: {mean_squared_error(y_train_predict,y_train):.4f}')
print(f'prediction mean_absolute_error score is: {mean_absolute_error(y_val_predict,y_val):.4f}')
print(f'train mean_absolute_error score is: {mean_absolute_error(y_train_predict,y_train):.4f}')





cross_val_r2 = cross_val_score(
    estimator = model,
    X = x.to_numpy(),
    y = y.to_numpy(),
    cv = 5,
    scoring = 'r2'    
)

cross_val_mse = cross_val_score(
    estimator = model,
    X = x.to_numpy(),
    y = y.to_numpy(),
    cv = 5,
    scoring = 'neg_mean_squared_error'    
)
cross_val_mse = - cross_val_mse



print(f'cv=5 cross_val_score r2 is: {cross_val_r2}')
print(f'cv=5 cross_val_score r2 mean is: {np.mean(cross_val_r2)}')
print(f'cv=5 cross_val_score r2 std is: {np.std(cross_val_r2)}')

print(f'cv=5 cross_val_score mse is: {cross_val_mse}')
print(f'cv=5 cross_val_score mse mean is: {np.mean(cross_val_mse)}')
print(f'cv=5 cross_val_score mse std is: {np.std(cross_val_mse)}')
    
    


#for submission 
x_test,x_id,_ = data_process(test,False)
test_acc_risk = model.predict(x_test.to_numpy())
outputs = pd.DataFrame({
    'id': x_id,
    'accident_risk': test_acc_risk   
})
print(outputs.head(5))
outputs.to_csv('/kaggle/input/playground-series-s5e10/submission.csv',index = False)

