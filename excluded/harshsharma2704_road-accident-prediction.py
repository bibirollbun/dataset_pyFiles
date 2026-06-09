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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score,train_test_split
import tensorflow
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Input
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


X = df.drop(columns=['id','accident_risk'])
y = df['accident_risk']
test_id = test['id']
test = test.drop(columns=['id'])


X['road_signs_present'] = X['road_signs_present'].astype(int)
X['public_road'] = X['public_road'].astype(int)
X['holiday'] = X['holiday'].astype(int)
X['school_season'] = X['school_season'].astype(int)


test['road_signs_present'] = test['road_signs_present'].astype(int)
test['public_road'] = test['public_road'].astype(int)
test['holiday'] = test['holiday'].astype(int)
test['school_season'] =test['school_season'].astype(int)

X.head()


# map_light = {'dim':2,'daylight':3,'night':1}
# map_weather = {'clear':3,'rainy':1,'foggy':2}
# map_time_of_day = {'morning':3,'evening':1,'afternoon':2}

# X['time_of_day']=X['time_of_day'].map(map_time_of_day)
# X['weather'] =X['weather'].map(map_weather)
# X['lighting']=X['lighting'].map(map_light)

# test['time_of_day']=test['time_of_day'].map(map_time_of_day)
# test['weather'] =test['weather'].map(map_weather)
# test['lighting']=test['lighting'].map(map_light)
# test


num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=['object','category']).columns


for i in num_cols:
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    sns.boxplot(x=X[i])
    plt.subplot(122)
    sns.kdeplot(x=X[i])


X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42)

num_pipe = Pipeline([
    ('scale',RobustScaler()),

])

cat_pipe = Pipeline([
    ('ohe',OneHotEncoder(handle_unknown='ignore'))
])
preprocess = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

model_lr = Pipeline([
    ('pre',preprocess),
    ('algo',LinearRegression())
])

model_lr.fit(X_train,y_train)
y_pred = model_lr.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_lr.predict(X_train))))
print('validatoin score',np.sqrt(mean_squared_error(y_valid,y_pred)))


model_lgb = Pipeline([
    ('pre',preprocess),
    ('algo',LGBMRegressor(max_depth=6,learning_rate=0.05,n_estimators=250,verbose=-1))
])

model_lgb.fit(X_train,y_train)
y_pred = model_lgb.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_lgb.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


model_dt = Pipeline([
    ('pre',preprocess),
    ('algo',DecisionTreeRegressor(max_depth=4,min_samples_leaf=40,min_samples_split=20))
])

model_dt.fit(X_train,y_train)
y_pred = model_dt.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_dt.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


model_xgb = Pipeline([
    ('pre',preprocess),
    ('algo',XGBRegressor(learning_rate=0.1,max_depth=450,reg_alpha=4,subsample=0.7,colsample_bytree=0.6,reg_lambda=4))
])

model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_xgb.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


importances = model_xgb.named_steps['algo'].feature_importances_
df_new = pd.DataFrame({
    'Features':model_xgb.named_steps['pre'].get_feature_names_out(),
    'importance':importances
}).sort_values(by='importance',ascending=False)
sns.barplot(y='Features',x='importance',data=df_new)


model_cat = Pipeline([
    ('pre',preprocess),
    ('algo',CatBoostRegressor(learning_rate=0.1,depth=4,n_estimators=750,verbose=False))
])

model_cat.fit(X_train,y_train)
y_pred = model_cat.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_cat.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


importances = model_cat.named_steps['algo'].feature_importances_
df_new = pd.DataFrame({
    'Features':model_cat.named_steps['pre'].get_feature_names_out(),
    'importance':importances
}).sort_values(by='importance',ascending=False)
sns.barplot(y='Features',x='importance',data=df_new)


model_rf = Pipeline([
    ('pre',preprocess),
    ('algo',RandomForestRegressor(n_estimators=350,max_depth=4,oob_score=True,bootstrap=True,n_jobs=-1))
])

model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_rf.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


model_rf.named_steps['algo'].oob_score_


model_gbd = Pipeline([
    ('pre',preprocess),
    ('algo',GradientBoostingRegressor(learning_rate=0.01,n_estimators=250,loss='huber'))
])

model_gbd.fit(X_train,y_train)
y_pred = model_gbd.predict(X_valid)
print('training score',np.sqrt(mean_squared_error(y_train,model_gbd.predict(X_train))))
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred)))


X_train_pre = preprocess.fit_transform(X_train)
X_valid_pre = preprocess.transform(X_valid)
shape =len(preprocess.get_feature_names_out())

model = Sequential([
    Input(shape=(shape,)),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1,activation='linear')
])

model.summary()


model.compile(loss='mse',optimizer='AdamW',metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train_pre,y_train,epochs=15,validation_split=0.2)


y_pred = model.predict(X_valid_pre)
print('Training score ',np.sqrt(mean_squared_error(y_train,model.predict(X_train_pre))))
print('validation score ',np.sqrt(mean_squared_error(y_valid,y_pred)))



test_pre= preprocess.transform(test)
test_pred = model.predict(test_pre)
submit = pd.DataFrame({'id':test_id,'accident_risk':test_pred.ravel()})
submit.to_csv('submission.csv',index=False)


result = pd.read_csv('submission.csv')
result.head()




