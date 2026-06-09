
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

import os
import warnings 
warnings.filterwarnings('ignore')


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
train_dataset.head()


test_dataset.shape, train_dataset.shape


train_dataset.info()


test_dataset.info()


test_dataset.loc[2707,'winddirection']=70.0


train_dataset.describe()


## Distribution of rainfall 
rainfall = train_dataset.rainfall.value_counts()
plt.title('Rainfall Distribution')
sns.barplot( x= rainfall.index, y= rainfall)
plt.show()


train_dataset.columns


cols = ['pressure','temparature', 'dewpoint','humidity', 'cloud', 'sunshine', 'windspeed' ]
for col in cols:
    fig, axs=plt.subplots(1,2, figsize=(12, 5))
    axs[0].set_title(f'{col} Distribution')
    sns.histplot(data =train_dataset, x= col, kde=True,ax=axs[0])
    axs[1].set_title(f'{col} Distribution with Rainfall')
    sns.histplot(data =train_dataset, x= col, kde=True, hue='rainfall', ax=axs[1])
plt.show()


fig, axs=plt.subplots(1,2, figsize=(12, 5))
axs[0].set_title('Pressure Distribution')
sns.histplot(data =train_dataset, x= 'pressure', kde=True,ax=axs[0])
axs[1].set_title('Pressure Distribution with Rainfall')
sns.histplot(data =train_dataset, x= 'pressure', kde=True, hue='rainfall', ax=axs[1])
plt.show()


## Scatter plot 
#1- Relation between Temparature and pressure 
plt.title ('Relation between Temparature and pressure')
sns.scatterplot(train_dataset, x= 'temparature', y='pressure')
plt.show()


## Scatter plot 
#2- Relation between humatity and cloud 
plt.title ('Relation between Temparature and cloud')
sns.scatterplot(train_dataset, x= 'cloud', y='temparature', hue ='rainfall' )
plt.show()


## Scatter plot 
#3- Relation between sunshine and cloud 
plt.title ('Relation between sunshine and cloud')
sns.scatterplot(train_dataset, x= 'cloud', y='sunshine')
plt.show()


corr = train_dataset[cols].corr()
sns.heatmap(corr, annot=True)
plt.show()


# Box plot 
sns.boxplot(train_dataset, x= 'rainfall', y='temparature')


sns.violinplot(train_dataset, x='rainfall', y= 'cloud')


# trend data over the time 
trend_data = train_dataset[['pressure', 'temparature', 'humidity']]
trend_data.pressure = trend_data.pressure/1000
start_date = pd.to_datetime('01/01/2019')
date = pd.date_range(start=start_date, periods=len(trend_data), freq='D')
trend_data.set_index(date, inplace=True, drop=True)
trend_data.plot(figsize=(12,6))
plt.title("Trend of pressure,temparature, humidity ")
plt.show()



test_dataset.winddirection.isna().sum()


def add_feature(data):
    data['diff_temp'] = data.maxtemp - data.mintemp
    data['sin_day'] = np.sin(2*np.pi*data.day/365)
    data['cos_day'] = np.cos(2*np.pi*data.day/365)
    data['sin_angle'] = np.sin(data.winddirection)
    data['cos_angle'] = np.cos(data.winddirection)
    data['temp_mul_down'] = data.temparature * data.dewpoint
    return data


train_dataset = add_feature(train_dataset)
test_dataset = add_feature(test_dataset)
test_dataset.shape, train_dataset.shape


all_data = pd.concat([train_dataset,test_dataset], axis=0)
all_data['day_1'] = all_data.rainfall.shift(1)
all_data['day_2'] = all_data.rainfall.shift(2)
all_data['day_3'] = all_data.rainfall.shift(3)
all_data['day_4'] = all_data.rainfall.shift(4)
all_data['day_5'] = all_data.rainfall.shift(5)
all_data['day_6'] = all_data.rainfall.shift(6)
all_data['day_7'] = all_data.rainfall.shift(7)


all_data= all_data.iloc[7:,:]
all_data


train_data=all_data.iloc[:-730,:]
train_data


test_data = all_data.iloc[-737:,:]
test_data


X,y = train_data.drop(['day', 'rainfall'], axis=1),train_data.rainfall
X.shape, y.shape


scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
X


train_x, val_x,train_y, val_y = train_test_split(X,y,test_size= 0.2, random_state=1)
train_x.shape, train_y.shape,val_x.shape, val_y.shape 


rf_param={'n_estimators': 338, 'max_depth': 11, 'min_samples_split': 4, 'min_samples_leaf': 1}
rf_model = RandomForestClassifier(**rf_param)


rf_model.fit(train_x, train_y)


rf_predict=rf_model.predict(val_x)
accuracy = accuracy_score(val_y, rf_predict)
print(f'Validation Accuracy: {accuracy:.4f}')



def optimizer(trial):
    param={
        'n_estimators':trial.suggest_int('n_estimators', 50,2000),
        'max_depth':trial.suggest_int('max_depth', 3,20),
        'min_samples_split':trial.suggest_int('min_samples_split', 2,20),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1,20),
        'max_features': trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
        
    }
    rf_opt = RandomForestClassifier(**param, random_state=1 )
    cv= cross_val_score(rf_opt,X, y,scoring='roc_auc')
    return np.mean(cv)


rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(optimizer, n_trials=100, timeout=1000)

# Best parameters
print("Best Hyperparameters:", rf_study.best_params)


rf_model = RandomForestClassifier(**rf_study.best_params,random_state=1 )
rf_model.fit(train_x, train_y)


rf_predict=rf_model.predict(val_x)
accuracy = accuracy_score(val_y, rf_predict)
print(f'Validation Accuracy: {accuracy:.4f}')


xgb_param = {'n_estimators': 500, 'learning_rate': 0.1,
         'max_depth': 6, 'subsample': 0.5, 'colsample_bytree': 0.5}
xgb = XGBClassifier(**xgb_param,eval_metric='logloss',random_state=1)


xgb.fit(train_x, train_y, eval_set=[(train_x, train_y),(val_x,val_y)])


y_predict = xgb.predict(val_x)
y_predict


accuracy = accuracy_score(val_y, y_predict)
print(f'Validation Accuracy: {accuracy:.4f}')



def optimizer(trial):
    param={
        'n_estimators':trial.suggest_int('n_estimators', 300,2000),
        'max_depth':trial.suggest_int('max_depth', 3,15),
        'min_child_weight':trial.suggest_int('min_child_weight', 2,15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1),
        'gamma': trial.suggest_float("gamma", 1e-4, 1.0),
        "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
        "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
        "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1)
    }
    xgb_opt = XGBClassifier(**param,eval_metric='logloss',use_label_encoder=False,random_state=1 )
    cv= cross_val_score(xgb_opt,X, y,scoring='roc_auc')
    return np.mean(cv)


xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(optimizer, n_trials=100, timeout=1000)

# Best parameters
print("Best Hyperparameters:", xgb_study.best_params)


xgb_model = XGBClassifier(**xgb_study.best_params,eval_metric='logloss')


xgb_model.fit(train_x, train_y, eval_set=[(train_x, train_y),(val_x,val_y)])


y_predict = xgb_model.predict(val_x)
y_predict


accuracy = accuracy_score(val_y, y_predict)
print(f'Validation Accuracy: {accuracy:.4f}')


model = Sequential([
    Input(shape=(23,)),
    Dense(512,activation='relu'),
    Dropout(0.5),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(32,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.summary()


model.fit(train_x, train_y, epochs=50, batch_size=128,validation_data=(val_x,val_y))





def pre_data(data):
    data['day_1'] = data.rainfall.shift(1)
    data['day_2'] = data.rainfall.shift(2)
    data['day_3'] = data.rainfall.shift(3)
    data['day_4'] = data.rainfall.shift(4)
    data['day_5'] = data.rainfall.shift(5)
    data['day_6'] = data.rainfall.shift(6)
    data['day_7'] = data.rainfall.shift(7)
    return data


test_data.columns


test_data.iloc[6:,11]


for i in range(7,len(test_data)):
    test_data = pre_data(test_data)
    row = test_data.drop(['day','rainfall'],axis=1).iloc[i]
    row = scaler.transform(row.values.reshape(1,-1))
    test_data.iloc[i,11] = np.round(model.predict(row),0)


submission.rainfall = test_data.iloc[-730:,11].values
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission




