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


from zipfile  import ZipFile

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns


path_data = '/kaggle/input/flight-delays-fall-2018'
path_data_pred = '/kaggle/flight-delay-prediction'
data_trein_name = 'flight_delays_train.csv'
data_test_name =  'flight_delays_test.csv'
sample_sub_name = 'sample_submission.csv'


with ZipFile(f'{path_data}/{data_trein_name}.zip',"r") as z:
    z.extractall(path_data_pred)
    
with ZipFile(f'{path_data}/{data_test_name}.zip',"r") as z:
    z.extractall(path_data_pred)
with ZipFile(f'{path_data}/{sample_sub_name}.zip',"r") as z:
    z.extractall(path_data_pred)


df_train=  pd.read_csv(f'{path_data_pred}/{data_trein_name}') 
df_test =  pd.read_csv(f'{path_data_pred}/{data_test_name}') 


le = LabelEncoder()
seasons = {
    1: 'Winter',
    2: 'Spring',
    3: 'Summer',
    4: 'Autumn'
}
MonthName = {
    1:  'January',
    2:  'February',
    3:  'March', 
    4:  'April',
    5:  'May',
    6:  'June', 
    7:  'July',
    8:  'August', 
    9:  'September', 
    10: 'October', 
    11: 'November',
    12: 'December' 
}
WeekName={
    1:  'Monday', 
    2:  'Tuesday', 
    3:  'Wednesday', 
    4:  'Thursday', 
    5:  'Friday', 
    6:  'Saturday', 
    7:  'Sunday' 
}
DistanceName= {
    0:  'Distance<500',
    1:  'Distance>500<1000',
    2:  'Distance>1000<1500',
    3:  'Distance>1500<2000',
    4:  'Distance>2000<2500',
    5:  'Distance>2500<3000',
    6:  'Distance>3000',
    7:  'Distance>3000',
    8:  'Distance>3000',
    9:  'Distance>3000',
    10: 'Distance>3000'

}
Time_Of_Day = {
    0:  'Night',
    1:  'Night', 
    2:  'Night', 
    3:  'Night', 
    4:  'Night', 
    5:  'Night',
    6:  'Morning',
    7:  'Morning',
    8:  'Morning',
    9:  'Morning',
    10: 'Morning',
    11: 'Morning',
    12: 'Afternoon',
    13: 'Afternoon',
    14: 'Afternoon',
    15: 'Afternoon',
    16: 'Afternoon',
    17: 'Afternoon',
    18: 'Evening',
    19: 'Evening',
    20: 'Evening',
    21: 'Evening',
    22: 'Evening',
    23: 'Evening'
   
}


def preprocess_data(data):
    data['FromTo']      = data['Origin'] + '-->' + data['Dest']
    data['Season']      = data['Month'].str[2:].astype(int).apply(lambda x: x %12 //3+1).map(seasons)
    data['Month']       = data['Month'].str[2:].astype(int).map(MonthName)
    data['DayOfWeek']   = data['DayOfWeek'].str[2:].astype(int).map(WeekName)
    data['DepTimeH']    = data['DepTime'].apply(lambda x: x//100).astype(int).apply(lambda x: 0 if x == 24 else (1 if x == 25 else x))
    data['TimeOfDay']   = data['DepTimeH'].map(Time_Of_Day)
    data['DepTimeM']    = data['DepTime'].apply(lambda x: x%100).astype(int)
    data['DayofMonth']  = data['DayofMonth'].str[2:].astype(int)
    #data['Delayed']     = le.fit_transform(data['dep_delayed_15min'])
    data['Weekends']    = data['DayOfWeek'].apply(lambda x: 'weekend' if x in ('Saturday','Sunday') else  ('Friday' if x == 'Friday' else 'weekday')) 
    data['DepTimeH']    = data['DepTimeH'].apply(lambda x: 0 if x == 24 else (1 if x == 25 else x))
    data['DistanceName'] = data['Distance']//500
    data['DistanceName'] = data['DistanceName'].map(DistanceName)
   
    data['UC_Dest']     = data['UniqueCarrier'] +'->' +data['Dest']

    return data


df_train = preprocess_data(df_train)
df_train['Delayed']     = le.fit_transform(df_train['dep_delayed_15min'])
df_test = preprocess_data(df_test)


df_train = df_train.drop(columns=['DepTime','Distance','dep_delayed_15min','Season','Weekends'],axis=1)
df_test = df_test.drop(columns=['DepTime','Distance','Season','Weekends'],axis=1)
categ_feat_idx = np.where(df_train.drop('Delayed', axis=1).dtypes == 'object')[0]
X_train = df_train.drop(columns=['Delayed']).values
y_train = df_train['Delayed'].values
X_test = df_test.values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train,
                                                                test_size=0.2,
                                                                random_state=1)
model = CatBoostClassifier( learning_rate= 0.1, n_estimators = 1000)
model.fit(X_train_part, y_train_part,cat_features=categ_feat_idx)


model_pred_v = model.predict_proba(X_valid)[:, 1]


roc_auc_score(y_valid, model_pred_v)


# model.fit(X_valid, y_valid,cat_features=categ_feat_idx)
model_prob = model.predict_proba(X_test)[:, 1]

sample_sub = pd.read_csv(f'{path_data_pred}/{sample_sub_name}', index_col='id')

sample_sub['dep_delayed_15min'] = model_prob

sample_sub.to_csv(sample_sub_name)

pd.read_csv(f'{path_data_pred}/{sample_sub_name}', index_col='id')

