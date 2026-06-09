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


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

all_data = pd.concat([train,test],axis = 0)

print('train:',train.shape)
print('test:',test.shape)
print('all data:',all_data.shape)


all_data.iloc[2185:].head(10)


#prettytable

from prettytable import PrettyTable 
mytable = PrettyTable(["Column", "Data Type","Missing in Train","Missing in Test"]) 

for column in all_data.columns:
    dtype = str(all_data[column].dtype)
    missing_train = f"{train[column].isna().sum()}/{train[column].shape[0]}"
    if column != 'rainfall':
        missing_test = f"{test[column].isna().sum()}/{test[column].shape[0]}"
    else:
        missing_test = '-'
        

    mytable.add_row([column,dtype,missing_train,missing_test])
print(mytable)


print(set(test['winddirection']))


np.mean(test['winddirection'])


#np.mean(all_data['winddirection']) #100
test['winddirection']=test['winddirection'].fillna(100)


test.isna().sum()


train = train.drop('id',axis = 1)
test = test.drop('id',axis =1)
y = train['rainfall']


full_cor=train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
full_cor=full_cor.rename(columns={'level_0':'output','level_1':'features',0:'score'})
full_cor[full_cor['output']== 'rainfall']


y


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def rf_importance(data,y):
    data = data.copy()
    data = data.drop('rainfall',axis =1)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_

    # View as DataFrame
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df.sort_values(by='importance', ascending=False, inplace=True)

    print(importance_df)


rf_importance(train,y)


#Sin and Cos Representation in Day Number:

def geo_of_day(data):
    day = data['day'].copy()
    sin_data = np.sin(2 * np.pi * day/365)
    cos_data = np.cos(2 * np.pi * day/365)
    
    return sin_data,cos_data

train['sin_day'],train['cos_day']=geo_of_day(train)
test['sin_day'],test['cos_day']=geo_of_day(test)



rf_importance(train,y)


train.head()


def add_month_and_season(df):
    import pandas as pd

    # Define seasons
    winter = [12, 1, 2]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    autumn = [9, 10, 11]

    # Convert day to month (approximate method)
    df = df.copy()  # avoid changing the original DataFrame
    df['month'] = ((df['day'] - 1) // 31) + 1

    # Map each month to a season
    def get_season(month):
        if month in winter:
            return 1
        elif month in spring:
            return 2
        elif month in summer:
            return 3
        elif month in autumn:
            return 4
        else:
            return pd.NA  # for safety

    df['season'] = df['month'].apply(get_season).astype('Int64')  # supports NaN

    return df

train=add_month_and_season(train)
test = add_month_and_season(test)


rf_importance(train,y)


#wind direction might affect the could so rainfall maybe. Lets go deeper:

print(set(train['winddirection']))
#lets categorise them by 45degree in 8 categories:
temp = []

def wind_direct_cat(data):

    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = [1,2,3,4,5,6,7,8] #east, upper north easth, lower north east,north, upper north west,lower north west etc,etc,,,,
    data['wind_direct_cat'] = pd.cut(data['winddirection'], bins=bins, labels=labels, right=False)

    return data
    
train = wind_direct_cat(train)   
test = wind_direct_cat(test)


rf_importance(train,y)


full_cor=train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
full_cor=full_cor.rename(columns={'level_0':'output','level_1':'features',0:'score'})
full_cor[full_cor['output']== 'rainfall']


type(train['winddirection'][0])


#wind_vector_x = wind_speed * cos(wind_direction)
#wind_vector_y = wind_speed * sin(wind_direction)

train['wind_vector_x'] = train['windspeed'] * np.sin(np.radians(train['winddirection']))
train['wind_vector_y'] = train['windspeed'] * np.cos(np.radians(train['winddirection']))

test['wind_vector_x'] = test['windspeed'] * np.sin(np.radians(test['winddirection']))
test['wind_vector_y'] = test['windspeed'] * np.cos(np.radians(test['winddirection']))



rf_importance(train,y)


#If wind is strong and pressure is low, the value is high â†’ possibly indicating rain.
#If pressure is high, the value drops, even if wind is strong.

train['wspeed_and_pressure'] = train['windspeed'] / train['pressure']
test['wspeed_and_pressure'] = test['windspeed'] / test['pressure']



rf_importance(train,y)


full_cor=train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
full_cor=full_cor.rename(columns={'level_0':'output','level_1':'features',0:'score'})
full_cor[full_cor['output']== 'rainfall']


# i would expect that cloud dew factor is high when rainfall is high, lets see
train['min_temp_and_dew_diff'] = (train['mintemp'] - train['dewpoint']).abs()
test['min_temp_and_dew_diff'] = (test['mintemp'] - test['dewpoint']).abs()

#train['cloud_dew_factor'] = train['cloud']/(train['min_temp_and_dew_diff']



rf_importance(train,y)


print(min(train['humidity']))
print(min(test['humidity']))


print(min(train['maxtemp']))
print(min(test['maxtemp']))
print(min(train['min_temp_and_dew_diff']))


#lets some interaction btw features:

#lets go with cloud:humidiy ratio
train['cloud_humid_ratio'] = train['cloud']/train['humidity'] #i checked that it never goes to inf.
test['cloud_humid_ratio'] = test['cloud']/test['humidity'] #i checked that it never goes to inf.

train['rolling_temp_and_dew_diff_3d'] = train['min_temp_and_dew_diff'].rolling(window=3, min_periods=1).mean()
train['rolling_pressure_3d'] = train['pressure'].rolling(window=3, min_periods=1).mean()
train['rolling_humid_3d'] = train['humidity'].rolling(window=3, min_periods=1).mean()

train['rolling_temp_and_dew_diff_7d'] = train['min_temp_and_dew_diff'].rolling(window=7, min_periods=1).mean()
train['rolling_pressure_7d'] = train['pressure'].rolling(window=7, min_periods=1).mean()
train['rolling_humid_7d'] = train['humidity'].rolling(window=7, min_periods=1).mean()

test['rolling_temp_and_dew_diff_3d'] = test['min_temp_and_dew_diff'].rolling(window=3, min_periods=1).mean()
test['rolling_pressure_3d'] = test['pressure'].rolling(window=3, min_periods=1).mean()
test['rolling_humid_3d'] = test['humidity'].rolling(window=3, min_periods=1).mean()

test['rolling_temp_and_dew_diff_7d'] = test['min_temp_and_dew_diff'].rolling(window=7, min_periods=1).mean()
test['rolling_pressure_7d'] = test['pressure'].rolling(window=7, min_periods=1).mean()
test['rolling_humid_7d'] = test['humidity'].rolling(window=7, min_periods=1).mean()


rf_importance(train,y)


train['rolling_temp_and_dew_diff_14d'] = train['min_temp_and_dew_diff'].rolling(window=14, min_periods=1).mean()
train['rolling_pressure_14d'] = train['pressure'].rolling(window=14, min_periods=1).mean()
train['rolling_humid_14d'] = train['humidity'].rolling(window=14, min_periods=1).mean()

test['rolling_temp_and_dew_diff_14d'] = test['min_temp_and_dew_diff'].rolling(window=14, min_periods=1).mean()
test['rolling_pressure_14d'] = test['pressure'].rolling(window=14, min_periods=1).mean()
test['rolling_humid_14d'] = test['humidity'].rolling(window=14, min_periods=1).mean()



rf_importance(train,y)


'''
# How quantile works:
import pandas as pd

data = pd.Series([10, 20, 30, 40, 50])

print(data.quantile(0.5))   # Output: 30.0  -> Middle value (median)
print(data.quantile(0.25))  # Output: 20.0  -> 25% of values are below this
print(data.quantile(0.75))  # Output: 40.0  -> 75% of values are below this
print(data.quantile(0.90))  


30.0
20.0
40.0
46.0
'''


#best and worst:
train['extreme_temp'] = (
    (train['temparature']>train['temparature'].quantile(0.90)) |
    (train['temparature']<train['temparature'].quantile(0.10))
    ).astype(int)

train['extreme_press'] = (
    (train['pressure']>train['pressure'].quantile(0.90)) |
    (train['pressure']<train['pressure'].quantile(0.10))
    ).astype(int)

train['extreme_humid'] = (
    (train['humidity']>train['humidity'].quantile(0.90)) |
    (train['humidity']<train['humidity'].quantile(0.10))
    ).astype(int)

#best and worst:
test['extreme_temp'] = (
    (test['temparature']>test['temparature'].quantile(0.90)) |
    (test['temparature']<test['temparature'].quantile(0.10))
    ).astype(int)

test['extreme_press'] = (
    (test['pressure']>test['pressure'].quantile(0.90)) |
    (test['pressure']<test['pressure'].quantile(0.10))
    ).astype(int)

test['extreme_humid'] = (
    (test['humidity']>test['humidity'].quantile(0.90)) |
    (test['humidity']<test['humidity'].quantile(0.10))
    ).astype(int)



train


rf_importance(train,y)


#min_temp_and_dew_diff and humidity interaction
train['min_temp_diff_humid_interact'] = train['min_temp_and_dew_diff']*train['humidity']
test['min_temp_diff_humid_interact'] = test['min_temp_and_dew_diff']*test['humidity']
#min_temp_and_dew_diff and pressure interaction
train['min_temp_diff_pressure_interact'] = train['min_temp_and_dew_diff']*train['pressure']
test['min_temp_diff_pressure_interact'] = test['min_temp_and_dew_diff']*test['pressure']
#cloud and wind_vector_y and windspeed
train['cloud_wind_y_wspeed'] = train['cloud'] * train['wind_vector_y']*train['windspeed']
test['cloud_wind_y_wspeed'] = test['cloud'] * test['wind_vector_y']*test['windspeed']


rf_importance(train,y)


train['pressure_3d_std'] = train['pressure'].rolling(window=3).std().fillna(0)
train['pressure_7d_std'] = train['pressure'].rolling(window=7).std().fillna(0)
train['pressure_14d_std'] = train['pressure'].rolling(window=14).std().fillna(0)

train['w_speed_3d_std'] = train['windspeed'].rolling(window=3).std().fillna(0)
train['w_speed_7d_std'] = train['windspeed'].rolling(window=7).std().fillna(0)
train['w_speed_14d_std'] = train['windspeed'].rolling(window=14).std().fillna(0)

train['humid_3d_std'] = train['humidity'].rolling(window=3).std().fillna(0)
train['humid_7d_std'] = train['humidity'].rolling(window=7).std().fillna(0)
train['humid_14d_std'] = train['humidity'].rolling(window=14).std().fillna(0)

train['cloud_3d_std'] = train['cloud'].rolling(window=3).std().fillna(0)
train['cloud_7d_std'] = train['cloud'].rolling(window=7).std().fillna(0)
train['cloud_14d_std'] = train['cloud'].rolling(window=14).std().fillna(0)



# PRESSURE rolling std
test['pressure_3d_std'] = test['pressure'].rolling(window=3).std().fillna(0)
test['pressure_7d_std'] = test['pressure'].rolling(window=7).std().fillna(0)
test['pressure_14d_std'] = test['pressure'].rolling(window=14).std().fillna(0)

# WINDSPEED rolling std
test['w_speed_3d_std'] = test['windspeed'].rolling(window=3).std().fillna(0)
test['w_speed_7d_std'] = test['windspeed'].rolling(window=7).std().fillna(0)
test['w_speed_14d_std'] = test['windspeed'].rolling(window=14).std().fillna(0)

# HUMIDITY rolling std
test['humid_3d_std'] = test['humidity'].rolling(window=3).std().fillna(0)
test['humid_7d_std'] = test['humidity'].rolling(window=7).std().fillna(0)
test['humid_14d_std'] = test['humidity'].rolling(window=14).std().fillna(0)

# CLOUD rolling std
test['cloud_3d_std'] = test['cloud'].rolling(window=3).std().fillna(0)
test['cloud_7d_std'] = test['cloud'].rolling(window=7).std().fillna(0)
test['cloud_14d_std'] = test['cloud'].rolling(window=14).std().fillna(0)



rf_importance(train,y)


#Control if train and test is the same

cntrl= len(train.columns)
chc = 0
for col in train.columns:
    chc +=1
    if col in test.columns:
        if chc == cntrl:
            print('... equal')
        chc +=1


def normalization_features(data):
    max_val=max(data)
    min_val=min(data)

    divider = max_val-min_val+0.000001
    
    norm_data = list(map(lambda x: (x-min_val)/divider,data))
    return norm_data


train.columns


y = train['rainfall']
X = train.drop('rainfall',axis = 1)
norm_train={}
for each in X.columns:
    if each !='rainfall':
        norm_train[each] = normalization_features(X[each])
    else:
         norm_train[each] = X[each]
norm_train = pd.DataFrame(norm_train)


norm_test={}
for each in test.columns:
    norm_test[each] = normalization_features(test[each])
   
norm_test = pd.DataFrame(norm_test)


norm_train.head(5) 


norm_test


print(norm_test.columns,len(norm_test.columns))
print(norm_train.columns,len(norm_train.columns))


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, callbacks


from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(norm_train, y, test_size=0.3, random_state=42)
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)


import matplotlib.pyplot as plt
import seaborn as sns


#The first feature:
"""
'day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
'sin_day', 'cos_day', 'month', 'season', 'wind_direct_cat',
'wind_vector_x', 'wind_vector_y', 'wspeed_and_pressure',
'min_temp_and_dew_diff' + cloud_sun_ratio
"""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)

importances = model.feature_importances_

# View as DataFrame
import pandas as pd
feature_names = X_train.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df.sort_values(by='importance', ascending=False, inplace=True)
top_20 = importance_df[:20]

sns.barplot(data=top_20.reset_index(), x='importance', y='feature')

# Optionally add a title and labels
plt.title('Bar Plot of Importance by Feature')
plt.xlabel('importance')
plt.ylabel('feature')

plt.show()



from sklearn.model_selection import KFold

from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score 


def check_other_model(X_train,y_train,model_name):

    """
    inputs: Xtrain and ytrain. It is because only to see feature importance. I dont want the model to touch test data yet.
    model_name: xgb, lgb or cat
    """
    xgb_params = {
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
        }
    
    lgb_params = {
            'objective': 'binary',
            'metric': 'logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose':-1
        }
    cb_params = {
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_state': 42,
        }

    if model_name == 'xgb':
        model = XGBClassifier()
    elif model_name == 'lgb':
        model = LGBMClassifier(verbose = -1)
    elif model_name == 'cat':
        model = CatBoostClassifier(verbose =0) #Set silent
    else:
        raise ValueError('Undefined model selected!')
    print(f"Selected model {model_name} is running:")
    print('---------'*10)
       
    feature_names = X_train.columns
    n_splits = 5
    f_score = 0        #each score in fold gonna be stored here.
    f_importance = []   #importances of each feature in fold gonna be stored here.
    
    kf = KFold(n_splits = n_splits,shuffle = True,random_state= 42) # 1 in 5 (%20)

    
    #it is for only seeing the feature importance. 
    for train_index,test_index in kf.split(X_train):
        X_train_fold , X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold , y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        #print(X_train_fold.shape) (1226, 48) (1533/5*4 == 1126)
        #print(X_test_fold.shape) (307, 48)
        #print(y_train_fold.shape) (1226,)
        #print(y_test_fold.shape) (307,)

        model.fit(X_train_fold,y_train_fold)
        #Only see the ROC AUC: -> use model.predict_proba()[:,1] 0-> negative pred, 1-> positive pred
        fold_predc = model.predict_proba(X_test_fold)[:,1]
        #print(fold_predc.shape) #(307,)
        #print(roc_auc_score(y_test_fold,fold_predc))
        f_score += roc_auc_score(y_test_fold,fold_predc)
        f_imp = model.feature_importances_ #send the importance in order of column of x
        f_importance.append(f_imp)
        
    print(f"Average AUC Score:{f_score/n_splits:.2f}")

    oof_feature_importance = np.mean(f_importance,axis=0)
    df_importance = pd.DataFrame({
        'feature':feature_names,
        'importance':oof_feature_importance
        })
    df_importance = df_importance.sort_values(by='importance', ascending=False)
    print(df_importance[:10].head(10))
    return df_importance

xgb_importance = check_other_model(X_train,y_train,'xgb')
lgb_importance = check_other_model(X_train,y_train,'lgb')
cat_importance = check_other_model(X_train,y_train,'cat')



def best_feature_finder(xgb,lgb,cb):
    n = 45 
    common_elements = set(xgb['feature'][:n]) & set(lgb['feature'][:n]) & set(cb['feature'][:n]) #gathering the cross features 
    return common_elements

feature_selection = best_feature_finder(xgb_importance,lgb_importance,cat_importance)


print('feature number:',len(xgb_importance))
print('the feature number to be trained, ',len(feature_selection))


feature_selection


#class balance:
total = y.count()
a=y.value_counts().reset_index()
class_one =a.loc[a['rainfall'] == 1, 'count'].squeeze()
class_zero =a.loc[a['rainfall'] == 0, 'count'].squeeze()

weight_one = total/(2*class_one)
weight_zero = total/(2*class_zero)
print(f'(Class0: {weight_zero:.2f} // Class1: {weight_one:.2f})')


#for XGB:
scale_pos_weight = round(class_zero/class_one,3)
print('scale_pos_weight,',scale_pos_weight)


from sklearn.model_selection import StratifiedKFold
import optuna
from xgboost import XGBClassifier 
from sklearn.metrics import f1_score

def model(lr,max_depth,subsample,colsample_bytree):
    xgb_params = {
        'learning_rate': lr,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'n_estimators': 100,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0
        }
    return XGBClassifier(**xgb_params)
        
def objective(trial):
    
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    threshold = trial.suggest_float("threshold", 0.1, 0.9)

    xgb = model(lr,max_depth,subsample,colsample_bytree)
    
    X_train, X_val, y_train, y_val = train_test_split(norm_train, y, test_size=0.3, random_state=42)
    xgb.fit(X_train, y_train,verbose = False)

    y_prob = xgb.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)

    return f1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)



print("Best params:", study.best_params)
print("Best F1 score:", study.best_value)


# xgb = model(lr,max_depth,subsample,colsample_bytree)
threshold = study.best_params['threshold']

final_model = model(study.best_params['learning_rate'],
          study.best_params['max_depth'],
          study.best_params['subsample'],
          study.best_params['colsample_bytree']
         )

X_train, X_val, y_train, y_val = train_test_split(norm_train, y, test_size=0.3, random_state=42)
    
final_model.fit(X_train, y_train,
               eval_set = [(X_val,y_val)],
               verbose = 0)


y_prob = final_model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
f1 = f1_score(y_val, y_pred)
print('f1 score: ',f1)

y_prob = final_model.predict_proba(norm_test)[:, 1]
#y_pred = (y_prob >= threshold).astype(int) because the prediction is supposed to be probability.

list_of_ids = sub['id'].tolist()

pred_xgb = {'id':sub['id'].tolist(),
     'rainfall':y_prob}

pred_xgb = pd.DataFrame(pred_xgb)

pred_xgb.to_csv('/kaggle/working/wo_blend.csv',index = False)


#let see the results:
pred_xgb.head()


from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import f1_score
optuna.logging.set_verbosity(optuna.logging.INFO)


def objective(trial):
    cboost_params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'iterations': trial.suggest_int("iterations",100,1000, step=100),
        'depth': trial.suggest_int("depth",4,10),
        'loss_function': 'Logloss',
        'eval_metric' : 'F1',
        'random_state': 42,
        'verbose': 100, #report results in every 100
        'scale_pos_weight' :scale_pos_weight, #inform the model about class imbalance,
        'task_type': 'GPU',           # âœ… Enables GPU usage
        'devices': '0'                # optional â€” selects which GPU to use
        }
    
    model = CatBoostClassifier(**cboost_params)

    #my data is imbalance, so i need to split data with sk.fold:
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    f1scores =[]
    for fold, (train_idx,val_idx) in enumerate (skf.split(norm_train,y)):
        ptrain = Pool(data = norm_train.iloc[train_idx], label = y.iloc[train_idx])
        pval = Pool(data = norm_train.iloc[val_idx], label = y.iloc[val_idx])

        model.fit(ptrain, eval_set = pval,early_stopping_rounds=30,verbose = 0)

        y_probs = model.predict_proba(norm_train.iloc[val_idx])[:,1]

        threshold = trial.suggest_float("threshold",0.3,0.8) 
        y_preds = (y_probs >= threshold).astype(int)

        f1 = f1_score(y.iloc[val_idx], y_preds)
        f1scores.append(f1)

    return np.mean(f1scores)

study = optuna.create_study(study_name = 'catboost_',
                            direction="maximize")
study.optimize(objective, n_trials=100)

# â¬‡ï¸� Add this to see best results manually
print(f"\nğŸ�¯ Best F1 Score: {study.best_value:.4f}")
print(f"ğŸ�† Best Parameters: {study.best_params}")


from sklearn.metrics import roc_auc_score


cboost_params = {
       'learning_rate': 0.08661836450838649,
        'iterations': 200,
        'depth': 10,
        'loss_function': 'Logloss',
        'eval_metric' : 'F1',
        'random_state': 42,
        'verbose': 100, #report results in every 100
        'scale_pos_weight' :scale_pos_weight, #inform the model about class imbalance,
        'task_type': 'GPU',           # âœ… Enables GPU usage
        'devices': '0',                # optional â€” selects which GPU to use
        'verbose': 1        
}
threshold = 0.36260351809770935
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
model = CatBoostClassifier(**cboost_params)

test_cross = []
for fold, (train_idx,val_idx) in enumerate (skf.split(norm_train,y)):
    ptrain = Pool(data = norm_train.iloc[train_idx], label = y.iloc[train_idx])
    pval = Pool(data = norm_train.iloc[val_idx], label = y.iloc[val_idx])

    model.fit(ptrain, eval_set = pval,early_stopping_rounds=30,verbose = 0)
    yprobs = model.predict_proba(norm_train.iloc[val_idx])[:, 1]

    y_preds = (yprobs >= threshold).astype(int)

    f1 = f1_score(y.iloc[val_idx], y_preds)
    print(f'Fold {fold} -> F1 Score: {f1}')
    print('-'*30)  
    
   
    test_pred=model.predict_proba(norm_test)[:, 1]
    test_cross.append(test_pred)


pred_cat = np.mean(np.array(test_cross), axis=0)

list_of_ids = sub['id'].tolist()

pred_cboost = {'id':sub['id'].tolist(),
     'rainfall':pred_cat}

pred_cboost = pd.DataFrame(pred_cboost)
pred_cboost.to_csv('/kaggle/working/woBlend_catboost.csv',index = False)


cluster = pd.concat([pred_cboost['rainfall'],pred_xgb['rainfall']],axis = 1)
cluster.columns = ['rainfall_cb','rainfall_xgb']
cluster.head(10)


import tensorflow as tf
global device

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
    device = 'gpu'
else:
    print("GPU is not available")
    device = 'cpu'


import warnings
warnings.filterwarnings("ignore")


xgb_params = {
        'learning_rate': 0.04207338706005672,
        'max_depth': 10,
        'subsample': 0.7190088938381993,
        'colsample_bytree': 0.5753256241976069,
        'n_estimators': 100,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0
        }

model1 = XGBClassifier(**xgb_params)
    
# LightGBM parameters
lgb_params = {
           'n_estimators': 200,
           'max_depth': 10, 
           'min_samples_leaf': 33, 
           'subsample': 0.8144362305468624, 
           'learning_rate': 0.00647777270150904, 
           'lambda_l1': 1.2991459277687692e-05, 
           'lambda_l2': 0.0007304768170358017,
           'objective': 'binary',  # Changed to binary
           'metric': 'auc',  # Changed to binary error
           'boosting_type': 'gbdt',
           'device': device,
           'random_state': 42,
            'eval_metric':'AUC',
           'verbose': -1
}
model2 = LGBMClassifier(**lgb_params)

cboost_params = {
       'learning_rate': 0.08661836450838649,
        'iterations': 200,
        'depth': 10,
        'loss_function': 'Logloss',
        'eval_metric' : 'F1',
        'random_state': 42,
        'verbose': 100, #report results in every 100
        'scale_pos_weight' :scale_pos_weight, #inform the model about class imbalance,
        'task_type': 'GPU',           # âœ… Enables GPU usage
        'devices': '0',                # optional â€” selects which GPU to use
        'verbose': 1        
}

model3 = CatBoostClassifier(**cboost_params)

model_list = [model1,model2,model3]



model_list


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import numpy as np

model_list = [model1, model2, model3]
final_test_preds = []  # This will store test prediction from each trial

def objective(trial):
    # Step 1: Suggest weights and normalize
    w1 = trial.suggest_float("xgb_weight", 0, 1)
    w2 = trial.suggest_float("lgb_weight", 0, 1)
    w3 = trial.suggest_float("cat_weight", 0, 1)

    w_list = np.array([w1, w2, w3])
    w_list = w_list / w_list.sum()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    test_preds_accumulator = []

    for fold_nmb, (train_idx, val_idx) in enumerate(skf.split(norm_train, y)):
        X_train_cv = norm_train.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = norm_train.iloc[val_idx]
        y_val_cv = y.iloc[val_idx]

        fold_val_preds = np.zeros((len(y_val_cv), len(model_list)))
        fold_test_preds = np.zeros((norm_test.shape[0], len(model_list)))

        for i, (model, w) in enumerate(zip(model_list, w_list)):
            # Clone model to avoid overwriting previously trained models
            fresh_model = clone(model)
            if isinstance(model,LGBMClassifier):
            # Fit model
                fresh_model.fit(X_train_cv, y_train_cv,
                            eval_set=[(X_val_cv, y_val_cv)]
                           )
                
            else: #not lgb
                # Fit model
                fresh_model.fit(X_train_cv, y_train_cv, verbose=False,
                            eval_set=[(X_val_cv, y_val_cv)]
                            )

            # Predict on validation and test sets
            fold_val_preds[:, i] = fresh_model.predict_proba(X_val_cv)[:, 1]
            fold_test_preds[:, i] = fresh_model.predict_proba(norm_test)[:, 1]

        # Blend validation predictions
        blended_val = np.dot(fold_val_preds, w_list)
        oof_preds[val_idx] = blended_val

        # Blend test predictions for this fold
        blended_test = np.dot(fold_test_preds, w_list)
        test_preds_accumulator.append(blended_test)

    # Score using ROC AUC on full OOF predictions
    roc_auc = roc_auc_score(y, oof_preds)

    # Average test predictions across folds
    final_test_pred = np.mean(test_preds_accumulator, axis=0)
    final_test_preds.append(final_test_pred)

    return roc_auc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)


print("Best params:", study.best_params)
print("Best auc score:", study.best_value)



blend_pred = np.mean(final_test_preds, axis=0)


list_of_ids = sub['id'].tolist()

k = {'id':sub['id'].tolist(),
     'rainfall':blend_pred['rainfall']}

blend_pred = pd.DataFrame(k)
blend_pred.head()
blend_pred.to_csv('/kaggle/working/predblends.csv',index = False)


cluster = pd.concat([cluster,blend_pred['rainfall']],axis = 1)
cluster.columns = ['rainfall_cb','rainfall_xgb', 'rainfall_blend']
cluster.head(10)

