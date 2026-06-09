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


#Encoders for categorial features
from sklearn.preprocessing import OneHotEncoder   
from sklearn.preprocessing import LabelEncoder
from category_encoders.count import CountEncoder

#Light Gradient Boosting Model
from lightgbm import LGBMClassifier

#Optuna for hyperparameters optimization
import optuna

#Other libraries
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.simplefilter("ignore")


Train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
X_val = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
print("Train shape: ", Train.shape)
print("Test shape: ", X_val.shape)
Train.head()


# statistics of Train Dataframe's numerical features
Train.describe()


# statistics of Test Dataframe's numerical features
X_val.describe()


#Frequency of each distinct categorial features in Train Dataframe
for column in Train.select_dtypes(include='object').columns:
    print("*"*25)
    print(Train[column].value_counts())


#Heat map  correlation of numerical features and target in Train Dataframe
sns.heatmap(Train.select_dtypes(include= np.number ).corr(), cmap= 'coolwarm')
#We can see higher correlation between duration - target and previous - target then between other features


#Let's analyze the distribution of successful calls by days of the month.

Df_group = Train.groupby("day")['y'].agg(['count', 'sum'])
Df_group["success"] = Df_group["count"]/Df_group["sum"]
plt.bar( Df_group.index, Df_group.success)

# We can see that:

#1. the share of successful calls differs significantly on different days
#2. On last day of month  share of successful calls is increasing, but model don't know which day is last in different months

#So, I am going to do:

#1. add feature "last_day" : "yes" or "no" : is "day" last day in "month"?
#2. feature "day" code whith OneHotEncoder 


# Let's analize dependence number calls  of day for each month. This data is for march. 
# If you get data for other months, you can see that we have days when number of call is very small.  
# This happens with a frequency of 7 days. Let's consider these days as Sundays. When I analyzed all the months,
# I concluded that June, July, August, September, October, November, December looks like 2020 year. 
# January, February, March, April, March and May looks like 2021 year.

# So I'm going to add feature "day_of_week" and analize dependence number calls  of day of week.
f_group_day = Train[(Train.month=='mar')].groupby(["day"], as_index=False)[["id"]].count()
f_group_day.rename(columns={'id': 'count_calls'}, inplace=True)
plt.bar( f_group_day.day, f_group_day.count_calls)


day_month = {"jun" : 0, "jul" : 30, "aug" : 61, "sep": 92, "oct" : 122, "nov":153, "dec":183, "jan":214, "feb":245, "mar": 273, "apr":304, "may": 334 }


Train["number_day"] = Train.apply(lambda x: x.day+day_month[x.month], axis=1) 
    #X["number_day_year"] = X.apply(lambda x: x.day+day_month_year[x.month], axis=1) 

Train["day_of_week"]="unknown"
Train["day_of_week"] = Train.apply(lambda x: "monday" if (x["number_day"] % 7 == 1) else x["day_of_week"], axis=1) 
Train["day_of_week"] = Train.apply(lambda x: "tuesday" if x["number_day"] % 7 == 2 else x["day_of_week"], axis=1) 
Train["day_of_week"] = Train.apply(lambda x: "wednesday" if x["number_day"] % 7 == 3 else x["day_of_week"], axis=1)
Train["day_of_week"] = Train.apply(lambda x: "thursday" if x["number_day"] % 7 == 4 else x["day_of_week"], axis=1)
Train["day_of_week"] = Train.apply(lambda x: "friday" if x["number_day"] % 7 == 5 else x["day_of_week"], axis=1)
Train["day_of_week"] = Train.apply(lambda x: "saterday" if x["number_day"] % 7 == 6 else x["day_of_week"], axis=1)
Train["day_of_week"] = Train.apply(lambda x: "sunday" if x["number_day"] % 7 == 0 else x["day_of_week"], axis=1)
Train["day_of_week"] = Train.apply(lambda x: "unknown" if x["day"] == -1 else x["day_of_week"] , axis=1)


Df_group = Train.groupby("day_of_week")['y'].agg(['count', 'sum'])
Df_group["success"] = Df_group["sum"]/Df_group["count"]

plt.bar( Df_group.index, Df_group.success)

# We can see that successeful rate depends on day of week
# So, I' am going to add feature "day of week"


Train = Train.drop(["day_of_week", "number_day"], axis=1 )


# Split data

X_train, X_test, y_train, y_test = train_test_split(Train.drop("y", axis=1), Train.y, stratify = Train.y,
                                                    train_size=0.7, 
                                                    random_state=42)


#  prepare data
def prepare_data(X_in):
    X=X_in.copy()
    last_day_month = {"jan" : 31, "feb" : 28, "mar" : 31, "apr": 30, "may" : 31, "jun":30, "jul":31, "aug":31, "sep":30, "oct": 31, "nov":30, "dec": 31 }
    day_month = {"jun" : 0, "jul" : 30, "aug" : 61, "sep": 92, "oct" : 122, "nov":153, "dec":183, "jan":214, "feb":245, "mar": 273, "apr":304, "may": 334 }
    X["last_day"] = X.apply(lambda x: "yes" if x['day'] == last_day_month[x['month']] else "no", axis=1)
    X.day = X.apply(lambda x: -1 if (x['last_day'] == "no") & (x['day'] == 31) else x["day"], axis=1)
    X.day = X.apply(lambda x: -1 if (x['month'] == "feb") & (x['day'] == 31) else x["day"], axis=1)
    X.day = X.apply(lambda x: -1 if (x['month'] == "feb") & (x['day'] == 30) else x["day"], axis=1)
    X.day = X.apply(lambda x: -1 if (x['month'] == "feb") & (x['day'] == 29) else x["day"], axis=1)
    #
    X["number_day"] = X.apply(lambda x: x.day+day_month[x.month], axis=1) 
    #X["number_day_year"] = X.apply(lambda x: x.day+day_month_year[x.month], axis=1) 

    X["day_of_week"]="unknown"
    X["day_of_week"] = X.apply(lambda x: "monday" if (x["number_day"] % 7 == 1) else x["day_of_week"], axis=1) 
    X["day_of_week"] = X.apply(lambda x: "tuesday" if x["number_day"] % 7 == 2 else x["day_of_week"], axis=1) 
    X["day_of_week"] = X.apply(lambda x: "wednesday" if x["number_day"] % 7 == 3 else x["day_of_week"], axis=1)
    X["day_of_week"] = X.apply(lambda x: "thursday" if x["number_day"] % 7 == 4 else x["day_of_week"], axis=1)
    X["day_of_week"] = X.apply(lambda x: "friday" if x["number_day"] % 7 == 5 else x["day_of_week"], axis=1)
    X["day_of_week"] = X.apply(lambda x: "saterday" if x["number_day"] % 7 == 6 else x["day_of_week"], axis=1)
    X["day_of_week"] = X.apply(lambda x: "sunday" if x["number_day"] % 7 == 0 else x["day_of_week"], axis=1)
    X["day_of_week"] = X.apply(lambda x: "unknown" if x["day"] == -1 else x["day_of_week"] , axis=1)
    
    X["weekend"] = X.apply(lambda x: "yes" if (x["day_of_week"] == "monday") | (x["day_of_week"] == "sanday") else "no" , axis=1)
    for column in list(X.select_dtypes(include='int64').columns):
        if column!="id":
            X[column] = X[column].astype("int32")
    X = X.drop("number_day", axis=1)


    return X 


# function for fit OneHotEncoder
def Ohe_fit(X_in, features_to_ohe):
    X=X_in.copy()
    ohe =OneHotEncoder(sparse_output=False, drop = "if_binary", handle_unknown='ignore')
    ohe.fit(X[features_to_ohe])
    return ohe


#function for transform data by OneHotEncoder
def Ohe_transform(ohe, X_in, features_to_ohe):
    X=X_in.copy()

    encoded_data = ohe.transform(X[features_to_ohe])
    encoded_data = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())
    X = pd.concat([X.reset_index(), encoded_data.reset_index()], axis=1)    
    for column in ohe.get_feature_names_out():
        if column!="id":
            X[column] = X[column].astype("int8")
    X = X.drop(features_to_ohe, axis=1)
    return X


#Data Preparation

X_train_tuned = prepare_data(X_train)
X_test_tuned = prepare_data(X_test)



# I am going to label encode all categorical features
features_to_label=['job', 'marital', 'education', 'default','housing','loan','contact','month','poutcome','last_day','day_of_week', 'weekend']
LE = LabelEncoder()
for feature in features_to_label:
    LE.fit(X_train_tuned[feature])
    X_train_tuned[feature+"_label"] = LE.transform(X_train_tuned[feature])
    X_test_tuned[feature+"_label"] = LE.transform(X_test_tuned[feature])


X_train_tuned.head()


# convert types to take care computational resources
for column in list(X_train_tuned.select_dtypes(include='int64').columns):
        if column!="id":
            X_train_tuned[column] = X_train_tuned[column].astype("int8")
            X_test_tuned[column] = X_test_tuned[column].astype("int8")


features_to_ohe=  ['job', 'marital', 'education', 'default','housing','loan','contact','month','poutcome','last_day','day_of_week', 'weekend','day']


X_train_tuned


#I am going to combine featuers pairs and encode pairs by count encoder

all_features=X_train_tuned.columns
features_to_count_encoding = []
for feature in all_features:
    if(feature!="id"):
        for feature1 in all_features:
           if(feature1!="id") and  (feature1!=feature) and ((feature1+"_"+feature) not in X_train_tuned.columns) and (feature not in features_to_label) and (feature1 not in features_to_label): 
               X_train_tuned[feature+"_"+feature1] = X_train_tuned[feature].astype(str)+"+"+X_train_tuned[feature1].astype (str)
               X_test_tuned[feature+"_"+feature1] = X_test_tuned[feature].astype(str)+"+"+X_test_tuned[feature1].astype(str)
               features_to_count_encoding.append(feature+"_"+feature1)


# Count Encoder
CE=CountEncoder(cols=features_to_count_encoding)



CE.fit(X_train_tuned, y_train)


feature_label = []
for feature in features_to_label:
    feature_label.append(feature+"_label")
feature_label
feature_label


X_test_tuned.head()


X_train_tuned=CE.transform(X_train_tuned)
X_test_tuned=CE.transform(X_test_tuned)
X_train_tuned=X_train_tuned.drop(feature_label, axis=1)
X_test_tuned=X_test_tuned.drop(feature_label, axis=1)


#fit One Hot Encoder
Ohe_train_my = Ohe_fit(X_train_tuned, features_to_ohe)



#Transform train data by One Hot Encoder
X_train_tuned = Ohe_transform(Ohe_train_my, X_train_tuned, features_to_ohe)
X_train_tuned.head()


# Transform test data by One Hot Encoder
X_test_tuned = Ohe_transform(Ohe_train_my, X_test_tuned, features_to_ohe)
X_test_tuned.head()


X_train_tuned=X_train_tuned.drop(["index", "id"], axis=1)
X_test_tuned=X_test_tuned.drop(["index", "id"], axis=1)


# Optuna Hyperparameters tuning LGBMClassifier

def objective(trial):    
    param = {
        'n_jobs':-1, 
        'eval_metric' : "auc",   
        'random_state':42,
        'verbosity':-1,
        'n_estimators':trial.suggest_int('n_estimators',1000,6000),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 30),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 30),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'learning_rate': trial.suggest_float('learning_rate',0.01, 0.04),
        'num_leaves': trial.suggest_int('num_leaves', 30, 200),
        'max_depth':trial.suggest_int('max_depth', 5, 15),
        'max_bin':trial.suggest_int('max_bin', 100, 5000),
        'min_child_samples':  trial.suggest_int('min_child_samples', 3, 30),
    }
    model = LGBMClassifier(**param)  
    model.fit(X_train_tuned, y_train)
    y_LGBM = model.predict_proba(X_test_tuned)
    roc_auc =roc_auc_score(y_test, y_LGBM[:,1])
    return roc_auc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


X_train_tuned.shape


#Repeat all steps on full data


Train_tuned = prepare_data(Train.drop("y", axis=1))
X_val_tuned = prepare_data(X_val)


features_to_label=['job', 'marital', 'education', 'default','housing','loan','contact','month','poutcome','last_day','day_of_week', 'weekend']
features_to_ohe=['job', 'marital', 'education', 'default','housing','loan','contact','month','poutcome','last_day','day_of_week', 'weekend', 'day']


Train_tuned.head()


LE = LabelEncoder()
for feature in features_to_label:
    LE.fit(Train_tuned[feature])
    Train_tuned[feature+"_label"] = LE.transform(Train_tuned[feature])
    X_val_tuned[feature+"_label"] = LE.transform(X_val_tuned[feature])




for column in list(Train_tuned.select_dtypes(include='int64').columns):
        if column!="id":
            Train_tuned[column] = Train_tuned[column].astype("int32")
            X_val_tuned[column] = X_val_tuned[column].astype("int32")



all_features=Train_tuned.columns
features_to_count_encoding = []
for feature in all_features:
    if(feature!="id"):
        for feature1 in all_features:
           if(feature1!="id") and (feature1!=feature) and ((feature1+"_"+feature) not in Train_tuned.columns) and (feature not in features_to_label) and (feature1 not in features_to_label): 
               Train_tuned[feature+"_"+feature1] = Train_tuned[feature].astype(str)+"+"+Train_tuned[feature1].astype(str)
               X_val_tuned[feature+"_"+feature1] = X_val_tuned[feature].astype(str)+"+"+X_val_tuned[feature1].astype(str)
               features_to_count_encoding.append(feature+"_"+feature1)


# Count Encoder
CE=CountEncoder(cols=features_to_count_encoding)


CE.fit(Train_tuned, Train.y)


feature_label = []
for feature in features_to_label:
    feature_label.append(feature+"_label")
feature_label
feature_label


Train_tuned=CE.transform(Train_tuned)



X_val_tuned=CE.transform(X_val_tuned)
Train_tuned=Train_tuned.drop(feature_label, axis=1)
X_val_tuned=X_val_tuned.drop(feature_label, axis=1)


#fit One Hot Encoder
Ohe_train = Ohe_fit(Train_tuned, features_to_ohe)


Train_tuned = Ohe_transform(Ohe_train, Train_tuned, features_to_ohe)
Train_tuned=Train_tuned.drop(["index", "id"], axis=1)

X_val_tuned = Ohe_transform(Ohe_train, X_val_tuned, features_to_ohe)
X_val_tuned=X_val_tuned.drop(["index", "id"], axis=1)


#Create LGBM with best parameters

LGBM = LGBMClassifier(
    n_jobs=-1, 
    eval_metric = "auc",   
    random_state=42,
    verbosity=-1,
    n_estimators=4263, 
    learning_rate= 0.036019426144927265,
    min_child_samples=26,
    subsample=0.8251149539458965,
    colsample_bytree= 0.5270990554726177,
    num_leaves=49,
    max_depth= 12,
    max_bin= 2916,
    
    reg_alpha=2.4356357622500435,
    reg_lambda=6.850902607219219,
    )

LGBM.fit(Train_tuned, Train.y)
y_LGBM = LGBM.predict_proba(X_val_tuned)


df_result = pd.concat([X_val["id"].reset_index(), pd.DataFrame(y_LGBM[:,1], columns=["y"]).reset_index()], axis=1)
df_result=df_result.drop("index", axis=1)
df_result.to_csv("sample_submission.csv", index=False)


df_result

