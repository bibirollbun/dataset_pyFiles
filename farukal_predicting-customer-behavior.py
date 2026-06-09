import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd



train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train.head()



test.head()


train.drop(columns = ["id"],axis = 1,inplace = True)
test.drop(columns = ["id"],axis = 1,inplace = True)



LE = LabelEncoder()
for column in train.columns:
    if train[column].dtype == "object":
        train[column] = LE.fit_transform(train[column].astype(str))

for column in test.columns:
    if test[column].dtype == "object":
        test[column] = LE.fit_transform(test[column].astype(str))




train.head()


test.head()


SS = StandardScaler()
y = train["y"]
train  = train.drop(columns = "y")
columns = train.columns
trains= SS.fit_transform(train)
tests = SS.fit_transform(test)
train = pd.DataFrame(trains , columns = columns)
test =  pd.DataFrame(tests , columns  = columns)
train["y"]  = y.reset_index(drop = True) 
train.head()
test.head()


y = train["y"]
x = train.drop(columns = "y")



X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,random_state = 8,stratify = y)



from catboost import CatBoostClassifier
import optuna
def objective (trial):
    params = {
        'depth' : trial.suggest_int('depth',3,10),
        'learning_rate' : trial.suggest_float('learning_rate',0.005,0.3,log = True),
        'iterations' : trial.suggest_int('iterations',150,1200),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg',0.01,10.0,log= True),
        'bagging_temperature': trial.suggest_float('bagging_temperature',0.0,1.0),
        'random_strength' : trial.suggest_float('random_strength',1e-9,10.0,log =True),
        'border_count' : trial.suggest_int('border_count',32,255),
        'random_state' : 98,
        'verbose' : 150,
        'task_type' :  'GPU'
    }
    model  = CatBoostClassifier(**params)
    model.fit(X_train,Y_train)
    pre = model.predict(X_test)
    return roc_auc_score(Y_test,pre)
    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective,n_trials= 10)


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(**study.best_params,verbose = 150,random_state = 98,task_type = 'GPU')
cat_model.fit(X_train,Y_train)


predictions3 = cat_model.predict_proba(X_test)
print("my auc score is",roc_auc_score(Y_test,predictions3[:,1]))



import optuna
def objective (trial):
    params = {
        'max_depth' : trial.suggest_int('max_depth',3,10),
        'learning_rate' : trial.suggest_float('lr',0.005,0.01),
        'subsample' : trial.suggest_float('ss',0.5,1),
        'n_estimators' : trial.suggest_int('n',600,1200),
        'colsample_bytree': trial.suggest_float('cs',0.5,1),
        'gamma' : trial.suggest_float('g',0,0.5),
        'min_child_weight' : trial.suggest_int('m',1,10),
        'random_state' : 98
    }
    model  = XGBClassifier(**params,
                          device = 'cuda')
    model.fit(X_train,Y_train)
    pre = model.predict(X_test)
    return roc_auc_score(Y_test,pre)
    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective,n_trials= 20)


xgb_model  = XGBClassifier(max_depth =  10
                           , learning_rate =  0.00960008360096706
                           , subsample =  0.7343265983322328
                           , n_estimators= 1036
                           , colsample_bytree =  0.5612644396085253
                           , gamma =  0.3047012122494639
                           , min_child_weight=  4
                           , random_state = 98
                           , device = 'cuda')
xgb_model.fit(X_train,Y_train,verbose = 1)




prediction1 = xgb_model.predict_proba(X_test)
print("My AUC score is",roc_auc_score(Y_test,prediction1[:,1]))



lgbm_model = LGBMClassifier(
    max_depth=  8, learning_rate= 0.00837639183587293, subsample=0.953557869970604, n_estimators= 847, colsample_bytree=0.808616862983552, gamma= 0.2395225512959197, min_child_weight= 8,
    verbose = -1,random_state = 7
)
lgbm_model.fit(X_train,Y_train)



prediction2 = lgbm_model.predict_proba(X_test)
print("My Auc Score is",roc_auc_score(Y_test,prediction2[:,1]))



voting = VotingClassifier(
    estimators = [('cat',cat_model),('xgb',xgb_model),("lgbm",lgbm_model)],
    voting = "soft"
)
voting.fit(X_train,Y_train)


from sklearn.model_selection import StratifiedKFold
fold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 788)
probs = np.zeros(len(test))
for (train,val) in fold.split(x,y):
    X_train1,Y_train1 = x.iloc[train],y.iloc[train]
    X_test1,Y_test1 = x.iloc[val],y.iloc[val]

    voting.fit(X_train1,Y_train1)
    prob = voting.predict_proba(test)
    probs += (prob[:,1])/5
    





sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sample.drop(columns = "y")
sample["y"] = probs
sample.to_csv('predictions.csv',index = False)



print(sample)

