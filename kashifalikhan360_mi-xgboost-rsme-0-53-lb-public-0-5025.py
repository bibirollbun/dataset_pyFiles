import pandas as pd
train=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/train.csv")
test=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/test.csv")
submission=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/sample_submission.csv")
def clean_data(df):
    df['sex']=df['sex'].map({'female':0,'male':1})
    df['smoker']=df['smoker'].map({'yes':1,'no':0}) 
    df=pd.concat([pd.get_dummies(df['region']).astype(int),df],axis=1).drop(columns=['region'])
    df['bmi_smoker']=((df['bmi']<30)&(df['smoker']==1)).astype(int)
    df['bmi_age']=((df['bmi']<30)&(df['age']>=30)).astype(int)
    df['childer_smoker']=((df['age']<20)&(df['smoker']==1)).astype(int)
    df['mean_age_bmi']=df.groupby('age')['bmi'].transform('mean')
    df['children_bmi_smoker']=np.sqrt(df.groupby('bmi')['children'].transform('mean'))
    df.drop(columns=['northwest'],inplace=True)
    return df


train_cl=clean_data(train)
test_cl=clean_data(test)


import matplotlib.pyplot as plt
sns.distplot(train_cl['charges'])


import seaborn as sns
sns.boxplot(x=df['charges'])
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['charges'] < (Q1 - 1.5 * IQR)) | (df['charges'] > (Q3 + 1.5 * IQR))]
print(outliers.shape)


df=train_cl.copy()
df['charges_log']=np.log1p(df['charges'])
import seaborn as sns
sns.boxplot(df['charges_log'])


import numpy as np
corre_m=train_cl.corr().abs()
upper=corre_m.where(np.triu(np.ones(corre_m.shape),k=1).astype(bool))
[col for col in upper.columns if any(upper[col]>0.6)]


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
sns.heatmap(train_cl.corr(),annot=True,fmt='.3f',cbar=True)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc=StandardScaler()
X=sc.fit_transform(train_cl.drop(columns=['id','charges']).values)
y=np.log1p(train[['charges']].values)
X_test_sub=sc.fit_transform(test_cl.drop(columns=['id']).values)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


{'n_estimators': 2399,
 'learning_rate': 0.016613565042818485,
 'subsample': 0.4604943873331562,
 'colsample_bytree': 0.7892061764930905,
 'gamma': 2}


import optuna
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer

def finetuna(trial):
    param={
        'n_estimators':trial.suggest_int('n_estimators',2300,3000),
        'learning_rate':trial.suggest_float('learning_rate',0.009,0.2,log=True),
        'subsample':trial.suggest_float('subsample',0.3,0.5),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.7,0.9),
        'gamma':trial.suggest_int('gamma',1,4),
        # 'device':'cuda',
        # 'tree_method':'hist'
    }
    model=XGBRegressor(**param)
    kf=KFold(n_splits=5, shuffle=True, random_state=55)
    scores=cross_val_score(model,X_train,y_train,cv=kf,scoring=make_scorer(mean_squared_error,greater_is_better=False))
    rmse = np.sqrt(-scores.mean())  # negative because greater_is_better=False
    print(f"Trial {trial.number}: RMSE={rmse:.4f}, Params={param}")
    return rmse
study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner())
study.optimize(finetuna,n_trials=25)


study.best_params


study.best_trial
print("Best RMSE:", study.best_value)
print("Best Params:", study.best_params)


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
pram={'n_estimators': 2399,
 'learning_rate': 0.016613565042818485,
 'subsample': 0.4604943873331562,
 'colsample_bytree': 0.7892061764930905,
 'gamma': 2}
xgr=XGBRegressor(**pram)
xgr.fit(X_train,y_train)
y_pred=xgr.predict(X_test)
import numpy as np
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
pram={'n_estimators': 2399,
 'learning_rate': 0.016613565042818485,
 'subsample': 0.4604943873331562,
 'colsample_bytree': 0.7892061764930905,
 'gamma': 2}

xgr=XGBRegressor(**pram)
sc=StandardScaler()

X=sc.fit_transform(train_cl.drop(columns=['id','charges']).values)
y=np.log1p(train[['charges']].values)
xgr.fit(X,y)

X_test_sub=sc.fit_transform(test_cl.drop(columns=['id']).values)
y_pred_sub=np.expm1(xgr.predict(X_test_sub))


y_pred_sub = pd.DataFrame(y_pred_sub,columns=["charges"])
sub = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')[['id']]
final_sub = pd.concat([sub, y_pred_sub], axis=1)
final_sub.to_csv('submission.csv', index=False)


y_pred_sub = pd.DataFrame(np.expm1(y_pred_sub), columns=['charges'])
y_pred_sub.shape


sub = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')[['id']]
final_sub = pd.concat([sub, y_pred_sub], axis=1)
final_sub.to_csv('submission.csv', index=False)


pd.read_csv('/kaggle/working/submission.csv')

