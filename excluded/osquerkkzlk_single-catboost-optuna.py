import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")


dir="/kaggle/input/playground-series-s5e10"

data_train=pd.read_csv(os.path.join(dir,"train.csv"))
data_test=pd.read_csv(os.path.join(dir,"test.csv"))

orig=[]
dir2="/kaggle/input/simulated-roads-accident-data"
for s in ['2','10',"100"]:
    orig.append(pd.read_csv(os.path.join(dir2,f"synthetic_road_accidents_{s}k.csv")))
orig=pd.concat(orig,axis=0)

print("< train.shape >",data_train.shape)
display(data_train.head())
print("\n< test.shape >",data_test.shape)
display(data_test.head())
print("\n< orig.shape >",orig.shape)
display(orig.head())


data_test["accident_risk"]=0.5
orig["id"]=data_test.id.max()+1
orig=orig[data_train.columns]

print("< train.shape >",data_train.shape)
print("\n< test.shape >",data_test.shape)
print("\n< orig.shape >",orig.shape)


data=pd.concat([data_train,data_test,orig],axis=0,ignore_index=True)
print("< data.shape >",data.shape)


Features=orig.columns[1:-1].tolist()
Target="accident_risk"


import scipy

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

z = clip(f)(data)
data["y"] = z.values


Features.append("y")


cat_f,num_f,target=[],[],["accident_risk"]

for col in data.columns:
    temp="CAT"
    if data[col].dtype == "object":
        cat_f.append(col)
    else:
        num_f.append(col)
        temp="NUM"
    nunique=data[col].nunique()
    isna=data[col].isna().sum()
    print(f"[{temp}] have {nunique:10} categories and have {isna:10} NA")


print("\nCAT Features",cat_f)
print("\nNUM Features",num_f)



for col in cat_f:
    data[col]= data[col].astype("category")


train_size,test_size,orig_size=data_train.shape[0],data_test.shape[0],orig.shape[0]

train=data.iloc[:train_size]
test=data.iloc[train_size:train_size+test_size]
orig=data.iloc[-orig_size:]

print("< train.shape >",train.shape)
print("\n< test.shape >",test.shape)
print("\n< orig.shape >",orig.shape)


TE=[]

for col in Features:
    print(f"orig_{col}",end=" ")
    temp=orig.groupby(col)[Target].mean()
    temp.name=f"orig_{col}"
    train=train.merge(temp,on=col,how="left")
    test=test.merge(temp,on=col,how="left")
    TE.append(f"orig_{col}")

Features.extend(TE)


from catboost import CatBoostRegressor
import catboost as cat

cat_params={
    "learning_rate":0.1,
    "iterations":1000,
    "train_dir":"CAT_Test",
    "early_stopping_rounds":20,
    "task_type":"GPU",
    "use_best_model":True,
}


from sklearn.model_selection import KFold

oof_preds=np.zeros(len(train))

kf=KFold(shuffle=True,n_splits=7,random_state=42)
for fold ,(train_idx,val_idx) in enumerate(kf.split(train)):
    print(f"\n-------------------{fold+1}/7-----------------------")
    
    x_train =train.iloc[train_idx][Features]
    y_train =train.iloc[train_idx][Target]-train.iloc[train_idx]["y"]
    
    x_val =train.iloc[val_idx][Features]
    y_val =train.iloc[val_idx][Target]-train.iloc[val_idx]["y"]
    y_val_add=train.iloc[val_idx]["y"]

    P_train=cat.Pool(x_train,label=y_train,cat_features=cat_f)
    P_val=cat.Pool(x_val,label=y_val,cat_features=cat_f)

    model=CatBoostRegressor(**cat_params).fit(
        P_train,
        eval_set=P_val,
        verbose=200)
    oof_preds[val_idx]=model.predict(P_val)+y_val_add


from sklearn.metrics import mean_squared_error

mean_squared_error(train["accident_risk"],oof_preds,squared=False)


model.eval_metrics(
    data=P_val,
    metrics=["RMSE"],
    ntree_start=0,
    ntree_end=0,
    eval_period=1,
    plot=True
)


importance=np.array(model.get_feature_importance(data=P_val,prettified=True,type="LossFunctionChange"))
importance.reshape(-1,2)
importance


importance=pd.DataFrame(importance,columns=["Feature","Importance"]).sort_values(by="Importance",ascending=False)
_=sns.barplot(y="Feature",x="Importance",data=importance)                                                                     


import optuna
from sklearn.model_selection import train_test_split

def objective(trial):

    x_train,x_val,y_train,y_val=train_test_split(train[Features],train[Target])
    y_train =y_train-x_train["y"]
    
    y_val =y_val-x_val["y"]
    y_val_add=x_val["y"]
    
    P_train=cat.Pool(x_train,label=y_train,cat_features=cat_f)
    P_val=cat.Pool(x_val,label=y_val,cat_features=cat_f)

    params={
        "learning_rate":trial.suggest_float("learning_rate",1e-3,0.2),
        "iterations":trial.suggest_int("iterations",300,3000),
        "task_type":"GPU",
        "use_best_model":True,
        "train_dir":"Single_CAT",
        "subsample":trial.suggest_float("subsample",0.6,1),
        "depth":trial.suggest_int("depth",3,10),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",0,50),
        "random_seed":42,
        "early_stopping_rounds":50,
        "bootstrap_type":"Bernoulli"
    }
    model=CatBoostRegressor(**params)
    model.fit(P_train,verbose=200,eval_set=P_val)

    oof_preds=model.predict(P_val)+y_val_add
    return np.sqrt(np.mean((oof_preds-(y_val+y_val_add))**2))


from tqdm import tqdm
N_TRIALS=1000

study=optuna.create_study(direction="minimize",
                         study_name="Single_Catboost",
                         storage="sqlite:///Single_Catboost_Optuna_2.db",
                         load_if_exists=True)
pbar=tqdm(total=N_TRIALS)

def logging_callback(study,trial):
    pbar.update(1)
    pbar.set_description(f"< Trial > {trial.number+1}/{N_TRIALS} , < Best Value > {study.best_value:5f}")

def Early_Stopping_Trial(study,trial):
    if trial.number>study.best_trial.number+20:
        study.stop()
    else:pass

study.optimize(objective,n_trials=N_TRIALS,callbacks=[logging_callback,Early_Stopping_Trial])
pbar.close()


# 最佳参数 
best_params=study.best_params
fixxed_params={
        "task_type":"GPU",
        "use_best_model":True,
        "train_dir":"Single_CAT",
        "random_seed":42,
        "early_stopping_rounds":50,
        "bootstrap_type":"Bernoulli"}

oof_preds=np.zeros(len(train))
test_preds=np.zeros(len(test))

kf=KFold(shuffle=True,n_splits=7,random_state=42)
for fold ,(train_idx,val_idx) in enumerate(kf.split(train)):
    print(f"\n-------------------{fold+1}/7-----------------------")
    
    x_train =train.iloc[train_idx][Features]
    y_train =train.iloc[train_idx][Target]-train.iloc[train_idx]["y"]
    
    x_val =train.iloc[val_idx][Features]
    y_val =train.iloc[val_idx][Target]-train.iloc[val_idx]["y"]
    y_val_add=train.iloc[val_idx]["y"]

    x_test=test[Features]
    y_test_add=test["y"]

    P_train=cat.Pool(x_train,label=y_train,cat_features=cat_f)
    P_val=cat.Pool(x_val,label=y_val,cat_features=cat_f)
    P_test=cat.Pool(x_test,cat_features=cat_f)

    model=CatBoostRegressor(**best_params,**fixxed_params)
    model.fit(P_train,eval_set=P_val,verbose=200)

    oof_preds[val_idx]=model.predict(P_val)+y_val_add
    test_preds += (model.predict(P_test)+y_test_add)/7



Overall_cv=mean_squared_error(train["accident_risk"],oof_preds,squared=False)
print("\nOverall CV",Overall_cv)


sub_=pd.read_csv(os.path.join(dir,"sample_submission.csv"))
sub_["accident_risk"]=test_preds
sub_.to_csv("Single_Catboost_Optuna_test.csv",index=False)

df=pd.DataFrame({"id":train.id,"Single_Catboost_Optuna_preds":oof_preds})
df.to_csv("Single_Catboost_Optuna_oof.csv",index=False)







