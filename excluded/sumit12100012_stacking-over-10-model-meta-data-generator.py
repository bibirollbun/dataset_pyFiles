import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# Preprocessing library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
# Evaluation metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import uniform,randint,loguniform

import random
# Ml models imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


df=pd.read_csv("train.csv")
df.drop("id",axis=1,inplace=True)


df.head()


from itertools import combinations
def feature_maker(df):
    cols=["road_type","lighting","weather","time_of_day"]
    Dict={}
    for i in range(1,5):
        for idx,comb in enumerate(combinations(cols,i)):
            Q=df[comb[0]]
            for x in range(1,i):
                Q=Q+"_"+df[comb[x]]
            Dict[f"cat_{i}_{idx}"]=Q
    cols=["road_signs_present","holiday","school_season"]
    for i in range(1,5):
        for idx,comb in enumerate(combinations(cols,i)):
            Q=df[comb[0]]
            for x in range(1,i):
                Q=Q & df[comb[x]]
            Dict[f"and_{i}_{idx}"]=Q
    for i in range(1,5):
        for idx,comb in enumerate(combinations(cols,i)):
            Q=df[comb[0]]
            for x in range(1,i):
                Q=Q | df[comb[x]]
            Dict[f"or_{i}_{idx}"]=Q
    for i in range(1,5):
        for idx,comb in enumerate(combinations(cols,i)):
            Q=df[comb[0]]
            for x in range(1,i):
                Q=Q ^ df[comb[x]]
            Dict[f"xor_{i}_{idx}"]=Q
    df=pd.concat([df,pd.DataFrame(Dict)],axis=1)
    return df
df=feature_maker(df)


X=df.drop("accident_risk",axis=1)
y=df["accident_risk"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


num_cat=list(X_train.select_dtypes(include=[np.number]).columns)
alp_cat=[col for col in X_train.columns if col not in num_cat]
pipeline=ColumnTransformer([
    ("scaling",StandardScaler(),num_cat),
    ("One_Hot",OneHotEncoder(sparse_output=False, handle_unknown='ignore'),alp_cat),
])
X_train=pipeline.fit_transform(X_train)
X_test=pipeline.transform(X_test)

features=num_cat+list(pipeline.named_transformers_["One_Hot"].get_feature_names_out(alp_cat))


X_train=pd.DataFrame(X_train,columns=features)
X_test=pd.DataFrame(X_test,columns=features)


def eval_model(model,Dict,features):
    predictions=cross_val_predict(model,X_train[features],y_train,cv=3,n_jobs=-1)
    mae=mean_absolute_error(y_train,predictions)
    mse=mean_squared_error(y_train,predictions)
    r2=r2_score(y_train,predictions)
    #==============================================================
    # storing score in Dict
    Dict["Model"].append(model.__class__.__name__)
    Dict["MSE"].append(mse)
    Dict["MAE"].append(mae)
    Dict["R2"].append(r2)
    #=================================================================
    # printing score
    print("âœ… Model Name is : ",model.__class__.__name__)
    print("ğŸ�¹ MAE  ",mae)
    print("ğŸ�¹ MSE  ",mse)
    print("ğŸ�¹ R2  ",r2)


Dict={"Model":[],"MSE":[],"MAE":[],"R2":[]}


f1=[col for col in features if col.startswith("xor")]
f2=[col for col in features if col.startswith("or")]
f3=[col for col in features if col.startswith("and")]
f4=[col for col in features if col.startswith("cat")]
f5=[col for col in features if col not in f1+f2+f3+f4]
feature_set = [f5 + f1,f5 + f2,f5 + f3,f5 + f4,f5 + f1 + f2,f5 + f1 + f3,f5 + f1 + f4,f5 + f2 + f3,f5 + f2 + f4,f5 + f3 + f4,
                f5 + f1 + f2 + f3 + f4]


regressors = {
    # ğŸš€ XGBoost variations
    "XGB_default": XGBRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "XGB_shallow_fast": XGBRegressor(
        n_estimators=200, learning_rate=0.2, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "XGB_deep": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=10,
        subsample=0.9, colsample_bytree=0.9, gamma=0.2, reg_lambda=2,
        random_state=42, n_jobs=-1
    ),
    "XGB_medium_lr": XGBRegressor(
        n_estimators=400, learning_rate=0.07, max_depth=7,
        subsample=0.85, colsample_bytree=0.85, gamma=0.1,
        random_state=42, n_jobs=-1
    ),
    "XGB_shallow_high_lr": XGBRegressor(
        n_estimators=150, learning_rate=0.3, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),

    # âš¡ LightGBM variations
    "LGBM_default": LGBMRegressor(
        n_estimators=300, learning_rate=0.1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "LGBM_large_leaves": LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=128,
        max_depth=-1, subsample=0.9, colsample_bytree=0.9,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, n_jobs=-1
    ),
    "LGBM_fast": LGBMRegressor(
        n_estimators=200, learning_rate=0.2, num_leaves=16,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "LGBM_medium_leaves": LGBMRegressor(
        n_estimators=400, learning_rate=0.07, num_leaves=64,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    ),
    "LGBM_shallow_high_lr": LGBMRegressor(
        n_estimators=250, learning_rate=0.25, num_leaves=16,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "LGBM_deep_slow": LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=128,
        max_depth=12, subsample=0.9, colsample_bytree=0.9,
        min_child_samples=15, reg_alpha=0.2, reg_lambda=0.5,
        random_state=42, n_jobs=-1
    )
}


for idx,model in enumerate(regressors.values()):
    print(f"âœ…Model {idx+1} Training Has started")
    model.fit(X_train[feature_set[idx]],y_train)
    eval_model(model,Dict,feature_set[idx])
    print(f"âœ…Model {idx+1} Training Has Completed")
    print("="*50)


def eval_result(model,Dict,features):
    predictions=model.predict(X_test[features])
    mae=mean_absolute_error(y_test,predictions)
    mse=mean_squared_error(y_test,predictions)
    r2=r2_score(y_test,predictions)
    #==============================================================
    # storing score in Dict
    Dict["Model"].append(model.__class__.__name__)
    Dict["MSE"].append(mse)
    Dict["MAE"].append(mae)
    Dict["R2"].append(r2)
    #=================================================================
    # printing score
    print("âœ… Model Name is : ",model.__class__.__name__)
    print("ğŸ�¹ MAE  ",mae)
    print("ğŸ�¹ MSE  ",mse)
    print("ğŸ�¹ R2  ",r2)
    return Dict


Dict={"Model":[],"MSE":[],"MAE":[],"R2":[]}


for idx,model in enumerate(regressors.values()):
    print(f"âœ…Model {idx+1} Training Has started")
    eval_result(model,Dict,feature_set[idx])
    print(f"âœ…Model {idx+1} Training Has Completed")
    print("="*50)


pd.DataFrame(Dict)


submission=pd.read_csv("test.csv")
ID=submission["id"]
submission.drop("id",axis=1,inplace=True)
submission=feature_maker(submission)
submission=pipeline.transform(submission)
submission=pd.DataFrame(submission,columns=features)


predictions=np.zeros((172585))
for idx,model in enumerate(regressors.values()):
    pred=model.predict(submission[feature_set[idx]])
    predictions+=pred


predictions=predictions/11
submit=pd.DataFrame(predictions,columns=["accident_risk"],index=ID)


submit.to_csv("11_blend.csv")


def stacking(train,valid,test,target,regressor,feature_set):
    train_oov=np.ones((train.shape[0],len(regressor)))
    valid_oov=np.ones((valid.shape[0],len(regressor)))
    test_oov=np.ones((test.shape[0],len(regressor)))
    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    for idx,model in enumerate(regressor.values()):
        print(f"Iteration {idx+1}")
        train_subset=train[feature_set[idx]]
        valid_subset=valid[feature_set[idx]]
        test_subset=test[feature_set[idx]]
        valid_temp_oov=np.ones((valid.shape[0],5))
        test_temp_oov=np.ones((test.shape[0],5))
        for fold_idx,(train_idx,valid_idx) in enumerate(kf.split(train,target)):
            X_train,X_test=train_subset.iloc[train_idx,:],train_subset.iloc[valid_idx,:]
            y_train=target.iloc[train_idx]
            model.fit(X_train,y_train)
            train_oov[valid_idx,idx]=model.predict(X_test)
            valid_temp_oov[:,fold_idx]=model.predict(valid_subset)
            test_temp_oov[:,fold_idx]=model.predict(test_subset)
        valid_oov[:,idx]=valid_temp_oov.mean(axis=1)
        test_oov[:,idx]=test_temp_oov.mean(axis=1)
    return train_oov,valid_oov,test_oov
meta_train,meta_valid,meta_test=stacking(X_train,X_test,submission,y_train,regressors,feature_set)


joblib.dump(meta_train,"Meta_train.joblib")
joblib.dump(meta_valid,"Meta_valid.joblib")
joblib.dump(meta_test,"Meta_test.joblib")


joblib.dump(y_train,"Meta_train_target.joblib")
joblib.dump(y_test,"Meta_test_target.joblib")
joblib.dump(ID,"Submission_id.joblib")






















