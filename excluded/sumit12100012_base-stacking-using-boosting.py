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
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import uniform,randint,loguniform

# Ml models imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")


df=pd.read_csv("train.csv")
df.drop("id",axis=1,inplace=True)
df


df.info()


df.isna().sum()


df.duplicated().sum()


df.drop_duplicates(inplace=True)


fig,ax=plt.subplots(1,2,figsize=(20,4))
sns.histplot(data=df,x="accident_risk",color="green",bins=60,ax=ax[0])
ax[0].set_title("Accident Risk")
sns.histplot(data=df,x="curvature",color="green",ax=ax[1])
ax[1].set_title("Curvature")
plt.show()


df.columns


fig,ax=plt.subplots(2,5,figsize=(30,8))
sns.countplot(data=df,x="road_type",palette="pastel",ax=ax[0,0])
sns.countplot(data=df,x="num_lanes",palette="pastel",ax=ax[0,1])
sns.countplot(data=df,x="lighting",palette="pastel",ax=ax[0,2])
sns.countplot(data=df,x="weather",palette="pastel",ax=ax[0,3])
sns.countplot(data=df,x="road_signs_present",palette="pastel",ax=ax[0,4])
sns.countplot(data=df,x="public_road",palette="pastel",ax=ax[1,0])
sns.countplot(data=df,x="time_of_day",palette="pastel",ax=ax[1,1])
sns.countplot(data=df,x="holiday",palette="pastel",ax=ax[1,2])
sns.countplot(data=df,x="school_season",palette="pastel",ax=ax[1,2])
sns.countplot(data=df,x="num_reported_accidents",palette="pastel",ax=ax[1,3])
plt.show()


cols=df.select_dtypes(include=[np.number]).columns
fig,ax=plt.subplots(1,5,figsize=(20,6))
for idx,col in enumerate(cols):
    sns.boxplot(data=df,y=col,ax=ax[idx])
plt.show()


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


df.head()


X=df.drop("accident_risk",axis=1)
y=df["accident_risk"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


num_cat=list(X_train.select_dtypes(include=[np.number]).columns)
alp_cat=[col for col in X_train.columns if col not in num_cat]
Pipeline=ColumnTransformer([
    ("scaling",StandardScaler(),num_cat),
    ("One_Hot",OneHotEncoder(),alp_cat),
])
X_train=Pipeline.fit_transform(X_train)
X_test=Pipeline.transform(X_test)

features=num_cat+list(Pipeline.named_transformers_["One_Hot"].get_feature_names_out(alp_cat))


def eval_model(model,Dict,features):
    predictions=cross_val_predict(model,X_train,y_train,cv=4,n_jobs=-1)
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
    if hasattr(model,"feature_importances_"):
        fig,ax=plt.subplots(1,2,figsize=(40,6))
        data=pd.DataFrame({"Name":features,"Imp":model.feature_importances_}).sort_values("Imp",ascending=False)
        sns.barplot(data=data.head(20),y="Name",x="Imp",palette="Set2",ax=ax[0])
        ax[0].set_title("Top 20 features")
        sns.barplot(data=data.iloc[20:40],y="Name",x="Imp",palette="Set2",ax=ax[1])
        ax[1].set_title("Top 20-40 features")


Dict={"Model":[],"MSE":[],"MAE":[],"R2":[]}


xgb_reg=XGBRegressor()
xgb_reg.fit(X_train,y_train)
eval_model(xgb_reg,Dict,features)


lgbm_reg=LGBMRegressor()
lgbm_reg.fit(X_train,y_train)
eval_model(lgbm_reg,Dict,features)


cat_reg=CatBoostRegressor()
cat_reg.fit(X_train,y_train)
eval_model(cat_reg,Dict,features)


pd.DataFrame(Dict)


xgb_params = {
    # --- General Parameters ---
    'learning_rate': uniform(0.01, 0.2),  # Step size shrinkage
    'n_estimators': randint(100, 1500),    # Number of boosting rounds (trees)

    # --- Tree-Specific Parameters ---
    'max_depth': randint(3, 10),           # Maximum depth of a tree
    'min_child_weight': randint(1, 10),    # Minimum sum of instance weight needed in a child
    'gamma': uniform(0, 0.5),              # Minimum loss reduction required to make a further partition
    'subsample': uniform(0.6, 0.4),        # Subsample ratio of the training instance (loc=0.6, scale=0.4 gives range [0.6, 1.0])
    'colsample_bytree': uniform(0.6, 0.4), # Subsample ratio of columns when constructing each tree

    # --- Regularization Parameters ---
    'reg_alpha': loguniform(1e-2, 1e2),    # L1 regularization term on weights
    'reg_lambda': loguniform(1e-2, 1e2),   # L2 regularization term on weights
}

lgb_params = {
    # --- General Parameters ---
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 2000),

    # --- Tree-Specific Parameters ---
    'num_leaves': randint(20, 150),        # Max number of leaves in one tree (main complexity control)
    'max_depth': randint(3, 12),           # Max tree depth, -1 for no limit
    'min_child_samples': randint(5, 50),   # Minimum number of data needed in a child (leaf)

    # --- Subsampling & Regularization ---
    'subsample': uniform(0.6, 0.4),        # Also known as bagging_fraction
    'colsample_bytree': uniform(0.6, 0.4), # Also known as feature_fraction
    'reg_alpha': loguniform(1e-2, 1e2),    # L1 regularization
    'reg_lambda': loguniform(1e-2, 1e2),   # L2 regularization
}

cat_params = {
    # --- General Parameters ---
    'iterations': randint(100, 1500),      # Alias for n_estimators
    'learning_rate': uniform(0.01, 0.2),

    # --- Tree-Specific Parameters ---
    'depth': randint(4, 10),               # Alias for max_depth
    'l2_leaf_reg': loguniform(1, 10),      # L2 regularization coefficient
    'border_count': [32, 64, 128, 200],    # Number of splits for numerical features

    # --- Overfitting Control ---
    'subsample': uniform(0.6, 0.4),        # Sample rate of rows
    'random_strength': uniform(0, 2),      # Adds randomness to scores to combat overfitting
    'bagging_temperature': uniform(0, 1),  # Controls intensity of bagging
}

xgb=XGBRegressor()
lgbm=LGBMRegressor()
cat=CatBoostRegressor()

xgb_cv=RandomizedSearchCV(xgb,xgb_params,cv=3,n_jobs=-1,n_iter=20,scoring="neg_mean_squared_error",verbose=True)
lgbm_cv=RandomizedSearchCV(lgbm,lgb_params,cv=3,n_jobs=-1,n_iter=20,scoring="neg_mean_squared_error",verbose=True)
cat_cv=RandomizedSearchCV(cat,cat_params,cv=3,n_jobs=-1,n_iter=20,scoring="neg_mean_squared_error",verbose=True)



for idx,tune_model in enumerate([xgb_cv,lgbm_cv,cat_cv]):
    print(f"âœ…âœ…Tuninig Model {idx+1}")
    tune_model.fit(X_train,y_train)
    print("Completed")
    print("-"*50)


xgb_tuned=xgb_cv.best_estimator_
lgbm_tuned=lgbm_cv.best_estimator_
cat_tuned=cat_cv.best_estimator_


xgb_tuned=joblib.load("XGB_categorie_gates.pkl")
lgbm_tuned=joblib.load("LGBM_categorie_gates.pkl")
cat_tuned=joblib.load("CAT_categorie_gates.pkl")


Dict_tuned={"Model":[],"MSE":[],"MAE":[],"R2":[]}


xgb_tuned.fit(X_train,y_train)
eval_model(xgb_tuned,Dict_tuned,features)


lgbm_tuned.fit(X_train,y_train)
eval_model(lgbm_tuned,Dict_tuned,features)


cat_tuned.fit(X_train,y_train)
eval_model(cat_tuned,Dict_tuned,features)


print(pd.DataFrame(Dict))
print(pd.DataFrame(Dict_tuned))


def final_result(model,Dict,features):
    predictions=model.predict(X_test)
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


Final={"Model":[],"MSE":[],"MAE":[],"R2":[]}


for model in [xgb_reg,lgbm_reg,cat_reg,xgb_tuned,lgbm_tuned,cat_tuned,model]:
    final_result(model,Final,features)


pd.DataFrame(Final)


def stacking(X,y,upload,models):
    n_train = X.shape[0]
    n_test = upload.shape[0]
    n_models = len(models)
    oof_train = np.zeros((n_train, n_models))
    oof_test = np.zeros((n_test, n_models))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for model_idx, model in enumerate(models):
        print(f"Training Model {model_idx + 1}/{n_models}...")
        fold_test_preds = np.zeros((n_test, kf.get_n_splits()))
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold = X[val_idx]
            model.fit(X_train_fold, y_train_fold)
            oof_train[val_idx, model_idx] = model.predict(X_val_fold)
            fold_test_preds[:, fold_idx] = model.predict(upload)
        oof_test[:, model_idx] = fold_test_preds.mean(axis=1)
        print(f"Model {model_idx + 1} complete.")
    return oof_train, oof_test


X=Pipeline.transform(df.drop("accident_risk",axis=1))
models=[xgb_reg,lgbm_reg,cat_reg,xgb_tuned,lgbm_tuned,cat_tuned]
train_meta,test_meta=stacking(X,y.values,upload,models)


meta_1=XGBRegressor()
meta_2=LGBMRegressor()

meta_1.fit(train_meta,y)
meta_2.fit(train_meta,y)


pred_1=meta_1.predict(test_meta)
pred_2=meta_2.predict(test_meta)


Final=pred_2*0.65+pred2*0.35
submission=pd.DataFrame(Final,index=ID,columns=["accident_risk"])
submission.to_csv("Meta.csv")




