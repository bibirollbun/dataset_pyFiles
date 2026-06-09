import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing,model_selection,linear_model,metrics,ensemble
from xgboost import XGBRegressor



df = pd.read_csv("/kaggle/input/5foldcv-road-acc/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


df.columns


features=[c for c in df.columns if c not in ('id','accident_risk','kfold')] #useful features by which we will be training our model on
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  #subsetting numerical columns from the 'useful features'
cat_cols=df[features].select_dtypes(include='object').columns.tolist() #subsetting categorical columns from the 'useful features'


df_test=df_test[features]  #to test our predictions we are subsetting the useful features 


num_cols


cat_cols


df.weather.value_counts()


final_predictions=[]  #here, we will append our test predictions
rmse_scores=[]
for fold in range(5):
    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.accident_risk.values
    yvalid=xvalid.accident_risk.values

    xtrain=xtrain[features]
    xvalid=xvalid[features]


    ohe=preprocessing.OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    xtrain_ohe=ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe=ohe.transform(xvalid[cat_cols])
    xtest_ohe=ohe.transform(xtest[cat_cols])

    xtrain_ohe=pd.DataFrame(xtrain_ohe,columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe=pd.DataFrame(xvalid_ohe,columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe=pd.DataFrame(xtest_ohe,columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain=pd.concat([xtrain,xtrain_ohe],axis=1)
    xvalid=pd.concat([xvalid,xvalid_ohe],axis=1)
    xtest=pd.concat([xtest,xtest_ohe],axis=1)

    xtrain=xtrain.drop(cat_cols,axis=1)
    xvalid=xvalid.drop(cat_cols,axis=1)
    xtest=xtest.drop(cat_cols,axis=1)

    scaler=preprocessing.StandardScaler()
    xtrain[num_cols]=scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols]=scaler.transform(xvalid[num_cols])
    xtest[num_cols]=scaler.transform(xtest[num_cols])


    model=XGBRegressor(random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid=model.predict(xvalid)
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
    rmse=metrics.mean_squared_error(yvalid,preds_valid,squared=False)
    print(f'Fold{str(fold+1)} RMSE: {rmse}')
    rmse_scores.append(rmse)

print(np.mean(rmse_scores))


preds = np.mean(np.column_stack(final_predictions), axis=1)


sample_submission.accident_risk = preds
sample_submission.to_csv("submission.csv", index=False)


for col in num_cols:
    df[col]=np.log1p(df[col])
    df_test[col]=np.log1p(df_test[col])
    
final_predictions=[]  #here, we will append our test predictions
rmse_scores=[]
for fold in range(5):
    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.accident_risk.values
    yvalid=xvalid.accident_risk.values

    xtrain=xtrain[features]
    xvalid=xvalid[features]


    ohe=preprocessing.OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    xtrain_ohe=ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe=ohe.transform(xvalid[cat_cols])
    xtest_ohe=ohe.transform(xtest[cat_cols])

    xtrain_ohe=pd.DataFrame(xtrain_ohe,columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe=pd.DataFrame(xvalid_ohe,columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe=pd.DataFrame(xtest_ohe,columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain=pd.concat([xtrain,xtrain_ohe],axis=1)
    xvalid=pd.concat([xvalid,xvalid_ohe],axis=1)
    xtest=pd.concat([xtest,xtest_ohe],axis=1)

    xtrain=xtrain.drop(cat_cols,axis=1)
    xvalid=xvalid.drop(cat_cols,axis=1)
    xtest=xtest.drop(cat_cols,axis=1)

    scaler=preprocessing.StandardScaler()
    xtrain[num_cols]=scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols]=scaler.transform(xvalid[num_cols])
    xtest[num_cols]=scaler.transform(xtest[num_cols])


    model=XGBRegressor(random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid=model.predict(xvalid)
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
    rmse=metrics.mean_squared_error(yvalid,preds_valid,squared=False)
    print(f'Fold{str(fold+1)} RMSE: {rmse}')
    rmse_scores.append(rmse)

print(np.mean(rmse_scores))


features=[c for c in df.columns if c not in ('id','accident_risk','kfold')] #useful features by which we will be training our model on
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  #subsetting numerical columns from the 'useful features'
cat_cols=df[features].select_dtypes(include='object').columns.tolist() #subsetting categorical columns from the 'useful features'




poly=preprocessing.PolynomialFeatures(degree=3,interaction_only=True,include_bias=False)
train_poly=poly.fit_transform(df[num_cols])
test_poly=poly.transform(df_test[num_cols])
df_poly=pd.DataFrame(train_poly,columns=[f'poly_{i}' for i in range(train_poly.shape[1])])
df_test_poly=pd.DataFrame(test_poly,columns=[f'poly_{i}' for i in range(test_poly.shape[1])])

df = pd.concat([df, df_poly], axis=1)
df_test = pd.concat([df_test, df_test_poly], axis=1)


features=[c for c in df.columns if c not in ('id','accident_risk','kfold')] #useful features by which we will be training our model on
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  #subsetting numerical columns from the 'useful features'
cat_cols=df[features].select_dtypes(include='object').columns.tolist() #subsetting categorical columns from the 'useful features'


final_predictions=[]  #here, we will append our test predictions
rmse_scores=[]
for fold in range(5):
    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.accident_risk.values
    yvalid=xvalid.accident_risk.values

    xtrain=xtrain[features]
    xvalid=xvalid[features]


    ohe=preprocessing.OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    xtrain_ohe=ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe=ohe.transform(xvalid[cat_cols])
    xtest_ohe=ohe.transform(xtest[cat_cols])

    xtrain_ohe=pd.DataFrame(xtrain_ohe,columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe=pd.DataFrame(xvalid_ohe,columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe=pd.DataFrame(xtest_ohe,columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain=pd.concat([xtrain,xtrain_ohe],axis=1)
    xvalid=pd.concat([xvalid,xvalid_ohe],axis=1)
    xtest=pd.concat([xtest,xtest_ohe],axis=1)

    xtrain=xtrain.drop(cat_cols,axis=1)
    xvalid=xvalid.drop(cat_cols,axis=1)
    xtest=xtest.drop(cat_cols,axis=1)


    model=XGBRegressor(random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid=model.predict(xvalid)
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
    rmse=metrics.mean_squared_error(yvalid,preds_valid,squared=False)
    print(f'Fold{str(fold+1)} RMSE: {rmse}')
    rmse_scores.append(rmse)

print(np.mean(rmse_scores))


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from xgboost import XGBRegressor


# Load data

df = pd.read_csv("/kaggle/input/5foldcv-road-acc/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


# Basic feature setup

features = [c for c in df.columns if c not in ('id','accident_risk','kfold')]
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()
cat_cols = df[features].select_dtypes(include='object').columns.tolist()

# =======================
# Target Encoding
# =======================
for col in cat_cols:
    temp_df = []
    temp_test_feat = None
    
    for fold in range(5):
        X_train = df[df.kfold != fold].reset_index(drop=True)
        X_valid = df[df.kfold == fold].reset_index(drop=True)
        
        feat = X_train.groupby(col)['accident_risk'].mean().to_dict()
        X_valid.loc[:, f'tar_enc_{col}'] = X_valid[col].map(feat)
        temp_df.append(X_valid)
        
        if temp_test_feat is None:
            temp_test_feat = df_test[col].map(feat)
        else:
            temp_test_feat += df_test[col].map(feat)
    
    temp_test_feat /= 5
    df_test.loc[:, f'tar_enc_{col}'] = temp_test_feat
    df = pd.concat(temp_df).reset_index(drop=True)


# Update features

features = [c for c in df.columns if c not in ('id','accident_risk','kfold')]
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()
cat_cols = df[features].select_dtypes(include='object').columns.tolist()


# Model Training

final_predictions = []
rmse_scores = []

for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    ytrain = xtrain.accident_risk.values
    yvalid = xvalid.accident_risk.values

    xtrain = xtrain[features].copy()
    xvalid = xvalid[features].copy()
    xtest = xtest[features].copy()
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[cat_cols] = ordinal_encoder.fit_transform(xtrain[cat_cols])
    xvalid[cat_cols] = ordinal_encoder.transform(xvalid[cat_cols])
    xtest[cat_cols] = ordinal_encoder.transform(xtest[cat_cols])

    model = XGBRegressor(random_state=42)
    model.fit(xtrain, ytrain)

    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    
    rmse = metrics.mean_squared_error(yvalid, preds_valid, squared=False)
    print(f'Fold {fold+1} RMSE: {rmse:.5f}')
    
    final_predictions.append(test_preds)
    rmse_scores.append(rmse)

# =======================
# Final Averaging
# =======================
print(f'Mean RMSE: {np.mean(rmse_scores):.5f}')
final_predictions = np.mean(np.column_stack(final_predictions), axis=1)




import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from xgboost import XGBRegressor
import optuna

df = pd.read_csv("/kaggle/input/5foldcv-road-acc/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

features = [c for c in df.columns if c not in ('id','accident_risk','kfold')]
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()
cat_cols = df[features].select_dtypes(include='object').columns.tolist()


def run(trial):
    
    fold=0
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.accident_risk.values
    yvalid=xvalid.accident_risk.values

    xtrain=xtrain[features]
    xvalid=xvalid[features]


    ohe=preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    xtrain_ohe=ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe=ohe.transform(xvalid[cat_cols])
    xtest_ohe=ohe.transform(xtest[cat_cols])

    xtrain_ohe=pd.DataFrame(xtrain_ohe,columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe=pd.DataFrame(xvalid_ohe,columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe=pd.DataFrame(xtest_ohe,columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain=pd.concat([xtrain,xtrain_ohe],axis=1)
    xvalid=pd.concat([xvalid,xvalid_ohe],axis=1)
    xtest=pd.concat([xtest,xtest_ohe],axis=1)

    xtrain=xtrain.drop(cat_cols,axis=1)
    xvalid=xvalid.drop(cat_cols,axis=1)
    xtest=xtest.drop(cat_cols,axis=1)


    model=XGBRegressor(random_state=42,tree_method = "hist", device = "cuda",n_estimators=7000,
                       early_stopping_rounds=300,
                        learning_rate=learning_rate,
                        reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        max_depth=max_depth)
    
    model.fit(xtrain,ytrain,eval_set=[(xvalid,yvalid)],verbose=False)
    preds_valid=model.predict(xvalid)
    
    rmse=metrics.mean_squared_error(yvalid,preds_valid,squared=False)
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(run, n_trials=20)


study.best_params


import joblib


joblib.dump(study, "/kaggle/working/optuna_study.pkl")
print("Optuna study saved successfully!")



features=[c for c in df.columns if c not in ('id','accident_risk','kfold')] #useful features by which we will be training our model on
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  #subsetting numerical columns from the 'useful features'
cat_cols=df[features].select_dtypes(include='object').columns.tolist() #subsetting categorical columns from the 'useful features'

df_test=df_test[features]


best_params=study.best_params
final_predictions=[]  #here, we will append our test predictions
rmse_scores=[]
for fold in range(5):
    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.accident_risk.values
    yvalid=xvalid.accident_risk.values

    xtrain=xtrain[features]
    xvalid=xvalid[features]


    ohe=preprocessing.OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    xtrain_ohe=ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe=ohe.transform(xvalid[cat_cols])
    xtest_ohe=ohe.transform(xtest[cat_cols])

    xtrain_ohe=pd.DataFrame(xtrain_ohe,columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe=pd.DataFrame(xvalid_ohe,columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe=pd.DataFrame(xtest_ohe,columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain=pd.concat([xtrain,xtrain_ohe],axis=1)
    xvalid=pd.concat([xvalid,xvalid_ohe],axis=1)
    xtest=pd.concat([xtest,xtest_ohe],axis=1)

    xtrain=xtrain.drop(cat_cols,axis=1)
    xvalid=xvalid.drop(cat_cols,axis=1)
    xtest=xtest.drop(cat_cols,axis=1)


    model=XGBRegressor(random_state=42,tree_method = "hist", device = "cuda",n_estimators=7000,
                       early_stopping_rounds=300,**best_params)
    model.fit(
    xtrain, ytrain,
    eval_set=[(xvalid, yvalid)],
    verbose=False
)
    preds_valid=model.predict(xvalid)
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
    
    rmse=metrics.mean_squared_error(yvalid,preds_valid,squared=False)
    print(f'Fold{str(fold+1)} RMSE: {rmse}')
    rmse_scores.append(rmse)

print(np.mean(rmse_scores))


preds = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission.accident_risk = preds
sample_submission.to_csv("submission2.csv", index=False)


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn import metrics


df = pd.read_csv("/kaggle/input/5foldcv-road-acc/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


features = [c for c in df.columns if c not in ('id', 'accident_risk', 'kfold')]
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()
cat_cols = df[features].select_dtypes(include='object').columns.tolist()

df_test = df_test[features]


final_predictions = []
rmse_scores = []


for fold in range(5):
    print(f"\n Fold {fold + 1}")

    # Split data
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.accident_risk.values
    yvalid = xvalid.accident_risk.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

   
    train_pool = Pool(xtrain, label=ytrain, cat_features=cat_cols)      
    valid_pool = Pool(xvalid, label=yvalid, cat_features=cat_cols)
    test_pool = Pool(xtest, cat_features=cat_cols)

    
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        task_type="GPU",       
        eval_metric="RMSE",
        loss_function="RMSE",
        early_stopping_rounds=200,
        verbose=500,
        ##**best_params  
    )

    
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

   
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)

   
    rmse = metrics.mean_squared_error(yvalid, preds_valid, squared=False)
    print(f"Fold {fold + 1} RMSE: {rmse:.5f}")
    rmse_scores.append(rmse)


print("\n Mean RMSE across folds:", np.mean(rmse_scores))


final_predictions = np.mean(np.column_stack(final_predictions), axis=1)





model.get_feature_importance(prettified=True)   



import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


df = pd.read_csv("/kaggle/input/5foldcv-road-acc/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


features = [c for c in df.columns if c not in ('id','accident_risk','kfold')]
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()
cat_cols = df[features].select_dtypes(include='object').columns.tolist()

df_test = df_test[features]


oof_xgb = np.zeros(len(df))
oof_cat = np.zeros(len(df))
test_preds_xgb = np.zeros((len(df_test), 5))
test_preds_cat = np.zeros((len(df_test), 5))
rmse_scores = []

best_params={'learning_rate': 0.03358845707372683,
 'reg_lambda': 1.0900608171378882e-05,
 'reg_alpha': 1.185869175426394e-07,
 'subsample': 0.9074476262602248,
 'colsample_bytree': 0.8511367922571782,
 'max_depth': 7}

for fold in range(5):
    print(f"\n Fold {fold + 1}")
    
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.accident_risk.values
    yvalid = xvalid.accident_risk.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

    
    ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe = ohe.transform(xvalid[cat_cols])
    xtest_ohe = ohe.transform(xtest[cat_cols])

    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain_xgb = pd.concat([xtrain.reset_index(drop=True), xtrain_ohe], axis=1).drop(cat_cols, axis=1)
    xvalid_xgb = pd.concat([xvalid.reset_index(drop=True), xvalid_ohe], axis=1).drop(cat_cols, axis=1)
    xtest_xgb = pd.concat([xtest.reset_index(drop=True), xtest_ohe], axis=1).drop(cat_cols, axis=1)

   
    model_xgb = XGBRegressor(
        random_state=42,
        tree_method="hist",
        device="cuda",
        n_estimators=7000,
        early_stopping_rounds=300,
        **best_params  # use tuned parameters
    )
    model_xgb.fit(
        xtrain_xgb, ytrain,
        eval_set=[(xvalid_xgb, yvalid)],
        verbose=False
    )

    preds_valid_xgb = model_xgb.predict(xvalid_xgb)
    preds_test_xgb = model_xgb.predict(xtest_xgb)
    oof_xgb[xvalid.index] = preds_valid_xgb
    test_preds_xgb[:, fold] = preds_test_xgb

    
    # Train CatBoost
   
    train_pool = Pool(xtrain, label=ytrain, cat_features=cat_cols)
    valid_pool = Pool(xvalid, label=yvalid, cat_features=cat_cols)
    test_pool = Pool(xtest, cat_features=cat_cols)

    model_cat = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        task_type="GPU",
        eval_metric="RMSE",
        loss_function="RMSE",
        early_stopping_rounds=200,
        verbose=False
    )

    model_cat.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    preds_valid_cat = model_cat.predict(xvalid)
    preds_test_cat = model_cat.predict(xtest)
    oof_cat[xvalid.index] = preds_valid_cat
    test_preds_cat[:, fold] = preds_test_cat

   
    # Evaluate fold
    
    preds_blend = (preds_valid_xgb + preds_valid_cat) / 2     #simple averaging
    rmse = metrics.mean_squared_error(yvalid, preds_blend, squared=False)
    print(f"Fold {fold + 1} RMSE (avg XGB+Cat): {rmse:.5f}")
    rmse_scores.append(rmse)


# 5️⃣ Train Meta-model (Stacking)

meta_train = np.vstack([oof_xgb, oof_cat]).T
meta_test = np.vstack([test_preds_xgb.mean(axis=1), test_preds_cat.mean(axis=1)]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, df.accident_risk)
final_preds = meta_model.predict(meta_test)

print("\n Mean Fold RMSE:", np.mean(rmse_scores))





submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
submission["accident_risk"] = final_preds
submission.to_csv("stacking_xgb_catboost_regression.csv", index=False)
print("\n Submission saved: stacking_xgb_catboost_regression.csv")


