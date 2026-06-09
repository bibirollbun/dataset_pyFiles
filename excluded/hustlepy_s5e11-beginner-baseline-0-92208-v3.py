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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing,model_selection,linear_model,metrics,ensemble
from xgboost import XGBClassifier


df = pd.read_csv("/kaggle/input/s5e11-5foldcv/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df.columns


features=[c for c in df.columns if c not in ('id','loan_paid_back','kfold')] #useful features by which we will be training our model on
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  #subsetting numerical columns from the 'useful features'
cat_cols=df[features].select_dtypes(include='object').columns.tolist() #subsetting categorical columns from the 'useful features'


df_test=df_test[features] 


final_predictions=[]  #here, we will append our test predictions
auc_scores=[]
for fold in range(5):
    xtrain=df[df.kfold!=fold].reset_index(drop=True)   
    xvalid=df[df.kfold==fold].reset_index(drop=True)
    xtest=df_test.copy()

    ytrain=xtrain.loan_paid_back.values
    yvalid=xvalid.loan_paid_back.values

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


    model=XGBClassifier(random_state=42,tree_method = "hist", device = "cuda",n_estimators=7000,
                       early_stopping_rounds=300,use_label_encoder=False,eval_metric="auc")
    
    model.fit(xtrain,ytrain,eval_set=[(xvalid, yvalid)],
        verbose=False)
    preds_valid = model.predict_proba(xvalid)[:, 1]
    test_preds = model.predict_proba(xtest)[:, 1]
    auc = metrics.roc_auc_score(yvalid, preds_valid)
    auc_scores.append(auc)
    print(f'Fold{str(fold+1)} AUC: {auc}')
    

print(np.mean(auc_scores))


import optuna
import numpy as np
from sklearn import preprocessing, metrics
from xgboost import XGBClassifier

def objective(trial):
    fold = 0  # only one fold(since it will be computationally expensive for all loops)

    
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.loan_paid_back.values
    yvalid = xvalid.loan_paid_back.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

    
    ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe = ohe.transform(xvalid[cat_cols])
    xtest_ohe = ohe.transform(xtest[cat_cols])

    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f"ohe_{i}" for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f"ohe_{i}" for i in range(xtest_ohe.shape[1])])

    xtrain = pd.concat([xtrain, xtrain_ohe], axis=1)
    xvalid = pd.concat([xvalid, xvalid_ohe], axis=1)
    xtest = pd.concat([xtest, xtest_ohe], axis=1)

    xtrain = xtrain.drop(cat_cols, axis=1)
    xvalid = xvalid.drop(cat_cols, axis=1)
    xtest = xtest.drop(cat_cols, axis=1)

    
    scaler = preprocessing.StandardScaler()
    xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols] = scaler.transform(xvalid[num_cols])
    xtest[num_cols] = scaler.transform(xtest[num_cols])

    
    model = XGBClassifier(
        random_state=42,
        tree_method="hist",
        device="cuda",
        n_estimators=7000,
        early_stopping_rounds=300,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric="auc"
    )

    model.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], verbose=False)
    preds_valid = model.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(yvalid, preds_valid)

    return auc 




study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\n Best trial:")
print(f"Best AUC: {study.best_value:.5f}")
print(f"Best Params: {study.best_params}")


import joblib

joblib.dump(study, "/kaggle/working/optuna_study.pkl")
print("Optuna study saved successfully!")



# Use best hyperparameters from your Optuna study
best_params = study.best_params

# Add constants or defaults that weren't tuned
best_params.update({
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "n_estimators": 7000,
    "early_stopping_rounds": 300,
    "use_label_encoder": False,
    "eval_metric": "auc"
})

final_predictions = []
auc_scores = []

for fold in range(5):
    print(f"=== Fold {fold} ===")

    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.loan_paid_back.values
    yvalid = xvalid.loan_paid_back.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

    # One-hot encoding
    ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe = ohe.transform(xvalid[cat_cols])
    xtest_ohe = ohe.transform(xtest[cat_cols])

    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    # Combine encoded and numerical features
    xtrain = pd.concat([xtrain, xtrain_ohe], axis=1).drop(cat_cols, axis=1)
    xvalid = pd.concat([xvalid, xvalid_ohe], axis=1).drop(cat_cols, axis=1)
    xtest = pd.concat([xtest, xtest_ohe], axis=1).drop(cat_cols, axis=1)

    # Scale numeric columns
    scaler = preprocessing.StandardScaler()
    xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols] = scaler.transform(xvalid[num_cols])
    xtest[num_cols] = scaler.transform(xtest[num_cols])

   
    model = XGBClassifier(**best_params)
    model.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], verbose=False)

    preds_valid = model.predict_proba(xvalid)[:, 1]
    preds_test = model.predict_proba(xtest)[:, 1]

    auc = metrics.roc_auc_score(yvalid, preds_valid)
    auc_scores.append(auc)
    final_predictions.append(preds_test)

    print(f"Fold {fold + 1} AUC: {auc:.5f}")


print(f"\nMean AUC across folds: {np.mean(auc_scores):.5f}")
final_test_predictions = np.mean(np.column_stack(final_predictions), axis=1)





sample_submission.loan_paid_back = final_test_predictions
sample_submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")




