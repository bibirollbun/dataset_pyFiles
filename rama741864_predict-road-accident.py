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


import pandas as pd
df=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


from sklearn import model_selection


df["kfold"]=-1


kf=model_selection.KFold(n_splits=5,shuffle=True,random_state=42)

for fold,(train_idx,val_idx) in enumerate(kf.split(X=df)):
  df.loc[val_idx,"kfold"]=fold


df.to_csv("train_folds.csv",index=False)


import pandas
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import numpy as np
from xgboost import XGBRegressor


df=pd.read_csv('train_folds.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import optuna
import xgboost



def run(trial):


  fold=0
  learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
  reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
  reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
  subsample = trial.suggest_float("subsample", 0.2, 1.0)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
  max_depth = trial.suggest_int("max_depth", 3, 10)

  xtrain = df[df.kfold != fold].reset_index(drop=True)
  xvalid = df[df.kfold == fold].reset_index(drop=True)

  categorical_cols = ['road_type','lighting','weather','time_of_day']
  numeric_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
  bool_cols = ['road_signs_present','public_road','holiday','school_season']

  # Convert boolean columns
  xtrain[bool_cols] = xtrain[bool_cols].astype(int)
  xvalid[bool_cols] = xvalid[bool_cols].astype(int)


  # Extract target
  y_train = xtrain.accident_risk
  y_valid = xvalid.accident_risk

  # Drop target BEFORE preprocessing
  xtrain = xtrain.drop(columns=['accident_risk'])
  xvalid = xvalid.drop(columns=['accident_risk'])

  # Preprocessing
  preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
          ('num', StandardScaler(), numeric_cols),
          ('bool', 'passthrough', bool_cols)
      ]
  )

  X_train = preprocessor.fit_transform(xtrain)
  X_valid = preprocessor.transform(xvalid)


  # Model
  model = XGBRegressor(
      random_state=42,

      n_estimators=7000,
    learning_rate=learning_rate,
    reg_lambda=reg_lambda,
    reg_alpha=reg_alpha,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    max_depth=max_depth,
      tree_method='hist',
    predictor='auto',
    device='cuda'

  )

  model.fit(X_train, y_train)
  preds = model.predict(X_valid)



  rmse = np.sqrt(mean_squared_error(y_valid, preds))
  print(f"Fold {fold}: RMSE = {rmse}")

  return rmse




study = optuna.create_study(direction="minimize")
study.optimize(run, n_trials=1)#n_trails=100


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
final_test_predict=[]
final_valid_prediction={}
score=[]
for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)

    valid_ids=xvalid.id.values.tolist()

    categorical_cols = ['road_type','lighting','weather','time_of_day']
    numeric_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
    bool_cols = ['road_signs_present','public_road','holiday','school_season']

    # Convert boolean columns
    xtrain[bool_cols] = xtrain[bool_cols].astype(int)
    xvalid[bool_cols] = xvalid[bool_cols].astype(int)
    df_test[bool_cols] = df_test[bool_cols].astype(int)

    # Extract target
    y_train = xtrain.accident_risk
    y_valid = xvalid.accident_risk

    # Drop target BEFORE preprocessing
    xtrain = xtrain.drop(columns=['accident_risk'])
    xvalid = xvalid.drop(columns=['accident_risk'])

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
            ('bool', 'passthrough', bool_cols)
        ]
    )

    X_train = preprocessor.fit_transform(xtrain)
    X_valid = preprocessor.transform(xvalid)
    X_test = preprocessor.transform(df_test)

    # Model
    model = XGBRegressor(
        random_state=42, n_estimators=7000,
    learning_rate= 0.0019884236966102127,
    reg_lambda=0.00012232007593815082,
    reg_alpha= 0.002671905355439339,
    subsample=0.8350799721033705,
    colsample_bytree=0.99673695060659,
    max_depth=7,
      tree_method='hist',
    predictor='auto',
    device='cuda'
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    test_preds = model.predict(X_test)
    final_test_predict.append(test_preds)
    final_valid_prediction.update(dict(zip(valid_ids,preds)))


    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"Fold {fold}: RMSE = {rmse}")
    score.append(rmse)
print(np.mean(score),np.std(score))



sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


final_valid_prediction = pd.DataFrame.from_dict(final_valid_prediction, orient="index").reset_index()

final_valid_prediction.columns=["id","prediction"]




final_valid_prediction.to_csv("train_pred_1.csv",index=False)


sample_submission.accident_risk=np.mean(np.column_stack(final_test_predict),axis=1)
sample_submission.columns=["id","pred1"]
sample_submission.to_csv("test_pred_1.csv",index=False)


# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error

# df = pd.read_csv("/kaggle/working/train_folds.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
# sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# categorical_cols = ['road_type','lighting','weather','time_of_day']
# numeric_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
# bool_cols = ['road_signs_present','public_road','holiday','school_season']

# df[bool_cols] = df[bool_cols].astype(int)
# df_test[bool_cols] = df_test[bool_cols].astype(int)



# from xgboost import XGBRegressor

# model_name = "xgb"

# final_test_predict = []
# final_valid_prediction = {}
# scores = []

# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)

#     valid_ids = xvalid.id.values.tolist()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     xtrain = xtrain.drop(columns=['accident_risk'])
#     xvalid = xvalid.drop(columns=['accident_risk'])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ("num", StandardScaler(), numeric_cols),
#             ("bool", "passthrough", bool_cols),
#         ]
#     )

#     X_train = preprocessor.fit_transform(xtrain)
#     X_valid = preprocessor.transform(xvalid)
#     X_test = preprocessor.transform(df_test)

#     model = XGBRegressor(
#         random_state=42,
#         n_estimators=7000,
#         learning_rate=0.002,
#         reg_lambda=0.00012,
#         reg_alpha=0.0026,
#         subsample=0.835,
#         colsample_bytree=0.996,
#         max_depth=7,
#         tree_method='hist',
#         device='cuda'
#     )

#     model.fit(X_train, y_train)

#     preds = model.predict(X_valid)
#     test_preds = model.predict(X_test)

#     final_test_predict.append(test_preds)
#     final_valid_prediction.update(dict(zip(valid_ids, preds)))

#     rmse = np.sqrt(mean_squared_error(y_valid, preds))
#     print(f"Fold {fold}: RMSE = {rmse}")
#     scores.append(rmse)

# # Save train preds
# final_valid_prediction = pd.DataFrame.from_dict(final_valid_prediction, orient="index").reset_index()
# final_valid_prediction.columns = ["id","prediction"]
# final_valid_prediction.to_csv(f"train_pred_{model_name}.csv",index=False)

# # Save test preds
# sample_submission[f"pred_{model_name}"] = np.mean(np.column_stack(final_test_predict),axis=1)
# sample_submission[['id',f"pred_{model_name}"]].to_csv(f"test_pred_{model_name}.csv",index=False)

# print("âœ… XGB Done")



# from catboost import CatBoostRegressor

# model_name = "cat1"

# final_test_predict = []
# final_valid_prediction = {}
# scores = []

# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)

#     valid_ids = xvalid.id.values.tolist()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     xtrain = xtrain.drop(columns=['accident_risk'])
#     xvalid = xvalid.drop(columns=['accident_risk'])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ("num", StandardScaler(), numeric_cols),
#             ("bool", "passthrough", bool_cols),
#         ]
#     )

#     X_train = preprocessor.fit_transform(xtrain)
#     X_valid = preprocessor.transform(xvalid)
#     X_test = preprocessor.transform(df_test)

#     model = CatBoostRegressor(
#         iterations=4000,
#         learning_rate=0.01,
#         depth=6,
#         verbose=0,
#         random_seed=42
#     )

#     model.fit(X_train, y_train)

#     preds = model.predict(X_valid)
#     test_preds = model.predict(X_test)

#     final_test_predict.append(test_preds)
#     final_valid_prediction.update(dict(zip(valid_ids, preds)))

#     rmse = np.sqrt(mean_squared_error(y_valid, preds))
#     print(f"Fold {fold}: RMSE = {rmse}")
#     scores.append(rmse)

# final_valid_prediction = pd.DataFrame.from_dict(final_valid_prediction, orient="index").reset_index()
# final_valid_prediction.columns = ["id","prediction"]
# final_valid_prediction.to_csv(f"train_pred_{model_name}.csv",index=False)

# sample_submission[f"pred_{model_name}"] = np.mean(np.column_stack(final_test_predict),axis=1)
# sample_submission[['id',f"pred_{model_name}"]].to_csv(f"test_pred_{model_name}.csv",index=False)

# print("âœ… CatBoost Done")



# from sklearn.ensemble import RandomForestRegressor

# model_name = "rf1"

# final_test_predict = []
# final_valid_prediction = {}
# scores = []

# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)

#     valid_ids = xvalid.id.values.tolist()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     xtrain = xtrain.drop(columns=['accident_risk'])
#     xvalid = xvalid.drop(columns=['accident_risk'])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ("num", StandardScaler(), numeric_cols),
#             ("bool", "passthrough", bool_cols),
#         ]
#     )

#     X_train = preprocessor.fit_transform(xtrain)
#     X_valid = preprocessor.transform(xvalid)
#     X_test = preprocessor.transform(df_test)

#     model = RandomForestRegressor(
#         n_estimators=800,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1
#     )

#     model.fit(X_train, y_train)

#     preds = model.predict(X_valid)
#     test_preds = model.predict(X_test)

#     final_test_predict.append(test_preds)
#     final_valid_prediction.update(dict(zip(valid_ids, preds)))

#     rmse = np.sqrt(mean_squared_error(y_valid, preds))
#     print(f"Fold {fold}: RMSE = {rmse}")
#     scores.append(rmse)

# final_valid_prediction = pd.DataFrame.from_dict(final_valid_prediction, orient="index").reset_index()
# final_valid_prediction.columns = ["id","prediction"]
# final_valid_prediction.to_csv(f"train_pred_{model_name}.csv",index=False)

# sample_submission[f"pred_{model_name}"] = np.mean(np.column_stack(final_test_predict),axis=1)
# sample_submission[['id',f"pred_{model_name}"]].to_csv(f"test_pred_{model_name}.csv",index=False)

# print("âœ… RandomForest Done")



# from lightgbm import LGBMRegressor

# model_name = "lgbm1"

# final_test_predict = []
# final_valid_prediction = {}
# scores = []

# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)

#     valid_ids = xvalid.id.values.tolist()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     xtrain = xtrain.drop(columns=['accident_risk'])
#     xvalid = xvalid.drop(columns=['accident_risk'])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ("num", StandardScaler(), numeric_cols),
#             ("bool", "passthrough", bool_cols),
#         ]
#     )

#     X_train = preprocessor.fit_transform(xtrain)
#     X_valid = preprocessor.transform(xvalid)
#     X_test = preprocessor.transform(df_test)

#     model = LGBMRegressor(
#         n_estimators=5000,
#         learning_rate=0.005,
#         num_leaves=31,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     preds = model.predict(X_valid)
#     test_preds = model.predict(X_test)

#     final_test_predict.append(test_preds)
#     final_valid_prediction.update(dict(zip(valid_ids, preds)))

#     rmse = np.sqrt(mean_squared_error(y_valid, preds))
#     print(f"Fold {fold}: RMSE = {rmse}")
#     scores.append(rmse)

# final_valid_prediction = pd.DataFrame.from_dict(final_valid_prediction, orient="index").reset_index()
# final_valid_prediction.columns = ["id","prediction"]
# final_valid_prediction.to_csv(f"train_pred_{model_name}.csv",index=False)

# sample_submission[f"pred_{model_name}"] = np.mean(np.column_stack(final_test_predict),axis=1)
# sample_submission[['id',f"pred_{model_name}"]].to_csv(f"test_pred_{model_name}.csv",index=False)

# print("âœ… LGBM Done")



# test1 = pd.read_csv("test_pred_xgb.csv")
# test2 = pd.read_csv("test_pred_lgbm1.csv")
# test3 = pd.read_csv("test_pred_cat1.csv")
# test4 = pd.read_csv("test_pred_rf1.csv")



# # Load predictions from each model
# df1 = pd.read_csv("train_pred_xgb.csv").rename(columns={"prediction": "pred_xgb"})
# df2 = pd.read_csv("train_pred_lgbm1.csv").rename(columns={"prediction": "pred_lgbm1"})
# df3 = pd.read_csv("train_pred_cat1.csv").rename(columns={"prediction": "pred_cat1"})
# df4 = pd.read_csv("train_pred_rf1.csv").rename(columns={"prediction": "pred_rf1"})


# df = df.merge(df1, on="id", how="left")
# df = df.merge(df2, on="id", how="left")
# df = df.merge(df3, on="id", how="left")
# df = df.merge(df4, on="id", how="left")



# df_test = df_test.merge(test1, on="id", how="left")
# df_test = df_test.merge(test2, on="id", how="left")
# df_test = df_test.merge(test3, on="id", how="left")
# df_test = df_test.merge(test4, on="id", how="left")


# from sklearn.linear_model import LinearRegression


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd

# # Make sure your merged df and df_test exist
# useful_features = ["pred_xgb", "pred_lgbm1", "pred_cat1", "pred_rf1"]

# # Drop rows with any missing values (very important!)
# df = df.dropna(subset=useful_features + ["accident_risk", "kfold"]).reset_index(drop=True)
# df_test = df_test[useful_features].dropna().reset_index(drop=True)

# final_test_predict = []
# scores = []

# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)
#     xtest = df_test.copy()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     # Select only useful features
#     xtrain = xtrain[useful_features]
#     xvalid = xvalid[useful_features]

#     # Double-check dimensions before training
#     print(f"Fold {fold} -> Train: {xtrain.shape}, Valid: {xvalid.shape}")

#     # Train meta model
#     model = LinearRegression()
#     model.fit(xtrain, y_train)

#     preds_valid = model.predict(xvalid)
#     test_preds = model.predict(xtest)

#     # Sanity check
#     if len(y_valid) != len(preds_valid):
#         print(f"âš ï¸� Length mismatch in fold {fold}: y_valid={len(y_valid)}, preds_valid={len(preds_valid)}")
#         continue

#     rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
#     print(f"Fold {fold}: RMSE = {rmse:.5f}")
#     scores.append(rmse)

#     final_test_predict.append(test_preds)

# print(f"\nâœ… CV Mean RMSE = {np.mean(scores):.5f}")
# print(f"âœ… CV Std RMSE = {np.std(scores):.5f}")

# # Average test predictions
# stacked_preds = np.mean(np.column_stack(final_test_predict), axis=1)

# # Save final stacked predictions
# sub = pd.DataFrame({
#     "id": range(len(df_test)),   # or df_test.id if available
#     "stacked_pred": stacked_preds
# })
# sub.to_csv("test_pred_stacked.csv", index=False)

# print("\nğŸ�¯ Stacking completed successfully!")



# df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error

# # =====================================================
# # Define stacking features (use your exact column names)
# # =====================================================
# useful_features = ["pred_xgb", "pred_lgbm1", "pred_cat1", "pred_rf1"]

# # Drop missing rows for safety
# df = df.dropna(subset=useful_features + ["accident_risk", "kfold"]).reset_index(drop=True)
# df_test = df_test[useful_features].dropna().reset_index(drop=True)

# final_test_predict = []
# scores = []

# # =====================================================
# # 5-Fold Stacking with XGBoost
# # =====================================================
# for fold in range(5):
#     xtrain = df[df.kfold != fold].reset_index(drop=True)
#     xvalid = df[df.kfold == fold].reset_index(drop=True)
#     xtest = df_test.copy()

#     y_train = xtrain.accident_risk
#     y_valid = xvalid.accident_risk

#     xtrain = xtrain[useful_features]
#     xvalid = xvalid[useful_features]

#     print(f"Fold {fold} -> Train: {xtrain.shape}, Valid: {xvalid.shape}")

#     # -------------------------------
#     # Meta-model (XGBoost Regressor)
#     # -------------------------------
#     model = XGBRegressor(
       
#         random_state=fold,
#         tree_method='hist',  # Fast GPU-compatible method
#         device='cuda' ,
#         max_depth=7   # Use GPU if available
#     )

#     model.fit(
#         xtrain, y_train,
        
#     )

#     preds_valid = model.predict(xvalid)
#     test_preds = model.predict(xtest)

#     # Validation RMSE
#     rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
#     print(f"Fold {fold}: RMSE = {rmse:.5f}")
#     scores.append(rmse)

#     # Save test preds for this fold
#     final_test_predict.append(test_preds)




# sample_submission.accident_risk=np.mean(np.column_stack(final_test_predict),axis=1)
# sample_submission.columns=["id","pred1"]
# sample_submission.to_csv("test_pred_1.csv",index=False)




