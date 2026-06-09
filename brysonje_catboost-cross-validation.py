import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import StratifiedKFold
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score


from catboost import CatBoostClassifier, Pool


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
X = pd.read_csv(train_path).drop("diagnosed_diabetes", axis = 1)
y = pd.read_csv(train_path)["diagnosed_diabetes"]
X_test = pd.read_csv(test_path)


X.drop("gender", axis = 1, inplace = True)
X_test.drop("gender", axis = 1, inplace = True)


## AGE & TRIGLYCERIDES
X["at_risk"] = X["triglycerides"] + X["triglycerides"] * X["age"] * 0.30
X["at_risk"] = X["at_risk"].astype(float)

X_test["at_risk"] = X_test["triglycerides"] + X_test["triglycerides"] * X_test["age"] * 0.30
X_test["at_risk"] = X_test["at_risk"].astype(float)

## AGE & FAMILY HISTORY INTERACTION
X["af_risk"] = X["family_history_diabetes"] + X["family_history_diabetes"] * X["age"] * 0.15
X["af_risk"] = X["af_risk"].astype(float)

X_test["af_risk"] = X_test["family_history_diabetes"] + X_test["family_history_diabetes"] * X_test["age"] * 0.15
X_test["af_risk"] = X_test["af_risk"].astype(float)


cat_feats = X.select_dtypes(include = "object").columns.tolist()
for feat in cat_feats:
    X[feat] = X[feat].astype("category").cat.codes
    X_test[feat] = X_test[feat].astype("category").cat.codes


kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 48)

todas_las_predicciones = []
scores_accuracy = []

oof_predictions = np.zeros(len(X))

scores_folds = []
predicciones_test_por_modelo = []


cbc_params = {"iterations": 3000,
              'learning_rate': 0.04359224667234545,
              'depth': 6,
              'l2_leaf_reg': 7.678939079305491,
              'random_strength': 0.6252073339200809,
              'min_data_in_leaf': 21,
              "verbose": 0,
              "task_type": "GPU",
              "devices": "0"}
              ## 'subsample': 0.9251861002711826}


cat_feats = X.select_dtypes("object").columns.tolist()


cat_features_indices = [X.columns.get_loc(col) for col in cat_feats]

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # Preparamos los datos del fold
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # 2. Creamos los Pools pasando los ÍNDICES numéricos
    train_pool = Pool(X_train_fold, y_train_fold, cat_features = cat_features_indices)
    val_pool = Pool(X_val_fold, y_val_fold, cat_features = cat_features_indices)

    # 3. Definimos el modelo y le pasamos los mismos índices
    cbc_model = CatBoostClassifier(**cbc_params, cat_features = cat_features_indices)

    # 4. Entrenamos
    cbc_model.fit(train_pool, eval_set = val_pool)

    prob_val = cbc_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = prob_val
    
    pred_val_clase = (prob_val > 0.5).astype(int) # Convertimos probabilidad a 0 o 1
    score_fold = accuracy_score(y_val_fold, pred_val_clase)
    scores_folds.append(score_fold)
    
    pred_test = cbc_model.predict_proba(X_test[X.columns])[:, 1]
    predicciones_test_por_modelo.append(pred_test)
    
    print(f"--> Fold {fold + 1} listo. Accuracy: {score_fold:.4f}")
    
    print(f"--> Fold {fold + 1} entrenado con éxito.")


prediccion_final_test = np.mean(predicciones_test_por_modelo, axis = 0)


sub_path = "/kaggle/input/playground-series-s5e12/sample_submission.csv"
df_sub = pd.read_csv(sub_path)
df_sub["diagnosed_diabetes"] = prediccion_final_test


df_sub.to_csv("submission.csv", index = False)




