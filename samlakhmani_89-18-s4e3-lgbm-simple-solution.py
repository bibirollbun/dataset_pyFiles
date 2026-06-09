import pandas as pd 

train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')


def feature_engineering(data):

    data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
    data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] -data['Steel_Plate_Thickness'].min()) / (data['Steel_Plate_Thickness'].max() - data['Steel_Plate_Thickness'].min())
    data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']

    # features_to_drop = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']
    # data = data.drop(features_to_drop,axis=1)

    return data


train = feature_engineering(train)
test = feature_engineering(test)


target_list = [
    'Pastry', 
    'Z_Scratch', 
    'K_Scatch', 
    'Stains',
    'Dirtiness', 
    'Bumps', 
    'Other_Faults'
]


import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import optuna.visualization as vis
from sklearn.utils.class_weight import compute_class_weight


from sklearn.model_selection import StratifiedKFold


def optimize_for_target(train, target):

    X = train.drop(columns=target_list + ['id'], errors='ignore')
    y = train[target].values  

        
    def objective(trial):
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pred_oof = np.zeros_like(y, dtype=float)
        
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y[train_index], y[val_index]

            weights_train = [1/np.mean(y_train, axis=0)]*len(y_train) 
            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(0),  # silence logging
                ],
            )

            pred = model.predict(X_val, num_iteration=model.best_iteration)
            pred_oof[val_index] = pred

        auc = roc_auc_score(y, pred_oof)

        return auc


    # ----- Run Optuna study -----
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    return study.best_trial.params


best_params_per_target = {}

for target in target_list :
    
    best_params_per_target[target] = optimize_for_target(train, target)


best_params_per_target


def get_oof_rocauc_and_test_proba(train, test, target, params, no_of_folds = 5):
    
    X = train.drop(columns=target_list + ['id'], errors='ignore')
    y = train[target].values
    
    X_test = test.drop(['id'],axis=1).values 

    params = params
    params["objective"] = "binary"
    params["metric"] = "auc"
    params["boosting_type"] = "gbdt"
    params["verbosity"] = -1
    params["n_estimators"] = 500
    
    skf = StratifiedKFold(n_splits=no_of_folds, shuffle=False)
    pred_oof = np.zeros_like(y, dtype=float)
    pred_test = np.zeros(len(X_test), dtype=float)
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        weights_train = [1/np.mean(y_train, axis=0)]*len(y_train) 
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weights_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(0),  # silence logging
            ],
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_oof[val_index] = val_pred
        pred_test = pred_test + test_pred/no_of_folds
        
    auc = roc_auc_score(y, pred_oof)
    
    return auc, pred_test, pred_oof


result = np.zeros((len(test),len(target_list)))
train_pred = np.zeros((len(train), len(target_list)))
counter = 0 
for target,best_params in best_params_per_target.items():
    
    auc, pred_test, pred_train = get_oof_rocauc_and_test_proba(train, test, target, best_params)
    print(f"Expected AUC for Label {target} - {auc}")
    
    train_pred[:,counter] = pred_train
    result[:,counter] = pred_test
    counter+=1


test[target_list] = result


train[[x+"__proba" for x in target_list]] = train_pred


submission = test[['id']+target_list]


submission.to_csv("submission.csv",index=False)

