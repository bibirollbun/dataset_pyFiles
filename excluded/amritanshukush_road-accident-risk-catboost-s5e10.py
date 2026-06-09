import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import catboost as catb

import optuna


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
samp_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train Data:")
display(train_data.head())

print("\nTest Data:")
display(test_data.head())

print("\nSample Submission:")
display(samp_sub.head())


print("!!!! TRAIN DATA STATISTICS !!!!\n")

print("1. Shape : " , train_data.shape)
print("=====================================")

display(train_data.describe().T)
print("=====================================")

print("2. Column Info :\n")
display(train_data.info())
print("=====================================")

print("\n3.Null Values Info : \n")
display(train_data.isnull().sum())


TARGET = 'accident_risk'

train_x = train_data.drop([TARGET , 'id'] , axis=1)
train_y = train_data[TARGET]

test_x = test_data.drop('id' , axis=1)

CAT_COLS = train_x.select_dtypes(include=['object' , 'bool']).columns.tolist()
NUM_COLS = train_x.select_dtypes(include=['int' , 'float']).columns.tolist()

print("Categorical Features : " , CAT_COLS)
print("Numerical Features : " , NUM_COLS)


def rmse(y_true , y_pred):
    return np.sqrt(mean_squared_error(y_true , y_pred))


# Optuna layout
def catb_objective(trial):
    params = {
        'learning_rate' : trial.suggest_float('learning_rate' , 0.01 , 0.1 , log = True),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg' , 0.01 , 0.1 , log = True),
        'colsample_bylevel' : trial.suggest_float('colsample_bylevel' , 0 , 1),
        'depth' : trial.suggest_int('depth' , 1 , 10),
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf' , 3 , 20),
        'subsample' : trial.suggest_float('subsample' , 0 , 1)
    }

    catb_model = catb.CatBoostRegressor(**params)

    val_rmse = 0.0
    oof_preds = np.zeros(len(train_x))
    fold_rmses=[]

    kf = KFold(n_splits = 5 , shuffle=True , random_state=0)

    for train_idx , val_idx in kf.split(train_x , train_y):

        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

        train_pool = catb.Pool(X_train , Y_train , cat_features = CAT_COLS)
        test_pool = catb.Pool(X_val , Y_val , cat_features = CAT_COLS)

        catb_model.fit(train_pool , eval_set = test_pool , verbose=100)
        val_pred = catb_model.predict(X_val)
        oof_preds[val_idx] = val_pred
    
        val_rmse = rmse(Y_val , val_pred)
        fold_rmses.append(val_rmse)

    return float(np.mean(fold_rmses))

# To run, you can un-comment below lines and run it
        
#study = optuna.create_study(direction='minimize')
#study.optimize(catb_objective, n_trials=5, show_progress_bar=True)
#best_params = study.best_trial.params
#print("\nBest Hyperparameters from Optuna:")
#print(best_params)


catb_params = {
    'learning_rate' : 0.06846073551293783,
    'l2_leaf_reg' : 0.03358471172334371,
    'colsample_bylevel' : 0.4681301004890497,
    'depth' : 8,
    'min_data_in_leaf' : 12,
    'subsample' : 0.6306114676661142
}

catb_model = catb.CatBoostRegressor(**catb_params)

oof_preds = np.zeros(len(train_x))
catb_models , catb_scores = [] , []
kf = KFold(n_splits = 5 , shuffle=True , random_state=0)

for train_idx , val_idx in kf.split(train_x , train_y):
    print('\nFold:' , len(catb_models) + 1)
    X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
    Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

    train_pool = catb.Pool(X_train , Y_train , cat_features = CAT_COLS)
    test_pool = catb.Pool(X_val , Y_val , cat_features = CAT_COLS)

    catb_model.fit(train_pool , eval_set = test_pool , verbose=100)
    val_pred = catb_model.predict(X_val)
    oof_preds[val_idx] = val_pred
    
    val_rmse = rmse(Y_val , val_pred)
    
    catb_models.append(catb_model) , catb_scores.append(val_rmse)

print('\nScores :', catb_scores)


test_preds = sum(catb_model.predict(test_x) for catb_model in catb_models) / len(catb_models)


submission = pd.DataFrame({'id': test_data['id'], 'accident_risk': test_preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())




