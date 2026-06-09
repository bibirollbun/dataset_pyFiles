import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import optuna 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import lightgbm as lgb
import xgboost as xgb
import catboost as catb

warnings.filterwarnings('ignore')
sub_file = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
training_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
testing_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print("<--------TRAINING DATA--------->")
display(training_data.head(3))

print('\n')
print("\n<--------TESTING DATA---------->")
display(testing_data.head(3))


print('Train size : ' , training_data.shape)
print('Test size : ' , testing_data.shape)


display(training_data.describe())
training_data.info()
training_data.isnull().sum()


train_x = training_data.drop(['BeatsPerMinute' , 'id'] , axis=1)
train_y = training_data['BeatsPerMinute']

test_x = testing_data.drop('id' , axis=1)


def objective(trial):
    params = {
    'learning_rate' : trial.suggest_float('learning_rate' , 0.01 , 0.1 , log=True),
    'num_leaves' : trial.suggest_int('num_leaves' , 5 , 50),
    'max_depth' : trial.suggest_int('max_depth' , 3 , 15),
    'reg_alpha' : trial.suggest_float('reg_alpha' , 0.01 , 0.1 , log=True),
    'reg_lambda' : trial.suggest_float('reg_lambda' , 0.01 , 0.1 , log=True),
    'subsample' : trial.suggest_float('subsample' , 0 , 1)
             }

    lgb_model = lgb.LGBMRegressor(**params)

    val_rmse = 0.0
    oof_preds = np.zeros(len(train_x))

    kf = KFold(n_splits = 5 , shuffle=True , random_state=0)

    for train_idx , val_idx in kf.split(train_x , train_y):

        X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
        Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

        lgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)])
        val_pred = lgb_model.predict(X_val)
        oof_preds[val_idx] = val_pred
    
        val_rmse = np.sqrt(mean_squared_error(Y_val , val_pred))
        return val_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5, show_progress_bar=True)
best_params = study.best_trial.params
print("\nBest Hyperparameters from Optuna:")
print(best_params)


lgb_model = lgb.LGBMRegressor(
    learning_rate = 0.0137,
    num_leaves = 43,
    max_depth = 7,
    reg_alpha = 0.0172,
    reg_lambda = 0.0134,
    subsample = 0.454
)

val_rmse = 0.0
oof_preds = np.zeros(len(train_x))
lgb_models , lgb_scores = [] , []
kf = KFold(n_splits = 5 , shuffle=True , random_state=0)

for train_idx , val_idx in kf.split(train_x , train_y):
    print('\nFold:' , len(lgb_models) + 1)
    X_train , X_val = train_x.iloc[train_idx] , train_x.iloc[val_idx]
    Y_train , Y_val = train_y.iloc[train_idx] , train_y.iloc[val_idx]

    lgb_model.fit(X_train , Y_train , eval_set = [(X_val , Y_val)])
    val_pred = lgb_model.predict(X_val)
    oof_preds[val_idx] = val_pred

    model_rmse = np.sqrt(mean_squared_error(Y_val , val_pred))

    lgb_models.append(lgb_model) , lgb_scores.append(model_rmse)

print('\nScores :', lgb_scores)


lgb_test_preds = sum(lgb_model.predict(test_x) for lgb_model in lgb_models) / len(lgb_models)
final_preds = lgb_test_preds
final_preds


submission = pd.DataFrame({'id': testing_data['id'], 'BeatsPerMinute': final_preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

