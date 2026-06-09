import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import warnings

warnings.filterwarnings("ignore")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!head /kaggle/input/playground-series-s5e3/sample_submission.csv
!head /kaggle/input/playground-series-s5e3/train.csv
!head /kaggle/input/playground-series-s5e3/test.csv


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")#.set_index("id")
train.head()


train.describe()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")#.set_index("id")
test.head()




# Define feature columns and target
features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
            'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target = 'rainfall'



train.isnull().sum()



test.isnull().sum()


test[test.isnull().any(axis=1)]


# Impute missing values (using mean strategy)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(train[features]), columns=features)
y = train[target]



X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def objective(params):
    model_type = params['model_type']
    
    if model_type == 'xgb':
        # Prepare parameters for XGBoost
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        params['seed'] = 42
        params['max_depth'] = int(params['max_depth'])  
        num_round = int(params['n_estimators'])
        params.pop('n_estimators')
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        model = xgb.train(params, dtrain, num_round,
                          evals=[(dvalid, 'eval')],
                          early_stopping_rounds=10,
                          verbose_eval=False
                         )
        preds = model.predict(dvalid)
    
    else:
        # Prepare parameters for LightGBM
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['verbose'] = -1
        params['max_depth'] = int(params['max_depth'])
        params['num_leaves'] = int(params['num_leaves'])
        num_round = int(params['n_estimators'])
        params.pop('n_estimators')
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
        
        model = lgb.train(params, lgb_train, num_round,
                          valid_sets=[lgb_valid],
                          #early_stopping_rounds=10,
                          #verbose_eval=False
                         )
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
    
    auc = roc_auc_score(y_valid, preds)
    print(f"Model: {model_type} | AUC: {auc:.4f}")
    return {'loss': -auc, 'status': STATUS_OK}



space = hp.choice('classifier', [
    {
        'model_type': 'xgb',
        'n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 25),
        'max_depth': hp.quniform('xgb_max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('xgb_learning_rate', np.log(0.01), np.log(0.2)),
        'subsample': hp.uniform('xgb_subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.7, 1.0),
        'gamma': hp.uniform('xgb_gamma', 0, 5)
    },
    {
        'model_type': 'lgb',
        'n_estimators': hp.quniform('lgb_n_estimators', 100, 500, 25),
        'max_depth': hp.quniform('lgb_max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.01), np.log(0.2)),
        'subsample': hp.uniform('lgb_subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.7, 1.0),
        'num_leaves': hp.quniform('lgb_num_leaves', 20, 100, 5)
    }
])


trials = Trials()
rng = np.random.default_rng(42)
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,  # Increase this number for a more thorough search
            trials=trials,
            rstate=rng)

print("Best hyperparameters:", best)



from hyperopt import space_eval
# 'best' is the raw result from fmin
best_params = space_eval(space, best)
print("Best hyperparameters:", best_params)


# Use the same imputer as during training
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(train[features]), columns=features)
y = train[target]

# Preprocess test set in the same way
X_test = pd.DataFrame(imputer.transform(test[features]), columns=features)


best_params


if best_params['model_type'] == 'xgb':
    # For XGBoost: cast integer parameters
    best_params['max_depth'] = int(best_params['max_depth'])
    n_estimators = int(best_params.pop('n_estimators'))
    
    # Set additional parameters
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['seed'] = 42
    
    # Prepare full training data as a DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    # Train the model on full training data
    model = xgb.train(best_params, dtrain, num_boost_round=n_estimators)
    
    # Prepare test data and predict
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)

elif best_params['model_type'] == 'lgb':
    # For LightGBM: cast integer parameters
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    n_estimators = int(best_params.pop('n_estimators'))
    
    # Set additional parameters
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbose'] = -1
    
    # Create a LightGBM dataset with the full training data
    lgb_train = lgb.Dataset(X, label=y)
    # Train the model on full training data
    model = lgb.train(best, lgb_train, num_boost_round=n_estimators)
    
    # Predict on test data
    preds = model.predict(X_test)


X_test = test[features]
X_test


# Predict probabilities on the test set

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)



submission

