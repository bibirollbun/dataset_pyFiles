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
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

SEED = 123

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')



target = 'accident_risk'
features = [col for col in train_df.columns if col != target]



for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].astype('category').cat.codes
        test_df[col] = test_df[col].astype('category').cat.codes



print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(train_df[features],train_df[target],test_size=0.2,random_state=SEED)


xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 100
}



model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_train,y_train,
    eval_set=[(X_test,y_test)],

    verbose = 500
    
)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))



print("Best iteration:", model.best_iteration)
print("RMSE: %.4f" % rmse)



import optuna
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 10000,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
        "max_depth": trial.suggest_int('max_depth', 3, 8),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        "subsample": trial.suggest_float('subsample', 0.6, 0.9),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 0.9),
        "min_child_weight": trial.suggest_int('min_child_weight', 1, 10),
        "reg_alpha": trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df))
    oof_rmse_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_df)):
        X_train, y_train = train_df.loc[train_idx, features], train_df.loc[train_idx, target]
        X_test, y_test = train_df.loc[test_idx, features], train_df.loc[test_idx, target]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds = model.predict(X_test)
        oof_preds[test_idx] = preds
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        oof_rmse_scores.append(rmse)

        print(f"Fold {fold + 1} RMSE: {rmse:.5f}")

    avg_rmse = np.mean(oof_rmse_scores)
    print(f"Trial {trial.number} - Avg RMSE: {avg_rmse:.5f}")
    return avg_rmse


# تشغيل عملية البحث باستخدام Optuna
study = optuna.create_study(direction='minimize', study_name='xgb_regression_tuning')
study.optimize(objective, n_trials=5)

print("\nBest Parameters:")
print(study.best_params)
print(f"Best RMSE: {study.best_value:.5f}")



best_params = study.best_params
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 10000,
    'random_state': SEED,
    'n_jobs': -1,
    'tree_method': 'hist',
    "early_stopping_rounds": 100
})

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

X = train_df[features]
y = train_df[target]

oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))
models = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"===== Fold {fold+1} =====")
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    model = xgb.XGBRegressor(**best_params)
    model. fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    preds = model.predict(X_test)
    oof_predictions[test_idx] = preds
    
    test_preds = model.predict(test_df)
    test_predictions += test_preds / kf.get_n_splits()
    
    models.append(model)

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\nTotal RMSE on OOF predictions: {oof_rmse:.5f}")



from sklearn.linear_model import LinearRegression


meta_X_train = pd.DataFrame({'xgb_pred': oof_predictions})
meta_X_test = pd.DataFrame({'xgb_pred': test_predictions})
meta_y_train = train_df[target]

meta_model = LinearRegression()
meta_model.fit(meta_X_train, meta_y_train)

final_predictions = meta_model.predict(meta_X_test)

print("✅ Final predictions have been generated from the stacking model.")

submission_df = pd.DataFrame({
    'id': test_df["id"],
    'accident_risk': final_predictions
})
submission_df.to_csv('submission.csv', index=False)

print("The submission file has been saved successfully. ✅ 'submission.csv'")



submission_df.head(10)




