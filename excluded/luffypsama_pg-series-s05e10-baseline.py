import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


#load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
train.head()


print(train.dtypes)


target = 'accident_risk'
useful_features = [c for c in train.columns if c not in ['id',target]]
cat_features = [c for c in train if train[c].dtypes == 'object']
num_features = [c for c in train if train[c].dtypes in ['int64','float64']]
train[cat_features] = train[cat_features].astype('category')


# test[cat_features] = test[cat_features].astype('category')
# X_test = test[useful_features]

# # Create an array to store predictions from each fold
# test_predictions = np.zeros(len(X_test))

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
#     print(f'---Fold {fold+1}/5---')

#     x_train,x_valid = train.iloc[train_idx][useful_features] , train.iloc[valid_idx][useful_features]
#     y_train,y_valid = train.iloc[train_idx][target], train.iloc[valid_idx][target]

#     model = XGBRegressor(
#         n_estimators=10000,
#         learning_rate=0.001,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         enable_categorical=True,
#         # device='cuda',
#         early_stopping_rounds=200,
#     )

#     model.fit(x_train,y_train,
#              eval_set = [(x_valid,y_valid)],
#              verbose = 1000
#              )

#     preds_valid = model.predict(x_valid)
#     rmse = mean_squared_error(y_valid,preds_valid,squared = False)
#     print(f"Fold {fold+1} RMSE: {rmse}")
#     # --- NEW PART: PREDICT ON TEST DATA AND ADD TO THE ARRAY ---
#     # We divide by the number of splits to get the average in the end
#     test_predictions += model.predict(X_test) / kf.get_n_splits()


# # --- CREATE SUBMISSION FILE AFTER THE LOOP ---
# submission_df = pd.DataFrame({'id': test['id'], target: test_predictions})
# submission_df.to_csv('submission_1.csv', index=False)

# print("\nSubmission file created successfully using averaged predictions!")
# print(submission_df.head())


import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# --- PREPARE YOUR DATA ---
# (Assuming 'train', 'cat_features', 'useful_features', and 'target' are already defined)
train[cat_features] = train[cat_features].astype('category')


# 1. DEFINE THE OBJECTIVE FUNCTION
# ---------------------------------
def objective(trial):
    # Define the hyperparameter search space
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'enable_categorical': True,
        'device': 'cuda',
        'random_state': 42,
    }

    # -- K-Fold Cross-validation --
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmses = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        x_train, x_valid = train.iloc[train_idx][useful_features], train.iloc[valid_idx][useful_features]
        y_train, y_valid = train.iloc[train_idx][target], train.iloc[valid_idx][target]

        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=100
        )
        
        model.fit(x_train, y_train,
                  eval_set=[(x_valid, y_valid)],
                  verbose=False)
        
        preds_valid = model.predict(x_valid)
        # =============================================================

        rmse = mean_squared_error(y_valid, preds_valid, squared=False)
        fold_rmses.append(rmse)

    # Return the average RMSE across all folds
    return np.mean(fold_rmses)


# 2. CREATE AND RUN THE OPTUNA STUDY
# ---------------------------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# 3. PRINT THE BEST RESULTS
# --------------------------
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial

print(f"  Value (RMSE): {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

best_params = best_trial.params







