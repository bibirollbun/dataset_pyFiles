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


import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd

# Adjust the file paths according to the Kaggle competition's dataset structure.
train_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_path = '/kaggle/input/playground-series-s5e4/test.csv'

try:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
except Exception as e:
    print("Error reading files:", e)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress formatting warnings (optional)
warnings.filterwarnings("ignore", message="invalid value encountered in")

train_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_path = '/kaggle/input/playground-series-s5e4/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic structure and summary
print("Train DataFrame type:", type(train))
print("Training Data Sample:")
print(train.head())
print("\nTrain Info:")
print(train.info())
print("\nTrain Summary Statistics:")
print(train.describe())

# Check missing values
print("\nMissing Values per Column in Train:")
print(train.isnull().sum())

print("\nMissing Values per Column in Test:")
print(test.isnull().sum())



# Visualize distributions of key numerical columns
train[['Episode_Length_minutes', 'Host_Popularity_percentage', 
       'Guest_Popularity_percentage', 'Listening_Time_minutes']].hist(bins=50, figsize=(15, 8))
plt.tight_layout()
plt.show()


# Impute numeric columns using the median:
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)


# Check missing values
print("\nMissing Values per Column in Train:")
print(train.isnull().sum())

print("\nMissing Values per Column in Test:")
print(test.isnull().sum())


from sklearn.preprocessing import LabelEncoder

# For purely categorical columns with a limited number of unique values:
cat_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

drop_cols = ['Podcast_Name', 'Episode_Title']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)


# drop_cols = ['Podcast_Name_length',	'Episode_Title_length',	'Episode_Title_word_count']
# train = train.drop(columns=drop_cols)
# test = test.drop(columns=drop_cols)


train.head()



# We assume "id" is an identifier and "Listening_Time_minutes" is our target.
# Therefore, our final feature list consists of all columns except 'id' and the target.

target = 'Listening_Time_minutes'
features = [col for col in train.columns if col not in ['id', target]]

# Print the final feature list to verify:
print("Final feature list:", features)


# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np

# target = 'Listening_Time_minutes'
# X = train[features]
# y = train[target]
# X_test = test[features]

# folds = KFold(n_splits=5, shuffle=True, random_state=42)
# oof_preds = np.zeros(X.shape[0])
# test_preds = np.zeros(X_test.shape[0])

# lgb_params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.01,
#     'num_leaves': 31,
#     'verbose': -1,
#     'seed': 42,
# }

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#     print(f"Starting Fold {fold_ + 1}")
#     X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
#     X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
#     train_data = lgb.Dataset(X_train, label=y_train)
#     valid_data = lgb.Dataset(X_valid, label=y_valid)
    
#     # Replace early_stopping_rounds with a callback for early stopping
#     clf = lgb.train(
#         lgb_params,
#         train_data,
#         num_boost_round=10000,
#         valid_sets=[train_data, valid_data],
#         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
#     )
    
#     oof_preds[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
#     test_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

# cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
# print("LightGBM CV RMSE:", cv_rmse)


# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error

# target = 'Listening_Time_minutes'
# X = train[features]
# y = train[target]
# X_test = test[features]

# folds = KFold(n_splits=5, shuffle=True, random_state=42)
# xgb_params = {
#     'objective': 'reg:squarederror',  # 'reg:linear' is deprecated
#     'eval_metric': 'rmse',
#     'eta': 0.01,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'seed': 42,
# }

# oof_preds_xgb = np.zeros(X.shape[0])
# test_preds_xgb = np.zeros(X_test.shape[0])

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#     print(f"XGBoost Fold {fold_ + 1}")
#     dtrain_fold = xgb.DMatrix(X.iloc[trn_idx], label=y.iloc[trn_idx])
#     dvalid_fold = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
    
#     watchlist = [(dtrain_fold, 'train'), (dvalid_fold, 'eval')]
#     model = xgb.train(
#         xgb_params, 
#         dtrain_fold,
#         num_boost_round=10000,
#         evals=watchlist,
#         early_stopping_rounds=100,
#         verbose_eval=100
#     )
    
#     oof_preds_xgb[val_idx] = model.predict(xgb.DMatrix(X.iloc[val_idx]), iteration_range=(0, model.best_iteration))
#     test_preds_xgb += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / folds.n_splits

# cv_rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
# print("XGBoost CV RMSE:", cv_rmse_xgb)


# ensemble_test_preds = (test_preds + test_preds_xgb)/2


# submission = pd.DataFrame({
#     'id': test['id'],
#     'Listening_Time_minutes': ensemble_test_preds  # or use test_preds from your preferred model
# })
# submission.to_csv('submission.csv', index=False)
# print("Submission file created!")


# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error

# # Assume train, test, target, and features are already defined and preprocessed.
# target = 'Listening_Time_minutes'
# X = train[features]
# y = train[target]
# X_test = test[features]

# # Setting up 5-fold cross-validation
# folds = KFold(n_splits=5, shuffle=True, random_state=42)
# oof_preds_lgba = np.zeros(X.shape[0])
# test_preds_lgba = np.zeros(X_test.shape[0])

# # Updated hyperparameter dictionary for LightGBM
# lgb_params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',       # You can try 'dart' or 'goss' as alternatives
#     'learning_rate': 0.005,        # Lower learning rate for finer training
#     'num_leaves': 63,              # Increased complexity; you might try [31, 63, 127]
#     'feature_fraction': 0.8,       # Random fraction of features to use per iteration
#     'bagging_fraction': 0.8,       # Random subset of data for each iteration
#     'bagging_freq': 5,             # Do bagging every 5 iterations
#     'min_data_in_leaf': 20,        # Minimum number of samples per leaf (try values like 20, 50)
#     'reg_alpha': 0.1,              # L1 regularization
#     'reg_lambda': 0.1,             # L2 regularization
#     'verbose': -1,
#     'seed': 42,
# }

# # Training loop
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#     print(f"\n[LightGBM] Starting Fold {fold_ + 1}")
#     X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
#     X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
#     train_data = lgb.Dataset(X_train, label=y_train)
#     valid_data = lgb.Dataset(X_valid, label=y_valid)
    
#     clf = lgb.train(
#         lgb_params,
#         train_data,
#         num_boost_round=20000,  # set a high maximum; early stopping will halt training when improvement stalls
#         valid_sets=[train_data, valid_data],
#         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
#     )
    
#     best_iter = clf.best_iteration
#     print(f"Fold {fold_ + 1} best iteration: {best_iter}")
#     oof_preds_lgba[val_idx] = clf.predict(X_valid, num_iteration=best_iter)
#     test_preds_lgba += clf.predict(X_test, num_iteration=best_iter) / folds.n_splits

# cv_rmse_lgba = np.sqrt(mean_squared_error(y, oof_preds))
# print(f"\n[LightGBM] Overall CV RMSE: {cv_rmse:.4f}")


# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb

# # Assume train, test, target, and features are already defined and preprocessed.
# target = 'Listening_Time_minutes'
# X = train[features]
# y = train[target]
# X_test = test[features]

# # Setting up 5-fold cross-validation
# folds = KFold(n_splits=5, shuffle=True, random_state=42)
# xgb_params = {
#     'objective': 'reg:squarederror',  # Current regression objective
#     'eval_metric': 'rmse',
#     'eta': 0.005,                     # Lower learning rate for finer tuning
#     'max_depth': 7,                   # Adjust depth as needed (e.g., 6-10)
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'min_child_weight': 10,           # Adjust to control tree complexity
#     'seed': 42,
#     'tree_method': 'hist',            # Use histogram method for GPU training
#     'device': 'cuda',                 # Specify GPU usage
#     'nthread': 8,                     # Use parallel threads as appropriate
# }

# oof_preds_xgbb = np.zeros(X.shape[0])
# test_preds_xgbb = np.zeros(X_test.shape[0])

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#     print(f"\n[XGBoost] Starting Fold {fold_ + 1}")
#     dtrain_fold = xgb.DMatrix(X.iloc[trn_idx], label=y.iloc[trn_idx])
#     dvalid_fold = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
    
#     watchlist = [(dtrain_fold, 'train'), (dvalid_fold, 'eval')]
#     model = xgb.train(
#         xgb_params,
#         dtrain_fold,
#         num_boost_round=50000,
#         evals=watchlist,
#         early_stopping_rounds=100,
#         verbose_eval=100
#     )
    
#     best_iter = model.best_iteration
#     print(f"Fold {fold_ + 1} best iteration: {best_iter}")
#     oof_preds_xgbb[val_idx] = model.predict(xgb.DMatrix(X.iloc[val_idx]), iteration_range=(0, best_iter))
#     test_preds_xgbb += model.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter)) / folds.n_splits
# cv_rmse_xgbb = np.sqrt(mean_squared_error(y, oof_preds_xgbb))
# print(f"\n[XGBoost] Overall CV RMSE: {cv_rmse_xgbb:.4f}")


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Assume train, test, target, and features are already defined and preprocessed.
target = 'Listening_Time_minutes'
X = train[features]
y = train[target]
X_test = test[features]

# Setting up 5-fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.003220710159405592,
    'max_depth': 10,
    'subsample': 0.7683608904751302,
    'colsample_bytree': 0.7868641460936892,
    'min_child_weight': 7,
    'gamma': 0.22924550099242594,
    'reg_alpha': 1.0404469630505464e-07,
    'reg_lambda': 3.0297275361051017e-07,
    'seed': 42,
    # GPU settings (new style)
    'tree_method': 'hist',
    'device': 'cuda',    # use GPU
    'nthread': 8,
}

oof_preds_xgbb = np.zeros(X.shape[0])
test_preds_xgbb = np.zeros(X_test.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"\n[XGBoost] Starting Fold {fold_ + 1}")
    dtrain_fold = xgb.DMatrix(X.iloc[trn_idx], label=y.iloc[trn_idx])
    dvalid_fold = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
    
    watchlist = [(dtrain_fold, 'train'), (dvalid_fold, 'eval')]
    model = xgb.train(
        xgb_params,
        dtrain_fold,
        num_boost_round=50000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    best_iter = model.best_iteration
    print(f"Fold {fold_ + 1} best iteration: {best_iter}")
    oof_preds_xgbb[val_idx] = model.predict(xgb.DMatrix(X.iloc[val_idx]), iteration_range=(0, best_iter))
    test_preds_xgbb += model.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter)) / folds.n_splits
cv_rmse_xgbb = np.sqrt(mean_squared_error(y, oof_preds_xgbb))
print(f"\n[XGBoost] Overall CV RMSE: {cv_rmse_xgbb:.4f}")


cv_rmse_xgbb = np.sqrt(mean_squared_error(y, oof_preds_xgbb))
print(f"\n[XGBoost] Overall CV RMSE: {cv_rmse_xgbb:.4f}")


submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': test_preds_xgbb  # or use test_preds from your preferred model
})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


# import optuna
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# target = 'Listening_Time_minutes'
# X = train[features]
# y = train[target]
# X_test = test[features]
# def objective(trial):
#     params = {
#         'objective': 'reg:squarederror',  # regression objective
#         'eval_metric': 'rmse',
#         'eta': trial.suggest_float('eta', 0.001, 0.02, log=True),
#         'max_depth': trial.suggest_int('max_depth', 6, 10),
#         'subsample': trial.suggest_float('subsample', 0.7, 0.9),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
#         'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
#         # Optionally add gamma and regularization:
#         'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
#         'seed': 42,
#         'tree_method': 'hist',  # use histogram method (GPU-friendly)
#         'device': 'cuda',       # use GPU for training
#     }

#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     rmse_list = []
    
#     # Assuming X and y are defined outside (training features and target)
#     for train_idx, valid_idx in cv.split(X, y):
#         dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
#         dvalid = xgb.DMatrix(X.iloc[valid_idx], label=y.iloc[valid_idx])
#         watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#         model = xgb.train(params,
#                           dtrain,
#                           num_boost_round=50000,
#                           evals=watchlist,
#                           early_stopping_rounds=100,
#                           verbose_eval=False)
#         preds = model.predict(xgb.DMatrix(X.iloc[valid_idx]), iteration_range=(0, model.best_iteration))
#         rmse = np.sqrt(mean_squared_error(y.iloc[valid_idx], preds))
#         rmse_list.append(rmse)
#     return np.mean(rmse_list)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)
# print("Best params:", study.best_trial.params)


# # After obtaining the best parameters from the study:
# best_params = study.best_trial.params
# print("Best parameters from Optuna:", best_params)

# # Train a final model using the best parameters on the full training set:
# dtrain_full = xgb.DMatrix(X, label=y)
# dtest = xgb.DMatrix(X_test)

# # Optionally, set a fixed number of boosting rounds or use early stopping with a validation split
# final_model = xgb.train(best_params, dtrain_full, num_boost_round=best_iteration_value)  # e.g., use best_iteration_value from tuning

# # Predict on the test set:
# final_predictions = final_model.predict(dtest)

# # Calculate overall training RMSE (if needed) by predicting on training set:
# train_preds = final_model.predict(dtrain_full)
# overall_rmse = np.sqrt(mean_squared_error(y, train_preds))
# print(f"Overall Training RMSE: {overall_rmse:.4f}")

# # Create submission file (update 'id_column' with your unique identifier column name for the submission):
# submission = pd.DataFrame({
#     'id': test['id_column'],  # Replace 'id_column' with the actual column name
#     'Listening_Time_minutes': final_predictions
# })
# submission.to_csv('submission.csv', index=False)
# print("Submission file created: submission.csv")

