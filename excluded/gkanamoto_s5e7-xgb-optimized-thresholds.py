# Importing liblies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize, minimize_scalar

import xgboost as xgb

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


SEED = 42
FOLDS = 10
N_REPEATS = 2


# check sample size
print('Training: ', train_df.shape)
print('    Test: ', train_df.shape)


# check missing values
print("----Train data----")
print(train_df.isnull().sum())
print("="*20)
print("----Test data----")
print(test_df.isnull().sum())


train_df.head()


test_df.head()


# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['Stage_fear', 'Drained_after_socializing']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train_df[cat_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
train_df = pd.concat([train_df.drop(columns=cat_cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test_df[cat_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
test_df = pd.concat([test_df.drop(columns=cat_cols), encoded_test_df], axis=1)


# # Encoding categorical features
# train_df['Stage_fear'] = train_df['Stage_fear'].map({'Yes':1, 'No': 0})
# train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes':1, 'No': 0})

# test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes':1, 'No': 0})
# test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes':1, 'No': 0})


# Encoding the target value
target_map = {
    'Extrovert' : 0,
    'Introvert' : 1
}

train_df['Personality'] = train_df['Personality'].map(target_map)


# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')

# missing_columns_train = train_df.columns[train_df.isnull().any()].tolist()
# missing_columns_test = test_df.columns[test_df.isnull().any()].tolist()

# train_missing = train_df[missing_columns_train]
# test_missing = test_df[missing_columns_test]

# train_missing_filled = pd.DataFrame(
#     imputer.fit_transform(train_df[missing_columns_train]),
#     columns=missing_columns_train,
#     index=train_df.index
# )
# test_missing_filled = pd.DataFrame(
#     imputer.transform(test_df[missing_columns_test]),
#     columns=missing_columns_test,
#     index=test_df.index
# )

# train_df.loc[:, missing_columns_train] = train_missing_filled
# test_df.loc[:, missing_columns_test] = test_missing_filled


X = train_df.drop(columns=["id", "Personality"])
y = train_df["Personality"]


# Function to optimize Accuracy score by adjusting thresholds
def evaluate_predictions(threshold, y_true, y_pred):
    y_pred = (y_pred >= threshold).astype(int)
    return -accuracy_score(y_true, y_pred)  # maximize Accuracy by minimizing negative

initial_thresholds = 0.5


## Optuna
import optuna
from sklearn.model_selection import train_test_split

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

best_params_per_fold = []

for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    print(f"ğŸ”� Fold {fold + 1}  Searching for Best params")
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = y.iloc[train_index], y.iloc[valid_index]

    def objective(trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.05),
            "max_bin": trial.suggest_int("max_bin", 256, 1024, step=64),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "random_state": SEED
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=0
        )

        y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))

        opt_result = minimize_scalar(
            evaluate_predictions, bounds=(0.0, 1.0), method='bounded',
            args=(y_val, y_pred_val),
            options={'xatol': 1e-4, 'maxiter': 1000}
        )
        best_threshold = opt_result.x

        y_pred_bin = (y_pred_val >= best_threshold).astype(int)
        score = accuracy_score(y_val, y_pred_bin)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print(f"âœ… Fold {fold + 1} -Best params: {study.best_params}")
    best_params_per_fold.append(study.best_params)

print("Best params for each models:")
for i, params in enumerate(best_params_per_fold):
    print(f"Fold{i+1}: {params}")


# Hyperparameters
# xgb_params =  {
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",
#     'learning_rate': 0.05, 
#     'max_depth': 32, 
#     'subsample': 0.8, 
#     'colsample_bytree': 0.8, 
#     "random_state": SEED
# }

# xgb_params =  {
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",
#     "learning_rate": ,
#     "max_depth": ,
#     "subsample": ,
#     "colsample_bytree": ,
#     "max_bin": ,
#     "min_child_weight": ,
#     "gamma": ,
#     "lambda": ,
#     "alpha": ,
#     "random_state": SEED
# }


# rkf = RepeatedKFold(n_splits=FOLDS, n_repeats=N_REPEATS, random_state=SEED)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

scores = []
models = []
best_thresholds = []

for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = y.iloc[train_index], y.iloc[valid_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    best_params = best_params_per_fold[fold]
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))

    #â­� Optimize thresholds for Accuracy score â­�
    opt_result = minimize_scalar(
        evaluate_predictions, bounds=(0.0, 1.0), method='bounded',
        args=(y_val, y_pred_val),
        options={
            'xatol': 1e-4,
            'maxiter': 1000
        }
    )
    best_threshold = opt_result.x
    best_thresholds.append(best_threshold)
    
    # Calculate the score
    y_pred_bin = (y_pred_val >= best_threshold).astype(int)
    score = accuracy_score(y_val, y_pred_bin)
    print(f'Fold: {fold+1} Accuracy score: {np.mean(score):.5f}') 

    scores.append(score)
    models.append(model)


print(f'\nAverage Accuracy Score : {np.mean(scores):.5f}, +-: {np.std(scores):.5f}')


X_test = test_df.drop(columns=["id"])
submit_score = []

dtest = xgb.DMatrix(X_test)
for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest, iteration_range=(0, model.best_iteration)) # XGBoost
    best_threshold = best_thresholds[fold_]
    pred_ = (pred_ >= best_threshold).astype(int)
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred
})

target_map = {
    0 : 'Extrovert',
    1 : 'Introvert'
}

submission['Personality'] = submission['Personality'].map(target_map)

# Save
submission.to_csv('submission.csv', index=False)


submission

