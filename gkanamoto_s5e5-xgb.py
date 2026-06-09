import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


# define RMSLE
def rmsle_score(y, preds):
    y = np.maximum(0, y)
    preds = np.maximum(0, preds)
    return np.sqrt(np.mean((np.log1p(preds) - np.log1p(y)) ** 2))


SEED = 42
N_SPLITS = 5


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# check datases
print("Train size:", train_df.shape)
print("Test size:", test_df.shape)


train_df.head()


test_df.head()


print("----Train data----")
print(train_df.isnull().sum())
print("="*20)
print("----Test data----")
print(test_df.isnull().sum())


## all data
train_df.describe()


# Male
male_df = train_df[train_df["Sex"]=="male"]
male_df.describe()


# Female
female_df = train_df[train_df["Sex"]=="female"]
female_df.describe()


# create Duration class column
bins = list(np.arange(1, 40, 5))
labels = [f'{b}-{b+4}' for b in bins[:-1]]

train_df['Duration_class'] = pd.cut(train_df['Duration'], bins=bins, labels=labels, right=False)
test_df['Duration_class'] = pd.cut(test_df['Duration'], bins=bins, labels=labels, right=False)


# create age class column
bins = list(np.arange(1, 90, 5))
labels = [f'{b}-{b+4}' for b in bins[:-1]]

train_df['age_class'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)
test_df['age_class'] = pd.cut(test_df['Age'], bins=bins, labels=labels, right=False)


# target encoding
# groubby --> Sex, Age_class, Duration_class
group_encod = train_df.groupby(['Sex', 'age_class', 'Duration_class'])['Calories'].median().reset_index()
group_encod.rename(columns={'Calories': 'Calories_encoded'}, inplace=True)

train_df = train_df.merge(group_encod, on=['Sex', 'age_class', 'Duration_class'], how='left')
test_df = test_df.merge(group_encod, on=['Sex', 'age_class', 'Duration_class'], how='left')


# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['Duration_class', 'age_class', 'Sex']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train_df[cat_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
train_df = pd.concat([train_df.drop(columns=cat_cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test_df[cat_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
test_df = pd.concat([test_df.drop(columns=cat_cols), encoded_test_df], axis=1)


# drop columns
# train_df = train_df.drop(columns=['Duration_class', 'age_class'])
# test_df = test_df.drop(columns=['Duration_class', 'age_class'])


X = train_df.drop(columns=["id", "Calories"])
y = train_df["Calories"]


# ## Optuna
# import optuna
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.1, random_state=SEED
# )


# def objective(trial):
#     param = {
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
#         "max_depth": trial.suggest_int("max_depth", 7, 10),
#         "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.05),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.05),
#         "max_bin": trial.suggest_int("max_bin", 256, 1024, step=128),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
#         "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
#         "random_state": SEED
#     }

#     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
#     scores = []

#     for train_index, valid_index in kf.split(X):
#         X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
#         y_train, y_val = y.iloc[train_index], y.iloc[valid_index]

#         y_train_log = np.log1p(y_train)
#         y_val_log = np.log1p(y_val)

#         dtrain = xgb.DMatrix(X_train, label=y_train_log)
#         dval = xgb.DMatrix(X_val, label=y_val_log)

#         model = xgb.train(
#             params=param,
#             dtrain=dtrain,
#             num_boost_round=2000,
#             evals=[(dval, "valid")],
#             early_stopping_rounds=100,
#             verbose_eval=0
#         )

#         y_pred_val_log = model.predict(dval, iteration_range=(0, model.best_iteration))
#         y_pred_val = np.expm1(y_pred_val_log)

#         score = rmsle_score(y_val, y_pred_val)
#         scores.append(score)

#     return np.mean(scores)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# print("Best hyperparameters:", study.best_params)
# print("Best RMSE score:", study.best_value)


params =  {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",
    'learning_rate': 0.02, 
    'max_depth': 7, 
    'subsample': 0.75, 
    'colsample_bytree': 0.85, 
    'max_bin': 896, 
    'min_child_weight': 8, 
    'gamma': 0.004203608130580876, 
    'lambda': 0.46161300537831074, 
    'alpha': 0.9179112663665772,
    "random_state": SEED
}


# params =  {
#     "objective": "reg:squarederror",
#     "eval_metric": "rmse",
#     "tree_method": "gpu_hist",
#     'learning_rate': 0.02, 
#     'max_depth': 10, 
#     'subsample': 0.8, 
#     'colsample_bytree': 0.8, 
#     "random_state": SEED
# }


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

scores = []
models = []

for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = y.iloc[train_index], y.iloc[valid_index]

    # ⭐ Transform the target variable using log1p
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    # ⭐ Convert the predictions back using expm1
    y_pred_val_log = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_pred_val = np.expm1(y_pred_val_log)

    # Calculate the score by comparing with y_val before the log1p transformation
    score = rmsle_score(y_val, y_pred_val)
    print(f'Fold: {fold+1} RMSLE score: {np.mean(score):.5f}') 

    scores.append(score)
    models.append(model)


print(f'Cross-validated RMSLE score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}') 


test_id = test_df["id"]
test = test_df.drop(columns=["id"])
submit_score = []

dtest = xgb.DMatrix(test)
for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest, iteration_range=(0, model.best_iteration))
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)
pred = np.expm1(pred)


submission = pd.DataFrame({
    'id': test_id,
    'Calories': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission




