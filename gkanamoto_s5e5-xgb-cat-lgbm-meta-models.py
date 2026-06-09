import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import itertools
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV, LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


# define RMSLE
def rmsle_score(y, preds):
    y = np.maximum(0, y)
    preds = np.maximum(0, preds)
    return np.sqrt(np.mean((np.log1p(preds) - np.log1p(y)) ** 2))


SEED = 42
N_SPLITS = 15
N_REPEATS = 3


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print("----Train data----")
print(train_df.isnull().sum())
print("="*20)
print("----Test data----")
print(test_df.isnull().sum())


# Add new features
def add_new_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df["Heart_Rate"] / (df["Duration"] + 1e-5)
    df['Calories_Burned'] = np.where(
        df['Sex'] == 'male',
        (-55.0969 + (0.6309 * df['Heart_Rate']) + (0.1988 * df['Weight']) + (0.2017 * df['Age'])) / 4.184 * df['Duration'],
        (-20.4022 + (0.4472 * df['Heart_Rate']) - (0.1263 * df['Weight']) + (0.074 * df['Age'])) / 4.184 * df['Duration']
    )
    df["Body_Temp_*_Duration"] = df["Body_Temp"] * df["Duration"]
    df["Heart_Rate_*_Duration"] = df["Heart_Rate"] * df["Duration"]
    df["Height_*_Duration"] = df["Height"] * df["Duration"]

    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max']:
            agg_val = train_df.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            df = df.merge(agg_val, on='Sex', how='left')
    return df

train_df = add_new_features(train_df)
test_df = add_new_features(test_df)


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


train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0}).astype("float32")
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0}).astype('float32')


# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['Duration_class', 'age_class']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train_df[cat_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
train_df = pd.concat([train_df.drop(columns=cat_cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test_df[cat_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
test_df = pd.concat([test_df.drop(columns=cat_cols), encoded_test_df], axis=1)


X = train_df.drop(columns=["id", "Calories"])
y = train_df["Calories"]

X_test = test_df.drop(columns=["id"])


# Hyperparameters
xgb_params =  {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",
    'learning_rate': 0.02, 
    'max_depth': 10, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    "random_state": SEED
}

cat_params = {
    "loss_function": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, 
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "task_type": "GPU"
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "n_estimators": 3000, 
    "num_leaves": 128,  
    "max_depth": 20, 
    "min_child_samples": 20, 
    "min_split_gain": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 3.0, 
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbosity": -1,
    "feature_fraction": 0.7,
    "force_col_wise": True
}


# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))
oof_preds_lgb = np.zeros(len(X))

test_preds_xgb = []
test_preds_cat = []
test_preds_lgb = []

for fold, (train_idx, val_idx) in enumerate(rkf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Transform the target variable using log1p
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    model_xgb = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    xgb_score = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration))
    oof_preds_xgb[val_idx] = xgb_score
    # oof_preds_xgb[val_idx] = np.expm1(xgb_score)

    dtest = xgb.DMatrix(X_test)
    xgb_pred = model_xgb.predict(dtest, iteration_range=(0, model_xgb.best_iteration))
    test_preds_xgb.append(xgb_pred)
    # test_preds_xgb.append(np.expm1(xgb_pred))

    # CatBoost
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_train, y_train_log, verbose=False)
    cat_score = model_cat.predict(X_val)
    oof_preds_cat[val_idx] = cat_score
    # oof_preds_cat[val_idx] = np.expm1(cat_score)

    cat_pred = model_cat.predict(X_test)
    test_preds_cat.append(cat_pred)
    # test_preds_cat.append(np.expm1(cat_pred))

    # LGBM
    model_lgb = LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train_log)
    lgb_score = model_lgb.predict(X_val)
    oof_preds_lgb[val_idx] = lgb_score
    # oof_preds_lgb[val_idx] = np.expm1(lgb_score)

    lgb_pred = model_cat.predict(X_test)
    test_preds_lgb.append(lgb_pred)
    # test_preds_lgb.append(np.expm1(lgb_pred))

    # Calculate the score by comparing with y_val before the log1p transformation
    avg_score = (np.expm1(xgb_score) + np.expm1(cat_score) + np.expm1(lgb_score)) / 3
    print(f'===============Fold: {fold+1} Average RMSLE score: {np.mean(rmsle_score(y_val, avg_score)):.5f}') 
    print(f'-------------------->  XGBoost RMSLE score: {np.mean(rmsle_score(y_val, np.expm1(xgb_score))):.5f}') 
    print(f'--------------------> CatBoost RMSLE score: {np.mean(rmsle_score(y_val, np.expm1(cat_score))):.5f}') 
    print(f'--------------------> LightGBM RMSLE score: {np.mean(rmsle_score(y_val, np.expm1(lgb_score))):.5f}') 
    print('')

# Creating train and val data for stacking
stacked_train = np.vstack([oof_preds_xgb, oof_preds_cat, oof_preds_lgb]).T
stacked_test = np.vstack([np.mean(test_preds_xgb, axis=0),
                          np.mean(test_preds_cat, axis=0),
                          np.mean(test_preds_lgb, axis=0)
                         ]).T


# Prediction - Using multiple meta models

y_log = np.log1p(y)

# Ridge
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0], cv=5)
ridge.fit(stacked_train, y_log)
ridge_preds = ridge.predict(stacked_test)

#LinearRegression
lr = LinearRegression()
lr.fit(stacked_train, y_log)
lr_preds = lr.predict(stacked_test)

# BayesianRidge
bayesian_ridge = make_pipeline(
    StandardScaler(),
    BayesianRidge(
        n_iter=2000,
        tol=1e-3,
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6
    )
)
bayesian_ridge.fit(stacked_train, y_log)
bayes_preds = bayesian_ridge.predict(stacked_test)

# Averaging
final_preds = (ridge_preds + lr_preds + bayes_preds) / 3
final_preds = np.expm1(final_preds)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_preds
})

# Save
submission.to_csv('submission.csv', index=False)


submission




