import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor #added 
from sklearn.linear_model import ElasticNet,SGDRegressor #added 


import shap #added


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
N_REPEATS = 3
N_FOLDS = 2


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print("----Train data----")
print(train_df.isnull().sum())
print("="*20)
print("----Test data----")
print(test_df.isnull().sum())


# # Add new features
# def add_statistical_features(df, num_features):
#     df_new = df.copy()
#     df_new["row_mean"] = df[num_features].mean(axis=1)
#     df_new["row_std"] = df[num_features].std(axis=1)
#     df_new["row_max"] = df[num_features].max(axis=1)
#     df_new["row_min"] = df[num_features].min(axis=1)
#     df_new["row_median"] = df[num_features].median(axis=1)
#     return df_new

# num_features = train_df.drop(columns=["id", "Sex", "Calories"]).columns
# train_df = add_statistical_features(train_df, num_features)
# test_df = add_statistical_features(test_df, num_features)


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


X = train_df.drop(columns=["id", "Calories"])
y = train_df["Calories"]

X_test = test_df.drop(columns=["id"])


#downcasting to save RAM
for df_ in (X, X_test):
    for col in df_.select_dtypes(include="float64"):
        df_[col] = pd.to_numeric(df_[col], downcast="float")
    for col in df_.select_dtypes(include="int64"):
        df_[col] = pd.to_numeric(df_[col], downcast="integer")


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
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, 
    "random_seed": SEED,
    "verbose": 0,
    "task_type": "GPU"
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "n_estimators": 2000, 
    "num_leaves": 64,  
    "max_depth": 8, 
    "min_child_samples": 20, 
    "min_split_gain": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0, 
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbosity": -1,
    "force_col_wise": True
}


#model mixer
model_mixer = {
    "xgb":    lambda: xgb.XGBRegressor(**xgb_params),
    "cat":    lambda: cb.CatBoostRegressor(**cat_params),
    "lgb":    lambda: lgb.LGBMRegressor(**lgb_params),
    "elastic":lambda: ElasticNet(random_state=SEED),
    #  "sgd":     lambda: SGDRegressor(max_iter=1000, tol=1e-3, random_state=SEED),
    # "rf_small":lambda: RandomForestRegressor(
    #                     n_estimators=50,
    #                     max_depth=8,
    #                     random_state=SEED
    #                 )
}


model_list = list(model_mixer.keys())
n_models = len(model_list)


#OOF & pred files
oof_pred =np.zeros((len(X),n_models),dtype="float32")
test_pred = np.memmap(
    "test_preds.dat",
    dtype="float32",
    mode="w+",
    shape=(len(X_test), n_models)
)


# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
from sklearn.model_selection import KFold

#  shuffled folds
all_splits = []
for r in range(N_REPEATS):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED + r
    )
    all_splits.extend(kf.split(X))

# CV loop over those folds
for fold, (train_idx, val_idx) in enumerate(all_splits,1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx],       y[val_idx]

    # log1p transform for RMSLE
    y_train_log = np.log1p(y_train)
    y_val_log   = np.log1p(y_val)

    # only boosters get early stopping
    fit_params = {
        "xgb": {"eval_set": [(X_val, y_val_log)],
                "early_stopping_rounds": 100,
                "verbose": False},
        "lgb": {"eval_set": [(X_val, y_val_log)],
                "callbacks": [lgb.early_stopping(stopping_rounds=100)]},
        "cat": {"eval_set": [(X_val, y_val_log)],
                "early_stopping_rounds": 100},
    }

    print(f"--- Fold {fold} ---")
    for i, name in enumerate(model_list):
        model = model_mixer[name]()
        params = fit_params.get(name, {})           # {} for models without early-stopping
        model.fit(X_train, y_train_log, **params)
        MAX_LOG = np.log(np.finfo(np.float32).max) - 1 
        raw_pred = model.predict(X_val)
        raw_clipped = np.minimum(raw_pred, MAX_LOG)
        val_pred    = np.expm1(raw_clipped).astype("float32")  # now guaranteed finite


        oof_pred[val_idx, i] = val_pred

        #for test_pred
        raw_test = model.predict(X_test)
        raw_test_clipped = np.minimum(raw_test, MAX_LOG)
        test_pred[:, i] += np.expm1(raw_test_clipped) / (N_SPLITS * N_REPEATS)


        

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f" {name:<7} RMSE: {rmse:.5f}")
        #free memory from this model
        del model
        gc.collect()
        
    print()

#fush to disk
test_pred.flush()
   


 # Creating train and val data for stacking
stacked_train = oof_pred
stacked_test = test_pred[:]

print("stacked_train shape:", stacked_train.shape)
print("stacked_test  shape:", stacked_test.shape)


#meta-learner
meta = RidgeCV(
    alphas=[0.1, 1.0, 10.0, 50.0, 100.0],
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
)


meta.fit(stacked_train, y)
final_preds = meta.predict(stacked_test)



# Prediction
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0], cv=5)
ridge.fit(stacked_train, y)
final_preds = ridge.predict(stacked_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_preds
})

# Save
submission.to_csv('submission.csv', index=False)


submission




