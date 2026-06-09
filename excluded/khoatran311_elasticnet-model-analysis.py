import numpy as np
import pandas as pd
import polars as pl
import seaborn as sea
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline as pipe

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import optuna
import joblib

import warnings
warnings.filterwarnings("ignore")


def format_pl():
  """FLOAT DISPLAY FORMATTING"""
  pl.Config.set_fmt_float("mixed")
  """STRING FORMATTING"""
  pl.Config.set_fmt_str_lengths(50)
  """TABLE FORMATTING"""
  pl.Config.set_tbl_rows(8)
  pl.Config.set_tbl_cols(30)
  pl.Config.set_tbl_width_chars(200)
  pl.Config.set_tbl_cell_alignment("RIGHT")
  pl.Config.set_tbl_hide_dtype_separator(True)
  pl.Config.set_tbl_hide_column_data_types(True)

format_pl()


## Helper functions
eps = 1e-7
logit   = lambda p: np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
sigmoid = lambda x: 1/(1 + np.exp(-x)) 
RMSE    = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred)**2))


train = pl.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.head()


test.head()


train.select(
    pl.all().is_null().sum()
)


train.select(
    pl.all().n_unique().sum()
)


train.is_duplicated().sum()


train.group_by(
    pl.col("road_type")
).agg(
    pl.col("accident_risk").mean().alias("mean_accident_risk"),
    pl.col("accident_risk").median().alias("median_accident_risk"),
    pl.col("road_type").count().alias("count")
)


train.group_by(
    pl.col("num_lanes")
).agg(
    pl.col("accident_risk").mean().alias("mean_accident_risk"),
    pl.col("accident_risk").median().alias("median_accident_risk"),
    pl.col("road_type").count().alias("count")
).sort(by="num_lanes")


train.group_by(
    pl.col("num_reported_accidents")
).agg(
    pl.col("accident_risk").mean().alias("mean_accident_risk"),
    pl.col("accident_risk").median().alias("median_accident_risk"),
    pl.col("road_type").count().alias("count")
).sort(by="num_reported_accidents")


time_weather_accident = train.group_by(
    pl.col("time_of_day", "weather")
).agg(
    pl.col("accident_risk").mean().alias("mean_accident_risk"),
    pl.col("accident_risk").median().alias("median_accident_risk"),
    pl.col("time_of_day").count().alias("count"),    
).sort(
    by=["time_of_day", "weather"]
).to_pandas()

time_weather_accident


sea.barplot(data=time_weather_accident, x="time_of_day", y="mean_accident_risk", hue="weather")
plt.xlabel("Time of Day")
plt.ylabel("Mean Accident Risk")


sea.scatterplot(x=train["curvature"], y=train["accident_risk"])


train_pd = train.to_pandas().drop(columns=["id"])
bool_features = ["road_signs_present", "public_road", "holiday", "school_season"]


X = train_pd.copy()
X[bool_features] = X[bool_features].astype(np.uint8)

y = X.pop("accident_risk")
y_logit = logit(y)


def cross_validate(model, X, y_logit, n_splits=10, store_oof=True, save_n_models=False, model_prefix=None):
    kfold = KFold(n_splits=n_splits, random_state=3126, shuffle=True)
    rmse  = np.zeros(n_splits)
    if store_oof:
        y_oof = np.zeros_like(y_logit)
    history = {}
    
    for k,(train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X.iloc[train_idx], y_logit[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y_logit[val_idx]
        
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        if save_n_models:
            model_filename = f"{model_prefix}_{k+1}.joblib"
            joblib.dump(cloned_model, model_filename)
            
        ## Get logit predictions
        y_pred = cloned_model.predict(X_val)
        ## If store_oof is enabled, stores the oof prediction
        if store_oof:
            y_oof[val_idx] = y_pred
        ## Stores RMSE
        rmse[k] = RMSE(sigmoid(y_val), sigmoid(y_pred))
        ## Saves model if wanted
       
            

    history["rmse"] = rmse
    if store_oof:
        history["y_oof"] = y_oof
    return history


cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']
num_features = ["curvature", "speed_limit"]

preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(drop="first"), cat_features),
     ("numerical", StandardScaler(), num_features)],
    remainder="passthrough",
    n_jobs=-1
)


# def objective(trial, X, y_logit):
#     # Suggest hyperparameters
#     alpha    = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     l1_ratio = trial.suggest_float("l1_ratio", 0.005, 1.0)

#     model = pipe(preprocessor,
#                  ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=3126, max_iter=10000))
#     history = cross_validate(model, X, y_logit, store_oof=False)
#     return history["rmse"].mean()

# study = optuna.create_study(direction="minimize")  
# study.optimize(lambda trial: objective(trial, X, y_logit), n_trials=100)


# ElasticNet Best params: {'alpha': 0.018360331209412065, 'l1_ratio': 0.4152962096107638}
# ElasticNet Best RMSE: 0.07530080149513488
enet_params = {
    'alpha': 0.018360331209412065, 
    'l1_ratio': 0.4152962096107638
}


# def objective(trial, X, y_logit):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)

#     model = pipe(preprocessor,
#                  Lasso(alpha=alpha, random_state=3126))
#     history = cross_validate(model, X, y_logit, store_oof=False)
#     return history["rmse"].mean()

# study = optuna.create_study(direction="minimize")  
# study.optimize(lambda trial: objective(trial, X, y_logit), n_trials=100)


# Lasso Best params:  {'alpha': 0.012852265770746055}
# Lasso Best RMSE:  0.07534884691731689
lasso_params = {
    'alpha': 0.012852265770746055
}


# def objective(trial, X, y_logit):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)

#     model = pipe(preprocessor,
#                  Ridge(alpha=alpha, random_state=3126))
#     history = cross_validate(model, X, y_logit, store_oof=False)
#     return history["rmse"].mean()

# study = optuna.create_study(direction="minimize")  
# study.optimize(lambda trial: objective(trial, X, y_logit), n_trials=100)


# Ridge Best params:  {'alpha': 9.995174955024535}
# Ridge Best RMSE:  0.07640762729883627
ridge_params = {
    'alpha': 9.995174955024535
}


enet = pipe(preprocessor,
            ElasticNet(**enet_params, random_state=3126, max_iter=10000))
history = cross_validate(enet, X, y_logit, store_oof=True, save_n_models=True, model_prefix="enet")


sea.scatterplot(x=np.arange(1,11), y=history["rmse"])
sea.lineplot(x=np.arange(1,11), y=history["rmse"])
plt.axhline(y=history["rmse"].mean(), linestyle="dashed", color="black")
plt.title("RMSE on Out-of-Fold Sets of ElasticNet")
plt.xlabel("Out-of-Fold Set")
plt.ylabel("RMSE")
plt.show()


plt.figure(figsize=(16,4))
plt.subplot(121)
sea.scatterplot(x=sigmoid(history["y_oof"]), y=y)
plt.xlabel("OOF Predicted Accident Risk")
plt.ylabel("True Accident Risk")
plt.title("ElasticNet's OOF Prediction Analysis")

plt.subplot(122)
sea.histplot(sigmoid(history["y_oof"])-y)
plt.axvline(x=(sigmoid(history["y_oof"])-y).mean(), color="black", linestyle="dashed")
plt.xlabel("OOF Prediction Residual")
plt.ylabel("Count")
plt.title("OOF Prediction Residual Distribution of ElasticNet")

plt.show()


sea.scatterplot(x=sigmoid(history["y_oof"]), y=sigmoid(history["y_oof"])-y, alpha=.25, color="black")
plt.xlabel("OOF Predicted Accident Risk")
plt.ylabel("Residual")
plt.title("OOF Prediction's Residuals of ElasticNet")
plt.show()


test.drop(columns=["id"], inplace=True)
test[bool_features] = test[bool_features].astype(np.uint8)

y_pred = np.zeros(len(test))
for k in range(1, 11):
    enet = joblib.load(f"/kaggle/working/enet_{k}.joblib")
    y_pred += sigmoid(enet.predict(test))
y_pred /= 10

sample_submission["accident_risk"] = y_pred
sample_submission


sample_submission.to_csv("submission.csv", index=False)

