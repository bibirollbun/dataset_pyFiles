# Install all required packages
!apt-get update && apt-get install -y r-cran-rcppeigen # RcppEigen
import rpy2.robjects as ro
ro.r("install.packages(c('midr', 'khroma', 'viridisLite', 'RColorBrewer'), quiet = TRUE)")
!pip install /kaggle/input/pyramid-learn/pyramid_learn-0.1.0-py3-none-any.whl -q

print('Installed all required packages.')


import os
import numpy as np
import pandas as pd

import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb
import midlearn as mid
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

import plotnine as p9
p9.theme_set(p9.theme_bw())

# Global Configurations
PATH  = "/kaggle/input/playground-series-s5e10/"
TARGET = 'accident_risk'
RUN_OPTUNA = False
CALC_INTERACTION = True # Calculation of two-dimensional MID requires a huge memory capacity.
COLD_RUN = True


train = pd.read_csv(os.path.join(PATH, "train.csv"))
test  = pd.read_csv(os.path.join(PATH, "test.csv"))
sub   = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))

train_X = train.drop(["id", TARGET], axis=1)
train_y = train[TARGET]
test_X  = test.drop(["id"], axis=1)

if COLD_RUN:
    train_X = train_X[:30000]
    train_y = train_y[:30000]

object_columns = train_X.select_dtypes(include=['object']).columns
train_X[object_columns] = train_X[object_columns].astype('category')
test_X[object_columns]  = test_X[object_columns].astype('category')

train_X.shape


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(train_X.shape[0])

    for train_idx, valid_idx in kf.split(train_X):
        X_train, X_valid = train_X.iloc[train_idx], train_X.iloc[valid_idx]
        y_train, y_valid = train_y.iloc[train_idx], train_y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        oof[valid_idx] = model.predict(X_valid)

    return mean_squared_error(oof, train_y, squared=False)

if RUN_OPTUNA:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)


params = {
    'n_estimators': 296,
    'learning_rate': 0.022116913287724724,
    'num_leaves': 245,
    'max_depth': 15,
    'min_child_samples': 42,
    'subsample': 0.8137115514953386,
    'colsample_bytree': 0.8751219376345558,
    'reg_alpha': 0.0005636473765537796,
    'reg_lambda': 0.011482076889580634,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    'importance_type': 'gain',
}

if RUN_OPTUNA:
    params.update(study.best_trial.params)
print("Parameters:\n", params, "\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(train_X.shape[0])
models = list()

for train_idx, valid_idx in kf.split(train_X):
    X_train, X_valid = train_X.iloc[train_idx], train_X.iloc[valid_idx]
    y_train, y_valid = train_y.iloc[train_idx], train_y.iloc[valid_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    oof[valid_idx] = model.predict(X_valid)
    models.append(model)

print("RMSE (Out-of-Fold):\n", round(np.sqrt(np.mean((oof - train_y) ** 2)), 6))


# Create an ensemble model of LGBMRegressors
ensemble = VotingRegressor(estimators=models)
ensemble.estimators_ = ensemble.estimators # Skip the fitting process

# Predict for testing dataset
pred_y = ensemble.predict(test_X)
sub[TARGET] = pred_y
display(sub.head())

sub.to_csv("submission_lgb.csv", index=False)


explainer = mid.MIDExplainer(
    estimator=ensemble,
    interaction=CALC_INTERACTION,
    params_main=100
)
explainer.fit(train_X)


print("R-squared score:", explainer.fidelity_score(test_X, pred_y))

p9.ggplot() \
+ p9.geom_abline(slope=1) \
+ p9.geom_point(p9.aes(x=pred_y, y=explainer.predict(test_X))) \
+ p9.labs(x="Predictions (LightGBM Regressor)", y="Predictions (Surrogate MID Explainer)")


imp = explainer.importance()
display(imp.importance.head(10))


imp.plot(theme='muted')


imp.plot(style='heatmap', color = '#505050')


for i, t in enumerate(imp.terms(interactions=False)):
    print(f"{i + 1}: {t}")
    display(explainer.plot(term=str(t)) + p9.lims(y=[-.2, .2]))


for i, t in enumerate(imp.terms(main_effects=False)[:4]):
    print(f"{i + 1}: {t}")
    display(explainer.plot(term=str(t), main_effects=False))
    display(explainer.plot(term=str(t), main_effects=True, theme='sunset'))


for i in range(3):
    print(f"train_X[{i}]")
    display(explainer.breakdown(row=i).plot(theme='muted_r'))


for i in range(3):
    print(f"test_X[{i}]")
    display(explainer.breakdown(data=test_X.iloc[[i]]).plot(theme='midr@qualitative'))


# Plot individual conditional expectations
explainer.conditional('curvature') \
.plot(style='centered', var_color='num_reported_accidents')


train_X.head()


def preprocess_lm(df: pd.DataFrame) -> pd.DataFrame:
    cat_columns = ['lighting', 'weather']
    _df = df.copy()
    _df[cat_columns] = _df[cat_columns].astype('category')
    _df = pd.get_dummies(_df, columns=cat_columns, drop_first=False)
    _df['speed_limit_lift_60'] = _df['speed_limit'] >= 60
    _df['num_reported_accidents_lift_3'] = _df['num_reported_accidents'] >= 3
    _df['num_reported_accidents_lift_6'] = _df['num_reported_accidents'] >= 6
    _df['low_curvature_many_reported'] = (_df['curvature'] <= 0.5) & (_df['num_reported_accidents'] >= 3)
    return _df


use_columns = [
    # 'road_type',
    # 'num_lanes',
    'curvature',
    # 'speed_limit',
    # 'road_signs_present',
    # 'public_road',
    # 'time_of_day',
    # 'holiday',
    # 'school_season',
    # 'num_reported_accidents',
    # 'lighting_daylight',
    # 'lighting_dim',
    'lighting_night',
    'weather_clear',
    # 'weather_foggy',
    # 'weather_rainy',
    'speed_limit_lift_60',
    'num_reported_accidents_lift_3',
    'num_reported_accidents_lift_6',
    'low_curvature_many_reported'
]

train_X_lm = preprocess_lm(train_X)[use_columns]
test_X_lm  = preprocess_lm(test_X)[use_columns]

display(train_X_lm.head())


linear = LinearRegression()
linear.fit(train_X_lm, train_y)

# Summary: intercept and coefficients
linear_df = pd.DataFrame(
    linear.coef_,
    index=linear.feature_names_in_,
    columns=['coefficient']
)
linear_df.loc['intercept'] = linear.intercept_
linear_df


pred_y_lm = linear.predict(test_X_lm)
print(f"R squared score: {r2_score(pred_y, pred_y_lm)}")

p9.ggplot() \
+ p9.geom_abline(slope=1) \
+ p9.geom_point(p9.aes(x=pred_y, y=pred_y_lm)) \
+ p9.labs(x="Predictions (LightGBM Regressor)", y="Predictions (Surrogate Linear Regressor)")


# Predict for testing dataset
sub[TARGET] = pred_y_lm
display(sub.head())

sub.to_csv("submission_lm.csv", index=False)

