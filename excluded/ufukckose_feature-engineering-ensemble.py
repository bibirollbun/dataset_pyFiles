import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# Initial Exploration
df.head()


df.info()


def ftr_eng(X) :
    """
    This function adds secondary features for the model
    
    Sources:
    (1) https://www.kaggle.com/code/imaadmahmood/road-accident-risk-prediction
    (2) https://www.kaggle.com/code/ravi20076/playgrounds5e10-public-baseline-v1
    """
    
    # Copy input to avoid modifying original DataFrame
    df = X.copy()
    
    ordinal_features     = ["lighting"]
    boolean_features     = ["road_signs_present", "public_road", "holiday", "school_season"]
    categorical_features = ["road_type", "weather", "time_of_day"]


    # Interaction Features
    df['lanes_speed']        = df['num_lanes'] * df['speed_limit']
    df["speed_accident"]     = df["speed_limit"] * df["num_reported_accidents"]
    df["curvature_speed"]    = df["curvature"] * df["speed_limit"]
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-5)
    df["lanes_accidents"]    = df["num_lanes"] * df["num_reported_accidents"]
    df["curvature_per_lane"] = df["curvature"] / (df["num_lanes"] + 1e-5)
    df["risky_conditions"]   = ((df["curvature"] > 0.5) & (df["speed_limit"] > 50) & (df["num_reported_accidents"] > 0)).astype(int)
    df["weather_time"]       = df["weather"] + "_" + df["time_of_day"]
    df["lighting_weather"]   = df["lighting"] + "_" + df["weather"]

    # Binned Features
    df["speed_bin"]          = pd.cut(
        df["speed_limit"], bins=[0, 35, 50, 70], labels=["low", "medium", "high"]
    )
    
    df["curvature_bin"] = pd.qcut(
        df["curvature"], q=4, labels=["very_low", "low", "high", "very_high"]
    )
    
    # Ratio & Log Transform Features
    df["log_accidents"]    = np.log1p(df["num_reported_accidents"])
    df["accident_density"] = df["num_reported_accidents"] / (df["speed_limit"] * df["num_lanes"] + 1e-5)
    
    # Ordinal Encoding
    lighting_order = {"daylight": 2, "dim": 1, "night": 0}
    df["lighting"] = df["lighting"].map(lighting_order)
        
    return df


from sklearn.base import BaseEstimator, TransformerMixin

# Wrapper class to use custom feature engineering function inside sklearn Pipelines
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func  # pass any custom function
    
    def fit(self, X, y=None):
        # Nothing to learn here (stateless transformer)
        return self
    
    def transform(self, X):
        # Apply the custom function to transform features
        return self.func(X)

# Wrap our defined feature engineering function
feature_engineering = FeatureEngineer(ftr_eng)


# Feature Groups
numeric_features = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents",
                    "lanes_speed", "speed_accident", "curvature_speed", 
                    "accidents_per_lane", "lanes_accidents", "curvature_per_lane",
                    "log_accidents", "accident_density"]

categorical_features = ["road_type", "lighting", "weather", "time_of_day",
                        "weather_time", "lighting_weather", "speed_bin", "curvature_bin"]

boolean_features = ["road_signs_present", "public_road", "holiday", "school_season", "risky_conditions"]


from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.utils.fixes import loguniform


# Preprocessing Pipeline (Rebuilt with Engineered Features)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("bool", "passthrough", boolean_features)
    ]
)


# Features & Target Split
X = df.drop(columns=["id", "accident_risk"])

y = df["accident_risk"]

X_test = test.drop(columns=["id"])


# Custom RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


rng = 42
xgb_param_dist = {
    "model__n_estimators": [100, 200, 400, 800],
    "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "model__max_depth": [3, 4, 6, 8],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "model__gamma": [0, 0.1, 0.5]
}

lgbm_param_dist = {
    "model__n_estimators": [100, 200, 400, 800],
    "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "model__num_leaves": [31, 63, 127, 200],
    "model__max_depth": [-1, 3, 5, 7],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.4, 0.6, 0.8, 1.0]
}

cat_param_dist = {
    "model__iterations": [200, 400, 800],
    "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "model__depth": [4, 6, 8, 10],
    "model__l2_leaf_reg": [1, 3, 5, 9]
}


# Model Tuning Helper Function
def tune_model(base_model, param_dist, X, y, n_iter=25, cv=3, n_jobs=-1, verbose=2, random_state=rng):
    """
    base_model: estimator (unfitted)
    param_dist: dict with keys matching pipeline param names, e.g. "model__learning_rate"
    """
    pipe = Pipeline([
        ("feature_engineering", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])
    
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=False
    )
    search.fit(X, y)
    print(f"Best score: {search.best_score_:.5f} | Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


# === tune each model (you can increase n_iter/cv if you have compute) ===
#n_iter = 10
#cv = 3

# XGB
#xgb_base = XGBRegressor(objective="reg:squarederror", random_state=rng, tree_method="hist", device="cuda")  # hist is faster if supported
#best_xgb_pipe, xgb_best_params, xgb_best_score = tune_model(xgb_base, xgb_param_dist, X, y, n_iter=n_iter, cv=cv)

# LGBM
#lgbm_base = LGBMRegressor(random_state=rng, n_jobs=-1, verbose=-1)
#best_lgbm_pipe, lgbm_best_params, lgbm_best_score = tune_model(lgbm_base, lgbm_param_dist, X, y, n_iter=n_iter, cv=cv)

# CatBoost
# set verbose=0 to silence logs; catboost learns faster with CPU threads
#cat_base = CatBoostRegressor(loss_function="RMSE", random_seed=rng, verbose=0)
#best_cat_pipe, cat_best_params, cat_best_score = tune_model(cat_base, cat_param_dist, X, y, n_iter=n_iter, cv=cv)



# Best Base Models (XGB, LGBM, CatBoost)
best_xgb = XGBRegressor(
    max_depth=8,
    learning_rate=0.01,
    n_estimators=2000,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="rmse"
)


best_lgbm = LGBMRegressor(
    random_state=rng,
    n_jobs=-1,
    subsample=0.6,
    num_leaves=200,
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    colsample_bytree=1.0,
    verbose=-1
)

best_cat = CatBoostRegressor(
    random_seed=rng,
    loss_function="RMSE",
    verbose=0,
    learning_rate=0.1,
    l2_leaf_reg=9,
    iterations=800,
    depth=8
)




# Build Stacking Regressor with Ridge
estimators = [
    ("xgb", best_xgb),
    ("cat", best_cat)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0, random_state=rng),
    passthrough=True,
    n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True, random_state=rng)
)



# Full Pipeline (Feature Engineering + Preprocessing + Stack)
stacking_pipeline = Pipeline([
    ("feature_engineering", feature_engineering),
    ("preprocessor", preprocessor),
    ("stack", stack)
])


# Fit on full training set and predict on test
stacking_pipeline.fit(X, y)
test_preds = stacking_pipeline.predict(X_test)

# Clip predictions to [0,1] if target is risk between 0 and 1
test_preds = np.clip(test_preds, 0.0, 1.0)

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv — head:")
print(submission.head())

