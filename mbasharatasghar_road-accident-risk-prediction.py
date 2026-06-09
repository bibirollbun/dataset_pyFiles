import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.head(5)


test.head(5)


train.shape


test.shape


train.info()


train.describe()


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


test.duplicated().sum()


num_cols = train.select_dtypes(include="number").drop(columns="id", axis=1).columns
num_cols


cat_cols = train.select_dtypes(include="object").columns
cat_cols


bool_cols = train.select_dtypes(include="bool").columns
bool_cols


for col in num_cols:
    sns.histplot(data=train, x=col, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    sns.boxplot(data=train, x=col)
    plt.show()


sns.histplot(data=train, x="curvature", kde=True, hue="road_type", element="step", stat="density", common_norm=False)
plt.title("Distribution of Curvature by Road Type")
plt.xlabel("Curvature")
plt.ylabel("Density")
plt.show()


sns.histplot(data=train, x="curvature", hue="speed_limit", kde=True, element="step", stat="density", common_norm=False)
plt.title("Distribution of Curvature by Speed Limit")
plt.xlabel("Curvature")
plt.ylabel("Density")
plt.show()


sns.boxplot(data=train, x="road_type", y="curvature")
plt.title("Curvature by Road Type")
plt.xlabel("Road Type")
plt.ylabel("Curvature")
plt.xticks(rotation=45)
plt.show()


for col in cat_cols:
    sns.countplot(data=train, y=col)
    plt.title(f"Count of Each Category of {col}")
    plt.show()


for col in bool_cols:
    sns.countplot(data=train, y=col)
    plt.title(f"Count of Each Category of {col}")
    plt.show()


for col in num_cols:
    sns.scatterplot(data=train, x=col, y="accident_risk", alpha=0.5)
    sns.regplot(data=train, x=col, y="accident_risk", scatter=False, color="orange")
    plt.title(f"{col} vs Accident Risk")
    plt.show()


sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


for col in cat_cols:
    sns.boxplot(data=train, x=col, y="accident_risk")
    plt.title(f"Accident Risk vs {col}")
    plt.xticks(rotation=45)
    plt.show()


train.groupby("time_of_day")["accident_risk"].mean().sort_values()


train.groupby("weather")["accident_risk"].mean().sort_values()


for col in bool_cols:
    sns.boxplot(data=train, x=col, y="accident_risk")
    plt.title(f"Accident Risk vs {col}")
    plt.xticks(rotation=45)
    plt.show()


train.groupby("school_season")["accident_risk"].mean().sort_values()


train.head(2)


train["accident_risk"]


X = train.drop(["accident_risk", "id"], axis=1)
y = train["accident_risk"]


X[bool_cols] = X[bool_cols].astype(int)
X.head(5)


num_cols = num_cols.drop("accident_risk")


num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

bool_transformer = "passthrough"


preprocessor = Pipeline(steps=[
    ("transformers", ColumnTransformer(
        transformers = [
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
            ("bool", bool_transformer, bool_cols)
        ]
    ))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def eval_model(pipeline, X_train, X_test, y_train, y_test):
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)

  rmse = mean_squared_error(y_test, y_pred, squared=False)

  return {"Root Mean Squared Error": f"{rmse:.4f}"}


lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


lr_results = eval_model(lr_pipeline, X_train, X_test, y_train, y_test)


lr_results


rmse_scorer = make_scorer(
    lambda y_test, y_pred: np.sqrt(mean_squared_error(y_test, y_pred)),
    greater_is_better=False
)


rmse_scores = -cross_val_score(
    lr_pipeline,
    X_train, y_train,
    cv=5,
    scoring = rmse_scorer,
    n_jobs = -1
)


print("RMSE per fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Std deviation:", rmse_scores.std())


rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators = 500,
        max_depth = 10,
        min_samples_leaf = 5,
        n_jobs = -1,
        random_state = 42
    ))
])


rf_results = eval_model(rf_pipeline, X_train, X_test, y_train, y_test)


rf_results


xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        n_jobs = -1,
        random_state = 42
    ))
])


xgb_results = eval_model(xgb_pipeline, X_train, X_test, y_train, y_test)


xgb_results


lgbm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", lgb.LGBMRegressor(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        num_leaves = 31,
        subsample = 0.8,
        colsample_bytree = 0.8,
        n_jobs = -1,
        random_state = 42
    ))
])


lgbm_results = eval_model(lgbm_pipeline, X_train, X_test, y_train, y_test)


lgbm_results


feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
feature_names


xgb_model = xgb_pipeline.named_steps["regressor"]


importances = xgb_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feat_imp.head(20))


feat_imp.head(20).plot(kind="barh", x="feature", y="importance", legend=False)
plt.title("XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.show()


from sklearn.preprocessing import FunctionTransformer

def add_interactions(X):
    X = X.copy()
    
    X["curvature_speed"] = X["curvature"] * X["speed_limit"]

    X["speed_daylight"] = X["speed_limit"] * (X["lighting"]=="daylight").astype(int)
    X["speed_night"] = X["speed_limit"] * (X["lighting"]=="night").astype(int)
    X["speed_dim"] = X["speed_limit"] * (X["lighting"]=="dim").astype(int)

    X["night_foggy"] = ((X["lighting"] == "night") & (X["weather"] == "foggy")).astype(int)
    X["night_rainy"] = ((X["lighting"] == "night") & (X["weather"] == "rainy")).astype(int)
    X["night_clear"] = ((X["lighting"] == "night") & (X["weather"] == "clear")).astype(int)

    return X


interaction_transformer = FunctionTransformer(add_interactions, validate=False)


xgb_interaction = Pipeline(steps=[
    ("interactions", interaction_transformer),
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        n_jobs = -1,
        random_state = 42
    ))
])


xgb_results_inter = eval_model(xgb_interaction, X_train, X_test, y_train, y_test)


xgb_results_inter


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "regressor__n_estimators": [200, 500, 800, 1000],
    "regressor__max_depth": np.arange(3, 11),
    "regressor__learning_rate": np.linspace(0.01, 0.3, 10),
    "regressor__min_child_weight": np.arange(1, 11),
    "regressor__subsample": np.linspace(0.6, 1.0, 5),
    "regressor__colsample_bytree": np.linspace(0.6, 1.0, 5)
}


search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=param_dist,
    n_iter=30,        # try 30 random combos
    scoring="neg_root_mean_squared_error",
    cv=5,             # 5-fold CV
    verbose=2,
    random_state=42,
    n_jobs=-1
)


search.fit(X_train, y_train)


print("Best params:", search.best_params_)
print("Best CV RMSE:", -search.best_score_)


best_xgb = search.best_estimator_


y_pred = best_xgb.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print("Test RMSE:", rmse)


y_pred = best_xgb.predict(test)


submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": y_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission File Created.")


import joblib
joblib.dump(best_xgb, "best_xgb.pkl")

