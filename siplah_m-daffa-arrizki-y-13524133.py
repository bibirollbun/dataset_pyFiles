# Bagian 1: Import & Load Data
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans

from category_encoders import TargetEncoder

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")



# Bagian 2: Data Cleaning & Konversi Tipe
drop_txt = [
    "name","description","neighborhood_overview",
    "host_name","host_about","host_location",
    "host_neighbourhood","host_verifications","bathrooms_text"
]
for df in (train, test):
    df.drop(columns=[c for c in drop_txt if c in df.columns], inplace=True)

for df in (train, test):
    for pct in ["host_response_rate","host_acceptance_rate"]:
        if pct in df:
            df[pct] = (
                df[pct].str.rstrip("%")
                      .replace("", np.nan)
                      .astype(float)
                      .div(100)
            )

for df in (train, test):
    for col in ["first_review","last_review","host_since"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

to_drop = train.isnull().mean().loc[lambda x: x>0.7].index.tolist()
for df in (train, test):
    df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)



# Bagian 3: Feature Engineering
import ast

def feature_engineering(df, is_train=True, clip_vals=None):
    df = df.copy()
    
    if "host_since" in df:
        df["host_experience_days"] = (pd.Timestamp("today") - df["host_since"]).dt.days

    df["amenities_count"] = (
        df["amenities"].fillna("[]")
        .apply(lambda x: len(ast.literal_eval(x)) if isinstance(x,str) else 0)
    )

    rev_cols = [
        "review_scores_rating","review_scores_accuracy","review_scores_cleanliness",
        "review_scores_checkin","review_scores_communication",
        "review_scores_location","review_scores_value"
    ]
    df["review_score_avg"] = df[rev_cols].mean(axis=1)

    df["availability_ratio"] = df["availability_365"] / 365

    for col in ["last_review","first_review"]:
        if col in df:
            df[f"days_since_{col}"] = (pd.Timestamp("today") - df[col]).dt.days

    if clip_vals is None and is_train:
        clip_vals = {}

    return df, clip_vals

train, _ = feature_engineering(train, is_train=True)
test, _  = feature_engineering(test, is_train=False)

coords = train[["latitude","longitude"]].dropna()
km = KMeans(n_clusters=25, random_state=42)
train["geo_cluster"] = km.fit_predict(coords)
test["geo_cluster"]  = km.predict(test[["latitude","longitude"]].fillna(0))

for df in (train, test):
    for c in ["amenities","host_since","first_review","last_review","availability_365"]:
        if c in df: df.drop(columns=c, inplace=True)



# Bagian 4: Feature Encoding 

bool_cols = [
    "host_is_superhost","host_has_profile_pic",
    "host_identity_verified","has_availability"
]
for df in (train, test):
    for c in bool_cols:
        if c in df:
            df[c] = df[c].map({"t":1,"f":0}).fillna(0).astype(int)

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
for df in (train, test):
    if "host_response_time" in df.columns:
        df["host_response_time_enc"] = oe.fit_transform(
            df[["host_response_time"]].fillna("missing")
        ) if df is train else oe.transform(
            df[["host_response_time"]].fillna("missing")
        )
        df.drop(columns="host_response_time", inplace=True)

for df in (train, test):
    if "neighbourhood" in df.columns:
        df.drop(columns="neighbourhood", inplace=True)

tenc_cols = ["neighbourhood_cleansed","city","property_type","room_type"]
te = TargetEncoder(cols=tenc_cols, smoothing=0.3)
train[tenc_cols] = te.fit_transform(train[tenc_cols], np.log1p(train["price"]))
test [tenc_cols] = te.transform(test [tenc_cols])



# Bagian 5: Text Features (TF-IDF + SVD)
df_desc = pd.read_csv("train.csv")[["description"]].fillna("")
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X_desc = tfidf.fit_transform(df_desc["description"])

svd = TruncatedSVD(n_components=50, random_state=42)
desc_svd = svd.fit_transform(X_desc)

train_svd = pd.DataFrame(desc_svd, columns=[f"svd_desc_{i}" for i in range(50)])
train = pd.concat([train.reset_index(drop=True), train_svd], axis=1)

df_desc_t = pd.read_csv("test.csv")[["description"]].fillna("")
X_desc_t = tfidf.transform(df_desc_t["description"])
desc_svd_t = svd.transform(X_desc_t)
test_svd = pd.DataFrame(desc_svd_t, columns=[f"svd_desc_{i}" for i in range(50)])
test = pd.concat([test.reset_index(drop=True), test_svd], axis=1)



# Bagian 6: Feature-Matrix & Preprocessing Pipeline

for df in (train, test):
    if "price_clipped" in df.columns:
        df.drop(columns="price_clipped", inplace=True)

numeric = train.select_dtypes(include=[np.number]).columns.drop("price")

X_train = train[numeric].copy()
y_train = np.log1p(train["price"])
X_test  = test [numeric].copy()

preproc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

X_train[numeric] = preproc.fit_transform(X_train[numeric])
X_test [numeric] = preproc.transform(X_test [numeric])



# Bagian 7: Tuning LightGBM dengan Optuna 
def objective(trial):
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 200),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_estimators":     1000,
        "random_state":     42,
        "n_jobs":           -1,
    }
    model = lgb.LGBMRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    return -scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)

best_params = study.best_params
print("Best CV RMSE:", study.best_value)
print("Best LightGBM params:", best_params)



# Bagian 8: Stacking 3 base models
base_models = [
    ("lgbm", lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)),
    ("xgb",  xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        learning_rate=0.05, n_estimators=500, random_state=42, n_jobs=-1)),
    ("cat",  cb.CatBoostRegressor(verbose=0, iterations=500, random_state=42))
]
stack = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV(),
    cv=KFold(5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=False
)

stack.fit(X_train, y_train)

y_pred_log = stack.predict(X_test)
y_pred     = np.expm1(y_pred_log)



# Bagian 9: submission
submission = pd.DataFrame({
    "id":    test["id"],
    "price": y_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission shape:", submission.shape)


