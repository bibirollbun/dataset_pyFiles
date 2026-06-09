%pip install -qq scikit-learn lightgbm xgboost --upgrade


import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.model_selection import cross_val_score, cross_validate
import optuna
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns", 140)
pd.set_option("display.max_rows", 250)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


df_train = pd.read_csv(
    "/kaggle/input/playground-series-s5e4/train.csv",
    usecols=[
        "Podcast_Name",
        "Episode_Title",
        "Episode_Length_minutes",
        "Genre",
        "Host_Popularity_percentage",
        "Publication_Day",
        "Publication_Time",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        "Episode_Sentiment",
        "Listening_Time_minutes",
    ],
    dtype={
        "Podcast_Name": "category",
        "Episode_Title": "category",
        "Genre": "category",
        "Publication_Day": "category",
        "Publication_Time": "category",
        "Episode_Sentiment": "category",
    },
)

df_test = pd.read_csv(
    "/kaggle/input/playground-series-s5e4/test.csv",
    usecols=[
        "id",
        "Podcast_Name",
        "Episode_Title",
        "Episode_Length_minutes",
        "Genre",
        "Host_Popularity_percentage",
        "Publication_Day",
        "Publication_Time",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        "Episode_Sentiment",
    ],
    dtype={
        "Podcast_Name": "category",
        "Episode_Title": "category",
        "Genre": "category",
        "Publication_Day": "category",
        "Publication_Time": "category",
        "Episode_Sentiment": "category",
    },
)

df_origin = pd.read_csv(
    "/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv",
    usecols=[
        "Podcast_Name",
        "Episode_Title",
        "Episode_Length_minutes",
        "Genre",
        "Host_Popularity_percentage",
        "Publication_Day",
        "Publication_Time",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        "Episode_Sentiment",
        "Listening_Time_minutes",
    ],
    dtype={
        "Podcast_Name": "category",
        "Episode_Title": "category",
        "Genre": "category",
        "Publication_Day": "category",
        "Publication_Time": "category",
        "Episode_Sentiment": "category",
    },
).dropna(subset="Listening_Time_minutes")


df_train = df_train[
    (
        (df_train["Episode_Length_minutes"] <= 320)
        & (df_train["Episode_Length_minutes"] >= 2)
    )
    & (df_train["Number_of_Ads"] <= 3)
]
df_train.drop_duplicates(
    subset={
        "Podcast_Name",
        "Episode_Title",
        "Episode_Length_minutes",
        "Listening_Time_minutes",
    },
    inplace=True,
)
df_test["Episode_Length_minutes"] = np.where(
    df_test["Episode_Length_minutes"] > 120.73,
    np.NAN,
    df_test["Episode_Length_minutes"],
)
df_test["Number_of_Ads"] = np.where(
    df_test["Number_of_Ads"] > 3, np.NAN, df_test["Number_of_Ads"]
)


TARGET = "Listening_Time_minutes"
train_cat_cols = df_train.select_dtypes(include="category").columns.to_list()
train_num_cols = df_train.select_dtypes(exclude="category").columns.to_list()
if "id" in df_test.columns.to_list():
    ids = df_test.id
    df_test.drop(["id"], axis=1, inplace=True)
test_cat_cols = df_test.select_dtypes(include="category").columns.to_list()
test_num_cols = df_test.select_dtypes(exclude="category").columns.to_list()
kf = KFold(n_splits=5, shuffle=True, random_state=42)


X = df_train.drop(TARGET, axis=1)
X_origin = df_origin.drop(TARGET, axis=1)
y = df_train[TARGET]
y_origin = df_origin[TARGET]


df_train


df_test


for col in train_cat_cols:
    print(df_train[col].value_counts())
    print("\n")


for col in test_cat_cols:
    print(df_test[col].value_counts())
    print("\n")


df_train[train_num_cols].describe()


df_test[test_num_cols].describe()


df_train[train_num_cols].plot.kde(subplots=True, figsize=(12, 10), layout=(3, 2))


df_test[test_num_cols].plot.kde(subplots=True, figsize=(12, 10), layout=(2, 2))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
for i, col in enumerate(
    [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
    ],
    1,
):
    plt.subplot(3, 2, i)
    df_train[col].value_counts().nlargest(48).plot.bar()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 10))
for i, col in enumerate(
    [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
    ],
    1,
):
    plt.subplot(3, 2, i)
    df_test[col].value_counts().nlargest(48).plot.bar()

plt.tight_layout()
plt.show()


from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class ColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        construct_features=True,
        drop_columns=True,
        mix_enable=False,
        normalization=True,
    ):
        self.columns = []
        self.construct_features = construct_features
        self.drop_columns = drop_columns
        self.mix_enable = mix_enable
        self.normalization = normalization

    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers

    def get_columns(self):
        return self.columns

    def transform(self, X: pd.DataFrame):
        X_transformed = X.copy()
            

        self.cat_cols = ["Podcast_Name", "Episode_Title", "Genre"]
        self.num_cols = X_transformed.select_dtypes(include=np.number).columns.to_list()
        X_transformed["Episode_Length_minutes"] = X_transformed.groupby(
            ["Podcast_Name", "Episode_Title"], observed=False
        )["Episode_Length_minutes"].transform(lambda x: x.fillna(x.median()))
        X_transformed["Guest_Popularity_percentage"] = X_transformed.groupby(
            ["Podcast_Name", "Episode_Title"], observed=False
        )["Guest_Popularity_percentage"].transform(lambda x: x.fillna(x.median()))
        X_transformed[self.num_cols] = SimpleImputer(strategy="median").fit_transform(
            X_transformed[self.num_cols]
        )
        if self.construct_features:
            X_transformed["SinEpLen"] = np.sin(
                2 * np.pi * X_transformed["Episode_Length_minutes"] / 60
            )
            X_transformed["CosEpLen"] = np.cos(
                2 * np.pi * X_transformed["Episode_Length_minutes"] / 60
            )
            X_transformed["ELen_Int"] = np.floor(
                X_transformed["Episode_Length_minutes"]
            )
            X_transformed["ELen_Dec"] = (
                X_transformed["Episode_Length_minutes"] - X_transformed["ELen_Int"]
            )
            X_transformed["Popularity_Score"] = X_transformed[
                "Guest_Popularity_percentage"
            ] / (X_transformed["Host_Popularity_percentage"] + 1e-3)
            X_transformed["Ad_Density"] = X_transformed[
                "Number_of_Ads"
            ] / X_transformed["Episode_Length_minutes"].clip(lower=1)
            X_transformed["Length_Adjusted_Popularity"] = X_transformed[
                "Popularity_Score"
            ] * np.log1p(X_transformed["Episode_Length_minutes"])
            if self.construct_features:
                self.num_cols = self.num_cols + [
                    "ELen_Int",
                    "ELen_Dec",
                    "Popularity_Score",
                    "Ad_Density",
                    "Length_Adjusted_Popularity",
                ]
            X_transformed["Length_Bucket"] = (
                pd.cut(
                    X_transformed["ELen_Int"],
                    bins=[-np.Infinity, 30, 60, 90, np.Infinity],
                    include_lowest=True,
                    labels=["short", "medium", "long", "very long"],
                )
                .map(
                    {
                        "short": 0,
                        "medium": 1,
                        "long": 2,
                        "very long": 3,
                    }
                )
                .astype(dtype="int8")
            )
            X_transformed[self.cat_cols] = SimpleImputer(
                strategy="most_frequent"
            ).fit_transform(X_transformed[self.cat_cols])
        X_transformed["Publication_Day"] = (
            X_transformed["Publication_Day"]
            .map(
                {
                    "Sunday": 0,
                    "Monday": 1,
                    "Tuesday": 2,
                    "Wednesday": 3,
                    "Thursday": 4,
                    "Friday": 5,
                    "Saturday": 6,
                }
            )
            .astype(dtype="int8")
        )
        X_transformed["Publication_Time"] = (
            X_transformed["Publication_Time"]
            .map(
                {
                    "Morning": 0,
                    "Afternoon": 1,
                    "Evening": 2,
                    "Night": 3,
                }
            )
            .astype(dtype="int8")
        )
        X_transformed["Episode_Sentiment"] = (
            X_transformed["Episode_Sentiment"]
            .map({"Negative": -1, "Neutral": 0, "Positive": 1})
            .astype(dtype="int8")
        )
        if self.construct_features:
            X_transformed["Is_Weekend"] = np.where(
                (
                    (X_transformed["Publication_Day"] == 6)
                    | (X_transformed["Publication_Day"] == 0)
                ),
                1,
                0,
            )
            X_transformed["Is_Add"] = np.where(
                (X_transformed["Number_of_Ads"] == 0),
                0,
                1,
            )
            X_transformed["Is_Morning"] = (
                X_transformed["Publication_Time"] == 0
            ).astype(dtype="int8")
            X_transformed["Is_Night"] = (X_transformed["Publication_Time"] == 3).astype(
                dtype="int8"
            )
            if self.mix_enable:
                X_mix_cat_cols = X_transformed.copy()[
                    self.cat_cols
                    + [
                        "Length_Bucket",
                        "Publication_Day",
                        "Publication_Time",
                        "Episode_Sentiment",
                        "Is_Weekend",
                        "Is_Add",
                        "Is_Morning",
                        "Is_Night",
                    ]
                ].astype(dtype="str")
                for left_col, right_col in combinations(
                    self.cat_cols
                    + [
                        "Length_Bucket",
                        "Publication_Day",
                        "Publication_Time",
                        "Episode_Sentiment",
                        "Is_Weekend",
                        "Is_Add",
                        "Is_Morning",
                        "Is_Night",
                    ],
                    2,
                ):
                    new_col = left_col + "-" + right_col
                    X_transformed[new_col] = (
                        X_mix_cat_cols[left_col].str.lower()
                        + " - "
                        + X_mix_cat_cols[right_col].str.lower()
                    )

                    if new_col not in self.cat_cols:
                        self.cat_cols.append(new_col)
                del X_mix_cat_cols
                for left_col, right_col in combinations(self.num_cols, 2):
                    colName = left_col + "_SUB_" + right_col + "_DIV_" + left_col
                    X_transformed[colName] = np.where(
                        (
                            ((X_transformed[left_col] - X_transformed[right_col]) == 0)
                            | (X_transformed[left_col] == 0)
                        ),
                        0,
                        (X_transformed[left_col] - X_transformed[right_col])
                        / X_transformed[left_col],
                    )
            X_transformed[self.cat_cols] = X_transformed[self.cat_cols].astype(
                dtype="category"
            )
            X_transformed["SinWeekday"] = np.sin(
                2 * np.pi * X_transformed["Publication_Day"] / 7
            )
            X_transformed["CosWeekday"] = np.cos(
                2 * np.pi * X_transformed["Publication_Day"] / 7
            )
            X_transformed["SinTime"] = np.sin(
                2 * np.pi * X_transformed["Publication_Time"] / 4
            )
            X_transformed["CosTime"] = np.cos(
                2 * np.pi * X_transformed["Publication_Time"] / 4
            )
        X_transformed[self.cat_cols] = (
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            .fit_transform(X_transformed[self.cat_cols])
            .astype(dtype="int16")
        )
        if self.normalization:

            X_transformed[self.num_cols] = StandardScaler().fit_transform(
                X_transformed[self.num_cols]
            )

        if self.drop_columns:
            X_transformed.drop(
                ["Number_of_Ads"],
                axis=1,
                inplace=True,
            )
        self.columns = X_transformed.columns.to_list()

        return X_transformed


df_train_trans = ColumnTransformer(drop_columns=False, normalization=False).transform(
    df_train
)
df_test_trans = ColumnTransformer(drop_columns=False, normalization=False).transform(
    df_test
)


extended_cols = train_num_cols + [
    "ELen_Int",
    "ELen_Dec",
    "Popularity_Score",
    "Ad_Density",
    "Length_Adjusted_Popularity",
]
extended_cols.remove("Listening_Time_minutes")


df_train_trans[extended_cols].plot.kde(subplots=True, figsize=(12, 18), layout=(3, 3))


df_test_trans[extended_cols].plot.kde(subplots=True, figsize=(12, 18), layout=(3, 3))


df_train_trans[extended_cols].describe()


df_test_trans[extended_cols].describe()


corr = df_train_trans.corr()


import seaborn as sns


plt.figure(figsize=(32, 24))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)


corr["Listening_Time_minutes"].sort_values()


def select_features(X, y, est, params, cv=5, scoring="neg_root_mean_squared_error"):
    X_trans = ColumnTransformer(mix_enable=False).fit_transform(X)
    model_all_features = est(**params)

    output = cross_validate(
        model_all_features,
        X_trans,
        y,
        cv=cv,
        scoring=scoring,
        return_estimator=True,
    )

    fi = []
    for estimator in output["estimator"]:
        fi.append(estimator.feature_importances_)

    fi = pd.DataFrame(
        np.array(fi).T,
        columns=["importance " + str(idx) for idx in range(len(fi))],
        index=X_trans.columns,
    )

    fi["mean_importance"] = fi.mean(axis=1)

    features = fi["mean_importance"]
    features = features.sort_values(ascending=True)

    features.plot.barh(figsize=(10, 20))

    score_all = np.abs(output["test_score"].mean())
    print("Mean Score: ", score_all)

    tol = 0.0001
    features_to_remove = []
    score_mean_list = []
    diff_auc_list = []
    count = 1

    features = list(features.index)

    for feature in features:
        print("=" * 50)
        print("Check feature: ", feature, " feature ", count, " of ", len(features))
        count = count + 1
        model = est(**params)

        scores = cross_val_score(
            model,
            X_trans.drop(features_to_remove + [feature], axis=1),
            y,
            scoring=scoring,
            cv=cv,
        )

        score_mean = np.abs(scores.mean())
        print("Score model after removing={}".format(score_mean))
        score_mean_list.append(score_mean)
        print("Score model with all features={}".format(score_all))
        diff_auc = score_all - score_mean
        diff_auc_list.append(diff_auc)
        print("Difference Score={}".format(diff_auc))
        if diff_auc <= tol:
            print("keep: ", feature)
        else:
            print("remove: ", feature)
            score_all = score_mean
            features_to_remove.append(feature)
        print("=" * 50)
    df = pd.DataFrame(
        {
            "feature": features,
            "score_mean": score_mean_list,
            "diff_score": diff_auc_list,
        }
    )

    print("DONE!!")
    print("Total features for removing: ", len(features_to_remove))

    features_to_keep = [x for x in features if x not in features_to_remove]

    print("Total features for keeping: ", len(features_to_keep))

    return (df.sort_values(by="diff_score", ascending=False), features_to_keep)


select_features(
    X,
    y,
    LGBMRegressor,
    {
        "device": "gpu",
        "random_state": 42,
        "importance_type": "gain",
        "verbosity": -1,
    },
)


select_features(
    X,
    y,
    XGBRegressor,
    {
        "random_state": 42,
        "verbosity": 0,
        "device": "cuda",
        "importance_type": "gain",
    },
)


select_features(
    X_origin,
    y_origin,
    RandomForestRegressor,
    {
        "random_state": 42,
        "n_jobs": -1
    },
)


SELECTED_LGB_FEATURES = [
    "Is_Morning",
    "Is_Night",
    "Is_Add",
    "Publication_Time",
    "Is_Weekend",
    "CosWeekday",
    "SinTime",
    "Popularity_Score",
    "SinWeekday",
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "ELen_Dec",
    "Guest_Popularity_percentage",
    "Episode_Sentiment",
    "Host_Popularity_percentage",
    "SinEpLen",
    "Ad_Density",
    "CosEpLen",
    "Length_Bucket",
    "Episode_Length_minutes",
    "ELen_Int",
]

SELECTED_RF_FEATURES = [
    "Is_Morning",
    "Is_Night",
    "Is_Weekend",
    "Publication_Time",
    "CosTime",
    "CosWeekday",
    "Episode_Sentiment",
    "Publication_Day",
    "SinWeekday",
    "Genre",
    "Length_Adjusted_Popularity",
    "SinEpLen",
    "Podcast_Name",
    "Guest_Popularity_percentage",
    "ELen_Dec",
    "Episode_Title",
    "ELen_Int",
    "CosEpLen",
    "Host_Popularity_percentage",
    "Ad_Density",
    "Episode_Length_minutes",
]

SELECTED_XGB_FEATURES = [
    "Is_Morning",
    "Is_Night",
    "Length_Bucket",
    "Publication_Time",
    "Popularity_Score",
    "CosWeekday",
    "Episode_Title",
    "Is_Add",
    "Publication_Day",
    "SinTime",
    "Guest_Popularity_percentage",
    "Is_Weekend",
    "ELen_Dec",
    "CosTime",
    "SinWeekday",
    "Genre",
    "Host_Popularity_percentage",
    "Episode_Sentiment",
    "SinEpLen",
    "Ad_Density",
    "CosEpLen",
    "Episode_Length_minutes",
    "ELen_Int",
]


class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_clip = X[self.columns]
        return X_clip.values


def optimize_lightgbm(
    X,
    y,
):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, shuffle=True
    )

    X_test = ColumnTransformer().fit_transform(X_test)[SELECTED_LGB_FEATURES]

    del X, y

    def objective(trial: optuna.Trial):
        """Define the objective function for LGB Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "num_leaves": trial.suggest_int("num_leaves", 2, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 500),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "max_bin": trial.suggest_int("max_bin", 2, 255),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100.0, log=True),
            "random_state": 42,
            "device": "gpu",
            "n_jobs": -1,
            "verbosity": -1,
        }

        model = LGBMRegressor(**params)
        scores = []
        for i, (tr, val) in enumerate(kf.split(X_train)):
            X_spt_train, X_val, y_spt_train, y_val = (
                X_train.iloc[tr, :],
                X_train.iloc[val, :],
                y_train.iloc[tr],
                y_train.iloc[val],
            )
            X_spt_train = ColumnTransformer().fit_transform(X_spt_train)[
                SELECTED_LGB_FEATURES
            ]
            X_val = ColumnTransformer().fit_transform(X_val)[SELECTED_LGB_FEATURES]
            model.fit(
                X_spt_train,
                y_spt_train,
                eval_set=[(X_val, y_val)],
                eval_names=["valid"],
                eval_metric=["rmse"],
                callbacks=[
                    early_stopping(stopping_rounds=100),
                ],
            )
            rmse = root_mean_squared_error(y_test, model.predict(X_test))
            scores.append(rmse)

            print(f"Fold {i}:", rmse)

        cv = abs(np.mean(scores))

        print("=" * 50)
        print("\n")

        return cv

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Get the best parameters
    best_params = study.best_params
    return best_params


def optimize_line(X, y):

    def objective(trial: optuna.Trial):
        """Define the objective function for Line Model"""

        params = {
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "random_state": 42,
        }

        model = Lasso(**params)
        model_pipe = make_pipeline(ColumnTransformer(), model)

        cv = abs(
            cross_val_score(
                model_pipe, X, y, cv=5, scoring="neg_root_mean_squared_error"
            ).mean()
        )

        return cv

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    # Get the best parameters
    best_params = study.best_params
    return best_params


def optimize_rf(X, y):

    def objective(trial: optuna.Trial):
        """Define the objective function for RF Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_features": trial.suggest_int("max_features", 1, 21),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
        }

        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model_pipe = make_pipeline(
            ColumnTransformer(), SelectFeatures(columns=SELECTED_RF_FEATURES), model
        )

        cv = abs(
            cross_val_score(
                model_pipe, X, y, cv=5, scoring="neg_root_mean_squared_error"
            ).mean()
        )

        return cv

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Get the best parameters
    best_params = study.best_params
    return best_params

def optimize_xgb(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, shuffle=True
    )

    X_test = ColumnTransformer().fit_transform(X_test)[SELECTED_XGB_FEATURES]

    del X, y

    def objective(trial: optuna.Trial):
        """Define the objective function for XGB Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-9, 0.5, log=True),
            "device": "cuda",
            "random_state": 42,
            "verbosity": 0,
        }

        model = XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )

        scores = []
        for i, (tr, val) in enumerate(kf.split(X_train)):
            X_spt_train, X_val, y_spt_train, y_val = (
                X_train.iloc[tr, :],
                X_train.iloc[val, :],
                y_train.iloc[tr],
                y_train.iloc[val],
            )
            X_spt_train = ColumnTransformer().fit_transform(X_spt_train)[
                SELECTED_XGB_FEATURES
            ]
            X_val = ColumnTransformer().fit_transform(X_val)[SELECTED_XGB_FEATURES]
            model.fit(X_spt_train, y_spt_train, eval_set=[(X_val, y_val)], verbose=0)
            rmse = root_mean_squared_error(y_test, model.predict(X_test))
            scores.append(rmse)

            print(f"Fold {i}:", rmse)

        cv = abs(np.mean(scores))

        print("=" * 50)
        print("\n")

        return cv

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Get the best parameters
    best_params = study.best_params
    return best_params


optimize_line(X_origin, y_origin)


optimize_lightgbm(X_origin, y_origin)


optimize_rf(X_origin, y_origin)


optimize_xgb(X_origin, y_origin)


lgbm_params = {
    "n_estimators": 477,
    "learning_rate": 0.022251843975067902,
    "max_depth": 4,
    "num_leaves": 55,
    "min_data_in_leaf": 16,
    "min_gain_to_split": 11.733150962349885,
    "max_bin": 34,
    "subsample": 0.9437484695495751,
    "subsample_freq": 5,
    "feature_fraction": 0.578182665774394,
    "reg_lambda": 1.8623068615119918e-08,
    "reg_alpha": 0.001621123364314717,
    "device": "gpu",
    "verbosity": -1,
    "random_state": 42,
}
rf_params = {
    "n_estimators": 138,
    "max_features": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}
xgb_params = {
    "n_estimators": 345,
    "learning_rate": 0.022240043176385078,
    "min_child_weight": 6,
    "max_depth": 4,
    "subsample": 0.7669411076539501,
    "colsample_bytree": 0.48011525616387596,
    "reg_lambda": 6.410016007993395e-05,
    "reg_alpha": 0.0023304343983831114,
    "gamma": 0.0017591964938196607,
    "device": "cuda",
    "verbosity": 0,
    "eval_metric": "rmse",
    "random_state": 42,
}


lgb_model = LGBMRegressor(**lgbm_params)
line_model = Lasso(alpha=0.0005268617929600615, random_state=42)
rf_model = RandomForestRegressor(**rf_params)
xgb_model = XGBRegressor(**xgb_params)

lgb_pipe = make_pipeline(SelectFeatures(columns=SELECTED_LGB_FEATURES), lgb_model)
rf_pipe = make_pipeline(SelectFeatures(columns=SELECTED_RF_FEATURES), rf_model)
xgb_pipe = make_pipeline(SelectFeatures(columns=SELECTED_XGB_FEATURES), xgb_model)


X_train, X_test, y_train, y_test = train_test_split(
    X_origin, y_origin, test_size=0.2, random_state=42, shuffle=True
)

X_train = ColumnTransformer().fit_transform(X_train)
X_test = ColumnTransformer().fit_transform(X_test)

lgb_pipe.fit(X_train, y_train)
line_model.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

meta_train_predictions = np.column_stack(
    [
        lgb_pipe.predict(X_train),
        line_model.predict(X_train),
        rf_pipe.predict(X_train),
        xgb_pipe.predict(X_train),
    ]
)


def optimize_meta_model():
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(
        meta_train_predictions, y_train, random_state=42, test_size=0.3, shuffle=True
    )
    def objective(trial: optuna.Trial):
        """Define the objective function for XGB Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-9, 0.5, log=True),
            "device": "cuda",
            "random_state": 42,
            "verbosity": 0,
        }

        model = XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )

        scores = []
        for i, (tr, val) in enumerate(kf.split(X_sub_train)):
            X_spt_train, X_val, y_spt_train, y_val = (
                X_sub_train[tr, :],
                X_sub_train[val, :],
                y_sub_train.iloc[tr],
                y_sub_train.iloc[val],
            )
            model.fit(X_spt_train, y_spt_train, eval_set=[(X_val, y_val)], verbose=0)
            rmse = root_mean_squared_error(y_sub_test, model.predict(X_sub_test))
            scores.append(rmse)

            print(f"Fold {i}:", rmse)

        cv = abs(np.mean(scores))

        print("=" * 50)
        print("\n")

        return cv

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    # Get the best parameters
    best_params = study.best_params
    return best_params


optimize_meta_model()


final_est_params = {
    "n_estimators": 137,
    "learning_rate": 0.1382964507273623,
    "min_child_weight": 8,
    "max_depth": 5,
    "subsample": 0.7864173849750221,
    "colsample_bytree": 0.8846450377954609,
    "reg_lambda": 1.9940538323978057e-08,
    "reg_alpha": 5.452890032850836,
    "gamma": 1.2369960080174684e-08,
    "random_state": 42,
    "device": "cuda",
    "verbosity": 0,
    "eval_metric": "rmse",
    "random_state": 42,
}


best_meta_model = XGBRegressor(**final_est_params)
best_meta_model.fit(meta_train_predictions, y_train)


meta_test_predictions = np.column_stack([
    lgb_pipe.predict(X_test),
    line_model.predict(X_test),
    rf_pipe.predict(X_test),
    xgb_pipe.predict(X_test),
])

final_predictions = best_meta_model.predict(meta_test_predictions)
rmse = root_mean_squared_error(y_test, final_predictions)
print(f"RMSE on Test Data: {rmse}")


from sklearn.ensemble import StackingRegressor

estimators = [
    ("LGBM", lgb_pipe),
    ("Lasso", line_model),
    ("RF", rf_pipe),
    ("XGBM", xgb_pipe),
]

stack_pipeline = make_pipeline(
    ColumnTransformer(),
    StackingRegressor(
        estimators=estimators, final_estimator=XGBRegressor(**final_est_params)
    ),
)


print(
    "Mean Score: ",
    np.mean(
        cross_val_score(
            stack_pipeline,
            X,
            y,
            verbose=10,
            scoring="neg_root_mean_squared_error",
        )
    ),
)


stack_pipeline.fit(X, y)


pred = stack_pipeline.predict(df_test)



submission = pd.DataFrame({"id": ids, TARGET: pred})
submission.to_csv("submission.csv", index=False)

