%pip install -qq scikit-learn lightgbm xgboost --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import Lasso, Ridge

from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor

import optuna

import seaborn as sns


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


df_train = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/train.csv",
    usecols=[
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "Calories",
    ],
)
df_test = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/test.csv",
    usecols=[
        "id",
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
    ],
)
df_origin = pd.read_csv(
    "/kaggle/input/calories-burnt-prediction/calories.csv",
    usecols=[
        "Age",
        "Gender",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "Calories",
    ],
)


df_train.drop_duplicates(keep="first", inplace=True)
df_origin.drop_duplicates(keep="first", inplace=True)


df_train


df_train.describe()


df_test.drop("id", axis=1).describe()


df_origin.describe()


df_train_copy = df_train.copy()

df_train_copy["Sex"] = np.where((df_train_copy["Sex"] == "male"), 1, 0).astype("int8")

corr = df_train_copy.corr()

corr["Calories"].sort_values()


plt.figure(figsize=(16, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

del df_train_copy


df_origin_copy = df_origin.copy()

df_origin_copy["Gender"] = np.where((df_origin_copy["Gender"] == "male"), 1, 0).astype(
    "int8"
)

corr = df_origin_copy.corr()

corr["Calories"].sort_values()


plt.figure(figsize=(16, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

del df_origin_copy


df_test_copy = df_test.copy()

df_test_copy["Sex"] = np.where((df_test_copy["Sex"] == "male"), 1, 0).astype("int8")

corr = df_test_copy.corr()

plt.figure(figsize=(16, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

del df_test_copy


df_train.plot.kde(subplots=True, figsize=(16, 20), layout=(4, 2))


df_origin.plot.kde(subplots=True, figsize=(16, 20), layout=(4, 2))


df_test.drop("id", axis=1).plot.kde(subplots=True, figsize=(16, 20), layout=(3, 2))


df_train["Sex"].value_counts().plot.hist()


df_origin["Gender"].value_counts().plot.hist()


df_test["Sex"].value_counts().plot.hist()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Age"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Age"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Height"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Height"].sort_values().unique()


df_train = df_train[(df_train["Height"] < 217) & (df_train["Height"] > 129)]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Height"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Height"].sort_values().unique()


df_origin = df_origin[(df_origin["Height"] < 217) & (df_origin["Height"] > 132)]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Weight"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Weight"].sort_values().unique()


df_train = df_train[df_train["Weight"] < 124]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Weight"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Weight"].sort_values().unique()


df_origin = df_origin[df_origin["Weight"] < 124]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Duration"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Duration"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Heart_Rate"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Heart_Rate"].sort_values().unique()


df_train = df_train[df_train["Heart_Rate"] < 126]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Heart_Rate"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Heart_Rate"].sort_values().unique()


df_origin = df_origin[df_origin["Heart_Rate"] < 128]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Body_Temp"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Body_Temp"].sort_values().unique()


df_train = df_train[df_train["Body_Temp"] > 37.9]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Body_Temp"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Body_Temp"].sort_values().unique()


df_origin = df_origin[df_origin["Body_Temp"] > 38.0]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Calories"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Calories"].sort_values().unique()


df_train = df_train[df_train["Calories"] < 289]


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Calories"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Calories"].sort_values().unique()


df_origin = df_origin[df_origin["Calories"] < 295]


df_train.shape, df_origin.shape


from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

TARGET = "Calories"
df_origin.rename(columns={"Gender": "Sex"}, inplace=True)
num_columns = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Body_Temp",
    "Heart_Rate",
    "BMI",
    "Max_Heart_Rate",
    "%_Max_Heart_Rate",
    "Temp_Deviation",
]


kf = KFold(n_splits=5, random_state=42, shuffle=True)


def preprocessing(
    df: pd.DataFrame,
    normalize=True,
    drop_target=True,
    TARGET=TARGET,
    mix_cat_cols=True,
    mix_num_cols=True,
):
    X = df.copy()
    y = None
    X["Age_Group"] = pd.cut(
        X["Age"],
        [18, 25, 40, 60, np.Infinity],
        include_lowest=True,
        labels=[1, 2, 3, 4],
    ).astype("int8")
    X["Sex"] = np.where((X["Sex"] == "male"), 1, 0).astype("int8")
    X["Sex_Reversed"] = 1 - X["Sex"]
    X["Max_Heart_Rate"] = 220 - X["Age"]
    X["%_Max_Heart_Rate"] = np.floor(
        X["Heart_Rate"] / X["Max_Heart_Rate"] * 100
    ).astype("int8")

    X["Moderate_intensity"] = np.where(X["%_Max_Heart_Rate"] < 70, 1, 0).astype("int8")
    X["Vigourous_intensity"] = np.where(X["%_Max_Heart_Rate"] >= 70, 1, 0).astype(
        "int8"
    )

    X["BMI"] = np.round(X["Weight"] / ((X["Height"] / 100) ** 2), 2)
    X["Body_Composition"] = pd.cut(
        X["BMI"],
        [-np.Infinity, 18.4, 24.9, 30.0, np.Infinity],
        include_lowest=True,
        labels=[-1, 0, 1, 2],
    ).astype("int8")

    X["Duration_Long"] = pd.cut(
        X["Duration"],
        bins=[0, 10, 20, np.Infinity],
        include_lowest=True,
        labels=[1, 2, 3],
    ).astype("int8")
    X["Overweight"] = np.where(X["Body_Composition"] > 0, 1, 0).astype("int8")
    X["Obese"] = np.where(X["Body_Composition"] > 1, 1, 0).astype("int8")
    X["Underweight"] = np.where(X["Body_Composition"] == -1, 1, 0).astype("int8")
    X["Temp_Deviation"] = np.round(X["Body_Temp"] - 36.6).astype("int8")
    new_cat_cols = []
    if mix_cat_cols:
        new_cat_cols = []
        for left_col, right_col in combinations(
            [
                "Age_Group",
                "Body_Composition",
                "Moderate_intensity",
                "Sex",
                "Duration_Long",
                "Vigourous_intensity",
            ],
            2,
        ):
            new_cat_col = left_col + "_vs_" + right_col
            if not (
                new_cat_col
                in [
                    "Overweight_vs_Underweight",
                    "Body_Composition_vs_Overweight",
                    "Body_Composition_vs_Underweight",
                    "Moderate_intensity__vs_Vigourous_intensity",
                ]
            ):
                X[new_cat_col] = (
                    X[left_col].astype(str) + " - " + X[right_col].astype(str)
                )
                new_cat_cols.append(new_cat_col)

    if mix_num_cols:
        for left_col, right_col in combinations(
            [
                "Age",
                "Body_Temp",
                "Duration",
                "Heart_Rate",
                "BMI",
            ],
            2,
        ):
            new_div_col = left_col + "_DIV_" + right_col
            new_mul_col = left_col + "_MUL_" + right_col
            X[new_div_col] = X[left_col] / X[right_col]
            X[new_mul_col] = X[left_col] * X[right_col]
        for f1 in ["Duration", "Heart_Rate", "Body_Temp"]:
            for f2 in ["Sex", "Sex_Reversed"]:
                X[f"{f1}_MUL_{f2}"] = X[f1] * X[f2]

        X.drop(columns=["Sex_Reversed"], inplace=True)

    if normalize:
        X[new_cat_cols] = (
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            .fit_transform(X[new_cat_cols])
            .astype(dtype="int16")
        )
        if TARGET in X.columns.to_list():
            X[TARGET] = np.log1p(X[TARGET])

    if drop_target:
        y = X[TARGET]
        X.drop(TARGET, axis=1, inplace=True)
    return X, y


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted


class CustomLGBMRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, params, features=None):
        self.params = params
        self.features = features

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    def fit(self, X, y):

        # Check that X and y have correct shape, set n_features_in_, etc.
        if self.features is None:
            X, y = validate_data(self, X, y)
            self.X_ = X
        else:
            X, y = validate_data(self, X[self.features], y)
            self.X_ = pd.DataFrame(X, columns=self.features)
        self.y_ = y

        X_train_, X_val_, y_train_, y_val_ = train_test_split(
            self.X_,
            self.y_,
            random_state=42,
            test_size=0.1,
            shuffle=True,
        )
        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(
            X_train_,
            y_train_,
            eval_set=[(X_val_, y_val_)],
            eval_names=["valid"],
            eval_metric=["rmse"],
            callbacks=[
                early_stopping(stopping_rounds=100),
            ],
        )
        return self.model_

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if self.features is None:
            X = validate_data(self, X, reset=False)
        else:
            X = pd.DataFrame(
                validate_data(self, X[self.features], reset=False),
                columns=self.features,
            )
        return self.model_.predict(X)


class CustomXGBMRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, params, features=None):
        self.params = params
        self.features = features

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    def fit(self, X, y=None):

        # Check that X and y have correct shape, set n_features_in_, etc.
        if self.features is None:
            X, y = validate_data(self, X, y)
            self.X_ = X
        else:
            X, y = validate_data(self, X[self.features], y)
            self.X_ = pd.DataFrame(X, columns=self.features)
        self.y_ = y

        X_train_, X_val_, y_train_, y_val_ = train_test_split(
            self.X_,
            self.y_,
            random_state=42,
            test_size=0.2,
            shuffle=True,
        )
        self.model_ = XGBRegressor(
            **self.params, early_stopping_rounds=50, eval_metrics="rmse"
        )
        self.model_.fit(X_train_, y_train_, eval_set=[(X_val_, y_val_)], verbose=0)
        return self.model_
        self.model_

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if self.features is None:
            X = validate_data(self, X, reset=False)
        else:
            X = pd.DataFrame(
                validate_data(self, X[self.features], reset=False),
                columns=self.features,
            )
        return self.model_.predict(X)


X_origin, y_origin = preprocessing(df_origin, mix_cat_cols=True, mix_num_cols=True)
X, y = preprocessing(df_train, mix_cat_cols=True, mix_num_cols=True)


def select_features(
    X: pd.DataFrame,
    y,
    est,
    params,
    cv=5,
    scoring="neg_root_mean_squared_error",
    fit_params=None,
):
    model_all_features = est(params, X.columns.to_list())

    output = cross_validate(
        model_all_features,
        X,
        y,
        cv=cv,
        scoring=scoring,
        params=fit_params,
        return_estimator=True,
    )

    fi = []
    for estimator in output["estimator"]:
        fi.append(estimator.feature_importances_)

    fi = pd.DataFrame(
        np.array(fi).T,
        columns=["importance " + str(idx) for idx in range(len(fi))],
        index=X.columns,
    )

    fi["mean_importance"] = fi.mean(axis=1)

    features = fi["mean_importance"]
    features = features.sort_values(ascending=True)

    features.plot.barh(figsize=(10, 20))

    score_all = np.abs(output["test_score"].mean())
    print("Mean Score: ", score_all)

    del output

    tol = 0.000001
    features_to_remove = []
    score_mean_list = []
    diff_auc_list = []
    count = 1

    features = list(features.index)

    for feature in features:
        print("=" * 50)
        print("Check feature: ", feature, " feature ", count, " of ", len(features))
        count = count + 1
        X_clip = X.drop(features_to_remove + [feature], axis=1)
        model = est(params, X_clip.columns.to_list())
        scores = cross_val_score(
            model,
            X_clip,
            y,
            scoring=scoring,
            params=fit_params,
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
        del scores, X_clip
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


# select_features(
#     X_origin,
#     y_origin,
#     LGBMRegressor,
#     {
#         "device": "gpu",
#         "random_state": 42,
#         "importance_type": "gain",
#         "verbosity": -1,
#     },
# )


# select_features(
#     X_origin,
#     y_origin,
#     XGBRegressor,
#     {
#         "random_state": 42,
#         "device": "cuda",
#         "importance_type": "gain",
#         "verbosity": 0,
#     },
# )


# select_features(
#     X,
#     y,
#     LGBMRegressor,
#     {
#         "device": "gpu",
#         "random_state": 42,
#         "importance_type": "gain",
#         "verbosity": -1,
#     },
# )


# select_features(
#     X,
#     y,
#     XGBRegressor,
#     {
#         "random_state": 42,
#         "verbosity": 0,
#         "device": "cuda",
#         "importance_type": "gain",
#     },
# )


SELECTED_LGBM_F = [
    "Body_Composition",
    "Duration_Long",
    "Moderate_intensity",
    "Overweight",
    "Vigourous_intensity",
    "Moderate_intensity_vs_Vigourous_intensity",
    "Underweight",
    "Obese",
    "Body_Composition_vs_Moderate_intensity",
    "Body_Composition_vs_Sex",
    "Age_Group",
    "Sex",
    "Temp_Deviation",
    "BMI",
    "Age_Group_vs_Moderate_intensity",
    "Body_Temp_DIV_BMI",
    "Age",
    "Body_Temp_MUL_BMI",
    "Moderate_intensity_vs_Sex",
    "Max_Heart_Rate",
    "Age_DIV_Heart_Rate",
    "Duration_MUL_BMI",
    "Age_Group_vs_Duration_Long",
    "Duration_DIV_BMI",
    "Age_MUL_Body_Temp",
    "Age_DIV_Body_Temp",
    "Sex_vs_Duration_Long",
    "Heart_Rate_DIV_BMI",
    "Body_Temp",
    "Body_Temp_DIV_Duration",
    "Height",
    "Age_DIV_Duration",
    "Body_Temp_MUL_Sex",
    "Duration",
    "Body_Temp_MUL_Duration",
    "Duration_DIV_Heart_Rate",
    "Heart_Rate_MUL_BMI",
    "Duration_MUL_Sex",
    "Body_Temp_DIV_Heart_Rate",
    "Heart_Rate",
    "Body_Temp_MUL_Sex_Reversed",
    "Age_DIV_BMI",
    "Heart_Rate_MUL_Sex",
    "Sex_vs_Vigourous_intensity",
    "Age_MUL_Heart_Rate",
    "Weight",
    "Duration_MUL_Sex_Reversed",
    "Age_MUL_Duration",
    "Body_Temp_MUL_Heart_Rate",
    "%_Max_Heart_Rate",
    "Duration_MUL_Heart_Rate",
]

SELECTED_LGBM_ORIGIN_F = [
    "Obese",
    "Overweight",
    "Moderate_intensity_vs_Vigourous_intensity",
    "Body_Composition_vs_Vigourous_intensity",
    "Body_Composition",
    "Moderate_intensity",
    "Age_Group_vs_Vigourous_intensity",
    "Body_Composition_vs_Moderate_intensity",
    "Age_Group",
    "Body_Composition_vs_Sex",
    "Age_Group_vs_Body_Composition",
    "Age_DIV_Heart_Rate",
    "Age_Group_vs_Moderate_intensity",
    "Moderate_intensity_vs_Duration_Long",
    "Body_Temp_DIV_BMI",
    "Age_MUL_BMI",
    "Weight",
    "Sex",
    "Age_MUL_Body_Temp",
    "Age",
    "Body_Temp_MUL_Sex",
    "Age_DIV_Body_Temp",
    "Height",
    "Age_Group_vs_Sex",
    "Heart_Rate_MUL_Sex",
    "Duration_MUL_Sex_Reversed",
    "Sex_vs_Vigourous_intensity",
    "Body_Composition_vs_Duration_Long",
    "Age_MUL_Heart_Rate",
    "Body_Temp_MUL_Sex_Reversed",
    "Moderate_intensity_vs_Sex",
    "Sex_vs_Duration_Long",
    "Max_Heart_Rate",
    "Heart_Rate_MUL_Sex_Reversed",
    "Age_DIV_Duration",
    "Duration_MUL_Sex",
    "Age_Group_vs_Duration_Long",
    "Temp_Deviation",
    "%_Max_Heart_Rate",
    "Heart_Rate_MUL_BMI",
    "Body_Temp_DIV_Heart_Rate",
    "Heart_Rate",
    "Body_Temp",
    "Duration_DIV_Heart_Rate",
    "Age_MUL_Duration",
    "Body_Temp_MUL_Duration",
    "Body_Temp_MUL_Heart_Rate",
    "Duration_MUL_BMI",
    "Duration_Long_vs_Vigourous_intensity",
    "Duration_Long",
    "Duration_MUL_Heart_Rate",
    "Duration_DIV_BMI",
    "Duration",
    "Body_Temp_DIV_Duration",
]

SELECTED_XGBM_F = [
    "Age_Group",
    "Vigourous_intensity",
    "Duration_Long",
    "Overweight",
    "Body_Composition",
    "Max_Heart_Rate",
    "Moderate_intensity",
    "Obese",
    "Moderate_intensity_vs_Vigourous_intensity",
    "Temp_Deviation",
    "Underweight",
    "Body_Composition_vs_Vigourous_intensity",
    "Age",
    "Body_Composition_vs_Duration_Long",
    "Body_Temp_MUL_BMI",
    "Moderate_intensity_vs_Duration_Long",
    "Age_DIV_Heart_Rate",
    "Age_Group_vs_Vigourous_intensity",
    "Body_Temp_DIV_Duration",
    "Body_Composition_vs_Sex",
    "Age_DIV_Body_Temp",
    "Heart_Rate_DIV_BMI",
    "Age_Group_vs_Body_Composition",
    "Age_Group_vs_Moderate_intensity",
    "Body_Temp_MUL_Duration",
    "Duration_DIV_BMI",
    "Duration_MUL_BMI",
    "Height",
    "Age_DIV_Duration",
    "Age_Group_vs_Duration_Long",
    "Age_MUL_Body_Temp",
    "Age_MUL_BMI",
    "Duration_DIV_Heart_Rate",
    "Sex",
    "Body_Temp_MUL_Sex",
    "Duration_Long_vs_Vigourous_intensity",
    "Heart_Rate",
    "Body_Temp",
    "Body_Temp_DIV_Heart_Rate",
    "Sex_vs_Duration_Long",
    "Duration",
    "Moderate_intensity_vs_Sex",
    "Heart_Rate_MUL_Sex",
    "Weight",
    "Duration_MUL_Sex",
    "Age_MUL_Duration",
    "Body_Temp_MUL_Sex_Reversed",
    "Age_Group_vs_Sex",
    "Sex_vs_Vigourous_intensity",
    "Body_Temp_MUL_Heart_Rate",
    "Duration_MUL_Sex_Reversed",
    "Heart_Rate_MUL_Sex_Reversed",
    "%_Max_Heart_Rate",
    "Duration_MUL_Heart_Rate",
]

SELECTED_XGBM_ORIGIN_F = [
    "Age_Group",
    "Vigourous_intensity",
    "Duration_Long",
    "Overweight",
    "Body_Composition",
    "Max_Heart_Rate",
    "Moderate_intensity",
    "Obese",
    "Moderate_intensity_vs_Vigourous_intensity",
    "Temp_Deviation",
    "Underweight",
    "Body_Composition_vs_Vigourous_intensity",
    "BMI",
    "Body_Composition_vs_Duration_Long",
    "Body_Temp_MUL_Duration",
    "Age_Group_vs_Duration_Long",
    "Duration_MUL_BMI",
    "Height",
    "Age_MUL_Body_Temp",
    "Body_Temp_DIV_Duration",
    "Age_DIV_Duration",
    "Age_Group_vs_Vigourous_intensity",
    "Duration_DIV_Heart_Rate",
    "Age_Group_vs_Moderate_intensity",
    "Age_Group_vs_Body_Composition",
    "Duration_DIV_BMI",
    "Duration",
    "Body_Temp_DIV_Heart_Rate",
    "Heart_Rate",
    "Moderate_intensity_vs_Sex",
    "Age_MUL_Heart_Rate",
    "Body_Composition_vs_Sex",
    "Duration_MUL_Sex",
    "Weight",
    "Body_Temp_MUL_Sex",
    "Heart_Rate_MUL_Sex",
    "Sex",
    "Age_MUL_Duration",
    "Age_Group_vs_Sex",
    "Sex_vs_Vigourous_intensity",
    "Duration_MUL_Sex_Reversed",
    "Heart_Rate_MUL_Sex_Reversed",
    "%_Max_Heart_Rate",
    "Duration_MUL_Heart_Rate",
]


def optimize_lightgbm(X, y, features):

    def objective(trial: optuna.Trial):
        """Define the objective function for LGB Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 0, 300),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 25.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 3.0, log=True),
            "random_state": 42,
            "device": "gpu",
            "verbosity": -1,
        }

        scores = []
        for i, (tr, val) in enumerate(kf.split(X)):
            X_tr, X_ts, y_tr, y_ts = (
                X.iloc[tr, :],
                X.iloc[val, :],
                y.iloc[tr],
                y.iloc[val],
            )
            model = CustomLGBMRegressor(
                {
                    **params,
                    "n_estimators": 9217,
                    "max_depth": 8,
                    "device": "gpu",
                    "random_state": 42,
                    "verbosity": -1,
                },
                features,
            )
            model.fit(X_tr, y_tr)
            rmse = root_mean_squared_error(y_ts, model.predict(X_ts))
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
    print("Best Params: ", best_params)
    return best_params


def optimize_xgbm(X, y, features):

    def objective(trial: optuna.Trial):
        """Define the objective function for XGB Model"""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0001, 0.5, log=True),
        }

        scores = []
        for i, (tr, val) in enumerate(kf.split(X)):
            X_tr, X_ts, y_tr, y_ts = (
                X.iloc[tr, :],
                X.iloc[val, :],
                y.iloc[tr],
                y.iloc[val],
            )
            model = CustomXGBMRegressor(
                {**params, "device": "cuda", "random_state": 42, "verbosity": 0},
                features,
            )
            model.fit(X_tr, y_tr)
            rmse = root_mean_squared_error(y_ts, model.predict(X_ts))
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
    print("Best Params: ", best_params)
    return best_params


# optimize_lightgbm(X_origin, y_origin, SELECTED_LGBM_ORIGIN_F)


# optimize_xgbm(X_origin, y_origin, SELECTED_XGBM_ORIGIN_F)


# optimize_lightgbm(X, y, SELECTED_LGBM_F)


# optimize_xgbm(X, y, SELECTED_XGBM_F)


lgbm_params = {
    "n_estimators": 9217,
    "max_depth": 8,
    "learning_rate": 0.012394273773537386,
    "num_leaves": 101,
    "min_data_in_leaf": 4,
    "subsample": 0.11335340936824273,
    "feature_fraction": 0.15376205873898768,
    "reg_lambda": 14.022827916839951,
    "reg_alpha": 1.2330813554539497e-09,
    "random_state": 42,
    "device": "gpu",
    "verbosity": -1,
}

lgbm_origin_params = {
    "n_estimators": 9217,
    "max_depth": 8,
    "learning_rate": 0.01698251430464904,
    "num_leaves": 8,
    "min_data_in_leaf": 23,
    "subsample": 0.3767548960897378,
    "feature_fraction": 0.18099939518519076,
    "reg_lambda": 0.0027705001484032788,
    "reg_alpha": 1.8966493214410135e-09,
    "random_state": 42,
    "device": "gpu",
    "verbosity": -1,
}

xgbm_params = {
    "n_estimators": 2017,
    "learning_rate": 0.01283222236370823,
    "min_child_weight": 2,
    "max_depth": 8,
    "subsample": 0.845812195258525,
    "colsample_bytree": 0.41344504982979446,
    "reg_lambda": 2.7695524056081875,
    "reg_alpha": 0.29041169508192066,
    "gamma": 0.0021293152218445025,
    "device": "cuda",
    "random_state": 42,
    "verbosity": 0,
}

xgbm_origin_params = {
    "n_estimators": 4974,
    "learning_rate": 0.012245974781740126,
    "min_child_weight": 2,
    "max_depth": 5,
    "subsample": 0.41398120857015897,
    "colsample_bytree": 0.4466788617929194,
    "reg_lambda": 1.242269314571232,
    "reg_alpha": 0.0007049912835321339,
    "gamma": 0.00011610980377983347,
    "device": "cuda",
    "random_state": 42,
    "verbosity": 0,
}


def train_test(
    X,
    y,
    features_1=X.columns.to_list(),
    features_2=X.columns.to_list(),
    model_1_params=lgbm_params,
    model_2_params=xgbm_params,
):
    avr_sets_scores = []
    lgbm_sets_scores = []
    xgbm_sets_scores = []

    for set in range(3):
        print(" " * 50, f"Set: {set}")

        avr_scores = []
        lgbm_scores = []
        xgbm_scores = []
        for i, (tr, val) in enumerate(kf.split(X)):
            X_tr, X_ts, y_tr, y_ts = (
                X.iloc[tr, :],
                X.iloc[val, :],
                y.iloc[tr],
                y.iloc[val],
            )
            model_1 = CustomLGBMRegressor(model_1_params, features_1)
            model_2 = CustomXGBMRegressor(model_2_params, features_2)
            model_1.fit(X_tr, y_tr)
            model_2.fit(X_tr, y_tr)
            lgbm_pred = model_1.predict(X_ts)
            xgbm_pred = model_2.predict(X_ts)
            avr_rmse = root_mean_squared_error(
                y_ts,
                np.mean(
                    np.column_stack(
                        [
                            lgbm_pred,
                            xgbm_pred,
                        ]
                    ),
                    axis=1,
                ),
            )
            lgbm_rmse = root_mean_squared_error(y_ts, lgbm_pred)
            xgbm_rmse = root_mean_squared_error(y_ts, xgbm_pred)

            avr_scores.append(avr_rmse)
            lgbm_scores.append(lgbm_rmse)
            xgbm_scores.append(xgbm_rmse)
            print(f"Fold {i}:", avr_rmse)
            print("=" * 100)
        avr_set_score = np.mean(avr_scores)
        lgbm_set_scores = np.mean(lgbm_scores)
        xgbm_set_score = np.mean(xgbm_scores)
        print("\n")
        print(f"Avr Set {set} Score:", avr_set_score)
        print(f"Light Set {set} Score:", lgbm_set_scores)
        print(f"XGB Set {set} Score:", xgbm_set_score)
        print("*" * 100)

        avr_sets_scores.append(avr_set_score)
        lgbm_sets_scores.append(lgbm_set_scores)
        xgbm_sets_scores.append(xgbm_set_score)

    cv = abs(np.mean(avr_sets_scores))
    print(f"Total Avr Score: {cv}")
    print(f"Total Light Score: {abs(np.mean(lgbm_sets_scores))}")
    print(f"Total XGB Score: {abs(np.mean(xgbm_sets_scores))}")
    print("\n")


# train_test(
#     X_origin,
#     y_origin,
#     SELECTED_LGBM_ORIGIN_F,
#     SELECTED_XGBM_ORIGIN_F,
#     lgbm_origin_params,
#     xgbm_origin_params,
# )

# Total Avr Score: 0.017158258666012855
# Total Light Score: 0.018245558972155972
# Total XGB Score: 0.018450311231374784


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=42, test_size=0.3, shuffle=True
# )

# X_train_origin, X_test_origin, y_train_origin, y_test_origin = train_test_split(
#     X_origin, y_origin, random_state=42, test_size=0.3, shuffle=True
# )


# model_1 = CustomLGBMRegressor(lgbm_params, SELECTED_LGBM_F)
# model_2 = CustomXGBMRegressor(xgbm_params, SELECTED_XGBM_F)

# model_3 = CustomLGBMRegressor(lgbm_origin_params, SELECTED_LGBM_ORIGIN_F)
# model_4 = CustomXGBMRegressor(xgbm_origin_params, SELECTED_XGBM_ORIGIN_F)

# model_1.fit(X_train, y_train)
# model_2.fit(X_train, y_train)

# model_3.fit(X_train_origin, y_train_origin)
# model_4.fit(X_train_origin, y_train_origin)


def predict(
    est_1,
    est_2,
    target,
):

    lgbm_pred = est_1.predict(target)
    xgbm_pred = est_2.predict(target)

    return np.column_stack([lgbm_pred, xgbm_pred])


# meta_train_features = predict(
#     model_1,
#     model_2,
#     target=X_train,
# )
# meta_train_origin_features = predict(
#     model_3,
#     model_4,
#     target=X_train_origin,
# )


# def optimize_meta_model(X, y):
#     def objective(trial: optuna.Trial):
#         """Define the objective function for XGB Model"""

#         params = {
#             "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
#             "random_state": 42,
#         }

#         scores = []
#         for i, (tr, ts) in enumerate(kf.split(X)):
#             X_tr, X_ts, y_tr, y_ts = (
#                 X[tr, :],
#                 X[ts, :],
#                 y.iloc[tr],
#                 y.iloc[ts],
#             )
#             model = Lasso(**params)
#             model.fit(X_tr, y_tr)
#             rmse = root_mean_squared_error(y_ts, model.predict(X_ts))
#             scores.append(rmse)

#             print(f"Fold {i}:", rmse)

#         cv = abs(np.mean(scores))

#         print("=" * 50)
#         print("\n")

#         return cv

#     # Create an Optuna study
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=50)

#     # Get the best parameters
#     best_params = study.best_params
#     return best_params


# optimize_meta_model(meta_train_features, y_train)


# meta_test_features = predict(
#     model_1,
#     model_2,
#     target=X_test,
# )
# meta_test_origin_features = predict(
#     model_3,
#     model_4,
#     target=X_test_origin,
# )


# best_meta_model = LassoCV(alphas=[0.00010131976133563833], random_state=42, cv=5)

# print(
#     abs(
#         np.mean(
#             cross_val_score(
#                 best_meta_model,
#                 meta_test_features,
#                 y_test,
#                 scoring="neg_root_mean_squared_error",
#             )
#         )
#     )
# )

# RMSE on Test Data: 0.05144253237861007


# best_meta_model = RidgeCV()

# print(
#     abs(
#         np.mean(
#             cross_val_score(
#                 best_meta_model,
#                 meta_test_origin_features,
#                 y_test_origin,
#                 scoring="neg_root_mean_squared_error",
#             )
#         )
#     )
# )

# 0.01711086107400208


# del X_train, X_test, y_train, y_test
# del X_train_origin, X_test_origin, y_train_origin, y_test_origin


from sklearn.ensemble import StackingRegressor

estimators = [
    ("LGBM", CustomLGBMRegressor(lgbm_params, SELECTED_LGBM_F)),
    ("XGBM", CustomXGBMRegressor(xgbm_params, SELECTED_XGBM_F)),
]

estimators_origin = [
    ("LGBM", CustomLGBMRegressor(lgbm_origin_params, SELECTED_LGBM_ORIGIN_F)),
    ("XGBM", CustomXGBMRegressor(xgbm_origin_params, SELECTED_XGBM_ORIGIN_F)),
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LassoCV(alphas=[0.00010131976133563833], random_state=42, cv=5),
)

stack_model_origin = StackingRegressor(estimators=estimators_origin)

lgbm = CustomLGBMRegressor(lgbm_params, SELECTED_LGBM_F)
xgbm = CustomXGBMRegressor(xgbm_params, SELECTED_XGBM_F)
lgbm_origin = CustomLGBMRegressor(lgbm_origin_params, SELECTED_LGBM_ORIGIN_F)
xgbm_origin = CustomXGBMRegressor(xgbm_origin_params, SELECTED_XGBM_ORIGIN_F)


# stack_model_test = StackingRegressor(
#     estimators=estimators,
#     final_estimator=Lasso(alpha=0.00010131976133563833, random_state=42),
#     cv=2
# )

# print(
#     abs(
#         np.mean(
#             cross_val_score(
#                 stack_model_test,
#                 X,
#                 y,
#                 scoring="neg_root_mean_squared_error",
#             )
#         )
#     )
# )


# Total Avr Score: 0.056263810181117445


# stack_model_origin_test = StackingRegressor(
#     estimators=estimators_origin, final_estimator=Ridge(random_state=42), cv=2
# )

# print(
#     abs(
#         np.mean(
#             cross_val_score(
#                 stack_model_origin_test,
#                 X_origin,
#                 y_origin,
#                 scoring="neg_root_mean_squared_error",
#             )
#         )
#     )
# )

# 0.0173340577853824


stack_model.fit(X, y)


stack_model_origin.fit(X_origin, y_origin)


lgbm.fit(X, y)
xgbm.fit(X, y)
lgbm_origin.fit(X_origin, y_origin)
xgbm_origin.fit(X_origin, y_origin)


meta_train_features = pd.DataFrame(
    {
        "Ensemble1": stack_model.predict(X),
        "Ensemble2": stack_model_origin.predict(X),
        "LGBM": lgbm.predict(X),
        "XGBM": xgbm.predict(X),
        "LGBM_ORIGIN": lgbm_origin.predict(X),
        "XGBM_ORIGIN": xgbm_origin.predict(X),
    }
)


np.random.seed(42)


def generate_normalized_array(rows, cols):
    random_array = np.random.random_sample((rows, cols))
    row_sums = random_array.sum(axis=1, keepdims=True)
    normalized_array = random_array / row_sums
    return normalized_array


best_score = 1
best_weights = np.zeros(6)

for weights in generate_normalized_array(1000, 6):
    pred = np.sum((meta_train_features.values * weights), axis=1)
    score = root_mean_squared_error(y, pred)
    if best_score > score:
        best_score = score
        best_weights = weights
print("Best Score: ", best_score)
print("Best Weights: ", best_weights)


X_test, _ = preprocessing(
    df_test, drop_target=False, mix_cat_cols=True, mix_num_cols=True
)

df_pred = X_test.id.to_frame()


meta_test_features = pd.DataFrame(
    {
        "Ensemble1": stack_model.predict(X_test),
        "Ensemble2": stack_model_origin.predict(X_test),
        "LGBM": lgbm.predict(X_test),
        "XGBM": xgbm.predict(X_test),
        "LGBM_ORIGIN": lgbm_origin.predict(X_test),
        "XGBM_ORIGIN": xgbm_origin.predict(X_test),
    }
)


df_pred[TARGET] = np.sum((meta_test_features.values * best_weights), axis=1)


submission = pd.DataFrame({"id": df_pred.id, TARGET: np.expm1(df_pred[TARGET])})
submission.to_csv("submission.csv", index=False)

