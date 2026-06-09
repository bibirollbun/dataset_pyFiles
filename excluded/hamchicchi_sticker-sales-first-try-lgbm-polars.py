# import modules
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold

# import optuna.integration.lightgbm as lgb
import lightgbm as lgb
# import shap

sns.set_theme(context="notebook", style="whitegrid", palette="Set2")
warnings.simplefilter(action="ignore", category=FutureWarning)

from datetime import datetime, timedelta
import time


# Stop watch
start_time = time.time()

# data reading
df = pl.read_csv("/kaggle/input/playground-series-s5e1/train.csv").drop("id")

df_obj = pl.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_obj_id = df_obj.select("id")
df_obj = df_obj.drop("id")


# Data preprocessing class
class CustomTramsformer:
    def __init__(self, target):
        self.target = target
        self.ordered_elements = {}
        self.not_need_other = {}
        self.feature_threshold = {}
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first")
        self.ct = None
        self.categories = None
        self.ct_feature_names = None
        self.num_features = None
        self.cat_features = None

    def reset_features(self, df: pl.DataFrame):
        self.num_features = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32]
        ]
        self.cat_features = [col for col in df.columns if df[col].dtype in [pl.String]]

    def set_threshold(
        self, df1: pl.DataFrame, df2: pl.DataFrame, feature: str, val: int
    ):
        self.feature_threshold[feature] = val
        df1 = self.set_categories_of_feature(df1, feature, val)
        df2 = self.set_categories_of_feature(df2, feature)

        if ("other" not in df1[feature].unique()) & (
            "other" not in df2[feature].unique()
        ):
            self.not_need_other[feature] = True
        else:
            self.not_need_other[feature] = False

        if self.not_need_other[feature]:
            self.ordered_elements[feature] = [
                x for x in self.ordered_elements[feature] if x != "other"
            ]
        return df1, df2

    def set_categories_of_feature(
        self, df: pl.DataFrame, feature: str, n_items: int = 0
    ):
        if n_items != 0:
            counts = df[feature].value_counts().sort(by="count", descending=True)
            self.ordered_elements[feature] = [
                col for col, ct in counts.to_numpy() if ct >= n_items
            ]

        df = df.with_columns(
            pl.when(pl.col(feature).is_in(self.ordered_elements[feature]))
            .then(pl.col(feature))
            .when(pl.col(feature).is_null())
            .then(pl.col(feature))
            .otherwise(pl.lit("other"))
            .alias(feature)
        )

        if feature not in self.not_need_other.keys():
            self.not_need_other[feature] = False

        if (
            not self.not_need_other[feature]
            and "other" not in self.ordered_elements[feature]
        ):
            self.ordered_elements[feature].append("other")
        return df

    def set_ct(self):
        num_wo_target = [val for val in self.num_features if val != self.target]
        self.categories = [self.ordered_elements[col] for col in self.cat_features]
        self.encoder = OneHotEncoder(drop="first", categories=self.categories)

        self.ct = ColumnTransformer(
            [
                ("scaler", self.scaler, num_wo_target),
                ("encoder", self.encoder, self.cat_features),
            ],
            remainder="passthrough",
        )

    def ct_transform(self, df: pl.DataFrame):
        if self.target in df.columns:
            df = df.drop(self.target)

        data_scaled = self.ct.fit_transform(
            df.to_pandas(use_pyarrow_extension_array=True)
        )
        if self.categories is None:
            self.categories = self.ct.named_transformers_["encoder"].categories_
        if self.ct_feature_names is None:
            self.ct_feature_names = self.ct.get_feature_names_out().tolist()
        return data_scaled

    def pipeline(self, df: pl.DataFrame, set_categories: bool = False):
        cols = self.cat_features
        if set_categories:
            n_threshes = []
            for col in cols:
                if col in self.feature_threshold.keys():
                    val = self.feature_threshold[col]
                else:
                    val = 1
                n_threshes.append(val)
        else:
            n_threshes = [0 for _ in range(len(cols))]

        for col, thresh in zip(cols, n_threshes):
            df = self.set_categories_of_feature(df, col, thresh)

        if set_categories:
            self.set_ct()
        return df


# function to plot cat features
def plot_cat(df1: pl.DataFrame, df2: pl.DataFrame, feature: str, order=None):

    if order is None:
        order = ut.ordered_elements[feature]

    if feature in ["month", "day", "weekday"]:
        order_num = sorted([int(val) for val in order])
        order = [str(x) for x in order_num]

    palette = dict(zip(order, sns.color_palette("colorblind", 32)[: len(order)]))
    _, ax = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    plt.subplots_adjust(right=0.85)
    sns.violinplot(
        df1.to_pandas(), x=feature, y=ut.target, ax=ax[0], order=order, palette=palette,
    )
    sns.countplot(df1.to_pandas(), x=feature, ax=ax[1], order=order, palette=palette)
    sns.countplot(df2.to_pandas(), x=feature, ax=ax[2], order=order, palette=palette)

    for i, ylabel in enumerate(["num_sold", "train cnt.", "test cnt."]):
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel("")
        # ax[i].set_xticks(order)
        ax[i].set_xticklabels(order, fontsize=11, rotation=90)
    plt.suptitle(feature)
    plt.show()

    if order is None:
        order = ut.ordered_elements[feature]


def nullcheck(df: pl.DataFrame, title: str):
    has_null = False
    df_nullcheck = pl.DataFrame()
    for col in df.columns:
        nc = df[col].null_count()
        if nc >= 1:
            df_nullcheck = pl.concat(
                [
                    df_nullcheck,
                    pl.DataFrame(
                        {"feature": col, "nulls": nc, "nulls_ratio": nc / len(df)}
                    ),
                ]
            )
            has_null = True

    if has_null:
        plt.figure(figsize=(8, 3))
        ax = sns.barplot(df_nullcheck.to_pandas(), y="feature", x="nulls")
        for container in ax.containers:
            ax.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=10)
        plt.title(f"Nulls check ({title})")
        plt.show()
    else:
        print(f"dateframe {title} has no null.")


nullcheck(df, "train")
nullcheck(df_obj, "test")


# drop nulls
df = df.drop_nulls()


ut = CustomTramsformer("num_sold")
ut.reset_features(df)
df = ut.pipeline(df, True)


# Extract year, month, day, weekday from Date
def handle_date(df: pl.DataFrame):

    date = pl.col("date")
    df_tmp = df.with_columns(date.str.to_date("%Y-%m-%d").alias("date")).with_columns(
        year=date.dt.year().cast(pl.Int32),
        month=date.dt.month().cast(pl.String),
        day=date.dt.day().cast(pl.String),
        weekday=date.dt.weekday().cast(pl.String),
    )

    conditions = [
        (pl.col("date").dt.month() == month) & (pl.col("date").dt.day() == day)
        for month, day in [
            (1, 1),
            (4, 17),
            (4, 18),
            (4, 20),
            (4, 21),
            (12, 24),
            (12, 25),
            (12, 26),
        ]
    ]

    # merge conditions
    is_holiday = conditions[0]
    for condition in conditions[1:]:
        is_holiday = is_holiday | condition

    # Create holiday flag
    df_tmp.with_columns(pl.when(is_holiday).then(1).otherwise(0).alias("is_holiday"))

    return df_tmp.drop("date")


df = handle_date(df)
df_obj = handle_date(df_obj)
ut.reset_features(df)


# trend by  each country

def plot_year(df: pl.DataFrame, hue: str):
    _, ax = plt.subplots(1, 1, figsize=(10, 3))
    sns.lineplot(
        data=df, x="year", y="num_sold", hue=hue, ax=ax,
    )
    plt.show()

df_trend = df.select(["year", "country", "num_sold"]).group_by("year", "country").mean()
plot_year(df, "country")

df_trend = df.select(["year", "product", "num_sold"]).group_by("year", "product").mean()
plot_year(df, "product")


# drop Norway data between 2010 and 2014
df = df.filter((pl.col("country") != "Norway") | (pl.col("year").is_in([2015, 2016])))

df = df.drop("year")
df_obj = df_obj.drop("year")
ut.reset_features(df)


# violinplot and countplot of train & test data
for feature in ut.cat_features:
    print(feature)
    ut.set_threshold(df, df_obj, feature, 1)
    plot_cat(df, df_obj, feature)


# Re: Null check
nullcheck(df, "train")
nullcheck(df_obj, "test")


ut.pipeline(df, True)
X = ut.ct_transform(df)
y = df.select(ut.target).to_numpy()


# train
from sklearn.metrics import mean_absolute_percentage_error


# definision of training function

def train_model(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(X.shape[0])
    models = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        params_lgb = {
            "objective": "regression",
            "metric": "mape",
            "random_state": 42,
            "verbose": -1,
            "learning_rate": 0.0020872203318988195,
            "max_depth": 9,
            "feature_pre_filter": False,
            "lambda_l1": 4.061383608715305e-05,
            "lambda_l2": 0.0013203576298249095,
            "num_leaves": 115,
            "feature_fraction": 0.948,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "min_child_samples": 20,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params_lgb,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=True),
                lgb.log_evaluation(250),
            ],
        )

        models.append(model)

        oof[valid_idx] = np.maximum(0, model.predict(X_valid))
        fold_mape = mean_absolute_percentage_error(y_valid, oof[valid_idx])
        print(f"Fold {fold + 1} MAPE: {fold_mape}")

    return models, oof


# train models
models, pred_valid = train_model(X, y)


# valid data result
print(mean_absolute_percentage_error(y, pred_valid))

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y.flatten(), y=pred_valid, alpha=0.01)
sns.lineplot(x=[0, y.max()], y=[0, y.max()], color="gray", linestyle="--", linewidth=1)
plt.xlabel("Premium Amount (true)")
plt.ylabel("predicted values")
plt.title("True vs Pred of train data")
plt.show()


X_obj = ut.ct_transform(df_obj)

obj_predictions = np.zeros(X_obj.shape[0])
for model in models:
    obj_predictions += np.maximum(0, model.predict(X_obj)) / len(models)


result = pl.concat(
    [df_obj_id, pl.DataFrame(obj_predictions, schema=["num_sold"])],
    how="horizontal",
)
result.write_csv("submission.csv")


end_time = time.time()
elapsed = end_time - start_time
print(f"elapsed time : {elapsed//3600}h {elapsed//60}min {elapsed%60:.1f}sec")

