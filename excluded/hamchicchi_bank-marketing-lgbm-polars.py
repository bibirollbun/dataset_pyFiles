# import modules

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context="notebook", style="ticks", font="Yu Gothic", font_scale=0.9, palette="Set2",
)


# read data
df = pl.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop("id")

df_obj = pl.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_obj_id = df_obj.select("id")
df_obj = df_obj.drop("id")


# data sample

display(df.to_pandas().info())
display(df_obj.to_pandas().info())


from sklearn.preprocessing import OneHotEncoder

class Handler:
    def __init__(self, df: pl.DataFrame, target: str):
        self.df = df
        self.target = target
        self.num_features, self.cat_features = [], []
        self.set_features()

        self.encoder = OneHotEncoder(sparse_output=False, drop="first")
        self.is_encoded = False

    def set_features(self):
        """return lists of num_features and cat_features"""

        num_f = [
            col for col in self.df.columns
            if (self.df[col].dtype in [pl.Int32, pl.Int64, pl.Float64])
            and (col != self.target)
        ]

        cat_f = [col for col in self.df.columns if self.df[col].dtype == pl.String]

        self.num_features = num_f
        self.cat_features = cat_f

    def target_encode(self):
        """convert cat to num for corr plot"""
        df_num = self.df

        for col in self.cat_features:
            df_num = df_num.with_columns(pl.col("y").mean().over(col).alias(col))
        return df_num

    def reset(self, df):

        self.df = df
        self.set_features()

    def onehot(self, df: pl.DataFrame):
        # Onehot encoding
        if not self.is_encoded:
            df_enc = self.encoder.fit_transform(
                df.select(self.cat_features).to_pandas()
            )
            self.is_encoded = True
            self.reset(df)

        else:
            df_enc = self.encoder.transform(df.select(self.cat_features).to_pandas())

        df = pl.concat([
                df.select(self.num_features),
                pl.DataFrame(
                    df_enc, schema=self.encoder.get_feature_names_out().tolist()
            )],
            how="horizontal",
        )

        return df


handler = Handler(df, "y")


# plot features
# num features
for i, col in enumerate(handler.num_features):
    _, ax = plt.subplots(1, 2, figsize=(9, 3), tight_layout=True, sharey=True)
    sns.violinplot(data=df.to_pandas(), y=col, x=handler.target, hue=handler.target, ax=ax[0])
    sns.violinplot(data=df_obj.to_pandas(), y=col, ax=ax[1])
    ax[0].set_title("train")
    ax[1].set_title("test")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="x-small")
    plt.suptitle(col)

    plt.show()


# Checking categorical features
# whether the number of unique values is the same

n_uni, n_uni_obj, n_uni_all = {}, {}, {}

for col in handler.cat_features:
    n_uni[col] = df[col].n_unique()
    n_uni_obj[col] = df_obj[col].n_unique()
    n_uni_all[col] = pl.concat([df[col], df_obj[col]], how="vertical").n_unique()

df_uni = pl.concat(
    [pl.DataFrame(n_uni), pl.DataFrame(n_uni_obj), pl.DataFrame(n_uni_all)],
    how="vertical",
)

df_uni.transpose(
    include_header=True,
    column_names=["train_features", "test_features", "all_features"],
)


# cat features
for i, col in enumerate(handler.cat_features):

    _, ax = plt.subplots(1, 2, figsize=(9, 3), tight_layout=True, sharey=True)
    sns.countplot(data=df.to_pandas(), y=col, hue=handler.target, ax=ax[0])
    sns.countplot(data=df_obj.to_pandas(), y=col, ax=ax[1])

    ax[0].set_title("train")
    ax[0].legend(fontsize="x-small")

    ax[1].set_title("test")
    plt.suptitle(col)

    plt.show()


# Correlation
df_heatmap = handler.target_encode().to_pandas().corr()

plt.figure(figsize=(9, 9))
sns.heatmap(
    df_heatmap,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=1,
    square=True,
    fmt=".2f",
    annot_kws={"size": 9},
    linecolor="white",
    cbar_kws={"shrink": 0.8},
)
plt.show()


# convert to int with yes-no categories
def convert_to_int(df, features):
    for col in features:
        df = df.with_columns(
            pl.when(pl.col(col) == "yes").then(1).otherwise(0).alias(col)
        )
    return df


features = ["default", "housing", "loan"]
df = convert_to_int(df, features)
df_obj = convert_to_int(df_obj, features)

handler.reset(df)


# month day to seasons

def handle_month_day(df):
    df = (
        df.with_columns(
            pl.when(pl.col("day") <= 10)
            .then(0)
            .when(pl.col("day") <= 20)
            .then(1)
            .otherwise(2)
            .alias("days")
        )
        .with_columns(
            season=(pl.col("month") + pl.col("days").cast(pl.String)).alias("season")
        )
        .drop("days", "month", "day")
    )
    return df


df = handle_month_day(df)
df_obj = handle_month_day(df_obj)

handler.reset(df)


def log_transform(df, col: str, base_int: int = 0):
    df = df.with_columns((pl.col(col) - base_int + 2).log().alias(col))
    return df


features = ["balance", "duration", "pdays", "campaign", "previous"]

for col in features:
    min_value = min([df[col].min(), df_obj[col].min()])
    df = log_transform(df, col, min_value)
    df_obj = log_transform(df_obj, col, min_value)

handler.reset(df)



def create_new_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        durationandpdays=pl.col("duration") * pl.col("pdays"),
        durationandprevious=pl.col("duration") * pl.col("previous"),
        durationandbalance=pl.col("duration") * pl.col("balance"),
        durationandcampaign=pl.col("duration") * pl.col("campaign"),
        baranceandpdays=pl.col("balance") * pl.col("pdays"),
    )
    return df


df = create_new_features(df)
df_obj = create_new_features(df_obj)

handler.reset(df)


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusterClassifier:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=15, random_state=42)
        self.scaler = StandardScaler()
        self.features = ["age", "balance", "duration", "pdays", "previous", "campaign"]

    def clustering_features(self, df: pl.DataFrame, fit=False) -> pl.DataFrame:
        # clustering_features = num_features + cat_features
        df_fit = df.select(self.features)

        if fit:
            self.scaler.fit(df_fit.to_numpy())

        scaled = self.scaler.transform(df_fit.to_numpy())

        if fit:
            self.kmeans.fit(scaled)

        cluster_labels = self.kmeans.predict(scaled)
        clustered_df = df.with_columns(pl.lit(cluster_labels).alias("cluster"))
        return clustered_df


cluster = ClusterClassifier()
df = cluster.clustering_features(df, True)
df_obj = cluster.clustering_features(df_obj)

handler.reset(df)


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
import shap


X = handler.onehot(df).to_pandas()
y = df[handler.target].to_numpy()

X_test = handler.onehot(df_obj).to_pandas()


# train model


def train_lightgbm_cv(X, y, X_test, n_splits=5):

    # result
    cv_results = {
        "fold": [],
        "precision": [],
        "accuracy": [],
        "roc": [],
        "y_true": [],
        "y_pred": [],
        "y_pred_proba": [],
        "model": [],
    }

    valid_pred_array = np.zeros_like(y, dtype=np.float64)

    # test data result of each fold
    test_predictions = []

    for loop, random_state in enumerate([21, 42, 84]):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 71,
            "feature_fraction_bynode": 0.20,
            "min_child_samples": 85,
            "verbose": -1,
            "random_state": random_state,
        }

        print("=== Starting LightGBM Cross Validation ===")

        # Cross Validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n--- Loop {loop + 1} Fold {fold + 1}/{n_splits} ---")

            # Fold
            X_train_fold = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            # LightGBM Data Set
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

            # train
            model = lgb.train(
                params,
                train_data,
                num_boost_round=10000,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.log_evaluation(250),
                    lgb.early_stopping(100, verbose=False),
                ]
            )

            # Prediction Valid data
            y_pred_proba = model.predict(X_val_fold, num_iteration=model.best_iteration)
            valid_pred_array[val_idx] += y_pred_proba / 3
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Prediction test data
            test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            test_predictions.append(test_pred_proba)

            # evaluation
            precision = precision_score(y_val_fold, y_pred)
            accuracy = accuracy_score(y_val_fold, y_pred)
            roc_score = roc_auc_score(y_val_fold, y_pred_proba)

            print(f"Precision: {precision:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_score:.4f}")

            # save result of fold
            cv_results["fold"].append(fold + 1)
            cv_results["precision"].append(precision)
            cv_results["accuracy"].append(accuracy)
            cv_results["roc"].append(roc_score)
            cv_results["y_true"].extend(y_val_fold)
            cv_results["y_pred"].extend(y_pred)
            cv_results["y_pred_proba"].extend(y_pred_proba)
            cv_results["model"].append(model)

    print(f"\n=== å…¨ä½“çµ�æ�œ ===")
    print(
        f"Average Precision: {np.mean(cv_results['precision']):.4f} Â± {np.std(cv_results['precision']):.4f}"
    )
    print(
        f"Average Accuracy: {np.mean(cv_results['accuracy']):.4f} Â± {np.std(cv_results['accuracy']):.4f}"
    )
    print(
        f"Average ROC-auc: {np.mean(cv_results['roc']):.4f} Â± {np.std(cv_results['roc']):.4f}"
    )

    # final result
    test_predictions_array = np.array(test_predictions)
    final_test_predictions = np.mean(test_predictions_array, axis=0)

    print(f"\nPrediction Complete: {len(final_test_predictions)} samples")

    return cv_results, valid_pred_array, final_test_predictions


cv_results, valid_predictions, final_test_predictions = train_lightgbm_cv(
    X, y, X_test, n_splits=5
)
df = pl.concat([df, pl.DataFrame(valid_predictions, schema=["pred"])], how="horizontal")


# calculate shap value

arr_importance_list = []

for model in cv_results["model"]:
    importance_gain = model.feature_importance(importance_type="gain")
    arr_importance_list.append(importance_gain)

arr_importance = np.vstack(arr_importance_list)
df_importance = (
    pl.DataFrame(
        arr_importance.mean(axis=0).reshape(-1, 1), schema=X_test.columns.to_list()
    )
    .transpose(include_header=True, header_name="feature", column_names=["importance"])
    .sort("importance", descending=True)
)

plt.figure(figsize=(8, 4))
sns.barplot(df_importance.head(15), x="importance", y="feature")
plt.show()


def visualize_cv_results(cv_results):

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 3. visualization
    sns.histplot(
        data=cv_results, x="y_pred_proba", bins=50,
        hue="y_true", alpha=0.3, ax=axes[0],
    )
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="threshold: 0.5")
    axes[0].set_title("Evaluation of valid data", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("count")
    axes[0].set_ylim((0, 5e4))

    axes[0].grid(True, alpha=0.3)

    # 4. Confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(cv_results["y_true"], cv_results["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt=",d", annot_kws={"size": 20}, cmap="Blues", ax=axes[1]
    )
    axes[1].set_title("Confusion matrix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted value")
    axes[1].set_ylabel("True value")

    plt.tight_layout()
    plt.show()

    return

visualize_cv_results(cv_results)


test_preds = pl.DataFrame(final_test_predictions, schema=[handler.target])

submission = pl.concat([df_obj_id, test_preds], how="horizontal")
submission.write_csv("submission.csv")

