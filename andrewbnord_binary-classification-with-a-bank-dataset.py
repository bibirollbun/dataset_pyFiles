%pip install -qq seaborn xgboost lightgbm scikit-learn prince --upgrade 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prince import FAMD

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tqdm import tqdm

sns.set_theme(style="darkgrid")
pd.set_option("display.max_columns", 100)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv").drop("id", axis=1)

df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

def converse_y(y):
    return 1 if y == "yes" else 0

df_origin = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";", converters={'y': converse_y})



TARGET = "y"
num_cols = df_train.select_dtypes(exclude="object").columns.to_list()
cat_cols = df_train.select_dtypes(include="object").columns.to_list()

num_cols.remove(TARGET)

SEED = 12637


from sklearn.discriminant_analysis import StandardScaler


sc = StandardScaler()


def normalize(X: pd.DataFrame):
    return sc.fit_transform(X)


from sklearn.calibration import LabelEncoder


le = LabelEncoder()


def label_encoder(X: pd.DataFrame, cat_cols=cat_cols):
    for col in cat_cols:
        X[col] = le.fit_transform(X[col]).astype("int8")
    return X[cat_cols]


df_train.info(), df_test.info(), df_origin.info()


df_train[TARGET].value_counts(normalize=True), df_origin[TARGET].value_counts(normalize=True)


display(
    df_train[num_cols].describe(),
    df_test[num_cols].describe(),
    df_origin[num_cols].describe(),
)


df_train_norm = df_train.copy().drop(TARGET, axis=1)
df_origin_norm = df_origin.copy().drop(TARGET, axis=1)

df_train_norm[num_cols] = normalize(df_train_norm[num_cols])
df_origin_norm[num_cols] = normalize(df_origin_norm[num_cols])

famd_train = FAMD(n_components=2, random_state=SEED)
famd_origin = FAMD(n_components=2, random_state=SEED)

df_train_norm[["FAMD_0", "FAMD_1"]] = famd_train.fit_transform(df_train_norm)

df_origin_norm[["FAMD_0", "FAMD_1"]] = famd_origin.fit_transform(df_origin_norm)

df_train_norm[TARGET] = df_train[TARGET]
df_origin_norm[TARGET] = df_origin[TARGET]

fig, axes = plt.subplots(1, 2, figsize=(20, 20))

sns.scatterplot(
    df_train_norm,
    x="FAMD_0",
    y="FAMD_1",
    hue=TARGET,
    ax=axes[0],
)

sns.scatterplot(
    df_origin_norm,
    x="FAMD_0",
    y="FAMD_1",
    hue=TARGET,
    ax=axes[1],
)

axes[0].set_title(f"Explained Variance:  {np.sum(famd_train.percentage_of_variance_) /100:.2f}")
axes[1].set_title(f"Explained Variance:  {np.sum(famd_origin.percentage_of_variance_) /100:.2f}")

plt.show()


df_train_copy = df_train[num_cols + cat_cols].copy()
df_test_copy = df_test[num_cols + cat_cols].copy()
df_origin_copy = df_origin.copy()

df_train_copy["age_group"] = pd.cut(
    df_train_copy["age"],
    bins=[18, 33, 39, 48, 70, np.Infinity],
    labels=["18-33", "33-39", "39-48", "48-70", "outliers"],
    include_lowest=True,
)
df_test_copy["age_group"] = pd.cut(
    df_test_copy["age"],
    bins=[18, 33, 39, 48, 70, np.Infinity],
    labels=["18-33", "33-39", "39-48", "48-70", "outliers"],
    include_lowest=True,
)
df_origin_copy["age_group"] = pd.cut(
    df_origin_copy["age"],
    bins=[18, 33, 39, 48, 70, np.Infinity],
    labels=["18-33", "33-39", "39-48", "48-70", "outliers"],
    include_lowest=True,
)
df_train_copy["balance_group"] = pd.cut(
    df_train_copy["balance"],
    bins=[-np.Infinity, 0, 630, 1388, 2000, np.Infinity],
    labels=["negative", "0-630$", "630-1388$", "1388-2000$", ">2000$"],
    include_lowest=True,
)
df_test_copy["balance_group"] = pd.cut(
    df_test_copy["balance"],
    bins=[-np.Infinity, 0, 630, 1388, 2000, np.Infinity],
    labels=["negative", "0-630$", "630-1388$", "1388-2000$", ">2000$"],
    include_lowest=True,
)
df_origin_copy["balance_group"] = pd.cut(
    df_origin_copy["balance"],
    bins=[-np.Infinity, 0, 630, 1388, 2000, np.Infinity],
    labels=["negative", "0-630$", "630-1388$", "1388-2000$", ">2000$"],
    include_lowest=True,
)
df_train_copy["campaign_group"] = pd.cut(
    df_train_copy["campaign"],
    bins=[0, 1, 2, 3, 6, np.Infinity],
    labels=["1", "2", "3", "3-6", "outlier"],
    include_lowest=True,
)
df_test_copy["campaign_group"] = pd.cut(
    df_test_copy["campaign"],
    bins=[0, 1, 2, 3, 6, np.Infinity],
    labels=["1", "2", "3", "3-6", "outlier"],
    include_lowest=True,
)
df_origin_copy["campaign_group"] = pd.cut(
    df_origin_copy["campaign"],
    bins=[0, 1, 2, 3, 6, np.Infinity],
    labels=["1", "2", "3", "3-6", "outlier"],
    include_lowest=True,
)
df_test_copy["day_group"] = pd.cut(
    df_test_copy["day"],
    bins=[0, 1, 9, 17, 21, np.Infinity],
    labels=["1", "<9", "<17", "<21", ">21"],
    include_lowest=True,
)
df_train_copy["day_group"] = pd.cut(
    df_train_copy["day"],
    bins=[0, 1, 9, 17, 21, np.Infinity],
    labels=["1", "<9", "<17", "<21", ">21"],
    include_lowest=True,
)
df_origin_copy["day_group"] = pd.cut(
    df_origin_copy["day"],
    bins=[0, 1, 9, 17, 21, np.Infinity],
    labels=["1", "<9", "<17", "<21", ">21"],
    include_lowest=True,
)

extend_num_cols = num_cols + ["balance_div_age"]

extend_cat_cols = cat_cols + [
    "age_group",
    "balance_group",
    "day_group",
    "campaign_group",
]

df_train_copy[TARGET] = df_train[TARGET]
df_origin_copy[TARGET] = df_origin[TARGET]

df_train_copy["Source"] = "Train"
df_test_copy["Source"] = "Test"

df_combine = pd.concat([df_train_copy, df_test_copy], ignore_index=True)


fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()
for ax, col in zip(axes, num_cols):
        sns.kdeplot(data=df_combine,  x=col, ax=ax, hue="Source")

plt.show()


fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

plt.suptitle("Train Dataset")

for ax, col in zip(axes, num_cols):
        sns.kdeplot(data=df_train,  x=col, ax=ax, hue=TARGET)

plt.show()


fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

plt.suptitle("Origin Dataset")

for ax, col in zip(axes, num_cols):
        sns.kdeplot(data=df_origin,  x=col, ax=ax, hue=TARGET)

plt.show()


fig, axes = plt.subplots(13, 1, figsize=(18, 52))
axes = axes.flatten()
for ax, col in zip(axes, extend_cat_cols):
        sns.histplot(data=df_combine,  x=col, ax=ax, hue="Source", multiple="stack")

plt.show()


fig, axes = plt.subplots(13, 1, figsize=(18, 52))
axes = axes.flatten()

plt.suptitle("Train Dataset")

for ax, col in zip(axes, extend_cat_cols):
    sns.histplot(data=df_train_copy, x=col, ax=ax, hue=TARGET, multiple="stack")

plt.show()


fig, axes = plt.subplots(13, 1, figsize=(18, 52))
axes = axes.flatten()

plt.suptitle("Origin Dataset")

for ax, col in zip(axes, extend_cat_cols):
    sns.histplot(data=df_origin_copy, x=col, ax=ax, hue=TARGET, multiple="stack")

plt.show()


def distribution_between_cat(ps: pd.Series):
    percents = ps / ps.sum()
    return percents


def highlight_cell(val):
    color = "background-color: ''"
    if val < 0.10:
        color = "background-color: blue"
    if val > 0.15:
        color = "background-color: red"
    return color


def detect_deviation(df: pd.DataFrame):

    fig, axes = plt.subplots(13, 1, figsize=(24, 56))
    axes = axes.flatten()
    for ax, col in zip(axes, extend_cat_cols):
        crosstab_table = pd.crosstab(df[TARGET], df[col], normalize=True)
        crosstab_table_by_percents = crosstab_table.apply(distribution_between_cat)

        sns.heatmap(crosstab_table, annot=True, cmap="coolwarm", ax=ax, vmin=0, vmax=1)
        display(crosstab_table_by_percents.style.map(highlight_cell, subset=1))

    plt.show()


detect_deviation(df_train_copy)


detect_deviation(df_origin_copy)


def plot_boxplot(df:pd.DataFrame, col):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title("Outlier Detection via Boxplot")
    plt.show()


for col in num_cols:
    plot_boxplot(df_train, col)


for col in num_cols:
    plot_boxplot(df_origin, col)


for col in num_cols:
    plot_boxplot(df_test, col)


df_copy = df_train[df_train["duration"] < 250]

plot_boxplot(df_copy, "duration")


# Detect & Replace Outliers (Z-score method)
# from scipy import stats

# skf = StratifiedKFold(n_splits=3, random_state=SEED, shuffle=True)

# # Copy dataset to avoid modifying original
# X, y = (
#     df_train.copy()
#     .drop(TARGET, axis=1)
#     .astype(
#         {
#             "job": "category",
#             "marital": "category",
#             "education": "category",
#             "default": "category",
#             "housing": "category",
#             "loan": "category",
#             "contact": "category",
#             "month": "category",
#             "poutcome": "category",
#         }
#     ),
#     df_train[TARGET],
# )

# scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)


# model = XGBClassifier(
#     scale_pos_weight=scale_pos_weight,
#     device="cuda",
#     random_state=SEED,
#     eval_metrics="auc",
#     enable_categorical=True,
#     verbosity=0,
# )


# best_auc = np.mean(cross_val_score(model, X, y, cv=skf, scoring="roc_auc", verbose=0))

# # Set Z-score threshold
# threshold = 3

# for col in num_cols:

#     # Calculate Z-scores
#     z_scores = np.abs(stats.zscore(X[col]))

#     # Print how many outliers detected
#     print(f"{col}: {np.sum(z_scores > threshold)} outliers detected")

#     # Compute boundaries (mean ± 3*std)
#     mean, std = X[col].mean(), X[col].std()
#     upper, lower = mean + threshold * std, mean - threshold * std

#     # Replace values outside boundaries with caps
#     # X[col] = np.where(
#     #     X[col] > upper,
#     #     upper,
#     #     np.where(X[col] < lower, lower, X[col]),
#     # )

#     X_target = X[X[col].between(lower, upper)]
#     y_target = y[X_target.index]

#     scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)

#     model = XGBClassifier(
#         scale_pos_weight=scale_pos_weight,
#         device="cuda",
#         random_state=SEED,
#         eval_metrics="auc",
#         enable_categorical=True,
#         verbosity=0,
#     )
#     target_auc = np.mean(
#         cross_val_score(model, X_target, y_target, cv=skf, scoring="roc_auc", verbose=0)
#     )
#     if target_auc > best_auc:
#         print(f"Before {best_auc:.4f} After remove AUC: {target_auc:.4f}")
#         print("Difference AUC: ", target_auc - best_auc, "\n")
#         print(f"Lower: {lower:.2f}", f"Upper: {upper:.2f}", "\n")
#         print("=" * 100)
#         best_auc = target_auc

# print("\n✅ Outliers replaced with boundary values (capped).")


from itertools import combinations


def combine_difference_from(feature_importance, df):
    top_left = [y for x, y in enumerate(feature_importance) if x % 2 == 0]
    top_right = [y for x, y in enumerate(feature_importance) if x % 2 != 0]
    for i in range(len(top_left)):
        for j in range(len(top_right)):
            colName = top_left[i] + "_SUB_" + top_right[j] + "_DIV_" + top_left[i]
            df[colName] = np.where(
                ((df[top_left[i]] - df[top_right[j]]) == 0) | (df[top_left[i]] == 0),
                0,
                (df[top_left[i]] - df[top_right[j]]) / df[top_left[i]],
            )
    return df


def preprocessing(
    df: pd.DataFrame, encode=False, mix_cat=False, mix_num=False, remove_outliers=True
):
    X = df.copy()

    if remove_outliers:
        X = X[
            (X["age"] >= 10)
            & (X["age"] <= 71)
            & (X["balance"] >= -7304.22)
            & (X["balance"] <= 9712.36)
            & (X["pdays"] >= -209.55)
            & (X["pdays"] <= 254.37)
        ]

    X.loc[X["job"] == "unknown", "job"] = "self-employed"
    X.loc[X["poutcome"] == "unknown", "poutcome"] = "failure"
    y = None

    X["age_group"] = pd.cut(
        X["age"],
        bins=[18, 33, 39, 48, 70],
        labels=["18-33", "33-39", "39-48", "48-70"],
        include_lowest=True,
    )
    X["age_marital"] = X["age_group"].astype(str) + " " + X["marital"]
    X["age_squared"] = X["age"] ** 2

    X["balance_group"] = pd.cut(
        X["balance"],
        bins=[-np.Infinity, 0, 630, 1388, 2000, np.Infinity],
        labels=["negative", "0-1388$", "630-1388$", "1388-2000$", ">2000$"],
        include_lowest=True,
    )
    X["balance_positive"] = (X["balance"] > 0).astype(int)
    X["balance_negative"] = (X["balance"] < 0).astype(int)
    X["balance_log"] = np.log1p(X["balance"] - X["balance"].min() + 1)

    X["campaign_group"] = pd.cut(
        X["campaign"],
        bins=[0, 1, 2, 3, 6],
        labels=["1", "2", "3", "3-6"],
        include_lowest=True,
    )
    X["campaign_intensity"] = X["campaign"] * X["duration"]

    X["credit_info"] = X["default"] + " " + X["housing"] + " " + X["loan"]

    X["contact_month_day_interaction"] = (
        X["month"].astype(str) + "_" + X["day"].astype(str)
    )

    X["day_group"] = pd.cut(
        X["day"],
        bins=[0, 1, 17, 21, np.Infinity],
        labels=["1", "<17", "<21", ">21"],
        include_lowest=True,
    )

    X["duration_log"] = np.log1p(X["duration"])
    X["duration_per_campaign"] = X["duration"] / (X["campaign"] + 1)
    X["duration_squared"] = X["duration"] ** 2
    X["duration_sqrt"] = np.sqrt(X["duration"])
    X["duration_to_age_ratio"] = X["duration"] / X["age"]

    X["job_education"] = X["job"].astype(str) + "_" + X["education"].astype(str)

    X["has_previous_contact"] = (X["previous"] > 0).astype(int)

    X["has_loan_and_housing"] = ((X["loan"] == "yes") & (X["housing"] == "yes")).astype(
        int
    )
    X["no_loan_no_housing"] = ((X["loan"] == "no") & (X["housing"] == "no")).astype(int)

    X["pdays_active"] = (X["pdays"] != -1).astype(int)
    X["pdays_bins"] = X["pdays"].replace(-1, 999)
    X["pdays_bins"] = pd.cut(
        X["pdays_bins"], bins=[-1, 0, 100, 200, 300, 400, 1000], labels=False
    )

    X["last_contact_info"] = (
        X["contact"] + " " + X["day"].astype(str) + " " + X["month"]
    )

    X["multiple_campaigns"] = (X["campaign"] > 1).astype(int)

    X["prev_success_indicator"] = (
        (X["poutcome"] == "success")
        | ((X["poutcome"] == "other") & (X["previous"] > 2))
    ).astype(int)

    X["prev_to_campaign_ratio"] = X["previous"] / (X["campaign"] + 1)

    X["previously_contacted"] = (X["pdays"] != -1).astype(int)
    X["previous_contact_outcome"] = X["previous"].astype(str) + " " + X["poutcome"]

    X["total_contacts"] = X["campaign"] + X["previous"]

    if TARGET in X.columns.to_list():
        y = X[TARGET]
        X.drop(TARGET, axis=1, inplace=True)

    X = X.astype(
        {
            "job": "category",
            "marital": "category",
            "education": "category",
            "default": "category",
            "housing": "category",
            "loan": "category",
            "contact": "category",
            "month": "category",
            "poutcome": "category",
            "age_marital": "category",
            "last_contact_info": "category",
            "previously_contacted": "category",
            "previous_contact_outcome": "category",
            "job_education": "category",
            "total_contacts": "category",
            "credit_info": "category",
            "pdays_bins": "category",
            "contact_month_day_interaction": "category",
        }
    )

    cat_cols = X.select_dtypes(include="category").columns.to_list()

    if mix_cat:
        cat_cols_set = set(cat_cols)
        for left_col, right_col in combinations(
            [
                "contact",
                "contact_month_day_interaction",
                "credit_info",
                "month",
                "no_loan_no_housing",
                "pdays_bins",
                "poutcome",
                "prev_success_indicator",
            ],
            2,
        ):
            mix_col_name = left_col + "_vs_" + right_col
            X[mix_col_name] = X[left_col].astype(str) + " - " + X[right_col].astype(str)
            cat_cols_set.add(mix_col_name)
        cat_cols = list(cat_cols_set)

    if mix_num:
        X = combine_difference_from(
            ["campaign", "duration"],
            X,
        )

    if encode:
        label_encoder(X, cat_cols)
    return X, y


# def target_encode(train, valid, test, col, target=TARGET, kfold=5, smooth=3):
#     train['kfold'] = ((train.index) % kfold)
#     col_name = '_'.join(col)
#     train[f'TE_MEAN_' + col_name] = 0.

#     np.random.seed(42)
    
#     for i in range(kfold):
#         df_tmp = train[train['kfold'] != i]
#         mn = train[target].mean()
#         df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
#         df_tmp.columns = col + ['mean', 'count']
#         df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
#         df_tmp_m = train[col + ['kfold', f'TE_MEAN_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
#         df_tmp_m.loc[df_tmp_m['kfold'] == i, f'TE_MEAN_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
#         train[f'TE_MEAN_' + col_name] = df_tmp_m[f'TE_MEAN_' + col_name].fillna(mn).values

#     df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
#     mn = train[target].mean()
#     df_tmp.columns = col + ['mean', 'count']
#     df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
    
#     df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
#     valid[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
#     valid[f'TE_MEAN_' + col_name] = valid[f'TE_MEAN_' + col_name].astype('float32')

#     df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
#     test[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
#     test[f'TE_MEAN_' + col_name] = test[f'TE_MEAN_' + col_name].astype('float32')

#     train = train.drop('kfold', axis=1)
#     train[f'TE_MEAN_' + col_name] = train[f'TE_MEAN_' + col_name].astype('float32')

#     return (train, valid, test)

# def count_encode(train, valid, test, col):
#     counts = train[col].value_counts()

#     train[f'CE_{col}'] = train[col].map(counts)
#     valid[f'CE_{col}'] = valid[col].map(counts).fillna(0)
#     test[f'CE_{col}'] = test[col].map(counts).fillna(0)
#     return (train, valid, test)


X, y = preprocessing(
    pd.concat(
        [df_train, df_origin.loc[df_origin[TARGET] == 1]],
    ),
    encode=True,
    mix_cat=False,
    mix_num=False,
)

# calculate class weights based on the training data
scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)



# top_10 = set()

# fig, axes = plt.subplots(1, 2, figsize=(24, 18))

# models = [
#     LGBMClassifier(
#         scale_pos_weight=scale_pos_weight,
#         device="gpu",
#         random_state=SEED,
#         importance_type="gain",
#         eval_metrics="auc",
#         verbosity=-1,
#     ),
#     XGBClassifier(
#         scale_pos_weight=scale_pos_weight,
#         device="cuda",
#         random_state=SEED,
#         importance_type="gain",
#         eval_metrics="auc",
#         verbosity=0,
#     ),
# ]

# fi = []

# for model, ax in zip(models, axes):
#     output = cross_validate(
#         model, X, y, cv=skf, scoring="roc_auc", return_estimator=True, verbose=0
#     )

#     fi = []
#     for estimator in output["estimator"]:
#         fi.append(estimator.feature_importances_)

#     fi = pd.DataFrame(
#         np.array(fi).T,
#         columns=["importance " + str(idx) for idx in range(len(fi))],
#         index=X.columns,
#     )

#     fi["mean_importance"] = fi.mean(axis=1)

#     features = fi["mean_importance"]
#     features = features.sort_values(ascending=True)

#     top_10.update(set(features.tail(10).index))

#     features.plot.barh(ax=ax, title=f"AUC: {np.abs(output['test_score'].mean()):.2f}")

# plt.show()
# top_10


X, y = preprocessing(
    pd.concat([df_train, df_origin.loc[df_origin[TARGET] == 1]]),
    encode=True,
    mix_cat=True,
    mix_num=True,
)

scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)


skf = StratifiedKFold(n_splits=3, random_state=SEED, shuffle=True)

def select_features(X, y, est, params, cv=skf, scoring="roc_auc", fit_params=None):
    model_all_features = est(**params)

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

    tol = 0.00001
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
            X.drop(features_to_remove + [feature], axis=1),
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
        if diff_auc >= tol:
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


# select_features(
#     X,
#     y,
#     LGBMClassifier,
#     {
#         "scale_pos_weight": scale_pos_weight,
#         "device": "gpu",
#         "random_state": SEED,
#         "importance_type": "gain",
#         "verbosity": -1,
#     },
# )


# select_features(
#     X,
#     y,
#     XGBClassifier,
#     {
#         "scale_pos_weight": scale_pos_weight,
#         "random_state": SEED,
#         "device": "cuda",
#         "importance_type": "gain",
#         "verbosity": 0,
#     },
# )


LGB_FEATURES = [
    "credit_info_vs_prev_success_indicator",
    "contact_vs_prev_success_indicator",
    "contact_vs_poutcome",
    "previous",
    "contact_vs_no_loan_no_housing",
    "prev_to_campaign_ratio",
    "pdays_bins_vs_prev_success_indicator",
    "prev_success_indicator",
    "credit_info_vs_poutcome",
    "balance_group",
    "previous_contact_outcome",
    "age_marital",
    "total_contacts",
    "campaign",
    "loan",
    "campaign_intensity",
    "month_vs_prev_success_indicator",
    "no_loan_no_housing_vs_pdays_bins",
    "credit_info_vs_pdays_bins",
    "job_education",
    "marital",
    "pdays",
    "education",
    "last_contact_info",
    "campaign_SUB_duration_DIV_campaign",
    "month_vs_no_loan_no_housing",
    "month_vs_poutcome",
    "age_squared",
    "month_vs_pdays_bins",
    "contact_month_day_interaction_vs_pdays_bins",
    "contact_month_day_interaction_vs_credit_info",
    "contact_month_day_interaction",
    "contact_month_day_interaction_vs_prev_success_indicator",
    "no_loan_no_housing_vs_prev_success_indicator",
    "contact_month_day_interaction_vs_poutcome",
    "balance",
    "day",
    "balance_log",
    "contact_vs_contact_month_day_interaction",
    "credit_info_vs_month",
    "contact_vs_pdays_bins",
    "duration_per_campaign",
    "duration_to_age_ratio",
    "contact_vs_month",
    "contact_month_day_interaction_vs_no_loan_no_housing",
    "poutcome",
    "duration_squared",
]
XGB_FEATURES = [
    "day_group",
    "housing",
    "balance_group",
    "age_marital",
    "credit_info_vs_prev_success_indicator",
    "prev_to_campaign_ratio",
    "credit_info_vs_poutcome",
    "contact_vs_prev_success_indicator",
    "previous_contact_outcome",
    "total_contacts",
    "pdays_bins_vs_prev_success_indicator",
    "contact_vs_no_loan_no_housing",
    "campaign_intensity",
    "age",
    "pdays_bins_vs_poutcome",
    "campaign",
    "month_vs_prev_success_indicator",
    "job_education",
    "pdays",
    "marital",
    "contact_month_day_interaction_vs_pdays_bins",
    "balance",
    "contact_vs_poutcome",
    "contact_month_day_interaction_vs_prev_success_indicator",
    "campaign_SUB_duration_DIV_campaign",
    "contact_month_day_interaction_vs_poutcome",
    "education",
    "day",
    "contact_month_day_interaction",
    "has_loan_and_housing",
    "month_vs_poutcome",
    "contact_month_day_interaction_vs_credit_info",
    "duration_to_age_ratio",
    "contact_vs_contact_month_day_interaction",
    "loan",
    "duration_per_campaign",
    "month_vs_no_loan_no_housing",
    "credit_info_vs_month",
    "contact_vs_pdays_bins",
    "contact_vs_month",
    "no_loan_no_housing_vs_poutcome",
    "no_loan_no_housing_vs_prev_success_indicator",
    "pdays_bins",
    "duration",
    "prev_success_indicator",
    "contact_vs_credit_info",
    "poutcome",
]


from lightgbm import early_stopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import validate_data, check_is_fitted

class CustomLGBMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, params, features=None):
        self.params = params
        self.features = features

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
            random_state=SEED,
            test_size=0.2,
            shuffle=True,
            stratify=self.y_,
        )
        self.model_ = LGBMClassifier(**self.params)
        self.model_.fit(
            X_train_,
            y_train_,
            eval_set=[(X_val_, y_val_)],
            eval_names=["valid"],
            eval_metric=["auc"],
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

    def predict_proba(self, X):

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
        return self.model_.predict_proba(X)


class CustomXGBMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, params, features=None):
        self.params = params
        self.features = features

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
            random_state=SEED,
            test_size=0.2,
            shuffle=True,
            stratify=self.y_,
        )
        self.model_ = XGBClassifier(
            **self.params, early_stopping_rounds=50, eval_metrics="auc"
        )
        self.model_.fit(X_train_, y_train_, eval_set=[(X_val_, y_val_)], verbose=0)
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

    def predict_proba(self, X):

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
        return self.model_.predict_proba(X)


import optuna

skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

def optimize_lightgbm(X, y, features):

    def objective(trial: optuna.Trial):
        """Define the objective function for LGB Model"""

        params = {
            "objective": "binary",
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100.0, log=True),
            'scale_pos_weight': scale_pos_weight,  # Added for imbalanced data
            "random_state": SEED,
            "device": "gpu",
            "verbosity": -1,
        }

        scores = []
        for i, (tr, val) in enumerate(skf.split(X, y)):
            X_tr, X_ts, y_tr, y_ts = (
                X.iloc[tr, :],
                X.iloc[val, :],
                y.iloc[tr],
                y.iloc[val],
            )
            model = CustomLGBMClassifier(
                {
                    **params,
                    "device": "gpu",
                    "random_state": SEED,
                    "verbosity": -1,
                },
                features,
            )
            model.fit(X_tr, y_tr)
            roc_auc = roc_auc_score(y_ts, model.predict_proba(X_ts)[:, 1])
            scores.append(roc_auc)

            print(f"Fold {i}:", roc_auc)

        cv = abs(np.mean(scores))

        print("=" * 50)
        print("\n")

        return cv

    # Create an Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Get the best parameters
    best_params = study.best_params
    print("Best Params: ", best_params)
    return best_params


def optimize_xgbm(X, y, features):

    def objective(trial: optuna.Trial):
        """Define the objective function for XGB Model"""

        params = {
            'objective': 'binary:logistic',
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-9, 1.0, log=True),
            'scale_pos_weight': scale_pos_weight,  # Added for imbalanced data
        }

        scores = []
        for i, (tr, val) in enumerate(skf.split(X, y)):
            X_tr, X_ts, y_tr, y_ts = (
                X.iloc[tr, :],
                X.iloc[val, :],
                y.iloc[tr],
                y.iloc[val],
            )
            model = CustomXGBMClassifier(
                {
                    **params,
                    "device": "cuda",
                    "random_state": SEED,
                    "verbosity": 0,
                },
                features,
            )
            model.fit(X_tr, y_tr)
            roc_auc = roc_auc_score(y_ts, model.predict_proba(X_ts)[:, 1])
            scores.append(roc_auc)

            print(f"Fold {i}:", roc_auc)

        cv = abs(np.mean(scores))

        print("=" * 50)
        print("\n")

        return cv

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Get the best parameters
    best_params = study.best_params
    print("Best Params: ", best_params)
    return best_params


# optimize_lightgbm(X, y, LGB_FEATURES)


# optimize_xgbm(X, y, XGB_FEATURES)


lgb_params = {
    "objective": "binary",
    "n_estimators": 8135,
    "max_depth": 12,
    "learning_rate": 0.027867891024600612,
    "num_leaves": 219,
    "min_data_in_leaf": 182,
    "subsample": 0.18178421488477295,
    "feature_fraction": 0.7613970808295375,
    "reg_lambda": 28.233862371584653,
    "reg_alpha": 0.00048704831648803095,
    "scale_pos_weight": scale_pos_weight,
    "device": "gpu",
    "random_state": SEED,
    "verbosity": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "n_estimators": 3964,
    "learning_rate": 0.013580886246766444,
    "min_child_weight": 63,
    "max_depth": 13,
    "subsample": 0.9497713646792296,
    "colsample_bytree": 0.8991990376992522,
    "reg_lambda": 94.91675895717631,
    "reg_alpha": 5.284728977887067e-09,
    "gamma": 0.11184406159493408,
    "scale_pos_weight": scale_pos_weight,
    "device": "cuda",
    "random_state": SEED,
    "verbosity": 0,
}


df_pred = df_test.id.to_frame()
X_target, _ = preprocessing(df_test.drop("id", axis=1), encode=True, mix_cat=True, mix_num=True, remove_outliers=False)


FOLDS = 10

skf = StratifiedKFold(n_splits=FOLDS, random_state=SEED, shuffle=True)

scores = []
df_pred[TARGET] = 0

for i, (tr, val) in tqdm(enumerate(skf.split(X, y))):
    X_tr, X_ts, y_tr, y_ts = (
        X.iloc[tr, :],
        X.iloc[val, :],
        y.iloc[tr],
        y.iloc[val],
    )
    lgb_model = CustomLGBMClassifier(
        lgb_params,
        LGB_FEATURES,
    )
    xgb_model = CustomXGBMClassifier(xgb_params, XGB_FEATURES)
    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    lgb_roc_auc = roc_auc_score(y_ts, lgb_model.predict_proba(X_ts)[:, 1])
    xgb_roc_auc = roc_auc_score(y_ts, xgb_model.predict_proba(X_ts)[:, 1])
    roc_auc = np.mean([lgb_roc_auc, xgb_roc_auc])
    df_pred[TARGET] += np.mean(
        np.column_stack(
            [
                lgb_model.predict_proba(X_target)[:, 1],
                xgb_model.predict_proba(X_target)[:, 1],
            ]
        ),
        axis=1,
    )

    scores.append(roc_auc)

    print(f"Fold {i}:", roc_auc)
print(f"Average Fold ROC-AUC: {np.mean(scores):.5f} \xb1 {np.std(scores):.5f}")
df_pred[TARGET] /= FOLDS
df_pred.to_csv("submission.csv", index=False)

