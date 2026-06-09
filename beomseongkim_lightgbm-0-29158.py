!pip install missingno
!pip install lightgbm
!pip install xgboost
!pip install bayesian-optimization


import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
from scipy import sparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from bayes_opt import BayesianOptimization

%matplotlib inline


path = "/kaggle/input/porto-seguro-safe-driver-prediction/"

train = pd.read_csv(path + "train.csv", index_col = "id")
test = pd.read_csv(path + "test.csv", index_col = "id")
submission = pd.read_csv(path + "sample_submission.csv", index_col = "id")


train.shape, test.shape


train.head()


test.head()


submission.head()


train.info()


train_copy = train.copy().replace(-1, np.NaN)

msno.bar(df = train_copy.iloc[:, 1:29], figsize = (10, 6));


msno.bar(df = train_copy.iloc[:, 29:], figsize = (10, 6));


# "Feature Summary Table

def resumetable(df):
    print(f"Dataset shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns = ["data type"])
    summary["num_NaN"] = (df == -1).sum().values
    summary["num_Unique"] = df.nunique().values
    summary["Data Type"] = None
    for col in df.columns:
        if "bin" in col or col == "target":
            summary.loc[col, "Data Type"] = "Binary"
        elif "cat" in col:
            summary.loc[col, "Data Type"] = "Nominal"
        elif df[col].dtype == float:
            summary.loc[col, "Data Type"] = "Continuous"
        elif df[col].dtype == int:
            summary.loc[col, "Data Type"] = "Ordinal"

    return summary


summary = resumetable(train)
summary


def write_percent(ax, total_size):
    """Iterates over shape objects and 
    displays the target value ratio at the top of each bar"""
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        left_coord = patch.get_x()
        percent = height/total_size * 100

        ax.text(x = left_coord + width/2.0,
                y = height + total_size * 0.001,
                s = f"{percent: 1.1f}%",
                ha = "center")


mpl.rc("font", size = 15)
plt.figure(figsize = (7, 6))

ax = sns.countplot(x = "target", data = train)
write_percent(ax, len(train))
ax.set_title("Target Distribution");


# Binary features

def plot_target_ratio_by_features(df, features, num_rows, num_cols, size = (12, 18)):
    mpl.rc("font", size = 9)
    plt.figure(figsize = size)
    grid = gridspec.GridSpec(num_rows, num_cols)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

    for idx, feature in enumerate(features):
        ax = plt.subplot(grid[idx])
        sns.barplot(x = feature, y = "target", data = df, palette = "Set2", ax = ax)


bin_features = summary[summary["Data Type"] == "Binary"].index
plot_target_ratio_by_features(train, bin_features, 6, 3)


# nominal features

nom_features = summary[summary["Data Type"] == "Nominal"].index
plot_target_ratio_by_features(train, nom_features, 7, 2)


# Ordinal Features

ord_features = summary[summary["Data Type"] == "Ordinal"].index
plot_target_ratio_by_features(train, ord_features, 8, 2, (12, 20))


# Continuous features 

cont_features = summary[summary["Data Type"] == "Continuous"].index

plt.figure(figsize = (12, 16))
grid = gridspec.GridSpec(5, 2)
plt.subplots_adjust(wspace = 0.2, hspace = 0.4)

for idx, cont_feature in enumerate(cont_features):
    train[cont_feature] = pd.cut(train[cont_feature], 5)
    ax = plt.subplot(grid[idx])
    sns.barplot(x = cont_feature, y = "target", data = train, palette ="Set2", ax = ax)
    ax.tick_params(axis = "x", labelrotation =10)


# correlation 

train_copy = train_copy.dropna()

plt.figure(figsize = (10, 8))
cont_corr = train_copy[cont_features].corr()
sns.heatmap(cont_corr, annot = True, cmap = "OrRd");


# data merge

all_data = pd.concat([train, test], ignore_index = True)
all_data = all_data.drop("target", axis = 1)

all_features = all_data.columns
all_features


# Nominal feature one-hot encoding

cat_features = [feature for feature in all_features if "cat" in feature]
onehot_encoder = OneHotEncoder()

encoded_cat_matrix = onehot_encoder.fit_transform(all_data[cat_features])


# Generate Derived Feature

all_data["num_missing"] = (all_data == -1).sum(axis = 1)
remaining_features = [feature for feature in all_features 
                      if ("cat" not in feature and "calc" not in feature)]

remaining_features.append("num_missing")


# feature that classficiation is "ind"

ind_features = [feature for feature in all_features if "ind" in feature]

is_first_feature = True

for ind_feature in ind_features:
    if is_first_feature:
        all_data["mix_ind"] = all_data[ind_feature].astype(str) + "_"
        is_first_feature = False
    else:
        all_data["mix_ind"] += all_data[ind_feature].astype(str) + "_"


# Add the count of unique values for Nominal Features as new feature

cat_count_features = []
for feature in cat_features + ["mix_ind"]:
    val_counts_dict = all_data[feature].value_counts().to_dict()
    all_data[f"{feature}_count"] = all_data[feature].apply(lambda x: val_counts_dict[x])
    cat_count_features.append(f"{feature}_count")


# Remove unnecessary Features

drop_features = ["ps_ind_14", "ps_ind_10_bin", "ps_ind_11_bin", "ps_ind_12_bin", "ps_ind_13_bin", "ps_car_14"]

all_data_remaining = all_data[remaining_features + cat_count_features].drop(drop_features, axis = 1)

all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining), encoded_cat_matrix], format = "csr")
all_data_sprs


# split the dataset

num_train = len(train)

X = all_data_sprs[: num_train]
X_test = all_data_sprs[num_train: ]

y = train["target"].values


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

bayes_dtrain = lgb.Dataset(X_train, y_train)
bayes_dvalid = lgb.Dataset(X_valid, y_valid)


param_bounds = {"num_leaves": (30, 40),
                 "lambda_l1": (0.7, 0.9),
                 "lambda_l2": (0.9, 1),
                 "feature_fraction": (0.6, 0.7),
                 "bagging_fraction": (0.6, 0.9),
                 "min_child_samples": (6, 10),
                 "min_child_weight": (10, 40)}

fixed_params = {"objective": "binary",
                "learning_rate": 0.005,
                "bagging_freq": 1,
                "randome_state": 1991}


# (bayesian optimization) evaluation function

def eval_function(num_leaves, lambda_l1, lambda_l2, feature_fraction, bagging_fraction, min_child_samples, min_child_weight):
    params = {"num_leaves": int(round(num_leaves)),
              "lambda_l1": lambda_l1,
             "lambda_l2": lambda_l2,
             "feature_fraction": feature_fraction,
             "bagging_fraction": bagging_fraction,
             "min_child_samples": int(round(min_child_samples)),
             "min_child_weight": min_child_weight,
             "feature_pre_filter": False}

    params.update(fixed_params)
    print("hyperparameter:", params)
    lgb_model = lgb.train(params = params,
                          train_set = bayes_dtrain,
                          num_boost_round = 250,
                          valid_sets = bayes_dvalid, feval = gini)

    preds = lgb_model.predict(X_valid)
    gini_score = eval_gini(y_valid, preds)
    print(f"gini coefficient: {gini_score}\n")

    return gini_score


optimizer = BayesianOptimization(f = eval_function,
                                 pbounds = param_bounds,
                                 random_state = 0)


optimizer.maximize(init_points = 3, n_iter = 6)


max_params = optimizer.max['params']
max_params


max_params["num_leaves"] = int(round(max_params["num_leaves"]))
max_params["min_child_samples"] = int(round(max_params["min_child_samples"]))


max_params.update(fixed_params)


max_params


def eval_gini(y_true, y_pred):
    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]
    L_mid = np.linspace(1 / n_samples, 1, n_samples) # diagonal line value

    # Gini Coefficient for predicted values
    pred_order = y_true[y_pred.argsort()]
    L_pred = np.cumsum(pred_order) / np.sum(pred_order) # Lorenz curve
    G_pred = np.sum(L_mid - L_pred) 

    # Gini Coefficient for perfect predictions
    true_order = y_true[y_true.argsort()]
    L_true = np.cumsum(true_order) / np.sum(true_order) 
    G_true = np.sum(L_mid - L_true) 

    return G_pred / G_true


def gini(preds, dtrain):
    labels = dtrain.get_label()
    return "gini", eval_gini(labels, preds), True


folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1991)

oof_val_preds = np.zeros(X.shape[0])
oof_test_preds = np.zeros(X_test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print("#" * 40, f"fold {idx + 1} / fold {folds.n_splits}", "#" * 40)

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)

    lgb_model = lgb.train(params = max_params,
                          train_set = dtrain,
                          num_boost_round = 2500,
                          valid_sets = dvalid, feval = gini)

    oof_test_preds += lgb_model.predict(X_test) / folds.n_splits
    oof_val_preds[valid_idx] += lgb_model.predict(X_valid)

    gini_score = eval_gini(y_valid, oof_val_preds[valid_idx])
    print(f"fold {idx + 1} gini coefficient: {gini_score}\n")


print("OOF valid data gini coefficient: ", eval_gini(y, oof_val_preds))


submission["target"] = oof_test_preds
submission.to_csv("submission.csv")

