import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt


os.makedirs("checkpoints/checkpoints_ft", exist_ok=True)
os.makedirs("checkpoints/checkpoints_lgb", exist_ok=True)
save_dir = "checkpoints/lgb_min_leaf_sweep/minleaf_50"
os.makedirs(save_dir, exist_ok=True)
assert os.access(save_dir, os.W_OK), "Directory is not writable!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



SEED = 42
N_SPLITS = 5

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": None,       # from sweep
    "max_depth": -1,
    "min_data_in_leaf": None, # from sweep
    "feature_fraction": None, # from sweep
    "bagging_fraction": None, # from sweep
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbosity": -1,
    "seed": SEED, 
}

os.makedirs("checkpoints", exist_ok=True)



def encode_categorical(train_df, test_df, features):
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = train_df[features].select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols)
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    return train_df, test_df

def load_data(train_path="/kaggle/input/diabetes-detection-dataset/data/train.csv", 
              test_path="/kaggle/input/diabetes-detection-dataset/data/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    TARGET = "diagnosed_diabetes"
    FEATURES = [c for c in train_df.columns if c not in [TARGET, "id"]]
    train_df, test_df = encode_categorical(train_df, test_df, FEATURES)
    X = train_df[FEATURES].values
    y = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    print("Train shape:", X.shape, "Test shape:", X_test.shape)
    return X, y, X_test, test_df, FEATURES, TARGET



class DiabetesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test



def train_lgb_cv(
    X,
    y,
    X_test,
    params = BASE_PARAMS,
    n_splits=5,
    num_boost_round=2000,
    early_stopping_rounds=100,
    seed=42,
    save_dir="checkpoints/lgb_base",
    train_from_scratch=True
):
    """
    Train LightGBM with Stratified K-Fold CV and checkpointing.

    Returns:
        oof_preds  : Out-of-fold predictions
        test_preds : Averaged test predictions
        cv_auc     : Overall CV AUC score
    """
    os.makedirs(save_dir, exist_ok=True)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nLGB Fold {fold+1}/{n_splits}")

        ckpt_path = f"{save_dir}/lgb_fold_{fold}.txt"

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if train_from_scratch or not os.path.exists(ckpt_path):
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(100),
                ],
            )

            model.save_model(ckpt_path)
        else:
            print(f"Loading checkpoint for fold {fold}")
            model = lgb.Booster(model_file=ckpt_path)

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration)
            / n_splits
        )

    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nLightGBM CV AUC: {cv_auc:.5f}")

    return oof_preds, test_preds, cv_auc



def sweep_min_data_in_leaf(
    X,
    y,
    X_test,
    base_params = BASE_PARAMS,
    start=25,
    stop=1025,
    step=25,
    n_splits=N_SPLITS,
    seed=SEED,
    num_boost_round=2000,
    early_stopping_rounds=100,
    save_dir="checkpoints/lgb_min_leaf_sweep",
):
    """
    Sweep min_data_in_leaf values and return the best one based on CV AUC.
    """
    os.makedirs(save_dir, exist_ok=True)

    min_leaf_values = list(range(start, stop, step))
    results = []

    for min_leaf in min_leaf_values:
        print(f"\nTesting min_data_in_leaf = {min_leaf}")

        params = base_params.copy()
        params["min_data_in_leaf"] = min_leaf

        oof_preds = np.zeros(len(y))
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        run_dir = f"{save_dir}/minleaf_{min_leaf}"
        os.makedirs(run_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Ensure numeric
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(early_stopping_rounds)],
            )

            model.save_model(f"{run_dir}/lgb_fold_{fold}.txt")

            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

        cv_auc = roc_auc_score(y, oof_preds)
        print(f"CV AUC: {cv_auc:.5f}")

        results.append({
            "min_data_in_leaf": min_leaf,
            "cv_auc": cv_auc
        })

    results_df = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
    results_df.to_csv(f"{save_dir}/results.csv", index=False)

    best_row = results_df.iloc[0]
    best_min_leaf = int(best_row["min_data_in_leaf"])
    best_auc = best_row["cv_auc"]

    print("\nBest min_data_in_leaf")
    print(results_df.head())
    print(f"\nBest value = {best_min_leaf} | CV AUC = {best_auc:.5f}")

    return best_min_leaf, results_df



def plot_lgb_sweep_results(
    results_df,
    param_name="min_data_in_leaf",
    metric_name="cv_auc",
    title="LightGBM Hyperparameter Sweep",
    figsize=(20, 5),
):
    """
    Plot CV AUC vs a swept LightGBM hyperparameter.
    """
    df = results_df.sort_values(param_name)

    best_row = df.loc[df[metric_name].idxmax()]
    best_param = int(best_row[param_name])
    best_metric = best_row[metric_name]

    plt.figure(figsize=figsize)
    plt.plot(df[param_name], df[metric_name], marker="o")
    plt.scatter(best_param, best_metric)
    plt.axvline(best_param, linestyle="--")
    plt.axhline(best_metric, linestyle="--")

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(metric_name)
    plt.grid(True)

    plt.text(
        best_param,
        best_metric,
        f"  Best: {best_param}\n  AUC: {best_metric:.5f}",
        verticalalignment="bottom",
    )

    plt.show()



def sweep_num_leaves(
    X,
    y,
    X_test,
    min_leaves=16,
    max_leaves=16,
    step=16,
    fixed_min_data_in_leaf=None,  
    n_splits=5,
    seed=42,
    save_dir="checkpoints/lgb_num_leaves_sweep"
):
    """
    Sweep LightGBM `num_leaves` values while keeping `min_data_in_leaf` fixed.

    Args:
        X, y, X_test : Training features, labels, and test features
        min_leaves, max_leaves, step : range for num_leaves sweep
        fixed_min_data_in_leaf : best min_data_in_leaf from previous sweep
        n_splits : number of CV folds
        seed : random seed
        save_dir : directory to save models and results

    Returns:
        best_num_leaves : integer, the best performing number of leaves
        results_df : DataFrame containing all sweep results
    """
    import os
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    if fixed_min_data_in_leaf is None:
        raise ValueError("You must provide fixed_min_data_in_leaf (e.g., the best from sweep_min_data_in_leaf)")

    os.makedirs(save_dir, exist_ok=True)

    base_params = BASE_PARAMS

    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for num_leaves in range(min_leaves, max_leaves + 1, step):
        print(f"\nTesting num_leaves = {num_leaves}")

        params = base_params.copy()
        params["num_leaves"] = num_leaves

        oof_preds = np.zeros(len(y))
        run_dir = f"{save_dir}/numleaves_{num_leaves}"
        os.makedirs(run_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold+1}/{n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=2000,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(100)],
            )

            model.save_model(f"{run_dir}/lgb_fold_{fold}.txt")

            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds

        cv_auc = roc_auc_score(y, oof_preds)
        print(f"CV AUC = {cv_auc:.5f}")

        results.append({
            "num_leaves": num_leaves,
            "cv_auc": cv_auc
        })

    results_df = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
    results_df.to_csv(f"{save_dir}/results.csv", index=False)

    best_row = results_df.iloc[0]
    best_num_leaves = int(best_row["num_leaves"])
    best_auc = best_row["cv_auc"]

    print("\nBEST RESULT")
    print(results_df)
    print(f"\nBest num_leaves = {best_num_leaves} | CV AUC = {best_auc:.5f}")

    return best_num_leaves, results_df



def sweep_feature_bagging_fraction(
    X,
    y,
    X_test,
    feature_fractions=(0.6, 0.7, 0.8, 0.9),
    bagging_fractions=(0.6, 0.7, 0.8),
    n_splits=N_SPLITS,
    seed=SEED,
    num_boost_round=2000,
    early_stopping_rounds=100,
    save_dir="checkpoints/lgb_feature_bagging_sweep"
):
    """
    Sweep feature_fraction × bagging_fraction together using Stratified K-Fold CV.
    """
    import os
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    os.makedirs(save_dir, exist_ok=True)

    results = []
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for ff in feature_fractions:
        for bf in bagging_fractions:
            print(f"\nTesting feature_fraction={ff}, bagging_fraction={bf}")

            params = BASE_PARAMS.copy()
            params["feature_fraction"] = ff
            params["bagging_fraction"] = bf
            params["bagging_freq"] = 1

            oof_preds = np.zeros(len(y))
            run_name = f"ff_{ff}_bf_{bf}"
            run_dir = f"{save_dir}/{run_name}"
            os.makedirs(run_dir, exist_ok=True)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f"  Fold {fold+1}/{n_splits}")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val)

                model = lgb.train(
                    params=params,
                    train_set=dtrain,
                    num_boost_round=num_boost_round,
                    valid_sets=[dtrain, dval],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(early_stopping_rounds)],
                )

                model.save_model(f"{run_dir}/lgb_fold_{fold}.txt")

                val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                oof_preds[val_idx] = val_preds

            cv_auc = roc_auc_score(y, oof_preds)
            print(f"CV AUC = {cv_auc:.5f}")

            results.append({
                "feature_fraction": ff,
                "bagging_fraction": bf,
                "cv_auc": cv_auc
            })

    results_df = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
    results_df.to_csv(f"{save_dir}/results.csv", index=False)

    best_row = results_df.iloc[0]
    best_ff = best_row["feature_fraction"]
    best_bf = best_row["bagging_fraction"]
    best_auc = best_row["cv_auc"]

    print("\nBEST RESULT")
    print(results_df.head())
    print(f"\nBest feature_fraction = {best_ff}")
    print(f"Best bagging_fraction = {best_bf}")
    print(f"Best CV AUC = {best_auc:.5f}")

    return best_ff, best_bf, results_df



'''
# --- Load preprocessed data ---
X, y, X_test, test_df, FEATURES, TARGET = load_data(
    train_path="../data/train.csv",
    test_path="../data/test.csv"
)

# ----------------------------------------------------
# 1) Sweep min_data_in_leaf
# ----------------------------------------------------
best_min_leaf, min_leaf_results = sweep_min_data_in_leaf(
    X,
    y,
    X_test,
    start=25,
    stop=1025,
    step=25,
    base_params=BASE_PARAMS,
    save_dir="checkpoints/lgb_min_leaf_sweep"
)

BASE_PARAMS["min_data_in_leaf"] = best_min_leaf


# ----------------------------------------------------
# 2) Sweep num_leaves (min_data_in_leaf fixed)
# ----------------------------------------------------
best_num_leaves, num_leaves_results = sweep_num_leaves(
    X,
    y,
    X_test,
    min_leaves=2,
    max_leaves=40,
    step=1,
    fixed_min_data_in_leaf=best_min_leaf,
    n_splits=5,
    seed=SEED,
    save_dir="checkpoints/lgb_num_leaves_sweep"
)

BASE_PARAMS["num_leaves"] = best_num_leaves


# ----------------------------------------------------
# 3) Sweep feature_fraction × bagging_fraction
#    (tree structure fully fixed)
# ----------------------------------------------------
best_ff, best_bf, fb_results = sweep_feature_bagging_fraction(
    X,
    y,
    X_test,
    feature_fractions=np.arange(0.65, 0.90, 0.05),
    bagging_fractions=np.arange(0.65, 0.90, 0.05),
    n_splits=5,
    seed=SEED,
    save_dir="checkpoints/lgb_feature_bagging_sweep"
)


BASE_PARAMS["feature_fraction"] = best_ff
BASE_PARAMS["bagging_fraction"] = best_bf
BASE_PARAMS["bagging_freq"] = 1
'''

# Comment this line if you will run the above excution
BASE_PARAMS["min_data_in_leaf"] = 500
BASE_PARAMS["num_leaves"] = 20
BASE_PARAMS["feature_fraction"] = np.float64(0.65)
BASE_PARAMS["bagging_fraction"] = np.float64(0.7500000000000001)
BASE_PARAMS["bagging_freq"] = 1

# ----------------------------------------------------
# Final tuned parameters
# ----------------------------------------------------
print("Best LightGBM hyperparameters after all sweeps:")
print(BASE_PARAMS)



def plot_lgb_sweep_results(
    results_csv_path,
    param_name,
    title=None,
    figsize=(20, 5)
):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv_path)

    # Sort for clean curve
    df = df.sort_values(param_name)

    best_row = df.loc[df["cv_auc"].idxmax()]
    best_value = best_row[param_name]
    best_auc = best_row["cv_auc"]

    plt.figure(figsize=figsize)
    plt.plot(df[param_name], df["cv_auc"], marker="o")
    plt.scatter(best_value, best_auc)
    plt.axvline(best_value, linestyle="--")
    plt.axhline(best_auc, linestyle="--")

    plot_title = title or f"LightGBM: {param_name} vs CV AUC"
    plt.title(plot_title)
    plt.xlabel(param_name)
    plt.ylabel("CV AUC")
    plt.grid(True)

    plt.text(
        best_value,
        best_auc,
        f"  Best: {best_value}\n  AUC: {best_auc:.5f}",
        verticalalignment="bottom"
    )

    plt.show()

    return best_value, best_auc



def plot_feature_bagging_heatmap(
    results_csv_path,
    figsize=(10, 6),
    cmap="viridis",
    annot=True
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load results
    df = pd.read_csv(results_csv_path)

    # Pivot to matrix form
    pivot = df.pivot(
        index="bagging_fraction",
        columns="feature_fraction",
        values="cv_auc"
    )

    # Find best configuration
    best_row = df.loc[df["cv_auc"].idxmax()]
    best_ff = best_row["feature_fraction"]
    best_bf = best_row["bagging_fraction"]
    best_auc = best_row["cv_auc"]

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=annot,
        fmt=".5f",
        linewidths=0.5
    )

    plt.title("LightGBM: feature_fraction × bagging_fraction (CV AUC)")
    plt.xlabel("feature_fraction")
    plt.ylabel("bagging_fraction")

    # Highlight best cell
    plt.scatter(
        pivot.columns.get_loc(best_ff) + 0.5,
        pivot.index.get_loc(best_bf) + 0.5,
        s=200,
        c="red",
        marker="o",
        edgecolors="black"
    )

    plt.text(
        pivot.columns.get_loc(best_ff) + 0.6,
        pivot.index.get_loc(best_bf) + 0.6,
        f"Best\nAUC={best_auc:.5f}",
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.show()

    return best_ff, best_bf, best_auc



'''
best_min_leaf_val, best_min_leaf_auc = plot_lgb_sweep_results(
    results_csv_path="checkpoints/lgb_min_leaf_sweep/results.csv",
    param_name="min_data_in_leaf",
    title="LightGBM: min_data_in_leaf Sweep Results"
)


best_num_leaves_val, best_num_leaves_auc = plot_lgb_sweep_results(
    results_csv_path="checkpoints/lgb_num_leaves_sweep/results.csv",
    param_name="num_leaves",
    title="LightGBM: num_leaves Sweep Results"
)

best_ff, best_bf, best_fb_auc = plot_feature_bagging_heatmap(
    results_csv_path="checkpoints/lgb_feature_bagging_sweep/results.csv",
    figsize=(12, 7),
    annot=True
)
'''
print()


from IPython.display import Image, display

# -------------------------------------------------
# 1️) min_data_in_leaf Sweep
# -------------------------------------------------
print("Effect of leaf-level regularization on CV ROC AUC")
display(Image("/kaggle/input/diabetes-detection-dataset/Images/min_data_in_leaf_sweep.png"))

# -------------------------------------------------
# 2️) num_leaves Sweep
# -------------------------------------------------
print("Interaction between tree capacity and fixed leaf constraints")
display(Image("/kaggle/input/diabetes-detection-dataset/Images/num_leaves_sweep.png"))

# -------------------------------------------------
# 3️) feature_fraction × bagging_fraction Heatmap
# -------------------------------------------------
print("Joint stochastic regularization behavior after fixing tree structure")
display(Image("/kaggle/input/diabetes-detection-dataset/Images/feature_bagging_heatmap.png"))



def train_lgb(
    X, y, X_test,
    n_splits=N_SPLITS,
    params=None,
    train_from_scratch=True,
    save_dir="checkpoints/checkpoints_lgb"
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    if params is None:
        from __main__ import BASE_PARAMS   # or import from wherever BASE_PARAMS is defined
        params = BASE_PARAMS.copy()

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=params.get("seed", 42))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nLGB Fold {fold+1}/{n_splits}")

        ckpt_path = f"{save_dir}/lgb_fold_{fold}.txt"
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if train_from_scratch or not os.path.exists(ckpt_path):
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=2000,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=100),
                ],
            )

            model.save_model(ckpt_path)
        else:
            print(f"Loading checkpoint for fold {fold}")
            model = lgb.Booster(model_file=ckpt_path)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

        test_preds += model.predict(X_test) / n_splits

    print("LightGBM CV AUC:", roc_auc_score(y, oof_preds))
    return oof_preds, test_preds



def ensemble_predictions(lgb_preds,  test_df, ft_preds = None, filename="ensemble_submission.csv", use_ft=True, ft_weight=0.5, lgb_weight=0.5):
    """
    Combine FT-Transformer and LightGBM predictions for final submission, optionally using FT.

    Parameters:
    -----------
    ft_preds : np.ndarray
        Predictions from FT-Transformer.
    lgb_preds : np.ndarray
        Predictions from LightGBM.
    test_df : pd.DataFrame
        Test dataframe containing 'id' column.
    filename : str
        Output CSV filename.
    use_ft : bool
        Whether to include FT-Transformer predictions.
    ft_weight : float
        Weight for FT-Transformer predictions (used only if use_ft=True).
    lgb_weight : float
        Weight for LightGBM predictions.

    Returns:
    --------
    pd.DataFrame
        Submission dataframe saved to CSV.
    """
    if use_ft:
        ensemble_preds = ft_weight * ft_preds + lgb_weight * lgb_preds
    else:
        ensemble_preds = lgb_preds

    submission = pd.DataFrame({"id": test_df["id"], "diagnosed_diabetes": ensemble_preds})
    submission.to_csv(filename, index=False)
    return submission



# Load data
X, y, X_test, test_df, FEATURES, TARGET = load_data()

# Train FT-Transformer (set train_from_scratch=True to retrain)
# oof_ft, test_ft = train_ft_transformer(X, y, X_test, train_from_scratch=False) # This was for the fft and now no need for it

# Train LightGBM
oof_lgb, test_lgb = train_lgb(X, y, X_test, train_from_scratch=True)

# Ensemble predictions
# Set use_ft=True to include FT predictions, False to use only LightGBM
ensemble_submission = ensemble_predictions(
    lgb_preds=test_lgb,
    test_df=test_df,
    filename="ensemble_submission_v3.csv",
    use_ft=False,   # change to False if you want LightGBM only
    ft_weight=0, # optional weighting for FT predictions
    lgb_weight=1 # optional weighting for LightGBM predictions
)

# Display the first few rows of submission
ensemble_submission.head()



