!unzip "/kaggle/input/sberbank-russian-housing-market/train.csv.zip"
!unzip "/kaggle/input/sberbank-russian-housing-market/test.csv.zip"
!unzip "/kaggle/input/sberbank-russian-housing-market/macro.csv.zip"


# import pandas as pd
# # From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
# macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
# "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
# "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

# df_train = pd.read_csv("/kaggle/working/train.csv", parse_dates=['timestamp'])
# df_test = pd.read_csv("/kaggle/working/test.csv", parse_dates=['timestamp'])
# df_macro = pd.read_csv("/kaggle/working/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# # Save the target variable before dropping columns
# y_train_all = df_train["price_doc"].values  # Keep raw price_doc before log transformation

# # Drop unnecessary columns
# df_train.drop(columns=["id", "price_doc"], inplace=True)
# df_test.drop(columns=["id"], inplace=True)

# # Merge Train with Macro Data before any preprocessing
# df_train = pd.merge_ordered(df_train, df_macro, on="timestamp", how="left")

# # Store column names before splitting
# feature_columns = df_train.columns.tolist()

# df_train.head()


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


# # Create bins for stratification (only for splitting)
# num_bins = 45  
# y_binned = pd.qcut(y_train_all, q=num_bins, labels=False, duplicates="drop")

# bin_counts = pd.Series(y_binned).value_counts().sort_index()
# print(bin_counts)

# plt.figure(figsize=(10, 5))
# plt.bar(bin_counts.index, bin_counts.values, width=0.5, edgecolor="black")
# plt.xlabel("Bin Number")
# plt.ylabel("Number of Samples")
# plt.title("Distribution of Samples Across Bins")
# plt.show()

# # Calculate dataset size
# N = len(y_train_all)

# # Compute bin counts using different rules
# sturges_bins = int(np.log2(N) + 1)
# rice_bins = int(2 * (N ** (1/3)))

# iqr = np.percentile(y_train_all, 75) - np.percentile(y_train_all, 25)
# fd_bin_width = 2 * iqr / (N ** (1/3))
# fd_bins = max(1, int((y_train_all.max() - y_train_all.min()) / fd_bin_width))

# # Print results
# print(f"Sturges' Rule Suggested Bins: {sturges_bins}")
# print(f"Rice Rule Suggested Bins: {rice_bins}")
# print(f"Freedman-Diaconis Rule Suggested Bins: {fd_bins}")



# # First split: 70% train, 30% (valid+test) using stratification
# X_train, X_valid_test, y_train, y_valid_test = train_test_split(
#     df_train, y_train_all, test_size=0.3, random_state=42, stratify=y_binned
# )

# # Create bins for valid+test split (again, only for stratification)
# y_valid_test_binned = pd.qcut(y_valid_test, q=num_bins, labels=False, duplicates="drop")

# # Second split: 50% valid, 50% test from the 30% portion, using stratification again
# X_valid, X_test, y_valid, y_test = train_test_split(
#     X_valid_test, y_valid_test, test_size=0.5, random_state=42, stratify=y_valid_test_binned
# )

# # Output data shapes
# print(f"Training Set: {X_train.shape}, Validation Set: {X_valid.shape}, Test Set: {X_test.shape}")


# # Convert back to DataFrames with column names
# train_df = pd.DataFrame(X_train, columns=feature_columns)
# train_df["price_doc"] = y_train  # Keep raw price_doc

# valid_df = pd.DataFrame(X_valid, columns=feature_columns)
# valid_df["price_doc"] = y_valid

# test_df = pd.DataFrame(X_test, columns=feature_columns)
# test_df["price_doc"] = y_test

# # Save raw datasets before preprocessing
# train_df.to_csv("train_set.csv", index=False)
# valid_df.to_csv("valid_set.csv", index=False)
# test_df.to_csv("test_set.csv", index=False)

print("Raw datasets saved as train_set.csv, valid_set.csv, and test_set.csv with column names.")

# ---- Now continue with preprocessing on train, valid, and test separately ----

# Feature Engineering (on train, valid, test separately)
def preprocess(df, y):
    df = df.copy()
    
    df = df.assign(
        month_year_cnt=lambda df: df["timestamp"].dt.month + df["timestamp"].dt.year * 100,
        week_year_cnt=lambda df: df["timestamp"].dt.isocalendar().week + df["timestamp"].dt.year * 100,
        month=lambda df: df["timestamp"].dt.month,
        dow=lambda df: df["timestamp"].dt.dayofweek,
        rel_floor=lambda df: df["floor"] / df["max_floor"].astype(float),
        rel_kitch_sq=lambda df: df["kitch_sq"] / df["full_sq"].astype(float),
    )
    
    # Map counts
    df["month_year_cnt"] = df["month_year_cnt"].map(df["month_year_cnt"].value_counts())
    df["week_year_cnt"] = df["week_year_cnt"].map(df["week_year_cnt"].value_counts())

    # Drop timestamp column to prevent overfitting
    df.drop(columns=["timestamp"], inplace=True)

    # Handle categorical values
    df_numeric = df.select_dtypes(exclude=["object"])
    df_categorical = df.select_dtypes(include=["object"]).apply(lambda col: pd.factorize(col)[0])
    # Merge categorical and numerical features
    df_values = pd.concat([df_numeric, df_categorical], axis=1)

    # Store column names for preprocessed data
    feature_columns_preprocessed = df_values.columns.tolist()

    # Replace NaN with 0 and handle Inf values
    df_values = df_values.astype(np.float64)
    df_values = np.nan_to_num(df_values, nan=0.0, posinf=1e6, neginf=-1e6)

    # Apply log1p transformation to target variable
    y_transformed = np.log1p(y)

    return df_values, y_transformed, feature_columns_preprocessed

# # Apply preprocessing separately to train, valid, and test
# X_train_preprocessed, y_train_preprocessed, feature_columns_preprocessed = preprocess(X_train, y_train)
# X_valid_preprocessed, y_valid_preprocessed, _ = preprocess(X_valid, y_valid)
# X_test_preprocessed, y_test_preprocessed, _ = preprocess(X_test, y_test)

# # Convert back to DataFrames with proper column names
# train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_columns_preprocessed)
# train_preprocessed_df["price_doc"] = y_train_preprocessed

# valid_preprocessed_df = pd.DataFrame(X_valid_preprocessed, columns=feature_columns_preprocessed)
# valid_preprocessed_df["price_doc"] = y_valid_preprocessed

# test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_columns_preprocessed)
# test_preprocessed_df["price_doc"] = y_test_preprocessed

# # Save the preprocessed datasets with column names
# train_preprocessed_df.to_csv("train_set_preprocessed.csv", index=False)
# valid_preprocessed_df.to_csv("valid_set_preprocessed.csv", index=False)
# test_preprocessed_df.to_csv("test_set_preprocessed.csv", index=False)

# print("Preprocessed datasets saved as train_set_preprocessed.csv, valid_set_preprocessed.csv, and test_set_preprocessed.csv with column names.")


import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed datasets
train_df = pd.read_csv("/kaggle/input/russian-housing-data/train_set_preprocessed.csv")
valid_df = pd.read_csv("/kaggle/input/russian-housing-data/valid_set_preprocessed.csv")
test_df = pd.read_csv("/kaggle/input/russian-housing-data/test_set_preprocessed.csv")


# Extract target variable
y_train = train_df["price_doc"].values
y_valid = valid_df["price_doc"].values
y_test = test_df["price_doc"].values

# Drop target column from features
X_train = train_df.drop(columns=["price_doc"]).values
X_valid = valid_df.drop(columns=["price_doc"]).values
X_test = test_df.drop(columns=["price_doc"]).values

print('Data loaded')

# Print dataset shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape : {X_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"X_test shape    : {X_test.shape}")

# XGBoost Parameters
xgb_params = {
    'eta': 0.026826111637055765,
    'max_depth': 6,
    'subsample': 0.8903097223566383,
    'colsample_bytree': 0.5419870495955178,
    'lambda': 6.802319056152236,
    'alpha': 0.14207928596262176,
    'min_child_weight': 2,
    'gamma': 0.2536073527964441,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}

# Evaluation Metrics
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Metrics") -> None:
    """Print multiple regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n--- {title} ---")
    print(f"RMSLE: {rmsle(y_true, y_pred):.6f}")
    print(f"RMSE : {np.sqrt(mse):.6f}")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RÂ²   : {r2_score(y_true, y_pred):.6f}\n")

# Cross-Validation Setup
rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
oof_preds = np.zeros(len(X_train))
best_iterations = []

# Repeated K-Fold CV
for fold, (train_idx, valid_idx) in enumerate(rkf.split(X_train), 1):
    X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
    X_fold_valid, y_fold_valid = X_train[valid_idx], y_train[valid_idx]

    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dvalid = xgb.DMatrix(X_fold_valid, label=y_fold_valid)

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dvalid, "eval")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    best_iterations.append(model.best_iteration)
    oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))

# Print overall CV (OOF) metrics
print_metrics(y_train, oof_preds, title="Overall CV (OOF) Metrics")
avg_best_iter = int(np.mean(best_iterations))
print(f"Average best boosting rounds from CV: {avg_best_iter}")

# Train Final Model on Full CV Training Set
dtrain_cv = xgb.DMatrix(X_train, label=y_train)
final_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain_cv,
    num_boost_round=avg_best_iter
)

# Evaluate on Hold-Out Validation Set
dvalid = xgb.DMatrix(X_valid, label=y_valid)
valid_preds = final_model.predict(dvalid, iteration_range=(0, avg_best_iter))
print_metrics(y_valid, valid_preds, title="Hold-Out Validation Metrics")


import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error
from optuna.pruners import HyperbandPruner

# Define RMSLE metric function
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Define the Optuna objective function
def objective(trial):
    """Objective function for Optuna to minimize RMSLE."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),  # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),  # L1 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),  # Minimum loss reduction
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "tree_method": "gpu_hist",  # Enable GPU
        "device": "cuda",  # Specify CUDA device
    }

    # Split data for cross-validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, valid_idx) in enumerate(rkf.split(X_train), 1):
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_valid, y_fold_valid = X_train[valid_idx], y_train[valid_idx]

        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dvalid = xgb.DMatrix(X_fold_valid, label=y_fold_valid)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            evals=[(dvalid, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Store OOF predictions
        oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))

    # Compute RMSLE score
    return rmsle(y_train, oof_preds)

# Run Optuna optimization
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(direction="minimize",pruner=pruner)
study.optimize(objective, timeout=3600)

# Best hyperparameters found
best_params = study.best_params
print(f"\nðŸ”¹ Best Hyperparameters:\n{best_params}")

# Train final model with best hyperparameters on GPU
best_params["tree_method"] = "gpu_hist"
best_params["device"] = "cuda"

dtrain_cv = xgb.DMatrix(X_train, label=y_train)
final_model = xgb.train(
    params=best_params,
    dtrain=dtrain_cv,
)

# Evaluate on Hold-Out Validation Set
dvalid = xgb.DMatrix(X_valid, label=y_valid)
valid_preds = final_model.predict(dvalid)
print_metrics(y_valid, valid_preds, title="Hold-Out Validation Metrics")



import shap
import matplotlib.pyplot as plt

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_train)

# Plot summary of SHAP feature importance
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train)



# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, RepeatedKFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.datasets import fetch_california_housing

# # Load California Housing dataset
# data = fetch_california_housing()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target, name="MedHouseVal")

# # Create bins for stratification (only for splitting)
# num_bins = 20  
# y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

# bin_counts = pd.Series(y_binned).value_counts().sort_index()
# print(bin_counts)

# plt.figure(figsize=(10, 5))
# plt.bar(bin_counts.index, bin_counts.values, width=0.5, edgecolor="black")
# plt.xlabel("Bin Number")
# plt.ylabel("Number of Samples")
# plt.title("Distribution of Samples Across Bins")
# plt.show()

# # Calculate dataset size
# N = len(y)

# # Compute bin counts using different rules
# sturges_bins = int(np.log2(N) + 1)
# rice_bins = int(2 * (N ** (1/3)))

# iqr = np.percentile(y, 75) - np.percentile(y, 25)
# fd_bin_width = 2 * iqr / (N ** (1/3))
# fd_bins = max(1, int((y.max() - y.min()) / fd_bin_width))

# # Print results
# print(f"Sturges' Rule Suggested Bins: {sturges_bins}")
# print(f"Rice Rule Suggested Bins: {rice_bins}")
# print(f"Freedman-Diaconis Rule Suggested Bins: {fd_bins}")

# # First split: 70% train, 30% (valid+test) using stratification
# X_train, X_valid_test, y_train, y_valid_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y_binned
# )

# # Create bins for valid+test split (again, only for stratification)
# y_valid_test_binned = pd.qcut(y_valid_test, q=num_bins, labels=False, duplicates="drop")

# # Second split: 50% valid, 50% test from the 30% portion, using stratification again
# X_valid, X_test, y_valid, y_test = train_test_split(
#     X_valid_test, y_valid_test, test_size=0.5, random_state=42, stratify=y_valid_test_binned
# )
# # Applying a log transformation
# y_train = np.log1p(y_train)
# y_valid = np.log1p(y_valid)
# y_test = np.log1p(y_test)

# # Concatenate features and target
# train_df = pd.concat([X_train, y_train], axis=1)
# valid_df = pd.concat([X_valid, y_valid], axis=1)
# test_df = pd.concat([X_test, y_test], axis=1)

# # Save datasets to CSV
# train_df.to_csv("train.csv", index=False)
# valid_df.to_csv("valid.csv", index=False)
# test_df.to_csv("test.csv", index=False)


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# # Load California Housing dataset
# data = fetch_california_housing()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target, name="MedHouseVal")

# # Split dataset into train, validation, and test sets
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Applying a log transformation
# y_train = np.log1p(y_train)
# y_valid = np.log1p(y_valid)
# y_test = np.log1p(y_test)

# # Concatenate features and target
# train_df = pd.concat([X_train, y_train], axis=1)
# valid_df = pd.concat([X_valid, y_valid], axis=1)
# test_df = pd.concat([X_test, y_test], axis=1)

# # Save datasets to CSV
# train_df.to_csv("train.csv", index=False)
# valid_df.to_csv("valid.csv", index=False)
# test_df.to_csv("test.csv", index=False)

# Load preprocessed datasets
train_df = pd.read_csv("/kaggle/input/california-housing-data/train.csv")
valid_df = pd.read_csv("/kaggle/input/california-housing-data/valid.csv")
test_df = pd.read_csv("/kaggle/input/california-housing-data/test.csv")

# Extract target variable
y_train = train_df["MedHouseVal"]
y_valid = valid_df["MedHouseVal"]
y_test = test_df["MedHouseVal"]

# Drop target column from features
X_train = train_df.drop(columns=["MedHouseVal"])
X_valid = valid_df.drop(columns=["MedHouseVal"])
X_test = test_df.drop(columns=["MedHouseVal"])

print('Data loaded')

# Print dataset shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape : {X_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"X_test shape    : {X_test.shape}")

# XGBoost Parameters
xgb_params = {
    'eta': 0.07250900083296051,
    'max_depth': 13,
    'subsample': 0.6616172000082796,
    'colsample_bytree': 0.6418773439284294,
    'lambda': 2.2926428387310045,
    'alpha': 0.5709733765314046,
    'min_child_weight': 9,
    'gamma': 0.011011205912381427,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}

# Evaluation Metrics
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Metrics") -> None:
    """Print multiple regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n--- {title} ---")
    print(f"RMSE : {np.sqrt(mse):.6f}")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RÂ²   : {r2_score(y_true, y_pred):.6f}\n")

# Cross-Validation Setup
rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
oof_preds = np.zeros(len(X_train))
best_iterations = []

# Repeated K-Fold CV
for fold, (train_idx, valid_idx) in enumerate(rkf.split(X_train), 1):
    X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_fold_valid, y_fold_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dvalid = xgb.DMatrix(X_fold_valid, label=y_fold_valid)

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dvalid, "eval")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    best_iterations.append(model.best_iteration)
    oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))

# Print overall CV (OOF) metrics
print_metrics(y_train, oof_preds, title="Overall CV (OOF) Metrics")
avg_best_iter = int(np.mean(best_iterations))
print(f"Average best boosting rounds from CV: {avg_best_iter}")

# Train Final Model on Full CV Training Set
dtrain_cv = xgb.DMatrix(X_train, label=y_train)
final_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain_cv,
    num_boost_round=avg_best_iter
)

# Evaluate on Hold-Out Validation Set
dvalid = xgb.DMatrix(X_valid, label=y_valid)
valid_preds = final_model.predict(dvalid, iteration_range=(0, avg_best_iter))
print_metrics(y_valid, valid_preds, title="Hold-Out Validation Metrics")


import shap
import matplotlib.pyplot as plt

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_train)

# Plot summary of SHAP feature importance
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train)



import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define RMSE metric function
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Define Optuna objective function
def objective(trial):
    """Objective function for Optuna to minimize RMSE."""
    params = {
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),  # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),  # L1 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),  # Minimum loss reduction
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "tree_method": "gpu_hist",  # Enable GPU acceleration
        "device": "cuda",  # Use CUDA
    }

    # Split data for cross-validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    oof_preds = np.zeros(len(X_train))
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(rkf.split(X_train), 1):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_valid, y_fold_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dvalid = xgb.DMatrix(X_fold_valid, label=y_fold_valid)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,
            evals=[(dvalid, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        best_iterations.append(model.best_iteration)
        oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))

    # Compute RMSE score
    return rmse(y_train, oof_preds)

# # Run Optuna optimization
# pruner = optuna.pruners.HyperbandPruner()
# study = optuna.create_study(direction="minimize",pruner=pruner)
# study.optimize(objective, timeout=3600) 

# # Best hyperparameters found
# best_params = study.best_params
# print(f"\nðŸ”¹ Best Hyperparameters:\n{best_params}")

# # Train final model with best hyperparameters on GPU
# best_params["tree_method"] = "gpu_hist"
# best_params["device"] = "cuda"

# dtrain_cv = xgb.DMatrix(X_train, label=y_train)
# final_model = xgb.train(
#     params=best_params,
#     dtrain=dtrain_cv,
#     num_boost_round=200
# )

# # Evaluate on Hold-Out Validation Set
# dvalid = xgb.DMatrix(X_valid, label=y_valid)
# valid_preds = final_model.predict(dvalid)

# print_metrics(y_valid, valid_preds, title="Hold-Out Validation Metrics")



train_df

