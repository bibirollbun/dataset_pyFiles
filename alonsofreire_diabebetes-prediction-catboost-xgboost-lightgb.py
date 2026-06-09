!pip install optuna 

# Load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import optuna.visualization as vis
import warnings
import json
from optuna.exceptions import ExperimentalWarning
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
#from optuna.integration import CatBoostPruningCallback
import lightgbm as lgb
from lightgbm import early_stopping
import xgboost as xgb
import catboost as ctb
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Silence Optuna experimental warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Silence LightGBM warnings
warnings.filterwarnings("ignore", message=".*LightGBM.*")

# Silence Optuna trial failure spam
optuna.logging.set_verbosity(optuna.logging.ERROR)


# This cell stores the function used in the notebook.

#########################################
# Function to load dfsets
#########################################
def load_data(file_path):
    """
    Load dfset from a given file path.

    Parameters:
    file_path (str): The path to the dfset file.

    Returns:
    pd.dfFrame: Loaded dfset as a pandas dfFrame.
    """
    print("="*100)
    print(f"Loading dfset from {file_path}...")
    print("="*100)

    try:
        df = pd.read_csv(file_path)
        print(f"dfset loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading dfset from {file_path}: {e}")
        return None

#########################################
# Function to show basic info of the dfset
#########################################
def show_data_info(df):
    """
    Display basic information about the dfset.

    Parameters:
    df (pd.dfFrame): The dfset to analyze.
    """
    print("="*100)
    print(f" Show Basic info...")
    print("="*100)

    print("\nShape of the dfset:")
    print(df.shape)
    print("\ndf types and non-null counts:")
    print(df.info())



##############################################
# Function to verify duplicate rows in the dfset
###############################################
def verify_duplicates(df):
    """
    Check for duplicate rows in the dfset.

    Parameters:
    df (pd.dfFrame): The dfset to check for duplicates.
    """
    print("="*100)
    print(f" Show duplicated values...")
    print("="*100)

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows in the dfset.")
    else:
        print("No duplicate rows found in the dfset.")

#########################################
# Function to visualize numerical features
# #########################################
def visualize_numerical_features(df):
    """
    Visualize numerical features in the dfset using histograms and box plots.

    Parameters:
    df (pd.dfFrame): The dfset to visualize.
    """
    print("="*100)
    print(f"Visualize numerical features...")
    print("="*100)

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    plt.figure(figsize=(25, 5 * len(numerical_features)))

    for feature in numerical_features:
        plt.subplot(len(numerical_features), 3, numerical_features.index(feature) + 1)
        sns.boxplot(x=df[feature])
        plt.title(f'Box plot of {feature}')

################################################
# Function to verfify missing data
################################################
def verify_missing_data(df):
    """
    Verify missing data in the dfset using a heatmap.

    Parameters:
    df (pd.dfFrame): The dfset to verify.
    """
    print("="*100)
    print(f"Verify missing data...")
    print("="*100)

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    missing_data = missing_data.sort_values(ascending=False)

    if missing_data.empty:
        print("No missing data found in the dfset.")
        return

    print(f"Total missing data: {df.isnull().sum().sum()}")
    print(f"Missing data per column: {missing_data}")
    print(f"Percentage of missing data per column: {missing_data / len(df) * 100}")
    print(f"Percentage of missing data per row: {df.isnull().sum(axis=1) / df.shape[1] * 100}")
    print(f"Percentage of missing data in the dfset: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100}")

#################################################
# Function to verify skweness
#################################################
def verify_skewness(df, threshold=1.0):
    """
    Verify skewness of numeric features and classify them using Bulmer's rule.

    Parameters:
    df (pd.DataFrame): The dataset to analyze
    threshold (float): Absolute skewness above which a feature is considered highly skewed
    """

    def classify_bulmer_skewness(skew_value):
        abs_skew = abs(skew_value)
        if abs_skew < 0.5:
            return "Approximately symmetric"
        elif abs_skew < 1.0:
            return "Moderately skewed"
        else:
            return "Highly skewed"

    print("=" * 100)
    print("Verify skewness of numeric features")
    print("=" * 100)

    numeric_cols = df.select_dtypes(include=[np.number])

    if numeric_cols.empty:
        print("No numeric features found.")
        return

    skewness = numeric_cols.skew().sort_values(
        key=lambda x: x.abs(), ascending=False
    )

    print(f"\nTotal numeric features: {len(skewness)}\n")

    print("-" * 50)
    print("Classification using Bulmer's skewness magnitude")
    print("-" * 50)

    for feature, skew_value in skewness.items():
        classification = classify_bulmer_skewness(skew_value)
        flag = "!!!" if abs(skew_value) > threshold else ""
        print(
            f"{feature:<30} "
            f"skew = {skew_value:>8.3f}  →  {classification}{flag}"
        )

##############################################
# Function to verify skewness by class
##############################################
def verify_skewness_by_class(df, target_col):
    """
    Verify skewness of numeric features separately for each target class.

    Parameters:
    df (pd.DataFrame): Dataset
    target_col (str): Name of the binary target column
    """

    print("=" * 100)
    print("Verify skewness of numeric features by target class")
    print("=" * 100)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(target_col, errors="ignore")

    if len(numeric_cols) == 0:
        print("No numeric features found.")
        return

    classes = sorted(df[target_col].dropna().unique())

    if len(classes) != 2:
        raise ValueError("Target column must be binary.")

    skew_table = []

    for feature in numeric_cols:
        row = {"feature": feature}

        for cls in classes:
            skew_val = df.loc[df[target_col] == cls, feature].skew()
            row[f"skew_class_{cls}"] = skew_val

        row["abs_diff"] = abs(row[f"skew_class_{classes[0]}"] -
                              row[f"skew_class_{classes[1]}"])
        skew_table.append(row)

    skew_df = pd.DataFrame(skew_table)
    skew_df = skew_df.sort_values("abs_diff", ascending=False)

    print(f"\nTop features with largest skewness difference between classes:\n")
    print(skew_df.head(15).to_string(index=False))

##############################################
# Function to visualize biggest skewness difference
###############################################
def visualise_skewness_diff_features(df, target_col, feature_list, palette='viridis'):
    """
    Plot distribution of features with largest skewness difference by target class.

    Parameters:
    - df: pd.DataFrame - Dataset containing features and target
    - target_col: str - Name of the binary target column
    - feature_list: list of str - Features to plot
    - palette: str or list - Color palette for plotting (default 'viridis')
    """
    plt.figure(figsize=(4 * len(feature_list), 5))

    for i, col in enumerate(feature_list, 1):
        plt.subplot(1, len(feature_list), i)
        unique_vals = df[col].nunique()

        if unique_vals <= 2:
            sns.countplot(data=df, x=col, hue=target_col, palette=palette)
            plt.title(f"Countplot of {col}")

        else:
            sns.kdeplot(data=df, x=col, hue=target_col, fill=True, common_norm=False, palette=palette)
            plt.title(f"KDE of {col}\n(Skewness Difference: High)")

    plt.tight_layout()
    plt.show()



##############################################
# Function to visualize outliers in numerical features
###############################################
def visualize_outliers(df):
    """
    Visualize outliers in numerical features using box plots.

    Parameters:
    df (pd.dfFrame): The dfset to visualize.
    """
    print("="*100)
    print(f"Visualize outliers in numerical features...")
    print("="*100)

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Count outliers using IQR method
    outlier_counts = {}
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].shape[0]
        outlier_counts[feature] = outlier_count

    for feature, count in outlier_counts.items():
        print(f"Feature '{feature}' has {count} outliers.")

############################################################
# Function to handle outliers in numerical features by capping
############################################################
def handle_outliers(df, column_name, method, action):
    """
    Handle outliers in the specified column of the dfset using the IQR method.

    Parameters:
    df (pd.dfFrame): Input dfset.
    column_name (str): Column name to check for outliers.
    method (str): Method to handle outliers ('IQR' or 'Z-Score').
    action (str): Action to take on outliers ('remove' or 'cap').

    Returns:
    pd.dfFrame: dfset with outliers handled.
    """
    print('='*100)
    print(f'Handling Outliers in {column_name}...')
    print('='*100)

    # Verify column exists
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the dfset.")
        return df

    # Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Column '{column_name}' is not numeric. Skipping.")
        return df

    # Validate method
    valid_methods = ['IQR', 'Z-Score']
    if method not in valid_methods:
        print(f"Invalid method '{method}'. Valid methods are: {valid_methods}")
        return df

    # Validate action
    valid_actions = ['remove', 'cap']
    if action not in valid_actions:
        print(f"Invalid action '{action}'. Valid actions are: {valid_actions}")
        return df

    # Compute bounds
    if method == 'IQR':
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(f"IQR Method: Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
        print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

    elif method == 'Z-Score':
        mean = df[column_name].mean()
        std = df[column_name].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        print(f"Z-Score Method: Mean = {mean}, Std = {std}")
        print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

    # Apply action
    if action == 'remove':
        df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        removed = len(df) - len(df_cleaned)
        print(f"Removed {removed} outliers from '{column_name}'.")

    elif action == 'cap':
        df_cleaned = df.copy()
        num_capped = ((df[column_name] < lower_bound) | (df[column_name] > upper_bound)).sum()
        df_cleaned[column_name] = df_cleaned[column_name].clip(lower=lower_bound, upper=upper_bound)
        print(f"Capped {num_capped} outliers in '{column_name}'.")

    print(f"dfset shape after handling outliers: {df_cleaned.shape}")

    return df_cleaned

###############################################################
# Function to visualize categorical features
###############################################################
def visualize_categorical_features(df):
    """
    Visualize categorical features in the dfset using count plots.

    Parameters:
    df (pd.dfFrame): The dfset to visualize.
    """
    print("="*100)
    print(f"Visualize categorical features...")
    print("="*100)

    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    plt.figure(figsize=(25, 5 * len(categorical_features)))

    for feature in categorical_features:
        plt.subplot(len(categorical_features), 3, categorical_features.index(feature) + 1)
        sns.countplot(x=df[feature])
        plt.title(f'Count plot of {feature}')

######################################################################
# Function to Encode categorical features
######################################################################
def handle_categorical_encoding_float(df):
    """
    Handles categorical columns encoding, ensuring binary features
    are converted to 0.0 and 1.0 (float64).

    Parameters:
    df (pd.DataFrame): The dataset to process.

    Returns:
    pd.DataFrame: The dataset with encoded categorical features.

    """
    print('='*100)
    print('Handling Categorical Columns Encoding (Output: float64 for Binary)...')
    print('='*100)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Separate columns by category count
    binary_columns = [col for col in categorical_columns if df[col].nunique() == 2]
    nominal_columns = [col for col in categorical_columns if df[col].nunique() > 2]

    # Label Encoding for Binary Columns (and forced float64)
    label_encoders = {}
    for column in binary_columns:
        le = LabelEncoder()

        df[column] = le.fit_transform(df[column].astype(str))

        df[column] = df[column].astype('float64')

        label_encoders[column] = le
        print(f'Label Encoded (Binary) & Converted to float64: {column}')
        print(f'Original Classes: {le.classes_} -> Mapped to: 0.0 and 1.0')

    # One-Hot Encoding for Multi-Category Nominal Columns
    if nominal_columns:
        print(f'\nOne-Hot Encoding (Nominal): {nominal_columns}')
        df = pd.get_dummies(df, columns=nominal_columns, drop_first=True, dtype='float64')

    return df

#############################################################################
# Function to visaulize correlation between features
#############################################################################
def mostly_correlated_features(df, threshold=0.8):
    """
    Identifies, prints, and visualizes features that are highly correlated
    above a certain threshold.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    threshold (float): Correlation threshold to consider.

    Returns:
    None

    """
    print("="*100)
    print(f"Detecting and Visualizing Highly Correlated Features (Threshold={threshold})...")
    print("="*100)

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Calculate the absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select the upper triangle of the correlation matrix (excluding diagonal and lower triangle)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_pairs = []

    # Iterate over the upper triangle to find pairs above the threshold
    for i in range(len(upper_triangle.columns)):
        for j in range(i + 1, len(upper_triangle.columns)):
            col1 = upper_triangle.columns[i]
            col2 = upper_triangle.columns[j]
            correlation = upper_triangle.iloc[i, j]

            if correlation > threshold:
                correlated_pairs.append((col1, col2, correlation))

    # Print the identified pairs
    if correlated_pairs:
        print("\n Highly Correlated Feature Pairs:")
        for col1, col2, corr in correlated_pairs:
            print(f"   - {col1} and {col2}: {corr:.4f}")
    else:
        print("No feature pairs found above the correlation threshold.")
        return

    # Determine Features to Drop
    # This logic correctly identifies the SECOND feature of the highly correlated pair
    features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    print(f"\n Features flagged to drop to reduce multicollinearity: {features_to_drop}")

###############################################################
# Function to create objective for Optuna hyperparameter tuning
###############################################################
def create_objective(data, model_name, fixed_params=None):
    """
    Create an Optuna objective function for hyperparameter tuning
    of multiple classifiers.

    Parameters:
    - data: pd.DataFrame with features and target.
    - model_name: str, one of ['LightGBM', 'XGBoost', 'CatBoost']

    Returns:
    - objective function to be passed to study.optimize()
    """

    X = data.drop(columns=['diagnosed_diabetes'])
    y = data['diagnosed_diabetes']

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype("category")

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    if fixed_params is None:
        fixed_params = {}

    def objective(trial):

        if model_name == 'LightGBM':
            trial_params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.02, 0.08),
                "num_leaves": trial.suggest_int("num_leaves", 2, 144),
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 6.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 6.0, log=True),
            }

            params = {**fixed_params, **trial_params}

            print("="*100)
            print("Classifier started:",model_name)
            print("="*100)
            print("Trial number:", trial.number)

            auc_scores = []
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

            for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                print("Fold number:", fold)

                X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

                X_train = X_train.copy()
                X_valid = X_valid.copy()

                for col in cat_cols:
                    X_train[col] = X_train[col].astype("category")
                    X_valid[col] = X_valid[col].astype("category")

                dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
                dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols)

                callbacks = [
                    early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)
                ]

                gbm = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    valid_sets=[dvalid],
                    callbacks=callbacks,
                )

                y_pred_prob = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
                auc = roc_auc_score(y_valid, y_pred_prob)
                auc_scores.append(auc)

                intermediate_avg = np.mean(auc_scores)
                trial.report(intermediate_avg, fold)

                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at fold {fold}")
                    raise optuna.exceptions.TrialPruned()

                print(f"Fold {fold} | AUC: {auc:.5f} | Best iter: {gbm.best_iteration}")

            trial.set_user_attr("best_iteration", gbm.best_iteration)

        elif model_name == 'XGBoost':
            trial_params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-2, 2.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 3.0, log=True)
            }

            params = {**fixed_params, **trial_params}

            print("="*100)
            print("Classifier started:",model_name)
            print("="*100)
            print("Trial number:", trial.number)

            auc_scores = []
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

            for fold, (train_index, val_index) in enumerate(skf.split(X,y), 1):
                X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

                X_train = X_train.copy()
                X_valid = X_valid.copy()

                for col in cat_cols:
                    X_train[col] = X_train[col].astype("category")
                    X_valid[col] = X_valid[col].astype("category")

                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

                xbm = xgb.train(
                   params,
                   dtrain,
                   num_boost_round=3000,
                   evals=[(dvalid, "valid")],
                   early_stopping_rounds=100,
                   verbose_eval=False,
                )

                y_pred_prob = xbm.predict(dvalid, iteration_range=(0, xbm.best_iteration))
                auc = roc_auc_score(y_valid, y_pred_prob)
                auc_scores.append(auc)

                intermediate_avg = np.mean(auc_scores)
                trial.report(intermediate_avg, fold)

                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at fold {fold}")
                    raise optuna.exceptions.TrialPruned()

                print(f"Fold {fold} | AUC: {auc:.5f} | Best iter: {xbm.best_iteration}")

            trial.set_user_attr("best_iteration", xbm.best_iteration)

        elif model_name == 'CatBoost':
            trial_params = {
                    "iterations": trial.suggest_int("iterations", 500, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 8, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 8.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.5)
                }

            params = {**fixed_params, **trial_params}

            print("="*100)
            print("Classifier started:",model_name)
            print("="*100)
            print("Trial number:", trial.number)

            auc_scores = []
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_features = [X.columns.get_loc(col) for col in cat_cols]

            for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

                X_train = X_train.copy()
                X_valid = X_valid.copy()

                cbm = ctb.CatBoostClassifier(**params)

                #Uncomment for CPU
                #pruning_callback = CatBoostPruningCallback(trial, "AUC")
                cbm.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=100,
                    cat_features=cat_features,
                    use_best_model=True,
                    #callbacks=[pruning_callback],
                )

                #pruning_callback.check_pruned()

                y_pred_prob = cbm.predict_proba(X_valid, ntree_end=cbm.best_iteration_)[:, 1]
                auc = roc_auc_score(y_valid, y_pred_prob)
                auc_scores.append(auc)

                intermediate_avg = np.mean(auc_scores)
                trial.report(intermediate_avg, fold)

                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at fold {fold}")
                    raise optuna.exceptions.TrialPruned()

                print(f"Fold {fold} | AUC: {auc:.5f} | Best iter: {cbm.best_iteration_}")

            trial.set_user_attr("best_iteration", cbm.best_iteration_)

        else:
            raise ValueError(f"Model {model_name} not supported")

        return np.mean(auc_scores)

    return objective

############################################################################
# Function to verify oof predictions
############################################################################
def verify_oof_predictions(oof_preds, X, y, skf, cat_features=None):
  print("="*50)
  print("Starting OOF predictions verification...")
  print("="*50)

  # Shape check
  if oof_preds.shape[0] == X.shape[0]:
      print(f"OOF predictions length matches dataset size: {oof_preds.shape[0]}")
  else:
      print(f"Length mismatch: OOF length {oof_preds.shape[0]} vs data length {X.shape[0]}")

  # Check NaNs
  if np.any(np.isnan(oof_preds)):
      print("NaN values found in OOF predictions!")
  else:
      print("No NaNs in OOF predictions")

  # Check train/val disjointness in each fold
  for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
      if len(set(train_idx) & set(val_idx)) == 0:
          print(f"Fold {fold} train/val sets are disjoint")
      else:
          print(f"Fold {fold} train/val sets overlap!")

  # Fold AUCs
  fold_aucs = []

  for fold, (_, val_idx) in enumerate(skf.split(X, y), 1):
      fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
      fold_aucs.append(fold_auc)
      print(f"Fold {fold} AUC: {fold_auc:.5f}")

  mean_fold_auc = np.mean(fold_aucs)

  print(f"Mean Fold AUC: {mean_fold_auc:.5f}")
  # Overall AUC
  overall_auc = roc_auc_score(y, oof_preds)
  print(f"Overall OOF AUC: {overall_auc:.5f}")

  # Check overall vs mean fold AUC difference
  diff = abs(overall_auc - mean_fold_auc)
  if diff < 0.01:
      print("Overall AUC is consistent with mean fold AUC")
  else:
      print("Overall AUC differs significantly from mean fold AUC, check for leakage")

  # Distribution histogram
  plt.figure(figsize=(8,4))
  plt.hist(oof_preds, bins=50, alpha=0.7)
  plt.title("Distribution of OOF predictions")
  plt.xlabel("Prediction")
  plt.ylabel("Frequency")
  plt.show()

  # Prediction distribution by target class
  df_oof = X.copy()
  df_oof['oof_pred'] = oof_preds
  df_oof['target'] = y.values
  plt.figure(figsize=(8,4))
  sns.boxplot(x='target', y='oof_pred', data=df_oof)
  plt.title("OOF prediction distribution by target class")
  plt.show()
  print("OOF verification complete.")


# Load datasets
train_data = load_data('/kaggle/input/playground-series-s5e12/train.csv')
train_data.head()


# show basic info of the dataset
show_data_info(train_data)
# Visualize numerical features
train_data.describe().T


# Drop ID as it has no value for prediction
train_data.drop(columns='id', inplace=True)


# Verfiy duplicate rows
verify_duplicates(train_data)


# Verify missing data
verify_missing_data(train_data)


# Verify skewness
verify_skewness(train_data,threshold=1)


# Verify skewness by class
verify_skewness_by_class(train_data, target_col='diagnosed_diabetes')


#Family history of diabetes and cardiovascular history:
#   Are highly skewed for both classes, but more skewed in class 0.
#   This means the shape of these binary/history features differs significantly
#   between positive and negative groups.

top_diff_features = [
    'family_history_diabetes',
    'physical_activity_minutes_per_week',
    'cardiovascular_history',
    'hypertension_history'
]

visualise_skewness_diff_features(train_data, target_col='diagnosed_diabetes', feature_list=top_diff_features)



# Visualize numerical features
visualize_numerical_features(train_data)


# Vizualize outliers in numerical features
visualize_outliers(train_data)


# Handle outliers in numerical features(cap)
outliers_cap = [
    'alcohol_consumption_per_week',
    'systolic_bp',
    'diastolic_bp',
    'bmi',
    'physical_activity_minutes_per_week',
    'screen_time_hours_per_day',
    'triglycerides',
    'waist_to_hip_ratio',
    'diastolic_bp',
    'systolic_bp',
    'heart_rate'
    ]
for feature in outliers_cap:
    train_data = handle_outliers(train_data, feature, method='IQR', action='cap')


# Visualize categorical features
visualize_categorical_features(train_data)


# Verify the most correlated features based on threshold
mostly_correlated_features(train_data, threshold=0.7)


# Drop highly highly correlated features
train_data.drop(columns=['bmi', 'cholesterol_total'], inplace=True)


# To keep all models' results for comparison
results = []


# CatBoost tuning
study_cb = optuna.create_study(direction="maximize")
fixed_cb_params = {
    "verbose": False,
    "task_type": "GPU",
    "devices": "0",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "allow_writing_files": False,
    "random_seed": 42,
    "border_count":128,
    "boosting_type": "Plain",
}

objective_cb = create_objective(train_data, 'CatBoost', fixed_params=fixed_cb_params)
study_cb.optimize(objective_cb, n_trials=20, show_progress_bar=True)
best_params_cb = {**fixed_cb_params, **study_cb.best_trial.params}
best_auc_cb = study_cb.best_trial.value
best_iter_cb = study_cb.best_trial.user_attrs.get('best_iteration', None)

results.append(
    {
        'Model': 'CatBoost', 
        'ROC-AUC': best_auc_cb,
        'Best_Params': best_params_cb,
        'Best_Iteration': best_iter_cb
    })


# Visualize the best parameters for Catboost
print("-" * 40)
print(f"{'Parameter':<25} | {'Value'}")
print("-" * 40)
print(f"{'Best AUC':<25} | {best_auc_cb:.4f}")
for key, value in best_params_cb.items():
    # Use :.4f for floats to avoid long decimals
    if isinstance(value, float):
        print(f"{key:<25} | {value:.4f}")
    else:
        print(f"{key:<25} | {value}")


# Visualize optimization history
evaluator = MeanDecreaseImpurityImportanceEvaluator()

display(vis.plot_param_importances(study_cb, evaluator=evaluator))
display(vis.plot_optimization_history(study_cb))


# Save the CatBoost best parameters
with open('best_params_catboost.json', 'w') as f:
    json.dump(best_params_cb, f)


# Train the CatBoost final models on full data
X_cbm = train_data.drop(columns=['diagnosed_diabetes'])
y_cbm = train_data['diagnosed_diabetes']
cat_cols = X_cbm.select_dtypes(include=['object', "category"]).columns.tolist()

cat_features = [X_cbm.columns.get_loc(col) for col in cat_cols]

print("=" * 100)
print(" Training final CatBoost model on FULL data")
print(f"• Samples: {X_cbm.shape[0]}")
print(f"• Features: {X_cbm.shape[1]}")
print(f"• Categorical features: {len(cat_features)}")
print(f"• Iterations: {best_iter_cb}")
print("-" * 80)

final_cb_model = CatBoostClassifier(
    **best_params_cb,
)

final_cb_model.fit(
    X_cbm,
    y_cbm,
    cat_features=cat_features,
    verbose=False
)

print("CatBoost training completed")
print(f"Total trees: {final_cb_model.tree_count_}")
print("=" * 100)


# Save Catboost model
final_cb_model.save_model('catboost_final_model.cbm')

# Save metadata
cbm_metadata = {
    'feature_names': X_cbm.columns.tolist(),
    'cat_features': cat_features
}

with open('cataboost_metadata.json', 'w') as f:
    json.dump(cbm_metadata, f)


# XGBoost tuning
study_xgb = optuna.create_study(direction="maximize")
fixed_xb_params = {
    "verbosity": 0,
    "objective": "binary:logistic",
    "device": "cuda",
    "eval_metric": "auc",
    "tree_method": "hist",
    "booster": "gbtree"
}
objective_xgb = create_objective(train_data, 'XGBoost', fixed_params=fixed_xb_params)
study_xgb.optimize(objective_xgb, n_trials=20, n_jobs=1, show_progress_bar=True)
best_params_xgb = {**fixed_xb_params, **study_xgb.best_trial.params}
best_auc_xgb = study_xgb.best_trial.value
best_iter_xgb = study_xgb.best_trial.user_attrs.get('best_iteration', None)

results.append({
    'Model': 'XGBoost',
    'ROC-AUC': best_auc_xgb,
    'Best_Params': best_params_xgb,
    'Best_Iteration': best_iter_xgb
})


# Visualize the best parameters for XGBoost
print("-" * 40)
print(f"{'Parameter':<25} | {'Value'}")
print("-" * 40)
print(f"{'Best AUC':<25} | {best_auc_xgb:.4f}")
for key, value in best_params_xgb.items():
    # Use :.4f for floats to avoid long decimals
    if isinstance(value, float):
        print(f"{key:<25} | {value:.4f}")
    else:
        print(f"{key:<25} | {value}")


# Visualize optimization history
evaluator = MeanDecreaseImpurityImportanceEvaluator()

display(vis.plot_param_importances(study_xgb, evaluator=evaluator))
display(vis.plot_optimization_history(study_xgb))


# Save the XGBoost best parameters
with open('best_params_xgboost.json', 'w') as f:
    json.dump(best_params_xgb, f)


# Train the XGBoost final models on full data
X_xgb = train_data.drop(columns=['diagnosed_diabetes']).copy()
y_xgb = train_data['diagnosed_diabetes']
cat_cols = X_xgb.select_dtypes(include=['object', 'category']).columns.tolist()


for col in cat_cols:
    X_xgb[col] = X_xgb[col].astype("category")


dtrain = xgb.DMatrix(X_xgb, label=y_xgb, enable_categorical=True)

params = {**best_params_xgb}

print("=" * 100)
print(" Training final XGBBoost model on FULL data")
print(f"• Samples: {X_xgb.shape[0]}")
print(f"• Features: {X_xgb.shape[1]}")
print(f"• Categorical features: {len(cat_cols)}")
print(f"• Iterations: {best_iter_xgb}")
print("-" * 80)

final_xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_iter_xgb
)
print("XGBoost training completed")
print("Total trees:", final_xgb_model.num_boosted_rounds())
print("=" * 100)


# Save XGboost model
final_xgb_model.save_model('xgboost_final_model.json')

# Save metadata
xgb_metadata = {
    'feature_names': X_xgb.columns.tolist(),
    'cat_cols': cat_cols
}

with open('xgboost_metadata.json', 'w') as f:
    json.dump(xgb_metadata, f)


# LightGBM tuning
study_lgb = optuna.create_study(direction="maximize")
fixed_lg_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_threads": 4,
    "verbosity": -1
}
objective_lgb = create_objective(train_data, 'LightGBM', fixed_params=fixed_lg_params )
study_lgb.optimize(objective_lgb, n_trials=20, show_progress_bar=True)
best_params_lgb = {**fixed_lg_params, **study_lgb.best_trial.params}
best_auc_lgb = study_lgb.best_trial.value
best_iter_lgb = study_lgb.best_trial.user_attrs.get('best_iteration', None)

results.append({
    'Model': 'LightGBM',
    'ROC-AUC': best_auc_lgb,
    'Best_Params': best_params_lgb,
    'Best_Iteration': best_iter_lgb
})


# Visualize the best parameters for LightGBM
print("-" * 40)
print(f"{'Parameter':<25} | {'Value'}")
print("-" * 40)
print(f"{'Best AUC':<25} | {best_auc_lgb:.4f}")
for key, value in best_params_lgb.items():
    # Use :.4f for floats to avoid long decimals
    if isinstance(value, float):
        print(f"{key:<25} | {value:.4f}")
    else:
        print(f"{key:<25} | {value}")


# Visualize optimization history
evaluator = MeanDecreaseImpurityImportanceEvaluator()

display(vis.plot_param_importances(study_lgb, evaluator=evaluator))
display(vis.plot_optimization_history(study_lgb))


# Save the LighGBM best parameters
with open('best_params_lgboost.json', 'w') as f:
    json.dump(best_params_lgb, f)


# Train the LighGBM final models on full data
X_lgb = train_data.drop(columns=['diagnosed_diabetes']).copy()
y_lgb = train_data['diagnosed_diabetes']
cat_cols = X_lgb.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")

dtrain = lgb.Dataset(X_lgb, label=y_lgb, categorical_feature=cat_cols)

params = {**best_params_lgb}

print("=" * 100)
print(" Training final LighGBM model on FULL data")
print(f"• Samples: {X_lgb.shape[0]}")
print(f"• Features: {X_lgb.shape[1]}")
print(f"• Categorical features: {len(cat_cols)}")
print(f"• Iterations: {best_iter_lgb}")
print("-" * 80)

final_lgb_model = lgb.train(
    params,
    dtrain,
    num_boost_round=best_iter_lgb,
)
print("LightGBM training completed")
print("Total trees:", final_lgb_model.current_iteration())
print("=" * 100)


# Save LightGBM model
final_lgb_model.save_model("lightgbm_final_model.txt")

# Save metadata
lgb_metadata = {
    "feature_names": X_lgb.columns.tolist(),
    "cat_cols": cat_cols
}

with open("lightgbm_metadata.json", "w") as f:
    json.dump(lgb_metadata, f)


comparison_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)

# Pick the best model (the one with highest ROC-AUC)
best_model_info = comparison_df.iloc[0]
best_model_name = best_model_info['Model']
best_params = best_model_info['Best_Params']

for _, row in comparison_df.iterrows():
    print(f"{row['Model']:10s} | ROC-AUC: {row['ROC-AUC']:.4f} | Best Iteration: {row['Best_Iteration']}")


X_oof = train_data.drop(columns=['diagnosed_diabetes']).copy()
y_oof = train_data['diagnosed_diabetes']
cat_cols = X_oof.select_dtypes(include=['object', "category"]).columns.tolist()

n_folds = 4
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


# Creat OOF predictions for CatBoost
oof_cb = np.zeros(len(X_oof))
fold_aucs = []
best_iter_cbm_oof = []

print("="*100)
print("Classifier started CatBoost")
print("="*100)

cat_features = [X_oof.columns.get_loc(col) for col in cat_cols]

for fold, (train_index, val_index) in enumerate(skf.split(X_oof, y_oof), 1):
    print("Fold number:", fold)

    X_train, X_valid = X_oof.iloc[train_index].copy(), X_oof.iloc[val_index].copy()
    y_train, y_valid = y_oof.iloc[train_index], y_oof.iloc[val_index]

    # New model per fold
    cb_model_oof = CatBoostClassifier(
        **best_params_cb          # Optuna-tuned params + fixed parameters
    )

    cb_model_oof.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        early_stopping_rounds=100
    )

    oof_cb[val_index] = cb_model_oof.predict_proba(
        X_valid,
        ntree_end=cb_model_oof.best_iteration_
    )[:, 1]

    fold_auc = roc_auc_score(y_valid, oof_cb[val_index])
    fold_aucs.append(fold_auc)

    fold_auc = roc_auc_score(y_valid, oof_cb[val_index])
    fold_aucs.append(fold_auc)
    best_iter_cbm_oof.append(cb_model_oof.best_iteration_)

    print(f"Fold {fold} | AUC: {fold_auc:.4f} | Best iter: {cb_model_oof.best_iteration_}")

print("="*100)
print(f"Overall OOF AUC: {roc_auc_score(y_oof, oof_cb):.4f}")
print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
print(f"Mean best_iteration: {int(np.mean(best_iter_cbm_oof))}")
print("="*100)


# Verify OOF predictions
verify_oof_predictions(oof_cb, X_oof, y_oof, skf, cat_features=cat_features)


# Save OOF to file
np.save("oof_cb_preds.npy", oof_cb)


# Creat OOF predictions for XGBoost

oof_xgb = np.zeros(len(X_oof))
fold_aucs = []
best_iterations_xgb = []

print("="*100)
print("Classifier started XGBoost")
print("="*100)

cat_cols = X_oof.select_dtypes(include=["object", "category"]).columns.tolist()

for fold, (train_index, val_index) in enumerate(skf.split(X_oof, y_oof), 1):
    print("Fold number:", fold)

    X_train, X_valid = X_oof.iloc[train_index].copy(), X_oof.iloc[val_index].copy()
    y_train, y_valid = y_oof.iloc[train_index], y_oof.iloc[val_index]

    # TARGET ENCODING (fit ONLY on training fold)
    encoder = TargetEncoder(cols=cat_cols, smoothing=0.3)
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_valid_enc = encoder.transform(X_valid)


    dtrain = xgb.DMatrix(X_train_enc, label=y_train)
    dvalid = xgb.DMatrix(X_valid_enc, label=y_valid)

    params = {
        **best_params_xgb,         # Optuna-tuned params + fixed parameters
        "objective": "binary:logistic",
        "device": "cuda",
        "booster": "gbtree",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": 42
    }

    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    oof_xgb[val_index] = xgb_model.predict(dvalid, iteration_range=(0, xgb_model.best_iteration))

    fold_auc = roc_auc_score(y_valid, oof_xgb[val_index])
    fold_aucs.append(fold_auc)

    best_iterations_xgb.append(xgb_model.best_iteration)

    print(f"Fold {fold} | AUC: {fold_auc:.4f} | Best iter: {xgb_model.best_iteration}")

print("="*100)
print(f"Overall OOF AUC: {roc_auc_score(y_oof, oof_cb):.4f}")
print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
print(f"Mean best_iteration: {int(np.mean(best_iterations_xgb))}")
print("="*100)


# Verify OOF predictions
verify_oof_predictions(oof_xgb, X_oof, y_oof, skf, cat_features=cat_features)


# Save OOF to file
np.save("oof_xgb_preds.npy", oof_xgb)


# Creat OOF predictions for LightGBM

oof_lgb = np.zeros(len(X_oof))
fold_aucs=[]
best_iterations_lgb = []

print("="*100)
print("Classifier started LGBMClassifier")
print("="*100)

cat_cols = X_oof.select_dtypes(include=["object", "category"]).columns.tolist()

for fold, (train_index, val_index) in enumerate(skf.split(X_oof, y_oof), 1):
    print("Fold number:", fold)

    X_train, X_valid = X_oof.iloc[train_index].copy(), X_oof.iloc[val_index].copy()
    y_train, y_valid = y_oof.iloc[train_index], y_oof.iloc[val_index]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols)


    params = {**best_params_lgb}         # Optuna-tuned params + fixed parameters
    
    callbacks = [
        early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=0)
    ]

    gbm = lgb.train(
        params,
        dtrain,
        num_boost_round=best_iter_lgb,
        valid_sets=[dvalid],
        callbacks=callbacks
    )

    oof_lgb[val_index] = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    fold_auc = roc_auc_score(y_valid, oof_lgb[val_index])
    fold_aucs.append(fold_auc)

    best_iterations_lgb.append(gbm.best_iteration)

    print(f"Fold {fold} | AUC: {fold_auc:.4f} | Best iter: {gbm.best_iteration}")

print("="*100)
print(f"Overall OOF AUC: {roc_auc_score(y_oof, oof_lgb):.4f}")
print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
print(f"Mean best_iteration: {int(np.mean(best_iterations_lgb))}")
print("="*100)


# Verify OOF predictions
verify_oof_predictions(oof_lgb, X_oof, y_oof, skf, cat_features=cat_features)


# Save OOF to file
np.save("oof_lgb_preds.npy", oof_lgb)


# Create stacking dataset
stacking_features = pd.DataFrame({
    'lgb_pred': oof_lgb,
    'xgb_pred': oof_xgb,
    'cb_pred': oof_cb
}).reset_index(drop=True)

stacking_target = train_data['diagnosed_diabetes'].reset_index(drop=True)


# Stacking using LogisticRegression
meta_model_lr = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)

meta_model_lr.fit(stacking_features, stacking_target)
stack_auc = roc_auc_score(stacking_target, meta_model_lr.predict_proba(stacking_features)[:, 1])
print(f"Stacked model AUC: {stack_auc:.4f}")


for name, coef in zip(
    ["LightGBM", "XGBoost", "CatBoost"],
    meta_model_lr.coef_[0]
):
    print(f"{name}: {coef:.4f}")


# Verfy the correlation 
stacking_features.corr()


plt.figure(figsize=(6,6))
meta_model_pred = meta_model_lr.predict_proba(stacking_features)[:, 1]

for name, preds in {
    "LightGBM": oof_lgb,
    "XGBoost": oof_xgb,
    "CatBoost": oof_cb,
    "Stacked": meta_model_pred
}.items():
    fpr, tpr, _ = roc_curve(stacking_target, preds)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Save meta oof
np.save("oof_stacked.npy", meta_model_pred)


# Prepare test set stacking features & predict

test_data = load_data('/kaggle/input/playground-series-s5e12/test.csv')
test_data.head()


# show basic info of the dataset
show_data_info(test_data)
# Visualize numerical features
test_data.describe().T


# Drop ID as it has no velue for prediction
test_ids = test_data['id']
test_data.drop(columns='id', inplace=True)


# Drop highly highly correlated features
test_data.drop(columns=['bmi', 'cholesterol_total'], inplace=True)
print(test_data[col].dtype)


# Creat a copy for understanding
X_test = test_data.copy()


# LightGBM predict
cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    if not isinstance(test_data[col].dtype, pd.CategoricalDtype):
        X_test[col] = X_test[col].astype('category')

    X_test[col] = X_test[col].astype('category')
    X_test[col] = X_test[col].cat.set_categories(X_test[col].cat.categories)

# Now predict
test_pred_lgb = final_lgb_model.predict(X_test)


# XGBoost predict
cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    X_test[col] = X_test[col].astype("category")

dtest = xgb.DMatrix(X_test, enable_categorical=True)
test_pred_xgb = final_xgb_model.predict(dtest)


cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    # Ensure train col is categorical dtype
    if not isinstance(X_test[col].dtype, pd.CategoricalDtype):
        X_test[col] = X_test[col].astype('category')

    if not isinstance(X_cbm[col].dtype, pd.CategoricalDtype):
        X_cbm[col] = X_cbm[col].astype('category')

    X_test[col] = X_test[col].cat.set_categories(X_cbm[col].cat.categories)

test_pred_cb = final_cb_model.predict_proba(X_test)[:, 1]


# Stack the predicitions
stacked_test = pd.DataFrame({
    'lgb_pred': test_pred_lgb,
    'xgb_pred': test_pred_xgb,
    'cb_pred': test_pred_cb
})

# Get the final prediction
final_preds = meta_model_lr.predict_proba(stacked_test)[:, 1]


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': final_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

