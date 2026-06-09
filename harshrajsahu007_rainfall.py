# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
)  # Import LogisticRegression
from sklearn.metrics import roc_auc_score
import optuna

warnings.filterwarnings("ignore")



RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
TARGET_COLUMN = "rainfall"

TRAIN_FILE = "/kaggle/input/playground-series-s5e3/train.csv"
TEST_FILE = "/kaggle/input/playground-series-s5e3/test.csv"
SAMPLE_SUBMISSION_FILE = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
OUTPUT_DIRECTORY = "results"
MODELS_DIRECTORY = "models"

# Create output and models directories
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(MODELS_DIRECTORY, exist_ok=True)


def load_data(train_path, test_path, sample_submission_path=None):
    """Loads training, test, and sample submission files, returns DataFrames."""
    print("=== [1] Data Loading ===")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = None

    if sample_submission_path and os.path.exists(
        sample_submission_path
    ):
        sample_submission_df = pd.read_csv(sample_submission_path)
        print(
            f"Training set dimensions: {train_df.shape}, Test set dimensions: {test_df.shape}, Sample submission file dimensions: {sample_submission_df.shape}"
        )
    else:
        print(
            f"Training set dimensions: {train_df.shape}, Test set dimensions: {test_df.shape}"
        )

    print("\nFirst 5 rows of training set:")
    print(train_df.head())
    if sample_submission_df is not None:
        print("\nFirst 5 rows of sample submission file:")
        print(sample_submission_df.head())

    return train_df, test_df, sample_submission_df


def basic_preprocessing(train_df, test_df, target_col):
    """Performs basic preprocessing on train and test sets (missing value handling, boolean column conversion)."""
    print("\n=== [2] Data Preprocessing ===")

    # Check missing values
    print("\nMissing values in training set:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
    print("\nMissing values in test set:")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

    # Fill missing values in numeric columns with median
    numeric_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_value)

    # Fill missing values in categorical columns with mode
    categorical_cols = train_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode(dropna=True)[
            0
        ]  # Get the first mode
        train_df[col] = train_df[col].fillna(mode_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(mode_value)

    # Convert boolean columns
    bool_mapping = {
        "Yes": True,
        "No": False,
        "yes": True,
        "no": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
    }
    for col in categorical_cols:
        unique_values = train_df[col].dropna().unique()
        if all(
            str(val).lower() in bool_mapping for val in unique_values
        ):
            print(f"Detected boolean column {col}, converting...")
            train_df[col] = train_df[col].map(bool_mapping)
            if col in test_df.columns:
                test_df[col] = test_df[col].map(bool_mapping)

    print("Basic preprocessing completed!")
    return train_df, test_df


def feature_engineering(train_df, test_df):
    """Performs feature engineering on train and test sets, generating new features."""
    print("\n=== [2.5] Feature Engineering ===")
    combined_data = pd.concat(
        [train_df, test_df], ignore_index=True, sort=False
    )

    # Log transformation for windspeed (avoiding zero values)
    combined_data["windspeed_log"] = np.log1p(
        combined_data["windspeed"]
    )  # log1p = log(1 + x)

    # Temperature-related features
    # combined_data["temp_range"] = (
    #     combined_data["maxtemp"] - combined_data["mintemp"]
    # )
    combined_data["avg_temp"] = (
        combined_data["maxtemp"] + combined_data["mintemp"]
    ) / 2

    # Cloud-related features
    combined_data["cloud_sun_ratio"] = combined_data["cloud"] / (
        combined_data["sunshine"] + 1e-6
    )
    combined_data["cloud_humidity_interaction"] = (
        combined_data["cloud"] * combined_data["humidity"]
    )
    combined_data["cloud_temp_interaction"] = (
        combined_data["cloud"] * combined_data["avg_temp"]
    )
    combined_data["cloud_wind_interaction"] = (
        combined_data["cloud"] * combined_data["windspeed_log"]
    )
    combined_data["cloud_pressure_ratio"] = combined_data["cloud"] / (
        combined_data["pressure"] + 1e-6
    )
    combined_data["cloud_temp_diff"] = (
        combined_data["cloud"] - combined_data["avg_temp"]
    )

    # Dew point extended feature
    combined_data["dew_point_spread"] = (
        combined_data["temparature"] - combined_data["dewpoint"]
    )

    # Interaction features
    combined_data["pressure_humidity_interaction"] = (
        combined_data["pressure"] * combined_data["humidity"]
    )
    combined_data["wind_cloud_interaction"] = (
        combined_data["windspeed_log"] * combined_data["cloud"]
    )

    # Temperature ratio
    combined_data["temp_ratio"] = (
        combined_data["temparature"] / combined_data["maxtemp"].max()
    )

    # Split back into training and test sets
    train_df = combined_data.iloc[: len(train_df)].copy()
    test_df = (
        combined_data.iloc[len(train_df) :]
        .copy()
        .drop(columns=["rainfall"], errors="ignore")
    )

    # Handle missing values in new features
    numeric_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_value)

    print(
        f"Training set dimensions after feature engineering: {train_df.shape}, Test set dimensions: {test_df.shape}"
    )
    return train_df, test_df


def tune_rf(X_train, y_train, X_val, y_val, n_trials=50):
    """Tunes RandomForestClassifier using Optuna."""

    def objective(trial):
        rf_params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000
            ),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy"]
            ),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        return auc_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best RF parameters: {best_params}")
    best_rf_model = RandomForestClassifier(
        **best_params, random_state=RANDOM_SEED, n_jobs=-1
    )
    best_rf_model.fit(X_train, y_train)
    return best_rf_model


def tune_xgb(X_train, y_train, X_val, y_val, n_trials=50):
    """Tunes XGBClassifier using Optuna."""

    def objective(trial):
        xgb_params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        xgb_model = XGBClassifier(
            **xgb_params, use_label_encoder=False, eval_metric="logloss"
        )
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        return auc_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best XGBoost parameters: {best_params}")
    best_xgb_model = XGBClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    best_xgb_model.fit(X_train, y_train)
    return best_xgb_model


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=50):
    """Tunes CatBoostClassifier using Optuna."""

    def objective(trial):
        cat_params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_state": RANDOM_SEED,
            "verbose": 0,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
        }
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=0,
        )
        y_pred_proba = cat_model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        return auc_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best CatBoost parameters: {best_params}")
    best_cat_model = CatBoostClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        verbose=0,
        loss_function="Logloss",
        eval_metric="AUC",
    )
    best_cat_model.fit(X_train, y_train)
    return best_cat_model


def train_stacked_model(train_df, target_col, eval_metric="roc_auc"):
    """Trains individual models with Optuna, then stacks them using StackingClassifier."""  
    print("\n=== [3] Model Training ===")
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print("Tuning and training RandomForest...")
    rf_model = tune_rf(X_train, y_train, X_val, y_val)
    print("Tuning and training XGBoost...")
    xgb_model = tune_xgb(X_train, y_train, X_val, y_val)
    print("Tuning and training CatBoost...")
    cat_model = tune_catboost(X_train, y_train, X_val, y_val)

    print("\nTraining StackingClassifier model...")
    estimators = [
        ("rf", rf_model),
        ("xgb", xgb_model),
        ("cat", cat_model),
    ]
    # Using LogisticRegression 
    stacked_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
    )
    stacked_model.fit(X_train, y_train)

    print("\nModel training completed!")
    print("=== Validation Set Evaluation ===")
    stacked_val_pred_proba = stacked_model.predict(X_val)
    auc_val_stacked = roc_auc_score(y_val, stacked_val_pred_proba)
    print(f"Stacked Model Validation AUC: {auc_val_stacked}")

    return (
        stacked_model,
        X_train.columns.tolist(),
    )


def predict(
    stacked_model,
    test_df,
    target_col,
    sample_submission_df=None,
    feature_names=None,
    prediction_type="prob"
):
    """Uses trained StackingClassifier model to predict probabilities on test set."""  
    print("\n=== [4] Test Set Prediction ===")
    test_df_processed = (
        test_df[feature_names] if feature_names else test_df
    ) 

    y_pred_proba = stacked_model.predict_proba(test_df_processed)[
        :, 1
    ] 

    return y_pred_proba


train_df, test_df, sample_submission_df = load_data(
    TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE
)
extra_data = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")  
extra_data["rainfall"] = extra_data["rainfall"].apply(
    lambda x: 1 if x == "yes" else 0
) 
train_df = pd.concat(
    [train_df, extra_data], ignore_index=True, sort=False
)


# 2. Basic Preprocessing
train_df, test_df = basic_preprocessing(
    train_df, test_df, TARGET_COLUMN
)

# 2.5 Feature Engineering
train_df, test_df = feature_engineering(train_df, test_df)

# Remove unnecessary columns
train_df = train_df.drop(columns=["id"], errors="ignore")
test_df = test_df.drop(columns=["id"], errors="ignore")


stacked_model, feature_names = train_stacked_model(
    train_df, TARGET_COLUMN, eval_metric="roc_auc"
)


submission = predict(
    stacked_model,
    test_df,
    TARGET_COLUMN,
    sample_submission_df,
    feature_names=feature_names,
    prediction_type="prob",  # Now directly probability from StackingClassifier
)
submission 


sample_sub = pd.read_csv(SAMPLE_SUBMISSION_FILE)
sample_sub["rainfall"] = submission


sample_sub.to_csv("submission.csv", index=False)


df=pd.read_csv("submission.csv")


df.head()

