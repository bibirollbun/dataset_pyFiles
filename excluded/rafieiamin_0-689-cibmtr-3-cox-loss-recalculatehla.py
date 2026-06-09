!pip install --quiet catboost lightgbm polars pandas numpy scikit-learn


import warnings
import time
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
#from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.stats import rankdata
from typing import Dict, List, Tuple
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# Load dataset
file_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"  # Update with actual file path
df = pd.read_csv(file_path)


# Display dataset shape
print(f"Dataset contains {df.shape[0]:,} entries and {df.shape[1]} columns.\n")


# Identify target variables
target_vars = ['efs', 'efs_time']
print(f"Target variables in EDA: {', '.join(target_vars)}\n")


# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Categorical variables: {', '.join(categorical_cols[:5])}... ({len(categorical_cols)} total)")
print(f"Numerical variables: {', '.join(numerical_cols[:5])}... ({len(numerical_cols)} total)\n")


# Count missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Variables with missing values:")
print(missing_values.head(10))  # Show top 10 variables with most missing values


# Identify key numerical variables related to HLA matching and clinical scores
key_numerical_features = ["hla_match_c_high", "donor_age", "karnofsky_score", "conditioning_intensity"]
print(f"\nKey numerical variables: {', '.join([col for col in key_numerical_features if col in numerical_cols])}")



# Set style for plots
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))


# Plot distribution of the target variable `efs`
sns.countplot(x=df["efs"], palette="coolwarm")
plt.title("Distribution of Event-Free Survival (efs)", fontsize=14)
plt.xlabel("Event-Free Survival (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


# Plot distribution of `efs_time`
plt.figure(figsize=(10, 6))
sns.histplot(df["efs_time"], bins=50, kde=True, color="teal")
plt.title("Distribution of Event-Free Survival Time (efs_time)", fontsize=14)
plt.xlabel("Time to Event-Free Survival")
plt.ylabel("Count")
plt.show()



# Plot distribution of race groups
plt.figure(figsize=(12, 6))
sns.countplot(y=df["race_group"], palette="viridis", order=df["race_group"].value_counts().index)
plt.title("Distribution of Patients by Race Group", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Race Group")
plt.show()


# Boxplot of efs_time by race group
plt.figure(figsize=(12, 6))
sns.boxplot(x="race_group", y="efs_time", data=df, palette="Set2")
plt.title("Event-Free Survival Time by Race Group", fontsize=14)
plt.xlabel("Race Group")
plt.ylabel("Event-Free Survival Time")
plt.xticks(rotation=45)
plt.show()



# Scatter plot of Age vs. Event-Free Survival Time
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["age_at_hct"], y=df["efs_time"], hue=df["efs"], alpha=0.5, palette="coolwarm")
plt.title("Age vs. Event-Free Survival Time", fontsize=14)
plt.xlabel("Age at HCT")
plt.ylabel("Event-Free Survival Time")
plt.show()


# Boxplot of comorbidity score vs. efs_time
plt.figure(figsize=(10, 6))
sns.boxplot(x="comorbidity_score", y="efs_time", data=df, palette="magma")
plt.title("Comorbidity Score vs. Event-Free Survival Time", fontsize=14)
plt.xlabel("Comorbidity Score")
plt.ylabel("Event-Free Survival Time")
plt.xticks(rotation=45)
plt.show()



# KDE Plot of Survival Time by Event-Free Status
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df["efs"] == 1]["efs_time"], shade=True, label="Survived", color="green")
sns.kdeplot(df[df["efs"] == 0]["efs_time"], shade=True, label="Not Survived", color="red")
plt.title("Kernel Density Estimation of Event-Free Survival Time", fontsize=14)
plt.xlabel("Event-Free Survival Time")
plt.ylabel("Density")
plt.legend()
plt.show()


# Correlation heatmap of numerical variables
plt.figure(figsize=(12, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()


class CFG:

    batch_size = 32768
    early_stop = 100
    penalizer = 0.01
    n_splits = 5

    # CatBoost parameters
    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.05,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 3000,
        'subsample': 0.8,
        'reg_lambda': 3.0,
        'depth': 8,
        'thread_count': 4
    }

    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 20,
        'num_iterations': 3000,
        'learning_rate': 0.03,
        'reg_lambda': 3.0,
        'num_leaves': 48,
        'metric': 'rmse',
        'max_depth': 8,
        'device': 'cpu',
        'num_threads': 4,
        'verbose': -1,
        'seed': 42
    }

    # Additional CatBoost model with different parameters
    ctb_params2 = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 3000,
        'subsample': 0.7,
        'reg_lambda': 5.0,
        'depth': 9,
        'thread_count': 4,
        'bootstrap_type': 'Bernoulli'
    }

    # Additional LightGBM model with different parameters
    lgb_params2 = {
        'objective': 'regression',
        'min_child_samples': 15,
        'num_iterations': 3000,
        'learning_rate': 0.02,
        'reg_lambda': 4.0,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 9,
        'device': 'cpu',
        'num_threads': 4,
        'verbose': -1,
        'seed': 42,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }


class FeatureEngineering:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def load_data(self, path: Path) -> pl.DataFrame:
        return pl.read_csv(path, batch_size=self.batch_size)

    def recalculate_hla_sums(self, df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns([
            (pl.col("hla_match_a_low").fill_null(0) +
             pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_nmdp_6"),

            (pl.col("hla_match_a_low").fill_null(0) +
             pl.col("hla_match_b_low").fill_null(0) +
             pl.col("hla_match_drb1_low").fill_null(0)).alias("hla_low_res_6"),

            (pl.col("hla_match_a_high").fill_null(0) +
             pl.col("hla_match_b_high").fill_null(0) +
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_high_res_6"),
        ])
        return df

    def cast_datatypes(self, df: pl.DataFrame) -> pl.DataFrame:

        # Define numeric columns
        num_cols = [
            'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6',
            'hla_low_res_6', 'hla_high_res_10', 'hla_low_res_10',
            'hla_match_dqb1_high', 'hla_match_dqb1_low',
            'hla_match_drb1_high', 'hla_match_drb1_low',
            'hla_nmdp_6', 'year_hct', 'hla_match_a_high',
            'hla_match_a_low', 'hla_match_b_high', 'hla_match_b_low',
            'hla_match_c_high', 'hla_match_c_low', 'donor_age',
            'age_at_hct', 'comorbidity_score', 'karnofsky_score',
            'efs', 'efs_time'
        ]

        # Handle numeric columns
        for col in num_cols:
            if col in df.columns:
                # Convert to string first to handle any binary data
                df = df.with_columns(pl.col(col).cast(pl.String))
                # Replace problematic values
                df = df.with_columns(
                    pl.when(pl.col(col).str.contains("N/A"))
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                # Convert to float, replacing invalid values with null
                df = df.with_columns(pl.col(col).cast(pl.Float32, strict=False))
                # Fill null values with column median or -1 if all null
                df = df.with_columns(
                    pl.col(col).fill_null(
                        pl.col(col).median().fill_null(-1)
                    )
                )

        # Handle categorical columns
        cat_cols = [col for col in df.columns if col not in num_cols and col != 'ID']
        for col in cat_cols:
            df = df.with_columns(
                pl.col(col)
                .fill_null("Unknown")
                .cast(pl.String)
            )

        # Handle ID column
        df = df.with_columns(pl.col('ID').cast(pl.Int32))

        return df

    def apply_fe(self, path: Path) -> Tuple[pd.DataFrame, List[str]]:
        #Apply all feature engineering steps.
        df = self.load_data(path)
        df = self.recalculate_hla_sums(df)
        df = self.cast_datatypes(df)
        df = df.to_pandas()

        cat_cols = [col for col in df.columns if isinstance(df[col].dtype, pd.CategoricalDtype)]
        return df, cat_cols


class ModelDevelopment:

    def __init__(self, early_stop: int, penalizer: float, n_splits: int):
        self.early_stop = early_stop
        self.penalizer = penalizer
        self.n_splits = n_splits

    def create_survival_targets(self, data: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:

        try:
            print("   > Attempting to create Cox-based targets...")
            # Prepare data for Cox model
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            cph_data = data.copy()

            # Handle numeric columns
            for col in numeric_cols:
                if cph_data[col].dtype in ['int64', 'float64']:
                    median_val = pd.to_numeric(cph_data[col], errors='coerce').median()
                    cph_data[col] = pd.to_numeric(cph_data[col], errors='coerce').fillna(median_val)

            # Create dummy variables for categorical columns
            cph_data = pd.get_dummies(cph_data, columns=cat_cols, drop_first=True)

            # Ensure all columns are numeric
            for col in cph_data.columns:
                if cph_data[col].dtype not in ['int64', 'float64']:
                    cph_data[col] = pd.to_numeric(cph_data[col], errors='coerce').fillna(0)

            # Fit Cox model
            cph = CoxPHFitter(penalizer=self.penalizer)
            cph.fit(cph_data, duration_col='efs_time', event_col='efs')
            data['target1'] = cph.predict_partial_hazard(cph_data)

        except Exception as e:
            print(f"   ! Cox model failed: {e}")
            print("   > Creating alternative target1...")
            # Create a simple risk score based on time and event
            data['target1'] = data['efs_time'] * (2 * data['efs'] - 1)

        # Create standard survival target
        data['target2'] = data['efs_time'] * data['efs']

        # Standardize targets
        scaler = StandardScaler()
        data['target1'] = scaler.fit_transform(data[['target1']])
        data['target2'] = scaler.fit_transform(data[['target2']])

        return data

    def train_model(self, data: pd.DataFrame, cat_cols: List[str],
                   params: Dict, model_type: str, target: str) -> Tuple[List, npt.NDArray]:
        """Train a model with cross-validation."""
        # Prepare features
        feature_cols = [col for col in data.columns
                       if col not in ['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3']]
        X = data[feature_cols].copy()
        y = data[target].copy()

        # Convert categorical columns to category type
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        # Handle numeric columns
        numeric_cols = [col for col in X.columns if col not in cat_cols]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

        models = []
        oof_preds = np.zeros(len(X))
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        successful_folds = 0
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X)):
            print(f"      > Training fold {fold + 1}/{self.n_splits}")
            X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()

            try:
                if model_type == 'lgb':
                    categorical_feature = [col for col in cat_cols if col in X.columns]
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        callbacks=[lgb.early_stopping(self.early_stop)],
                        categorical_feature=categorical_feature
                    )
                else:  # catboost
                    categorical_features = [col for col in cat_cols if col in X.columns]
                    model = CatBoostRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        cat_features=categorical_features,
                        early_stopping_rounds=self.early_stop,
                        verbose=False
                    )

                models.append(model)
                oof_preds[valid_idx] = model.predict(X_valid)
                successful_folds += 1

            except Exception as e:
                print(f"      ! Error in fold {fold + 1}: {str(e)}")
                continue

        if successful_folds < 1:
            print("      ! No folds were successfully trained")
            return [], oof_preds

        print(f"      > Successfully trained {successful_folds} out of {self.n_splits} folds")
        return models, oof_preds

    def predict(self, data: pd.DataFrame, cat_cols: List[str], models: List) -> npt.NDArray:
        """Generate predictions from trained models."""
        # Prepare features
        feature_cols = [col for col in data.columns
                       if col not in ['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3']]
        X = data[feature_cols].copy()

        # Convert categorical columns to category type
        for col in cat_cols:
            X[col] = X[col].astype('category')

        # Handle numeric columns
        numeric_cols = [col for col in X.columns if col not in cat_cols]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

        # Generate predictions
        return np.mean([model.predict(X) for model in models], axis=0)


def main():
    print("\n" + "="*50)
    print("Starting Training Pipeline")
    print("="*50 + "\n")

    start_time = time.time()

    # Initialize classes
    fe = FeatureEngineering(CFG.batch_size)
    md = ModelDevelopment(CFG.early_stop, CFG.penalizer, CFG.n_splits)
    # Load and prepare data
    print("1. Loading and preparing data...")
    train_data, cat_cols = fe.apply_fe(Path('/kaggle/input/equity-post-HCT-survival-predictions/train.csv'))
    test_data, _ = fe.apply_fe(Path('/kaggle/input/equity-post-HCT-survival-predictions/test.csv'))
    print(f"   ✓ Loaded {len(train_data)} training samples and {len(test_data)} test samples")

    # Create targets
    print("\n2. Creating targets...")
    train_data = md.create_survival_targets(train_data, cat_cols)
    print("   ✓ Successfully created and standardized targets")

    # Train models
    print("\n3. Training models...")
    model_predictions = {}

    # Train first CatBoost model
    print("\n   3.1 Training CatBoost model (configuration 1)...")
    try:
        ctb_models, ctb_oof = md.train_model(train_data, cat_cols, CFG.ctb_params, 'catboost', 'target1')
        if ctb_models:
            ctb_preds = md.predict(test_data, cat_cols, ctb_models)
            model_predictions['CatBoost1'] = ctb_preds
            rmse = np.sqrt(np.mean((ctb_oof - train_data['target1'])**2))
            print(f"      ✓ CatBoost1 training successful (RMSE: {rmse:.4f})")
    except Exception as e:
        print(f"      ! CatBoost1 training failed: {e}")

    # Train second CatBoost model
    print("\n   3.2 Training CatBoost model (configuration 2)...")
    try:
        ctb_models2, ctb_oof2 = md.train_model(train_data, cat_cols, CFG.ctb_params2, 'catboost', 'target2')
        if ctb_models2:
            ctb_preds2 = md.predict(test_data, cat_cols, ctb_models2)
            model_predictions['CatBoost2'] = ctb_preds2
            rmse = np.sqrt(np.mean((ctb_oof2 - train_data['target2'])**2))
            print(f"      ✓ CatBoost2 training successful (RMSE: {rmse:.4f})")
    except Exception as e:
        print(f"      ! CatBoost2 training failed: {e}")

    # Train first LightGBM model
    print("\n   3.3 Training LightGBM model (configuration 1)...")
    try:
        lgb_models, lgb_oof = md.train_model(train_data, cat_cols, CFG.lgb_params, 'lgb', 'target1')
        if lgb_models:
            lgb_preds = md.predict(test_data, cat_cols, lgb_models)
            model_predictions['LightGBM1'] = lgb_preds
            rmse = np.sqrt(np.mean((lgb_oof - train_data['target1'])**2))
            print(f"      ✓ LightGBM1 training successful (RMSE: {rmse:.4f})")
    except Exception as e:
        print(f"      ! LightGBM1 training failed: {e}")

    # Train second LightGBM model
    print("\n   3.4 Training LightGBM model (configuration 2)...")
    try:
        lgb_models2, lgb_oof2 = md.train_model(train_data, cat_cols, CFG.lgb_params2, 'lgb', 'target2')
        if lgb_models2:
            lgb_preds2 = md.predict(test_data, cat_cols, lgb_models2)
            model_predictions['LightGBM2'] = lgb_preds2
            rmse = np.sqrt(np.mean((lgb_oof2 - train_data['target2'])**2))
            print(f"      ✓ LightGBM2 training successful (RMSE: {rmse:.4f})")
    except Exception as e:
        print(f"      ! LightGBM2 training failed: {e}")

    # Check if we have enough models for ensemble
    if len(model_predictions) == 0:
        raise ValueError("No models were successfully trained")

    # Create ensemble predictions
    print(f"\n4. Creating ensemble predictions...")
    all_preds = list(model_predictions.values())
    model_names = list(model_predictions.keys())
    print(f"   ✓ Creating ensemble from {len(model_predictions)} models: {', '.join(model_names)}")

    # Rank transform predictions
    ranked_preds = np.array([rankdata(p) for p in all_preds])
    ensemble_preds = np.mean(ranked_preds, axis=0)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'prediction': ensemble_preds
    })
    submission.to_csv('submission.csv', index=False)

    # Final summary
    end_time = time.time()
    training_time = (end_time - start_time)/60

    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Total training time: {training_time:.2f} minutes")
    print(f"Models trained successfully: {len(model_predictions)}")
    print(f"Models used in ensemble: {', '.join(model_names)}")
    print(f"Submission file created: submission.csv")
    print(f"Predictions generated for {len(test_data)} test samples")
    print("="*50)

if __name__ == "__main__":
    main()

