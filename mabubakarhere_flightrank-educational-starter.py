import pandas as pd
df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/train.parquet")
df.sample(10)  # Random 10 rows



df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/test.parquet")
df.sample(10)  # Random 10 rows



df = pd.read_parquet("/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet")
df.sample(10)  # Random 10 rows



!pip install lightgbm
!pip install polars


# ✅ Optimized Polars + LightGBM Pipeline for AeroClub RecSys 2025
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import time

# --- Configuration ---
# Grouping settings here makes the script easier to manage and tweak.
class CFG:
    # File paths
    TRAIN_PATH = "/kaggle/input/aeroclub-recsys-2025/train.parquet"
    TEST_PATH = "/kaggle/input/aeroclub-recsys-2025/test.parquet"
    SUBMISSION_PATH = "submission.csv"
    TOP3_FLAT_PATH = "top3_choices_flat.csv"
    TOP3_PIVOTED_PATH = "top3_choices_pivoted.csv"

    # Feature lists
    # Defining features centrally to avoid repetition.
    BASE_FEATURES = [
        "totalPrice", "taxes",
        "legs0_duration", "legs1_duration",
        "miniRules0_monetaryAmount", "miniRules0_percentage",
        "miniRules1_monetaryAmount", "miniRules1_percentage",
        "pricingInfo_passengerCount"
    ]
    BOOLEAN_FEATURES = ["sex", "isVip", "isAccess3D", "bySelf", "pricingInfo_isAccessTP"]
    CATEGORICAL_FEATURES = ["frequentFlyer"]
    RANK_COLS = ["totalPrice", "taxes", "legs0_duration", "legs1_duration"]

    # Model parameters
    # LightGBM is often significantly faster than CatBoost for similar performance.
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1500, # Increased estimators, balanced by early stopping
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'class_weight': 'balanced'
    }
    LGBM_FIT_PARAMS = {
        "callbacks": [lgb.early_stopping(100, verbose=True)] # Increased patience
    }

    # Other settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

def feature_engineer(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies all feature engineering steps to the dataframe.
    This function encapsulates the logic to avoid code duplication.
    """
    # 1. Duration Conversion: Fast, idiomatic Polars expression.
    duration_cols = ["legs0_duration", "legs1_duration"]
    duration_expressions = []
    for col in duration_cols:
        duration_expr = pl.col(col).str.split(":")
        conversion_expr = (
            pl.when(duration_expr.list.len() == 3)
            .then(
                duration_expr.list.get(0).cast(pl.Int64, strict=False) * 60
                + duration_expr.list.get(1).cast(pl.Int64, strict=False)
                + duration_expr.list.get(2).cast(pl.Float64, strict=False) / 60
            )
            .otherwise(None) # Use None for failed conversions
            .cast(pl.Float32) # Use Float32 to save memory
            .fill_null(strategy="mean") # Impute with mean for robustness
            .alias(col)
        )
        duration_expressions.append(conversion_expr)

    df = df.with_columns(duration_expressions)

    # 2. Boolean to Integer Conversion
    # Explicitly cast boolean columns to integers using `cast(pl.Int8)`
    df = df.with_columns(
        [pl.col(c).cast(pl.Boolean, strict=False).cast(pl.Int8) for c in CFG.BOOLEAN_FEATURES]
    )

    # 3. Categorical Null Filling and Encoding
    # LightGBM works best with integer-encoded categoricals.
    df = df.with_columns(
        [pl.col(c).fill_null("None").cast(pl.Categorical) for c in CFG.CATEGORICAL_FEATURES]
    )

    # 4. Rank Features
    # This creates ranking within each group, a powerful feature.
    rank_expressions = []
    for col in CFG.RANK_COLS:
        rank_expressions.append(
            pl.col(col).rank(method='ordinal').over("ranker_id").alias(f"rank_{col}")
        )
    df = df.with_columns(rank_expressions)

    return df

def main():
    """Main function to run the entire pipeline."""
    start_time = time.time()

    # ========== Step 1: Load Train & Test Data ==========
    print("Step 1: Loading data...")
    train_df = pl.read_parquet(CFG.TRAIN_PATH)
    test_df = pl.read_parquet(CFG.TEST_PATH).with_row_index(name="row_order")
    print(f"Data loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ========== Step 2: Feature Engineering ==========
    print("Step 2: Performing feature engineering...")
    train_df = feature_engineer(train_df)
    test_df = feature_engineer(test_df)

    # Combine feature lists, including boolean features
    all_features = CFG.BASE_FEATURES + CFG.BOOLEAN_FEATURES + [f"rank_{col}" for col in CFG.RANK_COLS]

    # Ensure selected features exist in the dataframe
    missing_train_features = [f for f in all_features if f not in train_df.columns]
    missing_test_features = [f for f in all_features if f not in test_df.columns]

    if missing_train_features:
        print(f"Warning: Missing features in train_df: {missing_train_features}")
    if missing_test_features:
         print(f"Warning: Missing features in test_df: {missing_test_features}")

    all_features = [f for f in all_features if f in train_df.columns and f in test_df.columns]


    # Find categorical feature indices for LightGBM
    categorical_feature_indices = [all_features.index(col) for col in CFG.CATEGORICAL_FEATURES if col in all_features]

    print("Feature engineering complete.")

    # ========== Step 3: Prepare Data for Model ==========
    print("Step 3: Preparing data for LightGBM...")
    # Create labels
    train_df = train_df.with_columns(pl.col("selected").cast(pl.Int8).alias("label"))

    # Split data into train and validation sets as Polars DataFrames
    train_data, val_data = train_test_split(
        train_df.select(all_features + ["label"]), test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_STATE, stratify=train_df["label"]
    )

    # Convert Polars DataFrames to NumPy arrays for LightGBM
    X_train = train_data.select(all_features).to_numpy()
    y_train = train_data.select("label").to_numpy().flatten()
    X_val = val_data.select(all_features).to_numpy()
    y_val = val_data.select("label").to_numpy().flatten()

    print(f"Data prepared. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # ========== Step 4: Train LightGBM ==========
    print("Step 4: Training LightGBM model...")
    model = lgb.LGBMClassifier(**CFG.LGBM_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        callbacks=CFG.LGBM_FIT_PARAMS['callbacks'],
        # categorical_feature=[all_features.index(col) for col in CFG.CATEGORICAL_FEATURES if col in all_features], # Pass indices, not names
        feature_name=all_features # Pass feature names as a list
    )
    print("Model training complete.")

    # ========== Step 5: Predict & Rank ==========
    print("Step 5: Predicting on test set and ranking...")
    # Predict probabilities for the positive class (class 1) using the Polars DataFrame directly
    predicted_probs = model.predict_proba(test_df.select(all_features))[:, 1]

    test_df = test_df.with_columns(pl.Series(name="predicted_prob", values=predicted_probs))

    # Rank results based on prediction probability
    ranked_df = test_df.sort(["ranker_id", "predicted_prob"], descending=[False, True])
    ranked_df = ranked_df.with_columns(
        (pl.int_range(0, pl.len()).over("ranker_id") + 1).alias("selected")
    )
    print("Prediction and ranking complete.")

    # ========== Step 6: Output Files ==========
    print("Step 6: Generating output files...")
    # Create main submission file
    submission = ranked_df.sort("row_order").select(["Id", "ranker_id", "selected"])
    submission.write_csv(CFG.SUBMISSION_PATH)

    # Create file with top 3 choices
    top_k = ranked_df.filter(pl.col("selected").is_in([1, 2, 3]))

    choice_rank_expr = (
        pl.when(pl.col("selected") == 1).then(pl.lit("best"))
          .when(pl.col("selected") == 2).then(pl.lit("second_best"))
          .when(pl.col("selected") == 3).then(pl.lit("third_best"))
          .otherwise(pl.lit("other"))
          .alias("choice_rank")
    )

    top_k = top_k.with_columns(choice_rank_expr.cast(pl.Categorical))
    top_k.select(["Id", "ranker_id", "choice_rank", "predicted_prob"]).write_csv(CFG.TOP3_FLAT_PATH)

    # Create pivoted top 3 choices file
    pivoted = top_k.pivot(values="Id", index="ranker_id", on="choice_rank") # Changed columns to on
    pivoted.write_csv(CFG.TOP3_PIVOTED_PATH)

    total_time = time.time() - start_time
    print(f"All done! Submission files are ready. Total runtime: {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()


df = pd.read_csv("/kaggle/working/submission.csv")
df.sample(10)  # Random 10 rows


