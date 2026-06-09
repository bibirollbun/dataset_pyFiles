import polars as pl
import numpy as np
from catboost import CatBoostRegressor, Pool
import os

# Configuration
DATA_PATH = "/kaggle/input/china-real-estate-demand-prediction"
RANDOM_SEED = 42

print("Loading libraries and setting up environment...")

def load_and_preprocess_data(base_path):
    """Load all datasets with proper error handling and preprocessing"""
    datasets = {}
    
    # Define file paths and their specific handling
    file_configs = {
        'city_indexes': {'path': '/train/city_indexes.csv', 'drop_cols': ['total_fixed_asset_investment_10k']},
        'city_search_index': {'path': '/train/city_search_index.csv'},
        'sector_POI': {'path': '/train/sector_POI.csv'},
        'land_transactions': {'path': '/train/land_transactions.csv', 'schema': {
            'sector': pl.Utf8, 'month': pl.Utf8, 'land_area': pl.Float64,
            'construction_area': pl.Float64, 'price': pl.Float64, 'price_per_square': pl.Float64
        }},
        'land_transactions_nearby': {'path': '/train/land_transactions_nearby_sectors.csv'},
        'pre_owned_transactions': {'path': '/train/pre_owned_house_transactions.csv'},
        'pre_owned_nearby': {'path': '/train/pre_owned_house_transactions_nearby_sectors.csv'},
        'new_house_transactions': {'path': '/train/new_house_transactions.csv'},
        'new_house_nearby': {'path': '/train/new_house_transactions_nearby_sectors.csv'}
    }
    
    for name, config in file_configs.items():
        try:
            file_path = base_path + config['path']
            if 'schema' in config:
                df = pl.read_csv(file_path, schema_overrides=config['schema'])
            else:
                df = pl.read_csv(file_path, infer_schema_length=10000)
            
            # Apply dataset-specific preprocessing
            if 'drop_cols' in config:
                df = df.drop(config['drop_cols'])
            
            # Add prefix to columns (except sector and month)
            if name != 'city_search_index':
                prefix_map = {
                    'city_indexes': 'ci_', 'sector_POI': 'sp_', 'land_transactions': 'lt_',
                    'land_transactions_nearby': 'ltns_', 'pre_owned_transactions': 'pht_',
                    'pre_owned_nearby': 'phtns_', 'new_house_transactions': 'nht_',
                    'new_house_nearby': 'nhtns_'
                }
                if name in prefix_map:
                    prefix = prefix_map[name]
                    rename_dict = {col: col if col in ['sector', 'month'] else f"{prefix}{col}" 
                                 for col in df.columns}
                    df = df.rename(rename_dict)
            
            df = df.fill_null(-1)
            datasets[name] = df
            print(f"âœ“ Loaded {name}: {df.shape}")
            
        except Exception as e:
            print(f"âœ— Error loading {name}: {e}")
            # Fallback with ignore_errors
            df = pl.read_csv(base_path + config['path'], ignore_errors=True).fill_null(-1)
            datasets[name] = df
            print(f"  Loaded with fallback: {df.shape}")
    
    return datasets

def create_time_features(base_df):
    """Create comprehensive time-based features"""
    month_codes = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    return base_df.with_columns([
        pl.col("month").str.split("-").list.get(0).cast(pl.Int16).alias("year"),
        pl.col("month").str.split("-").list.get(1).replace(month_codes).cast(pl.Int8).alias("month_num")
    ]).with_columns([
        ((pl.col("year") - 2019) * 12 + pl.col("month_num") - 1).cast(pl.Int16).alias("time"),
        ((pl.col("month_num") - 1) / 6 * np.pi).cos().alias("season_cos"),
        ((pl.col("month_num") - 1) / 6 * np.pi).sin().alias("season_sin"),
        ((pl.col("month_num") - 1) / 3 * np.pi).cos().alias("season_cos2"),
        ((pl.col("month_num") - 1) / 3 * np.pi).sin().alias("season_sin2"),
        pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8).alias("sector_id")
    ])

def build_feature_dataset(datasets):
    """Build comprehensive feature dataset by joining all sources"""
    print("Building feature dataset...")
    
    # Create base grid of all sectors and months
    months = datasets['new_house_transactions']["month"].unique().sort()
    sectors = pl.concat([
        datasets['new_house_transactions']["sector"].unique(),
        pl.Series(["sector 95"])
    ]).unique()
    
    base_df = pl.DataFrame({"month": months}).join(
        pl.DataFrame({"sector": sectors}), how="cross"
    )
    
    # Add time features
    base_df = create_time_features(base_df)
    
    # Define join order
    join_order = [
        'new_house_transactions', 'new_house_nearby', 'pre_owned_transactions',
        'pre_owned_nearby', 'land_transactions', 'land_transactions_nearby'
    ]
    
    # Join all datasets
    for dataset_name in join_order:
        if dataset_name in datasets:
            base_df = base_df.join(
                datasets[dataset_name], on=["sector", "month"], how="left"
            ).fill_null(-1)
            print(f"  Joined {dataset_name}")
    
    # Join city indexes and POI data
    base_df = base_df.join(
        datasets['city_indexes'].rename({"ci_city_indicator_data_year": "year"}),
        on="year", how="left"
    ).fill_null(-1)
    
    base_df = base_df.join(
        datasets['sector_POI'], on="sector", how="left"
    ).fill_null(-1)
    
    return base_df

def optimize_memory_usage(df):
    """Optimize data types to reduce memory footprint"""
    print("Optimizing memory usage...")
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if dtype == pl.Int64:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min == 0 and c_max == 0:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
                
        elif dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
    
    return df

def create_advanced_features(df):
    """Create advanced features including lagged and rolling features"""
    print("Creating advanced features...")
    
    # Sort by sector and time for proper shifting
    df = df.sort(["sector_id", "time"])
    
    # Create lagged features for key columns
    target_col = "nht_amount_new_house_transactions"
    lag_periods = [1, 2, 3, 6, 12]
    
    for lag in lag_periods:
        df = df.with_columns([
            pl.col(target_col).shift(lag).over("sector_id").alias(f"target_lag_{lag}"),
            pl.col("nht_area_new_house_transactions").shift(lag).over("sector_id").alias(f"area_lag_{lag}"),
        ])
    
    # Create rolling statistics
    for window in [3, 6]:
        df = df.with_columns([
            pl.col(target_col).rolling_mean(window_size=window).over("sector_id").alias(f"target_roll_mean_{window}"),
            pl.col(target_col).rolling_std(window_size=window).over("sector_id").alias(f"target_roll_std_{window}"),
        ])
    
    return df

def prepare_model_data(features_df):
    """Prepare data for model training with proper target creation"""
    print("Preparing model data...")
    
    # Create target variable (next month's value)
    features_df = features_df.with_columns(
        pl.col("nht_amount_new_house_transactions")
        .shift(-1)
        .over("sector_id")
        .alias("target")
    )
    
    # Add advanced features
    features_df = create_advanced_features(features_df)
    
    # Remove columns with no variance or all nulls
    cols_to_drop = []
    for col in features_df.columns:
        if features_df[col].null_count() == len(features_df):
            cols_to_drop.append(col)
        elif features_df[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            if features_df[col].std() == 0:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        features_df = features_df.drop(cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} constant/null columns")
    
    return features_df

# Main execution pipeline
print("Starting China Real Estate Demand Prediction Pipeline...")

# Step 1: Load all datasets
print("\n1. Loading datasets...")
datasets = load_and_preprocess_data(DATA_PATH)

# Step 2: Process test data
print("\n2. Processing test data...")
test_df = pl.read_csv(f"{DATA_PATH}/test.csv")
test_processed = test_df.with_columns(
    pl.col("id").str.split("_").list.get(0).alias("month"),
    pl.col("id").str.split("_").list.get(1).alias("sector")
)
print(f"Test data: {test_processed.shape}")

# Step 3: Build feature dataset
print("\n3. Building feature dataset...")
features_df = build_feature_dataset(datasets)
print(f"Feature dataset shape: {features_df.shape}")

# Step 4: Optimize memory and prepare features
print("\n4. Feature engineering...")
features_df = optimize_memory_usage(features_df)
features_df = prepare_model_data(features_df)

# Step 5: Split data for training and validation
print("\n5. Preparing training data...")

# Use time-based split with proper validation target
train_mask = pl.col("time") <= 64  # Train on data up to time 64
val_mask = pl.col("time") == 65    # Validate on time 65 (target is time 66)

train_data = features_df.filter(train_mask).drop_nulls(subset=["target"])
validation_data = features_df.filter(val_mask).drop_nulls(subset=["target"])

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")

# Step 6: Prepare features for modeling
feature_columns = [col for col in train_data.columns 
                  if col not in ["target", "sector", "month", "year", "sector_id", "time"]]
categorical_features = ["month_num"]

print(f"Using {len(feature_columns)} features for modeling")

# Step 7: Create CatBoost pools with proper validation target
train_pool = Pool(
    data=train_data.select(feature_columns).to_pandas().fillna(-2),
    label=train_data["target"].to_pandas(),
    cat_features=categorical_features
)

# FIXED: Include target for validation set
val_pool = Pool(
    data=validation_data.select(feature_columns).to_pandas().fillna(-2),
    label=validation_data["target"].to_pandas(),  # This was missing!
    cat_features=categorical_features
)

# Step 8: Train model with improved configuration
print("\n6. Training CatBoost model...")
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3,
    random_strength=1,
    bagging_temperature=0.8,
    border_count=128,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=RANDOM_SEED,
    early_stopping_rounds=100,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool)

# Step 9: Prepare final training data (train on all available data)
print("\n7. Final model training...")
final_train_mask = pl.col("time") <= 65  # Use all data up to time 65
final_train_data = features_df.filter(final_train_mask).drop_nulls(subset=["target"])

final_train_pool = Pool(
    data=final_train_data.select(feature_columns).to_pandas().fillna(-2),
    label=final_train_data["target"].to_pandas(),
    cat_features=categorical_features
)

# Retrain final model on all data
final_model = CatBoostRegressor(
    iterations=model.best_iteration_ + 50,  # Use optimal iterations from validation
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3,
    random_seed=RANDOM_SEED,
    verbose=100
)

final_model.fit(final_train_pool)

# Step 10: Generate predictions for test period
print("\n8. Generating predictions...")

# Prepare test features (time = 66)
test_features = features_df.filter(pl.col("time") == 66)
test_pool = Pool(
    data=test_features.select(feature_columns).to_pandas().fillna(-2),
    cat_features=categorical_features
)

# Generate predictions
test_predictions = final_model.predict(test_pool)
test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative

# Create sector to prediction mapping
sector_predictions = {}
sectors = test_features["sector"].unique()
for sector, pred in zip(sectors, test_predictions):
    sector_predictions[sector] = float(pred)

# Step 11: Create submission file
print("\n9. Creating submission file...")
submission_template = pl.read_csv(f"{DATA_PATH}/sample_submission.csv")

# Generate predictions for all test periods
final_predictions = []
for row in submission_template.rows():
    id_str = row[0]
    month = id_str.split("_")[0]
    sector = "_".join(id_str.split("_")[1:])
    
    # Use the predicted value for all future months
    pred_value = sector_predictions.get(sector, 0.0)
    final_predictions.append(pred_value)

submission = submission_template.with_columns(
    pl.Series("new_house_transaction_amount", final_predictions)
)

# Step 12: Save submission and show results
submission.write_csv("submission.csv")
print("âœ“ Submission saved as 'submission.csv'")

# Display results summary
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Best validation RMSE: {model.best_score_['validation']['RMSE']:.2f}")
print(f"Final model iterations: {final_model.tree_count_}")
print(f"Predictions - Min: {min(final_predictions):.2f}, "
      f"Max: {max(final_predictions):.2f}, "
      f"Mean: {np.mean(final_predictions):.2f}")
print(f"Zero predictions: {sum(1 for x in final_predictions if x == 0)}/{len(final_predictions)}")
print("="*50)

# Feature importance
try:
    importance = final_model.get_feature_importance()
    feature_importance_df = pl.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort('importance', descending=True)
    
    print("\nTop 10 Most Important Features:")
    for row in feature_importance_df.head(10).rows():
        print(f"  {row[0]}: {row[1]:.4f}")
except Exception as e:
    print(f"\nCould not compute feature importance: {e}")

print("\nPipeline completed successfully! ðŸŽ¯")

