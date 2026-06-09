import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head(5)


print("Original Train Shape:", train.shape)
print("Original Test Shape:", test.shape)



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and categorical with mode."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string booleans to bool and unify formats."""
    bool_like = ['True', 'False', 'true', 'false']

    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isin(bool_like).any():
            df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False})

    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize categorical text values."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.strip().str.lower())

    if 'weather' in df.columns:
        df['weather'] = df['weather'].replace({
            'rain': 'rainy',
            'clear sky': 'clear'
        })

    if 'lighting' in df.columns:
        df['lighting'] = df['lighting'].replace({
            'day': 'daylight',
            'night time': 'night'
        })

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def remove_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Remove outliers using IQR for numeric columns."""
    if cols is None:
        cols = ['curvature', 'speed_limit', 'num_lanes', 'accident_risk']

    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def clean_dataset(df: pd.DataFrame, drop_duplicates=True, remove_outlier=False) -> pd.DataFrame:
    """Complete cleaning process for a single dataframe."""
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = standardize_categoricals(df)
    if drop_duplicates:
        df = remove_duplicates(df)
    if remove_outlier:
        df = remove_outliers(df)
    return df



train_clean = clean_dataset(train, drop_duplicates=True, remove_outlier=True)
test_clean = clean_dataset(test, drop_duplicates=False, remove_outlier=False)

print("Cleaned Train Shape:", train_clean.shape)
print("Cleaned Test Shape:", test_clean.shape)


# train_clean.head(5)



# ==========================================================
# Step 1: Data Cleaning
# ==========================================================
def clean_data(df):
    """Basic data cleaning: handle missing, strip text, fix types."""
    df = df.copy()
    
    # Drop duplicate rows if any
    df = df.drop_duplicates()
    
    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())
    
    # Fill missing values
    df = df.fillna({
        'weather': 'clear',
        'lighting': 'daylight',
        'time_of_day': 'afternoon',
        'road_signs_present': False,
        'public_road': True
    })
    
    return df


# ==========================================================
# Step 2: Feature Engineering
# ==========================================================
def feature_engineering(df):
    """Add domain-specific engineered features."""
    df = df.copy()
    
    # Interaction and ratio features
    df['curvature_speed_ratio'] = df['curvature'] / (df['speed_limit'] + 1e-6)
    df['lanes_per_speed'] = df['num_lanes'] / (df['speed_limit'] + 1e-6)
    
    # Combined categorical features
    df['lighting_weather'] = df['lighting'] + '_' + df['weather']
    df['road_time_combo'] = df['road_type'] + '_' + df['time_of_day']
    
    # Convert boolean to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # Derived numeric / categorical features
    df['is_high_speed'] = (df['speed_limit'] >= 60).astype(int)
    df['curvature_level'] = pd.cut(
        df['curvature'],
        bins=[0, 0.2, 0.5, 1.0],
        labels=['low', 'medium', 'high']
    )
    
    # Custom complexity feature
    df['complexity_score'] = (
        df['curvature'] * 2 +
        (1 / (df['num_lanes'] + 1)) +
        df['is_high_speed'] * 1.5
    )
    
    # Simplify time of day
    df['day_period'] = df['time_of_day'].replace({
        'morning': 'day',
        'afternoon': 'day',
        'evening': 'night'
    })
    
    return df


# ==========================================================
# Step 3: Encoding + Scaling
# ==========================================================
def encode_and_scale(train_df, test_df, target_col='accident_risk'):
    """Encodes categorical features and scales numeric ones consistently."""
    
    # Separate target and features
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.copy()
    
    # Identify numeric and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    cat_features = list(preprocessor.named_transformers_['cat']
                        .named_steps['encoder']
                        .get_feature_names_out(cat_cols))
    num_features = list(num_cols)
    feature_names = num_features + cat_features
    
    # Convert to DataFrames
    X_train_encoded = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_encoded = pd.DataFrame(X_test_processed, columns=feature_names)
    
    print("Encoding & scaling complete.")
    return X_train_encoded, X_test_encoded, y_train, preprocessor


# ==========================================================
# Step 4: Full Pipeline Wrapper
# ==========================================================
def full_feature_pipeline(train_raw, test_raw, target_col='accident_risk'):
    """Executes cleaning → feature engineering → encoding/scaling pipeline."""
    
    # Feature engineer
    train_feat = feature_engineering(train_clean)
    test_feat = feature_engineering(test_clean)
    
    # Encode & scale
    X_train_final, X_test_final, y_train, preprocessor = encode_and_scale(
        train_feat, test_feat, target_col
    )
    
    print("Full feature pipeline complete.")
    print("Train shape:", X_train_final.shape, "| Test shape:", X_test_final.shape)
    
    return X_train_final, X_test_final, y_train, preprocessor

# Run full pipeline
X_train_final, X_test_final, y_train, preprocessor = full_feature_pipeline(train_clean, test_clean)



from sklearn.model_selection import train_test_split

# Split 75/25 for training and validation
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_final, y_train, test_size=0.25, random_state=42
)



models = {
    # Linear Regression (baseline, very fast)
    "Linear Regression": LinearRegression(),
    
    # Random Forest (reduced trees, parallelized)
    "Random Forest": RandomForestRegressor(
        n_estimators=100,       # fewer trees
        max_depth=12,           # limit depth
        max_features='sqrt',    # sqrt features per split
        n_jobs=-1,              # use all cores
        random_state=42
    ),
    
    # XGBoost (faster for large data using hist method)
    "XGBoost": XGBRegressor(
        n_estimators=200,       # reduce slightly
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',     # faster for large datasets
        random_state=42,
        objective='reg:squarederror'
    ),
    
    # LightGBM (pass categorical features directly)
    "LightGBM": LGBMRegressor(
        n_estimators=200,       # fewer trees
        learning_rate=0.05,
        max_depth=6,            # slightly deeper
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


results = []

for name, model in models.items():
    print(f"\nTraining model: {name}...")
    model.fit(X_train, y_train_split)
    preds = model.predict(X_val)
    
    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    
    results.append((name, rmse, mae, r2))
    
    # Print results after each iteration
    print(f"Results for {name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.2f}")

# Convert all results to DataFrame at the end
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
results_df = results_df.sort_values(by="RMSE", ascending=False)

print("\n All model comparison results:")
display(results_df)



metric = "RMSE"  
ascending = False if metric == "R2" else True
results_df_sorted = results_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)

best_model_name = results_df_sorted.iloc[0]["Model"]
best_metric_value = results_df_sorted.iloc[0][metric]

print(f"Best model based on {metric}: {best_model_name}")
print(f"{metric} value: {best_metric_value:.4f}")

best_model = models[best_model_name]
best_model.fit(X_train_final, y_train)



# Predict on validation set (to estimate performance)
val_preds = best_model.predict(X_val)

# Compute metrics
rmse = mean_squared_error(y_val, val_preds, squared=False)
mae = mean_absolute_error(y_val, val_preds)
r2 = r2_score(y_val, val_preds)

print("Performance on validation set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")



# Predict on test set for submission
test_preds = best_model.predict(X_test_final)

# clip predictions between 0 and 1
test_preds = test_preds.clip(0, 1)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})

# Display sample submission
print("\nSample submission:")
display(submission.head())

# Save submission
submission.to_csv("submission.csv", index=False)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np

# Split data
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_final, y_train, test_size=0.25, random_state=42
)

# Base models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=12, max_features='sqrt', n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, tree_method='hist', random_state=42, objective='reg:squarederror'
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
}

# Store base model predictions
val_preds_df = pd.DataFrame()
test_preds_df = pd.DataFrame()

print("\nTraining base models for stacking...\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train_split)
    
    # Validation and test predictions
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_final)
    
    val_preds_df[name] = val_pred
    test_preds_df[name] = test_pred

# Evaluate base models
for name in models.keys():
    rmse = mean_squared_error(y_val, val_preds_df[name], squared=False)
    mae = mean_absolute_error(y_val, val_preds_df[name])
    r2 = r2_score(y_val, val_preds_df[name])
    print(f"\n{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# Meta-model (stacking model)
meta_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

print("\nTraining meta-model on base model predictions...")
meta_model.fit(val_preds_df, y_val)

# Meta-model validation prediction
meta_val_pred = meta_model.predict(val_preds_df)
meta_rmse = mean_squared_error(y_val, meta_val_pred, squared=False)
meta_mae = mean_absolute_error(y_val, meta_val_pred)
meta_r2 = r2_score(y_val, meta_val_pred)

print("\n=== Meta-Model Performance on Validation Set ===")
print(f"RMSE: {meta_rmse:.4f}")
print(f"MAE:  {meta_mae:.4f}")
print(f"R²:   {meta_r2:.4f}")

# Final prediction on test set (average base model predictions)
meta_test_pred = meta_model.predict(test_preds_df)
meta_test_pred = np.clip(meta_test_pred, 0, 1)

# Final submission
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": meta_test_pred
})

submission.to_csv("submission_stacked.csv", index=False)
print("\n✅ Stacking submission saved as 'submission_stacked.csv'")





























































































































