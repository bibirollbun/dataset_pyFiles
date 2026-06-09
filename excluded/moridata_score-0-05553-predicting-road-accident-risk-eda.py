# =============================================================================
# ğŸ“¦ STANDARD LIBRARIES
# =============================================================================
import warnings
import numpy as np
import pandas as pd

# =============================================================================
# ğŸ“Š VISUALIZATION
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ğŸ§  SCIKIT-LEARN (Modeling / Preprocessing)
# =============================================================================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# ğŸš€ BOOSTING LIBRARIES
# =============================================================================
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =============================================================================
# ğŸ”¢ ENCODING
# =============================================================================
from category_encoders import TargetEncoder

# =============================================================================
# ğŸ”§ SETTINGS / WARNINGS / ENVIRONMENT
# =============================================================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# Optional: if running inside Jupyter
# %matplotlib inline


# ğŸ“¥ Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

original = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")


# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

original['dataset'] = 'train'



# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# ğŸ§¾ Display dataset shape
print("Dataset shape:", df.shape)

# ğŸ‘�ï¸� Preview the data
df


df.shape



# ğŸ“‹ Check column types and non-null counts
df.info()


# âœ… Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# ğŸ”� Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


# ğŸ“Š Descriptive statistics for numerical columns
df[numerical_cols].describe()


# ğŸ”¢ Unique value counts for categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


# ğŸ�¯ Target Variable Distribution (Numerical)

# ---
## 1. Histogram: Showing the Frequency Distribution
# ---

plt.figure(figsize=(8, 5))
# Use 'histplot' for numerical data. Adjust the 'bins' parameter as needed.
sns.histplot(data=df, x='accident_risk', kde=True, bins=30, color='skyblue', edgecolor='black')

plt.title('Distribution of Accident Risk (Histogram)', fontsize=14)
plt.xlabel('Accident Risk Score', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# ---
## 2. Box Plot: Showing Central Tendency and Outliers
# ---

plt.figure(figsize=(8, 3))
# Use 'boxplot' to visualize the five-number summary and outliers
sns.boxplot(data=df, x='accident_risk', color='lightcoral')

plt.title('Accident Risk Distribution (Box Plot)', fontsize=14)
plt.xlabel('Accident Risk Score', fontsize=12)
plt.tight_layout()
plt.show()

# ---
## 3. Descriptive Statistics
# ---

print("\nğŸ“Š Accident Risk Descriptive Statistics:")
# Use .describe() for a numerical summary
print(df['accident_risk'].describe().round(3))


# Define the numerical columns
numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 
                  'num_reported_accidents', 'accident_risk']

# Separate the columns based on their nature
continuous_features = ['curvature', 'speed_limit', 'accident_risk']
discrete_features = ['num_lanes', 'num_reported_accidents']


for col in numerical_cols:
    print(f"--- Visualizing: {col} ---")
    
    # Set up a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Distribution of {col}', fontsize=16)

    if col in continuous_features:
        # Left: Histogram for density/shape
        sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Histogram (Shape & Density)')
        
        # Right: Boxplot for quartiles/outliers
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Outliers & Spread)')

    elif col in discrete_features:
        # Left: Count Plot for frequency of small integer values
        sns.countplot(x=df[col].dropna(), ax=axes[0], palette='viridis', edgecolor='black')
        axes[0].set_title('Count Plot (Frequency)')
        
        # Right: Boxplot can still be useful for summary statistics
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Outliers & Spread)')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()
    
    # Also print the numerical summary for quick reference
    print("\nDescriptive Statistics:")
    print(df[col].describe().round(3))
    print("\n" + "="*50 + "\n")


# ğŸ“Š Distribution of Categorical Features

# Updated list with the categorical columns from your dataset
cat_cols = [
    'road_type', 
    'lighting', 
    'weather', 
    'road_signs_present', 
    'public_road', 
    'time_of_day', 
    'holiday', 
    'school_season'
]

for col in cat_cols:
    # Adjust figure size for better readability, especially for features like 'weather'
    plt.figure(figsize=(10, 5)) 
    sns.countplot(
        data=df,
        x=col,
        # Ensure bars are ordered by frequency (most common first)
        order=df[col].value_counts().index, 
        palette='Set2',
        edgecolor='black'
    )
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Rotate x-labels if needed for long category names (e.g., weather)
    plt.xticks(rotation=30, ha='right') 
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ğŸ§® Print Category Proportions
    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


# Assuming you have already calculated the descriptive statistics for 'accident_risk'
# A good threshold for "high risk" is often the 75th percentile (Q3), which was 0.460.
risk_threshold = df['accident_risk'].quantile(0.75) 

# Create the new binary target column
df['high_risk_flag'] = (df['accident_risk'] > risk_threshold).astype(int)

# Check the distribution of the new flag (Optional)
print(f"âœ… Created 'high_risk_flag' (Threshold: >{risk_threshold:.3f})")
print(df['high_risk_flag'].value_counts(normalize=True).round(3))


# ğŸ�¨ Categorical Feature Distributions by Accident Risk Flag - Custom Colors

# Select key categorical columns to see their relationship with the risk flag
cols_to_plot = ['road_type', 'time_of_day', 'weather', 'lighting'] 

custom_palette = ['#1F77B4', '#FF7F0E']  # Blue for Low Risk (0), Orange for High Risk (1)

target_col = 'high_risk_flag' # Use the new binary target

for col in cols_to_plot:
    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df,
        x=col,
        hue=target_col, # Changed from 'y' to 'high_risk_flag'
        palette=custom_palette,
        edgecolor='black'
    )
    plt.title(f'Distribution of {col} by High Risk Status', fontsize=14)
    plt.xlabel(f'{col}', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Adjust rotation for potential long labels
    plt.xticks(rotation=20, ha='right') 
    
    # Update the legend to reflect the new target variable
    plt.legend(title='High Risk Flag', labels=['Low Risk (0)', 'High Risk (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# Define the list of numerical columns based on your dataset
num_cols = [
    'num_lanes', 
    'curvature', 
    'speed_limit', 
    'num_reported_accidents', 
    'accident_risk'
]

# ğŸ”— Correlation Between Numerical Features

plt.figure(figsize=(10, 8)) # Slightly increased size for better label visibility on a heatmap
sns.heatmap(
    # Calculate the correlation matrix for the specified columns
    df[num_cols].corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Between Numerical Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Define the numerical columns
num_cols = [
    'num_lanes', 
    'curvature', 
    'speed_limit', 
    'num_reported_accidents'
    # 'accident_risk' is the target (or source for the flag), so we don't plot it against itself
]

target_col = 'high_risk_flag' # Use the binary target we created earlier

plt.figure(figsize=(15, 8)) # Adjusted figure size for better layout

for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i + 1) # Adjusted subplot grid based on 4 features
    
    sns.boxplot(
        data=df,
        x=target_col, # Changed from 'y' to the binary target flag
        y=col,
        palette=['#1F77B4', '#FF7F0E'],  # Blue for Low Risk, Orange for High Risk
        linewidth=1.2,
        fliersize=4
    )
    plt.title(f'{col} by Accident Risk Status', fontsize=14, fontweight='semibold', color='#2E4057')
    
    # Updated X-label to reflect the new target
    plt.xlabel('High Risk Flag (0: Low Risk, 1: High Risk)', fontsize=12) 
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# Risk by Time of Day and Day Type
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df, 
    x='time_of_day', 
    y='accident_risk', 
    order=['morning', 'afternoon', 'evening'], # Define a logical order
    palette='viridis', 
    ci=None # Don't show confidence intervals for cleaner plot
)
plt.title('Average Accident Risk by Time of Day', fontsize=16)
plt.xlabel('Time of Day')
plt.ylabel('Mean Accident Risk')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Risk by Road Type & Weather/Lighting

# Requires filtering the weather to a few key types for readability
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df, 
    x='road_type', 
    y='accident_risk', 
    hue='weather', # Group by weather condition
    palette='Spectral',
    ci=None 
)
plt.title('Mean Accident Risk: Road Type vs. Weather', fontsize=16)
plt.xlabel('Road Type')
plt.ylabel('Mean Accident Risk')
plt.legend(title='Weather')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

# Plot the mean 'accident_risk' grouped by 'road_type' and 'weather'
sns.barplot(
    data=df, 
    x='road_type', 
    y='accident_risk', 
    hue='weather', # Groups the bars within each road_type
    palette='Spectral',
    ci=None # Plot the mean only
)

plt.title('Mean Accident Risk: Road Type vs. Weather Condition', fontsize=16)
plt.xlabel('Road Type')
plt.ylabel('Mean Accident Risk')
plt.legend(title='Weather', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print the numerical mean values for precision
print("\nğŸ“Š Mean Accident Risk by Road Type and Weather (Top Combinations):")
# Group and pivot the data for a clean, table-like view of the means
risk_pivot = df.groupby(['road_type', 'weather'])['accident_risk'].mean().unstack().round(4)
print(risk_pivot)
print("-" * 60)


plt.figure(figsize=(9, 5))

# Use barplot to show the mean 'accident_risk' for each 'speed_limit'
sns.barplot(
    data=df, 
    x='speed_limit', 
    y='accident_risk', 
    # Ensure speed limits are plotted in ascending order
    order=sorted(df['speed_limit'].unique()), 
    palette='magma', 
    ci=None # Plot the mean only
)

plt.title('Mean Accident Risk Across Different Speed Limits', fontsize=16)
plt.xlabel('Speed Limit')
plt.ylabel('Mean Accident Risk')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print the numerical mean values for precision
print("\nğŸ“Š Mean Accident Risk by Speed Limit:")
print(df.groupby('speed_limit')['accident_risk'].mean().sort_values(ascending=False).round(4))
print("-" * 40)


###########


df


# =============================================================================
# DATA PREPARATION
# =============================================================================
# Assume 'df' is pre-loaded
# Example:
# df = pd.read_csv("your_data.csv")

# Split train/test sets
train_df = df[df["dataset"] == "train"].copy()
test_df = df[df["dataset"] == "test"].copy()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Missing target values: {train_df['accident_risk'].isna().sum()}")

# Separate features and target
X_train = train_df.drop(["id", "accident_risk", "dataset", "high_risk_flag"], axis=1)
y_train = train_df["accident_risk"]

X_test = test_df.drop(["id", "accident_risk", "dataset", "high_risk_flag"], axis=1)


# =============================================================================
# FEATURE ENGINEERING (OPTIONAL)
# =============================================================================
# Uncomment and customize if needed
# def add_features(X: pd.DataFrame) -> pd.DataFrame:
#     """Add optional engineered features."""
#     X = X.copy()
#     X["lanes_x_speed"] = X["num_lanes"] * X["speed_limit"]
#     X["curvature_x_speed"] = X["curvature"] * X["speed_limit"]
#     X["lanes_x_curvature"] = X["num_lanes"] * X["curvature"]
#     X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
#     X["high_curvature"] = (X["curvature"] > 0.7).astype(int)
#     X["high_speed"] = (X["speed_limit"] > 60).astype(int)
#     X["few_lanes"] = (X["num_lanes"] <= 2).astype(int)
#     X["many_accidents"] = (X["num_reported_accidents"] >= 2).astype(int)
#     X["curvature_squared"] = X["curvature"] ** 2
#     X["speed_squared"] = X["speed_limit"] ** 2
#     X["speed_per_lane"] = X["speed_limit"] / (X["num_lanes"] + 1)
#     X["curvature_speed_ratio"] = X["curvature"] / (X["speed_limit"] + 1)
#     return X

# X_train = add_features(X_train)
# X_test = add_features(X_test)


# =============================================================================
# TARGET TRANSFORMATION (OPTIONAL)
# =============================================================================
use_target_transform = False  # Toggle on/off

if use_target_transform:
    print("\nApplying Yeo-Johnson target transformation...")
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    
    y_train_transformed = transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    print(f"Original target mean/std: {y_train.mean():.4f} / {y_train.std():.4f}")
    print(f"Transformed target mean/std: {y_train_transformed.mean():.4f} / {y_train_transformed.std():.4f}")
    
    y_train_original = y_train.copy()
    y_train = pd.Series(y_train_transformed, index=y_train.index)
    target_transformer = transformer
else:
    print("\nTarget transformation disabled.")
    target_transformer = None


# =============================================================================
# TARGET ENCODING + PREPROCESSING
# =============================================================================
# Define column groups
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
cat_cols = ["road_type", "weather", "time_of_day", "lighting"]
num_cols = [ "curvature", "speed_limit"]

# Columns for target encoding
cols_to_encode = cat_cols + bool_cols + ["num_lanes", "num_reported_accidents"]

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=25.0), cols_to_encode),
        ("scaler", StandardScaler(), num_cols)
    ],
    remainder="drop"
)


X_train


# =============================================================================
# MODEL DEFINITION (XGBoost Only)
# =============================================================================

# Define model
models = {
    "XGBoost": XGBRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
}


# =============================================================================
# CROSS-VALIDATION WITH MULTIPLE METRICS
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (10-Fold)")
print("=" * 80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

scoring = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2"
}

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)

    # Create pipeline (preprocessing + model)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Cross-validation
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    # Store averaged CV results
    cv_results[name] = {
        "RMSE": -cv_scores["test_RMSE"].mean(),
        "RMSE_std": cv_scores["test_RMSE"].std(),
        "MAE": -cv_scores["test_MAE"].mean(),
        "MAE_std": cv_scores["test_MAE"].std(),
        "R2": cv_scores["test_R2"].mean(),
        "R2_std": cv_scores["test_R2"].std()
    }

    print(f"RMSE: {cv_results[name]['RMSE']:.6f} (+/- {cv_results[name]['RMSE_std']:.6f})")
    print(f"MAE:  {cv_results[name]['MAE']:.6f} (+/- {cv_results[name]['MAE_std']:.6f})")
    print(f"RÂ²:   {cv_results[name]['R2']:.6f} (+/- {cv_results[name]['R2_std']:.6f})")


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

best_model_name = "XGBoost"
best_model = models[best_model_name]

print("\n" + "=" * 80)
print(f"FINAL MODEL: {best_model_name}")
print("=" * 80)
print(f"CV RMSE: {cv_results[best_model_name]['RMSE']:.6f}")
print(f"CV MAE:  {cv_results[best_model_name]['MAE']:.6f}")
print(f"CV RÂ²:   {cv_results[best_model_name]['R2']:.6f}")

# Train on full dataset
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

print(f"\nTraining final {best_model_name} model on full training set...")
final_pipeline.fit(X_train, y_train)
print("Final model training complete.")


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

test_predictions = final_pipeline.predict(X_test)

# Inverse transform if target was transformed
if use_target_transform and target_transformer is not None:
    print("Inverse transforming predictions...")
    test_predictions = target_transformer.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).ravel()

# Basic statistics on predictions
print(f"\nPrediction Statistics:")
print(f"Min:    {test_predictions.min():.6f}")
print(f"Max:    {test_predictions.max():.6f}")
print(f"Mean:   {test_predictions.mean():.6f}")
print(f"Median: {np.median(test_predictions):.6f}")
print(f"Std:    {test_predictions.std():.6f}")


# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================

submission = pd.DataFrame({
    'id': test_df['id'].values,
    'accident_risk': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nâœ“ Submission file saved: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

