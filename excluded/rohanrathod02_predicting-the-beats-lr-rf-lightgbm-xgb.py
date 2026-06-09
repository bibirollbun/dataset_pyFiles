pip install --upgrade xgboost


%pip install catboost -q


import kagglehub
gauravduttakiit_bpm_prediction_challenge_path = kagglehub.dataset_download('gauravduttakiit/bpm-prediction-challenge')

print('Data source import complete.')



# ====================================================
# Setup & Imports
# ====================================================

# Standard Libraries
import os
import warnings
import random

# Data Manipulation & Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor, RandomForestRegressor

# Gradient Boosting Models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Settings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ====================================================
# Load Data
# ====================================================
try:
    train_ps = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_ps = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

    print("Dataset Shapes:")
    print("Train:", train_ps.shape)
    print("Test:", test_ps.shape)
    print("Sample Submission:", sample_submission.shape)
    print()
except FileNotFoundError as e:
    print(f"âš ï¸� Error loading files: {e}")
    print("Please ensure train.csv, test.csv, and sample_submission.csv are available in the working directory.")



def data_info(df, df_name):
    """Comprehensive overview of a DataFrame with styled output."""

    print(f"\n{'='*70}")
    print(f"ğŸ“Š Comprehensive Information for DataFrame: {df_name}")
    print(f"{'='*70}\n")

    # --- Head & Tail ---
    print(f"--- {df_name} Head ---\n")
    display(df.head().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    print(f"--- {df_name} Tail ---\n")
    display(df.tail().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    # --- Info ---
    print(f"\n--- {df_name} Info ---\n")
    df.info()

    # --- Describe (numeric + categorical) ---
    print(f"\n--- {df_name} Describe (Numeric) ---\n")
    display(df.describe().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    print(f"\n--- {df_name} Describe (All Columns) ---\n")
    display(df.describe(include='all').transpose().style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
        {'selector': 'td', 'props': [('font-size', '10pt')]}
    ], overwrite=False))

    # --- Missing Values ---
    print(f"\n--- {df_name} Missing Values ---\n")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("âœ… No missing values found.")
    else:
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': (missing / len(df)) * 100
        }).query("`Missing Count` > 0")

        display(missing_df.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]},
            {'selector': 'td', 'props': [('font-size', '10pt')]}
        ], overwrite=False))

    print(f"\n{'='*70}\n")

# Apply to datasets
data_info(train_ps, "train_ps")
data_info(test_ps, "test_ps")



# Define excluded features
excluded_features = ['id', 'BeatsPerMinute']

# Separate numerical and categorical features
numerical_features = [
    col for col in train_ps.select_dtypes(include=np.number).columns
    if col not in excluded_features
]

categorical_features = [
    col for col in train_ps.select_dtypes(exclude=np.number).columns
    if col not in excluded_features
]

# Print results
print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")



def plot_correlation_heatmap(df, numerical_cols, df_name, annot=True):
    """
    Generates and displays a correlation heatmap for specified numerical columns,
    showing only the lower triangle.
    """
    corr = df[numerical_cols].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        cmap='Greens',
        fmt=".2f",
        vmin=-1, vmax=1,
        cbar_kws={"shrink": .8}
    )
    plt.title(f'Correlation Heatmap of Numerical Features ({df_name})')
    plt.show()

# Add target for correlation analysis
numerical_features_with_target = numerical_features + ['BeatsPerMinute']

plot_correlation_heatmap(train_ps, numerical_features_with_target, "train_ps")



def plot_numerical_distributions(train_df, test_df, numerical_cols):
    """
    Generates KDE and box plots for numerical features, comparing train vs test distributions.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Combine train and test for plotting
    combined_df = pd.concat([
        train_df[numerical_cols].assign(Source='Train'),
        test_df[numerical_cols].assign(Source='Test')
    ], axis=0, ignore_index=True)

    palette = ['#1f77b4', '#ff7f0e']  # Distinct colors for Train/Test

    for col in numerical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})

        # KDE Plot
        sns.kdeplot(
            data=combined_df, x=col, hue='Source', ax=axes[0], fill=True, palette=palette
        )
        axes[0].set_title(f'{col} Distribution (KDE)', fontsize=14)
        axes[0].set_xlabel('Density')
        axes[0].set_ylabel(col)

        # Box Plot
        sns.boxplot(
            data=combined_df, y=col, x='Source', ax=axes[1],
            orient='v', width=0.5, linewidth=1, fliersize=3, palette=palette
        )
        axes[1].set_title(f'{col} Boxplot', fontsize=14)
        axes[1].set_xlabel('Dataset')
        axes[1].set_ylabel(col)

        plt.tight_layout()
        plt.show()

# Call the function for numerical features
plot_numerical_distributions(train_ps, test_ps, numerical_features)



def plot_categorical_distributions(train_df, test_df, categorical_cols):
    """
    Generates count plots for each categorical feature, comparing train and test distributions.
    Uses side-by-side bars for clear comparison.
    """
    if len(categorical_cols) == 0:
        print("No categorical features to plot.")
        return

    palette = ['#1f77b4', '#ff7f0e']  # Train / Test colors

    for col in categorical_cols:
        plt.figure(figsize=(10, 6))

        # Combine train and test with a source column
        combined = pd.concat([
            train_df[[col]].assign(Source='Train'),
            test_df[[col]].assign(Source='Test')
        ], axis=0, ignore_index=True)

        sns.countplot(x=col, hue='Source', data=combined, palette=palette)

        plt.title(f'Distribution of {col} (Train vs Test)', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title='Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Call the function
plot_categorical_distributions(train_ps, test_ps, categorical_features)





plt.figure(figsize=(12, 5))
sns.kdeplot(data=train_ps, x='BeatsPerMinute', palette='viridis', fill=True, color='skyblue')
plt.title("Distribution of BeatsPerMinute (Target Variable)", fontsize=14)
plt.xlabel("Beats Per Minute")
plt.ylabel("Density")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Check total missing values
train_missing = train_ps.isnull().sum().sum()
test_missing = test_ps.isnull().sum().sum()

print(f"âœ… Total missing values in train_ps: {train_missing}")
print(f"âœ… Total missing values in test_ps: {test_missing}")



def reduce_memory_usage(df):
    """
    Reduce memory usage of a DataFrame by downcasting numerical columns and converting object columns to category.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:  # float
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df

# Apply memory reduction
train_ps = reduce_memory_usage(train_ps)
test_ps = reduce_memory_usage(test_ps)



import numpy as np
import pandas as pd

# Reload original data
try:
    train_ps_original = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_ps_original = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    print("âœ… Original data reloaded for preprocessing.")
except FileNotFoundError as e:
    print(f"Error reloading original files: {e}")
    exit()

# Store lengths for later splitting
train_len = len(train_ps_original)
test_len = len(test_ps_original)
print(f"Train rows: {train_len}, Test rows: {test_len}")

# Combine train and test (excluding target) for consistent preprocessing
combined_ps = pd.concat(
    [train_ps_original.drop('BeatsPerMinute', axis=1), test_ps_original],
    ignore_index=True
)

# Handle missing values (drop or optionally impute)
if combined_ps.isnull().sum().sum() > 0:
    print("âš ï¸� Missing values detected. Dropping rows with NaNs.")
combined_ps_cleaned = combined_ps.dropna()

# Transformations

# 1. Log-transform TrackDurationMs to reduce skewness
if 'TrackDurationMs' in combined_ps_cleaned.columns:
    combined_ps_cleaned['TrackDurationMs'] = np.log1p(combined_ps_cleaned['TrackDurationMs'])
    print("âœ… Applied log1p transformation to TrackDurationMs.")
else:
    print("âš ï¸� TrackDurationMs not found for log transformation.")

# 2. Relative Loudness: convert AudioLoudness (negative dB) to positive relative scale
if 'AudioLoudness' in combined_ps_cleaned.columns:
    max_loudness = combined_ps_cleaned['AudioLoudness'].max()
    combined_ps_cleaned['RelativeLoudness'] = max_loudness - combined_ps_cleaned['AudioLoudness']
    combined_ps_cleaned.drop('AudioLoudness', axis=1, inplace=True)
    print("âœ… Created RelativeLoudness and dropped AudioLoudness.")
else:
    print("âš ï¸� AudioLoudness not found for RelativeLoudness transformation.")

# Split back into train and test
train_processed = combined_ps_cleaned.iloc[:train_len, :].copy()
test_processed = combined_ps_cleaned.iloc[train_len:, :].copy()

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")



from sklearn.preprocessing import OrdinalEncoder

# Small epsilon to avoid division by zero
epsilon = 1e-6

# -----------------------------
# 1ï¸�âƒ£ Binning / Discretization
# -----------------------------
features_to_bin = ['RelativeLoudness', 'MoodScore']
for col in features_to_bin:
    if col in combined_ps_cleaned.columns:
        combined_ps_cleaned[f'{col}_Quartile'] = pd.qcut(
            combined_ps_cleaned[col], q=4, labels=False, duplicates='drop'
        )
        print(f"âœ… Binned {col} into quartiles.")
    else:
        print(f"âš ï¸� Column {col} not found for binning.")

# TrackDurationMs bucketed into short/medium/long
if 'TrackDurationMs' in combined_ps_cleaned.columns:
    combined_ps_cleaned['TrackDurationBucket'] = pd.qcut(
        combined_ps_cleaned['TrackDurationMs'],
        q=3,
        labels=['short', 'medium', 'long'],
        duplicates='drop'
    )

    # Ordinal encoding for bucket
    categories_order = [['short', 'medium', 'long']]
    ordinal_encoder = OrdinalEncoder(
        categories=categories_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    combined_ps_cleaned['TrackDurationBucket'] = ordinal_encoder.fit_transform(
        combined_ps_cleaned[['TrackDurationBucket']]
    )
    print("âœ… Created and encoded TrackDurationBucket.")
else:
    print("âš ï¸� TrackDurationMs not found for TrackDurationBucket creation.")

# -----------------------------
# 2ï¸�âƒ£ Interaction Features
# -----------------------------
interaction_cols_required = [
    'Energy', 'RhythmScore', 'MoodScore', 'VocalContent', 'InstrumentalScore', 'AcousticQuality'
]
if all(col in combined_ps_cleaned.columns for col in interaction_cols_required):
    combined_ps_cleaned['Energy_x_RhythmScore'] = combined_ps_cleaned['Energy'] * combined_ps_cleaned['RhythmScore']
    combined_ps_cleaned['MoodScore_x_RhythmScore'] = combined_ps_cleaned['MoodScore'] * combined_ps_cleaned['RhythmScore']
    combined_ps_cleaned['Vocals_vs_Instrument'] = combined_ps_cleaned['VocalContent'] / (
        combined_ps_cleaned['InstrumentalScore'] + epsilon
    )
    combined_ps_cleaned['Energy_minus_AcousticQuality'] = combined_ps_cleaned['Energy'] - combined_ps_cleaned['AcousticQuality']
    print("âœ… Created interaction features.")
else:
    print("âš ï¸� Necessary columns not found for interaction features.")

# -----------------------------
# 3ï¸�âƒ£ Domain-Specific Features
# -----------------------------
domain_cols_required = ['RelativeLoudness', 'TrackDurationMs', 'LivePerformanceLikelihood', 'Energy']
if all(col in combined_ps_cleaned.columns for col in domain_cols_required):
    combined_ps_cleaned['NormalizedLoudness'] = combined_ps_cleaned['RelativeLoudness'] / (
        combined_ps_cleaned['TrackDurationMs'] + epsilon
    )
    combined_ps_cleaned['LiveEnergy'] = combined_ps_cleaned['LivePerformanceLikelihood'] * combined_ps_cleaned['Energy']
    print("âœ… Created domain-specific features.")
else:
    print("âš ï¸� Necessary columns not found for domain feature creation.")

# -----------------------------
# List of all new features
# -----------------------------
new_features = [
    'RelativeLoudness_Quartile', 'MoodScore_Quartile', 'TrackDurationBucket',
    'Energy_x_RhythmScore', 'MoodScore_x_RhythmScore',
    'Vocals_vs_Instrument', 'Energy_minus_AcousticQuality',
    'NormalizedLoudness', 'LiveEnergy'
]

print(f"ğŸ“� Total new features created: {len(new_features)}")
print(new_features)



from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# -----------------------------
# 1ï¸�âƒ£ Identify numeric features for scaling
# -----------------------------
numerical_features_final = combined_ps_cleaned.select_dtypes(include=np.number).columns.tolist()
numerical_features_final = [col for col in numerical_features_final if col not in ['id', 'BeatsPerMinute']]
print(f"\nNumeric features to scale: {numerical_features_final}")

# -----------------------------
# 2ï¸�âƒ£ Split combined dataset into train and test
# -----------------------------
train_ps_processed = combined_ps_cleaned.iloc[:train_len].reset_index(drop=True).copy()
test_ps_processed = combined_ps_cleaned.iloc[train_len:].reset_index(drop=True).copy()

# -----------------------------
# 3ï¸�âƒ£ Apply StandardScaler
# -----------------------------
scaler = StandardScaler()
train_scaled_values = scaler.fit_transform(train_ps_processed[numerical_features_final])
test_scaled_values = scaler.transform(test_ps_processed[numerical_features_final])

# Convert scaled arrays back to DataFrames
train_scaled_df = pd.DataFrame(train_scaled_values, columns=numerical_features_final, index=train_ps_processed.index)
test_scaled_df = pd.DataFrame(test_scaled_values, columns=numerical_features_final, index=test_ps_processed.index)

# -----------------------------
# 4ï¸�âƒ£ Preserve IDs
# -----------------------------
train_scaled_df['id'] = train_ps_processed['id'].values
test_scaled_df['id'] = test_ps_processed['id'].values

# -----------------------------
# 5ï¸�âƒ£ Merge target variable (BeatsPerMinute) for train
# -----------------------------
# Using the original train CSV to preserve target
train_original = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
train_scaled_df = pd.merge(train_scaled_df, train_original[['id', 'BeatsPerMinute']], on='id', how='left')

# -----------------------------
# 6ï¸�âƒ£ Prepare final feature and target sets
# -----------------------------
final_features_list = [col for col in train_scaled_df.columns if col not in ['id', 'BeatsPerMinute']]

X_train_processed = train_scaled_df[final_features_list].copy()
y_train_processed = train_scaled_df['BeatsPerMinute'].copy()
X_test_processed = test_scaled_df[final_features_list].copy()

# -----------------------------
# 7ï¸�âƒ£ Check shapes and sample data
# -----------------------------
print("\nTrain features shape:", X_train_processed.shape)
print("Train target shape:", y_train_processed.shape)
print("Test features shape:", X_test_processed.shape)

print("\nHead of X_train_processed:")
display(X_train_processed.head())

print("\nHead of X_test_processed:")
display(X_test_processed.head())



import pandas as pd

print("--- Comparison of Features Before and After Feature Creation ---")

# -----------------------------
# Original numerical features (before feature engineering)
# -----------------------------
features_before = [col for col in numerical_features if col not in ['id', 'BeatsPerMinute']]
print("\nFeatures Before Feature Creation (Original Numerical Features):")
display(features_before)

# -----------------------------
# Final processed features (after feature engineering and scaling)
# -----------------------------
features_after = X_train_processed.columns.tolist()
print("\nFeatures After Feature Creation (Final Processed Features):")
display(features_after)

# -----------------------------
# Identify added or transformed features
# -----------------------------
added_or_transformed_features = [feature for feature in features_after if feature not in features_before]
print("\nFeatures Added or Transformed During Feature Creation:")
display(added_or_transformed_features)

# -----------------------------
# Identify original features that were kept
# -----------------------------
kept_original_features = [feature for feature in features_after if feature in features_before]
print("\nOriginal Numerical Features Kept in the Final Set:")
display(kept_original_features)

# -----------------------------
# Explanation of added/engineered features
# -----------------------------
print("\nExplanation of Added/Transformed Features:")
print("- RelativeLoudness: Derived from AudioLoudness.")
print("- TrackDurationMs: Log-transformed to reduce skewness.")
print("- MoodScore_Quartile, RelativeLoudness_Quartile, TrackDurationBucket: Binned/categorized features.")
print("- Energy_x_RhythmScore, MoodScore_x_RhythmScore, Vocals_vs_Instrument, Energy_minus_AcousticQuality: Interaction features.")
print("- NormalizedLoudness, LiveEnergy: Domain-specific engineered features.")
print("\nâœ… This summary shows the impact of feature engineering and what the models will learn from.")



# Select features for correlation analysis from the scaled training data
features_for_corr = train_ps_processed.drop('id', axis=1)
target_for_corr = y_train_processed

# Calculate the correlation matrix
correlation_matrix_scaled = features_for_corr.corr()

# Display the head of the correlation matrix
print("Correlation Matrix of Scaled Features:")
display(correlation_matrix_scaled.head())


def find_highly_correlated_pairs(corr_matrix, threshold):
    """
    Finds pairs of features with an absolute correlation greater than the threshold.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix.
        threshold (float): The absolute correlation threshold.

    Returns:
        list: A list of tuples, where each tuple contains a pair of highly
              correlated feature names.
    """
    highly_correlated_pairs = []
    # Use stack to easily iterate through unique pairs
    stacked_corr = corr_matrix.stack()
    # Filter for absolute correlation > threshold, excluding self-correlation (diagonal)
    highly_correlated_series = stacked_corr[abs(stacked_corr) > threshold]
    highly_correlated_series = highly_correlated_series[highly_correlated_series != 1.0]

    # Convert to list of tuples, ensuring each pair is listed only once
    for (col1, col2), value in highly_correlated_series.items():
        if (col2, col1) not in highly_correlated_pairs:
             highly_correlated_pairs.append((col1, col2))

    return highly_correlated_pairs

highly_correlated_features = find_highly_correlated_pairs(correlation_matrix_scaled, 0.9)

print("Highly correlated feature pairs (absolute correlation > 0.9):")
if highly_correlated_features:
    for pair in highly_correlated_features:
        print(pair)
else:
    print("No highly correlated feature pairs found above the threshold.")


from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1ï¸�âƒ£ Correlation with target
# -----------------------------
corr_with_target = X_train_processed.copy()
corr_with_target['BeatsPerMinute'] = y_train_processed
correlation_matrix = corr_with_target.corr()
target_corr = correlation_matrix['BeatsPerMinute'].drop('BeatsPerMinute').sort_values(ascending=False)

print("--- Top Correlated Features with Target ---")
display(target_corr.head(10))

# -----------------------------
# 2ï¸�âƒ£ Mutual Information
# -----------------------------
mi_scores = mutual_info_regression(X_train_processed, y_train_processed, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X_train_processed.columns).sort_values(ascending=False)

print("\n--- Top Features by Mutual Information ---")
display(mi_scores_series.head(10))

# -----------------------------
# 3ï¸�âƒ£ Random Forest Feature Importance
# -----------------------------
rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(X_train_processed, y_train_processed)
rf_importances = pd.Series(rf.feature_importances_, index=X_train_processed.columns).sort_values(ascending=False)

print("\n--- Top Features by Random Forest Importance ---")
display(rf_importances.head(10))

# -----------------------------
# 4ï¸�âƒ£ Lasso Feature Selection
# -----------------------------
lasso = LassoCV(cv=5, random_state=42, max_iter=500)
lasso.fit(X_train_processed, y_train_processed)
lasso_importances = pd.Series(np.abs(lasso.coef_), index=X_train_processed.columns).sort_values(ascending=False)

print("\n--- Top Features by Lasso ---")
display(lasso_importances.head(10))

# -----------------------------
# 5ï¸�âƒ£ Combine Results
# -----------------------------
feature_scores = pd.DataFrame({
    'Correlation': target_corr,
    'MutualInfo': mi_scores_series,
    'RandomForest': rf_importances,
    'Lasso': lasso_importances
})

# Normalize scores to 0-1 range
feature_scores_normalized = feature_scores.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Average normalized scores to create a combined score
feature_scores_normalized['CombinedScore'] = feature_scores_normalized.mean(axis=1)
feature_scores_normalized = feature_scores_normalized.sort_values(by='CombinedScore', ascending=False)

print("\n--- Top Features by Combined Score ---")
display(feature_scores_normalized.head(15))

# -----------------------------
# 6ï¸�âƒ£ Optional Visualization
# -----------------------------
plt.figure(figsize=(12,6))
sns.barplot(x=feature_scores_normalized['CombinedScore'].head(15), y=feature_scores_normalized.index[:15], palette='viridis')
plt.title("Top 15 Features by Combined Score")
plt.xlabel("Combined Score")
plt.ylabel("Feature")
plt.show()



import pandas as pd

print("\n--- 3.7 Final Data Preparation ---")

# âœ… Choose top N features (you can tune this)
top_features = feature_scores_normalized.sort_values(by="CombinedScore", ascending=False).head(50).index.tolist()

print(f"\nNumber of selected top features: {len(top_features)}")
print("Top selected features for modeling:")
display(top_features)

train_ps_processed = pd.merge(
    train_ps_processed,
    train_ps_original[['id', 'BeatsPerMinute']],
    on='id',
    how='left'
)

# âœ… Final train/test splits
X_train_processed = train_ps_processed[top_features].copy()
y_train_processed = train_ps_processed['BeatsPerMinute'].copy()

X_test_processed = test_ps_processed[top_features].copy()

print("\nShapes of Final Prepared Data:")
print("X_train_processed:", X_train_processed.shape)
print("y_train_processed:", y_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)

print("\nHead of Final X_train_processed:")
display(X_train_processed.head())

print("\nHead of Final X_test_processed:")
display(X_test_processed.head())



# 4. Train-Test Split for Beats Prediction

from sklearn.model_selection import train_test_split

# Our preprocessed features and target
X = X_train_processed
y = y_train_processed

# Split into train/validation sets (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Separate Test set (from competition):", X_test_processed.shape)



# 4.2 Base Model Definitions 
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Ridge Regression
lr_model = Ridge(alpha=1.0, random_state=42)  

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,       
    max_depth=5,           
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# LightGBM
lgbm_model = lgb.LGBMRegressor(
    n_estimators=1000,      # fewer boosting rounds
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose: -1
)

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# CatBoost
cat_model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=5,
    eval_metric='RMSE',
    verbose=False,
    random_state=42
)

# Combine into a dictionary for easy use in CV or stacking
base_models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model,
    "XGBoost": xgb_model,
    "CatBoost": cat_model
}





# 4.3 Base Model Evaluation (Professional & Advanced)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"Cross-validation strategy defined with {kf.get_n_splits()} folds.\n")

# Initialize storage for OOF predictions and fold scores
oof_preds = {name: np.zeros(y_train_processed.shape[0]) for name in base_models.keys()}
fold_rmse_scores = {name: [] for name in base_models.keys()}

print("Initialized OOF prediction arrays and fold score storage.\n")

# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_processed, y_train_processed)):
    print(f"--- Fold {fold+1}/{kf.get_n_splits()} ---")

    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_tr, y_val = y_train_processed.iloc[train_idx], y_train_processed.iloc[val_idx]

    for model_name, model in base_models.items():
        # Clone model for fold to prevent data leakage
        if model_name in ["LightGBM", "XGBoost", "CatBoost"]:
            # For boosting libraries, create fresh instances with same params
            model_fold = model.__class__(**model.get_params())
        else:
            from sklearn.base import clone
            model_fold = clone(model)

        # Fit model
        if model_name == "XGBoost":
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            params = model_fold.get_params()
            params_for_train = {k:v for k,v in params.items() if k not in ['n_estimators','eval_metric']}
            num_boost_round = params.get('n_estimators', 100)

            bst = xgb.train(
                params=params_for_train,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval,'eval')],
                verbose_eval=False
            )
            preds = bst.predict(dval)

        elif model_name == "LightGBM":
            model_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)],
                
            )
            preds = model_fold.predict(X_val)

        elif model_name == "CatBoost":
            model_fold.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            preds = model_fold.predict(X_val)

        else:
            model_fold.fit(X_tr, y_tr)
            preds = model_fold.predict(X_val)

        # Store OOF predictions
        oof_preds[model_name][val_idx] = preds

        # Calculate RMSE for this fold
        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_rmse_scores[model_name].append(fold_rmse)

        print(f"{model_name} Fold {fold+1} RMSE: {fold_rmse:.4f}")

    print("\n")

# Summary of CV results
overall_rmse = {}
print("=== Overall Cross-Validation Results (Mean RMSE Â± Std) ===")
for model_name, scores in fold_rmse_scores.items():
    mean_rmse = np.mean(scores)
    std_rmse = np.std(scores)
    overall_rmse[model_name] = mean_rmse
    print(f"{model_name}: {mean_rmse:.4f} Â± {std_rmse:.4f}")

print("\nOOF predictions are ready for stacking or meta-model training.")



pip install --upgrade xgboost


import pandas as pd
import numpy as np

# Combine OOF predictions
X_meta_train = pd.DataFrame({
    'lr_oof_preds': oof_preds['Linear Regression'],
    'rf_oof_preds': oof_preds['Random Forest'],
    'lgbm_oof_preds': oof_preds['LightGBM'],
    'xgb_oof_preds': oof_preds['XGBoost'],
    'cat_oof_preds': oof_preds['CatBoost']
})

y_meta_train = y_train_processed.values  # target variable


from sklearn.linear_model import Ridge

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_meta_train)



import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import catboost as cb

print("Training base models on full training data and generating test set predictions...")

lr_model_full = LinearRegression()
lr_model_full.fit(X_train_processed, y_train_processed)
print("Linear Regression model trained on full data.")

rf_model_full = RandomForestRegressor(**rf_model.get_params())
rf_model_full.fit(X_train_processed, y_train_processed)
print("Random Forest model trained on full data.")

lgbm_model_full = lgb.LGBMRegressor(**lgbm_model.get_params())
lgbm_model_full.fit(X_train_processed, y_train_processed)
print("LightGBM model trained on full data.")

xgb_model_full_for_test = xgb.XGBRegressor(**xgb_model.get_params())
xgb_model_full_for_test.fit(X_train_processed, y_train_processed, eval_set=[], verbose=False)
print("XGBoost model trained on full data.")

cat_model_full = cb.CatBoostRegressor(**cat_model.get_params())
cat_model_full.fit(X_train_processed, y_train_processed)
print("CatBoost model trained on full data.")

test_preds_lr = lr_model_full.predict(X_test_processed)
print("Generated test predictions for Linear Regression.")

test_preds_rf = rf_model_full.predict(X_test_processed)
print("Generated test predictions for Random Forest.")

test_preds_lgbm = lgbm_model_full.predict(X_test_processed)
print("Generated test predictions for LightGBM.")

test_preds_xgb = xgb_model_full_for_test.predict(X_test_processed)
print("Generated test predictions for XGBoost.")

test_preds_cat = cat_model_full.predict(X_test_processed)
print("Generated test predictions for CatBoost.")


X_meta_test = pd.DataFrame({
    'lr_oof_preds': test_preds_lr,
    'rf_oof_preds': test_preds_rf,
    'lgbm_oof_preds': test_preds_lgbm,
    'xgb_oof_preds': test_preds_xgb,
    'cat_oof_preds': test_preds_cat
})

print("\nCombined base model test predictions for meta-model:")
display(X_meta_test.head())

print("\nGenerating final ensemble predictions using the meta-model...")
final_ensemble_predictions = meta_model.predict(X_meta_test)
print("Final ensemble predictions generated.")

print("\nSample of final ensemble predictions:")
display(final_ensemble_predictions[:10])

overall_rmse_ensemble = np.sqrt(mean_squared_error(y_train_processed, meta_model.predict(X_meta_train)))
overall_r2_ensemble = r2_score(y_train_processed, meta_model.predict(X_meta_train))


print(f"\nStacked Ensemble Model Performance on Training Data (Out-of-Fold):")
print(f"Overall Root Mean Squared Error (RMSE): {overall_rmse_ensemble}")
print(f"Overall RÂ² Score: {overall_r2_ensemble}")


import pandas as pd

def verify_submission(submission_file, sample_submission):
    """Perform comprehensive verification of submission format"""
    verification = pd.read_csv(submission_file)

    print("SUBMISSION VERIFICATION:")
    print(f"1. Submission shape: {verification.shape}")
    print(f"   Expected shape: {sample_submission.shape}")

    print(f"\n2. Submission columns: {verification.columns.tolist()}")
    print(f"   Expected columns: {sample_submission.columns.tolist()}")

    columns_match = verification.columns.tolist() == sample_submission.columns.tolist()
    print(f"\n3. Columns match exactly: {'âœ… YES' if columns_match else 'â�Œ NO'}")

    id_col = sample_submission.columns[0]
    id_match = set(verification[id_col]) == set(sample_submission[id_col])
    print(f"\n4. ID values match sample: {'âœ… YES' if id_match else 'â�Œ NO'}")

    target_col = sample_submission.columns[1]
    print(f"\n5. Target column statistics:")
    print(f"   Min: {verification[target_col].min():.2f}")
    print(f"   Max: {verification[target_col].max():.2f}")
    print(f"   Mean: {verification[target_col].mean():.2f}")
    print(f"   Std: {verification[target_col].std():.2f}")

    if columns_match and id_match:
        print("\nâœ… SUBMISSION FORMAT LOOKS CORRECT! Ready to upload.")
    else:
        print("\nâ�Œ SUBMISSION FORMAT HAS ISSUES! Please fix before uploading.")

    return verification


test_features_ps = train_ps_processed.drop('id', axis=1)

submission_df = pd.DataFrame({'id': test_ps_original['id'], 'BeatsPerMinute': final_ensemble_predictions})

submission_file = 'submission.csv'
sample_submission_file = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

try:
    sample_submission = pd.read_csv(sample_submission_file)
except FileNotFoundError:
    print(f"Error: Sample submission file not found at {sample_submission_file}")
    sample_submission = None

submission_df.to_csv(submission_file, index=False)

if sample_submission is not None:
  final_verification = verify_submission(submission_file, sample_submission)

  if final_verification.columns.tolist() != sample_submission.columns.tolist():
      print("\nAttempting to fix column names one last time based on sample submission...")
      final_verification.columns = sample_submission.columns
      final_verification.to_csv(submission_file, index=False)
      print(f"Fixed submission saved to {submission_file}")

      print("\nVerifying after attempting column name fix:")
      verify_submission(submission_file, sample_submission)


print("\nSubmission file for Playground Series dataset created:")
display(submission_df.head())

