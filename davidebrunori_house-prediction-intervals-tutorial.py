import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder # Not explicitly used if relying on 'category' dtype
import matplotlib.pyplot as plt
import seaborn as sns

# Competition specific
ALPHA = 0.1 # For 90% prediction interval (100 - 90)/100

# --- 1. Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv("dataset.csv")
    test_df = pd.read_csv("test.csv")
    sample_submission = pd.read_csv("sample_submission.csv")
except FileNotFoundError:
    print("Make sure dataset.csv, test.csv, and sample_submission.csv are in the same directory or adjust paths.")
    # Fallback for Kaggle environment
    train_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
    test_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# --- 2. EDA & Preprocessing ---
print("\nPreprocessing...")

# Target variable
TARGET = 'sale_price'

# Log transform the target (common for prices)
train_df[TARGET] = np.log1p(train_df[TARGET])

def preprocess(df):
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # --- Date Features ---
    # Convert 'sale_date' to datetime objects to extract useful components
    df_copy['sale_date'] = pd.to_datetime(df_copy['sale_date'])
    df_copy['sale_year'] = df_copy['sale_date'].dt.year
    df_copy['sale_month'] = df_copy['sale_date'].dt.month
    df_copy['sale_dayofyear'] = df_copy['sale_date'].dt.dayofyear # Day number within the year (1-366)
    df_copy['sale_dayofweek'] = df_copy['sale_date'].dt.dayofweek # Day of the week (Monday=0, Sunday=6)
    df_copy = df_copy.drop('sale_date', axis=1) # Original date string no longer needed

    # --- Simple Feature Engineering ---
    # Creating new features from existing ones can often improve model performance.

    # Age of the property at the time of sale
    df_copy['age_at_sale'] = df_copy['sale_year'] - df_copy['year_built']
    df_copy['age_at_sale'] = df_copy['age_at_sale'].clip(lower=0) # Ensure age isn't negative (e.g., if sale_year < year_built due to data error)

    # Age since renovation at the time of sale
    df_copy['reno_age_at_sale'] = 0 # Initialize with 0
    mask_reno = df_copy['year_reno'] > 0 # Identify properties that have been renovated
    df_copy.loc[mask_reno, 'reno_age_at_sale'] = df_copy.loc[mask_reno, 'sale_year'] - df_copy.loc[mask_reno, 'year_reno']
    df_copy['reno_age_at_sale'] = df_copy['reno_age_at_sale'].clip(lower=0) # Ensure non-negative
    df_copy['was_renovated'] = (df_copy['year_reno'] > 0).astype(int) # Binary flag: 1 if renovated, 0 otherwise

    # Combined square footage and per-story square footage
    df_copy['total_sqft'] = df_copy['sqft'] + df_copy['sqft_fbsmt'] # Total living area
    df_copy['sqft_per_story'] = df_copy['sqft'] / df_copy['stories'].replace(0,1) # Avoid division by zero if 'stories' is 0

    # --- Handling Missing Values (Imputation) ---

    # Impute 'sale_nbr' (sales number, likely categorical or ordinal)
    # Note: For test data, ideally, we'd use the median from the *training* data.
    # This implementation uses the median of the current DataFrame for simplicity.
    if 'sale_nbr' in df_copy.columns:
         median_sale_nbr = df_copy['sale_nbr'].median()
         # This part about 'sale_nbr_train_median' is a bit complex for a simple preprocess function.
         # A more robust way is to calculate medians on the training set once and pass them to this function.
         if df_copy['sale_nbr'].isnull().sum() > 0:
             if 'sale_price' in df.columns: # Heuristic to check if it's train_df
                 # Storing it here isn't directly used later unless explicitly passed.
                 # Better to compute and store train medians outside and pass as arguments.
                 df_copy['sale_nbr_train_median_placeholder'] = median_sale_nbr
         df_copy['sale_nbr'] = df_copy['sale_nbr'].fillna(median_sale_nbr)


    # --- Handling Categorical Features ---
    # Explicitly define columns that should be treated as categorical
    categorical_cols_explicit = ['sale_warning', 'join_status', 'city', 'zoning',
                                 'subdivision', 'present_use', 'submarket']

    for col in categorical_cols_explicit:
        if col in df_copy.columns:
            # Convert to pandas 'category' dtype. LightGBM can handle this efficiently.
            df_copy[col] = df_copy[col].astype('category')

    # --- Impute Missing Numerical Features ---
    numerical_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    # Ensure TARGET is not processed if it's not present (e.g. in test_df)
    if TARGET in numerical_cols and TARGET not in df.columns: # This case shouldn't occur given the logic
        pass
    elif TARGET in numerical_cols and TARGET in df.columns: # If TARGET is present (train_df)
         numerical_cols.remove(TARGET) # Don't impute the target variable

    if 'id' in numerical_cols: # 'id' is an identifier, not a feature
        numerical_cols.remove('id')

    for col in numerical_cols:
        # Impute missing numerical values with the median of that column.
        # Median is generally preferred over mean as it's less sensitive to outliers.
        # Again, for test data, train medians are preferred.
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    return df_copy


# Preprocess train and test
# For a more robust imputation, medians/modes from X_train (or full X before split) should be stored and applied to X_val and X_test_final
train_processed = preprocess(train_df)
test_processed = preprocess(test_df)

# Separate features (X) and target (y) from the training data
train_labels = train_processed[TARGET] # This is our log-transformed sale_price
train_ids = train_processed['id']      # Store IDs for potential later use
test_ids = test_processed['id']        # Store IDs for the submission file

# Drop the target and id columns from the feature set
X = train_processed.drop([TARGET, 'id'], axis=1)
# Drop id from the test set (target is not present)
X_test_final = test_processed.drop(['id'], axis=1)


# --- Align Columns between Training and Test Sets ---
# This is a CRITICAL step. Models trained on a specific set of features (columns)
# expect to see the exact same features (and in the same order) when making predictions.

# Find columns in training set (X) but not in test set (X_test_final)
missing_cols_test = set(X.columns) - set(X_test_final.columns)
for c in missing_cols_test:
    print(f"Column '{c}' is missing in the test set. Adding and filling.")
    # If the column was categorical in X, fill with its mode (most frequent value) or a placeholder.
    if X[c].dtype.name == 'category':
        # Ensure the new category (mode or 'missing') is known to X_test_final's category type
        mode_val = X[c].mode()[0] if not X[c].mode().empty else 'missing' # Get mode, or 'missing' if mode is empty
        X_test_final[c] = mode_val # Add the column and fill with mode
        # Important: The new column in X_test_final needs to be of type category
        # and ideally have the same categories as X[c].
        # For simplicity here, we cast to category. More robust handling might involve
        # ensuring X[c].cat.categories are used for X_test_final[c].
        current_categories = X[c].cat.categories.tolist()
        if mode_val not in current_categories:
            current_categories.append(mode_val)
        X_test_final[c] = pd.Categorical(X_test_final[c], categories=current_categories)

    else: # If numerical, fill with its median from the training set X.
        X_test_final[c] = X[c].median()

# Find columns in test set (X_test_final) but not in training set (X)
extra_cols_test = set(X_test_final.columns) - set(X.columns)
for c in extra_cols_test:
    print(f"Column '{c}' is extra in the test set. Removing.")
X_test_final = X_test_final.drop(columns=list(extra_cols_test))

# Ensure the order of columns in X_test_final is the same as in X
X_test_final = X_test_final[X.columns]

# Identify categorical feature names for LightGBM
# LightGBM can handle categorical features directly if told which ones they are.
categorical_features_names = [col for col in X.columns if X[col].dtype.name == 'category']


print(f"Features shape after preprocessing: {X.shape}")
print(f"Test features shape after preprocessing: {X_test_final.shape}")
if X.shape[1] != X_test_final.shape[1]:
    print("WARNING: Mismatch in number of columns between train and test features!")
    print(f"X columns: {X.columns.tolist()}")
    print(f"X_test_final columns: {X_test_final.columns.tolist()}")

print(f"Categorical features for LGBM: {categorical_features_names}")


# --- 3. Winkler Score Implementation ---
def winkler_score(y_true, lower, upper, alpha=ALPHA):
    """
    Calculate the Winkler Score for prediction intervals.

    Args:
        y_true (np.array): True values.
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% PI).

    Returns:
        float: Mean Winkler Score.
    """
    score = np.zeros_like(y_true, dtype=float) # Initialize scores for each prediction

    # Case 1: True value is below the lower bound
    below_lower = y_true < lower
    score[below_lower] = (upper[below_lower] - lower[below_lower]) + (2.0/alpha) * (lower[below_lower] - y_true[below_lower])

    # Case 2: True value is above the upper bound
    above_upper = y_true > upper
    score[above_upper] = (upper[above_upper] - lower[above_upper]) + (2.0/alpha) * (y_true[above_upper] - upper[above_upper])

    # Case 3: True value is within the interval
    within_interval = (~below_lower) & (~above_upper) # Neither below lower nor above upper
    score[within_interval] = upper[within_interval] - lower[within_interval]

    return np.mean(score) # The final score is the average over all predictions

def coverage_metric(y_true, lower, upper):
    """
    Calculate the proportion of true values falling within the prediction interval.
    """
    return np.mean((y_true >= lower) & (y_true <= upper))


print("\nTraining models...")
X_train, X_val, y_train, y_val = train_test_split(X, train_labels, test_size=0.2, random_state=42)

LOWER_QUANTILE = ALPHA / 2.0  # e.g., 0.1 / 2 = 0.05
UPPER_QUANTILE = 1.0 - (ALPHA / 2.0) # e.g., 1.0 - 0.05 = 0.95

# Common LightGBM parameters
params = {
    'objective': 'quantile',    #  <-- Key for quantile regression
    'metric': 'quantile',       #  <-- Metric for evaluation during training
    'boosting_type': 'gbdt',
    'n_estimators': 500,        # Number of trees (can be stopped early)
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,               # Use all available cores
    'verbose': -1,              # Suppress informational messages from LightGBM
}

print("Training lower quantile model...")
params_lower = params.copy()
params_lower['alpha'] = LOWER_QUANTILE # Tell LGBM to target the lower quantile
model_lower = lgb.LGBMRegressor(**params_lower)
model_lower.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile', # Monitor quantile loss on validation set
                callbacks=[lgb.early_stopping(100, verbose=False)], # Stop if no improvement for 100 rounds
                categorical_feature=categorical_features_names) # Specify categorical features

print("Training upper quantile model...")
params_upper = params.copy()
params_upper['alpha'] = UPPER_QUANTILE # Tell LGBM to target the upper quantile
model_upper = lgb.LGBMRegressor(**params_upper)
model_upper.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile',
                callbacks=[lgb.early_stopping(100, verbose=False)],
                categorical_feature=categorical_features_names)


# --- 5. Validation ---
print("\nValidating models...")
val_preds_lower_log = model_lower.predict(X_val)
val_preds_upper_log = model_upper.predict(X_val)

# Ensure upper bound prediction is not less than lower bound prediction
# Add a small epsilon to prevent lower and upper being exactly equal, which might be problematic later.
val_preds_upper_log = np.maximum(val_preds_upper_log, val_preds_lower_log + 1e-6)

# Inverse transform predictions and true values to original scale
val_preds_lower = np.expm1(val_preds_lower_log)
val_preds_upper = np.expm1(val_preds_upper_log)
y_val_orig = np.expm1(y_val) # y_val is a pandas Series, np.expm1 works directly

# Post-processing: Prices cannot be negative, and upper must be >= lower
val_preds_lower = np.maximum(0, val_preds_lower) # Clip at 0
val_preds_upper = np.maximum(0, val_preds_upper) # Clip at 0
val_preds_upper = np.maximum(val_preds_lower, val_preds_upper) # Ensure upper >= lower after all transformations


# Calculate Winkler score and coverage on the validation set
# .values converts pandas Series to NumPy array, which these functions expect
val_winkler = winkler_score(y_val_orig.values, val_preds_lower, val_preds_upper, alpha=ALPHA)
val_coverage = coverage_metric(y_val_orig.values, val_preds_lower, val_preds_upper)

print(f"Validation Winkler Score: {val_winkler:.4f}")
print(f"Validation Coverage: {val_coverage:.4f} (Target: {1-ALPHA:.2f})")

# Plotting a sample of prediction intervals and true values (Corrected for clarity)
plt.figure(figsize=(12, 6))
sample_size = 100 # Number of samples to plot
if len(y_val_orig) < sample_size: # Handle cases with small validation sets
    sample_size = len(y_val_orig)

# Get random indices for sampling
sample_indices = np.random.choice(len(y_val_orig), size=sample_size, replace=False)

# Get sampled data (use .iloc for pandas Series before .values if needed, or directly if already numpy arrays)
# y_val_orig is a Series, so .iloc is appropriate for selecting rows by position
y_val_orig_sample = y_val_orig.iloc[sample_indices].values
# val_preds_lower and val_preds_upper are already numpy arrays from model.predict()
val_preds_lower_sample = val_preds_lower[sample_indices]
val_preds_upper_sample = val_preds_upper[sample_indices]

# Calculate midpoints and half-widths for error bars
mid_points_sample = (val_preds_lower_sample + val_preds_upper_sample) / 2
half_widths_sample = (val_preds_upper_sample - val_preds_lower_sample) / 2
# Ensure half_widths are non-negative (they should be if upper >= lower)
half_widths_sample = np.maximum(0, half_widths_sample)

# X-axis for plotting (simple enumeration of samples)
plot_x_axis = np.arange(len(sample_indices))

plt.errorbar(plot_x_axis, mid_points_sample,
             yerr=half_widths_sample,
             fmt='none', # Do not plot markers for midpoints, only the bars
             ecolor='lightgray', elinewidth=3, capsize=3, label=f'{int((1-ALPHA)*100)}% Prediction Interval')
plt.plot(plot_x_axis, y_val_orig_sample,
         'ro', markersize=4, label='True Value') # Plot true values as red dots

plt.title(f'Sample of {sample_size} Validation PIs and True Values')
plt.xlabel('Sample Index (Randomly Chosen)')
plt.ylabel('Sale Price (Original Scale)')
plt.legend()
plt.tight_layout() # Adjust plot to prevent labels from overlapping
plt.show()


# --- 6. Train on Full Data & Predict on Test ---
print("\nTraining final models on full dataset...")
# Note: We re-initialize the models.
# For a more robust approach, you might use the 'best_iteration' from the early stopping
# during validation to set n_estimators, or simply rely on the n_estimators in params.
# Here, we fit for the n_estimators specified in params, without early stopping on the full data.

model_lower_final = lgb.LGBMRegressor(**params_lower) # Uses params_lower with alpha for lower quantile
model_lower_final.fit(X, train_labels, categorical_feature=categorical_features_names)

model_upper_final = lgb.LGBMRegressor(**params_upper) # Uses params_upper with alpha for upper quantile
model_upper_final.fit(X, train_labels, categorical_feature=categorical_features_names)

print("Predicting on test set...")
test_preds_lower_log = model_lower_final.predict(X_test_final)
test_preds_upper_log = model_upper_final.predict(X_test_final)

# Ensure upper bound prediction is not less than lower bound prediction
test_preds_upper_log = np.maximum(test_preds_upper_log, test_preds_lower_log + 1e-6)

# Inverse transform predictions to original scale
pi_lower = np.expm1(test_preds_lower_log)
pi_upper = np.expm1(test_preds_upper_log)

# Post-processing: Prices cannot be negative, and upper must be >= lower
pi_lower = np.maximum(0, pi_lower) # Clip at 0
pi_upper = np.maximum(0, pi_upper) # Clip at 0
pi_upper = np.maximum(pi_lower, pi_upper) # Ensure upper >= lower


# --- 7. Create Submission File ---
print("\nCreating submission file...")
submission_df = pd.DataFrame({
    'id': test_ids,         # From original test.csv
    'pi_lower': pi_lower,   # Our predicted lower bounds
    'pi_upper': pi_upper    # Our predicted upper bounds
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
print("Sample of the submission file:")
print(submission_df.head())

print("\nDescriptive statistics of prediction intervals on test set:")
print(f"Mean Lower Bound: {submission_df['pi_lower'].mean():.2f}")
print(f"Mean Upper Bound: {submission_df['pi_upper'].mean():.2f}")
print(f"Mean Interval Width: {(submission_df['pi_upper'] - submission_df['pi_lower']).mean():.2f}")
print(f"Median Lower Bound: {submission_df['pi_lower'].median():.2f}") # Added Median
print(f"Median Upper Bound: {submission_df['pi_upper'].median():.2f}") # Added Median
print(f"Median Interval Width: {(submission_df['pi_upper'] - submission_df['pi_lower']).median():.2f}")

