import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna # For hyperparameter tuning
import gc # Garbage collection
import warnings
import os # To check for files

warnings.filterwarnings('ignore')

# --- Configuration ---
TARGET = 'Listening_Time_minutes'
N_SPLITS = 7 # Number of folds for cross-validation
N_TRIALS_OPTUNA = 6 # Increased Optuna trials for better search
RANDOM_SEED = 42
USE_OPTUNA = True # Set to False to use default params and skip tuning
USE_GPU = False # Will be set to True automatically if GPU detected and usable by LGBM

# Check for GPU availability for LightGBM
try:
    # A simple check: try creating a GPU model instance
    # Note: This requires LightGBM to be compiled with GPU support!
    lgb.LGBMRegressor(device='gpu')
    print("GPU detected and LightGBM GPU support found. Enabling GPU usage.")
    USE_GPU = True
except Exception as e:
    print(f"GPU not detected or LightGBM GPU support not available ({e}). Using CPU.")
    USE_GPU = False


# --- Data Loading Function ---
def load_data(base_path="/kaggle/input/playground-series-s5e4/"):
    """Loads train, test, and submission files from the base path."""
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")
    sample_sub_path = os.path.join(base_path, "sample_submission.csv")

    if not all(os.path.exists(p) for p in [train_path, test_path, sample_sub_path]):
        print(f"Error: Could not find required files in path: {base_path}")
        print(f"Checked paths:\n - {train_path}\n - {test_path}\n - {sample_sub_path}")
        raise FileNotFoundError("Competition data files not found. Please ensure they are in the correct directory.")

    print(f"Loading data from: {base_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_sub_path)
    print("Data loaded successfully.")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    # Basic check for expected test size (adjust 250000 if competition differs)
    if test_df.shape[0] < 10000: # A sanity check threshold
        print(f"WARNING: Test data has only {test_df.shape[0]} rows. Expected a much larger number for most competitions.")
    return train_df, test_df, sample_submission

# --- Determine Data Path (Adapt for Kaggle/Local) ---
# Standard Kaggle input directory structure
kaggle_input_path = "/kaggle/input/playground-series-s5e4" # <-- !!! ADJUST THIS SLUG IF NEEDED !!!
local_path = "." # Current directory

if os.path.exists(os.path.join(kaggle_input_path, "train.csv")):
    DATA_PATH = kaggle_input_path
elif os.path.exists(os.path.join(local_path, "train.csv")):
    DATA_PATH = local_path
else:
    print("Error: Could not determine data path for Kaggle or local setup.")
    print(f"Looked in: '{kaggle_input_path}' and '{local_path}'")
    # Raise error immediately if data path isn't clear, preventing dummy data issues
    raise FileNotFoundError("Cannot determine data directory. Place data in '.' or check Kaggle path.")


# --- Load Data ---
train_df_raw, test_df_raw, sample_submission = load_data(DATA_PATH)


# --- Preprocessing Function ---
def preprocess(df_train, df_test):
    print("Starting preprocessing...")
    train_ids = df_train['id']
    test_ids = df_test['id']
    n_train = len(df_train)

    # Store original lengths from test set for later clipping
    # Ensure 'id' column exists before using it
    if 'id' not in df_test.columns:
         raise KeyError("'id' column not found in the test dataframe.")
    test_lengths_orig = df_test[['id', 'Episode_Length_minutes']].copy()


    if TARGET in df_train.columns:
        y_train = df_train[TARGET]
        # Drop ID and Target from train
        df_train = df_train.drop(columns=[col for col in ['id', TARGET] if col in df_train.columns])
    else:
         raise KeyError(f"Target column '{TARGET}' not found in the training dataframe.")

    # Drop ID from test
    df_test = df_test.drop(columns=[col for col in ['id'] if col in df_test.columns])


    # Combine for processing
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    print(f"Combined shape before processing: {df_combined.shape}")

    # === Imputation & Outlier Handling (Train-based) ===
    print("Handling missing values and outliers...")

    # 1. Handle Specific Missing Value (Ads) - Use Train Median
    if 'Number_of_Ads' in df_train.columns:
        ads_median = df_train['Number_of_Ads'].median()
        df_combined['Number_of_Ads'] = df_combined['Number_of_Ads'].fillna(ads_median)
    else:
        print("Warning: 'Number_of_Ads' column not found. Skipping related processing.")


    # 2. Outlier Capping (using train data quantiles to prevent leakage)
    # Define caps safely, checking if columns exist
    len_cap_upper, len_cap_lower, ads_cap_upper = None, None, None
    if 'Episode_Length_minutes' in df_train.columns:
        len_cap_upper = df_train['Episode_Length_minutes'].quantile(0.995)
        len_cap_lower = df_train['Episode_Length_minutes'].quantile(0.005)
        df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].clip(
            lower=max(1.0, len_cap_lower) if pd.notna(len_cap_lower) else 1.0,
            upper=len_cap_upper if pd.notna(len_cap_upper) else None
        )
        print(f"Length capped between ~{len_cap_lower:.2f} and ~{len_cap_upper:.2f} (based on train quantiles)")

    if 'Number_of_Ads' in df_combined.columns: # Use combined column as it might have been created/filled
        ads_cap_upper = df_train['Number_of_Ads'].quantile(0.995) # Cap based on train
        df_combined['Number_of_Ads'] = df_combined['Number_of_Ads'].clip(
             lower=0,
             upper=ads_cap_upper if pd.notna(ads_cap_upper) else None)
        print(f"Ads capped at {ads_cap_upper:.2f} (based on train quantile)")

    # Logical caps for percentages
    if 'Host_Popularity_percentage' in df_combined.columns:
        df_combined['Host_Popularity_percentage'] = df_combined['Host_Popularity_percentage'].clip(lower=0, upper=100.0)
    if 'Guest_Popularity_percentage' in df_combined.columns:
        df_combined['Guest_Popularity_percentage'] = df_combined['Guest_Popularity_percentage'].clip(lower=0, upper=100.0)


    # === Feature Engineering - Part 1 (Before Complex Imputation) ===
    print("Starting feature engineering (Part 1)...")

    # Has_Guest feature (captures info before imputation)
    if 'Guest_Popularity_percentage' in df_combined.columns:
        df_combined['Has_Guest'] = (~df_combined['Guest_Popularity_percentage'].isnull()).astype(int)
    else:
        df_combined['Has_Guest'] = 0 # Assume no guest if column is missing
        print("Warning: 'Guest_Popularity_percentage' column not found. Setting 'Has_Guest' to 0.")


    # Simple text length features
    for col in ['Podcast_Name', 'Episode_Title']:
        if col in df_combined.columns:
            df_combined[col + '_Len'] = df_combined[col].astype(str).str.len()
        else:
             print(f"Warning: Column '{col}' not found for length feature generation.")

    # Basic Time Features
    if 'Publication_Day' in df_combined.columns:
        df_combined['Publication_Day'] = df_combined['Publication_Day'].astype('category')
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Ensure all categories are present, handling potential unseen values gracefully
        current_categories = df_combined['Publication_Day'].cat.categories
        valid_categories = [day for day in all_days if day in current_categories]
        df_combined['Publication_Day'] = df_combined['Publication_Day'].cat.set_categories(all_days, ordered=False) # Keep order False if not inherently ordinal
        df_combined['DayOfWeek'] = df_combined['Publication_Day'].cat.codes # Monday=0, Sunday=6 (NaNs become -1)
        df_combined['DayOfWeek'] = df_combined['DayOfWeek'].replace(-1, df_combined['DayOfWeek'].mode()[0]) # Impute NaNs with mode
        df_combined['Is_Weekend'] = df_combined['DayOfWeek'].isin([5, 6]).astype(int) # Saturday=5, Sunday=6
    else:
        print("Warning: 'Publication_Day' column not found. Skipping DayOfWeek/Is_Weekend features.")


    if 'Publication_Time' in df_combined.columns:
        time_map = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
        df_combined['TimeOfDay'] = df_combined['Publication_Time'].map(time_map).fillna(0).astype(int) # 0 for unknowns/NaNs
    else:
         print("Warning: 'Publication_Time' column not found. Skipping TimeOfDay feature.")


    # Sentiment Mapping
    if 'Episode_Sentiment' in df_combined.columns:
        sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        df_combined['Sentiment_Score'] = df_combined['Episode_Sentiment'].map(sentiment_map).fillna(0).astype(int) # 0 for unknowns/NaNs
    else:
         print("Warning: 'Episode_Sentiment' column not found. Skipping Sentiment_Score feature.")


    # === Imputation (Guest Pop Median & Length KNN) ===
    print("Performing imputation...")
    # Impute Guest Popularity (median is acceptable here after creating Has_Guest)
    if 'Guest_Popularity_percentage' in df_combined.columns and 'Guest_Popularity_percentage' in df_train.columns:
        guest_pop_median = df_train['Guest_Popularity_percentage'].median()
        df_combined['Guest_Popularity_percentage'] = df_combined['Guest_Popularity_percentage'].fillna(guest_pop_median)
    elif 'Guest_Popularity_percentage' in df_combined.columns:
         # If only in combined (means it was only in test), fill with a neutral value like 0 or 50
         df_combined['Guest_Popularity_percentage'] = df_combined['Guest_Popularity_percentage'].fillna(50)
         print("Warning: Guest Pop column missing in train, filled test NaNs with 50.")

    # Impute Episode Length using KNNImputer (more sophisticated)
    if 'Episode_Length_minutes' in df_combined.columns and df_combined['Episode_Length_minutes'].isnull().any():
        print("Performing KNN Imputation for Episode_Length_minutes...")
        # Define potential columns safely
        base_impute_cols = [
            'Host_Popularity_percentage', 'Guest_Popularity_percentage',
            'Number_of_Ads', 'Podcast_Name_Len', 'Episode_Title_Len',
            'DayOfWeek', 'TimeOfDay', 'Sentiment_Score', 'Has_Guest'
        ]
        # Filter list to include only columns that actually exist in df_combined
        impute_cols_for_knn = [col for col in base_impute_cols if col in df_combined.columns]

        # Ensure there are columns to impute with
        if not impute_cols_for_knn:
             print("Warning: No suitable columns found for KNN imputation context. Filling length NaNs with train median.")
             len_median_train = df_train['Episode_Length_minutes'].median() if 'Episode_Length_minutes' in df_train.columns else 30.0 # Fallback value
             df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].fillna(len_median_train)
        else:
            cols_for_imputation = ['Episode_Length_minutes'] + impute_cols_for_knn
            df_imputation_subset = df_combined[cols_for_imputation]

            # Fit KNNImputer ONLY on the training part of the data where length is NOT missing
            knn_imputer = KNNImputer(n_neighbors=7, weights='distance')
            # Get training data subset for fitting (handle case where train might not have length)
            if 'Episode_Length_minutes' in df_train.columns:
                train_impute_fit_data = df_imputation_subset.iloc[:n_train].dropna(subset=['Episode_Length_minutes'])
            else:
                train_impute_fit_data = pd.DataFrame() # Empty dataframe

            if not train_impute_fit_data.empty and not train_impute_fit_data[impute_cols_for_knn].isnull().any().any():
                print(f"Fitting KNNImputer on {len(train_impute_fit_data)} non-NaN training rows with columns: {cols_for_imputation}")
                try:
                    knn_imputer.fit(train_impute_fit_data)
                    # Transform the entire combined dataframe subset
                    imputed_values = knn_imputer.transform(df_imputation_subset)
                    df_combined[cols_for_imputation] = imputed_values
                    print("KNN Imputation complete.")
                except Exception as e:
                    print(f"Error during KNN Imputation: {e}. Falling back to median imputation for length.")
                    len_median_train = df_train['Episode_Length_minutes'].median() if 'Episode_Length_minutes' in df_train.columns else 30.0
                    df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].fillna(len_median_train)

            else:
                print("Warning: Not enough valid data or columns for KNNImputer fitting. Filling length NaNs with train median.")
                len_median_train = df_train['Episode_Length_minutes'].median() if 'Episode_Length_minutes' in df_train.columns else 30.0
                df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].fillna(len_median_train)
    else:
        print("Episode length has no missing values or column doesn't exist. Skipping KNN imputation.")


    # Re-apply clipping after imputation as KNN might predict outside bounds (only if column exists)
    if 'Episode_Length_minutes' in df_combined.columns:
        # Use previously calculated bounds if available
        final_len_cap_lower = max(1.0, len_cap_lower) if pd.notna(len_cap_lower) else 1.0
        final_len_cap_upper = len_cap_upper if pd.notna(len_cap_upper) else None
        df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].clip(
             lower=final_len_cap_lower,
             upper=final_len_cap_upper
        )
        # Ensure length is strictly positive after all operations
        df_combined['Episode_Length_minutes'] = df_combined['Episode_Length_minutes'].clip(lower=1.0)


    # === Feature Engineering - Part 2 (Post Imputation) ===
    print("Starting feature engineering (Part 2)...")
    epsilon = 1e-6

    # Calculate features only if required columns exist
    if 'Number_of_Ads' in df_combined.columns and 'Episode_Length_minutes' in df_combined.columns:
        df_combined['Ads_per_Minute'] = df_combined['Number_of_Ads'] / (df_combined['Episode_Length_minutes'] + epsilon)

    if 'Host_Popularity_percentage' in df_combined.columns and 'Guest_Popularity_percentage' in df_combined.columns and 'Has_Guest' in df_combined.columns:
        df_combined['Host_Guest_Pop_Diff'] = df_combined['Host_Popularity_percentage'] - (df_combined['Guest_Popularity_percentage'] * df_combined['Has_Guest'])
        df_combined['Pop_Product'] = df_combined['Host_Popularity_percentage'] * (df_combined['Guest_Popularity_percentage'] + epsilon) * df_combined['Has_Guest']

    # Interactions with Length (now imputed and clipped)
    if 'Episode_Length_minutes' in df_combined.columns:
        len_col = 'Episode_Length_minutes'
        if 'Host_Popularity_percentage' in df_combined.columns:
            df_combined['Length_x_HostPop'] = df_combined[len_col] * df_combined['Host_Popularity_percentage']
        if 'Ads_per_Minute' in df_combined.columns:
             df_combined['Length_x_AdsPerMin'] = df_combined[len_col] * df_combined['Ads_per_Minute']
        if 'Sentiment_Score' in df_combined.columns:
             df_combined['Length_x_Sentiment'] = df_combined[len_col] * df_combined['Sentiment_Score']
        if 'Number_of_Ads' in df_combined.columns:
             df_combined['Length_x_AdsNum'] = df_combined[len_col] * df_combined['Number_of_Ads']

    # Frequency Encoding for High Cardinality Features
    # Calculate frequency on the combined set
    for col in ['Podcast_Name', 'Genre']:
        if col in df_combined.columns:
            print(f"Calculating frequency encoding for {col}...")
            freq_map = df_combined[col].value_counts(normalize=True)
            df_combined[col + '_Freq'] = df_combined[col].map(freq_map).fillna(0)


    # === Drop Original/Intermediate Columns ===
    print("Dropping original categorical and text columns...")
    # Define columns to drop based on which ones were actually processed
    cols_to_drop = [
        'Podcast_Name', 'Episode_Title', 'Publication_Day',
        'Publication_Time', 'Episode_Sentiment', 'Genre'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_combined.columns]
    df_combined = df_combined.drop(columns=existing_cols_to_drop)
    print(f"Columns dropped: {existing_cols_to_drop}")
    print(f"Final columns before splitting: {df_combined.columns.tolist()}")


    # === Separate Train and Test ===
    df_train_processed = df_combined.iloc[:n_train].copy()
    df_test_processed = df_combined.iloc[n_train:].copy()

    # Assign original test IDs back
    df_test_processed['id'] = test_ids.values # Use .values to ensure alignment

    print(f"Processed Train shape: {df_train_processed.shape}, Processed Test shape: {df_test_processed.shape}")

    # === Final Clipping of Training Target ===
    if y_train is not None and 'Episode_Length_minutes' in df_train_processed.columns:
        print("Clipping training target based on processed Episode_Length_minutes...")
        # Ensure lengths are aligned with y_train (index should match)
        final_train_lengths = df_train_processed['Episode_Length_minutes']
        # Reindex y_train just in case, though concat should preserve order
        y_train = y_train.reindex(df_train_processed.index)

        y_train_final = np.minimum(y_train.values, final_train_lengths.values)
        y_train_final = np.maximum(y_train_final, 0) # Ensure non-negative
        y_train_final = pd.Series(y_train_final, index=y_train.index, name=TARGET) # Keep index and name
        print(f"Target clipping example (first 5): Original: {y_train.head().values.round(2)}, Clipped: {y_train_final.head().values.round(2)}")
    elif y_train is not None:
         print("Warning: Cannot clip training target as 'Episode_Length_minutes' is missing.")
         y_train_final = y_train # Return original target
    else:
        y_train_final = None # Should not happen if TARGET was found initially


    # --- Get final test lengths for prediction clipping ---
    # Merge processed lengths back to test_ids for later use
    if 'Episode_Length_minutes' in df_test_processed.columns:
        final_test_lengths = df_test_processed[['id', 'Episode_Length_minutes']].copy()
    else:
        # If length column was missing entirely, create a placeholder
        # In this case, clipping won't happen based on length later
        print("Warning: 'Episode_Length_minutes' missing in final test set. Cannot use for prediction clipping.")
        final_test_lengths = pd.DataFrame({'id': test_ids.values, 'Episode_Length_minutes': np.inf}) # Use infinity to avoid clipping


    # Drop id column from test set before returning for training
    if 'id' in df_test_processed.columns:
        df_test_processed = df_test_processed.drop(columns=['id'])


    del df_combined, df_train, df_test, df_imputation_subset; gc.collect() # Removed unused vars from dummy data scenario

    return df_train_processed, df_test_processed, y_train_final, test_ids, final_test_lengths

# --- Apply Preprocessing ---
X, X_test, y, test_ids, test_lengths_df = preprocess(train_df_raw, test_df_raw)

# Clean up raw dataframes
del train_df_raw, test_df_raw; gc.collect()

# Check if any non-numeric columns remain in X or X_test
print("\nChecking dtypes in processed data:")
non_numeric_X = X.select_dtypes(exclude=np.number).columns
non_numeric_X_test = X_test.select_dtypes(exclude=np.number).columns

if not non_numeric_X.empty:
    print(f"Non-numeric columns found in X: {non_numeric_X.tolist()}")
    print(X[non_numeric_X].head())
    raise ValueError("Non-numeric columns found in processed training data X! Check preprocessing steps.")
if not non_numeric_X_test.empty:
     print(f"Non-numeric columns found in X_test: {non_numeric_X_test.tolist()}")
     print(X_test[non_numeric_X_test].head())
     raise ValueError("Non-numeric columns found in processed test data X_test! Check preprocessing steps.")

print("All columns in processed X and X_test are numeric.")


# --- Aggregate Feature Generation Function (to be used inside CV) ---
def create_aggregate_features(df_train_fold, df_valid_fold, df_test_fold, group_col, agg_cols, agg_funcs=['mean', 'std', 'median']):
    """Calculates aggregate features on train fold and maps to valid/test folds."""
    new_feature_names = []
    # Store original indices to ensure alignment after potential merges/sorts
    train_fold_index = df_train_fold.index
    valid_fold_index = df_valid_fold.index
    test_fold_index = df_test_fold.index

    # Create a temporary combined df for calculation to avoid modifying originals directly yet
    temp_train = df_train_fold[[group_col] + agg_cols].copy()

    for col in agg_cols:
        # Skip if the aggregation column is the same as the grouping column
        if col == group_col:
            continue

        # Calculate aggregate on the training part of the fold ONLY
        # Handle potential empty groups or all-NaN groups during aggregation
        agg_map = temp_train.groupby(group_col)[col].agg(agg_funcs)

        # If agg_funcs is a list, agg_map will have MultiIndex columns
        if isinstance(agg_funcs, list):
            agg_map.columns = [f"{group_col}_agg_{func}_{col}" for func in agg_funcs]
            current_new_features = agg_map.columns.tolist()
        else: # If single function string
            new_feature_name = f"{group_col}_agg_{agg_funcs}_{col}"
            agg_map.name = new_feature_name
            current_new_features = [new_feature_name]

        new_feature_names.extend(current_new_features)

        # Map aggregates back to original fold dataframes using merge
        # Use left merge to keep all rows from the original fold dfs
        df_train_fold = pd.merge(df_train_fold, agg_map, on=group_col, how='left')
        df_valid_fold = pd.merge(df_valid_fold, agg_map, on=group_col, how='left')
        df_test_fold = pd.merge(df_test_fold, agg_map, on=group_col, how='left')


        # Fill potential NaNs in valid/test (if group value didn't exist in train fold)
        # Also fill NaNs in train_fold that might occur if a group was present but had all NaNs for 'col'
        for new_feature in current_new_features:
             # Determine fill value based on the *training fold's* distribution of the original column
             # This is more robust than using the aggregate map's mean/median which might be NaN
             fill_value = df_train_fold[col].median() if ('median' in new_feature or 'std' in new_feature) else df_train_fold[col].mean()
             if pd.isna(fill_value): # Ultimate fallback if even the train column is all NaN
                 fill_value = 0
             df_train_fold[new_feature] = df_train_fold[new_feature].fillna(fill_value)
             df_valid_fold[new_feature] = df_valid_fold[new_feature].fillna(fill_value)
             df_test_fold[new_feature] = df_test_fold[new_feature].fillna(fill_value)


    # Restore original index order
    df_train_fold = df_train_fold.set_index(train_fold_index).sort_index()
    df_valid_fold = df_valid_fold.set_index(valid_fold_index).sort_index()
    df_test_fold = df_test_fold.set_index(test_fold_index).sort_index()

    # Drop duplicates that might arise from merges if index wasn't unique (shouldn't happen with KFold index)
    # df_train_fold = df_train_fold[~df_train_fold.index.duplicated(keep='first')]
    # df_valid_fold = df_valid_fold[~df_valid_fold.index.duplicated(keep='first')]
    # df_test_fold = df_test_fold[~df_test_fold.index.duplicated(keep='first')]


    return df_train_fold, df_valid_fold, df_test_fold, new_feature_names


# --- Optuna Objective Function ---
def objective(trial, X_train_orig, y_train_orig, X_test_orig):
    # Define hyperparameters to tune
    # Expanded ranges slightly for potentially better models
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 800, 8000, step=200),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256), # Wider range
        'max_depth': trial.suggest_int('max_depth', 5, 25), # Wider range
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0), # Wider range
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0), # Wider range
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 20.0, log=True), # Wider range
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 20.0, log=True), # Wider range
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200), # Wider range
        'seed': RANDOM_SEED,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
    }
    if USE_GPU:
         params['device'] = 'gpu'
         params['gpu_platform_id'] = 0 # Often 0
         params['gpu_device_id'] = 0 # Often 0

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_mae_list = []
    oof_preds = np.zeros(X_train_orig.shape[0])
    all_fold_features = set(X_train_orig.columns) # Keep track of all features generated

    # --- Define columns for aggregation within the objective function ---
    # Ensure these columns exist in the preprocessed data X_train_orig
    base_grouping_cols = ['DayOfWeek', 'TimeOfDay', 'Has_Guest', 'Sentiment_Score', 'Is_Weekend']
    base_numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                           'Guest_Popularity_percentage', 'Number_of_Ads', 'Ads_per_Minute',
                           'Podcast_Name_Len', 'Episode_Title_Len',
                           'Podcast_Name_Freq', 'Genre_Freq'] # Added Freq features

    # Filter to existing columns
    grouping_cols_for_agg = [col for col in base_grouping_cols if col in X_train_orig.columns]
    numerical_cols_for_agg = [col for col in base_numerical_cols if col in X_train_orig.columns]

    current_features = X_train_orig.columns.tolist() # Base features

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_orig, y_train_orig)):
        X_train_fold, X_valid_fold = X_train_orig.iloc[train_idx].copy(), X_train_orig.iloc[valid_idx].copy()
        y_train_fold, y_valid_fold = y_train_orig.iloc[train_idx], y_train_orig.iloc[valid_idx]
        X_test_fold_iter = X_test_orig.copy() # Use a fresh copy for this fold's aggregation mapping

        # --- Generate Aggregate Features WITHIN the fold ---
        fold_new_agg_features = []
        for group_col in grouping_cols_for_agg:
             if group_col in X_train_fold.columns:
                 # Filter numerical columns to avoid aggregating a column by itself or constant columns
                 agg_cols_for_group = [c for c in numerical_cols_for_agg if c != group_col and X_train_fold[c].nunique() > 1]
                 if not agg_cols_for_group:
                     continue # Skip if no valid columns to aggregate for this group

                 X_train_fold, X_valid_fold, X_test_fold_iter, new_agg_features = create_aggregate_features(
                     X_train_fold, X_valid_fold, X_test_fold_iter,
                     group_col=group_col,
                     agg_cols=agg_cols_for_group
                 )
                 fold_new_agg_features.extend(new_agg_features)
             else:
                 print(f"Optuna Warning: Grouping column '{group_col}' not found in fold {fold}. Skipping.")


        # Combine original features with newly generated ones for this fold
        # Use set operations to avoid duplicates
        all_fold_features.update(fold_new_agg_features)
        final_fold_cols = list(all_fold_features) # Use the cumulative set from all folds up to this point

        # Reindex to ensure consistent columns across folds, fill missing with 0 (or median/mean if preferred)
        X_train_fold = X_train_fold.reindex(columns=final_fold_cols, fill_value=0)
        X_valid_fold = X_valid_fold.reindex(columns=final_fold_cols, fill_value=0)

        # --- Train Model ---
        model = lgb.LGBMRegressor(**params)
        # Reduce patience during Optuna trials for speed
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_valid_fold, y_valid_fold)],
                  eval_metric='mae',
                  callbacks=callbacks)

        # --- Predict and Evaluate ---
        fold_preds = model.predict(X_valid_fold)
        fold_preds = np.maximum(fold_preds, 0) # Ensure non-negative predictions
        oof_preds[valid_idx] = fold_preds
        fold_mae = mean_absolute_error(y_valid_fold, fold_preds)
        oof_mae_list.append(fold_mae)

        # Pruning for Optuna (optional but recommended)
        trial.report(fold_mae, fold)
        if trial.should_prune():
             raise optuna.exceptions.TrialPruned()

        del X_train_fold, X_valid_fold, y_train_fold, y_valid_fold, X_test_fold_iter; gc.collect()


    mean_oof_mae = np.mean(oof_mae_list)
    print(f"Trial {trial.number} finished. Mean OOF MAE: {mean_oof_mae:.5f}")

    # Handle cases where Optuna might prune a trial early or MAE is NaN
    if np.isnan(mean_oof_mae):
        return float('inf') # Return a large value if MAE calculation failed

    return mean_oof_mae # Optuna minimizes this value

# --- Run Optuna Study ---
if USE_OPTUNA:
    print("\n--- Starting Hyperparameter Optimization with Optuna ---")
    # Consider increasing n_jobs if you have multiple cores and memory allows
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    # Pass copies of the original processed data to avoid modification issues during parallel trials (if n_jobs > 1)
    study.optimize(lambda trial: objective(trial, X.copy(), y.copy(), X_test.copy()),
                   n_trials=N_TRIALS_OPTUNA,
                   # n_jobs=2 # Uncomment to use multiple cores if safe and memory allows
                  )
    best_params = study.best_params
    print("\nBest hyperparameters found by Optuna:")
    print(best_params)
    print(f"Best MAE achieved during tuning: {study.best_value:.5f}")
else:
    print("\n--- Skipping Optuna - Using Default Parameters ---")
    # Define some reasonable default parameters if not tuning
    best_params = {
        'objective': 'mae', 'metric': 'mae', 'n_estimators': 4000, # Increased default
        'learning_rate': 0.01, 'num_leaves': 100, 'max_depth': 15,
        'feature_fraction': 0.65, 'bagging_fraction': 0.65, 'bagging_freq': 4,
        'lambda_l1': 0.5, 'lambda_l2': 0.5, 'min_child_samples': 25,
        'seed': RANDOM_SEED, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
    }
    if USE_GPU:
         best_params['device'] = 'gpu'
         best_params['gpu_platform_id'] = 0
         best_params['gpu_device_id'] = 0
    print("Using defaults:", best_params)


# --- Final Model Training with Best Parameters ---
print("\n--- Training Final Model with Best Parameters using K-Fold CV ---")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
oof_final_preds = np.zeros(X.shape[0])
test_final_preds = np.zeros(X_test.shape[0])
feature_importance_df = pd.DataFrame()
final_fold_maes = []
all_training_features = set(X.columns) # Initialize with base features

# --- Define Aggregation Columns Again (consistent with objective function) ---
base_grouping_cols = ['DayOfWeek', 'TimeOfDay', 'Has_Guest', 'Sentiment_Score', 'Is_Weekend']
base_numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                       'Guest_Popularity_percentage', 'Number_of_Ads', 'Ads_per_Minute',
                       'Podcast_Name_Len', 'Episode_Title_Len',
                       'Podcast_Name_Freq', 'Genre_Freq'] # Added Freq features

# Filter to existing columns
grouping_cols_for_agg = [col for col in base_grouping_cols if col in X.columns]
numerical_cols_for_agg = [col for col in base_numerical_cols if col in X.columns]

current_features = X.columns.tolist() # Base features before aggregation


for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_valid_fold = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    X_test_fold = X_test.copy() # Use a copy for each fold's prediction generation

    # --- Generate Aggregate Features WITHIN the fold ---
    fold_new_agg_features = []
    for group_col in grouping_cols_for_agg:
        if group_col in X_train_fold.columns: # Check existence
             # Filter numerical columns to avoid aggregating a column by itself or constant columns
             agg_cols_for_group = [c for c in numerical_cols_for_agg if c != group_col and X_train_fold[c].nunique() > 1]
             if not agg_cols_for_group:
                 print(f"  Skipping aggregation for group '{group_col}' in fold {fold+1} - no valid columns to aggregate.")
                 continue

             print(f"  Generating aggregates for group: {group_col} using cols: {agg_cols_for_group}")
             X_train_fold, X_valid_fold, X_test_fold, new_agg_features = create_aggregate_features(
                 X_train_fold, X_valid_fold, X_test_fold,
                 group_col=group_col,
                 agg_cols=agg_cols_for_group
             )
             fold_new_agg_features.extend(new_agg_features)
             all_training_features.update(new_agg_features) # Add to global set of all features encountered
        else:
             print(f"Warning: Grouping column '{group_col}' not found in final fold {fold+1}. Skipping.")

    # Final feature list for this fold (cumulative set of all features seen so far)
    final_fold_cols = list(all_training_features)

    # Align columns for train, valid, test for this specific fold using the master list
    # Fill missing values (e.g., aggregate feature not generated in *this* fold but in others) with 0
    X_train_fold = X_train_fold.reindex(columns=final_fold_cols, fill_value=0)
    X_valid_fold = X_valid_fold.reindex(columns=final_fold_cols, fill_value=0)
    X_test_fold = X_test_fold.reindex(columns=final_fold_cols, fill_value=0)

    print(f"Shape after aggregation: Train={X_train_fold.shape}, Valid={X_valid_fold.shape}, Test={X_test_fold.shape}")
    print(f"Number of features for fold {fold+1}: {len(final_fold_cols)}")

    # --- Train Model ---
    # Ensure n_estimators is in best_params (Optuna might not always include it if default is used)
    if 'n_estimators' not in best_params:
         best_params['n_estimators'] = 4000 # Set a default if missing

    model = lgb.LGBMRegressor(**best_params)
    # Longer patience for final model, maybe log evaluation results
    callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)] # Increase patience
                # lgb.log_evaluation(period=200)] # Uncomment to see training progress less frequently

    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_valid_fold, y_valid_fold)],
              eval_metric='mae',
              callbacks=callbacks)

    # --- Predict ---
    valid_preds = model.predict(X_valid_fold)
    valid_preds = np.maximum(0, valid_preds) # Clip predictions at 0
    oof_final_preds[valid_idx] = valid_preds

    test_fold_preds = model.predict(X_test_fold)
    test_fold_preds = np.maximum(0, test_fold_preds) # Clip predictions at 0
    test_final_preds += test_fold_preds / N_SPLITS

    # --- Feature Importance ---
    fold_importance = pd.DataFrame({'feature': X_train_fold.columns,
                                    'importance': model.feature_importances_,
                                    'fold': fold + 1})
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

    fold_mae = mean_absolute_error(y_valid_fold, valid_preds)
    final_fold_maes.append(fold_mae)
    print(f"Fold {fold+1} OOF MAE: {fold_mae:.5f}")
    print(f"Fold {fold+1} Best Iteration: {model.best_iteration_}")

    del X_train_fold, X_valid_fold, y_train_fold, y_valid_fold, X_test_fold, model; gc.collect()


# --- Evaluate Overall OOF ---
final_oof_mae = mean_absolute_error(y, oof_final_preds)
print(f"\n--- Overall OOF MAE: {final_oof_mae:.5f} ---")
print(f"--- Std Dev OOF MAE across folds: {np.std(final_fold_maes):.5f} ---")


# --- Display Feature Importances ---
if not feature_importance_df.empty:
    # Aggregate importances using the global set of features encountered
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    mean_importance = mean_importance[mean_importance > 0] # Show only features with > 0 importance

    print(f"\n--- Top {min(50, len(mean_importance))} Feature Importances (Mean over Folds, >0 Importance) ---")
    print(mean_importance.head(50))

    plt.figure(figsize=(12, max(10, int(len(mean_importance.head(50)) * 0.3)))) # Adjust height dynamically
    sns.barplot(x=mean_importance.head(50).values, y=mean_importance.head(50).index)
    plt.title(f'Top {min(50, len(mean_importance))} Feature Importances (Mean over Folds)')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png') # Save the plot
    # plt.show() # Comment out plt.show() for Kaggle submission typically
    print("Feature importance plot saved as feature_importance.png")
else:
    print("\nFeature importance calculation skipped (possibly no successful folds).")


# --- Create Submission File ---
print("\n--- Creating Submission File ---")
submission_df = pd.DataFrame({'id': test_ids, TARGET: test_final_preds})

# --- Final Clipping of Test Predictions ---
# Merge the processed test lengths back to the submission df
submission_df = pd.merge(submission_df, test_lengths_df, on='id', how='left')

# Clip predictions: >= 0 and <= Episode_Length_minutes
# Handle potential NaNs in length if KNN failed or length was missing initially
submission_df[TARGET] = np.maximum(0, submission_df[TARGET])
# If 'Episode_Length_minutes' is NaN or Inf (placeholder), keep the prediction; otherwise, clip.
submission_df[TARGET] = np.minimum(submission_df[TARGET],
                                   submission_df['Episode_Length_minutes'].fillna(submission_df[TARGET]))

# Ensure final target column doesn't contain NaN/Inf from the length column merge/clip process
submission_df[TARGET] = submission_df[TARGET].replace([np.inf, -np.inf], np.nan)
# If any NaNs remain after clipping (highly unlikely), fill with OOF mean or simple median?
if submission_df[TARGET].isnull().any():
    oof_mean = np.mean(oof_final_preds) if len(oof_final_preds) > 0 else y.median() # Fallback
    print(f"Warning: Found NaNs in final predictions after clipping. Filling with OOF mean ({oof_mean:.2f}).")
    submission_df[TARGET] = submission_df[TARGET].fillna(oof_mean)
    # Re-apply floor clipping
    submission_df[TARGET] = np.maximum(0, submission_df[TARGET])


# Drop the temporary length column
submission_df = submission_df[['id', TARGET]]

# Final check on submission shape
print(f"Final submission shape: {submission_df.shape}")
expected_rows = sample_submission.shape[0]
if submission_df.shape[0] != expected_rows:
     print(f"ERROR: Submission has {submission_df.shape[0]} rows, but expected {expected_rows} rows!")
else:
     print(f"Submission row count ({submission_df.shape[0]}) matches sample submission.")

# Save submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
print(submission_df.head())

print("\nScript finished.")

