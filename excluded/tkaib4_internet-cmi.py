# defining necessary imports
import numpy as np
import pandas as pd
import os
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer # Added QuantileTransformer
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import random
import mord
from sklearn.impute import KNNImputer


# ignoring warnings
warnings.filterwarnings('ignore')


# defining constants
SEED = 3
n_splits = 10
optimize_params = True # change depending on if you want params changed or not
n_optuna_trials = 25 # n_trials for optuna
base_thresholds = [30, 50, 80] # Initial guess for thresholds


# defining paths
TABULAR_TRAIN_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
TABULAR_TEST_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'
ACTIGRAPHY_TRAIN_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet'
ACTIGRAPHY_TEST_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet'
SUBMISSION_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv'
OUTPUT_PATH = '/kaggle/working/'


train_full = pd.read_csv(TABULAR_TRAIN_PATH)
test = pd.read_csv(TABULAR_TEST_PATH)


def calculate_weights(series):
    # Create bins for the target variable and assign weights based on frequency
    bins = pd.cut(series, bins=10, labels=False)
    weights = bins.value_counts().reset_index()
    weights.columns = ['target_bins', 'count']
    weights['count'] = 1 / weights['count']
    weight_map = weights.set_index('target_bins')['count'].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()


def round_with_thresholds(raw_preds, thresholds):
    return np.where(raw_preds < thresholds[0], int(0),
                    np.where(raw_preds < thresholds[1], int(1),
                             np.where(raw_preds < thresholds[2], int(2), int(3))))


def perform_pca(train, test, n_components=None, random_state=33):
    """
    Performs PCA on train, describes the Explained Variance Ratio and transforms train and test

    Returns: transformed train, transformed test, pca object

    """
    pca = PCA(n_components=n_components, random_state=random_state)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio of the components:\n {explained_variance_ratio}")
    print(f"Cumulative explained variance: {np.sum(explained_variance_ratio)}") # Added cumulative variance

    train_pca_df = pd.DataFrame(train_pca, columns=[f'PC_{i+1}' for i in range(train_pca.shape[1])], index=train.index) # Preserve index
    test_pca_df = pd.DataFrame(test_pca, columns=[f'PC_{i+1}' for i in range(test_pca.shape[1])], index=test.index) # Preserve index

    return train_pca_df, test_pca_df, pca


def time_features(df):
    """Function extracting Features from ActiGraph data of an individual"""
    # Convert time_of_day to hours
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    # Basic features
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum(),
    ]

    # Define conditions for night, day, and no mask (full data)
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    no_mask = np.ones(len(df), dtype=bool)

    # List of columns of interest and masks
    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [no_mask, night, day]

    # Helper function for feature extraction
    def extract_stats(data):
        return [
            data.mean(),
            data.std(),
            data.max(),
            data.min(),
            data.diff().mean(),
            data.diff().std()
        ]

    # Iterate over keys and masks to generate the statistics
    for key in keys:
        for mask in masks:
            filtered_data = df.loc[mask, key]
            features.extend(extract_stats(filtered_data))

    return features


def process_file(filename, dirname):
    # Process file and extract time features
    try:
        df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
        df.drop('step', axis=1, inplace=True)
        return time_features(df), filename.split('=')[1]
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Return placeholder data or handle error appropriately
        return [0.0] * 74, filename.split('=')[1] # Adjust size based on expected features


# def load_time_series(dirname) -> pd.DataFrame:
#     # Load time series from directory in parallel
#     ids = os.listdir(dirname)

#     with ThreadPoolExecutor() as executor:
#         results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids), desc="Processing Time Series"))

#     stats_list, indexes = zip(*results)

#     # Check consistency of feature lengths
#     feature_length = len(stats_list[0])
#     if not all(len(s) == feature_length for s in stats_list):
#         print("Warning: Inconsistent feature lengths detected in time series processing.")
#         # Add logic here to handle inconsistent lengths if necessary, e.g., padding or error reporting

#     df = pd.DataFrame(stats_list, columns=[f"stat_{i}" for i in range(feature_length)])
#     df['id'] = indexes

#     return df


def load_time_series(dirname) -> pd.DataFrame:
    # Only process subdirectories
    ids = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(lambda fname: process_file(fname, dirname), ids),
            total=len(ids),
            desc="Processing Time Series"
        ))

    stats_list, indexes = zip(*results)

    feature_length = len(stats_list[0])
    if not all(len(s) == feature_length for s in stats_list):
        print("Warning: Inconsistent feature lengths detected.")

    df = pd.DataFrame(stats_list, columns=[f"stat_{i}" for i in range(feature_length)])
    df['id'] = indexes

    return df



train_ts = load_time_series(ACTIGRAPHY_TRAIN_PATH)
test_ts = load_time_series(ACTIGRAPHY_TEST_PATH)


df_train_ts = train_ts.drop('id', axis=1).set_index(train_ts['id']) # Set ID as index
df_test_ts = test_ts.drop('id', axis=1).set_index(test_ts['id'])


# scaling and imputation prior to PCA
# using a quantile transformer and KNN Imputer
qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
knn_imputer_ts = KNNImputer(n_neighbors=5) # Impute based on neighbors

df_train_ts_qt = qt.fit_transform(df_train_ts)
df_test_ts_qt = qt.transform(df_test_ts)

df_train_ts_imputed = knn_imputer_ts.fit_transform(df_train_ts_qt)
df_test_ts_imputed = knn_imputer_ts.transform(df_test_ts_qt)

df_train_ts = pd.DataFrame(df_train_ts_imputed, columns=df_train_ts.columns, index=df_train_ts.index)
df_test_ts = pd.DataFrame(df_test_ts_imputed, columns=df_test_ts.columns, index=df_test_ts.index)


print(f"TS Train shape after imputation: {df_train_ts.shape}")


# perform PCA
print("Performing PCA on time series features...")
df_train_pca, df_test_pca, pca = perform_pca(df_train_ts, df_test_ts, n_components=20, random_state=SEED) # Increased components slightly


# Merge PCA features back
train_full = pd.merge(train_full, df_train_pca, how="left", left_on='id', right_index=True)
test = pd.merge(test, df_test_pca, how="left", left_on='id', right_index=True)
print(f"Train shape after merging PCA: {train_full.shape}")


# feature cleaning
def clean_features(df):
    # Remove highly implausible values

    # Clip Grip
    df[['FGC-FGC_GSND', 'FGC-FGC_GSD']] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].clip(lower=9, upper=60)
    # Remove implausible body-fat
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] < 5, np.nan, df["BIA-BIA_Fat"])
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] > 60, np.nan, df["BIA-BIA_Fat"])
    # Basal Metabolic Rate
    df["BIA-BIA_BMR"] = np.where(df["BIA-BIA_BMR"] > 4000, np.nan, df["BIA-BIA_BMR"])
    # Daily Energy Expenditure
    df["BIA-BIA_DEE"] = np.where(df["BIA-BIA_DEE"] > 8000, np.nan, df["BIA-BIA_DEE"])
    # Bone Mineral Content
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] <= 0, np.nan, df["BIA-BIA_BMC"])
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] > 10, np.nan, df["BIA-BIA_BMC"])
    # Fat Free Mass Index - Corrected column name assuming it's BIA-BIA_FFM
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] <= 0, np.nan, df["BIA-BIA_FFM"])
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] > 300, np.nan, df["BIA-BIA_FFM"])
    # Fat Mass Index
    df["BIA-BIA_FMI"] = np.where(df["BIA-BIA_FMI"] < 0, np.nan, df["BIA-BIA_FMI"])
    # Extra Cellular Water
    df["BIA-BIA_ECW"] = np.where(df["BIA-BIA_ECW"] > 100, np.nan, df["BIA-BIA_ECW"])
    # Intra Cellular Water - commented out in original
    # df["BIA-BIA_ICW"] = np.where(df["BIA-BIA_ICW"] > 100, np.nan, df["BIA-BIA_ICW"])
    # Lean Dry Mass
    df["BIA-BIA_LDM"] = np.where(df["BIA-BIA_LDM"] > 100, np.nan, df["BIA-BIA_LDM"])
    # Lean Soft Tissue
    df["BIA-BIA_LST"] = np.where(df["BIA-BIA_LST"] > 300, np.nan, df["BIA-BIA_LST"])
    # Skeletal Muscle Mass
    df["BIA-BIA_SMM"] = np.where(df["BIA-BIA_SMM"] > 300, np.nan, df["BIA-BIA_SMM"])
    # Total Body Water
    df["BIA-BIA_TBW"] = np.where(df["BIA-BIA_TBW"] > 300, np.nan, df["BIA-BIA_TBW"])

    return df


train_full = clean_features(train_full)
test = clean_features(test)


# feature engineering
def feature_engineering(df):
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1, errors='ignore') # Use errors='ignore'

    # From here on own features
    def assign_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan # Return NaN if age is outside defined ranges

    # Age groups
    df["group"] = df['Basic_Demos-Age'].apply(assign_group)

    # BMI
    BMI_map = {0: 16.3,1: 15.9,2: 16.1,3: 16.8,4: 17.3,5: 19.2,6: 20.2,7: 22.3, 8: 23.6}
    # Handle potential NaN in group map result
    df['BMI_mean'] = df[['Physical-BMI', 'BIA-BIA_BMI']].mean(axis=1)
    df['group_BMI_mean'] = df["group"].map(BMI_map)
    df['BMI_mean_norm'] = df['BMI_mean'] / df['group_BMI_mean']
    df.drop(['BMI_mean', 'group_BMI_mean'], axis=1, inplace=True)


    # FGC zone aggregate
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
             'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
             'FGC-FGC_TL_Zone']

    # Ensure zones exist before calculating stats
    existing_zones = [z for z in zones if z in df.columns]
    if existing_zones:
        df['FGC_Zones_mean'] = df[existing_zones].mean(axis=1)
        df['FGC_Zones_min'] = df[existing_zones].min(axis=1)
        df['FGC_Zones_max'] = df[existing_zones].max(axis=1)

    # Grip
    GSD_max_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2, 5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4}
    GSD_min_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4, 5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1}

    df['GS_max_val'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].max(axis=1)
    df['GS_min_val'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].min(axis=1)
    df['group_GSD_max'] = df["group"].map(GSD_max_map)
    df['group_GSD_min'] = df["group"].map(GSD_min_map)

    df['GS_max'] = df['GS_max_val'] / df['group_GSD_max']
    df['GS_min'] = df['GS_min_val'] / df['group_GSD_min']
    df.drop(['GS_max_val', 'GS_min_val', 'group_GSD_max', 'group_GSD_min'], axis=1, inplace=True)


    # Curl-ups, push-ups, trunk-lifts... normalized based on age-group
    cu_map = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 14.0, 6: 20.0, 7: 20.0, 8: 20.0}
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {0: 8.0, 1: 8.0, 2: 8.0, 3: 9.0, 4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}

    df["CU_norm"] = df['FGC-FGC_CU'] / df['group'].map(cu_map)
    df["PU_norm"] = df['FGC-FGC_PU'] / df['group'].map(pu_map)
    df["TL_norm"] = df['FGC-FGC_TL'] / df['group'].map(tl_map)

    # Reach
    df["SR_min"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].min(axis=1)
    df["SR_max"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].max(axis=1)

    # BIA Features
    # Energy Expenditure
    bmr_map = {0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0, 4: 1283.0, 5: 1255.0, 6: 1481.0, 7: 1519.0, 8: 1650.0}
    dee_map = {0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0, 4: 2132.0, 5: 2121.0, 6: 2528.0, 7: 2566.0, 8: 2793.0}
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map)
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"] # Consider potential division by zero or NaNs

    # FMM
    ffm_map = {0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0, 4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0}
    # Handle potential NaN division
    df['group_ffm_map'] = df["group"].map(ffm_map)
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df['group_ffm_map']
    df.drop(['group_ffm_map'], axis=1, inplace=True)

    # ECW ICW
    # Handle potential division by zero or NaN
    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"].replace(0, np.nan) # Replace 0 with NaN before division

    # Drop original/intermediate features
    drop_feats = ['FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                  'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
                  'Physical-BMI', 'BIA-BIA_BMI', 'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
                 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_Frame_num', "BIA-BIA_FFM", "BIA-BIA_ICW", "BIA-BIA_ECW", # Added ICW, ECW
                 'group' # Drop the group feature itself after use
                 ]
    # Drop only columns that exist in the dataframe
    existing_drop_feats = [feat for feat in drop_feats if feat in df.columns]
    df = df.drop(existing_drop_feats, axis=1)
    return df


train_full = feature_engineering(train_full)
test = feature_engineering(test)


# Binning (OPTIONAL, test both with and without it to see which performs better)
# def bin_data(train, test, columns, n_bins=10):
#     # Combine train and test for consistent bin edges
#     combined = pd.concat([train, test], axis=0)

#     bin_edges = {}
#     for col in columns:
#         # Compute quantile bin edges
#         edges = pd.qcut(combined[col], n_bins, retbins=True, labels=range(n_bins), duplicates="drop")[1]
#         bin_edges[col] = edges

#     # Apply the same bin edges to both train and test
#     for col, edges in bin_edges.items():
#         train[col] = pd.cut(
#             train[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
#         ).astype(float)
#         test[col] = pd.cut(
#             test[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
#         ).astype(float)

#     return train, test

# columns_to_bin = [
#     "PAQ_A-PAQ_A_Total", "BMR_norm", "DEE_norm", "GS_min", "GS_max", "BIA-BIA_FFMI",
#     "BIA-BIA_BMC", "Physical-HeartRate", "BIA-BIA_ICW", "Fitness_Endurance-Time_Sec",
#     "BIA-BIA_LDM", "BIA-BIA_SMM", "BIA-BIA_TBW", "DEE_BMR", "ICW_ECW"
# ]
# # Bin specified columns in train and test
# train, test = bin_data(train, test, columns_to_bin, n_bins=10)


# define features
pciat_cols = [col for col in train_full.columns if 'PCIAT-' in col and col != 'PCIAT-Season'] # PCIAT score is needed later
y_model_col = "PCIAT-PCIAT_Total" # Intermediate score target
y_comp_col = "sii" # Final competition target


# Features available in both train and test after processing
base_features = [f for f in test.columns if f not in ['id']]
# Ensure all base_features are also in train
features = [f for f in base_features if f in train_full.columns]
# Make sure no target/ID columns slipped through
features = [f for f in features if f not in [y_comp_col, y_model_col]]


print(f"Number of features: {len(features)}")
print(f"Missing features in test compared to train_full (should be only targets/PCIAT): {set(train_full.columns) - set(test.columns) - set(['id'])}")


# separate labelled and unlabelled data
train_labeled = train_full[train_full[y_comp_col].notna()].copy()
train_unlabeled = train_full[train_full[y_comp_col].isna()].copy()
print(f"Labeled samples: {len(train_labeled)}, Unlabeled samples: {len(train_unlabeled)}")


# impute using KNN Imputer (instead of Impute_With_Model)
imputer = KNNImputer(n_neighbors=7) # Adjust n_neighbors as needed


#fit on labelled data only
imputer.fit(train_labeled[features])


train_labeled[features] = imputer.transform(train_labeled[features])
test[features] = imputer.transform(test[features])


lgb_params = {
    'objective': 'poisson', # Good for scores
    'metric': 'rmse', # Monitor RMSE during potential early stopping
    'n_estimators': 500, # Increase estimators, use early stopping
    'learning_rate': 0.03,
    'feature_fraction': 0.8, # Equivalent to colsample_bytree
    'bagging_fraction': 0.8, # Equivalent to subsample
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31, # Default is 31, adjust based on max_depth
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED,
    'boosting_type': 'gbdt',
    'max_depth': 5, # Limit depth
    'min_child_samples': 20, # Equivalent to min_data_in_leaf
}


# impute data for pseudo-labeling
if not train_unlabeled.empty:
     # Check if unlabeled data has missing values in features
    if train_unlabeled[features].isnull().sum().sum() > 0:
        train_unlabeled[features] = imputer.transform(train_unlabeled[features])
    else:
        print("No NaNs to impute in unlabeled feature set.")


# psuedo labelling
if not train_unlabeled.empty:
    print("Performing pseudo-labeling...")
    # 1. Train an initial model on labeled data (e.g., LGBM)
    # Using pre-defined params for simplicity, tune if needed
    pseudo_label_model = LGBMRegressor(**lgb_params, random_state=SEED+1, verbosity=-1) # Use different seed

    # Optional: Use PCIAT score as target if it's cleaner
    if y_model_col in train_labeled.columns and not train_labeled[y_model_col].isnull().all():
         print(f"Using {y_model_col} for initial pseudo-label model training.")
         weights_pseudo = calculate_weights(train_labeled[y_model_col]) # Reuse weight function
         pseudo_label_model.fit(train_labeled[features], train_labeled[y_model_col], sample_weight=weights_pseudo)
         pseudo_scores = pseudo_label_model.predict(train_unlabeled[features])
         # Use consistent thresholds (e.g., average from previous CV or base)
         # Find average thresholds from a preliminary CV run if possible, otherwise use base
         # avg_thresholds_pseudo = np.mean(np.array(preliminary_thresholds), axis=0)
         avg_thresholds_pseudo = base_thresholds # Fallback
         pseudo_labels = round_with_thresholds(pseudo_scores, avg_thresholds_pseudo)
    else:
        # Fallback: Train directly on 'sii' if PCIAT score isn't reliable/available
        # This might be less accurate as input 'sii' is ordinal
         print(f"Warning: Falling back to training pseudo-label model directly on {y_comp_col}.")
         pseudo_label_model.fit(train_labeled[features], train_labeled[y_comp_col])
         pseudo_labels = pseudo_label_model.predict(train_unlabeled[features]).round().astype(int).clip(0, 3) # Predict sii directly

    # 2. Add pseudo-labels to unlabeled data
    train_unlabeled[y_comp_col] = pseudo_labels
    # Optional: Add PCIAT score prediction if that was used
    if y_model_col in train_labeled.columns and not train_labeled[y_model_col].isnull().all():
        train_unlabeled[y_model_col] = pseudo_scores # Add the predicted score too

    # 3. Combine labeled and pseudo-labeled data
    train_combined = pd.concat([train_labeled, train_unlabeled], ignore_index=True)
    print(f"Combined training data size: {len(train_combined)}")

    # Optional: Assign lower weight to pseudo-labeled samples during final training
    sample_indices = np.arange(len(train_combined))
    pseudo_labeled_indices = sample_indices[len(train_labeled):]
    # Create a weight array (e.g., 1 for original, 0.5 for pseudo)
    # sample_weight_pseudo = np.ones(len(train_combined))
    # sample_weight_pseudo[pseudo_labeled_indices] = 0.5 # Adjust weight factor

else:
    print("No unlabeled data found, skipping pseudo-labeling.")
    train_combined = train_labeled.copy()
    # sample_weight_pseudo = np.ones(len(train_combined)) # All weights are 1

# Use train_combined for subsequent training
train = train_combined # Rename for consistency with the rest of the script


# threshold optimization
def round_with_thresholds(raw_preds, thresholds):
    # Ensure thresholds are sorted
    thresholds = np.sort(thresholds)
    return np.where(raw_preds < thresholds[0], int(0),
                    np.where(raw_preds < thresholds[1], int(1),
                             np.where(raw_preds < thresholds[2], int(2), int(3))))


def optimize_thresholds(y_true, raw_preds, start_vals=[0.5, 1.5, 2.5]):
    # Ensure y_true and raw_preds have the same length
    if len(y_true) != len(raw_preds):
         raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != raw_preds ({len(raw_preds)})")

    # Check for NaNs
    if np.isnan(raw_preds).any():
        print("Warning: NaNs found in raw_preds during threshold optimization. Replacing with mean.")
        raw_preds = np.nan_to_num(raw_preds, nan=np.nanmean(raw_preds))

    def fun(thresholds, y_true, raw_preds):
        # Ensure thresholds are sorted within the function
        sorted_thresholds = np.sort(thresholds)
        rounded_preds = round_with_thresholds(raw_preds, sorted_thresholds)
        # Return negative kappa score for minimization
        return -cohen_kappa_score(y_true, rounded_preds, weights='quadratic')

    # Use bounds to ensure thresholds remain ordered and reasonable
    bnds = [(min(raw_preds)-1, max(raw_preds)+1) for _ in range(len(start_vals))]
    # Add constraints to keep thresholds ordered t0 < t1 < t2
    constraints = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 1e-6}, # t1 > t0
                   {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 1e-6}) # t2 > t1

    res = minimize(fun, x0=np.sort(start_vals), args=(y_true, raw_preds), method='SLSQP', bounds=bnds, constraints=constraints) # Changed method to SLSQP for bounds/constraints

    if not res.success:
         print(f"Threshold optimization failed: {res.message}. Returning start_vals.")
         return np.sort(start_vals) # Return sorted start values on failure

    return np.sort(res.x) # Return sorted thresholds


def calculate_weights(series):
    # Create bins for the target variable and assign weights based on frequency
    # Handle potential NaNs in the series
    if series.isnull().any():
        print("Warning: NaNs found in target series for weight calculation. Dropping NaNs.")
        series = series.dropna()
    if series.empty:
        print("Warning: Empty series provided for weight calculation. Returning uniform weights.")
        return pd.Series(1.0, index=series.index) # Or handle as error

    # Increase robustness for series with few unique values or skewed distributions
    try:
        # Use pd.qcut for potentially better binning with skewed data
        bins = pd.qcut(series, q=min(10, series.nunique()), labels=False, duplicates='drop')
    except ValueError:
        # Fallback to cut if qcut fails (e.g., too few unique values)
        bins = pd.cut(series, bins=min(10, series.nunique()), labels=False, include_lowest=True)

    # If all values fall into one bin, return uniform weights
    if bins.nunique() <= 1:
        print("Warning: Target series has low variance. Returning uniform weights.")
        return pd.Series(1.0, index=series.index)

    weights_df = bins.value_counts().reset_index()
    weights_df.columns = ['target_bins', 'count']

    # Prevent division by zero if a bin somehow has zero count (shouldn't happen with value_counts)
    weights_df['count'] = weights_df['count'].replace(0, 1)

    weights_df['weight'] = 1 / weights_df['count']
    weight_map = weights_df.set_index('target_bins')['weight'].to_dict()

    # Map weights back, handle potential missing bins if qcut/cut produced fewer than expected
    final_weights = bins.map(weight_map).fillna(1.0) # Fill potential NaNs with 1

    # Normalize weights
    mean_weight = final_weights.mean()
    if mean_weight == 0: # Prevent division by zero
        print("Warning: Mean weight is zero. Returning uniform weights.")
        return pd.Series(1.0, index=series.index)

    normalized_weights = final_weights / mean_weight
    return normalized_weights


# cross validation
def cross_validate_regressor(model_, data, features, score_col, index_col, cv, sample_weights_series=None, verbose=False):
    """
    Perform cross-validation with a regressor model.
    Predicts the score_col, optimizes thresholds, and calculates QWK on index_col.

    Returns:
    float: Mean Kappa score across all folds.
    array: Out-of-fold score predictions.
    array: Out-of-fold index predictions (after thresholding).
    list: List of optimized thresholds per fold.
    """
    kappa_scores = []
    oof_score_predictions = np.zeros(len(data))
    oof_index_predictions = np.zeros(len(data), dtype=int)
    fold_thresholds = []
    models = [] # Store models from each fold

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        y_train_score = data[score_col].iloc[train_idx]
        y_train_index = data[index_col].iloc[train_idx]
        # y_val_score = data[score_col].iloc[val_idx] # Not strictly needed for score calculation here
        y_val_index = data[index_col].iloc[val_idx]

        current_model = clone(model_) # Clone model for each fold

        # Handle sample weights
        fit_params = {}
        if sample_weights_series is not None:
            # Ensure weights are aligned with training indices
            weights_train = sample_weights_series.iloc[train_idx].values
            # Check for NaNs or zeros in weights
            if np.isnan(weights_train).any() or np.isinf(weights_train).any() or (weights_train <= 0).any():
                 print(f"Warning: Invalid weights found in fold {fold_idx}. Using uniform weights for this fold.")
            else:
                # CatBoost uses 'sample_weight', LGBM/XGB use 'sample_weight' in fit
                fit_params['sample_weight'] = weights_train


        # Train model
        try:
             # Special handling for CatBoost verbosity if needed
            if isinstance(current_model, CatBoostRegressor):
                 current_model.fit(X_train, y_train_score, eval_set=[(X_val, data[score_col].iloc[val_idx])], early_stopping_rounds=50, verbose=0, **fit_params)
            else:
                 current_model.fit(X_train, y_train_score, **fit_params)
        except Exception as e:
             print(f"Error fitting model in fold {fold_idx}: {e}")
             # Handle error, maybe skip fold or use default predictions?
             continue # Skip this fold on error

        # Predict scores on validation set
        y_pred_val_score = current_model.predict(X_val)
        oof_score_predictions[val_idx] = y_pred_val_score

        # Optimize thresholds using validation scores and true validation index
        # Use train predictions and train index for optimizing thresholds to avoid overfitting thresholds to val set?
        # This is debatable. Original code used train set predictions. Let's stick to that for consistency.
        y_pred_train_score = current_model.predict(X_train)
        # Ensure no NaNs in prediction used for thresholding
        y_pred_train_score_clean = np.nan_to_num(y_pred_train_score, nan=np.nanmean(y_pred_train_score))

        # Use a robust starting point for thresholds based on score distribution
        start_thresholds = np.percentile(y_train_score.dropna(), [25, 50, 75]) # Use percentiles as start

        # Check if start_thresholds are distinct
        if len(np.unique(start_thresholds)) < 3:
            start_thresholds = base_thresholds # Fallback to base if percentiles are degenerate


        # Optimize thresholds on TRAIN predictions vs TRAIN index
        try:
             t_opt = optimize_thresholds(y_train_index, y_pred_train_score_clean, start_vals=start_thresholds)
             fold_thresholds.append(t_opt)
        except ValueError as e:
             print(f"Error optimizing thresholds in fold {fold_idx}: {e}. Using base thresholds.")
             t_opt = base_thresholds # Use base thresholds if optimization fails
             fold_thresholds.append(t_opt)


        # Apply optimized thresholds to validation score predictions
        y_pred_val_index = round_with_thresholds(y_pred_val_score, t_opt)
        oof_index_predictions[val_idx] = y_pred_val_index

        # Calculate Kappa score for the fold
        kappa_score = cohen_kappa_score(y_val_index, y_pred_val_index, weights='quadratic')
        kappa_scores.append(kappa_score)
        models.append(current_model) # Store the trained model

        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa = {kappa_score:.4f}, Thresholds = {np.round(t_opt, 2)}")

    mean_kappa = np.mean(kappa_scores)
    std_kappa = np.std(kappa_scores)
    if verbose:
        print(f"\n## Mean CV Kappa Score: {mean_kappa:.4f} ##")
        print(f"## Std CV Kappa Score: {std_kappa:.4f} ##\n")

    # Calculate overall OOF score using the index predictions from each fold
    overall_oof_kappa = cohen_kappa_score(data[index_col], oof_index_predictions, weights='quadratic')
    if verbose:
        print(f"## Overall OOF Kappa Score: {overall_oof_kappa:.4f} ##\n")


    return mean_kappa, oof_score_predictions, oof_index_predictions, fold_thresholds, models, overall_oof_kappa


# cross val for ordinal
def cross_validate_ordinal(model_, data, features, index_col, cv, verbose=False):
    """
    Perform cross-validation with an ordinal classification model (from mord).
    Directly predicts the index_col.

    Returns:
    float: Mean Kappa score across all folds.
    array: Out-of-fold index predictions.
    list: List of trained models per fold.
    float: Overall OOF Kappa score.
    """
    kappa_scores = []
    oof_index_predictions = np.zeros(len(data), dtype=int)
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        y_train_index = data[index_col].iloc[train_idx]
        y_val_index = data[index_col].iloc[val_idx]

        current_model = clone(model_)
        current_model.fit(X_train, y_train_index)

        y_pred_val_index = current_model.predict(X_val)
        oof_index_predictions[val_idx] = y_pred_val_index

        kappa_score = cohen_kappa_score(y_val_index, y_pred_val_index, weights='quadratic')
        kappa_scores.append(kappa_score)
        models.append(current_model)

        if verbose:
            print(f"Fold {fold_idx}: Ordinal Kappa = {kappa_score:.4f}")

    mean_kappa = np.mean(kappa_scores)
    std_kappa = np.std(kappa_scores)

    if verbose:
        print(f"\n## Mean CV Kappa Score (Ordinal): {mean_kappa:.4f} ##")
        print(f"## Std CV Kappa Score (Ordinal): {std_kappa:.4f} ##\n")

    # Calculate overall OOF score
    overall_oof_kappa = cohen_kappa_score(data[index_col], oof_index_predictions, weights='quadratic')
    if verbose:
        print(f"## Overall OOF Kappa Score (Ordinal): {overall_oof_kappa:.4f} ##\n")


    return mean_kappa, oof_index_predictions, models, overall_oof_kappa


# --- MODIFIED Optuna Objective Function ---
def objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights_series=None):
    """
    Optuna objective function using the new cross-validation functions.
    Performs ONE cross-validation run per trial.
    """
    is_ordinal = False # Flag to check if it's an ordinal model trial

    # --- Parameter Space Definitions ---

    # LightGBM
    if model_type == 'lightgbm':
        params = {
            'objective': trial.suggest_categorical('objective', ['poisson', 'regression_l1', 'rmse']),
            'metric': 'rmse',
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'n_jobs': -1,
            'verbosity': -1,
        }
        model = LGBMRegressor(**params)
        is_ordinal = False

    # --- NEW: XGBoost ---
    elif model_type == 'xgboost':
        params = {
            'objective': trial.suggest_categorical('objective', ['reg:squarederror', 'reg:pseudohubererror']), # Could add reg:tweedie if desired
            'eval_metric': 'rmse',
            'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),  # learning_rate
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5), # min_split_loss
            'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True),  # L2 reg
            'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),  # L1 reg
            'seed': SEED,
            'n_jobs': -1,
            # Add 'tree_method': 'hist' if using GPU or want faster histogram method
        }
        # Optional: Tweedie specific parameter
        # if params['objective'] == 'reg:tweedie':
        #     params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.0, 2.0)

        model = XGBRegressor(**params)
        is_ordinal = False

    # --- NEW: CatBoost ---
    elif model_type == 'catboost':
        params = {
            'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Poisson']), # Could add Tweedie
            'iterations': trial.suggest_int('iterations', 100, 1000), # n_estimators
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True), # Lambda L2 reg
            'subsample': trial.suggest_float('subsample', 0.6, 0.9), # Only if bootstrapping_type is Bayesian/Bernoulli
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9), # Feature fraction
            'random_seed': SEED,
            'verbose': 0, # Keep quiet during optuna
            'early_stopping_rounds': 50, # Use early stopping during CV folds
            'border_count': trial.suggest_int('border_count', 32, 255), # Controls discretization
             #'boosting_type': 'Plain', # Could try 'Ordered' but often slower
             #'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']) # If using subsample
        }
        # Optional: Tweedie specific parameter
        # if 'Tweedie' in params['loss_function']:
        #     params['loss_function'] = f"Tweedie:variance_power={trial.suggest_float('tweedie_variance_power', 1.0, 2.0)}"
        # Optional: Subsample requires bootstrap_type
        # if params.get('bootstrap_type') in ['Bayesian', 'Bernoulli']:
        #     params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)


        model = CatBoostRegressor(**params)
        is_ordinal = False

    # Ordinal Ridge
    elif model_type == 'ordinal_ridge' and mord is not None:
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        model = mord.OrdinalRidge(**params)
        is_ordinal = True

    # Unsupported type
    else:
        if model_type == 'ordinal_ridge' and mord is None:
             print("Skipping ordinal_ridge trial as 'mord' library is not installed.")
             return -1.0
        print(f"Warning: model_type '{model_type}' not recognized in objective function or dependencies missing.")
        return -1.0 # Return poor score for unrecognized types


    # --- Perform Cross-Validation (Single Run) ---
    try:
        if is_ordinal:
            mean_fold_kappa, _, _, overall_oof_kappa = cross_validate_ordinal(
                model, X, features, index_col, cv, verbose=False
            )
        else:
            mean_fold_kappa, _, _, _, _, overall_oof_kappa = cross_validate_regressor(
                model, X, features, score_col, index_col, cv,
                sample_weights_series=sample_weights_series, verbose=False
            )
        # Return the score Optuna should optimize
        return overall_oof_kappa

    except Exception as e:
        print(f"Error during cross-validation for trial {trial.number} ({model_type}): {e}")
        # Important: Return a poor score if CV fails, so Optuna doesn't favor failing parameters
        return -1.0 # Or appropriate value indicating failure


#def run_optimization
def run_optimization(X, features, score_col, index_col, model_type, n_trials=30, cv=None, sample_weights_series=None): # Accepts Series or None
    """
    Runs Optuna optimization using the objective function.

    Args:
        X (pd.DataFrame): Training data.
        features (list): List of feature names.
        score_col (str): Name of the intermediate score column (for regressors).
        index_col (str): Name of the final target index column (sii).
        model_type (str): Type of model to optimize ('lightgbm', 'ordinal_ridge', etc.).
        n_trials (int): Number of Optuna trials.
        cv (cross-validation generator): The CV object (e.g., StratifiedKFold).
        sample_weights_series (pd.Series, optional): Series containing sample weights. Defaults to None.
    """
    study = optuna.create_study(direction="maximize")

    # Pass the actual sample_weights_series (or None) to the objective function
    study.optimize(lambda trial: objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights_series),
                   n_trials=n_trials)

    print(f"\nOptuna finished for {model_type}")
    print(f"Best params found: {study.best_params}")
    # study.best_value holds the best overall_oof_kappa found during optimization
    print(f"Best Overall OOF Kappa score achieved: {study.best_value:.4f}")
    return study.best_params


# feature subsets
# Replace if subsets for features have been selected

# List of manually selected features NOT to include
exclude = [
    "PC_9", "PC_12", "Fitness_Endurance-Max_Stage", "Basic_Demos-Sex", "BMI_mean_norm", "PC_11",
    "PC_8", "FGC_Zones_min", "Physical-Systolic_BP", "PC_4", "BIA-BIA_FMI", "BIA-BIA_LST", "Physical-Diastolic_BP",
    "BIA-BIA_ECW", "Fitness_Endurance-Time_Mins", "PAQ_C-PAQ_C_Total", "PC_10", "BIA-BIA_Fat", "FFM_norm", "PC_14", "PC_7"
]

reduced_features = [f for f in features if f not in exclude]

lgb_features = reduced_features
xgb_features = reduced_features
cat_features = reduced_features
print(len(reduced_features))


selected_features = features # Start with all features after cleaning/eng
print(f"Using {len(selected_features)} features for modeling.")


# Model Parameters

# lgb_params = {
#     'objective': 'poisson', # Good for scores
#     'metric': 'rmse', # Monitor RMSE during potential early stopping
#     'n_estimators': 500, # Increase estimators, use early stopping
#     'learning_rate': 0.03,
#     'feature_fraction': 0.8, # Equivalent to colsample_bytree
#     'bagging_fraction': 0.8, # Equivalent to subsample
#     'bagging_freq': 1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'num_leaves': 31, # Default is 31, adjust based on max_depth
#     'verbose': -1,
#     'n_jobs': -1,
#     'seed': SEED,
#     'boosting_type': 'gbdt',
#     'max_depth': 5, # Limit depth
#     'min_child_samples': 20, # Equivalent to min_data_in_leaf
# }

xgb_params = {
    'objective': 'reg:squarederror', # Simpler objective, rely on thresholding
    'eval_metric': 'rmse',
    'eta': 0.03, # learning_rate
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'gamma': 0.0,
    'lambda': 1, # L2 reg
    'alpha': 0, # L1 reg
    'seed': SEED
}


cat_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'iterations': 500, # Increase estimators, use early stopping
    'learning_rate': 0.04,
    'depth': 5,
    'l2_leaf_reg': 3,
    'subsample': 0.7,
    'colsample_bylevel': 0.7, # CatBoost specific feature sampling
    'random_seed': SEED,
    'verbose': 0, # Suppress verbose output during CV
    'early_stopping_rounds': 50 # Use early stopping
}

xtrees_params = {
    'n_estimators': 300, # Slightly reduced
    'max_depth': 12, # Control complexity
    'min_samples_leaf': 15,
    'min_samples_split': 10,
    'random_state': SEED,
    'n_jobs': -1
}

ordinal_params = {
    'alpha': 1.0,
    'fit_intercept': True
}


if optimize_params:
    print("\n--- Running Optuna Hyperparameter Tuning ---")
    # Define a CV split specifically for Optuna (can be fewer splits for speed)
    kf_for_optuna = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Calculate weights based on the target score distribution IN THE FINAL 'train' DATAFRAME
    if y_model_col in train.columns and not train[y_model_col].isnull().all():
        weights_for_optuna = calculate_weights(train[y_model_col])
    else:
        weights_for_optuna = calculate_weights(train[y_comp_col]) # Fallback

    # Tune LightGBM
    print("\nOptimizing LightGBM...")
    best_lgb_params = run_optimization(
        train, selected_features, y_model_col, y_comp_col,
        model_type='lightgbm',
        n_trials=n_optuna_trials,
        cv=kf_for_optuna,
        sample_weights_series=weights_for_optuna
    )
    lgb_params.update(best_lgb_params) # Update the main dict with best params found
    print("Updated LGBM Params:", lgb_params) # Print updated params

    print("\nOptimizing XGBoost...")
    best_xgb_params = run_optimization(
        train, selected_features, y_model_col, y_comp_col, # Pass necessary args
        model_type='xgboost',                             # Specify model type
        n_trials=n_optuna_trials,
        cv=kf_for_optuna,
        sample_weights_series=weights_for_optuna         # Pass weights
    )
    xgb_params.update(best_xgb_params)                     # Update the dict
    print("Updated XGBoost Params:", xgb_params)

    print("\nOptimizing CatBoost...")
    best_cat_params = run_optimization(
        train, selected_features, y_model_col, y_comp_col, # Pass necessary args
        model_type='catboost',                            # Specify model type
        n_trials=n_optuna_trials,
        cv=kf_for_optuna,
        sample_weights_series=weights_for_optuna         # Pass weights
    )
    cat_params.update(best_cat_params)                     # Update the dict
    print("Updated CatBoost Params:", cat_params)

    # Tune Ordinal Ridge (if mord installed and objective defined)
    if mord is not None:
         print("\nOptimizing Ordinal Ridge...")
         # Make sure 'ordinal_ridge' is handled in the objective function
         best_ord_params = run_optimization(
             train, selected_features, y_model_col, y_comp_col, # score_col not used but passed
             model_type='ordinal_ridge',
             n_trials=n_optuna_trials // 2, # Maybe fewer trials for simpler model
             cv=kf_for_optuna,
             sample_weights_series=None # OrdinalRidge doesn't use sample_weight in fit
         )
         # If you had an ordinal_params dict, update it here. Otherwise, store separately.
         ordinal_params.update(best_ord_params) # Example
         print(f"Best OrdinalRidge params: {best_ord_params}") # Store/use as needed


    print("\n--- Optuna Tuning Finished ---")
    print("Updated LGBM Params:", lgb_params)
    # Print other updated params...

else:
    print("\nSkipping Optuna tuning, using predefined parameters.")


# Define models
lgb_model = LGBMRegressor(**lgb_params)
xgb_model = XGBRegressor(**xgb_params)
cat_model = CatBoostRegressor(**cat_params)
xtrees_model = ExtraTreesRegressor(**xtrees_params)

# !! NEW: Add Ordinal Model !!
ordinal_model = None
if mord is not None:
    try:
        # Initialize using the potentially updated ordinal_params dict
        ordinal_model = mord.OrdinalRidge(**ordinal_params)
        print(f"Initializing OrdinalRidge with parameters: {ordinal_params}")
    except Exception as e:
        # Fallback if initialization fails for some reason
        print(f"Error initializing OrdinalRidge with params {ordinal_params}: {e}")
        print("Falling back to basic OrdinalRidge initialization.")
        ordinal_model = mord.OrdinalRidge() # Simplest fallback


# --- Cross-Validation Execution ---
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Calculate weights based on the target score distribution in the combined training data
# Use PCIAT score if reliable, otherwise fallback to sii (though less ideal for regression)
if y_model_col in train.columns and not train[y_model_col].isnull().all():
    print(f"Calculating sample weights based on {y_model_col}")
    weights = calculate_weights(train[y_model_col])
else:
    print(f"Warning: {y_model_col} not suitable for weights. Calculating based on {y_comp_col}.")
    weights = calculate_weights(train[y_comp_col]) # Less ideal but fallback


print("\n--- Cross-Validating Regressor Models ---")
oof_preds_regressors = {}
oof_indices_regressors = {}
thresholds_regressors = {}
models_regressors = {}
overall_kappa_regressors = {}


# LGBM
print("Cross-validating LGBM...")
score_lgb, oof_preds_regressors['lgb'], oof_indices_regressors['lgb'], thresholds_regressors['lgb'], models_regressors['lgb'], overall_kappa_regressors['lgb'] = cross_validate_regressor(
    lgb_model, train, selected_features, y_model_col, y_comp_col, kf, sample_weights_series=weights, verbose=True
)

# XGB
print("\nCross-validating XGBoost...")
score_xgb, oof_preds_regressors['xgb'], oof_indices_regressors['xgb'], thresholds_regressors['xgb'], models_regressors['xgb'], overall_kappa_regressors['xgb'] = cross_validate_regressor(
    xgb_model, train, selected_features, y_model_col, y_comp_col, kf, sample_weights_series=weights, verbose=True
)

# CatBoost
print("\nCross-validating CatBoost...")
score_cat, oof_preds_regressors['cat'], oof_indices_regressors['cat'], thresholds_regressors['cat'], models_regressors['cat'], overall_kappa_regressors['cat'] = cross_validate_regressor(
    cat_model, train, selected_features, y_model_col, y_comp_col, kf, sample_weights_series=weights, verbose=True
)

# ExtraTrees
print("\nCross-validating ExtraTrees...")
# ExtraTrees doesn't natively support sample weights in fit, so we omit them here
score_xtrees, oof_preds_regressors['xtrees'], oof_indices_regressors['xtrees'], thresholds_regressors['xtrees'], models_regressors['xtrees'], overall_kappa_regressors['xtrees'] = cross_validate_regressor(
    xtrees_model, train, selected_features, y_model_col, y_comp_col, kf, sample_weights_series=None, verbose=True # No weights for ET
)


# cross val ordinal model
oof_indices_ordinal = {}
models_ordinal = {}
overall_kappa_ordinal = {}

if ordinal_model is not None:
    print("\n--- Cross-Validating Ordinal Model ---")
    score_ord, oof_indices_ordinal['ord'], models_ordinal['ord'], overall_kappa_ordinal['ord'] = cross_validate_ordinal(
        ordinal_model, train, selected_features, y_comp_col, kf, verbose=True
    )
else:
    print("\nSkipping Ordinal Model cross-validation (mord library not found or model not defined).")


# ensemble prep
avg_thresholds = {}
for model_name, fold_thresholds in thresholds_regressors.items():
    if fold_thresholds: # Check if list is not empty
         avg_thresholds[model_name] = np.mean(np.array(fold_thresholds), axis=0)
    else:
         print(f"Warning: No thresholds found for {model_name}. Using base thresholds.")
         avg_thresholds[model_name] = base_thresholds


# Prepare OOF predictions for stacking
# Use the thresholded index predictions from regressors and direct predictions from ordinal
oof_stacking_features = pd.DataFrame(oof_indices_regressors)
if 'ord' in oof_indices_ordinal:
    oof_stacking_features['ord'] = oof_indices_ordinal['ord']

print("\nOOF Predictions for Stacking:")
print(oof_stacking_features.head())
print(f"Correlation of OOF index predictions:\n{oof_stacking_features.corr()}")


# stacking ensemble
from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression(random_state=SEED, C=1.0) # Adjust C (regularization) if needed


print("Fitting meta-model on OOF features...")
meta_model.fit(oof_stacking_features, train[y_comp_col])
print("Meta-model fitting complete.")


# Evaluate meta-model on OOF predictions
oof_stacking_preds = meta_model.predict(oof_stacking_features)
stacking_oof_kappa = cohen_kappa_score(train[y_comp_col], oof_stacking_preds, weights='quadratic')
print(f"\nStacking Ensemble OOF Kappa: {stacking_oof_kappa:.4f}")


# Plot Confusion Matrix for Stacking Ensemble
print("\nPlotting Stacking Ensemble Confusion Matrix...")
conf_matrix_stacking = confusion_matrix(train[y_comp_col], oof_stacking_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacking, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='black',
            xticklabels=range(4), yticklabels=range(4))
plt.title('Stacking Ensemble Confusion Matrix (OOF)', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()


# final model trg

print("\n--- Training Final Models on Full Data ---")
final_models = {}

# Regressors
print("Training LGBM...")
final_models['lgb'] = clone(lgb_model).fit(train[selected_features], train[y_model_col], sample_weight=weights)
print("Training XGBoost...")
final_models['xgb'] = clone(xgb_model).fit(train[selected_features], train[y_model_col], sample_weight=weights)
print("Training CatBoost...")
# Use early stopping with a validation set split from train for CatBoost final fit
X_train_final, X_val_final, y_train_final, y_val_final, w_train_final, _ = train_test_split(
    train[selected_features], train[y_model_col], weights, test_size=0.1, random_state=SEED, stratify=train[y_comp_col] # Stratify by sii
)
final_models['cat'] = clone(cat_model).fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=0, sample_weight=w_train_final)
print("Training ExtraTrees...")
final_models['xtrees'] = clone(xtrees_model).fit(train[selected_features], train[y_model_col]) # No weights

# Ordinal
if ordinal_model is not None:
    print("Training OrdinalRidge...")
    final_models['ord'] = clone(ordinal_model).fit(train[selected_features], train[y_comp_col])


# Predict on test set with base models
test_preds_regressors = {}
test_indices_regressors = {}
test_indices_ordinal = {}

print("Predicting on test set with base models...")
for name, model in final_models.items():
    if name in avg_thresholds: # Regressor models
        test_scores = model.predict(test[selected_features])
        test_preds_regressors[name] = test_scores
        test_indices_regressors[name] = round_with_thresholds(test_scores, avg_thresholds[name])
    elif name == 'ord': # Ordinal model
        test_indices_ordinal[name] = model.predict(test[selected_features])


# Prepare test features for stacking meta-model
test_stacking_features = pd.DataFrame(test_indices_regressors)
if 'ord' in test_indices_ordinal:
    test_stacking_features['ord'] = test_indices_ordinal['ord']


# Ensure column order matches OOF features used for training meta-model
test_stacking_features = test_stacking_features[oof_stacking_features.columns]


# final preds using stacking ensemble
print("Predicting final labels using Stacking Ensemble...")
final_test_predictions = meta_model.predict(test_stacking_features)


print("Creating submission file...")
submission = pd.read_csv(SUBMISSION_PATH)
submission['sii'] = final_test_predictions.astype(int) # Ensure integer type

# Define the output path for the submission file
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")


submission.to_csv(submission_path, index=False)
print(f"Submission file saved to: {submission_path}")
print("Value Counts in Submission:")
print(submission['sii'].value_counts())

print("\nScript finished.")


# # --- !! ADD THIS SECTION TO PRINT FINAL OOF SCORES !! ---

# print("\n--- Final OOF QWK Scores ---")

# # Calculate and print QWK for each base regressor model
# print("Base Regressor Models (OOF):")
# for name, oof_preds in oof_indices_regressors.items():
#     if len(oof_preds) == len(train): # Ensure prediction array has correct length
#         qwk = cohen_kappa_score(train[y_comp_col], oof_preds, weights='quadratic')
#         print(f"  - {name.upper()}: {qwk:.4f}")
#     else:
#         print(f"  - {name.upper()}: Error calculating score (length mismatch)")

# # Calculate and print QWK for the ordinal model (if used)
# if 'ord' in oof_indices_ordinal:
#     print("\nBase Ordinal Model (OOF):")
#     if len(oof_indices_ordinal['ord']) == len(train):
#         qwk_ord = cohen_kappa_score(train[y_comp_col], oof_indices_ordinal['ord'], weights='quadratic')
#         print(f"  - OrdinalRidge: {qwk_ord:.4f}")
#     else:
#          print(f"  - OrdinalRidge: Error calculating score (length mismatch)")


# # Print the stacking ensemble QWK (already calculated)
# print("\nStacking Ensemble Model (OOF):")
# # Ensure stacking_oof_kappa was calculated previously
# try:
#     print(f"  - Stacking Ensemble: {stacking_oof_kappa:.4f}")
# except NameError:
#     # If stacking_oof_kappa wasn't calculated or stored, recalculate it
#     if 'oof_stacking_preds' in locals() and len(oof_stacking_preds) == len(train):
#          stacking_oof_kappa = cohen_kappa_score(train[y_comp_col], oof_stacking_preds, weights='quadratic')
#          print(f"  - Stacking Ensemble: {stacking_oof_kappa:.4f}")
#     else:
#          print(f"  - Stacking Ensemble: Error calculating score (predictions unavailable or length mismatch)")


# # --- (Continue with final test set prediction and submission generation...) ---


# import joblib


# # --- (Previous code: Final model training, including final_models dict and meta_model.fit()) ---

# # --- !! ADD THIS SECTION TO SAVE FINAL MODELS !! ---

# print("\n--- Saving Final Trained Models ---")

# # Define a directory to save the models
# MODEL_SAVE_DIR = os.path.join(OUTPUT_PATH, "final_models") # Using OUTPUT_PATH defined earlier
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create directory if it doesn't exist
# print(f"Models will be saved in: {MODEL_SAVE_DIR}")

# # Save the base models stored in the final_models dictionary
# for name, model in final_models.items():
#     save_path = os.path.join(MODEL_SAVE_DIR, f"{name}_model") # Base path without extension

#     try:
#         print(f"Saving model: {name}...")
#         if isinstance(model, LGBMRegressor):
#             model.booster_.save_model(f"{save_path}.lgbm") # Use LightGBM's method
#         elif isinstance(model, XGBRegressor):
#             model.save_model(f"{save_path}.xgb") # Use XGBoost's method
#         elif isinstance(model, CatBoostRegressor):
#             model.save_model(f"{save_path}.cbm") # Use CatBoost's method
#         elif isinstance(model, (ExtraTreesRegressor, mord.OrdinalRidge)): # Models compatible with joblib
#              joblib.dump(model, f"{save_path}.joblib")
#         else:
#             print(f"  - Warning: Unknown model type '{type(model)}' for '{name}'. Attempting joblib dump.")
#             joblib.dump(model, f"{save_path}.joblib") # Fallback attempt
#         print(f"  - Saved {name} successfully.")

#     except Exception as e:
#         print(f"  - Error saving model {name}: {e}")


# # Save the stacking meta-model (Logistic Regression)
# try:
#     print("Saving stacking meta-model...")
#     meta_model_save_path = os.path.join(MODEL_SAVE_DIR, "stacking_meta_model.joblib")
#     joblib.dump(meta_model, meta_model_save_path)
#     print("  - Saved stacking meta-model successfully.")
# except Exception as e:
#      print(f"  - Error saving stacking meta-model: {e}")

# print("--- Model Saving Complete ---")

# # --- (Continue with test set prediction, submission generation, etc.) ---

