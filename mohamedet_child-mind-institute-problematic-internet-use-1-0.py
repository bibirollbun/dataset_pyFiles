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


import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import glob
import scipy

from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import time
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.optimize import minimize
from collections import Counter
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import warnings
warnings.filterwarnings('ignore')


def set_global_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dict = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')


train.head(10)


train_cols = set(train.columns)
test_cols = set(test.columns)
columns_not_in_test = sorted(list(train_cols - test_cols))
data_dict[data_dict['Field'].isin(columns_not_in_test)]


pciat_min_max = train.groupby('sii')['PCIAT-PCIAT_Total'].agg(['min', 'max'])
pciat_min_max = pciat_min_max.rename(
    columns={'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'}
)
pciat_min_max


data_dict[data_dict['Field'] == 'PCIAT-PCIAT_Total']['Value Labels'].iloc[0]


train_with_sii = train[train['sii'].notna()][columns_not_in_test]
train_with_sii[train_with_sii.isna().any(axis=1)].head().style.applymap(
    lambda x: 'background-color: #FFC0CB' if pd.isna(x) else ''
)


PCIAT_cols = [f'PCIAT-PCIAT_{i+1:02d}' for i in range(20)]
recalc_total_score = train_with_sii[PCIAT_cols].sum(
    axis=1, skipna=True
)
(recalc_total_score == train_with_sii['PCIAT-PCIAT_Total']).all()


def recalculate_sii(row):
    if pd.isna(row['PCIAT-PCIAT_Total']):
        return np.nan
    max_possible = row['PCIAT-PCIAT_Total'] + row[PCIAT_cols].isna().sum() * 5
    if row['PCIAT-PCIAT_Total'] <= 30 and max_possible <= 30:
        return 0
    elif 31 <= row['PCIAT-PCIAT_Total'] <= 49 and max_possible <= 49:
        return 1
    elif 50 <= row['PCIAT-PCIAT_Total'] <= 79 and max_possible <= 79:
        return 2
    elif row['PCIAT-PCIAT_Total'] >= 80 and max_possible >= 80:
        return 3
    return np.nan

train['recalc_sii'] = train.apply(recalculate_sii, axis=1)


mismatch_rows = train[
    (train['recalc_sii'] != train['sii']) & train['sii'].notna()
]

mismatch_rows[PCIAT_cols + [
    'PCIAT-PCIAT_Total', 'sii', 'recalc_sii'
]].style.applymap(
    lambda x: 'background-color: #FFC0CB' if pd.isna(x) else ''
)


train['sii'] = train['recalc_sii']
train['complete_resp_total'] = train['PCIAT-PCIAT_Total'].where(
    train[PCIAT_cols].notna().all(axis=1), np.nan
)

sii_map = {0: '0 (None)', 1: '1 (Mild)', 2: '2 (Moderate)', 3: '3 (Severe)'}
train['sii'] = train['sii'].map(sii_map).fillna('Missing')

sii_order = ['Missing', '0 (None)', '1 (Mild)', '2 (Moderate)', '3 (Severe)']
train['sii'] = pd.Categorical(train['sii'], categories=sii_order, ordered=True)

train.drop(columns='recalc_sii', inplace=True)


sii_counts = train['sii'].value_counts().reset_index()
total = sii_counts['count'].sum()
sii_counts['percentage'] = (sii_counts['count'] / total) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SII
sns.barplot(x='sii', y='count', data=sii_counts, palette='Blues_d', ax=axes[0])
axes[0].set_title('Distribution of Severity Impairment Index (sii)', fontsize=14)
for p in axes[0].patches:
    height = p.get_height()
    percentage = sii_counts.loc[sii_counts['count'] == height, 'percentage'].values[0]
    axes[0].text(
        p.get_x() + p.get_width() / 2,
        height + 5, f'{int(height)} ({percentage:.1f}%)',
        ha="center", fontsize=12
    )

# PCIAT_Total for complete responses
sns.histplot(train['complete_resp_total'].dropna(), bins=20, ax=axes[1])
axes[1].set_title('Distribution of PCIAT_Total', fontsize=14)
axes[1].set_xlabel('PCIAT_Total for Complete PCIAT Responses')

plt.tight_layout()
plt.show()


len(train[train['complete_resp_total'] == 0])


assert train['Basic_Demos-Age'].isna().sum() == 0
assert train['Basic_Demos-Sex'].isna().sum() == 0


### ---- Sensor Summary Feature Functions ---- ###
def describe_series_basic_stats(series, prefix):
    """Extract basic statistics from a time series."""
    return {
        f"{prefix}_mean": series.mean(),
        f"{prefix}_std": series.std(),
        f"{prefix}_min": series.min(),
        f"{prefix}_max": series.max(),
        f"{prefix}_range": series.max() - series.min(),
        f"{prefix}_skew": series.skew(),
        f"{prefix}_kurtosis": series.kurtosis(),
    }

def describe_all_axes(df):
    """Summarize stats across X, Y, Z axis."""
    all_stats = {}
    for axis in ['X', 'Y', 'Z']:
        all_stats.update(describe_series_basic_stats(df[axis], axis.lower()))
    return all_stats

def describe_enmo(df):
    """Summarize ENMO (acceleration magnitude)."""
    return describe_series_basic_stats(df['enmo'], 'enmo')

def describe_anglez(df):
    """Summarize device angle (anglez)."""
    return describe_series_basic_stats(df['anglez'], 'anglez')

### ---- Light Feature Engineering ---- ###

light_bins = [
    (0, 5, 'Twilight'),
    (5, 10, 'Minimal_Street_Lighting'),
    (10, 50, 'Sunset'),
    (50, 80, 'Family_Living_Room'),
    (80, 100, 'Hallway'),
    (100, 320, 'Very_Dark_Overcast_Day'),
    (320, 500, 'Office_Lighting'),
    (500, 1000, 'Sunrise_Sunset'),
    (1000, 10000, 'Overcast_Day'),
    (10000, 25000, 'Full_Daylight'),
    (25000, 130000, 'Direct_Sunlight')
]

def categorize_light(value):
    """Categorize light level (in lux) based on standard environmental conditions."""
    for low, high, label in light_bins:
        if low <= value < high:
            return label
    return 'Unknown'

def describe_light(df):
    """Compute normalized proportions of time spent in each light category."""
    df['light_category'] = df['light'].apply(categorize_light)
    category_counts = df['light_category'].value_counts(normalize=True).to_dict()
    return {f"light_{label}": category_counts.get(label, 0) for _, _, label in light_bins}

### ---- Streak Features (Activity / Inactivity Windows) ---- ###

def compute_streaks(series, condition, top_n=5):
    """Generic function to compute longest streaks satisfying a condition."""
    streak_lengths = []
    current_streak = 0

    for val in condition:
        if val:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0

    if current_streak > 0:
        streak_lengths.append(current_streak)

    streak_lengths = sorted(streak_lengths, reverse=True)[:top_n]
    streak_lengths += [0] * (top_n - len(streak_lengths))
    return streak_lengths

def longest_inactivity_streaks(df, window_size=100, threshold=10, top_n=5):
    """Detect long stretches of inactivity (rolling ENMO sum below threshold)."""
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    inactive_windows = rolling_cumsum <= threshold
    return compute_streaks(df['enmo'], inactive_windows, top_n)

def longest_activity_streaks(df, window_size=100, threshold=5, top_n=5):
    """Detect long stretches of activity (rolling ENMO sum above threshold)."""
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    active_windows = rolling_cumsum > threshold
    return compute_streaks(df['enmo'], active_windows, top_n)

### ---- File Processing ---- ###

def process_file(filename, dirname):
    """Read a parquet file, extract features, and return feature dict with ID."""
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop(['step'], axis=1, inplace=True)

    features = {}

    # Axes stats
    features.update(describe_all_axes(df))

    # ENMO & anglez
    features.update(describe_enmo(df))
    features.update(describe_anglez(df))

    # Light analysis
    features.update(describe_light(df))

    # Activity ratio
    features['enmo_active_ratio'] = (df['enmo'] > 0).mean()

    # Longest inactivity/activity streaks
    for i, val in enumerate(longest_inactivity_streaks(df, threshold=1)):
        features[f'inact_streak_{i}'] = val

    for i, val in enumerate(longest_activity_streaks(df, threshold=5)):
        features[f'act_streak_{i}'] = val

    sample_id = filename.split('=')[1]
    return features, sample_id

### ---- Bulk Processing ---- ###

def load_time_series(dirname) -> pd.DataFrame:
    """Process all files in directory and assemble final dataframe."""
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    features_list, sample_ids = zip(*results)

    df = pd.DataFrame(features_list)
    df['id'] = sample_ids

    return df



train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")


def feature_engineering(df):

    for col, (col_min, col_max) in min_max_dict.items():
        df[col] = df[col].clip(lower=col_min, upper=col_max)

    bins = [0, 6, 12, 18, 100]
    labels = ['1 to 6', '7 to 12', '13 to 18', '19 to 100']
    df['Age_Binned'] = pd.cut(df['Basic_Demos-Age'], bins=bins, labels=labels, right=True)
    df['Age_Sex'] = df['Age_Binned'].astype(str) + '_' + df['Basic_Demos-Sex'].astype(str)
    
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    
    df['PreInt_FGC_CU_PU'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['FGC-FGC_CU'] * df['FGC-FGC_PU']
    df['FGC_GSND_GSD_Age'] = df['FGC-FGC_GSND'] * df['FGC-FGC_GSD'] * df['Basic_Demos-Age']
    df['SDS_Activity'] = df['BIA-BIA_Activity_Level_num'] * df['SDS-SDS_Total_T']
    
    df['CGasync_Score_Normalized'] = df['CGAS-CGAS_Score'] - df.groupby('Basic_Demos-Enroll_Season')['CGAS-CGAS_Score'].transform('mean')
    df['Internet_Physical_Difference'] = df['PreInt_EduHx-computerinternet_hoursday'] - df['PAQ_A-PAQ_A_Total']
   
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').astype('category')
    return df


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')


numeric_cols = train[test.columns].select_dtypes(include='number').columns
min_max_dict = {col: (train[col].min(), train[col].max()) for col in numeric_cols}

train = feature_engineering(train)
test = feature_engineering(test)

train = train.drop('id', axis=1)
test  = test .drop('id', axis=1)   

train = train.dropna(subset='sii')

target = train['PCIAT-PCIAT_Total']
sii_target = train['sii']
train = train[test.columns]


def map_pciat_to_sii(pciat_values):
    return np.select(
        [pciat_values <= 30, 
         (pciat_values > 30) & (pciat_values <= 49),
         (pciat_values > 49) & (pciat_values <= 79),
         pciat_values > 79],
        [0, 1, 2, 3],
        default=3  # For PCIAT values greater than 79
    )
    
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def select_subset(df, target, subset_size=0.8):
    df_subset = df.sample(frac=subset_size, random_state=42)
    target_subset = target.loc[df_subset.index]
    return df_subset, target_subset


def gaussian_noise_injection(df, target, noise_level, subset_size=0.2):

    # Select a subset of data for augmentation
    df_subset, target_subset = select_subset(df, target, subset_size)

    # Split numeric and non-numeric columns
    numeric_cols = df_subset.select_dtypes(include=['float64', 'int64'])
    non_numeric_cols = df_subset.select_dtypes(exclude=['float64', 'int64'])

    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy='mean')
    numeric_imputed = pd.DataFrame(imputer.fit_transform(numeric_cols), 
                                   columns=numeric_cols.columns, 
                                   index=numeric_cols.index)

    # Add noise to numeric columns
    augmented_numeric = numeric_imputed
    for col in augmented_numeric.columns:
        std_dev = augmented_numeric[col].std()
        if std_dev > 0:  # Add noise only if variability exists
            noise = np.random.normal(0, noise_level * std_dev, size=len(augmented_numeric))
            augmented_numeric[col] += noise

    # Concatenate back with non-numeric columns (align rows)
    augmented_df = pd.concat([augmented_numeric, non_numeric_cols], axis=1)

    # Ensure the column order matches the original subset
    augmented_df = augmented_df[df_subset.columns]
    return augmented_df, target_subset


def augment_data_with_nans(X, target, threshold=0.1, subset_size=0.2):
   
    df_subset, target_subset = select_subset(X, target, subset_size)
    X_augmented = df_subset.reset_index(drop=True).copy()
    
    # Identify columns that already contain NaN values
    columns_with_nan = [col for col in X.columns if X[col].isna().sum() > 0]
    
    # Mask for non-NaN values in columns that contain NaNs
    non_nan_mask = X_augmented[columns_with_nan].notna()
    
    # Randomly select which column to set to NaN (for each row) where there's a valid value
    for col in columns_with_nan:
        # Create a random mask for columns with valid values (non-NaN)
        random_mask = np.random.rand(len(X_augmented)) < threshold  # Adjust probability as needed
        
        # Apply the mask to select rows and set that column's value to NaN
        X_augmented.loc[random_mask, col] = np.nan
    
    return X_augmented, target_subset


def plot_confusion_matrix(y_true, y_pred, labels=None):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    
    if labels is None:
        labels = sorted(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def cross_validate_model(params, X, y, sii_target, label='', save_models=True, pruning_callback=None, n_repeats=5, return_qwk=False):
    features = X.columns
    start_time = time.time()
    oof = []
    y_oof = []
    qwk_list = []
    model_list = []
   
    n = 0
    for repeat in tqdm(range(n_repeats)):
        random_seed = np.random.randint(0, 10000)  # Generate a random seed for each repeat
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
        
        for fold, (idx_tr, idx_va) in enumerate(folds.split(X, sii_target)):
            params['random_seeds'] = n
            set_global_seed(n)
            X_tr = X.iloc[idx_tr]
            X_va = X.iloc[idx_va]
            y_tr = y.iloc[idx_tr]
            y_va = y.iloc[idx_va]
            
            
            nan_prone_columns = [
                col for col in X_tr.columns 
                if X_tr[col].isna().any()  # Has NaNs
            ]
    
            # Step 1: Perform augmentation on X_tr
            nan_augmented, nan_aug_target = augment_data_with_nans(X_tr, target, threshold=1, subset_size=0.2)
            noise_augmented, noise_aug_target = gaussian_noise_injection(X_tr, y_tr, noise_level=0.02, subset_size=0.5)
    
            X_tr_augmented = pd.concat(
                [nan_augmented, noise_augmented, X_tr[y_tr>49], X_tr[y_tr>49], X_tr[y_tr>49], X_tr[y_tr>79]],
                ignore_index=True).reset_index(drop=True)
            
            y_tr_augmented = pd.concat(
                [nan_aug_target, noise_aug_target, y_tr[y_tr>49], y_tr[y_tr>49], y_tr[y_tr>49], y_tr[y_tr>79]],
                ignore_index=True).reset_index(drop=True)


            X_tr_combined = pd.concat([X_tr, X_tr_augmented], ignore_index=True).reset_index(drop=True)
            y_tr_combined = pd.concat([y_tr, y_tr_augmented], ignore_index=True).reset_index(drop=True)

            shuffled_indices = np.random.permutation(X_tr_combined.index)
            X_tr_combined = X_tr_combined.iloc[shuffled_indices].reset_index(drop=True)
            y_tr_combined = y_tr_combined.iloc[shuffled_indices].reset_index(drop=True)

            
            dtrain = lgb.Dataset(X_tr_combined, label=y_tr_combined)
            dvalid = lgb.Dataset(X_va, label=y_va)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dvalid],
                num_boost_round=params['n_estimators'],
            )

            y_pred = model.predict(X_va)

            if save_models:
                model_list.append(model)
            oof.append(y_pred)
            y_oof.append(y_va)
            
            n +=1
    elapsed_time = time.time() - start_time

    y_oof_actuals = np.concatenate(y_oof)
    oof_preds = np.concatenate(oof)
    
    # Post-processing: Map predictions
    y_oof_sii = map_pciat_to_sii(y_oof_actuals)
    oof_sii = map_pciat_to_sii(oof_preds)

  
    qwk = cohen_kappa_score(y_oof_sii, oof_sii, weights='quadratic')
    mse = ((y_oof_actuals - oof_preds)**2).mean()  
    print(f"Overall QWK: {qwk:.3f}, MSE: {mse:.3f}, Time: {int((time.time() - start_time) / 60)} min")

    # Optimize thresholds
    threshold_optimizer = minimize(evaluate_predictions, 
                                   x0=[34, 49, 62], 
                                   args=(y_oof_sii, oof_preds), 
                                   method='Nelder-Mead')
    
    optimized_preds = threshold_Rounder(oof_preds, threshold_optimizer.x)
    optimized_qwk = cohen_kappa_score(y_oof_sii, optimized_preds, weights='quadratic')
    accuracy = (y_oof_sii==optimized_preds).astype(np.float32).mean()
    print(f"Optimized QWK: {optimized_qwk:.3f}, Accuracy: {accuracy:.3f}, Thresholds: {threshold_optimizer.x}")
    
    plot_confusion_matrix(y_oof_sii, oof_sii)
    
    if save_models:
        saved_models[label] = {'features': features, 'model_list': model_list}

    return optimized_qwk, threshold_optimizer.x


saved_models = {}
results = []
for i in range(1):
    params = {'verbosity': -1,  'device': 'cpu', 'metric': 'mse', 'n_estimators':150, 'max_depth':5, 'max_bin': 15, 'boosting_type': 'gbdt', 'lambda_l1': 0.0012071403780584485, 'lambda_l2': 19.943477818207878, 'min_child_weight': 0.01586977190723854, 'learning_rate': 0.030512450456770007, 'num_leaves': 295, 'colsample_bytree': 0.8569995659929517, 'bagging_fraction': 0.587037100215173, 'feature_fraction': 0.8955475330753205, 'bagging_freq': 1}
    qwk, qwk_thresholded = cross_validate_model(params, train, target, sii_target, label='trial', save_models=True, n_repeats=100)
    print(qwk)
    results.append(qwk)
print(f"'mean {np.mean(results)}")
print(f"diff {max(results) - min(results)}")


pred = [model.predict(test)  for model in saved_models['trial']['model_list']]

n = 16
i = 500
plt.hist(np.array(pred)[:, n][:i], bins=30, alpha=0.7)

# Get the mode
mode_val = stats.mode(np.array(pred)[:, n][:i].round())[0]  # mode.value[0]

# Overlay the mode on the histogram
plt.axvline(mode_val, color='k', linestyle='dashed', linewidth=2, label=f'Mode: {mode_val}')
plt.axvline(np.array(pred)[:, n][:i].mean(), color='r', linestyle='dashed', linewidth=2, label=f'mean: {np.array(pred)[:, n][:i].mean()}')
# Add a label
plt.legend()

plt.show()


predictions = stats.mode(threshold_Rounder(np.array([model.predict(test) for model in saved_models['trial']['model_list']]), qwk_thresholded).astype(np.int32))[0]


submission_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

submission_df['sii'] = predictions
submission_df.to_csv('submission.csv', index=False)
pd.read_csv('./submission.csv')

