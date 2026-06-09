# ==================== CONFIGURATIONS ====================
class CFG:
    
    EXTRA_TRAIN_PATH = '/kaggle/input/bpm-prediction-challenge/Train.csv'
    TRAIN_PATH       = '/kaggle/input/playground-series-s5e9/train.csv'
    TEST_PATH        = '/kaggle/input/playground-series-s5e9/test.csv'
    TARGET           = 'BeatsPerMinute'

    FOLDS            = 5
    SEED             = 42
    TASK_TYPE        = 'regression'

    ADD_EXTRA_DATA   = False
    ADD_FEATURES     = True
    ADD_INTERACTIONS = True
    ADD_RATIOS       = False
    OPTUNA_SEARCH    = False
    OPTUNA_N_TRIALS  = 20
    SEARCH_WEIGHTS   = False


# ==================== INSTALL & IMPORT LIBRARIES ====================

!pip install itables
!pip install optuna-integration[xgboost]==4.3.0
!pip install catboost==1.2.8
!pip install scikit-learn==1.3.1 # for target_encoder

# !pip install ydata_profiling
# !pip install imbalanced-learn==0.12.2

# =========================================================================

import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

# =========================================================================

# Sets the seed for reproducibility in numpy, random, torch CPU, and CUDA.
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(CFG.SEED)
    torch.cuda.manual_seed_all(CFG.SEED) # For multi-GPU setups.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #May slightly slow down training, but ensures reproducibility

# =========================================================================
# Set plot style
sns.set_style('darkgrid')

# # Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== HELPER FUNCTIONS ====================

def iqr_outlier_capping(train, valid=None, test=None, columns=None):
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy() # Avoid modifying the original DataFrame
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()  # All numerical columns

    # Calculate IQR-based thresholds from the training set
    # w/ .dropna() to handle cols with nans: required by np.percentile
    for col in columns:
        Q1 = np.percentile(train[col].dropna(), 25)
        Q3 = np.percentile(train[col].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Show Values
        # print(f'Columns {col}: \tLower Bound is: {lower_bound:.2f} \tUpper Bound is: {upper_bound:.2f}')

        # Cap outliers in the training set
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)

        # If validation set is provided, cap using training set thresholds
        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)

        # If test set is provided, cap using training set thresholds
        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped
    
    # EXAMPLE USE: TRAIN_capped, _, TEST_capped = iqr_outlier_capping(TRAIN_DF.dropna(), None, TEST_DF, columns=TRAIN_DF.select_dtypes('number').columns.difference([target]))

# Analysis of all NUMERIC features

# ============================================================
# Function to create and display plots for a single numerical variable
def numeric_univariate_plots(train, test, extra, target):

    # Select columns
    focus_cols = train.select_dtypes(np.number).columns.difference([target])

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train[focus_cols].copy()
    test_temp = test[focus_cols].copy()
    extra_temp = extra[focus_cols].copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    extra_temp["Dataset"] = "Extra"
    combined_data = pd.concat([train_temp, test_temp, extra_temp])

    # Start loop
    for col in focus_cols:

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}
    
        # Box plot
        sns.boxplot(data=combined_data, x=col, y="Dataset", palette='viridis', ax=axes[0])
        axes[0].set_xlabel(col)
        axes[0].set_title(f"Box Plot of {col}")
    
        # Histogram
        sns.histplot(data=combined_data, x=col, hue='Dataset', palette='viridis', bins=50, 
                     stat='density', common_norm=False, multiple='dodge')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Histogram of {col} [Train, Test, Extra]")
        # axes[1].legend()
        axes[1].annotate(f"Skewness (TRAIN): {train[col].skew():.2f}\nKurtosis (TRAIN): {train[col].kurt():.2f}",
                         xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
    
    
        # Adjust spacing and show
        plt.tight_layout()
        plt.show()
        
# ============================================================
def print_with_sep(text,sep="=",n=30):
  print("\n")
  print(sep*n)
  print('\t',text)
  print(sep*n)

# ============================================================
def print_dataset_overview(datasets):

    # Check shapes
    print_with_sep("Shapes")
    for name, df in datasets.items():
      print(f"{name} shape: {df.shape}")
    
    # Check duplicates
    print_with_sep("Duplicates")
    for name, df in datasets.items():
      print(f"{name} duplicates: {df.duplicated().sum()}")
    
    # Check nans
    print_with_sep("NaNs")
    for name, df in datasets.items():
      print(f"{name} NaNs: {df.isnull().sum().sum()}")
    
    # Check col difference
    print_with_sep("Columns not in test")
    print(set(TRAIN_DF.columns).difference(set(TEST_DF.columns)))

    # Check descriptive stats
    print_with_sep("Descriptive Statistics")
    for name, df in datasets.items():
      print(f"{name} Description:")
      percentage_missing = df.isnull().sum()/df.shape[0]; percentage_missing.name = '% Missing'
      data_types = df.dtypes; data_types.name = 'd_type'
    
      display(
          pd.concat([
              df.describe(include='all').T,
              percentage_missing,
              data_types],
                    axis=1).replace(np.nan,'-').style.background_gradient(cmap='Blues'))
      print("\n")


import os

# Load datasets (on Kaggle)
TRAIN_DF    = pd.read_csv(CFG.TRAIN_PATH, index_col = 'id')
TEST_DF     = pd.read_csv(CFG.TEST_PATH, index_col = 'id')
TRAIN_EXTRA = pd.read_csv(CFG.EXTRA_TRAIN_PATH)
TRAIN_DF.head()

# # Load datasets (on Colab)
# TRAIN_DF = pd.read_csv(os.path.join(playground_series_s5e9_path, 'train.csv'),index_col = 'id')
# TEST_DF = pd.read_csv(os.path.join(playground_series_s5e9_path, 'test.csv'),index_col = 'id')
# TRAIN_EXTRA = pd.read_csv(os.path.join(playground_series_s5e9_path_extra, '.csv'))


TRAIN_EXTRA.var()


# Check shapes of all 4 datasets
datasets = {
    'TRAIN_DF'   : TRAIN_DF,
    'TEST_DF'    : TEST_DF,
    'TRAIN_EXTRA': TRAIN_EXTRA
    }

# Check shapes
print_with_sep("Shapes")
for name, df in datasets.items():
  print(f"{name} shape: {df.shape}")

# Check duplicates
print_with_sep("Duplicates")
for name, df in datasets.items():
  print(f"{name} duplicates: {df.duplicated().sum()}")

# Check nans
print_with_sep("NaNs")
for name, df in datasets.items():
  print(f"{name} NaNs: {df.isnull().sum().sum()}")

# Check col difference
print_with_sep("Columns not in test")
print(set(TRAIN_DF.columns).difference(set(TEST_DF.columns)))

print_with_sep("Columns not in extra")
print(f'EXTRA df has same n features as TRAIN: {TRAIN_DF.shape[1] == TRAIN_EXTRA.shape[1]}')
print(set(TRAIN_DF.columns).difference(set(TRAIN_EXTRA.columns)))


# Check descriptive stats
print_with_sep("Descriptive Statistics")
for name, df in datasets.items():
  print(f"{name} Description:")
  percentage_missing = df.isnull().sum()/df.shape[0]; percentage_missing.name = '% Missing'
  data_types = df.dtypes; data_types.name = 'd_type'

  display(
      pd.concat([
          df.describe(include='all').T,
          percentage_missing,
          data_types],
                axis=1).replace(np.nan,'-').style.background_gradient(cmap='Blues'))
  print("\n")


# Create temporary dataset for analysis
temp_df = pd.concat([TRAIN_DF, TRAIN_EXTRA], axis=0)
temp_df['dataset'] = len(TRAIN_DF)*['ORIGINAL'] + len(TRAIN_EXTRA)*['EXTRA']

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=temp_df, x=CFG.TARGET, y='dataset', ax=axes[0], palette='viridis')
sns.histplot(data=temp_df, x=CFG.TARGET, hue='dataset', ax=axes[1], palette='viridis', 
             stat='density', common_norm=False, multiple='dodge') # for better visibility, due to different dataset sizes
axes[0].set_title(f'Box plot of {CFG.TARGET.capitalize()} (Target Variable in {name})')
axes[1].set_title(f'Histogram of {CFG.TARGET.capitalize()} (Target Variable in {name})')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Delete temporaty df
del temp_df


# Perform univariate analysis for each numerical variable
numeric_univariate_plots(TRAIN_DF, TEST_DF, TRAIN_EXTRA, CFG.TARGET)


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_DF.corr().round(4),
            annot=True,
            cmap='viridis',
            linewidth = 2
           ); plt.show(); plt.tight_layout()


# Import PCA
from sklearn.decomposition import PCA

# Import UMAP
from umap import UMAP

# Scaler 
from sklearn.preprocessing import StandardScaler


# Define combined df
cluster_df = pd.concat([
    TRAIN_DF, TRAIN_EXTRA
], axis = 0, ignore_index = True)

# Define dataset feature
dataset_dict = {
    'TRAIN' : TRAIN_DF,
    'EXTRA' : TRAIN_EXTRA,
}

if CFG.TASK_TYPE == 'classification':
    # Add cluster feature
    cluster_df['cluster'] = (cluster_df[CFG.TARGET].astype(str) + '_' + cluster_df['dataset'])
    
    # Encode with dummies (no sklearn pipeline needed here)
    cluster_df = pd.concat([
        pd.get_dummies(cluster_df,dtype=int),
        cluster_df],
             axis=1)
else:
    # Add cluster feature
    dataset_indicator = []
    for name, data in dataset_dict.items():
        dataset_indicator.extend([name]*data.shape[0])
    cluster_df['cluster'] = dataset_indicator


from sklearn.preprocessing import StandardScaler

# Row mask
TRAIN_MASK = cluster_df['cluster'] == 'TRAIN'
EXTRA_MASK = cluster_df['cluster'] == 'EXTRA'

# Identify categorical columns excluding the target
scaler = StandardScaler()
cluster_df.loc[TRAIN_MASK].iloc[:,:-1] = scaler.fit_transform(
    cluster_df.loc[TRAIN_MASK].iloc[:,:-1])
cluster_df.loc[EXTRA_MASK].iloc[:,:-1] = scaler.transform(
    cluster_df.loc[EXTRA_MASK].iloc[:,:-1])

cluster_df


# Downsample for faster computation
downsampled_cluster_df = pd.DataFrame()
for dataset in cluster_df['cluster'].unique():
    data = cluster_df.query(f"`cluster` == {str([dataset])}").sample(TRAIN_EXTRA.shape[0], random_state = CFG.SEED)
    downsampled_cluster_df = pd.concat([downsampled_cluster_df,data], axis = 0, ignore_index = True)

downsampled_cluster_df.shape


import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Initialize PCA and set n_components
pca = PCA(n_components=2)

# Fit on full TRAIN data
pca.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-1])

# Transform TRAIN and EXTRA data
X_pca = pca.transform(downsampled_cluster_df.iloc[:,:-1])

# Convert to dataframe
pca_df = pd.DataFrame(X_pca, columns = [
    'PC1', 
    'PC2',
])

# Combine PCA components and cluster labels into one DataFrame
pca_df['cluster'] = downsampled_cluster_df.cluster

# Plot with plotly.express 
fig = px.scatter(
    pca_df.sample(frac=.1, random_state=CFG.SEED),
    x       = 'PC1',
    y       = 'PC2',
    color   = 'cluster',
    title   = '2D PCA',
    opacity = 0.5,
); 

fig.show(renderer='iframe_connected')


# Initialize PCA and set n_components
pca = PCA(n_components=3)

# Fit on full TRAIN data
pca.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-1])

# Transform TRAIN and EXTRA data
X_pca = pca.transform(downsampled_cluster_df.iloc[:,:-1])

# Convert to dataframe
pca_df = pd.DataFrame(X_pca, columns = [
    'PC1', 
    'PC2', 
    'PC3'
])

# Combine PCA components and cluster labels into one DataFrame
pca_df['cluster'] = downsampled_cluster_df.cluster

# Plot with plotly.express 
fig = px.scatter_3d(
    pca_df.sample(frac=.1, random_state=CFG.SEED),
    x       = 'PC1',
    y       = 'PC2',
    z       = 'PC3',
    color   = 'cluster',
    title   = '3D PCA',
    opacity = 0.5,
); 

fig.show(renderer='iframe_connected')


# Initialize UMAP and set n_components
umap = UMAP(n_components=2)

# Fit on TRAIN data
umap.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-1].sample(frac=0.2, 
                                                       # random_state=CFG.SEED 
                                                      ))

# Transform TRAIN and EXTRA data
X_umap = umap.transform(downsampled_cluster_df.iloc[:,:-1])

# to DataFrame
umap_df = pd.DataFrame(data = X_umap,
                      columns = [
                          'UMAP_component1', 
                          'UMAP_component2',
                      ])

# Combine UMAP components and cluster labels into one DataFrame
umap_df['cluster'] = downsampled_cluster_df.cluster

# Plot with plotly.express 
fig = px.scatter(
    umap_df.sample(frac=.1, random_state=CFG.SEED),
    x       = 'UMAP_component1',
    y       = 'UMAP_component2',
    color   = 'cluster',
    title   = '2D UMAP',
    opacity = 0.5,
)

fig.show(renderer='iframe_connected')


# Initialize UMAP and set n_components
umap = UMAP(n_components=3)

# Fit on TRAIN data
umap.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-1].sample(frac=0.2, 
                                                       # random_state=CFG.SEED
                                                      ))

# Transform TRAIN and EXTRA data
X_umap = umap.transform(downsampled_cluster_df.iloc[:,:-1])

# to DataFrame
umap_df = pd.DataFrame(data = X_umap,
                      columns = [
                          'UMAP_component1', 
                          'UMAP_component2', 
                          'UMAP_component3'
                      ])

# Combine UMAP components and cluster labels into one DataFrame
umap_df['cluster'] = downsampled_cluster_df.cluster

# Plot with plotly.express 
fig = px.scatter_3d(
    umap_df.sample(frac=.1, random_state=CFG.SEED),
    x       = 'UMAP_component1',
    y       = 'UMAP_component2',
    z       = 'UMAP_component3',
    color   = 'cluster',
    title   = '3D UMAP',
    opacity = 0.33,
)

fig.show(renderer='iframe_connected')


import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.model_selection import cross_validate, KFold

# Make copies for AV
av_train = TRAIN_DF.copy()
av_extra = TRAIN_EXTRA.copy()

# Drop the target variable
av_train.drop(CFG.TARGET, axis = 1, inplace = True)
av_extra.drop(CFG.TARGET, axis = 1, inplace = True)

# Add the trextratest labels
av_train['AV_label'] = 1
av_extra['AV_label'] = 0

# Concatenate, shuffle, and split train and valid
merged_df = pd.concat([av_train, av_extra], axis = 0, ignore_index = True)
merged_df = merged_df.sample(frac = 1, random_state = CFG.SEED)

# Define estimator and cv
if CFG.TASK_TYPE == 'classifier':
    estimator = LGBMClassifier(objective=CFG.TASK_TYPE)
else:
    estimator = LGBMRegressor(objective=CFG.TASK_TYPE)

kf = KFold(n_splits = CFG.FOLDS , random_state = CFG.SEED, shuffle = True)

# Define X and y
X = merged_df.copy()
y = X.pop('AV_label')

# Cross validate
cv_data = pd.DataFrame(
    cross_validate(estimator,
                   X = X,
                   y = y,
                   cv = kf, 
                   scoring = 'roc_auc',
                   fit_params = {
                       "eval_metric": "auc",
                       # "callbacks": [early_stopping(stopping_rounds=100, verbose=200)],
                   }
                  ))

# Display results
display(cv_data)
print(f"Mean cv: {np.mean(cv_data['test_score'])}")


# https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(8, 6))

fprs = []
tprs = []
aucs = []

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    estimator.fit(X_train, y_train)
    y_hat = estimator.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)
    
    tprs.append(tpr)
    fprs.append(fpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {i+1} ROC curve (AUC = {roc_auc:.2f})')

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Chance')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curves from Adversarial Validation')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


estimator.fit(X, y)
fig, ax = plt.subplots(figsize=(12,4))
lgb.plot_importance(estimator, ax=ax)
plt.show();



# Merge Train and Extra
if CFG.ADD_EXTRA_DATA:
    TRAIN_DF = pd.concat([
        TRAIN_DF, TRAIN_EXTRA
    ], axis = 0, ignore_index = True)


%%capture
from tqdm import tqdm
from itertools import combinations

# Make small samples-copies for testing purposes
train_sample = TRAIN_DF.iloc[:10].copy()
test_sample = TEST_DF.iloc[:10].copy()

if CFG.ADD_FEATURES:
    # Select columns to encode
    columns_to_encode = TRAIN_DF.columns.difference([CFG.TARGET])
    
    # ==========================================================================
    # INTERACTIONS
    if CFG.ADD_INTERACTIONS:
        
        # Decide combo size
        combo_size = [2, 3, 4]
        
        # Make combos
        for n in combo_size[:1]: # use just a dual combo for now
            combinations_list = list(combinations(columns_to_encode, n))
            for cols in tqdm(combinations_list):
              new_col_name = '_x_'.join(cols)  # Feature name
        
              # Calculate the interaction term (product)
              TRAIN_DF[new_col_name] = TRAIN_DF[list(cols)].prod(axis=1)
              TEST_DF[new_col_name] = TEST_DF[list(cols)].prod(axis=1)
    
    # ==========================================================================
    # RATIOS
    if CFG.ADD_RATIOS:
        combinations_list = list(combinations(columns_to_encode, 2)) # ratios are done in pairs
        for cols in tqdm(combinations_list):
          col1, col2 = cols # Get the two columns for the ratio
        
          # Create the ratio features
          new_col_name_ratio1 = f'{col1}_/_{col2}'
        
          # Add a small epsilon to the denominator to avoid division by zero
          Îµ = 1e-5
        
          # Calculate the ratio term
          TRAIN_DF[new_col_name_ratio1] = TRAIN_DF[col1] / (TRAIN_DF[col2] + Îµ)
          TEST_DF[new_col_name_ratio1] = TEST_DF[col1] / (TEST_DF[col2] + Îµ)


import time
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

import lightgbm as lgbm
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# Define optimization function
def objective(trial, X, y, model_type):
    if model_type == "xgboost":
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
            # 'use_label_encoder': False,
            # 'eval_metric': 'logloss',
            'early_stopping_rounds' : 100,
            'device': 'cuda',
            'random_state': CFG.SEED,
        }
        model = XGBRegressor(**params)

    elif model_type == "lightgbm":
        params = {
            'n_estimators':      trial.suggest_int("n_estimators", 200, 1500),
            'learning_rate':     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'num_leaves':        trial.suggest_int("num_leaves", 15, 150),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
            'subsample':         trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'reg_alpha':         trial.suggest_float("reg_alpha", 0.0, 10.0),
            'reg_lambda':        trial.suggest_float("reg_lambda", 0.0, 10.0),
            'verbose':-1,
            'device': 'gpu',
            'random_state': CFG.SEED
        }
        model = LGBMRegressor(**params)

    elif model_type == "catboost":
        params = {
            'iterations':        trial.suggest_int("iterations", 200, 1000),
            'learning_rate':     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'depth':             trial.suggest_int("depth", 4, 10),
            'l2_leaf_reg':       trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            'random_state': CFG.SEED,
            'verbose': 0,
            # 'task_type': "GPU",
            'allow_writing_files': False
        }
        model = CatBoostRegressor(**params)

    elif model_type == "histgb":
        params = {
            'max_iter':          trial.suggest_int("max_iter", 200, 1500),
            'learning_rate':     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_leaf_nodes':    trial.suggest_int("max_leaf_nodes", 15, 255),
            'min_samples_leaf':  trial.suggest_int("min_samples_leaf", 10, 100),
            'l2_regularization': trial.suggest_float("l2_regularization", 0.0, 10.0),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'early_stopping': True,
            'n_iter_no_change': 100,
            'validation_fraction': 0.1,
            'verbose': 0,
            'random_state': CFG.SEED
        }
        model = HistGradientBoostingRegressor(**params)

    else:
        print(f"Error: Unsupported model_type received: {model_type}")
        raise ValueError("Unsupported model_type")

    # Initialize kf
    kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    
    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y.values)):
        print(f"\n{'#'*10} Fold {fold+1}/{CFG.FOLDS} {'#'*10}")
    
        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes('number').columns.difference([CFG.TARGET]))
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        start = time.time()

        # XGBoost, LightGBM, and CatBoost support early stopping natively via fit() arguments.
        # HistGradientBoostingRegressor from sklearn (as of now) does not have native early stopping.

        if model_type == "xgboost":
            model.fit(x_train, y_train,
                      eval_set=[(x_train, y_train),(x_valid, y_valid)],
                      verbose=False,
                     )
        
        if model_type == "lightgbm":
            model.fit(x_train, y_train,
                      eval_set=[(x_train, y_train),(x_valid, y_valid)],
                      callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)],
                    )

        if model_type == "catboost":
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid),
                      verbose=False,
                     )

        elif model_type == "histgb":
            model.fit(x_train, y_train)

        # else:
        #   raise ValueError("Unsupported model_type")

        preds = model.predict(x_valid); # print(f'DEBUG TEST: {preds}')
        score = mean_squared_error(y_valid, preds, squared=False)
        scores.append(score)

        # Fold score
        print(f"RMSE: {score:.4f} | Time: {time.time() - start:.2f}s")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Checkpoint description
    print(f"Mean k-FOLDS score: {np.mean(scores)} +- {np.std(scores)}")

    return np.mean(scores)


# Define X and y
X = TRAIN_DF.copy()
y = X.pop(CFG.TARGET)
X_test = TEST_DF.copy()

if CFG.OPTUNA_SEARCH: 
    # Optimize with Optuna
    study_results = dict()
    for model_type in ["xgboost", "lightgbm", "catboost", "histgb"]:
      study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
      study.optimize(lambda trial: objective(trial, X=X, y=y, model_type=model_type),
                    n_trials=CFG.OPTUNA_N_TRIALS, timeout=3600)
    
      # Nest params, value and trial within each model's dictionary
      model_results = dict()
      model_results["Model"] = model_type
      model_results["Best params:"] = study.best_params
      model_results["Best value:"] = study.best_value
      model_results["Best trial:"] = study.best_trial
    
      # Append to study results
      study_results[model_type] = model_results
    
    # Display the results
    study_df = pd.DataFrame.from_records(study_results)
    for model_type in ["xgboost", "lightgbm", "catboost", "histgb"]:
      display(pd.DataFrame.from_records(study_results[model_type]))


XGB_PARAMS = {
    'n_estimators': 381, 
    'learning_rate': 0.005292705365436975, 
    'max_depth': 6, 
    'subsample': 0.728034992108518, 
    'colsample_bytree': 0.8925879806965068, 
    'reg_alpha': 9.925166969962311e-08, 
    'reg_lambda': 0.00013878559259972322,

    'eval_metric': 'logloss',
    'early_stopping_rounds' : 100,
    'device': 'cuda',
    'random_state': CFG.SEED,
    }

LGBM_PARAMS = {
    'n_estimators': 470, 
    'learning_rate': 0.015197021147500827, 
    'num_leaves': 38, 
    'min_child_samples': 96, 
    'subsample': 0.5962024646862834, 
    'colsample_bytree': 0.8344528550358621, 
    'reg_alpha': 3.9921404016484106, 
    'reg_lambda': 6.195339431335992,

    'verbose':-1,
    'device': 'gpu',
    'random_state': CFG.SEED
    }

CAT_PARAMS = {
    'iterations': 681, 
    'learning_rate': 0.05675206026988748, 
    'depth': 4, 
    'l2_leaf_reg': 9.72918866945795,

    'random_state': CFG.SEED,
    'verbose': 0,
    # 'task_type': "GPU",
    'allow_writing_files': False
    } # stopped after 1 trial, baseline

HISTGB_PARAMS = {
    'max_iter': 1283, 
    'learning_rate': 0.0033572967053517922, 
    'max_leaf_nodes': 58, 
    'min_samples_leaf': 26, 
    'l2_regularization': 3.0424224295953772, 
    'max_depth': 8,

    'early_stopping': True,
    'n_iter_no_change': 100,
    'validation_fraction': 0.1,
    'verbose': 0,
    'random_state': CFG.SEED
    } # stopped after 1 trial, baseline


# Define estimators group (with best hyperparams)
estimators = [
    ('xgboost',  XGBRegressor(**XGB_PARAMS)),
    ('lightgbm', LGBMRegressor(**LGBM_PARAMS)),
    ('catboost', CatBoostRegressor(**CAT_PARAMS)),
    ('histgb',   HistGradientBoostingRegressor(**HISTGB_PARAMS))
    ]

# Initialize dictionaries to keep predicted probs from each model
OOF_PREDS = dict()
TEST_PREDS = dict()

# Start model loop
for model_type, model in estimators:
    
    # Initialize kf
    kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    
    # Define empty oof variables to fill
    oof_preds = np.zeros(shape = (len(X)))
    test_preds = np.zeros(shape = (len(X_test),)) # Shape as 1d array rather than using y.nunique() because I only need class_1 probs
    fold_scores = []
        
    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y.values)):
        print(f"\n{'#'*10} Fold {fold+1}/{CFG.FOLDS} {'#'*10}")
    
        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes('number').columns.difference([CFG.TARGET]))
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        x_test_loop = X_test.copy()

        start = time.time()
        
        if model_type == "xgboost":
            model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    verbose=False,
                    )
    
        if model_type == "lightgbm":
            model.fit(x_train, y_train,
                        eval_set=[(x_train, y_train),(x_valid, y_valid)],
                        callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=-1)],
                            )
        
        if model_type == "catboost":
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid),
                      verbose=False,
                        )

        elif model_type == "histgb":
            model.fit(x_train, y_train,
                        )

        # Get predictions and Predict OOF and test
        oof_preds[valid_idx] = model.predict(x_valid)
        test_preds += model.predict(x_test_loop)
        
        preds = model.predict(x_valid); # print(f'DEBUG TEST PREDS: {preds}')
        fold_score = mean_squared_error(y_valid, preds, squared=False)
        fold_scores.append(fold_score)
        print(f" Fold {fold+1}: RMSE Score: {fold_score:.5f}")
        
        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")
    
    mean_valid_score = np.mean(fold_scores); print(f"Mean RMSE: {mean_valid_score:.3f}")
    test_predictions = test_preds / CFG.FOLDS
    
    # Save OOF and test predictions by model
    OOF_PREDS[model_type] = oof_preds
    TEST_PREDS[model_type] = test_predictions


def objective(trial):

  # Sample model weights and normalize
  w1 = trial.suggest_float('w1', 0, 1)
  w2 = trial.suggest_float('w2', 0, 1)
  w3 = trial.suggest_float('w3', 0, 1)
  w4 = 1 - (w1 + w2 + w3) # Constraint: weights must sum to 1

  # Skip invalid combinations
  if w4 < 0 or w4 > 1:
    raise optuna.exceptions.TrialPruned()

  # Weighted ensemble of out-of-fold probabilities
  ensemble_preds = (
      w1 * OOF_PREDS['xgboost'] +
      w2 * OOF_PREDS['lightgbm'] +
      w3 * OOF_PREDS['catboost'] +
      w4 * OOF_PREDS['histgb']
  )

  score = mean_squared_error(y, ensemble_preds, squared=False)

  return score


# Optimize model weights
if CFG.SEARCH_WEIGHTS:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    study.optimize(objective, n_trials=500, timeout=3600)


# Create Submission File
# Define best weights and threshold
w1 = 0.00014777071043482596 
w2 = 0.7407971912371842 
w3 = 0.21468014719386622
w4 = 1 - (w1 + w2 + w3)

# Weighted ensemble of model probabilities with weights
ensemble_probs = (
    w1 * TEST_PREDS['xgboost'] +
    w2 * TEST_PREDS['lightgbm'] +
    w3 * TEST_PREDS['catboost'] +
    w4 * TEST_PREDS['histgb']
)

submission_df = pd.DataFrame({
    'id': list(X_test.index),
    'y': ensemble_probs
})

# Display the first 5 rows
display(submission_df.head())

# Plot preds distribution
sns.histplot(submission_df['y'])
plt.show()

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

