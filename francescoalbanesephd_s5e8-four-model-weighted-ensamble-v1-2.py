%%capture
!pip install itables
!pip install optuna-integration[xgboost]==4.3.0
!pip install catboost==1.2.8
!pip install scikit-learn==1.3.1 # for target_encoder
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
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU setups.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #May slightly slow down training, but ensures reproducibility

# =========================================================================
# Set plot style
sns.set_style('whitegrid')

# # Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os

# Load datasets (on Kaggle)
TRAIN_DF = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col = 'id')
TEST_DF = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col = 'id')
TRAIN_EXTRA = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')

# # Load datasets (on Colab)
# TRAIN_DF = pd.read_csv(os.path.join(playground_series_s5e8_path, 'train.csv'),index_col = 'id')
# TEST_DF = pd.read_csv(os.path.join(playground_series_s5e8_path, 'test.csv'),index_col = 'id')
# # TRAIN_EXTRA = pd.read_csv(os.path.join(playground_series_s5e8_path_extra, '.csv'))


# Print helper function
def print_with_sep(text,sep="=",n=30):
  print("\n")
  print(sep*n)
  print('\t',text)
  print(sep*n)

# Check shapes of all 4 datasets
datasets = {
    'TRAIN_DF': TRAIN_DF,
    'TEST_DF': TEST_DF,
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


# Identify target
target = 'y'

# Make target categorical
TRAIN_DF[target] = TRAIN_DF[target].astype('category')
TRAIN_EXTRA[target] = (TRAIN_EXTRA[target] == 'yes').astype(int)
TRAIN_EXTRA[target] = TRAIN_DF[target].astype('category')

# Distribution of the target variable
plt.figure(figsize=(12, 5))
sns.countplot(data=TRAIN_DF, y=target, order = TRAIN_DF[target].value_counts().index, palette='viridis')
plt.title(f'Distribution of {target.capitalize()} (Target Variable)')
plt.xlabel('Count')
plt.ylabel(f'{target.capitalize()}')
plt.show()


# Relationship between numerical features and the target variable (using boxplots)

# Get the numerical features excluding the target and the index
numerical_features = TRAIN_DF.select_dtypes(include=np.number).columns.difference([target])

# Set up the subplot grid
fig, axes = plt.subplots(3, 3, figsize=(18, 3 * 6))
axes = axes.flatten()

# Iterate through the numerical features and create boxplots
for i, feature in enumerate(numerical_features):
  sns.boxplot(y=target, x=feature, data=TRAIN_DF, ax=axes[i], hue = target, palette='viridis')
  axes[i].set_title(f'Target vs {feature}')
  axes[i].set_ylabel('Target')
  axes[i].set_xlabel(feature)

# Remove blank plots
for i in range(len(numerical_features), len(axes)):
  fig.delaxes(axes[i])

# Plot
plt.tight_layout()
plt.show()


# Analysis of all NUMERIC features
# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Function to create and display plots for a single numerical variable
def create_variable_plots(train, test, extra, variable):

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train.copy()
    test_temp = test.copy()
    extra_temp = extra.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    extra_temp["Dataset"] = "Extra"
    combined_data = pd.concat([train_temp, test_temp, extra_temp])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

    # Box plot
    sns.boxplot(data=combined_data, x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}")

    # Histogram
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    sns.histplot(data=extra, x=variable, color=custom_palette[0], kde=True, bins=30, label="Extra", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()
    axes[1].annotate(f"Skewness (TRAIN): {train[variable].skew():.2f}\nKurtosis (TRAIN): {train[variable].kurt():.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])


    # Adjust spacing and show
    plt.tight_layout()
    plt.show()


# Perform univariate analysis for each numerical variable
for variable in TRAIN_DF.select_dtypes(include='number'):
    create_variable_plots(TRAIN_DF, TEST_DF, TRAIN_EXTRA, variable)


# Define features to investigate
cat_cols = TRAIN_DF.select_dtypes(exclude='number').columns.difference([target])

# Visualise categorical variables
fig, axes = plt.subplots(3,3,figsize=(15, 10))
ax = axes.flatten()

for i, col in enumerate(TRAIN_DF[cat_cols]):
    sns.countplot(data=TRAIN_DF, y=col, order = TRAIN_DF[col].value_counts().index, palette='viridis', ax=ax[i])

plt.tight_layout()


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_DF.select_dtypes(include='number').corr().round(4),
            annot=True,
            cmap='viridis',
            linewidth = 2
           ); plt.show(); plt.tight_layout()


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

# Cap target outliers in train and validation sets
# TRAIN_capped, _, TEST_capped = iqr_outlier_capping(TRAIN_DF.dropna(), None, TEST_DF, columns=TRAIN_DF.select_dtypes('number').columns.difference([target]))


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

dataset_indicator = []
for name, data in dataset_dict.items():
    dataset_indicator.extend([name]*data.shape[0])

# Add dataset feature
cluster_df['dataset'] = dataset_indicator

# Add cluster feature
cluster_df['cluster'] = (cluster_df['y'].astype(str) + '_' + cluster_df['dataset'])

# Encode with dummies (no sklearn pipeline needed here)
cluster_df = pd.concat([
    pd.get_dummies(cluster_df.iloc[:,:-3],dtype=int),
    cluster_df.iloc[:,-3:]],
         axis=1)


from sklearn.preprocessing import StandardScaler

# Feature groups
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Row mask
TRAIN_MASK = cluster_df['dataset'] == 'TRAIN'
EXTRA_MASK = cluster_df['dataset'] == 'EXTRA'

# Identify categorical columns excluding the target
scaler = StandardScaler()
cluster_df.loc[TRAIN_MASK, numeric_features] = scaler.fit_transform(
    cluster_df.loc[TRAIN_MASK, numeric_features])
cluster_df.loc[EXTRA_MASK, numeric_features] = scaler.transform(
    cluster_df.loc[EXTRA_MASK, numeric_features])

cluster_df


# Downsample for faster computation
downsampled_cluster_df = pd.DataFrame()
for dataset in cluster_df['dataset'].unique():
    data = cluster_df.query(f"`dataset` == {str([dataset])}").sample(45_000, random_state = SEED)
    downsampled_cluster_df = pd.concat([downsampled_cluster_df,data], axis = 0, ignore_index = True)

downsampled_cluster_df.shape


import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Initialize PCA and set n_components
pca = PCA(n_components=2)

# Fit on full TRAIN data
pca.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-3])

# Transform TRAIN and EXTRA data
X_pca = pca.transform(downsampled_cluster_df.iloc[:,:-3])

# Convert to dataframe
pca_df = pd.DataFrame(X_pca, columns = [
    'PC1', 
    'PC2',
])

# Combine PCA components and cluster labels into one DataFrame
pca_df['cluster'] = downsampled_cluster_df.cluster

# Plot with plotly.express 
fig = px.scatter(
    pca_df.sample(frac=.1, random_state=SEED),
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
pca.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-3])

# Transform TRAIN and EXTRA data
X_pca = pca.transform(downsampled_cluster_df.iloc[:,:-3])

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
    pca_df.sample(frac=.1, random_state=SEED),
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
umap.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-3].sample(frac=0.2, random_state=SEED))

# Transform TRAIN and EXTRA data
X_umap = umap.transform(downsampled_cluster_df.iloc[:,:-3])

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
    umap_df.sample(frac=.1, random_state=SEED),
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
umap.fit(cluster_df.loc[TRAIN_MASK].iloc[:,:-3].sample(frac=0.2, random_state=SEED))

# Transform TRAIN and EXTRA data
X_umap = umap.transform(downsampled_cluster_df.iloc[:,:-3])

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
    umap_df.sample(frac=.1, random_state=SEED),
    x       = 'UMAP_component1',
    y       = 'UMAP_component2',
    z       = 'UMAP_component3',
    color   = 'cluster',
    title   = '3D UMAP',
    opacity = 0.33,
)

fig.show(renderer='iframe_connected')


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold

# Make copies for AV
av_train = TRAIN_DF.copy()
av_test = TEST_DF.copy()

# Identify categorical columns excluding the target
categorical_cols = av_train.select_dtypes(include='object').columns.tolist()

# Feature Encoding
oe = OrdinalEncoder()
av_train[cat_cols] = oe.fit_transform(av_train[cat_cols])
av_test[cat_cols] = oe.transform(av_test[cat_cols])

# Drop the target variable
av_train.drop('y', axis = 1, inplace = True)

# Add the train/test labels
av_train['AV_label'] = 1
av_test['AV_label'] = 0

# Concatenate and shuffle
merged_df = pd.concat([av_train, av_test], axis = 0, ignore_index = True)
merged_df = merged_df.sample(frac = 1, random_state = SEED)

# Define estimator and cv
estimator = XGBClassifier(objective='binary:logistic')
skf = StratifiedKFold(n_splits = 5, random_state = SEED, shuffle = True)

# create our DMatrix (the XGBoost data structure)
X = merged_df.drop(['AV_label'], axis=1)
y = merged_df['AV_label']

# Cross validate
pd.DataFrame(
    cross_validate(estimator, 
                   X = X,
                   y = y,
                   cv = skf, 
                   scoring = 'roc_auc')
)


# https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
from sklearn.metrics import roc_curve, auc

skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
estimator = XGBClassifier(objective='binary:logistic')

plt.figure(figsize=(8, 6))

fprs = []
tprs = []
aucs = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    estimator.fit(X_train, y_train)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    tprs.append(tpr)
    fprs.append(fpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {i+1} ROC curve (AUC = {roc_auc:.2f})')

# # Plot mean ROC curve
# mean_tpr = [x.mean() for x in tprs]
# mean_fpr = [x.mean() for x in fprs]

# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=f'Mean ROC curve (AUC = {mean_auc:.2f})',
#          lw=2, alpha=0.8)

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Chance')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curves from Adversarial Validation')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



MERGED_TRAIN_DF = pd.concat([
    TRAIN_DF, TRAIN_EXTRA
], axis = 0, ignore_index = True)


def circular_features_time(df):
    df = df.copy()

    # Map month
    month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month'] = df['month'].map(month_map) 

    # # Note: Sin-cos transformations are not strictly needed with tree based models
    # df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
    # df['day_of_month_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    # df['day_of_month_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Optional
    # df = df.drop(['month','day'], axis = 1)
    
    return df


def preprocessing(df, train_data = None):
    df = df.copy()

    # Apply function for time features
    df = circular_features_time(df)

    # Loans
    df[['housing','loan']] = (df[['housing','loan']] == 'yes').astype(int)
    df['n_loans'] = df[['housing','loan']].sum(axis=1)

    # Contact features
    df['previously_contacted'] = (df['pdays'] >= 0).astype(int)
    df['total_contacts'] = df['campaign'] + df['previous'] 

    # Age group
    df['age_groups'] = pd.cut(df['age'], bins=5, labels=False)

    # Education-Job
    df['education_job'] = df['education'] + '_' + df['job']

    # Contact-Job
    df['contact_job'] = df['contact'] + '_' + df['job']

    # Mapping educatin
    education_map = {'unknown'  : 0, 
                     'primary'  : 1, 
                     'secondary': 2, 
                     'tertiary' : 3}
    
    # df['education'] = df['education'].map(education_map) # Commented out to try target encoding

    # Balance features
    #---positive balance
    df['negative_balance'] = (df['balance'] < 0).astype(int)
    #---binning (accounting for negative-only bin)
    if train_data is not None:
        s = train_data['balance']
        auto_bins = np.linspace(s.min(), s.max(), 10)
        if 0 not in auto_bins:
            auto_bins = np.sort(np.append(auto_bins, 0))
        df['balance_groups'] = pd.cut(df['balance'], bins=auto_bins, labels=False, include_lowest=True)
    else:
        df['balance_groups'] = pd.cut(df['balance'], bins=10, labels=False, include_lowest=True)

    # Duration
    df['duration_minute'] = df['duration'] / 60

    # Dummify default
    df['default'] = (df['default'] == 'yes').astype(int)
    
    # Previous campaign's success
    # df['psuccess'] = (df['poutcome'] == 'success').astype(int)

    # Make target numeric
    if target in df.columns:
        df[target] = (df[target] == 1).astype(int)
    
    return df


# Process dfs
PROCESSED_TRAIN_DF = preprocessing(MERGED_TRAIN_DF, train_data = MERGED_TRAIN_DF)
PROCESSED_TEST_DF = preprocessing(TEST_DF, train_data = MERGED_TRAIN_DF)

show(PROCESSED_TRAIN_DF)


# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder,TargetEncoder

# # Identify categorical columns excluding the target
# categorical_cols = PROCESSED_TRAIN_DF.select_dtypes(include='object').columns.difference([target]).tolist()

# # Feature Encoding
# oe = OrdinalEncoder()
# PROCESSED_TRAIN_DF[categorical_cols] = oe.fit_transform(PROCESSED_TRAIN_DF[categorical_cols])
# PROCESSED_TEST_DF[categorical_cols] = oe.transform(PROCESSED_TEST_DF[categorical_cols])

# # Label encode the target variable
# label_encoder = LabelEncoder()
# PROCESSED_TRAIN_DF[target] = label_encoder.fit_transform(PROCESSED_TRAIN_DF[target])

# print("Categorical features encoded.")
# print("Encoded PROCESSED_TRAIN_DF head:")
# display(PROCESSED_TRAIN_DF.head())
# print("\nEncoded PROCESSED_TEST_DF head:")
# display(PROCESSED_TEST_DF.head())


import time
import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
# from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


FOLDS = 5

def objective(trial, X, y, model_type):
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
            # 'use_label_encoder': False,
            'eval_metric': 'logloss',
            'early_stopping_rounds' : 100,
            'device': 'cuda',
            'random_state': SEED,
        }
        model = XGBClassifier(**params)

    elif model_type == "lightgbm":
        import lightgbm as lgbm
        from lightgbm import LGBMClassifier
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 200, 1500),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 15, 150),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 10.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 10.0),
            'verbose':-1,
            'device': 'gpu',
            'random_state': SEED
        }
        model = LGBMClassifier(**params)

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        params = {
            'iterations': trial.suggest_int("iterations", 200, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'depth': trial.suggest_int("depth", 4, 10),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            'random_state': SEED,
            'verbose': 0,
            # 'task_type': "GPU",
            'allow_writing_files': False
        }
        model = CatBoostClassifier(**params)

    elif model_type == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = {
            'max_iter': trial.suggest_int("max_iter", 200, 1500),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 15, 255),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 10, 100),
            'l2_regularization': trial.suggest_float("l2_regularization", 0.0, 10.0),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'early_stopping': True,
            'n_iter_no_change': 100,
            'validation_fraction': 0.1,
            'verbose': 0,
            'random_state': SEED
        }
        model = HistGradientBoostingClassifier(**params)

    else:
        print(f"Error: Unsupported model_type received: {model_type}")
        raise ValueError("Unsupported model_type")

    # Handle class imbalance
    # imbalance_strategy = trial.suggest_categorical("imbalance_strategy", ["weights", "SMOTE"]) # !! temporarily disabled smote option, too demanding for this dataset
    imbalance_strategy = 'weights'
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
        print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        start = time.time()

        # XGBoost, LightGBM, and CatBoost support early stopping natively via fit() arguments.
        # HistGradientBoostingClassifier from sklearn (as of now) does not have native early stopping.

        use_weights = imbalance_strategy == "weights"
        use_smote = imbalance_strategy == "SMOTE"
        
        if use_smote:
          # SMOTE needs a df with NO nans
          imputer = SimpleImputer(strategy='median')
          x_imputed = imputer.fit_transform(x_train)
          x_imputed = pd.DataFrame(x_imputed,columns=x_train.columns)
          smote = SMOTE(random_state=SEED)
          x_train, y_train = smote.fit_resample(x_imputed, y_train)

        if model_type == "xgboost":
          model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    verbose=False,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        if model_type == "lightgbm":
          model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)],
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        if model_type == "catboost":
          model.fit(x_train, y_train,
                    eval_set=(x_valid, y_valid),
                    verbose=False,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        elif model_type == "histgb":
          model.fit(x_train, y_train,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        # else:
        #   raise ValueError("Unsupported model_type")

        proba = model.predict_proba(x_valid)[:, 1]; # print(f'DEBUG TEST PROBA: {proba}')
        score = roc_auc_score(y_valid, proba)
        scores.append(score)

        # Fold score
        print(f"ROC AUC: {score:.4f} | Time: {time.time() - start:.2f}s")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Checkpoint description
    print(f"Mean k-FOLDS score: {np.mean(scores)} +- {np.std(scores)}")

    return np.mean(scores)


FOLDS = 5

def objective(trial, X, y, model_type, ranked_features = None):
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
            # 'use_label_encoder': False,
            'eval_metric': 'logloss',
            'early_stopping_rounds' : 100,
            'device': 'cuda',
            'random_state': SEED,
        }
        model = XGBClassifier(**params)

    elif model_type == "lightgbm":
        import lightgbm as lgbm
        from lightgbm import LGBMClassifier
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 200, 1500),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 15, 150),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 10.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 10.0),
            'verbose':-1,
            'device': 'gpu',
            'random_state': SEED
        }
        model = LGBMClassifier(**params)

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        params = {
            'iterations': trial.suggest_int("iterations", 200, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'depth': trial.suggest_int("depth", 4, 10),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            'random_state': SEED,
            'verbose': 0,
            # 'task_type': "GPU",
            'allow_writing_files': False
        }
        model = CatBoostClassifier(**params)

    elif model_type == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = {
            'max_iter': trial.suggest_int("max_iter", 200, 1500),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 15, 255),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 10, 100),
            'l2_regularization': trial.suggest_float("l2_regularization", 0.0, 10.0),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'early_stopping': True,
            'n_iter_no_change': 100,
            'validation_fraction': 0.1,
            'verbose': 0,
            'random_state': SEED
        }
        model = HistGradientBoostingClassifier(**params)

    else:
        print(f"Error: Unsupported model_type received: {model_type}")
        raise ValueError("Unsupported model_type")

    # Handle class imbalance
    # imbalance_strategy = trial.suggest_categorical("imbalance_strategy", ["weights", "SMOTE"]) # !! temporarily disabled smote option, too demanding for this dataset
    imbalance_strategy = 'weights'

    # Feature selection
    # k_features = trial.suggest_categorical('k_features', [10, 25, 58]) if ranked_features is not None else 58
    # features = list(ranked_features.index[:k_features]) if ranked_features is not None else X.columns

    # Initialize skf
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    # Define categorical cols
    categorical_cols = X.select_dtypes(include='object').columns.difference([target]).tolist()
    
    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
        print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
    
        # Initialize target encoder
        target_enc = TargetEncoder(target_type='binary', smooth='auto', cv=4, shuffle=True, random_state=SEED)

        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes('number').columns.difference([target]))
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Encode
        x_train[categorical_cols] = target_enc.fit_transform(x_train[categorical_cols], y_train)
        x_valid[categorical_cols] = target_enc.transform(x_valid[categorical_cols])
        
        start = time.time()

        # XGBoost, LightGBM, and CatBoost support early stopping natively via fit() arguments.
        # HistGradientBoostingClassifier from sklearn (as of now) does not have native early stopping.

        use_weights = imbalance_strategy == "weights"
        use_smote = imbalance_strategy == "SMOTE"
        
        if use_smote:
          # SMOTE needs a df with NO nans
          imputer = SimpleImputer(strategy='median')
          x_imputed = imputer.fit_transform(x_train)
          x_imputed = pd.DataFrame(x_imputed,columns=x_train.columns)
          smote = SMOTE(random_state=SEED)
          x_train, y_train = smote.fit_resample(x_imputed, y_train)

        if model_type == "xgboost":
          model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    verbose=False,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        if model_type == "lightgbm":
          model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)],
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        if model_type == "catboost":
          model.fit(x_train, y_train,
                    eval_set=(x_valid, y_valid),
                    verbose=False,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        elif model_type == "histgb":
          model.fit(x_train, y_train,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )

        # else:
        #   raise ValueError("Unsupported model_type")

        proba = model.predict_proba(x_valid)[:, 1]; # print(f'DEBUG TEST PROBA: {proba}')
        score = roc_auc_score(y_valid, proba)
        scores.append(score)

        # Fold score
        print(f"ROC AUC: {score:.4f} | Time: {time.time() - start:.2f}s")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Checkpoint description
    print(f"Mean k-FOLDS score: {np.mean(scores)} +- {np.std(scores)}")

    return np.mean(scores)


# # Optimize with Optuna
# study_results = dict()
# for model_type in ["xgboost", "lightgbm", "catboost", "histgb"]:
#   study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
#   study.optimize(lambda trial: objective(trial, X=PROCESSED_TRAIN_DF.drop(columns=[target]), y=PROCESSED_TRAIN_DF[target], model_type=model_type, ranked_features = None),
#                 n_trials=5, timeout=3600)

#   # Nest params, value and trial within each model's dictionary
#   model_results = dict()
#   model_results["Model"] = model_type
#   model_results["Best params:"] = study.best_params
#   model_results["Best value:"] = study.best_value
#   model_results["Best trial:"] = study.best_trial

#   # Append to study results
#   study_results[model_type] = model_results

# # Display the results
# study_df = pd.DataFrame.from_records(study_results)
# for model_type in ["xgboost", "lightgbm", "catboost", "histgb"]:
#   display(pd.DataFrame.from_records(study_results[model_type]))


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
    'random_state': SEED,
    }

LGBM_PARAMS = {
    'n_estimators': 793, 
    'learning_rate': 0.08810003129071789, 
    'num_leaves': 42, 
    'min_child_samples': 56, 
    'subsample': 0.7962072844310213, 
    'colsample_bytree': 0.5232252063599989, 
    'reg_alpha': 6.075448519014383, 
    'reg_lambda': 1.7052412368729153,

    'verbose':-1,
    'device': 'gpu',
    'random_state': SEED
    }

CAT_PARAMS = {
    'iterations': 252, 
    'learning_rate': 0.22413234378101138, 
    'depth': 10, 
    'l2_leaf_reg': 8.275576133048151,

    'random_state': SEED,
    'verbose': 0,
    # 'task_type': "GPU",
    'allow_writing_files': False
    }

HISTGB_PARAMS = {
    'max_iter': 275, 
    'learning_rate': 0.13983740016490973, 
    'max_leaf_nodes': 159, 
    'min_samples_leaf': 74, 
    'l2_regularization': 0.20584494295802447, 
    'max_depth': 12,

    'early_stopping': True,
    'n_iter_no_change': 100,
    'validation_fraction': 0.1,
    'verbose': 0,
    'random_state': SEED
    }


# Define estimators group (with best hyperparams)
estimators = [
    ('xgboost', XGBClassifier(**XGB_PARAMS)),
    ('lightgbm', LGBMClassifier(**LGBM_PARAMS)),
    ('catboost', CatBoostClassifier(**CAT_PARAMS)),
    ('histgb', HistGradientBoostingClassifier(**HISTGB_PARAMS))
    ]

# Define X and y
X = PROCESSED_TRAIN_DF.drop(columns=[target])
y = PROCESSED_TRAIN_DF[target]
X_test = PROCESSED_TEST_DF.copy()

# Initialize dictionaries to keep predicted probs from each model
OOF_probs = dict()
test_probs = dict()

# Start model loop
for model_type, model in estimators:

    # Handle class imbalance
    imbalance_strategy = "weights"
    
    # Initialize skf
    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    # Define empty oof variables to fill
    oof_preds = np.zeros(shape = (len(X),2))
    test_preds = np.zeros(shape = (len(X_test),)) # Shape as 1d array rather than using y.nunique() because I only need class_1 probs
    fold_scores = []
    
    # Define categorical cols
    categorical_cols = X.select_dtypes(include='object').columns.difference([target]).tolist()
    
    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
        print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
    
        # Initialize target encoder
        target_enc = TargetEncoder(target_type='binary', smooth='auto', cv=4, shuffle=True, random_state=SEED)

        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes('number').columns.difference([target]))
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        x_test_loop = X_test.copy()

        # Encode
        x_train[categorical_cols] = target_enc.fit_transform(x_train[categorical_cols], y_train)
        x_valid[categorical_cols] = target_enc.transform(x_valid[categorical_cols])
        x_test_loop[categorical_cols] = target_enc.transform(x_test_loop[categorical_cols])

        start = time.time()
    
        use_weights = imbalance_strategy == "weights"
        use_smote = imbalance_strategy == "SMOTE"
        if use_smote:
            # SMOTE needs a df with NO nans
            imputer = SimpleImputer(strategy='median')
            x_imputed = imputer.fit_transform(x_train)
            x_imputed = pd.DataFrame(x_imputed,columns=x_train.columns)
            smote = SMOTE(random_state=SEED)
            x_train, y_train = smote.fit_resample(x_imputed, y_train)
    
        if model_type == "xgboost":
            model.fit(x_train, y_train,
                    eval_set=[(x_train, y_train),(x_valid, y_valid)],
                    verbose=False,
                    sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                    )
    
        if model_type == "lightgbm":
            model.fit(x_train, y_train,
                        eval_set=[(x_train, y_train),(x_valid, y_valid)],
                        callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=-1)],
                        sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                        )
        
        if model_type == "catboost":
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid),
                      verbose=False,
                      sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                      )

        elif model_type == "histgb":
            model.fit(x_train, y_train,
                      sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                      )

        # Get probabilities and Predict OOF and test
        oof_preds[valid_idx] = model.predict_proba(x_valid)
        test_preds += model.predict_proba(x_test_loop)[:,1] # Take only class 1 probs
        
        proba = model.predict_proba(x_valid)[:, 1]; # print(f'DEBUG TEST PROBA: {proba}')
        fold_score = roc_auc_score(y_valid, proba)
        fold_scores.append(fold_score)
        print(f" Fold {fold+1}: ROC AUC Score: {fold_score:.5f}")
        
        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")
    
    mean_valid_score = np.mean(fold_scores); print(f"Mean ROC AUC: {mean_valid_score:.3f}")
    test_predictions = test_preds / FOLDS
    
    # Save OOF and test probavilities by model
    OOF_probs[model_type] = oof_preds
    test_probs[model_type] = test_predictions


def objective(trial):

  # Sample model weights and normalize
  w1 = trial.suggest_float('w1', 0, 1)
  w2 = trial.suggest_float('w2', 0, 1)
  w3 = trial.suggest_float('w3', 0, 1)
  w4 = 1 - (w1 + w2 + w3) # Constraint: weights must sum to 1

  # Skip invalid combinations
  if w4 < 0 or w4 > 1:
    raise optuna.exceptions.TrialPruned()

  # Sample threshold
  threshold = trial.suggest_float('threshold', 0.3, 0.7)

  # Weighted ensemble of out-of-fold probabilities
  ensemble_probs = (
      w1 * OOF_probs['xgboost'][:,1] +
      w2 * OOF_probs['lightgbm'][:,1] +
      w3 * OOF_probs['catboost'][:,1] +
      w4 * OOF_probs['histgb'][:,1]
  )

  score = roc_auc_score(y, ensemble_probs)

  return score


# Optimize with Optuna
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
# study.optimize(objective, n_trials=1000, timeout=3600)


# Create Submission File

# Define best weights and threshold
w1 = 1.5104304591948773e-05
w2 = 0.6717043056735472
w3 =  0.1439690181901106
w4 = 1 - (w1 + w2 + w3)
threshold = 0.5753159828526501

# Weighted ensemble of model probabilities with weights
ensemble_probs = (
    w1 * test_probs['xgboost'] +
    w2 * test_probs['lightgbm'] +
    w3 * test_probs['catboost'] +
    w4 * test_probs['histgb']
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

