%%capture
!pip install itables
!pip install optuna-integration[xgboost]==4.3.0
!pip install catboost==1.2.8
# !pip install imbalanced-learn==0.12.2

import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

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

# Set plot style
sns.set_style('whitegrid')

# # Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os

# Load datasets (on Kaggle)
TRAIN_DF = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv",index_col = 'id')
TEST_DF = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv",index_col = 'id')
TRAIN_EXTRA = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")

# # Load datasets (on Colab)
# TRAIN_DF = pd.read_csv(os.path.join(playground_series_s5e7_path, 'train.csv'),index_col = 'id')
# TEST_DF = pd.read_csv(os.path.join(playground_series_s5e7_path, 'test.csv'),index_col = 'id')
# TRAIN_EXTRA = pd.read_csv(os.path.join(playground_series_s5e7_path_extra, 'personality_datasert.csv'))


# Print helper function
def print_with_sep(text,sep="=",n=30):
  print("\n")
  print(sep*n)
  print('\t',text)
  print(sep*n)

# Check shapes of all 4 datasets
datasets = {'TRAIN_DF': TRAIN_DF, 'TEST_DF': TEST_DF, 'TRAIN_EXTRA': TRAIN_EXTRA}

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
  percentage_missing = df.isnull().sum()/df.shape[0]
  percentage_missing.name = '% Missing'
  display(
      pd.concat([df.describe(include='all').T,percentage_missing],axis=1).replace(np.nan,'-').style.background_gradient(cmap='Blues'))
  print("\n")


# Drop duplicates
TRAIN_EXTRA.drop_duplicates(inplace=True)

# # Fillna (NOT necessary with tree based models)
# ## Number
# for name, df in datasets.items():
#   for col in df.select_dtypes(include=np.number).columns:
#     df[col].fillna(df[col].median(),inplace=True)

# ## Object
# for name, df in datasets.items():
#   for col in df.select_dtypes(exclude=np.number).columns:
#     df[col].fillna(df[col].mode()[0],inplace=True)


# Identify target
target = 'Personality'

# Distribution of the target variable
plt.figure(figsize=(12, 5))
sns.countplot(data=TRAIN_DF, y=target, order = TRAIN_DF[target].value_counts().index, palette='viridis')
plt.title(f'Distribution of {target.capitalize()} (Target Variable)')
plt.xlabel('Count')
plt.ylabel(f'{target.capitalize()}')
plt.show()


# Relationship between numerical features and the target variable (using boxplots)

# Get the numerical features excluding the target and the index
numerical_features = TRAIN_DF.select_dtypes(include=np.number).columns.tolist()

# Set up the subplot grid
fig, axes = plt.subplots(3, 2, figsize=(18, 3 * 6))
axes = axes.flatten()

# Iterate through the numerical features and create boxplots
for i, feature in enumerate(numerical_features):
  sns.boxplot(y=target, x=feature, data=TRAIN_DF, ax=axes[i], hue = target, palette='viridis')
  axes[i].set_title(f'Target vs {feature}')
  axes[i].set_ylabel('Target')
  axes[i].set_xlabel(feature)

plt.tight_layout()
plt.show()


# Analysis of all NUMERIC features
# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Function to create and display plots for a single numerical variable
def create_variable_plots(train, test, variable):

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train.copy()
    test_temp = test.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    combined_data = pd.concat([train_temp, test_temp])

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
    create_variable_plots(TRAIN_DF, TEST_DF, variable)


# Define features to investigate
cat_cols = TRAIN_DF.select_dtypes(exclude='number').columns.difference([target])

# Visualise categorical variables
fig, axes = plt.subplots(1,2,figsize=(15, 5))
ax = axes.flatten()

for i, col in enumerate(TRAIN_DF[cat_cols]):
    sns.countplot(data=TRAIN_DF, y=col, order = TRAIN_DF[col].value_counts().index, palette='viridis', ax=ax[i])


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


from tqdm import tqdm
from itertools import combinations

# Select columns to encode
cols_to_encode = TRAIN_DF.select_dtypes(include=np.number).columns

# ==========================================================================
  ## INTERACTIONS

# Combo size
combo_size = [2, 3, 4]

# Combo loop
for n in combo_size[:1]: # restrict to combo = 2 for now
  combination_lists = list(combinations(cols_to_encode, n))
  for cols in tqdm(combination_lists, desc = 'Computing interactions...'):
    new_col_name = '_x_'.join(cols)

    # compute interaction term
    TRAIN_DF[new_col_name] = TRAIN_DF[list(cols)].product(axis=1)
    TEST_DF[new_col_name] = TEST_DF[list(cols)].product(axis=1)

# ==========================================================================
  ## RATIOS
# combinations_list = list(combinations(cols_to_encode, 2)) # ratios are done in pairs
# for cols in tqdm(combinations_list, desc = 'Computing ratios...'):
#   col1, col2 = cols

#   # Create two ratio features: col1 / col2 and col2 / col1
#   new_col_name_ratio1 = f'{col1}_/_{col2}'
#   new_col_name_ratio2 = f'{col2}_/_{col1}'

#   # Add a small epsilon to the denominator to avoid division by zero
#   Îµ = 1e-5

#   # Compute the ratio terms
#   TRAIN_DF[new_col_name_ratio1] = TRAIN_DF[col1] / (TRAIN_DF[col2] + Îµ)
#   TRAIN_DF[new_col_name_ratio2] = TRAIN_DF[col2] / (TRAIN_DF[col1] + Îµ)
#   TEST_DF[new_col_name_ratio1] = TEST_DF[col1] / (TEST_DF[col2] + Îµ)
#   TEST_DF[new_col_name_ratio2] = TEST_DF[col2] / (TEST_DF[col1] + Îµ)


print("Modified TRAIN_DF with interaction terms:")
show(TRAIN_DF)
print("\nModified TEST_DF with interaction terms:")
show(TEST_DF)


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Identify categorical columns excluding the target
categorical_cols = TRAIN_DF.select_dtypes(include='object').columns.tolist()
categorical_cols.remove(target)

# Feature Encoding
oe = OrdinalEncoder()
TRAIN_DF[cat_cols] = oe.fit_transform(TRAIN_DF[cat_cols])
TEST_DF[cat_cols] = oe.transform(TEST_DF[cat_cols])

# Label encode the target variable
label_encoder = LabelEncoder()
TRAIN_DF[target] = label_encoder.fit_transform(TRAIN_DF[target])

print("Categorical features encoded.")
print("Encoded TRAIN_DF head:")
display(TRAIN_DF.head())
print("\nEncoded TEST_DF head:")
display(TEST_DF.head())


import time
import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


# FOLDS = 5

# def objective(trial, X, y, model_type):
#     if model_type == "xgboost":
#         from xgboost import XGBClassifier
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
#             'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
#             'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
#             # 'use_label_encoder': False,
#             'eval_metric': 'logloss',
#             'early_stopping_rounds' : 100,
#             'device': 'cuda',
#             'random_state': SEED,
#         }
#         model = XGBClassifier(**params)

#     elif model_type == "lightgbm":
#         import lightgbm as lgbm
#         from lightgbm import LGBMClassifier
#         params = {
#             'n_estimators': trial.suggest_int("n_estimators", 200, 1500),
#             'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#             'num_leaves': trial.suggest_int("num_leaves", 15, 150),
#             'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
#             'subsample': trial.suggest_float("subsample", 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
#             'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 10.0),
#             'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 10.0),
#             'verbose':-1,
#             'device': 'gpu',
#             'random_state': SEED
#         }
#         model = LGBMClassifier(**params)

#     elif model_type == "catboost":
#         from catboost import CatBoostClassifier
#         params = {
#             'iterations': trial.suggest_int("iterations", 200, 1000),
#             'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#             'depth': trial.suggest_int("depth", 4, 10),
#             'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
#             'random_state': SEED,
#             'verbose': 0,
#             # 'task_type': "GPU",
#             'allow_writing_files': False
#         }
#         model = CatBoostClassifier(**params)

#     elif model_type == "histgb":
#         from sklearn.ensemble import HistGradientBoostingClassifier
#         params = {
#             'max_iter': trial.suggest_int("max_iter", 200, 1500),
#             'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#             'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 15, 255),
#             'min_samples_leaf': trial.suggest_int("min_samples_leaf", 10, 100),
#             'l2_regularization': trial.suggest_float("l2_regularization", 0.0, 10.0),
#             'max_depth': trial.suggest_int("max_depth", 3, 12),
#             'early_stopping': True,
#             'n_iter_no_change': 100,
#             'validation_fraction': 0.1,
#             'verbose': 0,
#             'random_state': SEED
#         }
#         model = HistGradientBoostingClassifier(**params)

#     else:
#         print(f"Error: Unsupported model_type received: {model_type}")
#         raise ValueError("Unsupported model_type")

#     # Handle class imbalance
#     imbalance_strategy = trial.suggest_categorical("imbalance_strategy", ["weights", "SMOTE"])
#     skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
#     scores = []

#     print(f"\n{'='*20} Fitting {model_type} {'='*20}")
#     for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
#         print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
#         x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

#         start = time.time()

#         # XGBoost, LightGBM, and CatBoost support early stopping natively via fit() arguments.
#         # HistGradientBoostingClassifier from sklearn (as of now) does not have native early stopping.

#         use_weights = imbalance_strategy == "weights"
#         use_smote = imbalance_strategy == "SMOTE"
#         if use_smote:
#           # SMOTE needs a df with NO nans
#           imputer = SimpleImputer(strategy='median')
#           x_imputed = imputer.fit_transform(x_train)
#           x_imputed = pd.DataFrame(x_imputed,columns=x_train.columns)
#           smote = SMOTE(random_state=SEED)
#           x_train, y_train = smote.fit_resample(x_imputed, y_train)

#         if model_type == "xgboost":
#           model.fit(x_train, y_train,
#                     eval_set=[(x_train, y_train),(x_valid, y_valid)],
#                     verbose=False,
#                     sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
#                     )

#         if model_type == "lightgbm":
#           model.fit(x_train, y_train,
#                     eval_set=[(x_train, y_train),(x_valid, y_valid)],
#                     callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=False)],
#                     sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
#                     )

#         if model_type == "catboost":
#           model.fit(x_train, y_train,
#                     eval_set=(x_valid, y_valid),
#                     verbose=False,
#                     sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
#                     )

#         elif model_type == "histgb":
#           model.fit(x_train, y_train,
#                     sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
#                     )

#         # else:
#         #   raise ValueError("Unsupported model_type")

#         preds = model.predict(x_valid)
#         score = accuracy_score(y_valid, preds)
#         scores.append(score)

#         # Fold score
#         print(f"Accuracy: {score:.4f} | Time: {time.time() - start:.2f}s")

#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()

#     # Checkpoint description
#     print(f"Mean k-FOLDS score: {np.mean(scores)} +- {np.std(scores)}")

#     return np.mean(scores)


# # Optimize with Optuna
# study_results = dict()
# for model_type in ["xgboost", "lightgbm", "catboost", "histgb"][-1:]:
#   study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
#   study.optimize(lambda trial: objective(trial, X=TRAIN_DF.drop(columns=[target]), y=TRAIN_DF[target], model_type=model_type),
#                 n_trials=50, timeout=3600)

#   # nest params, value and trial within each model dictionary
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
    'n_estimators': 565,
    'learning_rate': 0.003920107890975799,
    'max_depth': 9,
    'subsample': 0.5388098992001636,
    'colsample_bytree': 0.7226536169552121,
    'reg_alpha': 5.9938051833451595,
    'reg_lambda': 1.8726109477891196e-05,

    # 'imbalance_strategy': 'weights'
    }

LGBM_PARAMS = {
    'n_estimators': 292,
    'learning_rate': 0.002870891248413428,
    'num_leaves': 150,
    'min_child_samples': 32,
    'subsample': 0.9623434875219083,
    'colsample_bytree': 0.5508149959901448,
    'reg_alpha': 8.786794906311602,
    'reg_lambda': 4.803133171428337,
    'verbose': -1

    # 'imbalance_strategy': 'weights'
    }

CAT_PARAMS = {
    'iterations': 502,
    'learning_rate': 0.0028055617972448117,
    'depth': 8,
    'l2_leaf_reg': 6.938385887172275,

    # 'imbalance_strategy': 'weights'
    }

HISTGB_PARAMS = {
    'max_iter': 945,
    'learning_rate': 0.03114307424904254,
    'max_leaf_nodes': 202,
    'min_samples_leaf': 98,
    'l2_regularization': 7.787011356651036,
    'max_depth': 5,

    # 'imbalance_strategy': 'weights'
    }


# Define estimators group (with best hyperparams)
estimators = [
    ('xgboost', XGBClassifier(**XGB_PARAMS)),
    ('lightgbm', LGBMClassifier(**LGBM_PARAMS)),
    ('catboost', CatBoostClassifier(**CAT_PARAMS)),
    ('histgb', HistGradientBoostingClassifier(**HISTGB_PARAMS))
    ]

# Define X and y
X=TRAIN_DF.drop(columns=[target])
y=TRAIN_DF[target]

# Initialize dictionaries to keep predicted probs from each model
OOF_probs = dict()

# Start model loop
for model_type, model in estimators:

  # Handle class imbalance
  imbalance_strategy = "weights"

  # Initialize skf
  FOLDS = 5
  skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

  # Define empty oof variables to fill
  oof_preds = np.zeros(shape = (len(X),)) # Shape as 1d array rather than using y.nunique() because I only need class_1 probs

  # Start fold loop
  print(f"\n{'='*20} Fitting {model_type} {'='*20}")
  for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
      print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
      x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
      y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

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
      oof_preds[valid_idx] = model.predict_proba(x_valid)[:,1] # Take only class 1 probs

      end = time.time()
      print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

  # Save OOF preds by model
  OOF_probs[model_type] = oof_preds


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
      w1 * OOF_probs['xgboost'] +
      w2 * OOF_probs['lightgbm'] +
      w3 * OOF_probs['catboost'] +
      w4 * OOF_probs['histgb']
  )

  preds = (ensemble_probs >= threshold).astype(int)
  score = accuracy_score(y, preds)

  return score


# Optimize with Optuna
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=1000, timeout=3600)


# On COLAB env
{'w1': 0.007892107923276638, 'w2': 0.05720400316750507, 'w3': 2.4847119116585235e-05, 'threshold': 0.6575929044648897} # Best is trial 263 with value: 0.9693370762254373

# On KAGGLE env
{'w1': 0.051677838012044654, 'w2': 0.00047488647513526966, 'w3': 0.21842773433196772, 'threshold': 0.6911635730953096} # Best is trial 58 with value: 0.9693910602461672.


# Define estimators group (with best hyperparams)
estimators = [
    ('xgboost', XGBClassifier(**XGB_PARAMS)),
    ('lightgbm', LGBMClassifier(**LGBM_PARAMS)),
    ('catboost', CatBoostClassifier(**CAT_PARAMS)),
    ('histgb', HistGradientBoostingClassifier(**HISTGB_PARAMS))
    ]

# Define X and y
X=TRAIN_DF.drop(columns=[target])
y=TRAIN_DF[target]
X_test = TEST_DF

# Initialize dictionaries to keep predicted probs from each model
test_probs = dict()

# Start model loop
for model_type, model in estimators:

  # Handle class imbalance
  imbalance_strategy = "weights"

  # Initialize skf
  FOLDS = 5
  skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

  # Define empty test variables to fill
  oof_preds = np.zeros(shape = (len(TRAIN_DF) ,y.nunique()))
  test_preds = np.zeros(shape = (len(X_test),)) # Shape as 1d array rather than using y.nunique() because I only need class_1 probs
  fold_scores = []

  # Start fold loop
  print(f"\n{'='*20} Fitting {model_type} {'='*20}")
  for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
      print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")
      x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
      y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

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
                  verbose=100,
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
                  verbose=100,
                  sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                  )

      elif model_type == "histgb":
        model.fit(x_train, y_train,
                  sample_weight = compute_sample_weight(class_weight='balanced',y=y_train) if use_weights else None
                  )


      # Get probabilities and Predict OOF and test
      oof_preds[valid_idx] = model.predict_proba(x_valid)
      test_preds += model.predict_proba(X_test)[:,1] # Take only class 1 probs

      # Calculate fold score
      valid_preds = model.predict(x_valid)
      fold_score = accuracy_score(y_valid, valid_preds)
      fold_scores.append(fold_score)
      print(f" Fold {fold+1}: Accuracy Score: {fold_score:.5f}")

      end = time.time()
      print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

  mean_valid_score = np.mean(fold_scores); print(f"Mean Accuracy: {mean_valid_score:.3f}")
  test_predictions = test_preds / FOLDS

  # Save test preds by model
  test_probs[model_type] = test_predictions


# Create Submission File

# Define best weights and threshold
w1 = 0.051677838012044654
w2 = 0.00047488647513526966
w3 = 0.21842773433196772
w4 = 1 - (w1 + w2 + w3)
threshold = 0.6911635730953096

# Weighted ensemble of model probabilities with weights
ensemble_probs = (
    w1 * test_probs['xgboost'] +
    w2 * test_probs['lightgbm'] +
    w3 * test_probs['catboost'] +
    w4 * test_probs['histgb']
)

final_preds = (ensemble_probs >= threshold).astype(int)

submission_df = pd.DataFrame({
    'id': list(X_test.index),
    'Personality': label_encoder.inverse_transform(final_preds)
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Display the first 5 rows
display(submission_df.head())

# Plot preds distribution
sns.histplot(submission_df['Personality'],bins=2)
plt.show()


