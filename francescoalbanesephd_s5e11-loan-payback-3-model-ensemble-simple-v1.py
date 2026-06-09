# ==================== CONFIGURATIONS ====================
import torch

class CFG:
    IS_KAGGLE        = True # False: colab environment
    
    DEVICE           = "gpu" if torch.cuda.is_available() else "cpu"
    KAGGLE_DATA_PATH = "/kaggle/input/playground-series-s5e11/"
    # COLAB_DATA_PATH  = kagglehub.competition_download('playground-series-s5e11')
    EXTRA_TRAIN_PATH = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"
    TARGET           = "loan_paid_back"
    
    FOLDS                 = 5
    SEED                  = 42
    ADVERSARIAL_THRESHOLD = .65
    TASK_TYPE             = "classification"
    PALETTE_1             = "mako"
    PALETTE_2             = "pastel"         # for better differentiation
    
    ADD_EXTRA_DATA   = True
    ADD_FEATURES     = True
    USE_TE           = False
    HP_SEARCH        = False
    OPTIMIZED_PARAMS = False
    OPTUNA_N_TRIALS  = 50
    SEARCH_WEIGHTS   = True



# ==================== INSTALL & IMPORT LIBRARIES ====================

!pip install itables
!pip install optuna-integration[xgboost]==4.3.0
!pip install catboost==1.2.8
!pip install scikit-learn==1.3.1 # for target_encoder

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
    # torch.backends.cudnn.benchmark = False

# =========================================================================
sns.set_style("darkgrid")

# Silence FutureWarnings
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
def numeric_univariate_plots(
    train, test, target
    # extra
    ):

    # Select columns
    focus_cols = train.select_dtypes(np.number).columns.difference([target])

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train[focus_cols].copy()
    test_temp = test[focus_cols].copy()
    # extra_temp = extra[focus_cols].copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    # extra_temp["Dataset"] = "Extra"
    combined_data = pd.concat([
        train_temp,
        test_temp,
        # extra_temp
        ])

    # Start loop
    for col in focus_cols:

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

        # Box plot
        sns.boxplot(data=combined_data, x=col, y="Dataset", palette=CFG.PALETTE_1, ax=axes[0])
        axes[0].set_xlabel(col)
        axes[0].set_title(f"Box Plot of {col}")

        # Histogram
        if combined_data[col].nunique() > 15:
            sns.histplot(data=combined_data, x=col, hue='Dataset', palette=CFG.PALETTE_1, bins=50,
                         stat='density', common_norm=False, multiple='dodge', kde=False)
            axes[1].set_xlabel(col)
            axes[1].set_ylabel("Frequency")
            axes[1].set_title(f"Histogram of {col} [Train, Test]")
            # axes[1].set_title(f"Histogram of {col} [Train, Test, Extra]")
            # axes[1].legend()
            axes[1].annotate(f"Skewness (TRAIN): {train[col].skew():.2f}\nKurtosis (TRAIN): {train[col].kurt():.2f}",
                             xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])
        else:
            sns.countplot(data=combined_data, x=col, hue='Dataset', palette=CFG.PALETTE_1)
            axes[1].set_xlabel(col)
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Histogram of {col} [Train, Test]")
            # axes[1].set_title(f"Countplot of {col} [Train, Test, Extra]")
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

if CFG.IS_KAGGLE:
  TRAIN_DF = pd.read_csv(CFG.KAGGLE_DATA_PATH+"train.csv", index_col="id")
  TEST_DF  = pd.read_csv(CFG.KAGGLE_DATA_PATH+"test.csv", index_col="id")
  if CFG.ADD_EXTRA_DATA and CFG.EXTRA_TRAIN_PATH !="":
    TRAIN_EXTRA = pd.read_csv(CFG.EXTRA_TRAIN_PATH)
else:
# # Load datasets (on Colab)
  TRAIN_DF = pd.read_csv(os.path.join(CFG.COLAB_DATA_PATH, 'train.csv'),index_col = 'id')
  TEST_DF  = pd.read_csv(os.path.join(CFG.COLAB_DATA_PATH, 'test.csv'),index_col = 'id')
  if CFG.ADD_EXTRA_DATA and CFG.EXTRA_TRAIN_PATH !="":
    TRAIN_EXTRA = pd.read_csv(CFG.EXTRA_TRAIN_PATH)


# Datasets overview
datasets = {
    'TRAIN_DF'   : TRAIN_DF,
    'TEST_DF'    : TEST_DF,
    'TRAIN_EXTRA': TRAIN_EXTRA
    }

# Overview
print_dataset_overview(datasets)
show(TRAIN_DF.head(50))


# Align columns train/extra
common_cols = [col for col in TRAIN_EXTRA.columns if col in TRAIN_DF.columns]
TRAIN_EXTRA = TRAIN_EXTRA[common_cols]


# Distribution of the target variable
plt.figure(figsize=(12, 5))
sns.countplot(data=TRAIN_DF, y=CFG.TARGET, order = TRAIN_DF[CFG.TARGET].value_counts().index, palette='viridis')
plt.title(f'Distribution of {CFG.TARGET} (Target Variable)')
plt.xlabel('Count')
plt.ylabel(f'{CFG.TARGET}')
plt.show()


# Perform univariate analysis for each numerical variable
numeric_univariate_plots(TRAIN_DF, TEST_DF, CFG.TARGET)


# Define features to investigate
cat_cols = TRAIN_DF.select_dtypes(exclude=['number']).columns.difference([CFG.TARGET])

# Visualise categorical variables
fig, axes = plt.subplots(2,3,figsize=(15, 10))
ax = axes.flatten()

for i, col in enumerate(TRAIN_DF[cat_cols]):
    sns.countplot(data=TRAIN_DF, y=col, order = TRAIN_DF[col].value_counts().index, palette=CFG.PALETTE_1, ax=ax[i])

plt.tight_layout()


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(
    data=TRAIN_DF.select_dtypes(np.number).corr(),
    annot=True,
    cmap=CFG.PALETTE_1,
    linewidth=2,
)
plt.tight_layout()
plt.show()


import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier

if CFG.ADD_EXTRA_DATA:
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import OrdinalEncoder
    
    # Make copies for AV
    av_train = TRAIN_DF.copy()
    av_extra = TRAIN_EXTRA.copy()
    
    # Drop the target variable
    av_train.drop(CFG.TARGET, axis=1, inplace=True)
    av_extra.drop(CFG.TARGET, axis=1, inplace=True)
    
    # Add the dataset labels
    av_train["av_label"] = 1
    av_extra["av_label"] = 0
    
    # Concatenate
    adversarial_df = pd.concat([
      av_train,
      av_extra
    ], axis=0, ignore_index=True)
    
    # Shuffle
    adversarial_df = adversarial_df.sample(frac=1, random_state=CFG.SEED)
    
    # Define estimator and cv
    estimator = LGBMClassifier(objective="binary",verbose=0)
    
    # Define X and y
    X = adversarial_df.copy()
    y = X.pop("av_label")
    
    # Encode categorical
    enc = OrdinalEncoder()
    cat_cols = X.copy().select_dtypes("object").columns
    X[cat_cols] = pd.DataFrame(enc.fit_transform(X[cat_cols].copy()), columns=cat_cols)
    
    # Turn booleans into integers
    bool_cols = X.copy().select_dtypes("bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    # Cross validate
    cv_data = pd.DataFrame(
      cross_validate(
      estimator,
          X=X,
          y=y,
          cv=5,    # documentation: "if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used"
          scoring="roc_auc",
          fit_params = {
              "eval_metric":"auc"
          }
      )
    )
    
    # Display results
    display(cv_data)
    AV_RESULTS = np.mean(cv_data['test_score'])
    print(f"Mean cv: {AV_RESULTS}")


if CFG.ADD_EXTRA_DATA:
  # https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
  from sklearn.model_selection import StratifiedKFold
  from sklearn.metrics import roc_curve, auc
  plt.figure(figsize=(8, 6))

  # Initialize skfold
  skf = StratifiedKFold(n_splits=CFG.FOLDS, random_state=CFG.SEED, shuffle=True)

  # Initialize empty lists to store results
  fprs = []
  tprs = []
  aucs = []

  for i, (train_index, test_index) in enumerate(skf.split(X, y)):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      estimator.fit(X_train, y_train)
      y_hat = estimator.predict_proba(X_test)[:, 1]

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


if CFG.ADD_EXTRA_DATA:
  # Preliminary feature importance check
  fig, ax =plt.subplots(figsize=(12,6))
  lgb.plot_importance(estimator,ax=ax)
  plt.show()


# Add extra data to train dataset
if CFG.ADD_EXTRA_DATA and AV_RESULTS < CFG.ADVERSARIAL_THRESHOLD:
    TRAIN_DF = pd.concat([
        TRAIN_DF,
        TRAIN_EXTRA,
    ], axis=0, ignore_index=True)


from sklearn.preprocessing import PolynomialFeatures

# Define X and y
X = TRAIN_DF.copy()
y = X.pop(CFG.TARGET)
X_test = TEST_DF.copy()

def split_grade_subgrade(df):
  temp_df = df.copy()

  temp_df["grade"]    = temp_df["grade_subgrade"].str[0]
  # Correctly convert the second character of 'grade_subgrade' to an integer for each row
  temp_df["subgrade"] = temp_df["grade_subgrade"].str[1].astype(int)
  temp_df.drop("grade_subgrade", axis=1, inplace=True)

  return temp_df

def interact(df):
  temp_df = df.copy()

  # Select numerical columns for interactions
  numerical_cols_for_interactions = temp_df.select_dtypes(np.number).columns

  # Interactions
  interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
  X_interactions = interactions.fit_transform(temp_df[numerical_cols_for_interactions])

  ## Get feature names for the new interaction terms
  interaction_names = interactions.get_feature_names_out(numerical_cols_for_interactions)
  interaction_names = [name.replace(" ", "_X_") for name in interaction_names]

  # Create a DataFrame for the new interaction features
  # Ensure the index of this new DataFrame matches the original DataFrame's index
  interactions_df = pd.DataFrame(X_interactions, columns=interaction_names, index=temp_df.index)

  return interactions_df

def preprocessing(df):
  temp_df = df.copy()

  # First, split grade_subgrade so 'subgrade' (numeric) is available for 'interact'
  temp_df = split_grade_subgrade(temp_df)
  # Then, generate interactions using all current numerical features, including the new 'subgrade'
  temp_df = interact(temp_df)

  return temp_df

# Preprocess data
if CFG.ADD_FEATURES:
    X       = preprocessing(X)
    X_test  = preprocessing(X_test)


import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier

# Preliminary feature importance check

# Identify categorical features
CAT_COLS    = X.select_dtypes("object").columns.tolist()
X[CAT_COLS] = X[CAT_COLS].astype("category")

params = {
    'objective'    : 'binary',
    'metric'       : 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves'   : 11,
    'learning_rate': 0.05,
    "verbose"      :0,
    "random_state" :CFG.SEED
}

# Define and fit estimator
estimator = LGBMClassifier(**params)
estimator.fit(X, y, categorical_feature=CAT_COLS)

# Plot
fig, ax = plt.subplots(figsize=(12,6))
lgb.plot_importance(estimator,ax=ax, color=sns.color_palette(CFG.PALETTE_1)[0], ) # Use the first color from the palette
plt.title('Feature Importance', fontsize=16, weight='bold') # Add title
plt.xlabel('Importance', fontsize=14) # Add x-label
plt.ylabel('Features', fontsize=14) # Add y-label
plt.tight_layout()
plt.show()


import time
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

import lightgbm as lgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


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


def objective(trial, X, y, model_type):
    if model_type == "xgboost":
        from xgboost import XGBRegressor
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
            'device': CFG.DEVICE,
            'random_state': CFG.SEED,
        }
        model = XGBRegressor(**params)

    elif model_type == "lightgbm":
        import lightgbm as lgbm
        from lightgbm import LGBMRegressor
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
            'device': CFG.DEVICE,
            'random_state': CFG.SEED
        }
        model = LGBMRegressor(**params)

    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        params = {
            'iterations': trial.suggest_int("iterations", 200, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'depth': trial.suggest_int("depth", 4, 10),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            'random_state': CFG.SEED,
            'verbose': 0,
            'task_type': CFG.DEVICE.upper(),
            'allow_writing_files': False
        }
        model = CatBoostRegressor(**params)

    else:
        print(f"Error: Unsupported model_type received: {model_type}")
        raise ValueError("Unsupported model_type")

    # Initialize skf
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)

    # # Define TE cols
    # TE_cols = [""] # TE can be applied also with low cardinality numeric features
    # X[TE_cols] = X[TE_cols].astype("category") # not necessary, but good practice

    scores = []

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, pd.qcut(y, q = 10).cat.codes)):
        print(f"\n{'#'*10} Fold {fold+1}/{CFG.FOLDS} {'#'*10}")

        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes('number').columns.difference([CFG.TARGET]))
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if CFG.USE_TE:
            # Initialize target encoder
            target_enc = TargetEncoder(target_type='continuous', smooth='auto', cv=4, shuffle=True, random_state=CFG.SEED)

            # Encode
            x_train[TE_cols] = target_enc.fit_transform(x_train[TE_cols], y_train)
            x_valid[TE_cols] = target_enc.transform(x_valid[TE_cols])

        start = time.time()

        if model_type == "xgboost":
            model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)], # Use validation set for early stopping
                    verbose=200,
                    # early_stopping_rounds=100
                    )

        if model_type == "lightgbm":
            model.fit(x_train, y_train,
                      categorical_feature=CAT_COLS,
                      eval_set=(x_valid, y_valid),
                      callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=200)],
                            )
        if model_type == "catboost":
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid), # Use validation set for early stopping
                      cat_features=CAT_COLS, # Pass categorical features to CatBoost
                      verbose=200,
                      early_stopping_rounds=100
                        )

        # else:
        #   raise ValueError("Unsupported model_type")

        preds  = model.predict_proba(x_valid)[:,1]; # print(f'DEBUG TEST PREDS: {preds}')
        score = roc_auc_score(y_valid, preds)
        scores.append(score)
        end = time.time()

        print(f"AUC: {score:.4f} | Time: {end - start:.2f}s")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Checkpoint description
    print(f"Mean k-FOLDS score: {np.mean(scores)} +- {np.std(scores)}")

    return np.mean(scores)


if CFG.HP_SEARCH:
    # Optimize with Optuna
    study_results = dict()
    for model_type in ["xgboost", "lightgbm", "catboost"]:
      study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
      study.optimize(lambda trial: objective(trial, X=TRAIN_DF.drop(columns=[CFG.TARGET]), y=TRAIN_DF[CFG.TARGET], model_type=model_type),
                    n_trials=CFG.OPTUNA_N_TRIALS, timeout=7200)

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
    for model_type in ["xgboost", "lightgbm", "catboost"]:
      display(pd.DataFrame.from_records(study_results[model_type]))


if CFG.OPTIMIZED_PARAMS:
  XGB_PARAMS  = {
      'n_estimators': 1009,
      'learning_rate': 0.011943333544787549,
      'max_depth': 8,
      'subsample': 0.9143965635679931,
      'colsample_bytree': 0.7670542318585548,
      'reg_alpha': 3.0303575179672426e-05,
      'reg_lambda': 1.0218338193771584e-06,
      "device": CFG.DEVICE,
      "random_state":CFG.SEED
  }

  LGBM_PARAMS = {
      'n_estimators': 1384,
      'learning_rate': 0.030159004697110012,
      'num_leaves': 81,
      'min_child_samples': 56,
      'subsample': 0.7668992634198386,
      'colsample_bytree': 0.7539783365264743,
      'reg_alpha': 2.5249266600932376,
      'reg_lambda': 1.2022289464728586,
      "device": CFG.DEVICE,
      "random_state":CFG.SEED
  }
  CAT_PARAMS  = {
      'iterations': 909,
      'learning_rate': 0.1056470526055604,
      'depth': 6,
      'l2_leaf_reg': 9.897294454000193,
      "task_type": CFG.DEVICE.upper(),
      "random_state":CFG.SEED
  }

else:
  XGB_DEFAULT = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': "hist",  # The 'hist' method is required for native categorical support
    'enable_categorical':True,
    'device': CFG.DEVICE,
    'random_state': CFG.SEED
}

LGBM_DEFAULT = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'objective': 'binary',
    'metric': 'auc',
    'device': CFG.DEVICE,
    'random_state': CFG.SEED
}

CAT_DEFAULT = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': CFG.DEVICE.upper(),
    'random_state': CFG.SEED,
    'verbose': 0
}



# Define X, y, and X_test
X      = TRAIN_DF.copy()
y      = X.pop(CFG.TARGET)
X_test = TEST_DF.copy()

# Identify categorical features (to enable embedded categorical handling)
CAT_COLS         = X.select_dtypes("object").columns.tolist()
X[CAT_COLS]      = X[CAT_COLS].astype("category")
X_test[CAT_COLS] = X_test[CAT_COLS].astype("category")


# Define estimators group with best hyperparams (if available)
estimators = [
    ('xgboost',  XGBClassifier(**XGB_OPTIMIZED)      if CFG.OPTIMIZED_PARAMS  else XGBClassifier(**XGB_DEFAULT)),
    ('lightgbm', LGBMClassifier(**LGBM_OPTIMIZED)    if CFG.OPTIMIZED_PARAMS  else LGBMClassifier(**LGBM_DEFAULT)),
    ('catboost', CatBoostClassifier(**CAT_OPTIMIZED) if CFG.OPTIMIZED_PARAMS  else CatBoostClassifier(**CAT_DEFAULT)),
    ]

# Initialize dictionaries to keep predictions from each model
OOF_PREDS   = dict()
TEST_PREDS  = dict()
FOLD_SCORES = dict()

# Start model loop
for model_type, model in estimators:

    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)

    # Define empty oof variables to fill
    oof_preds   = np.zeros(shape = (len(X)))
    test_preds  = np.zeros(shape = (len(X_test),))
    fold_scores = []

    scores = []

    # Define TE cols
    TE_cols = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"] # TE can be applied also with low cardinality numeric features
    # X[TE_cols] = X[TE_cols].astype("category")

    print(f"\n{'='*20} Fitting {model_type} {'='*20}")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'#'*10} Fold {fold+1}/{CFG.FOLDS} {'#'*10}")

        # Define splits
        x_train, x_valid, _ = iqr_outlier_capping(X.iloc[train_idx], X.iloc[valid_idx], None, columns = X.select_dtypes(np.number).columns.difference([CFG.TARGET]))
        y_train, y_valid    = y.iloc[train_idx], y.iloc[valid_idx]
        x_test_loop         = X_test.copy()

        if CFG.USE_TE:
            # Initialize target encoder
            target_enc = TargetEncoder(target_type='auto', smooth='auto', cv=4, shuffle=True, random_state=CFG.SEED)

            # Encode
            x_train[TE_cols]     = target_enc.fit_transform(x_train[TE_cols], y_train)
            x_valid[TE_cols]     = target_enc.transform(x_valid[TE_cols])
            x_test_loop[TE_cols] = target_enc.transform(x_test_loop[TE_cols])

        start = time.time()

        if model_type == "xgboost":
            model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)], # Use validation set for early stopping
                    verbose=200,
                    early_stopping_rounds=100
                    )

        if model_type == "lightgbm":
            model.fit(x_train, y_train,
                      categorical_feature=CAT_COLS,
                      eval_set=(x_valid, y_valid),
                      callbacks = [lgbm.early_stopping(stopping_rounds=100, verbose=200)],
                            )
        if model_type == "catboost":
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid), # Use validation set for early stopping
                      cat_features=CAT_COLS, # Pass categorical features to CatBoost
                      verbose=200,
                      early_stopping_rounds=100
                        )

        # Get predictions and Predict OOF and test
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:,1] # corrected slicing
        test_preds += model.predict_proba(x_test_loop)[:,1]

        preds      = model.predict_proba(x_valid)[:,1]
        fold_score = roc_auc_score(y_valid, preds) # Calculate AUC score
        fold_scores.append(fold_score)
        print(f" Fold {fold+1}: AUC Score: {fold_score:.5f}")

        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

    mean_valid_score = np.mean(fold_scores); print(f"Mean AUC: {mean_valid_score:.3f}")
    test_predictions = test_preds / CFG.FOLDS

    # Save OOF and test predictions + fold scores by model
    OOF_PREDS[model_type]   = oof_preds
    TEST_PREDS[model_type]  = test_predictions
    FOLD_SCORES[model_type] = fold_scores


scores_df = pd.DataFrame(FOLD_SCORES)

# Boxplots
plt.figure(figsize=(12,6))
sns.boxplot(data=scores_df, palette=CFG.PALETTE_1, orient='h')

# Add titles and labels
plt.title('Score Distribution Across Targets', fontsize=16, weight='bold')
plt.xlabel('ROC-AUC', fontsize=14)
plt.ylabel('Models', fontsize=14)
plt.tight_layout()
plt.show()


def objective(trial):

    # Sample model weights and normalize
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = 1 - (w1 + w2) # Constraint: weights must sum to 1

    # Skip invalid combinations
    if w3 < 0 or w3 > 1:
        raise optuna.exceptions.TrialPruned()

    # Weighted ensemble of out-of-fold probabilities
    ensemble_preds = (
        w1 * OOF_PREDS["xgboost"]  +
        w2 * OOF_PREDS["lightgbm"] +
        w3 * OOF_PREDS["catboost"]
    )

    score = roc_auc_score(y, ensemble_preds)
    return score


if CFG.SEARCH_WEIGHTS:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    study.optimize(objective, n_trials = CFG.OPTUNA_N_TRIALS*10, timeout=3600)


# Define best weights and threshold
w1 = 0.08998770950649088
w2 = 0.9094910839659932
w3 = 1 - (w2+w1)

# Weighted ensemble of model predictions with weights
ensemble_predictions = (
    w1 * TEST_PREDS["xgboost"]  +
    w2 * TEST_PREDS["lightgbm"] +
    w3 * TEST_PREDS["catboost"]
)

# Prepare submission df
submission_df = pd.DataFrame({
    "id":list(X_test.index),
    "y":ensemble_predictions
})

# Display the first rows (sanity check)
display(submission_df.head())

# Save to CSV
submission_df.to_csv('submission.csv', index=False)


# # Plot distribution predictions (train data vs predictions)
# concatenated_df = pd.DataFrame()
# concatenated_df["source_set"] = ["PREDICTED_SCORES"] * submission_df.shape[0] + \
#                                 ["TRAIN_SCORES"] * TRAIN_DF.shape[0]

# concatenated_df[CFG.TARGET] = pd.concat([
#     submission_df['y'],
#     TRAIN_DF[CFG.TARGET]
# ],axis=0,ignore_index=True)

# plt.figure(figsize=(12,6))
# sns.histplot(data=concatenated_df, x=CFG.TARGET, hue='source_set', palette=CFG.PALETTE_1,
#              stat='density', common_norm=False, multiple='dodge', kde=True); plt.show()

