# Install necessary components
!pip -q install optuna xgboost matplotlib plotly
!pip install optuna-integration[lightgbm]


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import random 
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)
from IPython.display import display  # for nicer display in notebooks
from pandas.api.types import is_categorical_dtype
import optuna
import optuna.visualization as ov
import matplotlib.pyplot as plt
%matplotlib inline


# Define some awesome utilities, from C4rl05/V on kaggle
def configure_notebook(seed=548, float_precision=3, max_columns=15, max_rows=25):
    """
    Configure notebook settings:
      - Disables warnings for cleaner output.
      - Sets pandas display options for better table formatting.
      - Returns a seed value for reproducibility.
    
    Parameters:
      seed (int): Random seed (default 548).
      float_precision (int): Number of decimal places for floats (default 3).
      max_columns (int): Maximum number of columns to display (default 15).
      max_rows (int): Maximum number of rows to display (default 25).

    Returns:
      int: The provided seed.
    """
    # Disable all warnings
    warnings.filterwarnings("ignore")
    
    # Set pandas display options for nicer output
    pd.options.display.float_format = f"{{:,.{float_precision}f}}".format
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)

    # Set seeds for reproducibility in numpy and the standard random module
    np.random.seed(seed)
    random.seed(seed)
    
    return seed

# Apply configuration and set random seeds for reproducibility
seed = configure_notebook()    


# Configurable flag to control whether GPU is used
USE_GPU = True


def eda_summary(df):
    # 1. Display the first few rows
    print("======== First 5 Rows ========")
    display(df.head().T)
    
    # 2. DataFrame information (data types, non-null counts, etc.)
    print("\n======== DataFrame Info ========")
    df.info()
    
    # 3. Descriptive statistics for numeric columns
    print("\n======== Descriptive Statistics (Numeric Columns) ========")
    display(df.describe())
    
    # 4. Descriptive statistics for categorical columns (if any)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    print("\n======== Descriptive Statistics (Categorical Columns) ========")
    if not categorical_df.empty:
        display(categorical_df.describe())
    else:
        print("No categorical columns found.")
    
    # 5. Missing values summary
    print("\n======== Missing Values Summary ========")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": missing_percent
    })
    display(missing_summary)
    
    # 6. Count of duplicated rows
    print("\n======== Duplicated Rows ========")
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    
    # 7. Count of each data type
    print("\n======== Data Types Count ========")
    display(df.dtypes.value_counts())
    
    # 8. Correlation matrix for numeric variables (if more than one exists)
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        print("\n======== Correlation Matrix (Numeric Columns) ========")
        display(numeric_cols.corr())
    else:
        print("\n======== Correlation Matrix ========")
        print("Not enough numeric columns to compute correlation.")
    
    # 9. Value counts for categorical variables with low cardinality
    print("\n======== Value Counts for Categorical Columns (Low Cardinality) ========")
    if not categorical_df.empty:
        for col in categorical_df.columns:
            if df[col].nunique() <= 20:
                print(f"\nValue Counts for '{col}':")
                display(df[col].value_counts())
    else:
        print("No categorical columns found.")



PLAYGROUND_PATH = '/kaggle/input/playground-series-s5e8/'


training_df = pd.read_csv(PLAYGROUND_PATH + 'train.csv')
training_df


test_df = pd.read_csv(PLAYGROUND_PATH + 'test.csv')
test_df


training_df.drop('id', axis=1, inplace=True)
training_df.head()


# Save a copy with the ID for use later
test_ids = test_df['id'].copy()

test_df.drop('id', axis=1, inplace=True)
test_df.head()


eda_summary(training_df)


# Plot Target Distribution
ax = (training_df['y'].value_counts(normalize=True)
        .rename({0:'No',1:'Yes'})
        .plot(kind='bar'))
plt.title('Target Distribution'); 
plt.xlabel('Opened Account'); 
plt.ylabel('Share'); 
plt.show()


# Plot duration
training_df['duration'].clip(upper=training_df['duration'].quantile(0.99)).plot(kind='hist', bins=50)
plt.title('Duration (clipped 99th)'); 
plt.xlabel('seconds'); 
plt.show()


# Plot pdays missingness
(training_df['pdays'].eq(-1) | training_df['pdays'].isna()).value_counts(normalize=True).plot(kind='bar')
plt.title('pdays missing / never-contacted rate'); 
plt.show()


eda_summary(test_df)


# Plot duration
test_df['duration'].clip(upper=test_df['duration'].quantile(0.99)).plot(kind='hist', bins=50)
plt.title('Duration (clipped 99th)'); 
plt.xlabel('seconds'); 
plt.show()


# Plot pdays missingness
(test_df['pdays'].eq(-1) | test_df['pdays'].isna()).value_counts(normalize=True).plot(kind='bar')
plt.title('pdays missing / never-contacted rate'); 
plt.show()


def transform_job(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'job' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['job']):
        print("The 'job' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    out['job'] = out['job'].astype('category')
        
    return out


def fit_marital(df: pd.DataFrame):
    global marital_encoder
    
    # If the 'marital' feature isn't present, then remark about it and return
    if 'marital' not in df.columns:
        print("The 'marital' feature has already been transformed.  Operation cancelled.")
        return
        
    marital_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    marital_encoder.fit(df[['marital']])

def transform_marital(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    """Idempotent one-hot transform for 'marital'.
    - Drops any previously generated marital one-hot columns.
    - Works even if called multiple times.
    - Robust to unseen categories if encoder was created with handle_unknown='ignore'.
    """
    # If the 'marital' feature isn't present, then remark about it and return
    if 'marital' not in df.columns:
        print("The 'marital' feature has already been transformed.  Operation cancelled.")
        return df
        
    # Ensure encoder is fitted
    if not hasattr(marital_encoder, "categories_"):
        marital_encoder.fit(df[["marital"]])

    # Compute consistent column names
    cols = marital_encoder.get_feature_names_out(["marital"])

    # Get a copy of the DataFrame to work upon.
    out = df.copy()
    
    # Transform to sparse matrix
    X = marital_encoder.transform(out[["marital"]])

    # Drop existing one-hot columns if present
    to_drop = [c for c in cols if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)

    # Build a (sparse) DataFrame aligned to df.index
    df_ohe = pd.DataFrame.sparse.from_spmatrix(X, index=out.index, columns=cols)
    df_ohe = df_ohe.astype("Int8")
    
    # Optionally drop the original column
    if drop_original and "marital" in out.columns:
        out = out.drop(columns=["marital"])

    # Concatenate and return
    return pd.concat([out, df_ohe], axis=1)

def save_marital_encoder():
    joblib.dump(marital_encoder, "marital_encoder.pkl")
    print('marital_encoder saved')


education_encoder = LabelEncoder()

def fit_education(df: pd.DataFrame):
    # If the 'education' feature isn't present, then remark about it and return
    if 'education' not in df.columns:
        print("The 'education' feature has already been transformed.  Operation cancelled.")
        return

    education_encoder.fit(df['education'])

def transform_education(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'education' feature isn't present, then remark about it and return
    if 'education' not in df.columns:
        print("The 'education' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    out['encoded_education'] = education_encoder.transform(out['education'])
    out['education_unknown'] = (out['education'] == 'unknown').astype('int8')

    if drop_original:
        out.drop('education', axis=1, inplace=True)
        
    return out

def save_education_encoder():
    joblib.dump(education_encoder, "education_encoder.pkl")
    print('education_encoder saved')


def convert_default(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'default' feature isn't present, then remark about it and return
    if 'default' not in df.columns:
        print("The 'default' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['default_bool'] = out['default'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out.drop('default', axis=1, inplace=True)

    return out


def convert_housing(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'housing' feature isn't present, then remark about it and return
    if 'housing' not in df.columns:
        print("The 'housing' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['housing_bool'] = out['housing'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out.drop('housing', axis=1, inplace=True)

    return out


def convert_loan(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'loan' feature isn't present, then remark about it and return
    if 'loan' not in df.columns:
        print("The 'loan' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['loan_bool'] = out['loan'].str.lower().map({"yes": 1, "no": 0}).astype("Int8")

    if drop_original:
        out.drop('loan', axis=1, inplace=True)

    return out


def transform_contact(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'contact' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['contact']):
        print("The 'contact' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['contact'] = out['contact'].astype('category')

    return out


# First, let's see if there are any leap-year dates in the dataset
training_df[(training_df['month'] == 'feb') & (training_df['day'] == 29)]


# Alright, we found both Feb 29 as well as out of range day values.
# So, we'll choose 2020 as the year, since that's a leap year.
# Wealso take action to coerce the bad dates into good ones.
def convert_calendar_features(df, drop_originals: bool):
    # If the 'month' feature isn't present, then remark about it and return
    if 'month' not in df.columns:
        print("The 'month' feature has already been transformed.  Operation cancelled.")
        return df

    # Work on a copy so we don't mess stuff up if this fails halfway throug
    out = df.copy()

    # Create a numbered month series
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    month = out["month"].str.lower().map(month_map)

    # Create a year series that is the correct length
    year_array = np.empty(out["month"].size)
    year_array.fill(2020)
    year = pd.Series(data=year_array, name='year')

    # Assemble those series into a DataFrame suitable for
    raw_date_df = pd.DataFrame(
        {'year': year,
         'month': month,
         'day': out['day']
        }
    )

    # Look for bad dates and try to show them
    bad = pd.to_datetime(raw_date_df, errors="coerce")
    raw_date_df.loc[bad.isna()]

    # There's some bogus dats in the dataset, so we need to do some hygiene.
    # 1) month-end day for each row (handles leap years)
    month_start = pd.to_datetime(dict(year=raw_date_df["year"],
                                      month=raw_date_df["month"],
                                      day=1))
    month_end_day = (month_start + pd.offsets.MonthEnd(0)).dt.day
    
    # 2) clip day into [1, month_end_day]
    day_clipped = raw_date_df["day"].clip(lower=1, upper=month_end_day)

    # 3) build safe dates
    date_df = pd.to_datetime(dict(year=raw_date_df["year"],
                                  month=raw_date_df["month"],
                                  day=day_clipped))
    
    # Flag rows that were adjusted
    adjusted = day_clipped.ne(raw_date_df["day"])
    print(f"Adjusted {adjusted.sum()} rows where day exceeded month length.")
    
    # Now, create new features for the day of year and cyclical encoding.
    out["day_of_year"] = date_df.dt.dayofyear
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)

    if drop_originals:
        out.drop('month', axis=1, inplace=True)
        out.drop('day', axis=1, inplace=True)

    return out


def transform_previous(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    # If the 'previous' feature isn't present, then remark about it and return
    if 'previous' not in df.columns:
        print("The 'previous' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['prev_any'] = (out['previous'] > 0).astype('int8')

    out['prev_log'] = np.where(out['previous'] > 0,
                                   np.log1p(out["previous"]),
                                   0)

    if drop_original:
        out.drop('previous', axis=1, inplace=True)

    return out    


def transform_poutcome(df: pd.DataFrame) -> pd.DataFrame:
    # If the 'poutcome' feature is already categorical, then remark about it and return
    if is_categorical_dtype(df['poutcome']):
        print("The 'poutcome' feature has already been transformed.  Operation cancelled.")
        return df

    out = df.copy()
    
    out['poutcome'] = out['poutcome'].astype('category')

    return out


# Fit the features
def fit_features(df: pd.DataFrame):
    fit_marital(df)
    fit_education(df)

    save_marital_encoder()
    save_education_encoder()


# Transform features
def transform_features(df: pd.DataFrame, drop_original: bool) -> pd.DataFrame:
    out_df = df.copy()

    out_df = transform_job(out_df)
    out_df = transform_marital(out_df, drop_original)
    out_df = transform_education(out_df, drop_original)
    out_df = convert_default(out_df, drop_original)
    out_df = convert_housing(out_df, drop_original)
    out_df = convert_loan(out_df, drop_original)
    out_df = transform_contact(out_df)
    out_df = convert_calendar_features(out_df, drop_original)
    
    # Add logarithmic duration
    out_df['log_duration'] = np.log1p(out_df['duration'])

    # The 'pdays' feature uses -1 to indicate nothing.  Replace all these with NaNs and add a 'pdays_missing' feature
    out_df.loc[out_df['pdays'] == -1, 'pdays'] = np.nan
    out_df["pdays_missing"] = (out_df["pdays"].isna()).astype("int8")

    out_df = transform_previous(out_df, drop_original)
    out_df = transform_poutcome(out_df)

    return out_df


fit_features(training_df)


training_df = transform_features(training_df, True)
training_df.head(10)


test_df = transform_features(test_df, True)
test_df.head(10)


# Examine the transformed training dataset
eda_summary(training_df)


# Examine the transformed test dataset
eda_summary(test_df)


# Separate the predictors(X) from the target(y)
TARGET = "y"

X = training_df.drop(columns=[TARGET])
y = training_df[TARGET]


def stratified_subsample(X, y, n=200_000, seed=42):
    import numpy as np
    pos_idx = y[y==1].index.to_numpy()
    neg_idx = y[y==0].index.to_numpy()
    rng = np.random.default_rng(seed)
    k_pos = int(n * y.mean())
    k_neg = n - k_pos
    keep = np.r_[rng.choice(pos_idx, k_pos, replace=False),
                 rng.choice(neg_idx, k_neg, replace=False)]
    keep.sort()
    return X.loc[keep], y.loc[keep]

X_tune, y_tune = stratified_subsample(X, y, n=200_000, seed=seed)
folds_tune = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_tune, y_tune))


# imbalance weight
pos = y.sum()
neg = len(y) - pos
spw = neg / pos

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "tree_method": "hist" if not USE_GPU else "gpu_hist",
    "predictor": "auto" if not USE_GPU else "gpu_predictor",
    "learning_rate": 0.03,
    "max_depth": 7,               # ≈ num_leaves ~ 128
    "min_child_weight": 50,       # ≈ LGBM min_data_in_leaf
    "subsample": 0.8,             # ≈ bagging_fraction
    "colsample_bytree": 0.8,      # ≈ feature_fraction
    "alpha": 0.0,                 # L1
    "lambda": 5.0,                # L2
    "gamma": 0.0,
    "scale_pos_weight": spw,
    "max_bin": 512,
    "colsample_bylevel": 0.9,
    "seed": seed,
}

dall = xgb.DMatrix(X, label=y, enable_categorical=True)

# Use the SAME folds in xgb.cv to pick rounds
print("Starting cross-fold validation")
cv_res = xgb.cv(
    params,
    dall,
    num_boost_round=10000,
    folds=folds_tune,
    early_stopping_rounds=300,
    seed=seed,
    verbose_eval=200,
)
best_n_rounds = len(cv_res)
print("Done.")
print("CV AUC:", cv_res["test-auc-mean"].iloc[-1], "@ rounds:", best_n_rounds)

# Final model on ALL data for exactly that many rounds
print("Training model...")
model = xgb.train(params, dall, num_boost_round=best_n_rounds)
print("Done.")

y_all_pred = model.predict(dall)
print("Validation AUC:", roc_auc_score(y, y_all_pred))


# Plot feature importances
xgb.plot_importance(model, max_num_features=20)
plt.title('XGBoost Feature Importance'); plt.show()


# Reuse the K-folds from the initial training
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# folds = list(kf.split(X, y))

sampler = optuna.samplers.TPESampler(
    multivariate=True, group=True, n_startup_trials=10, seed=seed
)

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)


def build_fold_dmats(X, y, folds, enable_cat=True):
    out = []
    for tr, va in folds:
        dtr = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr], enable_categorical=enable_cat)
        dva = xgb.DMatrix(X.iloc[va], label=y.iloc[va], enable_categorical=enable_cat)
        out.append((dtr, dva, va))
    return out

fold_dmats = build_fold_dmats(X_tune, y_tune, folds_tune, enable_cat=True)


def objective(trial):
    grow_mode = trial.suggest_categorical("grow_mode", ["depth", "lossguide"])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "predictor": "gpu_predictor" if USE_GPU else "auto",
        "enable_categorical": True,
        "scale_pos_weight": spw,
        "seed": seed,
        "verbosity": 0,

        # tuned knobs
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
    }

    if grow_mode == "depth":
        params["max_depth"] = trial.suggest_int("max_depth", 6, 9)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 20.0, 200.0, log=True)
    else:
        params["grow_policy"] = "lossguide"
        params["max_depth"] = 0
        params["max_leaves"] = trial.suggest_int("max_leaves", 128, 512, log=True)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 10.0, 120.0, log=True)

    # Use the subsample label length
    oof = np.full(len(y_tune), np.nan, dtype=float)
    
    for dtr, dva, va_idx in fold_dmats:
        booster = xgb.train(
            params, dtr, num_boost_round=3000,
            evals=[(dva, "valid")],
            early_stopping_rounds=150,
            verbose_eval=False,
            callbacks=[optuna.integration.XGBoostPruningCallback(trial, "valid-auc")],
        )
        oof[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    return roc_auc_score(y_tune, oof)


study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

seed_params = params.copy()

# Make it consistent with the search space:
seed_params.update({
    "grow_mode": "depth",
    "learning_rate": min(max(seed_params.get("learning_rate", 0.05), 0.02), 0.05),
    "subsample": min(max(seed_params.get("subsample", 0.8), 0.6), 1.0),
    "colsample_bytree": min(max(seed_params.get("colsample_bytree", 0.8), 0.6), 1.0),
    "colsample_bylevel": 0.9,
    "colsample_bynode": 0.8,
    "gamma": max(seed_params.get("gamma", 0.0), 0.0),
    "alpha": max(seed_params.get("alpha", 1e-6), 1e-6),
    "lambda": max(seed_params.get("lambda", 5.0), 1e-4),
    "max_bin": 256,
    "max_depth": 7,
})

# Supply the study with the ad hoc parameters
study.enqueue_trial(seed_params)

study.optimize(objective, n_trials=150, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)


# Plot Optuna optimization history plots
ov.plot_optimization_history(study)


# Plot Optuna's idea of the importance of each of the parameters
ov.plot_param_importances(study)


def finalize_params(best_params, use_gpu=False, spw=1.0, seed=42):
    p = best_params.copy()

    # Map our custom branch key → real XGBoost params
    if "grow_mode" in p:
        gm = p.pop("grow_mode")
        if gm == "lossguide":
            p["grow_policy"] = "lossguide"
            p["max_depth"] = 0                 # required with lossguide
            # expect 'max_leaves' to be present from tuning
        else:
            p.pop("max_leaves", None)          # depth mode doesn't use this

    # Ensure required fixed bits
    p.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist" if not USE_GPU else "gpu_hist",
        "predictor": "auto" if not USE_GPU else "gpu_predictor",
        "enable_categorical": True,
        "scale_pos_weight": float(spw),
        "seed": seed,
        "verbosity": 1,
    })

    # Remove any None/unexpected keys just in case
    for k in ["grow_mode"]:
        p.pop(k, None)
    return p

best_params_final = finalize_params(study.best_params, use_gpu=USE_GPU, spw=spw, seed=seed)



# Train final with best settings
print("Training model on optuna-derived settings....")

# Start from Optuna's best and add the fixed bits
best_params = finalize_params(study.best_params, use_gpu=USE_GPU, spw=spw, seed=seed)

# Use CV to get best number of rounds with early stopping
cv_res = xgb.cv(
    best_params,
    dall,
    num_boost_round=10000,
    folds=folds_tune,
    early_stopping_rounds=300,
    seed=seed,
    verbose_eval=200,
)
best_n_rounds = len(cv_res)
print("Best CV AUC:", cv_res["test-auc-mean"].iloc[-1], " @ rounds:", best_n_rounds)

# Train final on ALL data for best_n_rounds
final_model = xgb.train(
    best_params,
    dall,
    num_boost_round=best_n_rounds
)

print("Done")


# Let's see how the fully trained model does
y_final_valid_pred = final_model.predict(dvalid)
print("Validation AUC:", roc_auc_score(y, y_final_valid_pred))


ORIGINAL_PATH = '/kaggle/input/bank-marketing-dataset-full/'


import pandas as pd

csv_path = ORIGINAL_PATH + "bank-full.csv"
original_df = pd.read_csv(csv_path, sep=";", quotechar='"')
original_df.head()


eda_summary(original_df)


# Take the data through the transformations
original_df = transform_features(original_df, True)
original_df.head()


eda_summary(original_df)


# Prepare the data
X_original = original_df.drop(columns=[TARGET])
y_original = original_df[TARGET].map({"yes": 1, "no": 0}).astype("int8")
doriginal = xgb.DMatrix(X_original, label=y_original, enable_categorical=True)


# Let's see how the fully trained model does on the original dataset
y_original_pred = final_model.predict(doriginal)
print("Validation AUC:", roc_auc_score(y_original, y_original_pred))


dtest = xgb.DMatrix(test_df, enable_categorical=True)
test_preds = final_model.predict(dtest)

submission = pd.DataFrame({"id": test_ids, "y": test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


submission.head()

