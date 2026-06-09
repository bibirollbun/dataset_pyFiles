# install rdkit 
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# import
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd
import numpy as np
import os
import warnings
import time
import gc
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# Seaborn
rc = {
    #FAEEE9
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4
}
sns.set(rc=rc)

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RDLogger.DisableLog('rdApp.*')


# Load the data
data_path = "/kaggle/input/neurips-open-polymer-prediction-2025"
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

print("Dataset shapes:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Sample submission: {sample_submission.shape}")

print("\nTrain dataset columns:")
print(train_df.columns.tolist())

print("\nTrain dataset info:")
print(train_df.info())

print("\nFirst 5 rows of train data:")
print(train_df.head())

print("\nTarget columns statistics:")
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
print(train_df[target_cols].describe())

print("\nMissing values in train data:")
print(train_df.isnull().sum())

print("\nTest dataset info:")
print(test_df.info())

print("\nFirst 5 rows of test data:")
print(test_df.head())

print("\nSample submission format:")
print(sample_submission.head())

# Check supplementary data
print("\nSupplementary datasets:")
supplement_path = os.path.join(data_path, "train_supplement")
for file in os.listdir(supplement_path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(supplement_path, file))
        print(f"{file}: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        print()


# 1. Distribution of Target Variables
print("\n--- Distribution of Target Variables ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train_df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}", color="#FFFFFF")
plt.tight_layout()


# 2. Missing Values Visualization
print("\n--- Missing Values Visualization ---")
plt.figure(figsize=(10, 6))
sns.heatmap(train_df[target_cols].isnull(), cbar=True, cmap='crest')
plt.title("Missing Values Heatmap for Target Properties", color="#FFFFFF")


# 3. Basic SMILES Analysis (Length)
print("\n--- SMILES Length Analysis ---")
train_df["SMILES_len"] = train_df["SMILES"].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(train_df["SMILES_len"], kde=True)
plt.title("Distribution of SMILES Length", color="#FFFFFF")


print("SMILES length statistics:")
print(train_df["SMILES_len"].describe())


# 4. Correlation Matrix for Target Variables
print("\n--- Correlation Matrix of Target Variables ---")
plt.figure(figsize=(8, 6))
sns.heatmap(train_df[target_cols].corr(), annot=True, cmap='crest', fmt=".2f")
plt.title("Correlation Matrix of Target Properties", color="#FFFFFF")


# 5. RDKit Descriptors (Example: Molecular Weight and LogP)
print("\n--- RDKit Descriptors Analysis (Example) ---")
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return pd.Series({
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol)
            })
        else:
            return pd.Series({"MW": None, "LogP": None, "NumHDonors": None, "NumHAcceptors": None})
    except:
        return pd.Series({"MW": None, "LogP": None, "NumHDonors": None, "NumHAcceptors": None})

# Apply descriptor calculation to a sample of the data due to potential runtime
# For full EDA, this would be applied to the entire dataset.
print("Calculating RDKit descriptors for a sample of 1000 polymers...")
sample_df = train_df.sample(min(1000, len(train_df)), random_state=42).copy()
descriptor_df = sample_df["SMILES"].apply(calculate_descriptors)
sample_df = pd.concat([sample_df, descriptor_df], axis=1)

sample_df.head()


# Plot distributions of some descriptors
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(sample_df["MW"].dropna(), kde=True)
plt.title("Distribution of Molecular Weight (MW)", color="#FFFFFF")

plt.subplot(1, 3, 2)
sns.histplot(sample_df["LogP"].dropna(), kde=True)
plt.title("Distribution of LogP", color="#FFFFFF")

plt.subplot(1, 3, 3)
sns.histplot(sample_df["NumHDonors"].dropna(), kde=True)
plt.title("Distribution of NumHDonors", color="#FFFFFF")

plt.tight_layout()


# 6. Analyze supplementary datasets
print("\n--- Supplementary Data Analysis ---")
supplement_path = os.path.join(data_path, "train_supplement")
supplementary_files = [f for f in os.listdir(supplement_path) if f.endswith(".csv")]

for file_name in supplementary_files:
    file_path = os.path.join(supplement_path, file_name)
    sup_df = pd.read_csv(file_path)
    print(f"\nAnalysis of {file_name}:")
    print(f"Shape: {sup_df.shape}")
    print(f"Columns: {sup_df.columns.tolist()}")
    print(f"Missing values:\n{sup_df.isnull().sum()}")
    print(f"First 5 rows:\n{sup_df.head()}")

    # If it contains a target column, plot its distribution
    if "TC_mean" in sup_df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(sup_df["TC_mean"].dropna(), kde=True)
        plt.title(f"Distribution of TC_mean in {file_name}", color="#FFFFFF")
        file_name_clean = file_name.replace('.csv', '')
    if "Tg" in sup_df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(sup_df["Tg"].dropna(), kde=True)
        plt.title(f"Distribution of Tg in {file_name}", color="#FFFFFF")
        file_name_clean = file_name.replace('.csv', '')
    if "FFV" in sup_df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(sup_df["FFV"].dropna(), kde=True)
        plt.title(f"Distribution of FFV in {file_name}", color="#FFFFFF")
        file_name_clean = file_name.replace('.csv', '')


print("EDA script finished.")


# Load the data
data_path = "/kaggle/input/neurips-open-polymer-prediction-2025/"
train_df = pd.read_csv(os.path.join(data_path, "train.csv" ))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# Define target columns
target_cols = ["Tg", "FFV", "Tc", "Density", "Rg"]

# Load supplementary data
print("Loading supplementary data...")
supplement_path = os.path.join(data_path, "train_supplement/")

# Original supplementary data (from competition data)
dataset1 = pd.read_csv(os.path.join(supplement_path, "dataset1.csv"))
dataset3 = pd.read_csv(os.path.join(supplement_path, "dataset3.csv"))
dataset4 = pd.read_csv(os.path.join(supplement_path, "dataset4.csv"))

# New supplementary data (from Kaggle datasets)
extra_data_tg3 = pd.read_excel("/kaggle/input/smiles-extra-data/data_tg3.xlsx")
extra_data_dnst1 = pd.read_excel("/kaggle/input/smiles-extra-data/data_dnst1.xlsx")
jcim_sup_bigsmiles = pd.read_csv("/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv")
tc_smiles_df = pd.read_csv("/kaggle/input/tc-smiles/Tc_SMILES.csv")

# --- Data Preparation and Merging ---

# Prepare competition train data
comp_train_df = train_df.copy()

# Prepare supplementary dataframes, ensuring consistent column names and adding dummy IDs

# dataset1 (Tc)
dataset1_processed = dataset1.rename(columns={"TC_mean": "Tc"})
dataset1_processed["id"] = "sup1_" + dataset1_processed.index.astype(str)
for col in target_cols:
    if col not in dataset1_processed.columns:
        dataset1_processed[col] = np.nan

# dataset3 (Tg)
dataset3_processed = dataset3.copy()
dataset3_processed["id"] = "sup3_" + dataset3_processed.index.astype(str)
for col in target_cols:
    if col not in dataset3_processed.columns:
        dataset3_processed[col] = np.nan

# dataset4 (FFV)
dataset4_processed = dataset4.copy()
dataset4_processed["id"] = "sup4_" + dataset4_processed.index.astype(str)
for col in target_cols:
    if col not in dataset4_processed.columns:
        dataset4_processed[col] = np.nan

# extra_data_tg3 (Tg)
extra_data_tg3_processed = extra_data_tg3.rename(columns={"Tg (C)": "Tg"})
extra_data_tg3_processed["id"] = "ext_tg3_" + extra_data_tg3_processed.index.astype(str)
for col in target_cols:
    if col not in extra_data_tg3_processed.columns:
        extra_data_tg3_processed[col] = np.nan

# extra_data_dnst1 (Density)
extra_data_dnst1_processed = extra_data_dnst1.rename(columns={"Density (g/cm^3)": "Density"})
extra_data_dnst1_processed["id"] = "ext_dnst1_" + extra_data_dnst1_processed.index.astype(str)
for col in target_cols:
    if col not in extra_data_dnst1_processed.columns:
        extra_data_dnst1_processed[col] = np.nan

# tc_smiles_df (Tc)
tc_smiles_df_processed = tc_smiles_df.rename(columns={"TC_mean": "Tc"})
tc_smiles_df_processed["id"] = "tc_sml_" + tc_smiles_df_processed.index.astype(str)
for col in target_cols:
    if col not in tc_smiles_df_processed.columns:
        tc_smiles_df_processed[col] = np.nan

# jcim_sup_bigsmiles (SMILES only, for feature engineering)
jcim_smiles_only = jcim_sup_bigsmiles[["SMILES"]].copy()
jcim_smiles_only["id"] = "jcim_" + jcim_smiles_only.index.astype(str)
for col in target_cols:
    jcim_smiles_only[col] = np.nan # No target values

# Combine all training-like dataframes for feature engineering
# This includes competition train, all supplementary data with targets, and jcim_sup_bigsmiles for its SMILES
combined_train_like_df = pd.concat([
    comp_train_df,
    dataset1_processed,
    dataset3_processed,
    dataset4_processed,
    extra_data_tg3_processed,
    extra_data_dnst1_processed,
    tc_smiles_df_processed,
    jcim_smiles_only # Include for its SMILES for feature generation
], ignore_index=True)

# Combine all unique SMILES for feature engineering (from combined_train_like_df and test_df)
all_smiles_for_fe = pd.concat([
    combined_train_like_df[["id", "SMILES"]],
    test_df[["id", "SMILES"]]
], ignore_index=True)

# Drop duplicates based on SMILES to avoid redundant calculations
all_smiles_for_fe.drop_duplicates(subset=["SMILES"], inplace=True)

print("Starting feature engineering from SMILES...")

def smiles_to_advanced_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Define all possible descriptor names to ensure consistent columns
    descriptor_names = [
        "MW", "LogP", "NumHDonors", "NumHAcceptors", "TPSA",
        "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
        "NumSaturatedRings", "NumHeteroatoms", "FractionCSP3",
        "HeavyAtomCount", "NHOHCount", "NOCount", "RingCount", "MolMR"
    ]
    # Add Morgan Fingerprint names (2048 bits)
    for i in range(2048):
        descriptor_names.append(f"MorganFP_{i}")

    if mol is None:
        # Return a Series of NaNs for all expected descriptor columns
        return pd.Series({name: np.nan for name in descriptor_names})
    
    descriptors = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NHOHCount": Descriptors.NHOHCount(mol),
        "NOCount": Descriptors.NOCount(mol),
        "RingCount": Descriptors.RingCount(mol),
        "MolMR": Descriptors.MolMR(mol),
    }
    
    # Add Morgan Fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    for i in range(fp.GetNumBits()):
        descriptors[f"MorganFP_{i}"] = fp[i]

    return pd.Series(descriptors)

# Apply feature engineering to all SMILES strings in chunks
# This helps manage memory usage for large datasets
chunk_size = 500 # Adjusted chunk size
all_smiles_features_list = []

for i in range(0, len(all_smiles_for_fe), chunk_size):
    chunk = all_smiles_for_fe.iloc[i:i + chunk_size]
    chunk_features = chunk["SMILES"].apply(smiles_to_advanced_descriptors)
    all_smiles_features_list.append(chunk_features)

all_smiles_features = pd.concat(all_smiles_features_list, ignore_index=True)
all_smiles_for_fe = pd.concat([all_smiles_for_fe.reset_index(drop=True), all_smiles_features], axis=1)

# Drop rows where feature engineering failed (i.e., all descriptors are NaN)
all_smiles_for_fe.dropna(subset=all_smiles_features.columns, how='all', inplace=True)

# Additional simple features
all_smiles_for_fe["SMILES_len"] = all_smiles_for_fe["SMILES"].apply(len)
all_smiles_for_fe["num_C"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("C"))
all_smiles_for_fe["num_O"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("O"))
all_smiles_for_fe["num_N"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("N"))

print("Advanced feature engineering complete. Combined dataframe shape:", all_smiles_for_fe.shape)

# Merge features back to the combined_train_like_df and test_df based on SMILES
# Use a left merge to keep all original train/test entries
processed_train_df = pd.merge(combined_train_like_df, all_smiles_for_fe.drop(columns=["id"]), on="SMILES", how="left")
processed_test_df = pd.merge(test_df, all_smiles_for_fe.drop(columns=["id"]), on="SMILES", how="left")

# Identify columns present in processed_train_df but not in processed_test_df
# Exclude target columns from being dropped from processed_train_df
missing_in_test = set(processed_train_df.columns) - set(processed_test_df.columns)
columns_to_drop_from_train = [col for col in missing_in_test if col not in target_cols]

# Drop these columns from processed_train_df to ensure consistency
processed_train_df = processed_train_df.drop(columns=columns_to_drop_from_train)

print("Processed train shape:", processed_train_df.shape)
print("Processed test shape:", processed_test_df.shape)

# Save processed data (optional, for debugging/next steps)
#processed_train_df.to_csv("processed_train_advanced.csv", index=False)
#processed_test_df.to_csv("processed_test_advanced.csv", index=False)
print("Processed data saved to processed_train_advanced.csv and processed_test_advanced.csv")

print("Preprocessing and feature engineering complete.")




import optuna
# --------------------------- CONFIG ---------------------------
class CFG:
    # General
    TARGET_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]
    N_FOLDS = 5
    SEED = 42
    MODELS_TO_RUN = ["lgbm", "xgb", "catboost"]

    # Feature Engineering & Selection
    N_FEATURES_TO_SELECT = 100 # Select top N features for each target

    # Hyperparameter Tuning
    USE_HYPERTUNING = True  # Set to False to use default params
    N_TRIALS = 1  # Number of optimization trials per model
    TUNING_TIMEOUT = 100  # INCREASE THIS: Timeout in seconds (recommend 1 hour)
    
    # Default Model Parameters (used if hypertuning is disabled)
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "num_leaves": 128,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED,
        "boosting_type": "gbdt",
    }

    XGB_PARAMS = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "verbose": 0,
        "tree_method": "hist",
        "seed": SEED,
    }

    CATBOOST_PARAMS = {
        "objective": "MAE",
        "eval_metric": "MAE",
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 0.1,
        "random_strength": 0.1,
        "loss_function": "MAE",
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1,
    }

# -------------------------- UTILITY FUNCTIONS --------------------------

def create_stratified_folds(df, target_col, n_splits, seed):
    """Create stratified folds for regression by binning the target."""
    df['bins'] = pd.cut(df[target_col], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X=df, y=df['bins']))
    df = df.drop('bins', axis=1)
    return folds

def select_features(train_df, test_df, features, target_col, n_features):
    """Select top features using LightGBM feature importance."""
    print(f"  Performing feature selection for {target_col}...")
    temp_model = lgb.LGBMRegressor(**CFG.LGBM_PARAMS)
    temp_model.fit(train_df[features], train_df[target_col])
    importances = pd.DataFrame({
        'feature': features,
        'importance': temp_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importances['feature'].head(n_features).tolist()
    print(f"  Selected {len(top_features)} features.")
    return top_features

# -------------------------- HYPERPARAMETER TUNING --------------------------

def suggest_lgbm_params(trial):
    """Suggest hyperparameters for LightGBM."""
    return {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0, log=True),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
        "seed": CFG.SEED,
    }

def suggest_xgb_params(trial):
    """Suggest hyperparameters for XGBoost."""
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": -1,
        "verbose": 0,
        "tree_method": "hist",
        "seed": CFG.SEED,
    }

def suggest_catboost_params(trial):
    """Suggest hyperparameters for CatBoost."""
    return {
        "objective": "MAE",
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": CFG.SEED,
        "verbose": 0,
        "thread_count": -1,
    }

def objective_function(trial, model_name, X_train, y_train, folds):
    """Objective function for hyperparameter optimization."""
    
    # Get model parameters based on model type
    if model_name == 'lgbm':
        params = suggest_lgbm_params(trial)
        model_class = lgb.LGBMRegressor
    elif model_name == 'xgb':
        params = suggest_xgb_params(trial)
        model_class = xgb.XGBRegressor
    elif model_name == 'catboost':
        params = suggest_catboost_params(trial)
        model_class = cat.CatBoostRegressor
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create model
        model = model_class(**params)
        
        # Create pipeline
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Prepare validation data for early stopping
        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()
        X_tr_processed = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val_processed = scaler.transform(imputer.transform(X_val))
        
        # Set up fit parameters
        if model_name == 'lgbm':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__callbacks': [lgb.early_stopping(100, verbose=False)]
            }
        elif model_name == 'xgb':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__callbacks': [xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                'model__verbose': False
            }
        elif model_name == 'catboost':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__early_stopping_rounds': 100
            }
        
        # Fit and predict
        pipeline.fit(X_tr, y_tr, **fit_params)
        pred = pipeline.predict(X_val)
        score = mean_absolute_error(y_val, pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

def tune_hyperparameters(model_name, X_train, y_train, folds, n_trials=500, timeout=100):
    """Tune hyperparameters using Optuna."""
    print(f"    Tuning hyperparameters for {model_name.upper()}...")
    
    study = optuna.create_study(direction='minimize', 
                               sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    
    study.optimize(
        lambda trial: objective_function(trial, model_name, X_train, y_train, folds),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    print(f"    Best {model_name.upper()} score: {study.best_value:.6f}")
    print(f"    Best {model_name.upper()} params: {study.best_params}")
    
    return study.best_params

# -------------------------- MODELING PIPELINE --------------------------

print("Starting enhanced model training with hyperparameter tuning...\n")
start_time = time.time()

test_predictions = pd.DataFrame({"id": processed_test_df["id"]})
oof_predictions = {}
best_params_per_target = {}

initial_features = [
    col for col in processed_train_df.columns
    if col not in ["id", "SMILES"] + CFG.TARGET_COLS and col in processed_test_df.columns
]

for target in CFG.TARGET_COLS:
    print(f"--- Training models for {target} ---")
    
    # Prepare data for the current target
    train_df = processed_train_df.dropna(subset=[target]).reset_index(drop=True)
    y_train = train_df[target]
    
    # Dynamic Feature Selection
    selected_features = select_features(train_df, processed_test_df, initial_features, target, CFG.N_FEATURES_TO_SELECT)
    X_train = train_df[selected_features]
    X_test = processed_test_df[selected_features]
    
    # Create stratified folds for this target
    folds = create_stratified_folds(train_df, target, CFG.N_FOLDS, CFG.SEED)
    
    # Store best parameters for this target
    best_params_per_target[target] = {}
    
    # Hyperparameter tuning phase
    if CFG.USE_HYPERTUNING:
        print(f"\n  === HYPERPARAMETER TUNING PHASE ===")
        for model_name in CFG.MODELS_TO_RUN:
            best_params = tune_hyperparameters(
                model_name, X_train, y_train, folds, 
                CFG.N_TRIALS, CFG.TUNING_TIMEOUT
            )
            best_params_per_target[target][model_name] = best_params
    
    # Training phase with best parameters
    print(f"\n  === TRAINING PHASE ===")
    oof_preds_ensemble = np.zeros((len(X_train), len(CFG.MODELS_TO_RUN)))
    test_preds_ensemble = np.zeros((len(X_test), len(CFG.MODELS_TO_RUN)))
    
    for model_idx, model_name in enumerate(CFG.MODELS_TO_RUN):
        print(f"\n  Training {model_name.upper()} model...")
        
        # Get parameters (tuned or default)
        if CFG.USE_HYPERTUNING:
            if model_name == 'lgbm':
                model_params = best_params_per_target[target][model_name]

            elif model_name == 'xgb':
                model_params = best_params_per_target[target][model_name]

            elif model_name == 'catboost':
                model_params = best_params_per_target[target][model_name]

        else:
            if model_name == 'lgbm':
                model_params = CFG.LGBM_PARAMS
            elif model_name == 'xgb':
                model_params = CFG.XGB_PARAMS
            elif model_name == 'catboost':
                model_params = CFG.CATBOOST_PARAMS
    
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"    Fold {fold + 1}/{CFG.N_FOLDS}")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create model with best parameters
            if model_name == 'lgbm':
                model = lgb.LGBMRegressor(**model_params)
            elif model_name == 'xgb':
                model = xgb.XGBRegressor(**model_params)
            elif model_name == 'catboost':
                model = cat.CatBoostRegressor(**model_params)
            
            # Create pipeline
            pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Prepare validation data for early stopping
            imputer = KNNImputer(n_neighbors=5)
            scaler = StandardScaler()
            X_tr_processed = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_val_processed = scaler.transform(imputer.transform(X_val))
            
            # Set up fit parameters
            if model_name == 'lgbm':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__callbacks': [lgb.early_stopping(100, verbose=False)]
                }
            elif model_name == 'xgb':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__callbacks': [xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                    'model__verbose': False
                }
            elif model_name == 'catboost':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__early_stopping_rounds': 100,
                }
            
            # Fit and predict
            pipeline.fit(X_tr, y_tr, **fit_params)
            oof_preds_ensemble[val_idx, model_idx] = pipeline.predict(X_val)
            test_preds_ensemble[:, model_idx] += pipeline.predict(X_test) / CFG.N_FOLDS
    
    # Average the predictions from all models (simple ensemble)
    final_oof = np.mean(oof_preds_ensemble, axis=1)
    final_preds = np.mean(test_preds_ensemble, axis=1)
    
    test_predictions[target] = final_preds
    oof_predictions[target] = final_oof
    print(f"\nEnsemble MAE for {target}: {mean_absolute_error(y_train, final_oof):.4f}\n")
    gc.collect()

# -------------------------- SUBMISSION & SUMMARY --------------------------

# Save best parameters
#if CFG.USE_HYPERTUNING:
    # import json
    # with open('best_hyperparameters.json', 'w') as f:
    #     json.dump(best_params_per_target, f, indent=2)
    # print("Best hyperparameters saved to 'best_hyperparameters.json'")

submission_df = test_predictions.copy()
if "Tg" in submission_df.columns:
    submission_df["Tg"] += 273.15 # Postprocess Tg to Kelvin
    submission_df['FFV'] **= 2
submission_df.to_csv("submission.csv", index=False)
print("Submission saved as 'submission.csv'.")

print("\n--- OOF MAE Summary (Ensembled with Hypertuning) ---")
for target, oof in oof_predictions.items():
    y_true = processed_train_df.dropna(subset=[target])[target]
    print(f"{target}: MAE = {mean_absolute_error(y_true, oof):.4f}")

print(f"\nTotal training time: {time.time() - start_time:.2f} seconds.")


# Check out predictions
test_predictions

