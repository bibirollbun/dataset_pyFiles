!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd
import numpy as np
import os
import warnings
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# Set display options and suppress all warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RDLogger.DisableLog('rdApp.*')

# Load the data
data_path = "/kaggle/input/neurips-open-polymer-prediction-2025/"
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
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
combined_train_like_df = pd.concat([
    comp_train_df,
    dataset1_processed,
    dataset3_processed,
    dataset4_processed,
    extra_data_tg3_processed,
    extra_data_dnst1_processed,
    tc_smiles_df_processed,
    jcim_smiles_only
], ignore_index=True)

# Combine all unique SMILES for feature engineering
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
chunk_size = 500
all_smiles_features_list = []

for i in range(0, len(all_smiles_for_fe), chunk_size):
    chunk = all_smiles_for_fe.iloc[i:i + chunk_size]
    chunk_features = chunk["SMILES"].apply(smiles_to_advanced_descriptors)
    all_smiles_features_list.append(chunk_features)

all_smiles_features = pd.concat(all_smiles_features_list, ignore_index=True)
all_smiles_for_fe = pd.concat([all_smiles_for_fe.reset_index(drop=True), all_smiles_features], axis=1)

# Drop rows where feature engineering failed
all_smiles_for_fe.dropna(subset=all_smiles_features.columns, how='all', inplace=True)

# Additional simple features
all_smiles_for_fe["SMILES_len"] = all_smiles_for_fe["SMILES"].apply(len)
all_smiles_for_fe["num_C"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("C"))
all_smiles_for_fe["num_O"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("O"))
all_smiles_for_fe["num_N"] = all_smiles_for_fe["SMILES"].apply(lambda x: x.count("N"))

print("Advanced feature engineering complete. Combined dataframe shape:", all_smiles_for_fe.shape)

# Merge features back to the combined_train_like_df and test_df based on SMILES
processed_train_df = pd.merge(combined_train_like_df, all_smiles_for_fe.drop(columns=["id"]), on="SMILES", how="left")
processed_test_df = pd.merge(test_df, all_smiles_for_fe.drop(columns=["id"]), on="SMILES", how="left")

# Identify columns present in processed_train_df but not in processed_test_df
missing_in_test = set(processed_train_df.columns) - set(processed_test_df.columns)
columns_to_drop_from_train = [col for col in missing_in_test if col not in target_cols]

# Drop these columns from processed_train_df to ensure consistency
processed_train_df = processed_train_df.drop(columns=columns_to_drop_from_train)

print("Processed train shape:", processed_train_df.shape)
print("Processed test shape:", processed_test_df.shape)
print("Preprocessing and feature engineering complete.")

# --- Model Training ---

# Features to use for modeling
all_possible_features = [col for col in processed_train_df.columns if col not in ["id", "SMILES"] + target_cols]
features = [col for col in all_possible_features if col in processed_test_df.columns]

# Initialize dictionary to store predictions
test_predictions = pd.DataFrame({"id": processed_test_df["id"]})

# Store OOF predictions for wMAE calculation
oofs = {}

print("Starting model training...")

# Train a separate XGBoost model for each target property
for target in target_cols:
    print(f"\n--- Training model for {target} ---")
    
    # Filter out rows where the current target is NaN for training
    train_target_df = processed_train_df.dropna(subset=[target]).copy()
    
    X = train_target_df[features]
    y = train_target_df[target]
    
    # Align columns
    X_test = processed_test_df[features]
    
    # Ensure all features are numeric and handle potential inf/-inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Impute NaNs in features with mean
    for col in X.columns:
        if X[col].isnull().any():
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)
            X_test[col].fillna(mean_val, inplace=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    fold_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        xgb_params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbosity": 0
        }
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val)
        fold_preds.append(model.predict(X_test))
        
    test_predictions[target] = np.mean(fold_preds, axis=0)
    oofs[target] = oof_preds
    print(f"MAE for {target}: {mean_absolute_error(y, oof_preds):.4f}")

print("Model training complete.")

# Create submission file
submission_df = test_predictions.copy()
submission_df['Tg'] += 273.15
submission_df['FFV']**=2
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

# Print OOF MAE for each target
print("\n--- OOF MAE for each target ---")
for target, oof_pred in oofs.items():
    y_true_for_oof = processed_train_df.dropna(subset=[target])[target]
    print(f"OOF MAE for {target}: {mean_absolute_error(y_true_for_oof, oof_pred):.4f}")

print("Note: Full wMAE calculation requires competition-specific reweighting factors from the hidden test set.")

# Display final predictions
print("\nFinal test predictions:")
print(test_predictions.head())

