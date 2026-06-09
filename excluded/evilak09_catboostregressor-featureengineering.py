!pip install /kaggle/input/packages/pkgs/wheels/rdkit-2025.3.2-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, MolSurf, Lipinski
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
train.head()


tg = train[["SMILES", "Tg"]]
ffv = train[["SMILES", "FFV"]]
tc = train[["SMILES", "Tc"]]
density =train[["SMILES", "Density"]]
rg = train[["SMILES", "Rg"]]


# RDKit physicochemical properties
def rdkit_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * 8
        return [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            MolSurf.TPSA(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Crippen.MolMR(mol)
        ]
    except:
        return [np.nan] * 8


# Morgan fingerprint (convert bit vector properly)
def morgan_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(list(fp))  # convert bit vector to list fir


# MACCS keys (convert bit vector properly)
def maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(fp))


def generate_features_stepwise(df, smiles_col='SMILES'):
    df = df.copy()

    # Step 1: RDKit physicochemical properties
    property_names = ['MolWt', 'LogP', 'TPSA', 'Rings', 'HDonors', 'HAcceptors', 'RotatableBonds', 'MolMR']
    rdkit_props = df[smiles_col].apply(rdkit_properties).apply(pd.Series)
    rdkit_props.columns = property_names
    rdkit_props.index = df.index
    df = pd.concat([df, rdkit_props], axis=1)


    # # Step 2: Morgan fingerprint
    # nBits = 1024
    # morgan_features = df[smiles_col].apply(morgan_fp)
    # # Convert series of arrays into dataframe
    # morgan_df = pd.DataFrame(morgan_features.tolist(), columns=[f"morgan_{i}" for i in range(nBits)], index=df.index)
    # df = pd.concat([df, morgan_df], axis=1)


    # Step 3: MACCS keys
    maccs_features = df[smiles_col].apply(maccs_fp)
    maccs_df = pd.DataFrame(maccs_features.tolist(), columns=[f"maccs_{i}" for i in range(167)], index=df.index)
    df = pd.concat([df, maccs_df], axis=1)

    return df


import rdkit
from rdkit import Chem

def train_and_blend(X, y, clip_threshold=1e6):
    # Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]

    # Clean data
    if isinstance(X, pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X.clip(lower=-clip_threshold, upper=clip_threshold)
    else:
        X = np.nan_to_num(X, nan=0.0, posinf=clip_threshold, neginf=-clip_threshold)
        X = np.clip(X, -clip_threshold, clip_threshold)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nAfter cleaning:")
    print(f"  X_train any NaN? {np.any(np.isnan(X_train))}")
    print(f"  X_train any Inf? {np.any(np.isinf(X_train))}")
    print(f"  X_train max value: {np.nanmax(X_train)}")
    print(f"  X_train min value: {np.nanmin(X_train)}\n")

    model = CatBoostRegressor(depth=8, learning_rate=0.05, iterations=2000, verbose=0, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    print("R2 score: ", r2_score(y_val, preds))
    print("MAE: ", mean_absolute_error(y_val, preds))
    print("MSE: ", mean_squared_error(y_val, preds))

    return model

def predict():
    test = tg
    train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")

    # Keep only the SMILES column in the test as the target Tg may not be present there
    test = test[['SMILES']]

    print(f"train shape: {train.shape}\t\t test shape: {test.shape}")

    # Generate features for both train and test
    feature_train = generate_features_stepwise(train)
    feature_test = generate_features_stepwise(test)

    # Extract target variable (assuming 'Tg' is the target column)
    y_train = feature_train['Tg'].values

    # Drop columns that should not be used as features
    # Usually, drop SMILES and the target column from features
    drop_cols = ['SMILES', 'Tg']
    X_train = feature_train.drop(columns=drop_cols, errors='ignore')
    X_test = feature_test.drop(columns=['SMILES'], errors='ignore')

    print(f"Feature train shape: {X_train.shape}")
    print(f"Feature test shape: {X_test.shape}")

    # Train model
    model = train_and_blend(X_train, y_train)

    # Clean test data same way as train
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict on test
    preds_test = model.predict(X_test_clean)

    # Add predictions to test dataframe
    test['Tg'] = preds_test

    # Print some predictions
    print(test[['SMILES', 'Tg']].head())

    # Optionally save to CSV
    test.to_csv("predicted_test_results.csv", index=False)
    print("Predictions saved to tgp.csv")

predict()


def generate_features_stepwise(df, smiles_col='SMILES'):
    df = df.copy()

    # Step 1: RDKit physicochemical properties
    property_names = ['MolWt', 'LogP', 'TPSA', 'Rings', 'HDonors', 'HAcceptors', 'RotatableBonds', 'MolMR']
    rdkit_props = df[smiles_col].apply(rdkit_properties).apply(pd.Series)
    rdkit_props.columns = property_names
    rdkit_props.index = df.index
    df = pd.concat([df, rdkit_props], axis=1)


    # Step 2: Morgan fingerprint
    nBits = 1024
    morgan_features = df[smiles_col].apply(morgan_fp)
    # Convert series of arrays into dataframe
    morgan_df = pd.DataFrame(morgan_features.tolist(), columns=[f"morgan_{i}" for i in range(nBits)], index=df.index)
    df = pd.concat([df, morgan_df], axis=1)

    # Step 3: MACCS keys
    maccs_features = df[smiles_col].apply(maccs_fp)
    maccs_df = pd.DataFrame(maccs_features.tolist(), columns=[f"maccs_{i}" for i in range(167)], index=df.index)
    df = pd.concat([df, maccs_df], axis=1)
    return df


def evaluate_model(model, X, y, target_name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    joblib.dump(model, f"model_{target_name}.pkl")
    preds = model.predict(X_val)

    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"--- {target_name} ---")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}\n")

    return model, preds, {"R2": r2, "MAE": mae, "RMSE": rmse}


# --- Main prediction pipeline ---
def predict_test():
    test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

    # Feature extraction
    test_with_feature = generate_features_stepwise(test)

    # Define targets and drop columns
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    drop_col = ['id', 'SMILES']

    # Store predictions
    predictions = {'id': test_with_feature['id']}

    # Prepare test feature matrix
    X_test = test_with_feature.drop(columns=targets + drop_col, errors='ignore')

    for target in targets:
        print(f"Predicting target: {target}")

        # Load trained model
        model_filename = f"model_{target}.pkl"
        model = joblib.load(model_filename)

        # Predict
        preds = model.predict(X_test)

        # Store
        predictions[target] = preds

    # Create DataFrame from predictions
    preds_df = pd.DataFrame(predictions)
    print("\nSample predictions:")
    print(preds_df.head())

    # Optionally save to CSV
    preds_df.to_csv("submission.csv", index=False)


ffv.head()



def main():
    global ffv, tc, density, rg
    # ==================== Load datasets ====================
    tg = pd.read_csv("/kaggle/working/predicted_test_results.csv")
    ffv = ffv
    tc = tc
    density = density
    rg = rg
    test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

    # Drop missing targets
    tg = tg.dropna(subset=['Tg'])
    ffv = ffv.dropna(subset=['FFV'])
    tc = tc.dropna(subset=['Tc'])
    density = density.dropna(subset=['Density'])
    rg = rg.dropna(subset=['Rg'])

    # ==================== Generate Features ====================
    feature_tg = generate_features_stepwise(tg)
    feature_ffv = generate_features_stepwise(ffv)
    feature_tc = generate_features_stepwise(tc)
    feature_density = generate_features_stepwise(density)
    feature_rg = generate_features_stepwise(rg)

    # ==================== Prepare X, y ====================
    X_tg = feature_tg.drop(columns=["SMILES", "Tg"], errors='ignore')
    y_tg = feature_tg["Tg"]

    X_ffv = feature_ffv.drop(columns=["SMILES", "FFV"], errors='ignore')
    y_ffv = feature_ffv["FFV"]

    X_tc = feature_tc.drop(columns=["SMILES", "Tc"], errors='ignore')
    y_tc = feature_tc["Tc"]

    X_density = feature_density.drop(columns=["SMILES", "Density"], errors='ignore')
    y_density = feature_density["Density"]

    X_rg = feature_rg.drop(columns=["SMILES", "Rg"], errors='ignore')
    y_rg = feature_rg["Rg"]

    # ==================== Train Models ====================
    # Tg: Best with CatBoost
    tg_model = CatBoostRegressor(depth=8, learning_rate=0.05, iterations=2000, verbose=0, random_state=42)
    evaluate_model(tg_model, X_tg, y_tg, "Tg")

    # FFV: Best with Stacking
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import StackingRegressor
    ffv_base_models = [
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)),
        ('lgbm', LGBMRegressor(num_leaves=64, learning_rate=0.05, n_estimators=1000, random_state=42)),
        ('catboost', CatBoostRegressor(depth=8, learning_rate=0.05, iterations=1000, verbose=0, random_state=42))
    ]
    ffv_model = StackingRegressor(estimators=ffv_base_models, final_estimator=RidgeCV())
    evaluate_model(ffv_model, X_ffv, y_ffv, "FFV")

    # Tc: Also best with stacking
    tc_base_models = ffv_base_models  # reuse
    tc_model = StackingRegressor(estimators=tc_base_models, final_estimator=RidgeCV())
    evaluate_model(tc_model, X_tc, y_tc, "Tc")

    # Density: LightGBM
    density_model = LGBMRegressor(num_leaves=64, learning_rate=0.05, n_estimators=1000, random_state=42)
    evaluate_model(density_model, X_density, y_density, "Density")

    # Rg: Random Forest
    rg_model = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=42, n_jobs=-1)
    evaluate_model(rg_model, X_rg, y_rg, "Rg")



import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, MolSurf, Lipinski
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib



main()


predict_test()




