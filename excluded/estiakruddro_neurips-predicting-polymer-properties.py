import pandas as pd
import numpy as np
from xgboost import XGBRegressor # Import XGBoost regressor
from sklearn.model_selection import KFold # Added for potential future use with cross-validation


# --- 1. Load the training and test datasets ---
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


# Display initial information (optional, for verification)
print("\nTrain DataFrame head:")
print(train_df.head())
print("\nTrain DataFrame info:")
print(train_df.info())
print("\nTest DataFrame head:")
print(test_df.head())
print("\nTest DataFrame info:")
print(test_df.info())


# --- 2. Feature Engineering from SMILES Strings ---
def extract_smiles_features(smiles):
    """
    Extracts simple features from a SMILES string.
    This function creates rudimentary features based on string properties.
    For a more robust solution, a cheminformatics library like RDKit would be used.
    """
    features = {}

    if pd.isna(smiles):
        # Return default values for NaN SMILES to avoid errors
        return {
            'SMILES_len': 0,
            'C_count': 0, 'N_count': 0, 'O_count': 0, 'F_count': 0,
            'S_count': 0, 'Cl_count': 0, 'Br_count': 0, 'I_count': 0,
            'P_count': 0, 'equal_count': 0, 'hash_count': 0, 'ring_count': 0
        }

    features['SMILES_len'] = len(smiles)

    # Atom counts (simple character counts)
    features['C_count'] = smiles.count('C')
    features['N_count'] = smiles.count('N')
    features['O_count'] = smiles.count('O')
    features['F_count'] = smiles.count('F')
    features['S_count'] = smiles.count('S')
    features['Cl_count'] = smiles.count('Cl')
    features['Br_count'] = smiles.count('Br')
    features['I_count'] = smiles.count('I')
    features['P_count'] = smiles.count('P')

    # Structural feature counts (heuristic)
    features['equal_count'] = smiles.count('=') # Double bonds
    features['hash_count'] = smiles.count('#')   # Triple bonds
    # Simple ring count (counts 'c' for aromatic and '1', '2' for cycle closures)
    features['ring_count'] = smiles.count('c') + smiles.count('1') + smiles.count('2')

    return features

print("\nExtracting features from SMILES strings...")
train_features = train_df['SMILES'].apply(extract_smiles_features)
test_features = test_df['SMILES'].apply(extract_smiles_features)

train_features_df = pd.DataFrame(train_features.tolist())
test_features_df = pd.DataFrame(test_features.tolist())

print("\nTrain features head:")
print(train_features_df.head())
print("\nTest features head:")
print(test_features_df.head())


# --- 3. Define Target Columns and Train Models ---
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Initialize a dictionary to store trained models
models = {}
# Initialize a DataFrame to store test predictions
test_predictions = pd.DataFrame({'id': test_df['id']})

print("\nStarting model training and prediction for each property using XGBoost...")
for target in target_columns:
    print(f"Training XGBoost model for {target}...")

    # Filter out rows where the current target is NaN for training
    train_data_for_target = train_df.dropna(subset=[target])
    
    # Ensure features correspond to the filtered target data
    X_train = train_features_df.loc[train_data_for_target.index]
    y_train = train_data_for_target[target]

    # Initialize and train the XGBoost Regressor
    # Using default parameters for simplicity, can be tuned later
    # n_jobs=-1 uses all available CPU cores.
    model = XGBRegressor(random_state=42, n_jobs=-1) 
    model.fit(X_train, y_train)
    models[target] = model

    # Make predictions on the test features using the trained model
    test_predictions[target] = model.predict(test_features_df)
    print(f"Predictions made for {target}.")

print("\nAll XGBoost models trained and predictions generated.")


# --- 4. Create Submission File ---
submission_file_name = 'submission.csv'
test_predictions.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully.")
print("The first few rows of the generated submission file:")
print(test_predictions.head())







