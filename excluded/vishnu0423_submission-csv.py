# Polymer Property Prediction Competition - Final Working Solution
# Handles missing values and generates submission.csv

# Install required packages
!pip install rdkit scikit-learn pandas numpy --quiet

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings("ignore")

# Competition constants
INPUT_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025"
WORKING_DIR = "/kaggle/working"
PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
SUBMISSION_FILE = "submission.csv"

def load_data(filename):
    """Load and preprocess competition data"""
    try:
        path = os.path.join(INPUT_DIR, filename)
        df = pd.read_csv(path)
        
        # Convert SMILES to fingerprints
        df['features'] = df['SMILES'].apply(
            lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(str(x)), 
                radius=2, 
                nBits=2048
            )) if pd.notnull(x) else None
        )
        
        # Remove invalid entries
        df = df.dropna(subset=['features'])
        print(f"Successfully loaded {filename}")
        return df
    
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None

def train_models(train_df):
    """Train models with missing value handling"""
    X = np.stack(train_df['features'].values)
    models = {}
    imputers = {}
    
    for prop in PROPERTIES:
        print(f"\nTraining {prop} model...")
        y = train_df[prop].values.reshape(-1, 1)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        y_imputed = imputer.fit_transform(y).ravel()
        imputers[prop] = imputer
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y_imputed)
        models[prop] = model
    
    return models, imputers

def generate_submission(models, imputers, test_df):
    """Create competition submission file"""
    X_test = np.stack(test_df['features'].values)
    submission = test_df[['id']].copy()
    
    for prop in PROPERTIES:
        # Predict and ensure no NaN values
        preds = models[prop].predict(X_test)
        preds = imputers[prop].transform(preds.reshape(-1, 1)).ravel()
        submission[prop] = preds
    
    # Save to required location
    submission_path = os.path.join(WORKING_DIR, SUBMISSION_FILE)
    submission.to_csv(submission_path, index=False, float_format='%.4f')
    print(f"\nSubmission file created at {submission_path}")
    return submission

# Main execution
print("=== Polymer Property Prediction Competition ===")
print("Loading data...")

train_df = load_data("train.csv")
test_df = load_data("test.csv")

if train_df is not None and test_df is not None:
    # Check for missing values
    print("\nMissing values in training data:")
    print(train_df[PROPERTIES].isnull().sum())
    
    print("\nTraining models...")
    models, imputers = train_models(train_df)
    
    print("\nGenerating submission...")
    submission = generate_submission(models, imputers, test_df)
    
    print("\nSubmission preview:")
    print(submission.head())
    
    # Verify file creation
    if os.path.exists(os.path.join(WORKING_DIR, SUBMISSION_FILE)):
        print("\nVERIFICATION: submission.csv successfully created")
    else:
        print("\nERROR: Failed to create submission file")
else:
    print("\nData loading failed. Please check:")
    print(f"- Competition dataset is added")
    print(f"- Files exist in {INPUT_DIR}")
    print(f"- Directory contains: {os.listdir(INPUT_DIR)}")

# Final verification
print("\nContents of /kaggle/working:")
!ls /kaggle/working


# Install required packages
!pip install rdkit pandas numpy scikit-learn xgboost torch torch-geometric matplotlib seaborn shap

