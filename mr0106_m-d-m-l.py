# -*- coding: utf-8 -*-
"""
Molecular Machine Learning: Predicting Molecular Properties
"""

# ======================
# Required Libraries
# ======================
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import sys
import subprocess
import importlib

# ======================
# RDKit Installation Check
# ======================
def install_and_import(package, import_name=None):
    """Install and import a package with retry logic"""
    if import_name is None:
        import_name = package
        
    for _ in range(2):  # Try twice
        try:
            module = importlib.import_module(import_name)
            return module
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    raise ImportError(f"Failed to install and import {package}")

try:
    # First try standard installation
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
except ImportError:
    # If standard import fails, try alternative installation methods
    try:
        # Try installing via conda if available
        print("Attempting conda installation...")
        subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "rdkit", "-y"])
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    except:
        # Fall back to pip with specific version
        print("Falling back to pip installation...")
        install_and_import("rdkit-pypi", "rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Verify RDKit installation
try:
    print(f"RDKit version: {Chem.rdBase.rdkitVersion}")
except:
    raise ImportError("RDKit installation failed - please install manually: \n"
                     "conda install -c conda-forge rdkit\n"
                     "or\n"
                     "pip install rdkit-pypi")

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# ======================
# Rest of your code continues...
# (The data loading, feature generation, modeling, etc. from previous solution)
# ======================


# ======================
# Data Loading
# ======================
def load_data(train_path, test_path, sample_sub_path):
    """Load training, test and sample submission data"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_sub_path)
    
    print("\nFirst 3 rows of training data:")
    display(train.head(3))
    
    print("\nFirst 3 rows of test data:")
    display(test.head(3))
    
    return train, test, sub

# File paths (modify as needed)
TRAIN_PATH = '/kaggle/input/molecular-machine-learning/train.csv'
TEST_PATH = '/kaggle/input/molecular-machine-learning/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/molecular-machine-learning/sample_submission.csv'

train, test, sub = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH)


# ======================
# RDKit Feature Generation
# ======================
def generate_rdkit_features(smiles_series):
    """
    Generate molecular features using RDKit
    
    Args:
        smiles_series: Pandas series containing SMILES strings
        
    Returns:
        numpy array containing computed features
    """
    num_samples = len(smiles_series)
    rdkit_props = np.zeros((num_samples, 10), dtype=float)
    
    for idx, smile in enumerate(smiles_series):
        try:
            # Convert SMILES to molecule with hydrogens
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError(f"Failed to parse SMILES: {smile}")
                
            mol3d = Chem.AddHs(mol)
            
            # Generate 3D structure
            AllChem.EmbedMolecule(mol3d, randomSeed=random.randint(1, 1000000))
            
            # Optimize structure with force field
            AllChem.MMFFOptimizeMolecule(mol3d)
            
            # Calculate molecular properties
            rdkit_props[idx, 0] = Descriptors.MolWt(mol3d)  # Molecular weight
            rdkit_props[idx, 1] = rdMolDescriptors.CalcNumHBA(mol3d)  # Hydrogen bond acceptors
            rdkit_props[idx, 2] = rdMolDescriptors.CalcNumHBD(mol3d)  # Hydrogen bond donors
            rdkit_props[idx, 3] = Descriptors.MolLogP(mol3d)  # LogP
            rdkit_props[idx, 4] = rdMolDescriptors.CalcAsphericity(mol3d)  # Asphericity
            rdkit_props[idx, 5] = rdMolDescriptors.CalcRadiusOfGyration(mol3d)  # Radius of gyration
            rdkit_props[idx, 6] = Descriptors.TPSA(mol3d)  # Polar surface area
            rdkit_props[idx, 7] = rdMolDescriptors.CalcNumRings(mol3d)  # Number of rings
            rdkit_props[idx, 8] = rdMolDescriptors.CalcNumRotatableBonds(mol3d)  # Rotatable bonds
            rdkit_props[idx, 9] = rdMolDescriptors.CalcNumHeteroatoms(mol3d)  # Heteroatoms
            
        except Exception as e:
            print(f"Error processing molecule {idx}: {str(e)}")
            rdkit_props[idx, :] = np.nan
    
    return rdkit_props

def add_rdkit_features(df):
    """Add RDKit features to DataFrame"""
    rdkit_features = generate_rdkit_features(df['Smiles'])
    feature_names = [
        'RDKit_MolWt', 'RDKit_HBA', 'RDKit_HBD', 'RDKit_LogP',
        'RDKit_Asphericity', 'RDKit_Rg', 'RDKit_TPSA',
        'RDKit_NumRings', 'RDKit_RotatableBonds', 'RDKit_Heteroatoms'
    ]
    
    rdkit_df = pd.DataFrame(rdkit_features, columns=feature_names)
    return pd.concat([df, rdkit_df], axis=1)

# Add features to training and test data
print("\nGenerating RDKit features for training data...")
train = add_rdkit_features(train)

print("\nGenerating RDKit features for test data...")
test = add_rdkit_features(test)

# Show training data after feature addition
print("\nTraining data with RDKit features:")
display(train.head(3))


# ======================
# Modeling and Prediction
# ======================
def custom_regression_model(X):
    """
    Custom regression model for molecular property prediction
    
    Args:
        X: DataFrame containing features
        
    Returns:
        numpy array containing predictions
    """
    t = X['T2'] * X['TDOS3.2'] * X['LUMOp1(eV)']
    
    part1 = (X['T8'] - t) * (1.7020332667085079 * X['SDOS3.9'])
    
    denominator = (X['T14'] / X['TDOS3.6']) + (0.8998248512734 - X['RDKit_RotatableBonds'])
    numerator = X['TDOS3.7'] - X['SDOS3.8'] - X['TDOS4.3'] - X['TDOS2.7']
    
    part2_1 = numerator / denominator
    part2_2 = (X['LUMO(eV)'] + t) * ((X['LUMO(eV)'] + t) - X['TDOS3.2'])
    
    result = part1 + (part2_1 + part2_2)
    return result

# Prepare data for modeling
X_train = train.iloc[:, 3:]
y_train = train['T80']
X_test = test.iloc[:, 3:]

# Make predictions
print("\nMaking predictions...")
train_pred = custom_regression_model(X_train)
test_pred = custom_regression_model(X_test)


# ======================
# Model Evaluation
# ======================
def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    scores = []
    for i in range(len(y_true)):
        scores.append(mean_squared_log_error(y_true[i:i+1], y_pred[i:i+1]))
    
    # Plot evaluation results
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='MSLE per sample')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Acceptance threshold (MSLE=0.01)')
    plt.title('Model Performance on Training Samples')
    plt.xlabel('Sample Number')
    plt.ylabel('Mean Squared Logarithmic Error')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Count good predictions
    good_predictions = np.isclose(scores, 0, atol=1e-2).sum()
    print(f"\nNumber of good predictions (MSLE < 0.01): {good_predictions} out of {len(y_true)}")
    
    # Evaluate on last two samples
    last_two_msle = mean_squared_log_error(y_true[-2:], y_pred[-2:])
    print(f"MSLE for last two samples: {last_two_msle:.6f}")

# Evaluate model
print("\nEvaluating model performance on training data...")
evaluate_model(y_train, train_pred)



# ======================
# Submission Preparation
# ======================
def prepare_submission(test_df, predictions, sample_sub):
    """Prepare competition submission file"""
    # Ensure non-negative values
    predictions = np.clip(predictions, 0, None)
    
    submission = pd.DataFrame({
        'Batch_ID': test_df['Batch_ID'],
        'T80': predictions
    })
    
    # Save file
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file saved successfully!")
    
    return submission

print("\nPreparing submission file...")
final_submission = prepare_submission(test, test_pred, sub)

# Show first 10 predictions
print("\nFirst 10 predictions:")
display(final_submission.head(10))

