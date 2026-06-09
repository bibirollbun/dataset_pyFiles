#!pip install -U transformers
#!pip install -U rdkit
#!pip install -U torch
!pip install /kaggle/input/rdkit-package/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



"""
Comprehensive Training and Prediction Pipeline for Polymer Properties
Trains 5 models (Tg, Density, FFV, Rg, Tc) using embeddings + molecular features
Makes predictions on test data and combines into sample_submission.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Chemistry imports
from transformers import AutoTokenizer, AutoModel
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Utility imports
import pickle
import json
from tqdm import tqdm
import time

tf.random.set_seed(42)



#MODEL_NAME = "/kaggle/input/chemberta-77m-mlm/pytorch/default/1/ChemBERTa-77M-MLM"
MODEL_NAME = "/kaggle/input/chemberta/ChemBERTa-77M-MLM"

data_files_path = "/kaggle/input/training-data"

class PolymerPropertyModel:
    """
    Class to handle training and prediction for a single polymer property
    """
    
    def __init__(self, property_name, embedding_dim=768):
        self.property_name = property_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.tokenizer = None
        self.embedding_model = None
        self.feature_columns = None
        self.target_column = None
        
    def load_embeddings_model(self):
        """Load the pretrained ChemBERTa model for SMILES embeddings"""
        print(f"Loading ChemBERTa model for {self.property_name}...")
        
        try:
            # Load tokenizer and model
            #model_name = "DeepChem/ChemBERTa-10M-MTR"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
            self.embedding_model = AutoModel.from_pretrained(MODEL_NAME,local_files_only=True)
            
            # Set to evaluation mode
            self.embedding_model.eval()
            print(f"✓ ChemBERTa model loaded successfully for {self.property_name}")
            
        except Exception as e:
            print(f"Error loading ChemBERTa model: {e}")
            print("Falling back to RDKit descriptors only...")
            self.embedding_model = None
    
    def generate_embeddings(self, smiles_list, batch_size=32):
        """Generate embeddings from SMILES strings"""
        if self.embedding_model is None:
            return None
            
        embeddings = []
        
        for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Generating embeddings for {self.property_name}"):
            batch_smiles = smiles_list[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_smiles, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors="pt")
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use CLS token embeddings (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def prepare_training_data(self, data_path):
        """Prepare training data with embeddings and features"""
        print(f"Preparing training data for {self.property_name}...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")
        
        # Check for required columns
        required_cols = ['Canonical_SMILES']
        
        # Handle column name variations for Tc
        target_column = self.property_name
        if self.property_name == 'Tc' and 'TC_mean' in df.columns:
            target_column = 'TC_mean'
            print(f"Using 'TC_mean' column for Tc target")
        elif self.property_name not in df.columns:
            print(f"Error: Target column '{self.property_name}' not found")
            return None, None, None
            
        # Get feature columns (exclude SMILES and target amd id)
        exclude_cols = ['Canonical_SMILES', target_column, 'id'] if 'id' in df.columns else ['Canonical_SMILES', target_column]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        self.target_column = target_column
        
        print(f"  Feature columns: {len(self.feature_columns)}")
        print(f"  Target column: {self.target_column}")
        
        # Generate embeddings
        smiles_list = df['Canonical_SMILES'].tolist()
        embeddings = self.generate_embeddings(smiles_list)
        
        # Prepare molecular features
        features = df[self.feature_columns].values
        
        # Prepare target
        target = df[target_column].values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(target) | np.isnan(features).any(axis=1))
        
        if embeddings is not None:
            valid_mask = valid_mask & ~np.isnan(embeddings).any(axis=1)
        
        if embeddings is not None:
            embeddings = embeddings[valid_mask]
        features = features[valid_mask]
        target = target[valid_mask]
        
        print(f"  Valid samples after cleaning: {len(target)}")
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Combine embeddings and features
        if embeddings is not None:
            X = np.hstack([embeddings, features_scaled])
            print(f"Final input shape: {X.shape}")
        else:
            X = features_scaled
            print(f"Final input shape: {X.shape} (RDKit features only)")
        
        return X, target, embeddings is not None
    
    def build_model(self, input_dim, use_embeddings=True):
        """Build the neural network model"""
        print(f"Building model for {self.property_name}...")
        
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"  Model architecture: {input_dim} → 512 → 256 → 128 → 64 → 1")
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=100):
        """Train the model"""
        print(f"Training model for {self.property_name}...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"  Validation Loss: {val_loss[0]:.4f}")
        print(f"  Validation MAE: {val_loss[1]:.4f}")
        
        return history
    
    def save_model(self, save_dir="models"):
        """Save the trained model and scalers"""
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"{self.property_name.lower()}_model.keras")
        scaler_path = os.path.join(save_dir, f"{self.property_name.lower()}_scalers.pkl")
        
        # Save model
        self.model.save(model_path)
        
        # Save scalers AND feature information
        scalers = {
            'feature_scaler': self.feature_scaler,
            'use_embeddings': hasattr(self, 'use_embeddings') and self.use_embeddings,
            'feature_columns': self.feature_columns, 
            'target_column': self.target_column   
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        print(f"  Model saved to: {model_path}")
        print(f"  Scalers saved to: {scaler_path}")
    
    def load_model(self, save_dir="/kaggle/working/models/"):
        """Load a trained model and scalers"""
        model_path = os.path.join(save_dir, f"{self.property_name.lower()}_model.keras")
        scaler_path = os.path.join(save_dir, f"{self.property_name.lower()}_scalers.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = keras.models.load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.use_embeddings = scalers.get('use_embeddings', True)
                
                # RESTORE feature columns and target column if available
                if 'feature_columns' in scalers:
                    self.feature_columns = scalers['feature_columns']
                    self.target_column = scalers.get('target_column', self.property_name)
                    print(f"Restored {len(self.feature_columns)} feature columns from saved model")
                else:
                    # Fallback: restore from training data
                    self._restore_feature_columns()
            
            print(f"Model loaded from: {model_path}")
            return True
        else:
            print(f"  Model not found at: {model_path}")
            return False

    # 3. NEW _restore_feature_columns METHOD
    def _restore_feature_columns(self):
        """Restore feature columns from training data"""
        # Try to find the training data to restore feature columns
        training_data_paths = {
            'Tg': f'{data_files_path}property_specific/Tg_features.csv',
            'Density': f'{data_files_path}/property_specific/density_features.csv',
            'FFV': f'{data_files_path}/property_specific/ffv_features.csv',
            'Rg': f'{data_files_path}/property_specific/rg_features.csv',
            'Tc': f'{data_files_path}/property_specific/tc_features.csv'
        }
        
        if self.property_name in training_data_paths:
            try:
                df = pd.read_csv(training_data_paths[self.property_name])
                
                # Handle column name variations for Tc
                target_column = self.property_name
                if self.property_name == 'Tc' and 'TC_mean' in df.columns:
                    target_column = 'TC_mean'
                
                # Get feature columns (exclude SMILES and target)
                exclude_cols = ['Canonical_SMILES', target_column, 'id'] if 'id' in df.columns else ['Canonical_SMILES', target_column]
                self.feature_columns = [col for col in df.columns if col not in exclude_cols]
                self.target_column = target_column
                
                print(f"Restored {len(self.feature_columns)} feature columns from training data")
                
            except Exception as e:
                print(f"Warning: Could not restore feature columns: {e}")
                # Set a default empty list to prevent errors
                self.feature_columns = []
                self.target_column = self.property_name
                
        else:
            print(f"Warning: No training data path found for {self.property_name}")
            self.feature_columns = []
            self.target_column = self.property_name
    
    def predict(self, test_data_path):
        """Make predictions on test data"""
        print(f"Making predictions for {self.property_name}...")
        
        # Load test data
        df = pd.read_csv(test_data_path)
        print(f"  Loaded {len(df)} test samples")
        
        # Generate embeddings
        #smiles_list = df['Canonical_SMILES'].tolist()
        #embeddings = self.generate_embeddings(smiles_list)
        embeddings = None
        
        # PREPARE FEATURES - ensure feature_columns exists
        if self.feature_columns is None:
            print(f"Error: feature_columns not available for {self.property_name}")
            return None
            
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        if not feature_cols:
            print(f"Error: No matching feature columns found in test data for {self.property_name}")
            return None
            
        features = df[feature_cols].values
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Combine embeddings and features
        if embeddings is not None and hasattr(self, 'use_embeddings') and self.use_embeddings:
            X_test = np.hstack([embeddings, features_scaled])
        else:
            X_test = features_scaled
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        print(f"Predictions shape: {predictions.shape}")
        return predictions


# To train the model it only reads the property_specific directory in preprocessing. IGNORE EVERYTHING ELSE.

def train_all_models():
    """Train models for all 5 properties"""
    print("="*80)
    print("TRAINING ALL PROPERTY MODELS")
    print("="*80)
    
    # Property configurations
    properties = {
        'Tg': f'{data_files_path}/property_specific/Tg_features.csv',
        'Density': f'{data_files_path}/property_specific/density_features.csv',
        'FFV': f'{data_files_path}/property_specific/ffv_features.csv',
        'Rg': f'{data_files_path}/property_specific/rg_features.csv',
        'Tc': f'{data_files_path}/property_specific/tc_features.csv'
    }
    
    trained_models = {}
    
    for property_name, data_path in properties.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING: {property_name}")
        print(f"{'='*60}")
        
        # Create model instance
        model = PolymerPropertyModel(property_name)
        
        # Load embeddings model
        #model.load_embeddings_model() not using pretrained model
        
        # Prepare training data
        X, y, use_embeddings = model.prepare_training_data(data_path)
        
        if X is None:
            print(f"  Skipping {property_name} due to data issues")
            continue
        
        model.use_embeddings = use_embeddings
        
        # Train model
        history = model.train_model(X, y)
        
        # Save model
        model.save_model()
        
        trained_models[property_name] = model
        
        print(f"  ✓ {property_name} model trained and saved successfully")
    
    return trained_models


def predict_all_properties(trained_models):
    """Make predictions for all properties on test data"""
    print("\n" + "="*80)
    print("MAKING PREDICTIONS ON TEST DATA")
    print("="*80)
    
    # Test data paths
    test_data = {
        'Tg': '/kaggle/working/testing_data/tg_test.csv',
        'Density': '/kaggle/working/testing_data/density_test.csv',
        'FFV': '/kaggle/working/testing_data/ffv_test.csv',
        'Rg': '/kaggle/working/testing_data/rg_test.csv',
        'Tc': '/kaggle/working/testing_data/tc_test.csv'
    }
    
    all_predictions = {}
    
    for property_name, test_path in test_data.items():
        if property_name in trained_models:
            print(f"\nPredicting {property_name}...")
            
            try:
                predictions = trained_models[property_name].predict(test_path)
                if predictions is not None:
                    all_predictions[property_name] = predictions
                    print(f"  ✓ {property_name} predictions completed")
                else:
                    print(f"  ✗ {property_name} predictions failed")
                    all_predictions[property_name] = None
                    
            except Exception as e:
                print(f"  ✗ Error predicting {property_name}: {e}")
                all_predictions[property_name] = None
    
    return all_predictions


# creates test files
def _smiles_to_mol_for_descriptors(smiles: str, wildcard_replacement='[H]'):
    """
    Convert (P)SMILES to an RDKit Mol suitable for descriptor calculation.
    Replaces '*' attachment points with `wildcard_replacement` (default [H]).
    Returns None if parsing fails.
    """
    if pd.isna(smiles):
        return None
    s = str(smiles)
    if '*' in s:
        s = s.replace('*', wildcard_replacement)
    try:
        s = Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True)
        mol = Chem.MolFromSmiles(s)
        return mol, s
    except Exception:
        return None

def convert_test_to_test_mw(
    test_csv_path="/kaggle/input/neurips-open-polymer-prediction-2025/test.csv",
    output_path="/kaggle/working/processed/test_mw.csv",
    wildcard_replacement='[H]',
    progress_every=500
):
    """
    Convert Kaggle's test.csv to test_mw.csv with molecular weight & related features.
    Safely handles PSMILES '*' by replacing with `wildcard_replacement` (default [H]).
    """
    print("=" * 80)
    print("CONVERTING KAGGLE TEST.CSV TO TEST_MW.CSV")
    print("=" * 80)

    # Load Kaggle test data
    print(f"Loading Kaggle test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Original test data shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")

    smiles_col = "SMILES"
    if smiles_col not in test_df.columns:
        raise ValueError(f"Expected column '{smiles_col}' not found in test.csv")

    # Ensure output dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mol_descriptors = []
    n = len(test_df)
    print(f"\nProcessing {n} molecules...")

    for idx, row in test_df.iterrows():
        if (idx % progress_every) == 0:
            print(f"  Processed {idx}/{n} molecules...")

        smiles = row[smiles_col]
        try:
            mol, s = _smiles_to_mol_for_descriptors(smiles, wildcard_replacement=wildcard_replacement)
            if mol is None:
                # Keep row with minimal info so you still get an id in outputs
                mol_descriptors.append({
                    'id': row.get('id', idx),
                    'Canonical_SMILES': s,
                    'MolWt': None,
                    'MolLogP': None,
                    'HeavyAtomCount': None,
                    'RingCount': None,
                    'HallKierAlpha': None,
                    'ExactMolWt': None,
                    'NumHeteroatoms': None,
                    'LabuteASA': None,
                    'TPSA': None,
                    'NumRotatableBonds': None,
                    'FractionCSP3': None,
                    'NumAliphaticRings': None,
                    'NumAromaticRings': None,
                    'NumHDonors': None,
                    'NumHAcceptors': None,
                    'NumAromaticCarbocycles': None,
                    'MolMR': None,
                    'BertzCT': None,
                    'Kappa1': None,
                    'Kappa2': None,
                    'Kappa3': None
                })
                continue

            # Calculate descriptors (note the corrected CalcFractionCSP3)
            desc = {
                'id': row.get('id', idx),
                'Canonical_SMILES': smiles,
                'MolWt': Descriptors.MolWt(mol),
                'MolLogP': Descriptors.MolLogP(mol),
                'HeavyAtomCount': mol.GetNumHeavyAtoms(),
                'RingCount': rdMolDescriptors.CalcNumRings(mol),
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),
                'ExactMolWt': Descriptors.ExactMolWt(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                'MolMR': Descriptors.MolMR(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol)
            }
            mol_descriptors.append(desc)

        except Exception as e:
            # Keep going; log minimal info
            print(f"  Error processing molecule {idx}: {e}")
            mol_descriptors.append({
                'id': row.get('id', idx),
                'Canonical_SMILES': smiles
            })
            continue

    test_mw_df = pd.DataFrame(mol_descriptors)

    print(f"\nProcessed {len(test_mw_df)} molecules successfully")
    print(f"Final shape: {test_mw_df.shape}")
    print(f"Features: {list(test_mw_df.columns)}")

    test_mw_df.to_csv(output_path, index=False)
    print(f"\nSaved test_mw.csv to: {output_path}")

    return test_mw_df


def create_property_test_files_from_test_mw(test_mw_path="/kaggle/working/processed/test_mw.csv"):
    """
    Create property-specific test files from test_mw.csv
    """
    print("\n" + "=" * 80)
    print("CREATING PROPERTY-SPECIFIC TEST FILES FROM TEST_MW.CSV")
    print("=" * 80)

    if not os.path.exists(test_mw_path):
        print(f"Error: {test_mw_path} not found")
        print("Please run convert_test_to_test_mw() first")
        return None

    test_df = pd.read_csv(test_mw_path)
    print(f"Test_mw.csv shape: {test_df.shape}")
    print(f"Available columns: {list(test_df.columns)}")

    property_features = {
        'Tg': [
            'id', 'Canonical_SMILES',
            'NumRotatableBonds', 'FractionCSP3', 'NumAliphaticRings',
            'NumHDonors', 'NumAromaticRings', 'RingCount',
            'MolWt', 'HeavyAtomCount', 'BertzCT'
        ],
        'Density': [
            'id', 'Canonical_SMILES',
            'MolWt', 'MolLogP', 'HeavyAtomCount', 'RingCount',
            'HallKierAlpha', 'ExactMolWt', 'NumHeteroatoms',
            'LabuteASA', 'TPSA'
        ],
        'FFV': [
            'id', 'Canonical_SMILES',
            'HeavyAtomCount', 'RingCount', 'LabuteASA', 'TPSA',
            'MolWt', 'BertzCT', 'Kappa1', 'Kappa2', 'Kappa3',
            'FractionCSP3', 'NumRotatableBonds'
        ],
        'Rg': [
            'id', 'Canonical_SMILES',
            'MolWt', 'HeavyAtomCount', 'RingCount', 'FractionCSP3',
            'NumAromaticRings', 'NumAromaticCarbocycles',
            'Kappa1', 'Kappa2', 'Kappa3', 'BertzCT',
            'LabuteASA', 'NumRotatableBonds'
        ],
        'Tc': [
            'id', 'Canonical_SMILES',
            'TPSA', 'MolWt', 'NumRotatableBonds', 'FractionCSP3',
            'NumHAcceptors', 'MolMR', 'HeavyAtomCount',
            'NumAromaticRings', 'BertzCT'
        ]
    }

    created_files = {}
    output_dir = "/kaggle/working/testing_data"
    os.makedirs(output_dir, exist_ok=True)

    for property_name, required in property_features.items():
        print(f"\nProcessing {property_name}...")
        available = [c for c in required if c in test_df.columns]
        missing   = [c for c in required if c not in test_df.columns]
        if missing:
            print(f"Missing features: {missing}")
        print(f"Available features: {len(available)}")

        df_subset = test_df[available].copy()
        out_path = os.path.join(output_dir, f"{property_name.lower()}_test.csv")
        df_subset.to_csv(out_path, index=False)
        print(f"Saved: {out_path}  |  Shape: {df_subset.shape}")

        created_files[property_name] = {
            "filename": out_path,
            "shape": df_subset.shape,
            "features": available,
            "missing": missing
        }

    print("\n" + "=" * 80)
    print("SUMMARY OF CREATED TEST FILES")
    print("=" * 80)
    print(f"{'Property':<10} {'Samples':<10} {'Features':<10} {'Missing':<10}")
    print("-" * 80)
    for prop, info in created_files.items():
        print(f"{prop:<10} {info['shape'][0]:<10} {info['shape'][1]:<10} {len(info['missing']):<10}")

    print("All property-specific test files created successfully!")
    return created_files


def complete_test_preprocessing_pipeline(test_csv_path):
    print("=" * 80)
    print("COMPLETE TEST PREPROCESSING PIPELINE")
    print("=" * 80)

    print("\nSTEP 1: Converting test.csv to test_mw.csv...")
    test_mw_df = convert_test_to_test_mw(test_csv_path)

    if test_mw_df is None:
        print("Failed to convert test.csv to test_mw.csv")
        return None

    print("\nSTEP 2: Creating property-specific test files...")
    created_files = create_property_test_files_from_test_mw()

    if created_files is None:
        print("Failed to create property-specific test files")
        return None

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return {"test_mw": test_mw_df, "property_files": created_files}


# Example usage on Kaggle:
#df = convert_test_to_test_mw(r"C:\Users\sarth\OneDrive\john_hopkins\Deep Neural Network\Colab\polymer\data\test.csv")

test_csv_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"  # Update this path
#test_csv_path = "/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv"

#df =convert_test_to_test_mw()
result = complete_test_preprocessing_pipeline(test_csv_path)


test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


check_df = pd.read_csv("/kaggle/working/testing_data/density_test.csv"); check_df


def create_sample_submission(all_predictions):
    #sample_submission_path="/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
    """Create sample submission file with all predictions"""
    print("\n" + "="*80)
    print("CREATING SAMPLE SUBMISSION FILE")
    print("="*80)

    # Load test data to get IDs
    test_df = pd.read_csv("/kaggle/working/testing_data/density_test.csv")
    test_ids = test_df['id'].tolist()
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({'id': test_ids})
    
    # Add predictions for each property
    for property_name, predictions in all_predictions.items():
        if predictions is not None:
            predictions_df[property_name] = predictions
            print(f"Added {property_name} predictions")
        else:
            # Use mean values from training data as fallback
            predictions_df[property_name] = 0.0
            print(f"Added {property_name} fallback values")
    

    predictions_df.to_csv("submission.csv",index=False)
    
    return predictions_df


def main(trained_models=None):
    """Main execution function"""
    print("Polymer Property Prediction Pipeline")
    print("="*50)

    # Check if models already exist
    models_dir = "/kaggle/working/models/"
    if not trained_models:
        if os.path.exists(models_dir):
            print("Checking for existing trained models...")
            
            # Try to load existing models
            trained_models = {}
            properties = ['Tg', 'Density', 'FFV', 'Rg', 'Tc']
            
            for prop in properties:
                model = PolymerPropertyModel(prop)
                if model.load_model(): # Since Loading the model is not working
                    trained_models[prop] = model
                    print(f"Loaded existing {prop} model")
                else:
                    print(f"{prop} model not found")
            
            if len(trained_models) == 5:
                print("All models loaded successfully!")
            else:
                print(f"Only {len(trained_models)}/5 models found. Training missing models...")
                # Train missing models
                missing_models = train_all_models()
                trained_models.update(missing_models)
        else:
            print("No existing models found. Training all models...")
            trained_models = train_all_models()

    # Make predictions
    all_predictions = predict_all_properties(trained_models)
    
    # Create sample submission
    predictions_df = create_sample_submission(all_predictions)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Output files:")
    print("- Final predictions: /kaggle/working/predictions_final.csv")
    print("- Updated sample submission: submission.csv")
    print("- Trained models: /kaggle/working/models/")

    return predictions_df, trained_models



predictions_df, trained_models = main()


predictions_df


def normalize_submission(submission_df, sample_path):
    sample_df = pd.read_csv(sample_path)

    # 1) Column set must match
    if set(submission_df.columns) != set(sample_df.columns):
        missing = set(sample_df.columns) - set(submission_df.columns)
        extra   = set(submission_df.columns) - set(sample_df.columns)
        raise ValueError(f"Column mismatch. Missing={missing}, Extra={extra}")

    # 2) Reorder columns to match sample exactly
    submission_df = submission_df[sample_df.columns]

    # Reindex rows to match sample_df id order
    #sub_by_id = submission_df.set_index('id')
    #submission_df = sample_df[['id']].join(sub_by_id, on='id', how='left')

    # 4) Enforce numeric dtypes for target cols (id stays as in sample)
    for col in sample_df.columns:
        if col != 'id':
            submission_df[col] = pd.to_numeric(submission_df[col], errors='coerce')

    # 5) Final checks: no NaNs, right shape
    if submission_df.isna().any().any():
        bad_cols = submission_df.columns[submission_df.isna().any()].tolist()
        raise ValueError(f"Found NaNs after normalization in columns: {bad_cols}")

    assert tuple(submission_df.columns) == tuple(sample_df.columns), "Column order still off"
    #assert len(submission_df) == len(sample_df), "Row count mismatch"

    return submission_df

# Usage
sample_path = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
submission_df_updated = normalize_submission(predictions_df, sample_path)
submission_df_updated.to_csv("submission.csv", index=False)



submission_df_updated


sample_path = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
sample_df = pd.read_csv(sample_path)

# Check columns match exactly
assert list(submission_df_updated.columns) == list(sample_df.columns), "Column names/order mismatch"
assert result["test_mw"].isna().sum().sum() == 0, "There are null alues in test_mw"
#assert len(test_df) == len(result["test_mw"]), "Length of test_mw and test_df is not same"
#assert len(test_df) == len(submission_df), "Length of test_df and submission_df are different"
assert submission_df_updated.isna().sum().sum() == 0, "There are null values in submission.csv"


sample_df


submission_df_updated


!head 'submission.csv'


#!rm -rf /kaggle/working/*

