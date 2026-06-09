!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from typing import List
from tqdm import tqdm
import optuna
import xgboost as xgb
import joblib


folder='/kaggle/input/neurips-open-polymer-prediction-2025/'
df_train=pd.read_csv(folder+'train.csv')
df_test=pd.read_csv(folder+'test.csv')


print('maximum number of null columns/ total columns % = ',round(df_train.isnull().sum(axis = 1).max() / df_train.isnull().count(axis = 1).max() * 100,2))


null_counts=df_train.isnull().sum()
round(100*(null_counts[null_counts>0]/len(df_train.index)), 2)


print("Median value of Tg {0}".format(df_train['Tg'].median()))
print("Median value of FFV {0}".format(df_train['FFV'].median()))
print("Median value of Tc {0}".format(df_train['Tc'].median()))
print("Median value of Density {0}".format(df_train['Density'].median()))
print("Median value of Rg {0}".format(df_train['Rg'].median()))


df_train.fillna( {
        'Tg': 74.04018308,
        'FFV': 0.364263595,
        'Tc': 0.236,
        'Density': 0.948193246,
        'Rg': 15.052194175} , inplace=True)


train_smiles = df_train['SMILES'].values
train_targets = df_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]


class PolymerPredictor:
    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.01):
        """Initialize the polymer property predictor.
        
        Args:
            n_estimators: Number of trees in XGBoost model
            learning_rate: Learning rate for XGBoost
        """
        print("Initializing PolymerPredictor model...")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = {}
        self.scalers = {}
        
        
        # Define reasonable ranges for each property
        # These ranges help in normalizing the weighted MAE metric
        self.property_ranges = {
            'Tg': 500,      # Glass transition temperature range in K
            'FFV': 0.5,     # Fractional free volume (0-1 range)
            'Tc': 1000,     # Critical temperature range in K
            'Density': 2.0, # Density range in g/cm³
            'Rg': 100       # Radius of gyration range in Å
        }
        print(f"Model initialized with {n_estimators} trees and learning rate {learning_rate}")
        
    def _extract_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract chemical features from SMILES string using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            np.ndarray: Feature vector containing molecular descriptors
        """
        print("Starting feature extraction from SMILES strings...")
        features = []
        # Use the new MorganGenerator API to avoid deprecation warnings
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        
        for smiles in tqdm(smiles_list, desc="Extracting features"):
            try:
                # Convert SMILES to RDKit molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES string: {smiles}")
                    features.append(np.zeros(200))  # Default feature vector
                    continue
                
                # Calculate molecular descriptors
                feature_vector = []
                
                # Basic descriptors
                feature_vector.extend([
                    Descriptors.MolWt(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.NumHeteroatoms(mol)
                ])
                
                # Morgan fingerprints (ECFP4) using the new generator API
                fp = morgan_gen.GetFingerprint(mol).ToBitString()
                fp_bits = [int(x) for x in fp]
                feature_vector.extend(fp_bits)
                
                # Additional descriptors
                feature_vector.extend([
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                    rdMolDescriptors.CalcNumAliphaticRings(mol),
                    rdMolDescriptors.CalcNumSaturatedRings(mol),
                    rdMolDescriptors.CalcNumHeterocycles(mol),
                    rdMolDescriptors.CalcNumSpiroAtoms(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                    rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
                ])
                
                # Pad or truncate to ensure consistent feature vector size
                feature_vector = feature_vector[:200] + [0] * (200 - len(feature_vector))
                features.append(feature_vector)
                
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {str(e)}")
                features.append(np.zeros(200))
        
        print(f"Successfully extracted features for {len(features)} molecules")
        return np.array(features)
        
    def _weighted_mae(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Calculate weighted MAE as per competition metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: Weighted MAE score
        """
        print("Calculating weighted MAE score...")
        
        # Count number of samples for each property
        n_p = {col: len(y_true[col].dropna()) for col in y_true.columns}
        print(f"Number of samples per property: {n_p}")
        
        # Calculate weights based on sample count and property range
        weights = {p: (n_p[p]**-0.5)/self.property_ranges[p] for p in self.property_ranges}
        total_weight = sum(weights.values())
        norm_weights = {p: weights[p] * len(weights)/total_weight for p in weights}
        print(f"Normalized weights: {norm_weights}")
        
        # Calculate weighted errors
        errors = np.abs(y_true - y_pred)
        weighted_score = np.mean([errors[p].mean() * norm_weights[p] for p in y_true.columns])
        
        print(f"Final weighted MAE score: {weighted_score:.4f}")
        return weighted_score
        
    def optimize_xgb(self, X, y, param_space, n_trials=20):
        """
        Use Optuna to find the best XGBoost hyperparameters for a given property.
        Args:
            X: Feature matrix
            y: Target vector
            param_space: Dictionary specifying the search space for each hyperparameter
            n_trials: Number of Optuna trials
        Returns:
            dict: Best hyperparameters found
        """
        print(f"Starting Optuna optimization for property with {n_trials} trials...")
        def objective(trial):
            params = {}
            for k, v in param_space.items():
                if isinstance(v, tuple) and len(v) == 2:
                    # Float or int range
                    if all(isinstance(x, int) for x in v):
                        params[k] = trial.suggest_int(k, v[0], v[1])
                    else:
                        params[k] = trial.suggest_float(k, v[0], v[1], log=True if k == 'learning_rate' else False)
                elif isinstance(v, list):
                    params[k] = trial.suggest_categorical(k, v)
            params['objective'] = 'reg:absoluteerror'  # Changed to MAE objective
            params['random_state'] = 42

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            preds = model.predict(X_valid)
            return mean_absolute_error(y_valid, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f"Best params found: {study.best_params}")
        return study.best_params

    def fit(self, train_smiles: list, train_targets: pd.DataFrame, optimize=False, param_space=None, n_trials=20) -> None:
        """Train the model on the provided data. Optionally use Optuna for hyperparameter optimization.
        
        Args:
            train_smiles: List of SMILES strings
            train_targets: DataFrame with target properties
            optimize: Whether to use Optuna for hyperparameter search
            param_space: Dictionary specifying the search space for each hyperparameter
            n_trials: Number of Optuna trials
        """
        print("Starting model training process...")
        
        # Extract features from SMILES strings
        print("Extracting molecular features from SMILES strings using RDKit...")
        X = self._extract_features(train_smiles)
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} molecules")
        
        # Train separate models for each property
        for property_name in train_targets.columns:
            print(f"\nTraining model for {property_name}...")
            
            # Get valid indices for this property
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                print(f"No valid data for {property_name}, skipping...")
                continue
                
            X_prop = X[valid_idx]
            y_prop = train_targets[property_name][valid_idx]
            print(f"Training data size for {property_name}: {len(y_prop)} samples")
            
            # Scale features
            print("Scaling features using StandardScaler...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            self.scalers[property_name] = scaler
            
            # Hyperparameter optimization with Optuna if requested
            if optimize and param_space is not None:
                print(f"Optimizing hyperparameters for {property_name} using Optuna...")
                best_params = self.optimize_xgb(X_scaled, y_prop, param_space, n_trials)
                model_params = best_params.copy()
                model_params['objective'] = 'reg:absoluteerror'
                model_params['random_state'] = 42
            else:
                model_params = dict(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    objective='reg:absoluteerror',
                    random_state=42,
                    max_depth=7,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0
                )
            
            # Train XGBoost model
            print("Training XGBoost model with parameters: {}".format(model_params))
            model = xgb.XGBRegressor(**model_params, early_stopping_rounds=100)
            print("Starting model training with early stopping...")
            model.fit(
                X_scaled, y_prop,
                eval_set=[(X_scaled, y_prop)],
                verbose=100
            )
            
            self.models[property_name] = model
            print(f"Completed training for {property_name}")
            
    def predict(self, test_smiles: list) -> pd.DataFrame:
        """Make predictions for new SMILES strings.
        
        Args:
            test_smiles: List of SMILES strings to predict
            
        Returns:
            pd.DataFrame: Predictions for all properties
        """
        print("Starting prediction process...")
        
        # Extract features from test SMILES
        print("Extracting features from test molecules using RDKit...")
        X = self._extract_features(test_smiles)
        print(f"Extracted features for {len(test_smiles)} test molecules")
        
        # Make predictions for each property
        predictions = {}
        for property_name, model in self.models.items():
            print(f"Making predictions for {property_name}...")
            X_scaled = self.scalers[property_name].transform(X)
            
            # Create DMatrix for faster prediction
            dtest = xgb.DMatrix(X_scaled)
            predictions[property_name] = model.predict(X_scaled)
            
        print("Completed all predictions")
        return pd.DataFrame(predictions)

    def fit_predict_best_model(self, train_smiles, train_targets, test_smiles, best_params):
        """
        Fit and predict in one shot using the best parameters for all properties.
        Args:
            train_smiles: List of SMILES strings for training
            train_targets: DataFrame with target properties
            test_smiles: List of SMILES strings for prediction
            best_params: Dict of best hyperparameters (from Optuna or manual)
        Returns:
            pd.DataFrame: Predictions for all properties
        """
        print("Fitting and predicting with best parameters in one shot...")
        # Extract features
        X_train = self._extract_features(train_smiles)
        X_test = self._extract_features(test_smiles)
        predictions = {}

        # Create directory for saving models if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)

        for property_name in train_targets.columns:
            print(f"Processing property: {property_name}")
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                print(f"No valid data for {property_name}, skipping...")
                continue

            X_prop = X_train[valid_idx]
            y_prop = train_targets[property_name][valid_idx]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[property_name] = scaler

            # Merge best params with default XGBoost params
            model_params = best_params.copy()
            model_params['objective'] = 'reg:absoluteerror'  # Changed to MAE objective
            model_params['random_state'] = 42

            print(f"Training XGBoost model for {property_name} with best parameters: {model_params}")
            model = xgb.XGBRegressor(**model_params, early_stopping_rounds=100)
            model.fit(X_scaled, y_prop, eval_set=[(X_scaled, y_prop)], verbose=100)
            self.models[property_name] = model

            # Save model and scaler
            model_path = os.path.join('saved_models', f'{property_name}_model.joblib')
            scaler_path = os.path.join('saved_models', f'{property_name}_scaler.joblib')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"Saved model and scaler for {property_name}")

            # Predict
            predictions[property_name] = model.predict(X_test_scaled)

        print("Completed fit and predict for all properties.")
        return pd.DataFrame(predictions)



    
    
    #--- Optuna Hyperparameter Optimization Usage ---
    param_space = {
        'n_estimators': (2500, 7500),                # Integer range
        'learning_rate': (0.01, 0.05),              # Float range (log scale for learning_rate)
        'max_depth': (10, 32),                       # Integer range
        'subsample': (0.5, 0.9),                    # Float range
        'colsample_bytree': (0.5, 0.9),             # Float range
        'gamma': (0, 1.0),                          # Float range
        'reg_alpha': (0, 1.0),                      # Float range
        'reg_lambda': (0, 2.0),                     # Float range
    }
    n_trials = 100
    print("Initializing and training model with Optuna hyperparameter optimization...")
    model = PolymerPredictor()
    
    # Dictionary to store best parameters for each property
    best_params_per_property = {}
    
    # Train models for each property
    for property_name in train_targets.columns:
        print(f"\nOptimizing hyperparameters for {property_name}...")
        valid_idx = ~train_targets[property_name].isna()
        if not valid_idx.any():
            print(f"No valid data for {property_name}, skipping...")
            continue
            
        X = model._extract_features(train_smiles[valid_idx])
        y = train_targets[property_name][valid_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimize hyperparameters
        best_params = model.optimize_xgb(X_scaled, y, param_space, n_trials)
        best_params_per_property[property_name] = best_params
        
        # Save scaler
        scaler_path = f'{property_name}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler for {property_name} to {scaler_path}")
    
    # Print best parameters for each property
    print("\nBest hyperparameters for each property:")
    best_params_df = pd.DataFrame(best_params_per_property).T
    print("\nBest Hyperparameters per Property:")
    print(best_params_df)
    
    # Train final models with best parameters
    model.fit(
        train_smiles,
        train_targets,
        optimize=False  # We already have the best parameters
    )
    
    # Save models
    for property_name, model_obj in model.models.items():
        model_path = f'{property_name}_model.joblib'
        joblib.dump(model_obj, model_path)
        print(f"Saved model for {property_name} to {model_path}")
    
    print("Making predictions on test set...")
    predictions = model.predict(df_test['SMILES'].values)
    print("Creating submission file...")
    submission = pd.DataFrame({
        'id': df_test['id'],
        **predictions
    })
    print("Saving submission file...")
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("Process completed successfully!")




