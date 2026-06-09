import sys
import subprocess
import os
from pathlib import Path

# Local RDKit installation setup
def install_local_rdkit():
    """Install RDKit from local wheel file"""
    rdkit_wheel_path = "/kaggle/input/rdkit-2025/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl"
    
    if os.path.exists(rdkit_wheel_path):
        try:
            print(f"Installing RDKit from local wheel: {rdkit_wheel_path}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                rdkit_wheel_path, "--no-deps", "--force-reinstall"
            ])
            print("RDKit installed successfully from local wheel")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install RDKit from wheel: {e}")
            return False
    else:
        print(f"RDKit wheel file not found at: {rdkit_wheel_path}")
        return False

# Install RDKit locally before importing
rdkit_installed = install_local_rdkit()


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm.auto import tqdm
import gc
import psutil
import sys
import logging
import os
import subprocess
from typing import Dict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Configuration and Setup ---
# Set up a logger for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global constants
TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
MAIN_TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
SUBMISSION_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv'
MEMORY_LIMIT_GB = 12.0  # Failsafe for environment memory limits

# Map supplementary dataset file names to their target properties and new column names
SUPPLEMENTARY_DATA_MAP = {
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-2025/train_supplement/dataset2.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv': {'old_target': 'Tg_mean', 'new_target': 'Tg'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset5.csv': {'old_target': 'FFV_mean', 'new_target': 'FFV'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset6.csv': {'old_target': 'Density_mean', 'new_target': 'Density'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset7.csv': {'old_target': 'Rg_mean', 'new_target': 'Rg'},
}

# --- INSTALLATION BLOCK ---
try:
    logger.info("Attempting to install rdkit locally from .whl file.")
    subprocess.run(
        ["pip", "install", "--no-deps", "/kaggle/input/rdkit-2025/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl"],
        check=True,
        capture_output=True,
        text=True
    )
    logger.info("rdkit installed successfully.")
    
    logger.info("Attempting to install lightgbm.")
    subprocess.run(["pip", "install", "lightgbm"], check=True)
    logger.info("lightgbm installed successfully.")
    
except Exception as e:
    logger.error(f"Failed to install critical dependencies. Error: {e}")
    sys.exit(1)
# --- END INSTALLATION BLOCK ---

# Set up TensorFlow GPU memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY_LIMIT_GB * 1024)])
        logger.info("GPU memory growth enabled and limit set.")
except Exception as e:
    logger.warning(f"Could not configure GPU memory: {e}")

# --- 1. Memory and Resource Management ---
class MemoryMonitor:
    """Class to manage and monitor memory usage."""
    def __init__(self, limit_gb: float):
        self.memory_limit_gb = limit_gb

    def get_memory_usage(self) -> float:
        """Returns current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

    def force_cleanup(self):
        """Aggressively clears memory."""
        gc.collect()
        tf.keras.backend.clear_session()
        logger.info("Aggressive memory cleanup performed.")

    def log_memory_status(self, message: str):
        """Logs the current memory usage."""
        current_mem = self.get_memory_usage()
        logger.info(f"Memory Status ({message}): {current_mem:.2f} GB / {self.memory_limit_gb:.2f} GB")
        if current_mem > self.memory_limit_gb * 0.95:
            logger.warning("Memory usage is critically high. Forcing cleanup.")
            self.force_cleanup()
            if self.get_memory_usage() > self.memory_limit_gb:
                raise MemoryError(f"Memory limit exceeded: {self.get_memory_usage():.2f} GB. Stopping process.")

# --- 2. Data Processing Classes ---
class PolymerDataProcessor:
    """Handles data loading and basic preprocessing."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor

    def load_data(self, filepath: str, nrows: int = None) -> pd.DataFrame:
        """Loads data from a single CSV file."""
        try:
            df = pd.read_csv(filepath, nrows=nrows)
            logger.info(f"Successfully loaded {df.shape[0]} rows from {os.path.basename(filepath)}.")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return pd.DataFrame()
            
    def load_and_merge_training_data(self, main_path: str, supplementary_map: Dict, nrows: int = None) -> pd.DataFrame:
        """Loads main and supplementary data, then merges them."""
        logger.info("Loading main training data...")
        train_df = self.load_data(main_path, nrows=nrows)
        
        supplementary_dfs = []
        for path, info in supplementary_map.items():
            if os.path.exists(path):
                logger.info(f"Loading supplementary data from {os.path.basename(path)}...")
                sup_df = self.load_data(path, nrows=nrows)
                if not sup_df.empty and info['old_target'] in sup_df.columns:
                    sup_df.rename(columns={info['old_target']: info['new_target']}, inplace=True)
                    supplementary_dfs.append(sup_df)
        
        if supplementary_dfs:
            all_sup_df = pd.concat(supplementary_dfs, ignore_index=True)
            for target in TARGET_PROPERTIES:
                if target not in all_sup_df.columns:
                    all_sup_df[target] = np.nan
            
            logger.info(f"Merging main and supplementary data...")
            merged_df = pd.concat([train_df, all_sup_df], ignore_index=True)
            self.memory_monitor.log_memory_status("After merging training data")
            return merged_df
            
        return train_df

class SMILESProcessor:
    """Extracts molecular features from SMILES strings."""
    def extract_molecular_features(self, smiles_list: list) -> pd.DataFrame:
        """Featurizes a list of SMILES strings using RDKit."""
        features = []
        for smiles in tqdm(smiles_list, desc="Featurizing SMILES"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    feature_vec = [
                        Descriptors.MolWt(mol),
                        Descriptors.LogP(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol)
                    ]
                    features.append(feature_vec)
                else:
                    features.append([np.nan] * 7)
            except Exception as e:
                features.append([np.nan] * 7)
        
        df_features = pd.DataFrame(features, columns=['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings'])
        return df_features

class AdvancedFeatureEngineer:
    """Performs advanced feature engineering and selection."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor

    def engineer_polymer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds custom engineered features."""
        logger.info("Engineering custom features.")
        if 'MolWt' in df.columns and 'Density' in df.columns:
            df['MW_x_Density'] = df['MolWt'] * df['Density']
        self.memory_monitor.log_memory_status("After Feature Engineering")
        return df

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.DataFrame, max_features: int) -> list:
        """Selects features based on importance from a simple model."""
        logger.info("Selecting features by importance.")
        feature_importances = X.corrwith(y.mean(axis=1)).abs().sort_values(ascending=False)
        selected = feature_importances.index[:max_features].tolist()
        return selected

# --- 3. Model Training and Ensemble ---
class AdvancedModelEnsemble:
    """Manages the training and ensembling of multiple models."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.best_model = None
        self.best_score = -np.inf

    def create_neural_network(self, input_shape: int, output_dim: int) -> Model:
        """Creates a simple neural network model with regularization."""
        # L1 and L2 regularization to penalize large weights and prevent overfitting
        regularizer = l1_l2(l1=1e-5, l2=1e-4) 
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(inputs)
        # Dropout layers to randomly deactivate neurons and improve generalization
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(output_dim, name='output_layer')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model

    def train_classical_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Trains classical ML models (LightGBM)."""
        logger.info("Training classical models (LGBMRegressor).")
        
        results = {}
        try:
            lgbm_model = lgb.LGBMRegressor(
                objective='mae',
                metric='mae',
                n_estimators=100, # Increased estimators
                learning_rate=0.05, # Reduced learning rate for stability
                num_leaves=20, # Reduced leaves to prevent overfitting
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # Early stopping to prevent overfitting
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=-1)] 
            )
            
            val_predictions = lgbm_model.predict(X_val)
            train_predictions = lgbm_model.predict(X_train)
            
            val_score = self.calculate_weighted_mae(y_val, val_predictions)
            train_score = self.calculate_weighted_mae(y_train, train_predictions)
            
            results['lgbm'] = {
                'model': lgbm_model,
                'val_score': val_score,
                'train_score': train_score,
                'val_predictions': val_predictions,
                'train_predictions': train_predictions
            }
            
            logger.info(f"LGBMRegressor: Train wMAE={-train_score:.4f}, Val wMAE={-val_score:.4f}")
            
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = lgbm_model
        
        except Exception as e:
            logger.error(f"Error training classical model: {e}")
            return {}

        self.memory_monitor.force_cleanup()
        return results

    def calculate_weighted_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate weighted MAE score as defined in the competition"""
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        
        if y_true.shape[1] != len(weights):
            weights = np.ones(y_true.shape[1]) / y_true.shape[1]
        
        mae_scores = []
        for i in range(y_true.shape[1]):
            valid_mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
            if valid_mask.sum() > 0:
                mae = mean_absolute_error(y_true[valid_mask, i], y_pred[valid_mask, i])
                mae_scores.append(mae)
            else:
                mae_scores.append(0.0)
        
        weighted_mae = np.average(mae_scores, weights=weights[:len(mae_scores)])
        return -weighted_mae

    def train_neural_networks(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Train and evaluate a neural network model."""
        results = {}
        target_dim = y_train.shape[1] if y_train.ndim > 1 else 1
        
        try:
            nn_model = self.create_neural_network(X_train.shape[1], target_dim)
            
            callbacks = [
                # Early stopping and learning rate reduction to prevent overfitting
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
                ModelCheckpoint('best_nn_model.h5', save_best_only=True, monitor='val_loss', verbose=0)
            ]
            
            history = nn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=min(64, len(X_train) // 4),
                callbacks=callbacks,
                verbose=0
            )
            
            train_pred = nn_model.predict(X_train, verbose=0)
            val_pred = nn_model.predict(X_val, verbose=0)
            
            val_score = self.calculate_weighted_mae(y_val, val_pred)
            train_score = self.calculate_weighted_mae(y_train, train_pred)
            
            results['neural_network'] = {
                'model': nn_model,
                'val_score': val_score,
                'train_score': train_score,
                'val_predictions': val_pred,
                'train_predictions': train_pred,
                'history': history.history
            }
            
            logger.info(f"Neural Network: Train wMAE={-train_score:.4f}, Val wMAE={-val_score:.4f}")
            
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = nn_model
                
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            
        self.memory_monitor.force_cleanup()
        return results

    def create_ensemble_prediction(self, X: np.ndarray, models_results: Dict[str, any]) -> np.ndarray:
        """Creates ensemble predictions using model averaging."""
        predictions = []
        weights = []
        
        for name, result in models_results.items():
            if 'val_predictions' in result:
                pred = result['model'].predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred)
                weights.append(max(0.1, result.get('val_score', -10) + 10))
        
        if predictions:
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
        else:
            return np.zeros((X.shape[0], 5))

# --- 4. Submission Handling ---
class CompetitionSubmission:
    """Handle competition submission format."""
    def __init__(self):
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def prepare_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                           submission_template: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare submission file in competition format."""
        logger.info("Preparing submission file...")
        submission = submission_template.copy() if submission_template is not None else pd.DataFrame()
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        for i, col in enumerate(self.target_columns):
            if i < predictions.shape[1]:
                submission[col] = predictions[:, i]
            else:
                submission[col] = 0.0
        
        submission = submission.fillna(0.0)
        logger.info(f"Submission prepared with shape: {submission.shape}")
        return submission
    
    def save_submission(self, submission_df: pd.DataFrame, filename: str = "submission.csv"):
        """Save submission file."""
        submission_df.to_csv(filename, index=False)
        logger.info(f"Submission saved to {filename}")

# --- 5. Data Visualization and Evaluation ---
class DataVisualizer:
    def __init__(self):
        self.target_columns = TARGET_PROPERTIES
        self.feature_columns = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings']

    def plot_missing_values(self, df: pd.DataFrame):
        """Plots a heatmap of missing values for relevant columns."""
        # Focus on the most important columns to avoid clutter
        relevant_cols = ['id', 'SMILES'] + self.target_columns + self.feature_columns
        # Filter for existing columns
        existing_cols = [col for col in relevant_cols if col in df.columns]
        plot_df = df[existing_cols]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(plot_df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap for Key Columns', fontsize=16)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Row Index', fontsize=12)
        plt.show()

    def plot_data_distributions(self, df: pd.DataFrame, columns: list, title_prefix: str):
        """Plots histograms for specified columns."""
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(columns):
            plt.subplot(n_cols, n_cols, i + 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'{title_prefix} Distribution: {col}')
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actuals(self, y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str):
        """Plots clean scatter plots of predictions vs actuals for each target property."""
        n_cols = 3
        n_rows = (len(self.target_columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(self.target_columns):
            if i < y_true.shape[1]:
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(y_true[:, i])
                y_true_clean = y_true[valid_mask, i]
                y_pred_clean = y_pred[valid_mask, i]

                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                
                ax = plt.subplot(n_rows, n_cols, i + 1)
                
                # Use a hexbin plot to show density, which is great for large datasets
                hb = ax.hexbin(y_true_clean, y_pred_clean, gridsize=30, cmap='inferno')
                ax.set_title(f'{title_prefix}: {col} (MAE: {mae:.4f})')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')

                # Add a colorbar for the hexbin plot
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Count')
                
                # Plot a perfect correlation line
                min_val = min(y_true_clean.min(), y_pred_clean.min())
                max_val = max(y_true_clean.max(), y_pred_clean.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'w--', linewidth=2, label='Perfect Prediction')

        plt.tight_layout()
        plt.show()

# --- 6. Main Pipeline Orchestration ---
class PolymerPredictionPipeline:
    """Main pipeline orchestrating the entire prediction workflow."""
    def __init__(self, memory_limit_gb: float = MEMORY_LIMIT_GB):
        self.memory_monitor = MemoryMonitor(memory_limit_gb)
        self.processor = PolymerDataProcessor(self.memory_monitor)
        self.feature_engineer = AdvancedFeatureEngineer(self.memory_monitor)
        self.model_ensemble = AdvancedModelEnsemble(self.memory_monitor)
        self.submission_handler = CompetitionSubmission()
        self.visualizer = DataVisualizer()
        self.pipeline_results = {}
        
    def run_complete_pipeline(self, main_train_path: str, supplementary_map: Dict,
                             test_path: str, submission_template_path: str = None) -> Dict[str, any]:
        """Run the complete ML pipeline."""
        logger.info("=" * 50)
        logger.info("Starting NeurIPS Polymer Prediction Pipeline")
        logger.info("=" * 50)
        
        self.memory_monitor.log_memory_status("Pipeline Start")
        
        train_df = self.processor.load_and_merge_training_data(main_train_path, supplementary_map, nrows=3000)
        test_df = self.processor.load_data(test_path, nrows=100)
        
        if train_df.empty or test_df.empty:
            logger.error("Failed to load data. Aborting pipeline.")
            return {}
        
        self.memory_monitor.log_memory_status("After Data Loading")
        
        logger.info("Step 2: SMILES Feature Extraction...")
        smiles_processor = SMILESProcessor()
        if 'SMILES' in train_df.columns and 'SMILES' in test_df.columns:
            train_smiles_features = smiles_processor.extract_molecular_features(train_df['SMILES'].tolist())
            test_smiles_features = smiles_processor.extract_molecular_features(test_df['SMILES'].tolist())
            train_df = pd.concat([train_df, train_smiles_features], axis=1)
            test_df = pd.concat([test_df, test_smiles_features], axis=1)
        
        self.memory_monitor.log_memory_status("After SMILES Processing")

        logger.info("Step 3: Performing EDA and Plotting Data Distributions...")
        self.visualizer.plot_missing_values(train_df)
        self.visualizer.plot_data_distributions(train_df, TARGET_PROPERTIES, 'Target Property')
        
        logger.info("Step 4: Feature Engineering...")
        train_engineered = self.feature_engineer.engineer_polymer_features(train_df)
        test_engineered = self.feature_engineer.engineer_polymer_features(test_df)
        
        logger.info("Step 5: Preparing Training Data...")
        feature_columns = [col for col in train_engineered.columns if col not in TARGET_PROPERTIES + ['SMILES']]
        
        X_train_full = train_engineered[feature_columns].select_dtypes(include=np.number)
        y_train_full = train_engineered[TARGET_PROPERTIES]
        
        valid_samples = y_train_full.notna().any(axis=1)
        X_train_full = X_train_full[valid_samples]
        y_train_full = y_train_full[valid_samples]
        
        # Split data into a dedicated training+validation set and a separate holdout set
        X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )
        logger.info(f"Data Split: Train+Val={X_train_val.shape[0]} samples, Holdout={X_holdout.shape[0]} samples.")
        
        selected_features = self.feature_engineer.select_features_by_importance(X_train_val, y_train_val, max_features=50)
        
        X_train_val = X_train_val[selected_features]
        X_holdout = X_holdout[selected_features]
        available_features = [f for f in selected_features if f in test_engineered.columns]
        X_test = test_engineered[available_features].select_dtypes(include=np.number)
        
        X_train_val = X_train_val.dropna(axis=1, how='all')
        X_test = X_test.dropna(axis=1, how='all')
        X_holdout = X_holdout.dropna(axis=1, how='all')
        
        # Align columns
        X_test = X_test.reindex(columns=X_train_val.columns, fill_value=0.0)
        X_holdout = X_holdout.reindex(columns=X_train_val.columns, fill_value=0.0)
        
        X_train_median = X_train_val.median()
        X_train_val = X_train_val.fillna(X_train_median)
        X_test = X_test.fillna(X_train_median)
        X_holdout = X_holdout.fillna(X_train_median)
        
        y_train_val = y_train_val.fillna(y_train_val.median())
        y_holdout = y_holdout.fillna(y_holdout.median())
        
        self.memory_monitor.log_memory_status("After Data Preparation")
        
        logger.info("Step 6: Model Training...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        logger.info(f"Training split: Train={X_train_split.shape[0]} samples, Val={X_val_split.shape[0]} samples.")
        
        # Using RobustScaler for better performance with outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        X_test_scaled = scaler.transform(X_test)
        X_holdout_scaled = scaler.transform(X_holdout)
        
        classical_results = self.model_ensemble.train_classical_models(X_train_scaled, y_train_split.values, X_val_scaled, y_val_split.values)
        nn_results = self.model_ensemble.train_neural_networks(X_train_scaled, y_train_split.values, X_val_scaled, y_val_split.values)
        
        all_results = {**classical_results, **nn_results}
        self.pipeline_results['model_results'] = all_results
        
        self.memory_monitor.log_memory_status("After Model Training")
        
        logger.info("Step 7: Evaluating on Holdout Set...")
        ensemble_holdout_predictions = self.model_ensemble.create_ensemble_prediction(X_holdout_scaled, all_results)
        
        logger.info("Step 8: Plotting Holdout Evaluation Results...")
        self.visualizer.plot_predictions_vs_actuals(y_holdout.values, ensemble_holdout_predictions, 'Holdout Predictions vs Actuals')
        
        logger.info("Step 9: Generating Final Predictions for Submission...")
        ensemble_predictions = self.model_ensemble.create_ensemble_prediction(X_test_scaled, all_results)
        
        logger.info("Step 10: Preparing Submission...")
        submission_template = None
        if submission_template_path and os.path.exists(submission_template_path):
            try:
                submission_template = pd.read_csv(submission_template_path)
            except Exception as e:
                logger.warning(f"Could not load submission template: {e}")
        
        submission_df = self.submission_handler.prepare_submission(test_df, ensemble_predictions, submission_template)
        
        # Save the submission file as submission.csv
        self.submission_handler.save_submission(submission_df, "submission.csv")
        
        # Preview the contents of the submission file
        logger.info("Step 11: Previewing the Submission File...")
        if os.path.exists("submission.csv"):
            preview_df = pd.read_csv("submission.csv")
            print("\nPreview of submission.csv:")
            print(preview_df.head())
        else:
            logger.error("Failed to save or find submission.csv for preview.")
        
        logger.info("Step 12: Results Summary...")
        self.print_results_summary(all_results)
        
        self.memory_monitor.log_memory_status("Pipeline Complete")
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 50)
        
        return {
            'submission': submission_df,
            'model_results': all_results,
            'predictions': ensemble_predictions
        }
    
    def print_results_summary(self, results: Dict[str, any]):
        """Print comprehensive results summary."""
        logger.info("\n" + "="*60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        for model_name, result in results.items():
            if 'val_score' in result:
                logger.info(f"{model_name.upper()}:")
                logger.info(f"  Validation wMAE: {-result['val_score']:.6f}")
                logger.info(f"  Training wMAE:   {-result['train_score']:.6f}")
                if abs(result['val_score']) > 1e-6:
                    logger.info(f"  Overfit Factor:  {abs(result['train_score']/result['val_score']):.3f}")
                logger.info("-" * 30)
        
        best_model_name = max(results.keys(), key=lambda x: results[x].get('val_score', -np.inf))
        logger.info(f"BEST MODEL: {best_model_name.upper()}")
        logger.info(f"Best Validation Score: {-results[best_model_name]['val_score']:.6f}")

# Main execution
if __name__ == "__main__":
    pipeline = PolymerPredictionPipeline()
    
    try:
        results = pipeline.run_complete_pipeline(
            main_train_path=MAIN_TRAIN_PATH,
            supplementary_map=SUPPLEMENTARY_DATA_MAP,
            test_path=TEST_PATH,
            submission_template_path=SUBMISSION_PATH
        )
        
        logger.info("Pipeline execution completed successfully!")
        logger.info("Check 'submission.csv' for final submission.")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.memory_monitor.force_cleanup()
        logger.info("Final memory cleanup completed.")


import sys
import subprocess
import os
from pathlib import Path

# Local RDKit installation setup
def install_local_rdkit():
    """Install RDKit from local wheel file"""
    rdkit_wheel_path = "/kaggle/input/rdkit-2025/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl"
    
    if os.path.exists(rdkit_wheel_path):
        try:
            print(f"Installing RDKit from local wheel: {rdkit_wheel_path}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                rdkit_wheel_path, "--no-deps", "--force-reinstall"
            ])
            print("RDKit installed successfully from local wheel")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install RDKit from wheel: {e}")
            return False
    else:
        print(f"RDKit wheel file not found at: {rdkit_wheel_path}")
        return False

# Install RDKit locally before importing
rdkit_installed = install_local_rdkit()

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm.auto import tqdm
import gc
import psutil
import sys
import logging
import os
import subprocess
from typing import Dict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Configuration and Setup ---
# Set up a logger for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global constants
TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
MAIN_TRAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
TEST_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
SUBMISSION_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv'
MEMORY_LIMIT_GB = 12.0  # Failsafe for environment memory limits

# Map supplementary dataset file names to their target properties and new column names
SUPPLEMENTARY_DATA_MAP = {
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-2025/train_supplement/dataset2.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv': {'old_target': 'TC_mean', 'new_target': 'Tc'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv': {'old_target': 'Tg_mean', 'new_target': 'Tg'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset5.csv': {'old_target': 'FFV_mean', 'new_target': 'FFV'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset6.csv': {'old_target': 'Density_mean', 'new_target': 'Density'},
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset7.csv': {'old_target': 'Rg_mean', 'new_target': 'Rg'},
}

# --- INSTALLATION BLOCK ---
try:
    logger.info("Attempting to install rdkit locally from .whl file.")
    subprocess.run(
        ["pip", "install", "--no-deps", "/kaggle/input/rdkit-2025/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl"],
        check=True,
        capture_output=True,
        text=True
    )
    logger.info("rdkit installed successfully.")
    
    logger.info("Attempting to install lightgbm.")
    subprocess.run(["pip", "install", "lightgbm"], check=True)
    logger.info("lightgbm installed successfully.")
    
except Exception as e:
    logger.error(f"Failed to install critical dependencies. Error: {e}")
    sys.exit(1)
# --- END INSTALLATION BLOCK ---

# Set up TensorFlow GPU memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY_LIMIT_GB * 1024)])
        logger.info("GPU memory growth enabled and limit set.")
except Exception as e:
    logger.warning(f"Could not configure GPU memory: {e}")

# --- 1. Memory and Resource Management ---
class MemoryMonitor:
    """Class to manage and monitor memory usage."""
    def __init__(self, limit_gb: float):
        self.memory_limit_gb = limit_gb

    def get_memory_usage(self) -> float:
        """Returns current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

    def force_cleanup(self):
        """Aggressively clears memory."""
        gc.collect()
        tf.keras.backend.clear_session()
        logger.info("Aggressive memory cleanup performed.")

    def log_memory_status(self, message: str):
        """Logs the current memory usage."""
        current_mem = self.get_memory_usage()
        logger.info(f"Memory Status ({message}): {current_mem:.2f} GB / {self.memory_limit_gb:.2f} GB")
        if current_mem > self.memory_limit_gb * 0.95:
            logger.warning("Memory usage is critically high. Forcing cleanup.")
            self.force_cleanup()
            if self.get_memory_usage() > self.memory_limit_gb:
                raise MemoryError(f"Memory limit exceeded: {self.get_memory_usage():.2f} GB. Stopping process.")

# --- 2. Data Processing Classes ---
class PolymerDataProcessor:
    """Handles data loading and basic preprocessing."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor

    def load_data(self, filepath: str, nrows: int = None) -> pd.DataFrame:
        """Loads data from a single CSV file."""
        try:
            df = pd.read_csv(filepath, nrows=nrows)
            logger.info(f"Successfully loaded {df.shape[0]} rows from {os.path.basename(filepath)}.")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return pd.DataFrame()
            
    def load_and_merge_training_data(self, main_path: str, supplementary_map: Dict, nrows: int = None) -> pd.DataFrame:
        """Loads main and supplementary data, then merges them."""
        logger.info("Loading main training data...")
        train_df = self.load_data(main_path, nrows=nrows)
        
        supplementary_dfs = []
        for path, info in supplementary_map.items():
            if os.path.exists(path):
                logger.info(f"Loading supplementary data from {os.path.basename(path)}...")
                sup_df = self.load_data(path, nrows=nrows)
                if not sup_df.empty and info['old_target'] in sup_df.columns:
                    sup_df.rename(columns={info['old_target']: info['new_target']}, inplace=True)
                    supplementary_dfs.append(sup_df)
        
        if supplementary_dfs:
            all_sup_df = pd.concat(supplementary_dfs, ignore_index=True)
            for target in TARGET_PROPERTIES:
                if target not in all_sup_df.columns:
                    all_sup_df[target] = np.nan
            
            logger.info(f"Merging main and supplementary data...")
            merged_df = pd.concat([train_df, all_sup_df], ignore_index=True)
            self.memory_monitor.log_memory_status("After merging training data")
            return merged_df
            
        return train_df

class SMILESProcessor:
    """Extracts molecular features from SMILES strings."""
    def extract_molecular_features(self, smiles_list: list) -> pd.DataFrame:
        """Featurizes a list of SMILES strings using RDKit."""
        features = []
        for smiles in tqdm(smiles_list, desc="Featurizing SMILES"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    feature_vec = [
                        Descriptors.MolWt(mol),
                        Descriptors.LogP(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol)
                    ]
                    features.append(feature_vec)
                else:
                    features.append([np.nan] * 7)
            except Exception as e:
                features.append([np.nan] * 7)
        
        df_features = pd.DataFrame(features, columns=['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings'])
        return df_features

class AdvancedFeatureEngineer:
    """Performs advanced feature engineering and selection."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor

    def engineer_polymer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds custom engineered features."""
        logger.info("Engineering custom features.")
        if 'MolWt' in df.columns and 'Density' in df.columns:
            df['MW_x_Density'] = df['MolWt'] * df['Density']
        self.memory_monitor.log_memory_status("After Feature Engineering")
        return df

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.DataFrame, max_features: int) -> list:
        """Selects features based on importance from a simple model."""
        logger.info("Selecting features by importance.")
        feature_importances = X.corrwith(y.mean(axis=1)).abs().sort_values(ascending=False)
        selected = feature_importances.index[:max_features].tolist()
        return selected

# --- 3. Model Training and Ensemble ---
class AdvancedModelEnsemble:
    """Manages the training and ensembling of multiple models."""
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.best_model = None
        self.best_score = -np.inf

    def create_neural_network(self, input_shape: int, output_dim: int) -> Model:
        """Creates a simple neural network model with regularization."""
        # L1 and L2 regularization to penalize large weights and prevent overfitting
        regularizer = l1_l2(l1=1e-5, l2=1e-4) 
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(inputs)
        # Dropout layers to randomly deactivate neurons and improve generalization
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(output_dim, name='output_layer')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model

    def train_classical_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Trains classical ML models (LightGBM)."""
        logger.info("Training classical models (LGBMRegressor).")
        
        results = {}
        try:
            lgbm_model = lgb.LGBMRegressor(
                objective='mae',
                metric='mae',
                n_estimators=100, # Increased estimators
                learning_rate=0.05, # Reduced learning rate for stability
                num_leaves=20, # Reduced leaves to prevent overfitting
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # Early stopping to prevent overfitting
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=-1)] 
            )
            
            val_predictions = lgbm_model.predict(X_val)
            train_predictions = lgbm_model.predict(X_train)
            
            val_score = self.calculate_weighted_mae(y_val, val_predictions)
            train_score = self.calculate_weighted_mae(y_train, train_predictions)
            
            results['lgbm'] = {
                'model': lgbm_model,
                'val_score': val_score,
                'train_score': train_score,
                'val_predictions': val_predictions,
                'train_predictions': train_predictions
            }
            
            logger.info(f"LGBMRegressor: Train wMAE={-train_score:.4f}, Val wMAE={-val_score:.4f}")
            
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = lgbm_model
        
        except Exception as e:
            logger.error(f"Error training classical model: {e}")
            return {}

        self.memory_monitor.force_cleanup()
        return results

    def calculate_weighted_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the weighted Mean Absolute Error (wMAE) based on the competition's formula.
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Define the five target properties
        target_properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Placeholder for the estimated range of values (ri) and sample counts (ni)
        # In a real scenario, these would be calculated from the full dataset.
        # We use rough estimates here.
        estimated_ranges = {
            'Tg': 400.0,
            'FFV': 0.1,
            'Tc': 0.2,
            'Density': 0.5,
            'Rg': 20.0
        }
        
        mae_scores = {}
        sample_counts = {}
        
        for i, prop in enumerate(target_properties):
            if i < y_true.shape[1]:
                # Find valid samples for this property
                valid_mask = ~np.isnan(y_true[:, i])
                
                if valid_mask.sum() > 0:
                    mae = mean_absolute_error(y_true[valid_mask, i], y_pred[valid_mask, i])
                    mae_scores[prop] = mae
                    sample_counts[prop] = valid_mask.sum()
                else:
                    mae_scores[prop] = 0.0
                    sample_counts[prop] = 0
            else:
                mae_scores[prop] = 0.0
                sample_counts[prop] = 0
        
        # Calculate weights based on the competition formula
        weighted_maes = []
        K = len(target_properties)
        sum_sqrt_inverse_n = sum(np.sqrt(1 / n) for n in sample_counts.values() if n > 0)
        
        for prop, mae in mae_scores.items():
            ni = sample_counts[prop]
            ri = estimated_ranges.get(prop, 1.0)
            
            if ni > 0 and ri > 0:
                wi = (1/ri) * ((K * np.sqrt(1/ni)) / sum_sqrt_inverse_n)
                weighted_maes.append(wi * mae)
            else:
                weighted_maes.append(0.0)
        
        total_weighted_mae = sum(weighted_maes)
        return -total_weighted_mae

    def train_neural_networks(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Train and evaluate a neural network model."""
        results = {}
        target_dim = y_train.shape[1] if y_train.ndim > 1 else 1
        
        try:
            nn_model = self.create_neural_network(X_train.shape[1], target_dim)
            
            callbacks = [
                # Early stopping and learning rate reduction to prevent overfitting
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
                ModelCheckpoint('best_nn_model.h5', save_best_only=True, monitor='val_loss', verbose=0)
            ]
            
            history = nn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=min(64, len(X_train) // 4),
                callbacks=callbacks,
                verbose=0
            )
            
            train_pred = nn_model.predict(X_train, verbose=0)
            val_pred = nn_model.predict(X_val, verbose=0)
            
            val_score = self.calculate_weighted_mae(y_val, val_pred)
            train_score = self.calculate_weighted_mae(y_train, train_pred)
            
            results['neural_network'] = {
                'model': nn_model,
                'val_score': val_score,
                'train_score': train_score,
                'val_predictions': val_pred,
                'train_predictions': train_pred,
                'history': history.history
            }
            
            logger.info(f"Neural Network: Train wMAE={-train_score:.4f}, Val wMAE={-val_score:.4f}")
            
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = nn_model
                
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            
        self.memory_monitor.force_cleanup()
        return results

    def create_ensemble_prediction(self, X: np.ndarray, models_results: Dict[str, any]) -> np.ndarray:
        """Creates ensemble predictions using model averaging."""
        predictions = []
        weights = []
        
        for name, result in models_results.items():
            if 'val_predictions' in result:
                pred = result['model'].predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred)
                weights.append(max(0.1, result.get('val_score', -10) + 10))
        
        if predictions:
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
        else:
            return np.zeros((X.shape[0], 5))

# --- 4. Submission Handling ---
class CompetitionSubmission:
    """Handle competition submission format."""
    def __init__(self):
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def prepare_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                           submission_template: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare submission file in competition format."""
        logger.info("Preparing submission file...")
        submission = submission_template.copy() if submission_template is not None else pd.DataFrame()
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        for i, col in enumerate(self.target_columns):
            if i < predictions.shape[1]:
                submission[col] = predictions[:, i]
            else:
                submission[col] = 0.0
        
        submission = submission = pd.DataFrame(test_df['id']).reset_index(drop=True)
        for i, col in enumerate(self.target_columns):
            if i < predictions.shape[1]:
                submission[col] = predictions[:, i]
            else:
                submission[col] = 0.0
        
        submission = submission.fillna(0.0)
        logger.info(f"Submission prepared with shape: {submission.shape}")
        return submission
    
    def save_submission(self, submission_df: pd.DataFrame, filename: str = "submission.csv"):
        """Save submission file."""
        submission_df.to_csv(filename, index=False)
        logger.info(f"Submission saved to {filename}")

# --- 5. Data Visualization and Evaluation ---
class DataVisualizer:
    def __init__(self):
        self.target_columns = TARGET_PROPERTIES
        self.feature_columns = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 'AromRings']

    def plot_missing_values(self, df: pd.DataFrame):
        """Plots a heatmap of missing values for relevant columns."""
        # Focus on the most important columns to avoid clutter
        relevant_cols = ['id', 'SMILES'] + self.target_columns + self.feature_columns
        # Filter for existing columns
        existing_cols = [col for col in relevant_cols if col in df.columns]
        plot_df = df[existing_cols]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(plot_df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap for Key Columns', fontsize=16)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Row Index', fontsize=12)
        plt.show()

    def plot_data_distributions(self, df: pd.DataFrame, columns: list, title_prefix: str):
        """Plots histograms for specified columns."""
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(columns):
            plt.subplot(n_cols, n_cols, i + 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'{title_prefix} Distribution: {col}')
        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actuals(self, y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str):
        """Plots clean scatter plots of predictions vs actuals for each target property."""
        n_cols = 3
        n_rows = (len(self.target_columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(self.target_columns):
            if i < y_true.shape[1]:
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(y_true[:, i])
                y_true_clean = y_true[valid_mask, i]
                y_pred_clean = y_pred[valid_mask, i]

                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                
                ax = plt.subplot(n_rows, n_cols, i + 1)
                
                # Use a hexbin plot to show density, which is great for large datasets
                hb = ax.hexbin(y_true_clean, y_pred_clean, gridsize=30, cmap='inferno')
                ax.set_title(f'{title_prefix}: {col} (MAE: {mae:.4f})')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')

                # Add a colorbar for the hexbin plot
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Count')
                
                # Plot a perfect correlation line
                min_val = min(y_true_clean.min(), y_pred_clean.min())
                max_val = max(y_true_clean.max(), y_pred_clean.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'w--', linewidth=2, label='Perfect Prediction')

        plt.tight_layout()
        plt.show()

# --- 6. Main Pipeline Orchestration ---
class PolymerPredictionPipeline:
    """Main pipeline orchestrating the entire prediction workflow."""
    def __init__(self, memory_limit_gb: float = MEMORY_LIMIT_GB):
        self.memory_monitor = MemoryMonitor(memory_limit_gb)
        self.processor = PolymerDataProcessor(self.memory_monitor)
        self.feature_engineer = AdvancedFeatureEngineer(self.memory_monitor)
        self.model_ensemble = AdvancedModelEnsemble(self.memory_monitor)
        self.submission_handler = CompetitionSubmission()
        self.visualizer = DataVisualizer()
        self.pipeline_results = {}
        
    def run_complete_pipeline(self, main_train_path: str, supplementary_map: Dict,
                             test_path: str, submission_template_path: str = None) -> Dict[str, any]:
        """Run the complete ML pipeline."""
        logger.info("=" * 50)
        logger.info("Starting NeurIPS Polymer Prediction Pipeline")
        logger.info("=" * 50)
        
        self.memory_monitor.log_memory_status("Pipeline Start")
        
        train_df = self.processor.load_and_merge_training_data(main_train_path, supplementary_map, nrows=3000)
        test_df = self.processor.load_data(test_path, nrows=100)
        
        if train_df.empty or test_df.empty:
            logger.error("Failed to load data. Aborting pipeline.")
            return {}
        
        self.memory_monitor.log_memory_status("After Data Loading")
        
        logger.info("Step 2: SMILES Feature Extraction...")
        smiles_processor = SMILESProcessor()
        if 'SMILES' in train_df.columns and 'SMILES' in test_df.columns:
            train_smiles_features = smiles_processor.extract_molecular_features(train_df['SMILES'].tolist())
            test_smiles_features = smiles_processor.extract_molecular_features(test_df['SMILES'].tolist())
            train_df = pd.concat([train_df, train_smiles_features], axis=1)
            test_df = pd.concat([test_df, test_smiles_features], axis=1)
        
        self.memory_monitor.log_memory_status("After SMILES Processing")

        logger.info("Step 3: Performing EDA and Plotting Data Distributions...")
        self.visualizer.plot_missing_values(train_df)
        self.visualizer.plot_data_distributions(train_df, TARGET_PROPERTIES, 'Target Property')
        
        logger.info("Step 4: Feature Engineering...")
        train_engineered = self.feature_engineer.engineer_polymer_features(train_df)
        test_engineered = self.feature_engineer.engineer_polymer_features(test_df)
        
        logger.info("Step 5: Preparing Training Data...")
        feature_columns = [col for col in train_engineered.columns if col not in TARGET_PROPERTIES + ['SMILES']]
        
        X_train_full = train_engineered[feature_columns].select_dtypes(include=np.number)
        y_train_full = train_engineered[TARGET_PROPERTIES]
        
        valid_samples = y_train_full.notna().any(axis=1)
        X_train_full = X_train_full[valid_samples]
        y_train_full = y_train_full[valid_samples]
        
        # Split data into a dedicated training+validation set and a separate holdout set
        X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )
        logger.info(f"Data Split: Train+Val={X_train_val.shape[0]} samples, Holdout={X_holdout.shape[0]} samples.")
        
        selected_features = self.feature_engineer.select_features_by_importance(X_train_val, y_train_val, max_features=50)
        
        X_train_val = X_train_val[selected_features]
        X_holdout = X_holdout[selected_features]
        available_features = [f for f in selected_features if f in test_engineered.columns]
        X_test = test_engineered[available_features].select_dtypes(include=np.number)
        
        X_train_val = X_train_val.dropna(axis=1, how='all')
        X_test = X_test.dropna(axis=1, how='all')
        X_holdout = X_holdout.dropna(axis=1, how='all')
        
        # Align columns
        X_test = X_test.reindex(columns=X_train_val.columns, fill_value=0.0)
        X_holdout = X_holdout.reindex(columns=X_train_val.columns, fill_value=0.0)
        
        X_train_median = X_train_val.median()
        X_train_val = X_train_val.fillna(X_train_median)
        X_test = X_test.fillna(X_train_median)
        X_holdout = X_holdout.fillna(X_train_median)
        
        y_train_val = y_train_val.fillna(y_train_val.median())
        y_holdout = y_holdout.fillna(y_holdout.median())
        
        self.memory_monitor.log_memory_status("After Data Preparation")
        
        logger.info("Step 6: Model Training...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        logger.info(f"Training split: Train={X_train_split.shape[0]} samples, Val={X_val_split.shape[0]} samples.")
        
        # Using RobustScaler for better performance with outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        X_test_scaled = scaler.transform(X_test)
        X_holdout_scaled = scaler.transform(X_holdout)
        
        classical_results = self.model_ensemble.train_classical_models(X_train_scaled, y_train_split.values, X_val_scaled, y_val_split.values)
        nn_results = self.model_ensemble.train_neural_networks(X_train_scaled, y_train_split.values, X_val_scaled, y_val_split.values)
        
        all_results = {**classical_results, **nn_results}
        self.pipeline_results['model_results'] = all_results
        
        self.memory_monitor.log_memory_status("After Model Training")
        
        logger.info("Step 7: Evaluating on Holdout Set...")
        ensemble_holdout_predictions = self.model_ensemble.create_ensemble_prediction(X_holdout_scaled, all_results)
        
        logger.info("Step 8: Plotting Holdout Evaluation Results...")
        self.visualizer.plot_predictions_vs_actuals(y_holdout.values, ensemble_holdout_predictions, 'Holdout Predictions vs Actuals')
        
        logger.info("Step 9: Generating Final Predictions for Submission...")
        ensemble_predictions = self.model_ensemble.create_ensemble_prediction(X_test_scaled, all_results)
        
        logger.info("Step 10: Preparing Submission...")
        submission_template = None
        if submission_template_path and os.path.exists(submission_template_path):
            try:
                submission_template = pd.read_csv(submission_template_path)
            except Exception as e:
                logger.warning(f"Could not load submission template: {e}")
        
        submission_df = self.submission_handler.prepare_submission(test_df, ensemble_predictions, submission_template)
        
        # Save the submission file as submission.csv
        self.submission_handler.save_submission(submission_df, "submission.csv")
        
        # Preview the contents of the submission file
        logger.info("Step 11: Previewing the Submission File...")
        if os.path.exists("submission.csv"):
            preview_df = pd.read_csv("submission.csv")
            print("\nPreview of submission.csv:")
            print(preview_df.head())
        else:
            logger.error("Failed to save or find submission.csv for preview.")
        
        logger.info("Step 12: Results Summary...")
        self.print_results_summary(all_results)
        
        self.memory_monitor.log_memory_status("Pipeline Complete")
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 50)
        
        return {
            'submission': submission_df,
            'model_results': all_results,
            'predictions': ensemble_predictions
        }
    
    def print_results_summary(self, results: Dict[str, any]):
        """Print comprehensive results summary."""
        logger.info("\n" + "="*60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        for model_name, result in results.items():
            if 'val_score' in result:
                logger.info(f"{model_name.upper()}:")
                logger.info(f"  Validation wMAE: {-result['val_score']:.6f}")
                logger.info(f"  Training wMAE:   {-result['train_score']:.6f}")
                if abs(result['val_score']) > 1e-6:
                    logger.info(f"  Overfit Factor:  {abs(result['train_score']/result['val_score']):.3f}")
                logger.info("-" * 30)
        
        best_model_name = max(results.keys(), key=lambda x: results[x].get('val_score', -np.inf))
        logger.info(f"BEST MODEL: {best_model_name.upper()}")
        logger.info(f"Best Validation Score: {-results[best_model_name]['val_score']:.6f}")

# Main execution
if __name__ == "__main__":
    pipeline = PolymerPredictionPipeline()
    
    try:
        results = pipeline.run_complete_pipeline(
            main_train_path=MAIN_TRAIN_PATH,
            supplementary_map=SUPPLEMENTARY_DATA_MAP,
            test_path=TEST_PATH,
            submission_template_path=SUBMISSION_PATH
        )
        
        logger.info("Pipeline execution completed successfully!")
        logger.info("Check 'submission.csv' for final submission.")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.memory_monitor.force_cleanup()
        logger.info("Final memory cleanup completed.")




