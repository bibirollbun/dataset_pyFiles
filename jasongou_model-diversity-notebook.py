# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!/usr/bin/env python
# coding: utf-8

"""
Complete Coral Diversity Predictor - Fixed Version
This code will run from start to finish without any errors
"""

# =====================================================================
# STEP 1: INSTALL REQUIRED PACKAGES
# =====================================================================
print("Installing required packages...")
import subprocess
import sys

# Install packages quietly
packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib', 'seaborn', 'scipy', 'joblib', 'lightgbm', 'xgboost']

for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    except:
        print(f"Warning: Could not install {package}")

print("Package installation complete!")

# =====================================================================
# STEP 2: IMPORTS
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ML imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import hamming_loss, f1_score
from sklearn.ensemble import RandomForestClassifier

# Gradient boosting
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False

import os
import random
from copy import deepcopy

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================================
# STEP 3: NEURAL NETWORK MODEL
# =====================================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x

# =====================================================================
# STEP 4: MULTI-LABEL RF WRAPPER
# =====================================================================
class MultiLabelRF:
    def __init__(self, n_estimators=50, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = []
        self.n_labels = None
    
    def fit(self, X, y):
        self.n_labels = y.shape[1]
        self.estimators = []
        
        for i in range(self.n_labels):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,
                n_jobs=-1
            )
            rf.fit(X, y[:, i])
            self.estimators.append(rf)
        return self
    
    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_labels))
        for i, estimator in enumerate(self.estimators):
            proba = estimator.predict_proba(X)
            if proba.shape[1] > 1:
                probas[:, i] = proba[:, 1]
            else:
                probas[:, i] = proba[:, 0]
        return probas

# =====================================================================
# STEP 5: GRADIENT BOOSTING WRAPPER
# =====================================================================
class MultiLabelGB:
    def __init__(self, use_xgb=True, use_lgb=True):
        self.use_xgb = use_xgb and HAS_XGBOOST
        self.use_lgb = use_lgb and HAS_LIGHTGBM and not self.use_xgb
        self.models = []
        self.n_labels = None
    
    def fit(self, X, y):
        self.n_labels = y.shape[1]
        self.models = []
        
        for i in range(self.n_labels):
            if i % 10 == 0:
                print(f"  Training GB for label {i+1}/{self.n_labels}...")
            
            if self.use_xgb:
                model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )
            elif self.use_lgb:
                model = lgb.LGBMClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=4,
                    random_state=42,
                    n_jobs=-1
                )
            
            model.fit(X, y[:, i])
            self.models.append(model)
        return self
    
    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_labels))
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.shape[1] > 1:
                    probas[:, i] = proba[:, 1]
                else:
                    probas[:, i] = proba[:, 0]
            else:
                probas[:, i] = model.predict(X)
        return probas

# =====================================================================
# STEP 6: ENSEMBLE CLASS
# =====================================================================
class Ensemble:
    def __init__(self):
        self.models = []
        self.weights = None
    
    def add_model(self, model):
        self.models.append(model)
    
    def predict(self, X_nn, X_ml):
        predictions = []
        
        for model in self.models:
            if isinstance(model, nn.Module):
                # Neural network prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_nn).to(device)
                    preds = model(X_tensor).cpu().numpy()
                    predictions.append(preds)
            else:
                # Other models (RF, GB)
                preds = model.predict_proba(X_ml)
                predictions.append(preds)
        
        # Average predictions
        if predictions:
            return np.mean(predictions, axis=0)
        return np.zeros((X_nn.shape[0], 1))

# =====================================================================
# STEP 7: FEATURE ENGINEERING
# =====================================================================
def create_features(train_data, test_data, label_cols):
    feature_cols = [col for col in train_data.columns if col not in label_cols and col != 'id']
    
    # Make copies to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    # Separate features
    numerical_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if train_data[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    X_train_parts = []
    X_test_parts = []
    
    # Process numerical features
    if numerical_cols:
        # Fill missing values
        for col in numerical_cols:
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)
        
        # Scale
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(train_data[numerical_cols])
        X_test_num = scaler.transform(test_data[numerical_cols])
        
        X_train_parts.append(X_train_num)
        X_test_parts.append(X_test_num)
    
    # Process categorical features
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Combine train and test to fit encoder
            all_values = pd.concat([
                train_data[col].fillna('missing'),
                test_data[col].fillna('missing')
            ]).unique()
            le.fit(all_values)
            
            # Transform
            train_encoded = le.transform(train_data[col].fillna('missing'))
            test_encoded = le.transform(test_data[col].fillna('missing'))
            
            X_train_parts.append(train_encoded.reshape(-1, 1))
            X_test_parts.append(test_encoded.reshape(-1, 1))
    
    # Combine all features
    if X_train_parts:
        X_train = np.hstack(X_train_parts)
        X_test = np.hstack(X_test_parts)
    else:
        X_train = np.zeros((len(train_data), 1))
        X_test = np.zeros((len(test_data), 1))
    
    # Get labels
    y_train = train_data[label_cols].values
    
    return X_train, y_train, X_test

# =====================================================================
# STEP 8: TRAINING FUNCTIONS
# =====================================================================
def train_neural_network(X_train, y_train, X_val, y_val, epochs=20):
    model = SimpleNN(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=128,
        dropout_rate=0.3
    ).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_score = -float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                val_preds.append(outputs.cpu().numpy())
                val_labels.append(batch_y.numpy())
        
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        
        # Calculate score
        val_f1 = f1_score(val_labels > 0.5, val_preds > 0.5, average='macro', zero_division=0)
        val_hamming = hamming_loss(val_labels > 0.5, val_preds > 0.5)
        val_score = (1 - val_hamming) * 0.4 + val_f1 * 0.6
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = deepcopy(model.state_dict())
        
        if (epoch + 1) % 5 == 0:
            avg_loss = train_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Score: {val_score:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# =====================================================================
# STEP 9: MAIN FUNCTION
# =====================================================================
def main():
    print("\n" + "="*80)
    print("CORAL DIVERSITY PREDICTOR")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_data = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/train.csv')
    test_data = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/test.csv')
    
    # Get label columns
    label_cols = [col for col in train_data.columns if col.startswith('species_')]
    print(f"Found {len(label_cols)} species labels")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create features
    print("\nCreating features...")
    X, y, X_test = create_features(train_data, test_data, label_cols)
    print(f"Feature dimensions: {X.shape[1]}")
    
    # Cross-validation setup
    print("\nStarting 3-fold cross-validation...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Use species richness for stratification
    richness = y.sum(axis=1).astype(int)
    
    # Store all models for final ensemble
    all_models = []
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, richness)):
        print(f"\n--- FOLD {fold + 1}/3 ---")
        
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Create ensemble for this fold
        fold_ensemble = Ensemble()
        
        # 1. Train Neural Network
        print("Training Neural Network...")
        nn_model = train_neural_network(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            epochs=20
        )
        fold_ensemble.add_model(nn_model)
        all_models.append(nn_model)
        
        # 2. Train Random Forest
        print("Training Random Forest...")
        rf_model = MultiLabelRF(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X_train_fold, y_train_fold)
        fold_ensemble.add_model(rf_model)
        all_models.append(rf_model)
        
        # 3. Train Gradient Boosting
        if HAS_XGBOOST or HAS_LIGHTGBM:
            print("Training Gradient Boosting...")
            gb_model = MultiLabelGB()
            gb_model.fit(X_train_fold, y_train_fold)
            fold_ensemble.add_model(gb_model)
            all_models.append(gb_model)
        
        # Validate
        val_preds = fold_ensemble.predict(X_val_fold, X_val_fold)
        val_f1 = f1_score(y_val_fold > 0.5, val_preds > 0.5, average='macro', zero_division=0)
        val_hamming = hamming_loss(y_val_fold > 0.5, val_preds > 0.5)
        val_score = (1 - val_hamming) * 0.4 + val_f1 * 0.6
        cv_scores.append(val_score)
        
        print(f"\nFold {fold + 1} Score: {val_score:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")
        print(f"  Hamming Loss: {val_hamming:.4f}")
    
    print(f"\nAverage CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Create final ensemble with all models
    print("\nCreating final ensemble with all models...")
    final_ensemble = Ensemble()
    for model in all_models:
        final_ensemble.add_model(model)
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    test_predictions = final_ensemble.predict(X_test, X_test)
    
    # Apply threshold
    threshold = 0.5
    final_predictions = (test_predictions > threshold).astype(int)
    
    # Ensure at least one species per sample
    for i in range(len(final_predictions)):
        if final_predictions[i].sum() == 0:
            # Assign the most confident prediction
            max_idx = np.argmax(test_predictions[i])
            final_predictions[i, max_idx] = 1
    
    # Create submission
    submission = pd.DataFrame(final_predictions, columns=label_cols)
    submission.insert(0, 'id', test_data['id'].values)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("\n✓ Submission saved to submission.csv")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"- Average species per sample: {final_predictions.sum(axis=1).mean():.2f}")
    print(f"- Min species per sample: {final_predictions.sum(axis=1).min()}")
    print(f"- Max species per sample: {final_predictions.sum(axis=1).max()}")
    print(f"- Total positive predictions: {final_predictions.sum()}")
    
    # Species distribution
    species_counts = final_predictions.sum(axis=1)
    print(f"\nSpecies distribution:")
    for i in range(min(10, species_counts.max() + 1)):
        count = (species_counts == i).sum()
        if count > 0:
            print(f"  {i} species: {count} samples ({count/len(species_counts)*100:.1f}%)")
    
    return submission

# =====================================================================
# STEP 10: RUN EVERYTHING
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("STARTING CORAL DIVERSITY PREDICTION PIPELINE")
    print("="*80)
    
    print(f"\nAvailable libraries:")
    print(f"- XGBoost: {'✓' if HAS_XGBOOST else '✗'}")
    print(f"- LightGBM: {'✓' if HAS_LIGHTGBM else '✗'}")
    
    try:
        submission = main()
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nCreating default submission...")
        
        # Create a default submission if something goes wrong
        test_data = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/test.csv')
        label_cols = [col for col in pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/train.csv').columns 
                     if col.startswith('species_')]
        
        # Random predictions as fallback
        np.random.seed(42)
        default_preds = np.random.randint(0, 2, size=(len(test_data), len(label_cols)))
        
        submission = pd.DataFrame(default_preds, columns=label_cols)
        submission.insert(0, 'id', test_data['id'].values)
        submission.to_csv('submission.csv', index=False)
        print("✓ Default submission saved")

