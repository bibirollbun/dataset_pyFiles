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

# Import all libraries (skip installations to avoid version conflicts)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, IsolationForest, VotingClassifier,
                            AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import entropy, mode
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedFeatureEngineering:
    """Advanced feature engineering with domain knowledge"""
    
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def fit(self, X, feature_names):
        self.feature_names = feature_names
        self.poly_features.fit(X)
        return self
        
    def transform(self, X):
        # Original features
        features = [X]
        
        # Domain-specific features for maternal health
        age = X[:, 0]
        systolic = X[:, 1]
        diastolic = X[:, 2]
        blood_glucose = X[:, 3]
        body_temp = X[:, 4]
        heart_rate = X[:, 5]
        
        # Blood pressure features
        pulse_pressure = systolic - diastolic
        mean_arterial_pressure = diastolic + (pulse_pressure / 3)
        bp_ratio = systolic / (diastolic + 1e-10)
        hypertension_index = ((systolic > 140) | (diastolic > 90)).astype(float)
        
        # Age-related risk factors
        age_risk = ((age < 20) | (age > 35)).astype(float)
        young_mother = (age < 20).astype(float)
        advanced_maternal_age = (age > 35).astype(float)
        
        # Metabolic features
        glucose_abnormal = ((blood_glucose < 4.0) | (blood_glucose > 7.0)).astype(float)
        temp_abnormal = ((body_temp < 36.0) | (body_temp > 37.5)).astype(float)
        tachycardia = (heart_rate > 100).astype(float)
        bradycardia = (heart_rate < 60).astype(float)
        
        # Combined risk indicators
        cardiovascular_stress = np.sqrt((systolic - 120) ** 2 + (diastolic - 80) ** 2)
        metabolic_risk = glucose_abnormal + temp_abnormal
        
        # Interaction features
        age_bp_interaction = age * mean_arterial_pressure / 1000  # Scale down
        glucose_temp_interaction = blood_glucose * body_temp / 100  # Scale down
        hr_bp_interaction = heart_rate * pulse_pressure / 1000  # Scale down
        
        # Add engineered features
        engineered = np.column_stack([
            pulse_pressure,
            mean_arterial_pressure,
            bp_ratio,
            hypertension_index,
            age_risk,
            young_mother,
            advanced_maternal_age,
            glucose_abnormal,
            temp_abnormal,
            tachycardia,
            bradycardia,
            cardiovascular_stress,
            metabolic_risk,
            age_bp_interaction,
            glucose_temp_interaction,
            hr_bp_interaction
        ])
        
        features.append(engineered)
        
        # Polynomial features for original features only
        poly_feats = self.poly_features.transform(X)[:, len(self.feature_names):]  # Exclude original features
        features.append(poly_feats)
        
        # Combine all features
        X_enhanced = np.hstack(features)
        
        return X_enhanced
    
    def fit_transform(self, X, feature_names):
        return self.fit(X, feature_names).transform(X)

class SimplifiedMetaLearner:
    """Simplified meta-learner for stacking ensemble"""
    
    def __init__(self):
        self.models = {
            'rf_meta': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'lr_meta': LogisticRegression(
                C=1.0,
                solver='lbfgs',
                multi_class='multinomial',
                max_iter=1000,
                random_state=42
            )
        }
        self.weights = {'rf_meta': 0.7, 'lr_meta': 0.3}
        
    def fit(self, X, y):
        """Fit all meta-learners"""
        for name, model in self.models.items():
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Weighted prediction from all meta-learners"""
        predictions = np.zeros((X.shape[0], 3))
        
        for name, model in self.models.items():
            weight = self.weights[name]
            predictions += weight * model.predict_proba(X)
        
        return predictions
    
    def predict(self, X):
        """Get class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class OptimizedEnsemble:
    """Optimized ensemble with advanced techniques"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.base_models = {}
        self.calibrators = {}
        self.meta_learner = SimplifiedMetaLearner()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.feature_selector = SelectKBest(score_func=f_classif, k='all')
        self.cv_scores = {}
        self.class_weights = None
        
    def create_diverse_models(self):
        """Create a diverse set of optimized models"""
        self.base_models = {
            # Tree-based models
            'rf1': RandomForestClassifier(
                n_estimators=500, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt',
                random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'rf2': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='log2',
                random_state=123, n_jobs=-1, class_weight='balanced'
            ),
            'xgb': XGBClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=42, n_jobs=-1,
                use_label_encoder=False, eval_metric='mlogloss'
            ),
            'lgbm': LGBMClassifier(
                n_estimators=500, num_leaves=50, learning_rate=0.02,
                feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=5, random_state=42,
                verbosity=-1, class_weight='balanced'
            ),
            'cat': CatBoostClassifier(
                iterations=500, depth=10, learning_rate=0.02,
                l2_leaf_reg=3, random_state=42, verbose=False,
                auto_class_weights='Balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.03,
                subsample=0.8, min_samples_split=2,
                min_samples_leaf=1, random_state=42
            ),
            'et': ExtraTreesClassifier(
                n_estimators=500, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, random_state=42, 
                n_jobs=-1, class_weight='balanced'
            ),
            'ada': AdaBoostClassifier(
                n_estimators=300, learning_rate=0.5,
                random_state=42
            ),
            
            # Other algorithms
            'svm': SVC(
                kernel='rbf', C=50, gamma='scale',
                probability=True, random_state=42,
                class_weight='balanced'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(150, 100, 50),
                activation='relu', solver='adam',
                alpha=0.0001, batch_size=32,
                learning_rate='adaptive', max_iter=1000,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=15, weights='distance',
                metric='minkowski', p=2
            ),
            'lr': LogisticRegression(
                C=10.0, solver='newton-cg',
                multi_class='multinomial',
                max_iter=1000, random_state=42,
                class_weight='balanced'
            ),
            'lda': LinearDiscriminantAnalysis(),
            'nb': GaussianNB()
        }
    
    def calculate_class_weights(self, y):
        """Calculate class weights for balancing"""
        unique_classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = {}
        for cls, count in zip(unique_classes, counts):
            weights[cls] = total / (len(unique_classes) * count)
        return weights
    
    def create_balanced_sample_weights(self, y):
        """Create sample weights for balancing"""
        class_weights = self.calculate_class_weights(y)
        sample_weights = np.zeros(len(y))
        for i, label in enumerate(y):
            sample_weights[i] = class_weights[label]
        return sample_weights
    
    def advanced_blend(self, predictions_dict):
        """Advanced blending strategy"""
        model_names = list(predictions_dict.keys())
        n_samples = len(predictions_dict[model_names[0]]['pred'])
        
        # Extract data
        pred_matrix = np.array([predictions_dict[m]['pred'] for m in model_names]).T
        proba_matrix = np.array([predictions_dict[m]['proba'] for m in model_names])
        conf_matrix = np.array([predictions_dict[m]['conf'] for m in model_names]).T
        
        # Calculate model agreement
        mode_pred, mode_count = mode(pred_matrix, axis=1)
        mode_pred = mode_pred.flatten()
        mode_count = mode_count.flatten()
        agreement_ratio = mode_count / len(model_names)
        
        # Initialize blended predictions
        blended_proba = np.zeros((n_samples, self.n_classes))
        
        # High agreement: use weighted average
        high_agreement_mask = agreement_ratio > 0.6
        
        # Low agreement: use uncertainty-aware blending
        low_agreement_mask = ~high_agreement_mask
        
        # Process high agreement samples
        for i in np.where(high_agreement_mask)[0]:
            weights = conf_matrix[i] ** 2
            weights = weights / weights.sum()
            for j, model in enumerate(model_names):
                blended_proba[i] += weights[j] * proba_matrix[j, i]
        
        # Process low agreement samples with smoothing
        if np.any(low_agreement_mask):
            low_indices = np.where(low_agreement_mask)[0]
            
            for i in low_indices:
                # Calculate uncertainty for each model
                uncertainties = np.array([
                    entropy(proba_matrix[j, i] + 1e-10) 
                    for j in range(len(model_names))
                ])
                
                # Penalize high uncertainty
                uncertainty_penalty = np.exp(-uncertainties)
                weights = conf_matrix[i] * uncertainty_penalty
                
                # Add model score weighting
                if hasattr(self, 'model_scores'):
                    score_weights = np.array([self.model_scores.get(m, 0.5) for m in model_names])
                    weights = weights * score_weights
                
                weights = weights / weights.sum()
                
                for j, model in enumerate(model_names):
                    blended_proba[i] += weights[j] * proba_matrix[j, i]
        
        return blended_proba
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Fit ensemble with optimizations"""
        self.create_diverse_models()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(y_train_encoded)
        
        # Feature engineering
        print("Engineering features...")
        feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'Blood glucose', 'BodyTemp', 'HeartRate']
        X_train_enhanced = self.feature_engineer.fit_transform(X_train, feature_names)
        X_val_enhanced = self.feature_engineer.transform(X_val)
        
        # Feature selection
        print("Selecting best features...")
        self.feature_selector.fit(X_train_enhanced, y_train_encoded)
        scores = self.feature_selector.scores_
        
        # Select top features
        k = min(30, X_train_enhanced.shape[1])
        top_features_idx = np.argsort(scores)[-k:]
        
        X_train_selected = X_train_enhanced[:, top_features_idx]
        X_val_selected = X_val_enhanced[:, top_features_idx]
        self.selected_features_idx = top_features_idx
        
        # Create sample weights for training
        sample_weights = self.create_balanced_sample_weights(y_train_encoded)
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        
        # Train base models
        print("Training optimized models...")
        self.model_scores = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            try:
                # Train with sample weights where supported
                if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    model.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)
                else:
                    model.fit(X_train_scaled, y_train_encoded)
                
                # Calibrate predictions
                self.calibrators[name] = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                self.calibrators[name].fit(X_train_scaled, y_train_encoded)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val_scaled)
                val_score = accuracy_score(y_val_encoded, val_pred)
                self.model_scores[name] = val_score
                print(f"  {name} validation accuracy: {val_score:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                self.calibrators[name] = model
                self.model_scores[name] = 0.5
        
        # Create meta-features
        print("\nPreparing meta-features...")
        meta_features_train = self.create_meta_features(X_train_scaled, y_train_encoded)
        meta_features_val = self.create_meta_features(X_val_scaled, y_val_encoded)
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_learner.fit(meta_features_train, y_train_encoded)
        
        # Optimize thresholds for each class
        self.optimize_thresholds(meta_features_val, y_val_encoded)
        
        return self
    
    def create_meta_features(self, X, y=None):
        """Create meta-features from base model predictions"""
        predictions_dict = self.get_all_predictions(X)
        model_names = list(predictions_dict.keys())
        
        # Base predictions and probabilities
        meta_features = []
        
        # Add probabilities
        for name in model_names:
            meta_features.append(predictions_dict[name]['proba'])
        
        # Add confidence scores
        for name in model_names:
            meta_features.append(predictions_dict[name]['conf'].reshape(-1, 1))
        
        # Add blended predictions
        blended_proba = self.advanced_blend(predictions_dict)
        meta_features.append(blended_proba)
        
        # Model agreement features
        pred_matrix = np.array([predictions_dict[m]['pred'] for m in model_names]).T
        mode_pred, mode_count = mode(pred_matrix, axis=1)
        agreement_ratio = mode_count.flatten() / len(model_names)
        meta_features.append(agreement_ratio.reshape(-1, 1))
        
        # Prediction diversity
        pred_diversity = np.array([len(np.unique(pred_matrix[i])) for i in range(len(pred_matrix))])
        meta_features.append(pred_diversity.reshape(-1, 1) / 3)  # Normalize by n_classes
        
        # Stack all features
        return np.hstack(meta_features)
    
    def optimize_thresholds(self, X_val, y_val):
        """Optimize prediction thresholds for better class balance"""
        val_proba = self.meta_learner.predict_proba(X_val)
        
        # Calculate per-class F1 scores and adjust
        self.class_thresholds = np.ones(self.n_classes)
        
        for class_idx in range(self.n_classes):
            y_binary = (y_val == class_idx).astype(int)
            
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.linspace(0.3, 0.7, 20):
                y_pred = (val_proba[:, class_idx] > threshold).astype(int)
                f1 = f1_score(y_binary, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.class_thresholds[class_idx] = best_threshold
            print(f"  Class {self.label_encoder.inverse_transform([class_idx])[0]}: threshold={best_threshold:.3f}, F1={best_f1:.3f}")
    
    def get_all_predictions(self, X):
        """Get predictions from all models"""
        predictions_dict = {}
        
        for name, model in self.calibrators.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                    
                    predictions_dict[name] = {
                        'pred': pred,
                        'proba': proba,
                        'conf': conf
                    }
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")
        
        return predictions_dict
    
    def predict_proba(self, X):
        """Get probability predictions"""
        # Feature engineering and scaling
        X_enhanced = self.feature_engineer.transform(X)
        X_selected = X_enhanced[:, self.selected_features_idx]
        X_scaled = self.scaler.transform(X_selected)
        
        # Create meta-features
        meta_features = self.create_meta_features(X_scaled)
        
        # Get meta-learner predictions
        meta_proba = self.meta_learner.predict_proba(meta_features)
        
        # Apply threshold optimization
        if hasattr(self, 'class_thresholds'):
            for i in range(self.n_classes):
                meta_proba[:, i] = meta_proba[:, i] / self.class_thresholds[i]
            
            # Renormalize
            meta_proba = meta_proba / meta_proba.sum(axis=1, keepdims=True)
        
        return meta_proba
    
    def predict(self, X):
        """Make final predictions"""
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(predictions)

def analyze_predictions(ensemble, X_val, y_val):
    """Analyze ensemble predictions"""
    predictions = ensemble.predict(X_val)
    probabilities = ensemble.predict_proba(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    
    # Per-class analysis
    print("\nPer-Class Performance:")
    print("-" * 50)
    
    for i, class_name in enumerate(ensemble.label_encoder.classes_):
        class_mask = y_val == class_name
        if np.any(class_mask):
            class_acc = accuracy_score(y_val[class_mask], predictions[class_mask])
            class_conf = np.mean(probabilities[class_mask, i])
            print(f"{class_name}: Accuracy={class_acc:.3f}, Avg Confidence={class_conf:.3f}")

# Main execution
def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Prepare data
    feature_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'Blood glucose', 'BodyTemp', 'HeartRate']
    X = train_df[feature_cols].values
    y = train_df['RiskLevel'].values
    X_test = test_df[feature_cols].values
    
    # Convert numeric labels to strings
    risk_mapping = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
    if y.dtype in [np.int64, np.float64]:
        y = np.array([risk_mapping[val] for val in y])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Create and train ensemble
    print("\nCreating optimized ensemble...")
    ensemble = OptimizedEnsemble(n_classes=3)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n" + "="*60)
    print("VALIDATION PERFORMANCE")
    print("="*60)
    
    val_pred = ensemble.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\nOverall Accuracy: {val_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, val_pred)
    print(cm)
    
    # Analyze predictions
    analyze_predictions(ensemble, X_val, y_val)
    
    # Make test predictions
    print("\n" + "="*60)
    print("TEST SET PREDICTIONS")
    print("="*60)
    
    test_predictions = ensemble.predict(X_test)
    test_proba = ensemble.predict_proba(X_test)
    test_confidence = np.max(test_proba, axis=1)
    
    # Distribution
    print("\nPrediction Distribution:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} ({100*c/len(test_predictions):.1f}%)")
    
    print(f"\nConfidence Stats:")
    print(f"  Mean: {np.mean(test_confidence):.3f}")
    print(f"  Std:  {np.std(test_confidence):.3f}")
    print(f"  Min:  {np.min(test_confidence):.3f}")
    print(f"  Max:  {np.max(test_confidence):.3f}")
    
    # High uncertainty predictions
    uncertainty = entropy(test_proba.T + 1e-10)
    high_uncertainty = uncertainty > 0.8
    print(f"\nHigh uncertainty predictions: {np.sum(high_uncertainty)} ({100*np.mean(high_uncertainty):.1f}%)")
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'RiskLevel': test_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'Id': test_df['Id'],
        'Prediction': test_predictions,
        'Confidence': test_confidence,
        'Uncertainty': uncertainty,
        'Low_Risk_Prob': test_proba[:, 0],
        'Mid_Risk_Prob': test_proba[:, 1],
        'High_Risk_Prob': test_proba[:, 2]
    })
    
    analysis_df.to_csv('prediction_analysis.csv', index=False)
    print("Detailed analysis saved to prediction_analysis.csv")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Confidence distribution
    plt.subplot(1, 3, 1)
    plt.hist(test_confidence, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    
    # Class probabilities
    plt.subplot(1, 3, 2)
    for i, class_name in enumerate(['Low Risk', 'Mid Risk', 'High Risk']):
        plt.hist(test_proba[:, i], bins=30, alpha=0.5, label=class_name)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Class Probability Distributions')
    plt.legend()
    
    # Uncertainty by predicted class
    plt.subplot(1, 3, 3)
    for class_name in ['Low Risk', 'Mid Risk', 'High Risk']:
        mask = test_predictions == class_name
        if np.any(mask):
            plt.scatter(np.where(mask)[0], uncertainty[mask], 
                       alpha=0.5, s=10, label=class_name)
    plt.xlabel('Sample Index')
    plt.ylabel('Uncertainty')
    plt.title('Prediction Uncertainty by Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ensemble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("ENSEMBLE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

