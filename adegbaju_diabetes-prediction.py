# Install required packages
!pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna imbalanced-learn seaborn matplotlib plotly umap-learn
!pip install --upgrade xgboost lightgbm catboost optuna protobuf



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import traceback
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel

# Set random seed
SEED = 42
np.random.seed(SEED)



class Config:
    """Configuration parameters"""
    TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
    SUBMISSION_PATH = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    OUTPUT_DIR = 'output/'
    
    TEST_SIZE = 0.2
    N_SPLITS = 5
    CV_STRATEGY = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    LOG_LEVEL = logging.INFO

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.OUTPUT_DIR}pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimplePreprocessor:
    """Simple preprocessing for the competition"""
    
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        self.scaler = StandardScaler()
        
    def preprocess(self, train_df, test_df):
        """Main preprocessing method"""
        
        # Separate features and target
        X_train = train_df.drop(columns=['diagnosed_diabetes', 'id'])
        y_train = train_df['diagnosed_diabetes']
        X_test = test_df.drop(columns=['id'])
        
        print(f"Original shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Identify column types
        self.numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical columns: {len(self.numerical_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")
        
        # Handle missing values
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Numerical imputation
        if self.numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            X_train_processed[self.numerical_cols] = num_imputer.fit_transform(X_train[self.numerical_cols])
            X_test_processed[self.numerical_cols] = num_imputer.transform(X_test[self.numerical_cols])
        
        # Categorical imputation and encoding
        if self.categorical_cols:
            # Impute missing categorical values
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_processed[self.categorical_cols] = cat_imputer.fit_transform(X_train[self.categorical_cols])
            X_test_processed[self.categorical_cols] = cat_imputer.transform(X_test[self.categorical_cols])
            
            # One-hot encode categorical variables
            X_train_processed = pd.get_dummies(X_train_processed, columns=self.categorical_cols, drop_first=True)
            X_test_processed = pd.get_dummies(X_test_processed, columns=self.categorical_cols, drop_first=True)
            
            # Align columns (test might have different categories)
            X_train_processed, X_test_processed = X_train_processed.align(X_test_processed, join='left', axis=1, fill_value=0)
        
        # Scale numerical features
        numerical_cols_final = X_train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numerical_cols_final:
            X_train_processed[numerical_cols_final] = self.scaler.fit_transform(X_train_processed[numerical_cols_final])
            X_test_processed[numerical_cols_final] = self.scaler.transform(X_test_processed[numerical_cols_final])
        
        print(f"Processed shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        return X_train_processed, y_train, X_test_processed



class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        self.results = {}
        
    def train_models(self):
        """Train multiple models"""
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=-1),
            'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric='logloss', use_label_encoder=False, n_estimators=100),
            'LightGBM': LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, n_estimators=100),
            'CatBoost': CatBoostClassifier(random_state=SEED, verbose=0, thread_count=-1, iterations=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=SEED, n_estimators=100),
        }
        
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  # Use 3-fold for speed
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                           cv=cv, scoring='roc_auc', n_jobs=-1)
                
                # Train on full training set
                model.fit(self.X_train, self.y_train)
                
                self.models[name] = model
                self.results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'scores': cv_scores
                }
                
                print(f"  CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        # Display results
        print("\n" + "-"*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("-"*50)
        for name, scores in sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True):
            print(f"{name:20} ROC AUC: {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")
        
        return self.results
    
    def create_ensemble(self):
        """Create ensemble model"""
        
        # Get top 3 models
        top_models = sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        
        if len(top_models) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(ensemble, self.X_train, self.y_train, 
                                   cv=cv, scoring='roc_auc', n_jobs=-1)
        
        ensemble.fit(self.X_train, self.y_train)
        
        self.models['Ensemble'] = ensemble
        self.results['Ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"\nEnsemble ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return ensemble
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None
        
        best_name = max(self.results.items(), key=lambda x: x[1]['cv_mean'])[0]
        return self.models[best_name], best_name


class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    
    def evaluate(self, model, model_name):
        """Evaluate model on validation set"""
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_val, y_pred_proba)
        accuracy = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        
        print(f"\n{model_name} - Validation Results:")
        print(f"  ROC AUC:  {roc_auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(self.y_val, y_pred, model_name)
        
        # Plot ROC curve
        self.plot_roc_curve(self.y_val, y_pred_proba, model_name, roc_auc)
        
        return {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, roc_auc):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}roc_curve_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()



def run_pipeline():
    """Main pipeline execution"""
    
    print("="*70)
    print("DIABETES PREDICTION CHALLENGE")
    print("Playground Series - Season 5, Episode 12")
    print("="*70)
    
    try:
        # Step 1: Load data
        print("\nSTEP 1: LOADING DATA")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
        sample_submission = pd.read_csv(Config.SUBMISSION_PATH)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Target distribution:\n{train_df['diagnosed_diabetes'].value_counts()}")
        
        # Step 2: Preprocessing
        print("\nSTEP 2: PREPROCESSING")
        preprocessor = SimplePreprocessor()
        X_train, y_train, X_test = preprocessor.preprocess(train_df, test_df)
        
        # Step 3: Split data
        print("\nSTEP 3: SPLITTING DATA")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=Config.TEST_SIZE, 
            random_state=SEED,
            stratify=y_train
        )
        
        print(f"Training set: {X_train_split.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 4: Train models
        print("\nSTEP 4: TRAINING MODELS")
        trainer = ModelTrainer(X_train_split, y_train_split)
        results = trainer.train_models()
        
        # Step 5: Create ensemble
        print("\nSTEP 5: CREATING ENSEMBLE")
        ensemble = trainer.create_ensemble()
        
        # Step 6: Evaluate on validation set
        print("\nSTEP 6: VALIDATION")
        evaluator = Evaluator(X_val, y_val)
        
        # Evaluate best model
        best_model, best_name = trainer.get_best_model()
        if best_model:
            val_results = evaluator.evaluate(best_model, best_name)
        
        # Step 7: Train final model on full data
        print("\nSTEP 7: TRAINING FINAL MODEL")
        
        # Use ensemble if available, otherwise use best model
        if ensemble:
            final_model = ensemble
            final_name = 'Ensemble'
        else:
            final_model = best_model
            final_name = best_name
        
        # Train on full dataset
        final_model.fit(X_train, y_train)
        
        # Step 8: Make predictions
        print("\nSTEP 8: MAKING PREDICTIONS")
        test_predictions = final_model.predict_proba(X_test)[:, 1]
        
        # Step 9: Create submission
        print("\nSTEP 9: CREATING SUBMISSION")
        submission = pd.DataFrame({
            'id': test_df['id'],
            'diagnosed_diabetes': test_predictions
        })
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f'{Config.OUTPUT_DIR}submission_{final_name}_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        
        print(f"\nSubmission saved to: {submission_file}")
        print("\nSubmission sample:")
        print(submission.head())
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(test_predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Predictions', fontsize=14)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{Config.OUTPUT_DIR}prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        
        return submission, final_model
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        traceback.print_exc()
        return None, None



def quick_analysis():
    """Quick data analysis"""
    
    print("\nQUICK DATA ANALYSIS")
    print("="*50)
    
    train_df = pd.read_csv(Config.TRAIN_PATH)
    
    # Basic info
    print(f"\nDataset shape: {train_df.shape}")
    print(f"\nColumns: {train_df.columns.tolist()}")
    
    # Target distribution
    target_counts = train_df['diagnosed_diabetes'].value_counts()
    print(f"\nTarget distribution:")
    print(f"  0 (No Diabetes): {target_counts[0]} ({target_counts[0]/len(train_df)*100:.1f}%)")
    print(f"  1 (Diabetes):    {target_counts[1]} ({target_counts[1]/len(train_df)*100:.1f}%)")
    
    # Numerical features
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'diagnosed_diabetes']]
    
    print(f"\nNumerical features ({len(numerical_cols)}):")
    print(numerical_cols[:10])
    
    # Categorical features
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nCategorical features ({len(categorical_cols)}):")
    print(categorical_cols)
    
    # Check for missing values
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\nMissing values found:")
        print(missing)
    else:
        print("\nNo missing values found.")
    
    return train_df


def analyze_feature_importance(model, X_train, y_train):
    """Analyze feature importance"""
    
    print("\nFEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Train a simple Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_names = X_train.columns
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                   color='skyblue', alpha=0.7, edgecolor='black')
    
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances', fontsize=14)
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        plt.text(importance * 1.01, i, f'{importance:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{Config.OUTPUT_DIR}feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    return importance_df



if __name__ == "__main__":
    
    print("Choose an option:")
    print("1. Run complete pipeline")
    print("2. Quick data analysis")
    print("3. Run with XGBoost only (fast)")
    
    try:
        choice = int(input("\nEnter your choice (1-3): "))
    except:
        choice = 1
    
    if choice == 1:
        # Run complete pipeline
        submission, model = run_pipeline()
        
    elif choice == 2:
        # Quick analysis
        train_df = quick_analysis()
        
        # Run a simple model
        print("\n\nTraining simple model for feature importance...")
        
        # Preprocess
        test_df = pd.read_csv(Config.TEST_PATH)
        preprocessor = SimplePreprocessor()
        X_train, y_train, X_test = preprocessor.preprocess(train_df, test_df)
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(None, X_train, y_train)
        
        # Train and save a simple model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        test_predictions = model.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            'id': test_df['id'],
            'diagnosed_diabetes': test_predictions
        })
        
        submission_file = f'{Config.OUTPUT_DIR}quick_analysis_submission.csv'
        submission.to_csv(submission_file, index=False)
        print(f"\nSubmission saved to: {submission_file}")
        
    elif choice == 3:
        # Fast XGBoost only
        print("\nRunning XGBoost only (fast)...")
        
        # Load data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
        
        # Simple preprocessing
        X_train = train_df.drop(columns=['diagnosed_diabetes', 'id'])
        y_train = train_df['diagnosed_diabetes']
        X_test = test_df.drop(columns=['id'])
        
        # One-hot encode categoricals
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        test_predictions = model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'diagnosed_diabetes': test_predictions
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f'{Config.OUTPUT_DIR}xgboost_fast_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        
        print(f"\nSubmission saved to: {submission_file}")
        print("\nDone!")
    
    else:
        print("Invalid choice. Running complete pipeline...")
        submission, model = run_pipeline()

