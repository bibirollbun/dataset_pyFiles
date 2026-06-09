"""
California Homelessness Prediction Solution - Version
Optimized for small datasets with proper R² estimation and robust validation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Preprocessing and Model Selection
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Models optimized for small datasets
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                 HuberRegressor, LinearRegression)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility
np.random.seed(42)

class ImprovedHomelessnessPredictor:
    """
    Enhanced predictor with proper R² estimation and robust validation
    """
    
    def __init__(self):
        self.feature_selector = None
        self.scaler = None
        self.models = {}
        self.feature_names = []
        self.cv_results = {}
        self.trained_models = {}
        
    def load_data(self):
        """Load and prepare the data"""
        print("Loading data...")
        self.train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
        self.test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')
        self.sample_sub = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/sample_submission.csv')
        
        # Separate components
        self.train_ids = self.train['ID'].values
        self.test_ids = self.test['ID'].values
        self.y = self.train['HOMELESS_RATE'].values
        self.X = self.train.drop(['ID', 'HOMELESS_RATE'], axis=1)
        self.X_test = self.test.drop(['ID'], axis=1)
        
        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}")
        print(f"Zero targets: {np.sum(self.y == 0)}/{len(self.y)}")
        
        # Basic target statistics
        print(f"\nTarget statistics:")
        print(f"Mean: {np.mean(self.y):.6f}")
        print(f"Std: {np.std(self.y):.6f}")
        print(f"Min: {np.min(self.y):.6f}")
        print(f"Max: {np.max(self.y):.6f}")
        print(f"Skewness: {stats.skew(self.y):.3f}")
        
    def create_domain_features(self, X):
        """Create domain-informed features based on homelessness research"""
        X_new = X.copy()
        
        # Vulnerability indicators
        X_new['age_vulnerability'] = (X['AGE_U18_PCT'] + 
                                    X['AGE_65_69_PCT'] + 
                                    X['AGE_70_79_PCT'] + 
                                    X['AGE_80_PLUS_PCT'])
        
        X_new['working_age_pop'] = (X['AGE_25_34_PCT'] + 
                                   X['AGE_35_44_PCT'] + 
                                   X['AGE_45_54_PCT'])
        
        # Housing instability
        X_new['single_households'] = (X['NONFAMILY_SINGLE_MALE_PCT'] + 
                                     X['NONFAMILY_SINGLE_FEMALE_PCT'])
        
        # Social isolation
        X_new['social_isolation'] = X['INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']
        
        # Economic vulnerability proxies
        X_new['minority_population'] = 100 - X['RACE_WHITE_NH_PCT']
        X_new['disability_burden'] = X['DISABILITY_POP_PCT']
        
        # Protective factors
        safe_denominator = X['FAMILY_HH_TOTAL'] + 1e-8
        X_new['family_stability'] = X['FAMILY_HH_CHILD_LT18_PCT'] / safe_denominator
        
        # Veterans with disabilities (high-risk interaction)
        X_new['veteran_disability_risk'] = X['VETERAN_POP_PCT'] * X['DISABILITY_POP_PCT'] / 100
        
        return X_new
    
    def prepare_features(self):
        """Feature engineering and selection"""
        print("\nPreparing features...")
        
        # Apply domain feature engineering
        X_eng = self.create_domain_features(self.X)
        X_test_eng = self.create_domain_features(self.X_test)
        
        print(f"Features after engineering: {X_eng.shape[1]}")
        
        # Remove highly correlated features (>0.95)
        corr_matrix = X_eng.corr()
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # Remove second feature in each highly correlated pair
        features_to_drop = []
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat2 not in features_to_drop:
                features_to_drop.append(feat2)
                print(f"Removing {feat2} (corr with {feat1}: {corr_val:.3f})")
        
        X_eng = X_eng.drop(columns=features_to_drop)
        X_test_eng = X_test_eng.drop(columns=features_to_drop)
        
        # Feature selection - conservative approach for small dataset
        max_features = min(15, X_eng.shape[1] - 2)  # Very conservative
        
        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_regression, k=max_features)
        X_selected = selector.fit_transform(X_eng, self.y)
        X_test_selected = selector.transform(X_test_eng)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.feature_names = X_eng.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.feature_names)} features")
        print("Selected features:", self.feature_names[:10])
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_final = self.scaler.fit_transform(X_selected)
        self.X_test_final = self.scaler.transform(X_test_selected)
        
        print(f"Final feature matrix shape: {self.X_final.shape}")
        
    def define_models(self):
        """Define models optimized for small datasets"""
        print("\nDefining models...")
        
        self.model_configs = {
            'linear_regression': LinearRegression(),
            
            'ridge': Ridge(alpha=1.0),
            
            'lasso': Lasso(alpha=0.001, max_iter=2000),
            
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
            
            'bayesian_ridge': BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
            ),
            
            'huber': HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01),
            
            'svr_linear': SVR(kernel='linear', C=1.0, epsilon=0.001),
            
            'svr_rbf': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.001),
            
            'knn': KNeighborsRegressor(
                n_neighbors=min(5, len(self.X_final)//4), 
                weights='distance'
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=50, max_depth=3, min_samples_split=8,
                min_samples_leaf=4, random_state=42
            ),
            
            'lgbm': LGBMRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                num_leaves=7, min_child_samples=8, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbose=-1
            )
        }
        
        print(f"Defined {len(self.model_configs)} models")
    
    def calculate_r2_manually(self, y_true, y_pred):
        """Calculate R² manually to avoid NaN issues"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0  # Perfect prediction case
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def evaluate_models_loo(self):
        """Evaluate models using Leave-One-Out Cross-Validation with proper R² calculation"""
        print("\nEvaluating models with Leave-One-Out CV...")
        
        loo = LeaveOneOut()
        
        for name, model in self.model_configs.items():
            print(f"Evaluating {name}...", end=" ")
            
            try:
                y_pred_loo = np.zeros(len(self.y))
                
                # Perform LOO manually to get proper predictions
                for train_idx, val_idx in loo.split(self.X_final):
                    X_train_fold, X_val_fold = self.X_final[train_idx], self.X_final[val_idx]
                    y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
                    
                    # Train model on fold
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_loo[val_idx] = model.predict(X_val_fold)
                
                # Calculate metrics
                mse = mean_squared_error(self.y, y_pred_loo)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y, y_pred_loo)
                r2 = self.calculate_r2_manually(self.y, y_pred_loo)
                
                self.cv_results[name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mse': mse,
                    'predictions': y_pred_loo
                }
                
                # Train on full dataset for final model
                model.fit(self.X_final, self.y)
                self.trained_models[name] = model
                
                print(f"RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def create_ensemble(self):
        """Create ensemble based on performance metrics"""
        print("\nCreating ensemble...")
        
        # Sort models by RMSE (primary) and R² (secondary)
        model_performance = []
        for name, results in self.cv_results.items():
            # Composite score: lower RMSE is better, higher R² is better
            # Normalize and combine
            composite_score = results['rmse'] - results['r2'] * 0.001  # Small weight for R²
            model_performance.append((name, composite_score, results))
        
        model_performance.sort(key=lambda x: x[1])  # Sort by composite score
        
        # Select top models for ensemble
        n_ensemble = min(5, len(model_performance))
        top_models = model_performance[:n_ensemble]
        
        print(f"\nTop {n_ensemble} models:")
        ensemble_predictions = []
        weights = []
        
        for name, score, results in top_models:
            print(f"  {name}: RMSE={results['rmse']:.6f}, R²={results['r2']:.4f}")
            
            # Get predictions from trained model
            pred = self.trained_models[name].predict(self.X_test_final)
            ensemble_predictions.append(pred)
            
            # Weight by inverse RMSE
            weight = 1.0 / (results['rmse'] + 1e-8)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        print(f"\nEnsemble weights:")
        for (name, _, _), w in zip(top_models, weights):
            print(f"  {name}: {w:.4f}")
        
        # Create weighted ensemble prediction
        ensemble_pred = np.average(ensemble_predictions, weights=weights, axis=0)
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        # Apply reasonable upper bound
        upper_bound = np.percentile(self.y, 99) * 1.5
        ensemble_pred = np.minimum(ensemble_pred, upper_bound)
        
        return ensemble_pred, top_models, weights
    
    def create_submission(self, predictions):
        """Create submission file"""
        print("\nCreating submission...")
        
        submission = pd.DataFrame({
            'ID': self.test_ids,
            'HOMELESS_RATE': predictions
        })
        
        submission.to_csv('improved_submission.csv', index=False)
        
        print(f"Submission statistics:")
        print(f"  Mean: {np.mean(predictions):.6f}")
        print(f"  Std: {np.std(predictions):.6f}")
        print(f"  Min: {np.min(predictions):.6f}")
        print(f"  Max: {np.max(predictions):.6f}")
        print(f"  Zeros: {np.sum(predictions == 0)}")
        
        print(f"\nComparison to training data:")
        print(f"  Train mean: {np.mean(self.y):.6f} vs Pred mean: {np.mean(predictions):.6f}")
        print(f"  Train std: {np.std(self.y):.6f} vs Pred std: {np.std(predictions):.6f}")
        
        return submission
    
    def visualize_results(self, predictions, top_models, weights):
        """Create visualization of results"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution comparison
        axes[0, 0].hist(self.y, bins=20, alpha=0.7, label='Training', color='blue', edgecolor='black')
        axes[0, 0].hist(predictions, bins=20, alpha=0.7, label='Predictions', color='red', edgecolor='black')
        axes[0, 0].set_title('Target vs Prediction Distribution')
        axes[0, 0].set_xlabel('Homeless Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. Model performance comparison
        model_names = list(self.cv_results.keys())
        rmse_values = [self.cv_results[name]['rmse'] for name in model_names]
        r2_values = [self.cv_results[name]['r2'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        axes[0, 1].bar(x_pos, rmse_values, alpha=0.7, color='orange')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].set_title('Model RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        
        # 3. R² comparison
        axes[0, 2].bar(x_pos, r2_values, alpha=0.7, color='green')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].set_title('Model R² Comparison')
        axes[0, 2].set_ylabel('R² Score')
        
        # 4. Feature importance (if available)
        if 'random_forest' in self.trained_models:
            rf_model = self.trained_models['random_forest']
            importances = rf_model.feature_importances_
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:]
            axes[1, 0].barh(range(len(top_indices)), importances[top_indices])
            axes[1, 0].set_yticks(range(len(top_indices)))
            axes[1, 0].set_yticklabels([self.feature_names[i] for i in top_indices])
            axes[1, 0].set_title('Top 10 Feature Importances')
            axes[1, 0].set_xlabel('Importance')
        
        # 5. Ensemble weights
        ensemble_names = [name for name, _, _ in top_models]
        axes[1, 1].pie(weights, labels=ensemble_names, autopct='%1.1f%%')
        axes[1, 1].set_title('Ensemble Model Weights')
        
        # 6. Prediction vs actual scatter (using LOO predictions)
        best_model_name = min(self.cv_results.items(), key=lambda x: x[1]['rmse'])[0]
        best_predictions = self.cv_results[best_model_name]['predictions']
        
        axes[1, 2].scatter(self.y, best_predictions, alpha=0.6)
        axes[1, 2].plot([0, np.max(self.y)], [0, np.max(self.y)], 'r--', label='Perfect prediction')
        axes[1, 2].set_xlabel('Actual Homeless Rate')
        axes[1, 2].set_ylabel('Predicted Homeless Rate')
        axes[1, 2].set_title(f'Actual vs Predicted ({best_model_name})')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('improved_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        print("="*60)
        print("IMPROVED CALIFORNIA HOMELESSNESS PREDICTION")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Feature engineering
        self.prepare_features()
        
        # Step 3: Define models
        self.define_models()
        
        # Step 4: Evaluate with proper R² calculation
        self.evaluate_models_loo()
        
        # Step 5: Create ensemble
        predictions, top_models, weights = self.create_ensemble()
        
        # Step 6: Create submission
        submission = self.create_submission(predictions)
        
        # Step 7: Visualize results
        self.visualize_results(predictions, top_models, weights)
        
        # Step 8: Final report
        self.print_final_report(top_models)
        
        return submission
    
    def print_final_report(self, top_models):
        """Print comprehensive final report"""
        print("\n" + "="*60)
        print("FINAL SOLUTION REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"  Training samples: {len(self.y)}")
        print(f"  Test samples: {len(self.test_ids)}")
        print(f"  Features used: {len(self.feature_names)}")
        print(f"  Zero targets in training: {np.sum(self.y == 0)}")
        
        print(f"\nModel Performance (Leave-One-Out CV):")
        best_model = min(self.cv_results.items(), key=lambda x: x[1]['rmse'])
        print(f"  Best model: {best_model[0]}")
        print(f"  Best RMSE: {best_model[1]['rmse']:.6f}")
        print(f"  Best R²: {best_model[1]['r2']:.4f}")
        print(f"  Best MAE: {best_model[1]['mae']:.6f}")
        
        print(f"\nEnsemble Information:")
        print(f"  Models in ensemble: {len(top_models)}")
        print(f"  Validation method: Leave-One-Out Cross-Validation")
        print(f"  Weighting: Inverse RMSE")
        
        print(f"\nKey Improvements:")
        print(f"  ✓ Proper R² calculation (no NaN values)")
        print(f"  ✓ Leave-One-Out CV for reliable small dataset validation")
        print(f"  ✓ Conservative feature engineering")
        print(f"  ✓ Domain-informed feature creation")
        print(f"  ✓ Robust ensemble weighting")
        print(f"  ✓ Non-negative prediction constraints")
        
        print(f"\nTop Features:")
        for i, feature in enumerate(self.feature_names[:8], 1):
            print(f"  {i:2d}. {feature}")
        
        print(f"\nOutput Files:")
        print(f"  - improved_submission.csv (submission file)")
        print(f"  - improved_analysis.png (analysis plots)")
        
        print(f"\nPipeline completed successfully!")

# Main execution
if __name__ == "__main__":
    predictor = ImprovedHomelessnessPredictor()
    submission = predictor.run_pipeline()

