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


"""
Plugin Examples - Custom Models and Agents
===========================================
This file demonstrates how to create custom models and agents that will be
automatically discovered and used by the pipeline without any manual registration.

Simply save this file in your plugins directory and they will be auto-discovered!
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Import base classes from main pipeline
# In practice: from ultimate_pipeline_autodiscovery import AutoDiscoverableModel, AutoDiscoverableAgent, AutoDiscoverableTransformer, AutoMLTool

logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM MODEL EXAMPLES
# ============================================================================

# Example 1: Custom Neural Network Model
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    class CustomNeuralNetworkModel(AutoDiscoverableModel):
        """Custom neural network that will be auto-discovered"""
        
        model_type = "neural_network"
        complexity = "high"
        handles_missing = False
        handles_categorical = False
        interpretable = False
        scalable = True
        requires_scaling = True
        estimated_training_time = "medium"
        
        def _build_model(self):
            """Build a PyTorch neural network"""
            
            class SimpleNN(nn.Module):
                def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1):
                    super().__init__()
                    
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2)
                        ])
                        prev_dim = hidden_dim
                    
                    layers.append(nn.Linear(prev_dim, output_dim))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x).squeeze()
            
            # Will be initialized when we know input dimension
            return SimpleNN
        
        def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
            """Custom fit for neural network"""
            import time
            start_time = time.time()
            
            # Initialize network with correct input dimension
            input_dim = X.shape[1]
            network_class = self._build_model()
            self.model = network_class(input_dim)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            epochs = self.model_params.get('epochs', 100)
            
            for epoch in range(epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.fitted = True
            self.training_time = time.time() - start_time
            logger.info(f"Neural network trained in {self.training_time:.2f}s")
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Custom predict for neural network"""
            if not self.fitted:
                raise ValueError("Model not fitted")
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor).numpy()
            
            return predictions
        
        @classmethod
        def is_compatible(cls, X: np.ndarray, y: np.ndarray) -> bool:
            """Check if neural network is suitable"""
            # Good for medium to large datasets
            return X.shape[0] >= 100
    
except ImportError:
    pass  # PyTorch not available

# Example 2: Custom Ensemble Model
class CustomVotingEnsembleModel(AutoDiscoverableModel):
    """Custom voting ensemble that will be auto-discovered"""
    
    model_type = "ensemble"
    complexity = "medium"
    handles_missing = False
    interpretable = True
    scalable = True
    requires_scaling = False
    estimated_training_time = "slow"
    
    def _build_model(self):
        """Build ensemble of different models"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import VotingRegressor
        
        # Create diverse base models
        models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=self.config.random_seed)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=self.config.random_seed)),
            ('ridge', Ridge(alpha=1.0, random_state=self.config.random_seed))
        ]
        
        return VotingRegressor(models)
    
    @classmethod
    def estimate_performance(cls, n_samples: int, n_features: int) -> float:
        """Ensembles typically perform well"""
        return 0.85

# Example 3: Custom Polynomial Model
class PolynomialRegressionModel(AutoDiscoverableModel):
    """Polynomial regression that will be auto-discovered"""
    
    model_type = "polynomial"
    complexity = "medium"
    handles_missing = False
    interpretable = True
    scalable = True
    requires_scaling = True
    estimated_training_time = "fast"
    
    def _build_model(self):
        """Build polynomial regression model"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        
        degree = self.model_params.get('degree', 2)
        
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=1.0, random_state=self.config.random_seed))
        ])
    
    @classmethod
    def is_compatible(cls, X: np.ndarray, y: np.ndarray) -> bool:
        """Polynomial regression works best with smaller feature sets"""
        return X.shape[1] <= 20  # Limited to avoid feature explosion

# Example 4: Custom Robust Regression Model
class RobustRegressionModel(AutoDiscoverableModel):
    """Robust regression for outlier-heavy data"""
    
    model_type = "robust"
    complexity = "low"
    handles_missing = False
    interpretable = True
    scalable = True
    requires_scaling = True
    estimated_training_time = "fast"
    
    def _build_model(self):
        """Build robust regression model"""
        from sklearn.linear_model import HuberRegressor
        
        return HuberRegressor(
            epsilon=self.model_params.get('epsilon', 1.35),
            alpha=self.model_params.get('alpha', 0.001),
            max_iter=1000
        )

# ============================================================================
# CUSTOM AGENT EXAMPLES
# ============================================================================

# Example 1: Feature Engineering Agent
class AdvancedFeatureEngineeringAgent(AutoDiscoverableAgent):
    """Advanced feature engineering agent"""
    
    agent_type = "feature_engineering"
    priority = 85  # Runs after preprocessing but before model selection
    required_inputs = ["X_train_processed", "X_test_processed", "y_train"]
    provided_outputs = ["X_train_engineered", "X_test_engineered", "feature_names"]
    
    def execute(self, X_train_processed: np.ndarray, 
                X_test_processed: np.ndarray,
                y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Execute advanced feature engineering"""
        logger.info("Advanced feature engineering")
        
        feature_names = []
        X_train_features = []
        X_test_features = []
        
        # Original features
        X_train_features.append(X_train_processed)
        X_test_features.append(X_test_processed)
        feature_names.extend([f"original_{i}" for i in range(X_train_processed.shape[1])])
        
        # Polynomial features (only for small datasets)
        if X_train_processed.shape[1] <= 10:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_processed)
            X_test_poly = poly.transform(X_test_processed)
            
            X_train_features.append(X_train_poly)
            X_test_features.append(X_test_poly)
            feature_names.extend([f"poly_{i}" for i in range(X_train_poly.shape[1])])
        
        # Statistical features
        X_train_stats = self._create_statistical_features(X_train_processed)
        X_test_stats = self._create_statistical_features(X_test_processed)
        
        X_train_features.append(X_train_stats)
        X_test_features.append(X_test_stats)
        feature_names.extend(["mean", "std", "min", "max", "median"])
        
        # Combine all features
        X_train_engineered = np.hstack(X_train_features)
        X_test_engineered = np.hstack(X_test_features)
        
        # Feature selection if too many features
        if X_train_engineered.shape[1] > 100:
            from sklearn.feature_selection import SelectKBest, f_regression
            selector = SelectKBest(f_regression, k=100)
            X_train_engineered = selector.fit_transform(X_train_engineered, y_train)
            X_test_engineered = selector.transform(X_test_engineered)
            
            # Update feature names
            selected_indices = selector.get_support()
            feature_names = [name for name, selected in zip(feature_names, selected_indices) if selected]
        
        return {
            "X_train_engineered": X_train_engineered,
            "X_test_engineered": X_test_engineered,
            "feature_names": feature_names
        }
    
    def _create_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """Create statistical features"""
        return np.column_stack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.min(X, axis=1),
            np.max(X, axis=1),
            np.median(X, axis=1)
        ])

# Example 2: Outlier Detection Agent
class OutlierDetectionAgent(AutoDiscoverableAgent):
    """Outlier detection and removal agent"""
    
    agent_type = "outlier_detection"
    priority = 95  # High priority - runs early
    required_inputs = ["X_train", "y_train"]
    provided_outputs = ["X_train_clean", "y_train_clean", "outlier_indices"]
    
    def execute(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Detect and remove outliers"""
        logger.info("Detecting outliers")
        
        from sklearn.ensemble import IsolationForest
        
        # Detect outliers using Isolation Forest
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(X_train)
        
        # Keep only inliers
        inlier_mask = outlier_labels == 1
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        X_train_clean = X_train[inlier_mask]
        y_train_clean = y_train[inlier_mask]
        
        logger.info(f"Removed {len(outlier_indices)} outliers")
        
        return {
            "X_train_clean": X_train_clean,
            "y_train_clean": y_train_clean,
            "outlier_indices": outlier_indices
        }

# Example 3: Cross-Validation Agent
class CrossValidationAgent(AutoDiscoverableAgent):
    """Cross-validation evaluation agent"""
    
    agent_type = "evaluation"
    priority = 45  # Runs after model training
    required_inputs = ["trained_models", "X_train_scaled", "y_train"]
    provided_outputs = ["cv_scores", "best_model_index"]
    
    def execute(self, trained_models: List, 
                X_train_scaled: np.ndarray, 
                y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform cross-validation"""
        logger.info("Running cross-validation")
        
        cv_scores = []
        
        for i, model in enumerate(trained_models):
            try:
                # Get the underlying sklearn model
                if hasattr(model, 'model'):
                    sklearn_model = model.model
                else:
                    sklearn_model = model
                
                # Cross-validate
                scores = cross_val_score(
                    sklearn_model, X_train_scaled, y_train,
                    cv=5, scoring='r2', n_jobs=-1
                )
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                cv_scores.append({
                    'model': model.get_name() if hasattr(model, 'get_name') else str(model),
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores.tolist()
                })
                
                logger.info(f"Model {i}: CV Score = {mean_score:.4f} (+/- {std_score:.4f})")
                
            except Exception as e:
                logger.warning(f"Could not cross-validate model {i}: {str(e)}")
                cv_scores.append({
                    'model': str(model),
                    'mean_score': -1,
                    'std_score': 0,
                    'scores': []
                })
        
        # Find best model
        best_index = np.argmax([s['mean_score'] for s in cv_scores])
        
        return {
            "cv_scores": cv_scores,
            "best_model_index": best_index
        }

# Example 4: Model Interpretability Agent
class ModelInterpretabilityAgent(AutoDiscoverableAgent):
    """Model interpretability agent"""
    
    agent_type = "interpretability"
    priority = 30  # Runs after ensemble creation
    required_inputs = ["ensemble", "X_train_scaled", "feature_names"]
    provided_outputs = ["feature_importance", "model_explanations"]
    
    def execute(self, ensemble: List, 
                X_train_scaled: np.ndarray,
                feature_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Generate model interpretability insights"""
        logger.info("Generating model interpretability")
        
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X_train_scaled.shape[1])]
        
        all_importances = []
        explanations = []
        
        for model in ensemble:
            try:
                # Get feature importance
                if hasattr(model, 'feature_importance_'):
                    importance = model.feature_importance_
                elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                    importance = model.model.feature_importances_
                else:
                    importance = None
                
                if importance is not None:
                    all_importances.append(importance)
                    
                    # Create explanation
                    top_features_idx = np.argsort(importance)[-5:][::-1]
                    top_features = [(feature_names[i], importance[i]) for i in top_features_idx]
                    
                    explanations.append({
                        'model': model.get_name() if hasattr(model, 'get_name') else str(model),
                        'top_features': top_features
                    })
                    
            except Exception as e:
                logger.warning(f"Could not get importance: {str(e)}")
        
        # Average importance across models
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
        else:
            avg_importance = np.ones(len(feature_names)) / len(feature_names)
        
        # Create final importance ranking
        importance_ranking = sorted(
            zip(feature_names, avg_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "feature_importance": importance_ranking,
            "model_explanations": explanations
        }

# ============================================================================
# CUSTOM TRANSFORMER EXAMPLES
# ============================================================================

class FourierTransformer(AutoDiscoverableTransformer):
    """Fourier transform for time-series or spectral data"""
    
    transformer_type = "fourier"
    preserves_shape = False
    requires_fit = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FourierTransformer':
        """No fitting required for FFT"""
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Fourier transform"""
        from scipy.fft import rfft
        
        # Apply FFT to each sample
        X_fft = []
        for sample in X:
            fft_values = np.abs(rfft(sample))
            X_fft.append(fft_values[:X.shape[1]//2])  # Keep half due to symmetry
        
        return np.array(X_fft)

class LogTransformer(AutoDiscoverableTransformer):
    """Log transformation for skewed data"""
    
    transformer_type = "log"
    preserves_shape = True
    requires_fit = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LogTransformer':
        """No fitting required"""
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log transformation"""
        # Add small constant to avoid log(0)
        return np.log1p(np.abs(X))

# ============================================================================
# CUSTOM AUTOML TOOL EXAMPLE
# ============================================================================

class CustomAutoMLTool(AutoMLTool):
    """Custom AutoML implementation"""
    
    tool_name = "custom_automl"
    supports_regression = True
    supports_classification = False
    supports_multioutput = False
    requires_install = []  # Uses only sklearn
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Simple AutoML using model selection"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search
        self.model = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.fitted = True
        
        logger.info(f"Best parameters: {self.model.best_params_}")
        logger.info(f"Best score: {self.model.best_score_:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_custom_plugins():
    """Demonstrate that custom plugins are auto-discovered"""
    
    # This would normally be in a separate file
    logger.info("\n" + "="*80)
    logger.info("CUSTOM PLUGINS LOADED")
    logger.info("="*80)
    
    # Check what's been registered
    from ultimate_pipeline_autodiscovery import GLOBAL_REGISTRY
    
    models = GLOBAL_REGISTRY.get_models()
    agents = GLOBAL_REGISTRY.get_agents()
    transformers = GLOBAL_REGISTRY.get_transformers()
    automl_tools = GLOBAL_REGISTRY.get_automl_tools()
    
    logger.info(f"\nCustom Models Discovered: {list(models.keys())}")
    logger.info(f"Custom Agents Discovered: {list(agents.keys())}")
    logger.info(f"Custom Transformers Discovered: {list(transformers.keys())}")
    logger.info(f"Custom AutoML Tools Discovered: {list(automl_tools.keys())}")
    
    return True

if __name__ == "__main__":
    # When this file is loaded, all classes are automatically registered
    demonstrate_custom_plugins()

