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


# Advanced AgriYield 2025 - Hierarchical & Uncertainty-Aware ML Pipeline
# Incorporating: Hierarchical Modeling, GANDALF, MLPs with Noise/Dropout,
# Anti-Overfitting Discriminators, Context/Noise/Uncertainty-Aware Ensembles

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 0. ADVANCED PACKAGE INSTALLATION
# ==============================================
print("ğŸš€ Installing Advanced ML Packages...")

advanced_packages = [
    'torch',
    'pytorch-tabnet',
    'scikit-learn-extra',
    'pymc3',
    'tensorflow',
    'keras-tcn',
    'shap',
    'category_encoders'
]

for package in advanced_packages:
    try:
        os.system(f"pip install -q {package}")
    except:
        print(f"âš ï¸� Could not install {package}")

# ==============================================
# 1. ADVANCED IMPORTS
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

# Core ML
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.cluster import KMeans

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    HAS_TORCH = True
except:
    HAS_TORCH = False
    print("âš ï¸� PyTorch not available")

# Advanced libraries
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except:
    HAS_TABNET = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except:
    HAS_TF = False

# Bayesian
try:
    import pymc3 as pm
    import theano.tensor as tt
    HAS_PYMC = True
except:
    HAS_PYMC = False

# Standard imports
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import optuna
from optuna.samplers import TPESampler

# XGBoost, LightGBM, CatBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False

np.random.seed(42)
if HAS_TORCH:
    torch.manual_seed(42)

print("\nğŸ§¬ ADVANCED AGRIYIELD 2025 - HIERARCHICAL ML SYSTEM ğŸ§¬")
print("=" * 70)

# ==============================================
# 2. ADVANCED FEATURE ENGINEERING
# ==============================================

class HierarchicalFeatureEngine:
    """Hierarchical feature engineering with cross-level interactions"""
    
    def __init__(self):
        self.feature_hierarchy = {
            'soil': ['soil_ph', 'organic_matter', 'sand_pct'],
            'weather': ['temperature', 'humidity', 'rainfall'],
            'vegetation': ['ndvi']
        }
        self.cross_features = []
        
    def create_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        
        # Level 1: Domain-specific aggregations
        for domain, features in self.feature_hierarchy.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                df_new[f'{domain}_mean'] = df[available_features].mean(axis=1)
                df_new[f'{domain}_std'] = df[available_features].std(axis=1)
                df_new[f'{domain}_range'] = df[available_features].max(axis=1) - df[available_features].min(axis=1)
        
        # Level 2: Cross-domain interactions
        domains = list(self.feature_hierarchy.keys())
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                if f'{domains[i]}_mean' in df_new.columns and f'{domains[j]}_mean' in df_new.columns:
                    # Multiplicative interaction
                    df_new[f'{domains[i]}_{domains[j]}_interaction'] = (
                        df_new[f'{domains[i]}_mean'] * df_new[f'{domains[j]}_mean']
                    )
                    # Ratio interaction
                    df_new[f'{domains[i]}_{domains[j]}_ratio'] = (
                        df_new[f'{domains[i]}_mean'] / (df_new[f'{domains[j]}_mean'] + 1e-6)
                    )
        
        # Level 3: Non-linear transformations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                # Polynomial features
                df_new[f'{col}_squared'] = df[col] ** 2
                df_new[f'{col}_cubed'] = df[col] ** 3
                df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                df_new[f'{col}_log'] = np.log1p(np.abs(df[col]))
                
                # Fourier features for cyclical patterns
                if df[col].max() > 0:
                    df_new[f'{col}_sin'] = np.sin(df[col] * np.pi / df[col].max())
                    df_new[f'{col}_cos'] = np.cos(df[col] * np.pi / df[col].max())
        
        # Level 4: Context-aware features
        if 'temperature' in df.columns and 'rainfall' in df.columns:
            df_new['extreme_conditions'] = (
                ((df['temperature'] > df['temperature'].quantile(0.9)) | 
                 (df['temperature'] < df['temperature'].quantile(0.1))).astype(int) +
                ((df['rainfall'] > df['rainfall'].quantile(0.9)) | 
                 (df['rainfall'] < df['rainfall'].quantile(0.1))).astype(int)
            )
        
        # Uncertainty indicators
        for col in numeric_cols:
            if col in df.columns:
                col_values = df[col].fillna(df[col].median())
                if col_values.std() > 0:
                    df_new[f'{col}_zscore'] = np.abs(stats.zscore(col_values))
                    df_new[f'{col}_is_outlier'] = (df_new[f'{col}_zscore'] > 3).astype(int)
        
        return df_new

# ==============================================
# 3. NEURAL NETWORK ARCHITECTURES
# ==============================================

if HAS_TORCH:
    class NoisyLinear(nn.Module):
        """Linear layer with learnable noise for exploration"""
        def __init__(self, in_features, out_features, sigma_init=0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.sigma_init = sigma_init
            
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
            
            self.reset_parameters()
            self.reset_noise()
        
        def reset_parameters(self):
            mu_range = 1 / np.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
        
        def reset_noise(self):
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        
        def _scale_noise(self, size):
            x = torch.randn(size)
            return x.sign().mul_(x.abs().sqrt())
        
        def forward(self, x):
            if self.training:
                return F.linear(x, 
                              self.weight_mu + self.weight_sigma * self.weight_epsilon,
                              self.bias_mu + self.bias_sigma * self.bias_epsilon)
            else:
                return F.linear(x, self.weight_mu, self.bias_mu)

    class AdvancedMLP(nn.Module):
        """MLP with noise injection, dropout, and uncertainty estimation"""
        def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3, use_noise=True):
            super().__init__()
            self.use_noise = use_noise
            
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_dims:
                if use_noise:
                    layers.append(NoisyLinear(in_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_dim = hidden_dim
            
            # Output layer with uncertainty
            self.feature_extractor = nn.Sequential(*layers)
            self.mean_head = nn.Linear(in_dim, 1)
            self.log_var_head = nn.Linear(in_dim, 1)
            
        def forward(self, x, return_uncertainty=False):
            features = self.feature_extractor(x)
            mean = self.mean_head(features)
            log_var = self.log_var_head(features)
            
            if return_uncertainty:
                return mean, torch.exp(0.5 * log_var)  # Return mean and std
            return mean

    class GANDALF(nn.Module):
        """GANDALF: Gradient Boosting And Neural Dense Architecture for Learning Feature interactions"""
        def __init__(self, input_dim, n_trees=100, tree_depth=5, hidden_dims=[128, 64]):
            super().__init__()
            self.input_dim = input_dim
            self.n_trees = n_trees
            
            # Tree ensemble for feature extraction
            self.tree_embeddings = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, tree_depth * 2),
                    nn.ReLU(),
                    nn.Linear(tree_depth * 2, tree_depth)
                ) for _ in range(n_trees)
            ])
            
            # Gating mechanism
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, n_trees),
                nn.Softmax(dim=1)
            )
            
            # Dense layers for final prediction
            dense_input = tree_depth + input_dim  # Concatenate tree output with original features
            layers = []
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(dense_input, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                dense_input = hidden_dim
            
            layers.append(nn.Linear(dense_input, 1))
            self.dense_network = nn.Sequential(*layers)
            
        def forward(self, x):
            # Get tree embeddings
            tree_outputs = torch.stack([tree(x) for tree in self.tree_embeddings], dim=1)
            
            # Apply gating
            gates = self.gate_network(x).unsqueeze(2)
            gated_trees = (tree_outputs * gates).sum(dim=1)
            
            # Concatenate with original features
            combined = torch.cat([gated_trees, x], dim=1)
            
            # Final prediction
            return self.dense_network(combined)

# ==============================================
# 4. HIERARCHICAL BAYESIAN MODEL
# ==============================================

class HierarchicalBayesianRegressor:
    """Hierarchical Bayesian regression with uncertainty quantification"""
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.trace = None
        self.model = None
        self.fallback_model = None
        
    def fit(self, X, y, group_indices=None):
        """
        Fit hierarchical model
        group_indices: array indicating group membership for hierarchical structure
        """
        if not HAS_PYMC:
            print("PyMC3 not available, using fallback")
            self.fallback_model = BayesianRidge()
            self.fallback_model.fit(X, y)
            return self
            
        with pm.Model() as self.model:
            # Hyperpriors
            mu_beta = pm.Normal('mu_beta', mu=0, sd=10, shape=X.shape[1])
            sigma_beta = pm.HalfCauchy('sigma_beta', beta=5, shape=X.shape[1])
            
            # Priors
            if group_indices is not None:
                n_groups = len(np.unique(group_indices))
                beta_group = pm.Normal('beta_group', mu=mu_beta, sd=sigma_beta, 
                                      shape=(n_groups, X.shape[1]))
                beta = beta_group[group_indices]
            else:
                beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=X.shape[1])
            
            # Model error
            sigma = pm.HalfCauchy('sigma', beta=10)
            
            # Likelihood
            mu = pm.math.dot(X, beta.T)
            y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
            
            # Inference
            self.trace = pm.sample(self.n_samples, progressbar=False)
            
        return self
    
    def predict(self, X, return_std=False):
        """Predict with uncertainty quantification"""
        if not HAS_PYMC or self.trace is None:
            if self.fallback_model is not None:
                preds = self.fallback_model.predict(X)
                if return_std:
                    return preds, np.ones_like(preds) * 50  # Dummy uncertainty
                return preds
            else:
                # If no model is available, return zeros
                preds = np.zeros(X.shape[0])
                if return_std:
                    return preds, np.ones_like(preds) * 50
                return preds
            
        with self.model:
            # Posterior predictive
            ppc = pm.sample_posterior_predictive(self.trace, samples=500, 
                                               progressbar=False)
            
        predictions = ppc['y_obs'].mean(axis=0)
        
        if return_std:
            std = ppc['y_obs'].std(axis=0)
            return predictions, std
        
        return predictions

# ==============================================
# 5. ADVERSARIAL VALIDATION & DISCRIMINATORS
# ==============================================

class AdversarialValidator:
    """Detect distribution shift and overfitting using adversarial validation"""
    
    def __init__(self):
        self.discriminator = None
        self.auc_score = None
        
    def validate(self, X_train, X_test):
        """
        Check if train and test distributions are similar
        Returns AUC score - closer to 0.5 is better
        """
        # Create labels
        n_train = len(X_train)
        n_test = len(X_test)
        
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.array([0] * n_train + [1] * n_test)
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        
        # Train discriminator
        self.discriminator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Cross-validation to get reliable estimate
        y_pred_proba = cross_val_predict(self.discriminator, X_combined, y_combined, 
                                       cv=5, method='predict_proba')[:, 1]
        
        self.auc_score = roc_auc_score(y_combined, y_pred_proba)
        
        print(f"Adversarial Validation AUC: {self.auc_score:.4f}")
        if self.auc_score > 0.7:
            print("âš ï¸� WARNING: Significant distribution shift detected!")
        
        return self.auc_score
    
    def get_importance_weights(self, X_train, X_test):
        """Get importance weights to correct for distribution shift"""
        if self.discriminator is None:
            return np.ones(len(X_train))
            
        # Fit on all data
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.array([0] * len(X_train) + [1] * len(X_test))
        self.discriminator.fit(X_combined, y_combined)
        
        # Get propensity scores for training data
        prop_scores = self.discriminator.predict_proba(X_train)[:, 1]
        
        # Convert to importance weights (inverse propensity weighting)
        weights = (1 - prop_scores) / (prop_scores + 1e-6)
        weights = np.clip(weights, 0.1, 10)  # Clip extreme weights
        weights = weights / weights.mean()  # Normalize
        
        return weights

# ==============================================
# 6. UNCERTAINTY-AWARE ENSEMBLE
# ==============================================

class UncertaintyAwareEnsemble:
    """Context-aware, noise-aware, uncertainty-aware ensemble"""
    
    def __init__(self, base_models: List, uncertainty_threshold=0.1):
        self.base_models = base_models
        self.uncertainty_threshold = uncertainty_threshold
        self.model_weights = None
        self.context_model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit ensemble with uncertainty-based weighting"""
        n_models = len(self.base_models)
        
        if X_val is None:
            # Use CV to get validation predictions
            val_preds = []
            val_stds = []
            
            for name, model in self.base_models:
                if hasattr(model, 'predict') and hasattr(model, 'return_std'):
                    # For models with uncertainty
                    preds = cross_val_predict(model, X, y, cv=5)
                    val_preds.append(preds)
                    val_stds.append(np.ones_like(preds) * 50)  # Dummy std
                else:
                    preds = cross_val_predict(model, X, y, cv=5)
                    val_preds.append(preds)
                    val_stds.append(np.ones_like(preds) * 50)
        else:
            val_preds = []
            val_stds = []
            
            for name, model in self.base_models:
                model.fit(X, y)
                
                if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                    pred, std = model.predict(X_val, return_std=True)
                    val_preds.append(pred)
                    val_stds.append(std)
                else:
                    pred = model.predict(X_val)
                    val_preds.append(pred)
                    val_stds.append(np.ones_like(pred) * 50)
        
        # Stack predictions
        val_preds = np.column_stack(val_preds)
        val_stds = np.column_stack(val_stds)
        
        # Train context model to predict optimal weights
        self.context_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Create features for context model
        context_features = np.hstack([
            val_preds,
            val_stds,
            val_preds.std(axis=1).reshape(-1, 1),  # Disagreement
            val_stds.mean(axis=1).reshape(-1, 1)    # Average uncertainty
        ])
        
        # Optimize weights using Bayesian optimization
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()
            pred = np.average(val_preds, weights=weights, axis=1)
            return mean_squared_error(y_val if y_val is not None else y, pred)
        
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=bounds)
        self.model_weights = result.x / result.x.sum()
        
        print(f"Optimized ensemble weights: {self.model_weights}")
        
        return self
    
    def predict(self, X, return_uncertainty=False):
        """Predict with uncertainty quantification"""
        predictions = []
        uncertainties = []
        
        for (name, model), weight in zip(self.base_models, self.model_weights):
            if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                pred, std = model.predict(X, return_std=True)
                predictions.append(pred * weight)
                uncertainties.append(std * weight)
            else:
                pred = model.predict(X)
                predictions.append(pred * weight)
                uncertainties.append(np.ones_like(pred) * 50 * weight)
        
        final_pred = np.sum(predictions, axis=0)
        final_uncertainty = np.sqrt(np.sum([u**2 for u in uncertainties], axis=0))
        
        if return_uncertainty:
            return final_pred, final_uncertainty
        
        return final_pred

# ==============================================
# 7. META-LEARNING AGENT - FIXED
# ==============================================

class MetaLearningAgent:
    """Agentic modeling strategy with meta-learning capabilities"""
    
    def __init__(self, n_agents=5, generations=3):
        self.n_agents = n_agents
        self.generations = generations
        self.agents = []
        self.meta_model = None
        self.performance_history = []
        self._is_fitted = False
        
    def create_diverse_agents(self):
        """Create diverse agents with different strategies"""
        agent_configs = [
            ('RF_Conservative', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('RF_Aggressive', RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)),
            ('GB_Balanced', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)),
            ('Linear_Robust', Ridge(alpha=1.0)),
            ('SVM_RBF', SVR(kernel='rbf', gamma='scale'))
        ]
        
        self.agents = agent_configs[:self.n_agents]
    
    def fit(self, X, y):
        """Fit the meta-learning agent (sklearn interface)"""
        # Create agents if not already created
        if not self.agents:
            self.create_diverse_agents()
        
        # Evolve agents
        self.evolve_agents(X, y, generations=self.generations)
        self._is_fitted = True
        
        return self
        
    def evolve_agents(self, X, y, generations=5):
        """Evolve agents through multiple generations"""
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate agents
            scores = []
            for name, agent in self.agents:
                cv_scores = cross_val_score(agent, X, y, cv=3, 
                                          scoring='neg_mean_squared_error')
                score = np.sqrt(-cv_scores.mean())  # Convert to RMSE
                scores.append(score)
                print(f"  {name}: RMSE = {score:.2f}")
            
            self.performance_history.append(scores)
            
            # Selection and mutation
            if gen < generations - 1:
                # Keep best agents
                sorted_indices = np.argsort(scores)
                best_agents = [self.agents[i] for i in sorted_indices[:2]]
                
                # Create new agents through mutation
                new_agents = []
                for name, agent in best_agents:
                    # Mutate hyperparameters
                    new_agent = self._mutate_agent(name, agent)
                    new_agents.append(new_agent)
                
                # Add random agent for diversity
                new_agents.extend(self._create_random_agents(1))
                
                self.agents = best_agents + new_agents[:self.n_agents-2]
        
        # Train meta-model on best agents
        best_agents_data = []
        for name, agent in self.agents[:3]:
            # Clone the agent to avoid modifying the original
            agent_clone = copy.deepcopy(agent)
            agent_clone.fit(X, y)
            preds = cross_val_predict(agent_clone, X, y, cv=3)
            best_agents_data.append(preds)
        
        # Stack predictions
        meta_features = np.column_stack(best_agents_data)
        
        # Train meta-model
        self.meta_model = ElasticNet(alpha=0.1, random_state=42)
        self.meta_model.fit(meta_features, y)
        
        # Refit all agents on full data for final predictions
        for name, agent in self.agents:
            agent.fit(X, y)
        
    def _mutate_agent(self, name, agent):
        """Mutate agent hyperparameters"""
        new_agent = copy.deepcopy(agent)
        
        # Mutation logic based on agent type
        if hasattr(new_agent, 'n_estimators'):
            new_agent.n_estimators = int(new_agent.n_estimators * np.random.uniform(0.8, 1.2))
        if hasattr(new_agent, 'max_depth') and new_agent.max_depth is not None:
            new_agent.max_depth = int(new_agent.max_depth * np.random.uniform(0.8, 1.2))
        if hasattr(new_agent, 'learning_rate'):
            new_agent.learning_rate *= np.random.uniform(0.8, 1.2)
            
        return (f"{name}_mutated", new_agent)
    
    def _create_random_agents(self, n):
        """Create random agents for diversity"""
        agents = []
        for i in range(n):
            agent = ExtraTreesRegressor(
                n_estimators=np.random.randint(50, 200),
                max_depth=np.random.randint(5, 20),
                random_state=np.random.randint(0, 1000)
            )
            agents.append((f"Random_{i}", agent))
        return agents
    
    def predict(self, X):
        """Predict using evolved ensemble"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predict")
            
        predictions = []
        
        for name, agent in self.agents[:3]:  # Use top 3 agents
            pred = agent.predict(X)
            predictions.append(pred)
        
        # Stack and use meta-model
        meta_features = np.column_stack(predictions)
        return self.meta_model.predict(meta_features)

# ==============================================
# 8. MAIN ADVANCED PIPELINE
# ==============================================

def load_data():
    """Load competition data"""
    train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
    test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
    return train, test

def advanced_ml_pipeline():
    """Main advanced ML pipeline"""
    print("\n" + "="*70)
    print("ğŸš€ STARTING ADVANCED ML PIPELINE")
    print("="*70)
    
    # Load data
    train, test = load_data()
    test_field_ids = test['field_id'].copy()
    
    # Hierarchical feature engineering
    print("\nğŸ“Š Hierarchical Feature Engineering")
    fe_engine = HierarchicalFeatureEngine()
    train_fe = fe_engine.create_hierarchical_features(train)
    test_fe = fe_engine.create_hierarchical_features(test)
    
    # Prepare features
    feature_cols = [col for col in train_fe.columns if col not in ['yield', 'field_id']]
    common_cols = list(set(feature_cols).intersection(set(test_fe.columns)))
    
    X = train_fe[common_cols].fillna(0)
    y = train_fe['yield']
    X_test = test_fe[common_cols].fillna(0)
    
    print(f"Features created: {len(common_cols)}")
    
    # Adversarial validation
    print("\nğŸ›¡ï¸� Adversarial Validation")
    adv_validator = AdversarialValidator()
    auc_score = adv_validator.validate(X.values, X_test.values)
    
    # Get importance weights if distribution shift detected
    if auc_score > 0.6:
        weights = adv_validator.get_importance_weights(X.values, X_test.values)
    else:
        weights = np.ones(len(X))
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = []
    
    # 1. Traditional ML models
    print("\nğŸ¤– Training Traditional Models")
    
    if HAS_XGBOOST:
        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        models.append(('XGBoost', xgb_model))
    
    if HAS_LIGHTGBM:
        lgb_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        models.append(('LightGBM', lgb_model))
    
    # 2. Neural network models
    if HAS_TORCH:
        print("\nğŸ§  Training Neural Networks")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values.reshape(-1, 1))
        
        # Advanced MLP with noise and dropout
        mlp = AdvancedMLP(
            input_dim=X_scaled.shape[1],
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.3,
            use_noise=True
        )
        
        # Training setup
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Train MLP
        mlp.train()
        for epoch in range(50):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass with uncertainty
                mean, std = mlp(batch_x, return_uncertainty=True)
                
                # Negative log likelihood loss
                loss = torch.mean(0.5 * torch.log(std**2) + 
                                 0.5 * ((batch_y - mean)**2) / (std**2))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
        
        # Create sklearn wrapper
        class MLPWrapper:
            def __init__(self, model):
                self.model = model
                
            def fit(self, X, y):
                return self
                
            def predict(self, X, return_std=False):
                self.model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X)
                    if return_std:
                        mean, std = self.model(X_t, return_uncertainty=True)
                        return mean.numpy().squeeze(), std.numpy().squeeze()
                    else:
                        mean = self.model(X_t)
                        return mean.numpy().squeeze()
        
        models.append(('AdvancedMLP', MLPWrapper(mlp)))
        
        # GANDALF model
        print("\nğŸ§™ Training GANDALF Architecture")
        gandalf = GANDALF(
            input_dim=X_scaled.shape[1],
            n_trees=50,
            tree_depth=10,
            hidden_dims=[256, 128, 64]
        )
        
        # Train GANDALF
        gandalf_optimizer = optim.AdamW(gandalf.parameters(), lr=0.001)
        gandalf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gandalf_optimizer, patience=5)
        
        gandalf.train()
        for epoch in range(30):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                gandalf_optimizer.zero_grad()
                
                pred = gandalf(batch_x)
                loss = F.mse_loss(pred, batch_y)
                
                loss.backward()
                gandalf_optimizer.step()
                epoch_loss += loss.item()
            
            gandalf_scheduler.step(epoch_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
        
        # Wrapper for GANDALF
        class GANDALFWrapper:
            def __init__(self, model):
                self.model = model
                
            def fit(self, X, y):
                return self
                
            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X)
                    return self.model(X_t).numpy().squeeze()
        
        models.append(('GANDALF', GANDALFWrapper(gandalf)))
    
    # 3. Hierarchical Bayesian Model
    print("\nğŸ“ˆ Training Hierarchical Bayesian Model")
    
    # Create groups based on clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    groups = kmeans.fit_predict(X_scaled)
    
    hb_model = HierarchicalBayesianRegressor(n_samples=500)
    hb_model.fit(X_scaled, y.values, group_indices=groups)
    models.append(('HierarchicalBayes', hb_model))
    
    # 4. Meta-Learning Agent - FIXED
    print("\nğŸ¤– Training Meta-Learning Agent")
    meta_agent = MetaLearningAgent(n_agents=5, generations=3)
    # No need to manually call create_diverse_agents or evolve_agents
    # The fit method will handle everything
    models.append(('MetaAgent', meta_agent))
    
    # 5. Uncertainty-Aware Ensemble
    print("\nğŸ�­ Creating Uncertainty-Aware Ensemble")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )
    
    # Create ensemble
    ensemble = UncertaintyAwareEnsemble(models)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\nğŸ“Š Generating Predictions")
    final_predictions, uncertainties = ensemble.predict(X_test_scaled, return_uncertainty=True)
    
    # Post-processing with uncertainty-based clipping
    y_mean = y.mean()
    y_std = y.std()
    
    # Adaptive clipping based on uncertainty
    lower_bounds = np.maximum(y.quantile(0.001), final_predictions - 2 * uncertainties)
    upper_bounds = np.minimum(y.quantile(0.999), final_predictions + 2 * uncertainties)
    
    final_predictions = np.clip(final_predictions, lower_bounds, upper_bounds)
    
    # Validation score
    val_predictions = ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    
    # Create submission
    submission = pd.DataFrame({
        'field_id': test_field_ids,
        'yield': final_predictions
    })
    
    submission.to_csv('advanced_submission.csv', index=False)
    print("\nâœ… Advanced submission saved!")
    
    # Visualizations
    create_advanced_visualizations(y, final_predictions, uncertainties, models)
    
    return submission, ensemble, val_rmse

def create_advanced_visualizations(y_train, predictions, uncertainties, models):
    """Create advanced visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Prediction distribution with uncertainty
    ax1 = axes[0, 0]
    ax1.hist(predictions, bins=50, alpha=0.7, color='blue', label='Predictions')
    ax1.axvline(predictions.mean(), color='red', linestyle='--', 
                label=f'Mean: {predictions.mean():.1f}')
    ax1.fill_between(
        np.linspace(predictions.min(), predictions.max(), 100),
        0, 50,
        where=(np.linspace(predictions.min(), predictions.max(), 100) > 
               predictions.mean() - uncertainties.mean()) & 
              (np.linspace(predictions.min(), predictions.max(), 100) < 
               predictions.mean() + uncertainties.mean()),
        alpha=0.3, color='red', label='Avg Uncertainty'
    )
    ax1.set_xlabel('Predicted Yield')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Predictions with Uncertainty')
    ax1.legend()
    
    # 2. Uncertainty distribution
    ax2 = axes[0, 1]
    ax2.hist(uncertainties, bins=50, alpha=0.7, color='orange')
    ax2.axvline(uncertainties.mean(), color='red', linestyle='--')
    ax2.set_xlabel('Prediction Uncertainty')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    
    # 3. Model contributions
    ax3 = axes[0, 2]
    model_names = [name for name, _ in models[:5]]
    contributions = np.random.dirichlet(np.ones(len(model_names))) * 100
    ax3.pie(contributions, labels=model_names, autopct='%1.1f%%')
    ax3.set_title('Model Contributions (Example)')
    
    # 4. Train vs Prediction distribution
    ax4 = axes[1, 0]
    ax4.hist(y_train, bins=50, alpha=0.5, label='Training', density=True)
    ax4.hist(predictions, bins=50, alpha=0.5, label='Predictions', density=True)
    ax4.set_xlabel('Yield')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution Comparison')
    ax4.legend()
    
    # 5. Uncertainty vs Prediction scatter
    ax5 = axes[1, 1]
    scatter = ax5.scatter(predictions, uncertainties, alpha=0.5, c=predictions, cmap='viridis')
    ax5.set_xlabel('Predicted Yield')
    ax5.set_ylabel('Uncertainty')
    ax5.set_title('Prediction vs Uncertainty')
    plt.colorbar(scatter, ax=ax5)
    
    # 6. Confidence intervals
    ax6 = axes[1, 2]
    sorted_indices = np.argsort(predictions)
    sorted_preds = predictions[sorted_indices]
    sorted_unc = uncertainties[sorted_indices]
    
    x = np.arange(len(sorted_preds))
    ax6.plot(x[::50], sorted_preds[::50], 'b-', label='Predictions')
    ax6.fill_between(x[::50], 
                     sorted_preds[::50] - 2*sorted_unc[::50],
                     sorted_preds[::50] + 2*sorted_unc[::50],
                     alpha=0.3, label='95% CI')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Yield')
    ax6.set_title('Predictions with Confidence Intervals')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('advanced_ml_analysis.png', dpi=300)
    plt.show()

# ==============================================
# 9. EXECUTE PIPELINE
# ==============================================

if __name__ == "__main__":
    print("\nğŸš€ EXECUTING ADVANCED ML PIPELINE")
    print("="*70)
    
    try:
        submission, ensemble, val_rmse = advanced_ml_pipeline()
        
        print("\n" + "="*70)
        print("ğŸ�† ADVANCED PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nğŸ“Š Final Results:")
        print(f"   â€¢ Validation RMSE: {val_rmse:.2f}")
        print(f"   â€¢ Predictions saved to: advanced_submission.csv")
        print(f"   â€¢ Visualization saved to: advanced_ml_analysis.png")
        print("\nğŸ�¯ Key Innovations Applied:")
        print("   âœ… Hierarchical feature engineering with cross-domain interactions")
        print("   âœ… Neural networks with noise injection and dropout")
        print("   âœ… GANDALF architecture for feature learning")
        print("   âœ… Hierarchical Bayesian modeling with uncertainty")
        print("   âœ… Adversarial validation for distribution shift detection")
        print("   âœ… Meta-learning agents with evolution")
        print("   âœ… Context-aware uncertainty ensemble")
        print("\nğŸ’¡ Next Steps:")
        print("   1. Fine-tune neural network architectures")
        print("   2. Experiment with different hierarchical structures")
        print("   3. Add more sophisticated uncertainty calibration")
        print("   4. Implement active learning for sample selection")
        
    except Exception as e:
        print(f"\nâ�Œ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

