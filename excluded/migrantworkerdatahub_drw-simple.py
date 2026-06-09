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


!pip install flaml ray
import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
import os
import time
import cupy as cp
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import gc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================
# CONFIGURATION SETTINGS
# ============================================

class PipelineConfig:
    """
    Central configuration for the DRW Crypto Prediction Pipeline.
    All parameters are organized by pipeline stage for easy adjustment.
    Optimized for memory efficiency while maintaining model quality.
    """
    
    # ====================
    # DATA SAMPLING CONFIGURATION
    # ====================
    # Controls how much data is used in different pipeline stages
    
    TRAIN_DATA_PERCENTAGE = 40  # Increased from 20% since we're reducing features
    TRAIN_DATA_RECENT_ONLY = True  # If True, uses most recent data; if False, uses random sampling
    INTERACTION_DISCOVERY_SAMPLE_SIZE = 50000  # Reduced from 100000
    
    # ====================
    # FEATURE INTERACTION DISCOVERY
    # ====================
    # Parameters for the AI-powered feature interaction search
    
    # Search Algorithm Settings
    SEARCH_N_PHASES = 3  # Reduced from 6 - focus on most effective strategies
    SEARCH_PAIRS_PER_PHASE = 1000  # Reduced from 5000
    SEARCH_TOP_FEATURES = 100  # Reduced from 300
    SEARCH_TOP_INTERACTIONS = 300  # Reduced from 2000
    
    # Sampling Configuration for Search
    SEARCH_INITIAL_SAMPLE_SIZE = 1000  # Reduced from 2000
    SEARCH_MAX_SAMPLE_SIZE = 30000  # Reduced from 100000
    
    # Search Quality Thresholds
    SEARCH_CORRELATION_THRESHOLD = 0.03  # Increased from 0.02 for higher quality
    SEARCH_MIN_CONFIDENCE = 0.85  # Slightly reduced from 0.9
    SEARCH_EXPLORATION_BONUS = 0.2  # Unchanged
    
    # ====================
    # FEATURE ENGINEERING
    # ====================
    # Controls feature creation and transformation
    
    N_INTERACTIONS_TO_CREATE = 200  # Reduced from 800
    N_INTERACTIONS_TO_SELECT = 150  # Reduced from 600
    N_PCA_COMPONENTS = 50  # Reduced from 150
    
    # ====================
    # FINAL MODEL PREPARATION
    # ====================
    # Parameters for the final feature set and model training
    
    N_FINAL_FEATURES = 150  # Reduced from 350
    
    # ====================
    # MODEL TRAINING (FLAML AutoML)
    # ====================
    # AutoML training configuration
    
    TRAINING_TIME_HOURS = 2  # Unchanged
    TRAINING_N_CONCURRENT_TRIALS = 4  # Unchanged
    TRAINING_METRIC = "mse"  # Optimization metric (do not change per requirements)
    TRAINING_TASK = "regression"  # Task type (do not change per requirements)
    TRAINING_FREE_MEM_RATIO = 0.2  # Increased from 0.1 for better memory management
    
    # ====================
    # OUTPUT AND DISPLAY
    # ====================
    # Controls what information is displayed and saved
    
    # Display Settings
    N_TOP_SYNERGIES_TO_DISPLAY = 10  # Top synergistic interactions to show
    N_TOP_FEATURES_TO_DISPLAY = 20  # Top important features to display
    N_TOP_PCA_TO_DISPLAY = 10  # Top PCA components to show
    
    # File I/O Settings
    INTERACTIONS_FILENAME = 'feature_interactions.csv'  # Saved interaction discovery results
    SUBMISSION_FILENAME = 'submission_optimized_pipeline.csv'  # Final predictions output
    
    # ====================
    # SYSTEM CONFIGURATION
    # ====================
    # Technical settings for processing
    
    N_JOBS = -1  # Number of CPU cores to use (-1 = all available)
    RANDOM_SEED = 42  # Random seed for reproducibility

# Create config instance
config = PipelineConfig()

# Set random seeds using config
seed = config.RANDOM_SEED
np.random.seed(seed)
random.seed(seed)

# ============================================
# FEATURE INTERACTION DISCOVERY CLASSES
# ============================================

@dataclass
class InteractionCandidate:
    """Store information about a feature interaction candidate"""
    feat1_idx: int
    feat2_idx: int
    score: float
    sample_size: int
    confidence: float
    exploration_count: int = 1
    interaction_type: str = 'multiply'
    metadata: Dict = field(default_factory=dict)

class WorldClassInteractionSearch:
    """State-of-the-art adaptive search for feature interactions"""
    
    def __init__(self, 
                 initial_sample_size: int = None,
                 max_sample_size: int = None,
                 n_top_features: int = None,
                 n_pairs_per_phase: int = None,
                 n_phases: int = None,
                 top_interactions: int = None,
                 min_confidence: float = None,
                 exploration_bonus: float = None,
                 correlation_threshold: float = None,
                 n_jobs: int = None):
        # Use config values if not provided
        self.initial_sample_size = initial_sample_size or config.SEARCH_INITIAL_SAMPLE_SIZE
        self.max_sample_size = max_sample_size or config.SEARCH_MAX_SAMPLE_SIZE
        self.n_top_features = n_top_features or config.SEARCH_TOP_FEATURES
        self.n_pairs_per_phase = n_pairs_per_phase or config.SEARCH_PAIRS_PER_PHASE
        self.n_phases = n_phases or config.SEARCH_N_PHASES
        self.top_interactions = top_interactions or config.SEARCH_TOP_INTERACTIONS
        self.min_confidence = min_confidence or config.SEARCH_MIN_CONFIDENCE
        self.exploration_bonus = exploration_bonus or config.SEARCH_EXPLORATION_BONUS
        self.correlation_threshold = correlation_threshold or config.SEARCH_CORRELATION_THRESHOLD
        self.n_jobs = n_jobs or config.N_JOBS
        
        self.candidates = {}
        self.feature_participation = defaultdict(int)
        self.feature_stats = {}
        self.interaction_cache = {}
    
    def search(self, X_gpu: cp.ndarray, y_gpu: cp.ndarray, 
               feature_names: List[str]) -> pd.DataFrame:
        """Main search method"""
        n_samples, n_features = X_gpu.shape
        
        print(f"Starting world-class adaptive search")
        print(f"Features: {n_features}, Samples: {n_samples:,}")
        print(f"Possible interactions: {n_features * (n_features - 1) // 2:,}")
        print(f"Search depth: {self.n_phases} phases, {self.n_pairs_per_phase:,} pairs/phase")
        print("-" * 70)
        
        # Preprocess
        try:
            X_clean, y_clean, valid_features = self._preprocess_and_cache(X_gpu, y_gpu, feature_names)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return pd.DataFrame()
        
        if len(valid_features) < 2:
            print("Not enough valid features!")
            return pd.DataFrame()
        
        # Define all strategies
        all_strategies = [
            ("Multi-resolution Sampling", self._intelligent_sampling),
            ("Information-theoretic Discovery", self._information_theoretic_search),
            ("Tree-based Importance", self._ensemble_importance_search),
            ("Evolutionary Search", self._evolutionary_search),
            ("Spectral Analysis", self._spectral_search),
            ("Bayesian Refinement", self._bayesian_refinement)
        ]
        
        strategies_to_run = all_strategies[:self.n_phases]
        
        for phase, (name, method) in enumerate(strategies_to_run, 1):
            try:
                print(f"\n=== Phase {phase}/{self.n_phases}: {name} ===")
                start = time.time()
                method(X_clean, y_clean, valid_features)
                elapsed = time.time() - start
                print(f"Completed in {elapsed:.1f}s | Total candidates: {len(self.candidates)}")
            except Exception as e:
                print(f"Warning: {name} failed - {str(e)[:100]}")
                continue
        
        # Final validation
        try:
            print("\n=== Final Validation ===")
            final_results = self._final_validation_advanced(X_clean, y_clean, valid_features)
        except Exception as e:
            print(f"Error in final validation: {e}")
            final_results = self._emergency_results(valid_features)
        
        self._cleanup_memory()
        return final_results
    
    def _preprocess_and_cache(self, X: cp.ndarray, y: cp.ndarray, 
                             feature_names: List[str]) -> Tuple[cp.ndarray, cp.ndarray, List[str]]:
        """Preprocess data and cache statistics"""
        print("Preprocessing data...")
        
        valid_mask = cp.ones(X.shape[1], dtype=bool)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            if cp.any(cp.isnan(col)) or cp.any(cp.isinf(col)):
                col_clean = col[~cp.isnan(col) & ~cp.isinf(col)]
                if len(col_clean) < X.shape[0] * 0.1:
                    valid_mask[i] = False
                    continue
                median_val = cp.median(col_clean)
                col = cp.where(cp.isnan(col) | cp.isinf(col), median_val, col)
                X[:, i] = col
            
            if cp.std(col) < 1e-10:
                valid_mask[i] = False
                continue
        
        valid_indices = cp.where(valid_mask)[0]
        X_clean = X[:, valid_indices]
        valid_feature_names = [feature_names[i] for i in cp.asnumpy(valid_indices)]
        
        print(f"Valid features: {len(valid_feature_names)}/{len(feature_names)}")
        
        X_mean = cp.mean(X_clean, axis=0)
        X_std = cp.std(X_clean, axis=0) + 1e-8
        X_normalized = (X_clean - X_mean) / X_std
        
        y_clean = cp.where(cp.isnan(y) | cp.isinf(y), cp.median(y[~cp.isnan(y) & ~cp.isinf(y)]), y)
        y_mean = cp.mean(y_clean)
        y_std = cp.std(y_clean) + 1e-8
        y_normalized = (y_clean - y_mean) / y_std
        
        return X_normalized, y_normalized, valid_feature_names
    
    def _intelligent_sampling(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Multi-resolution sampling with importance weighting"""
        n_samples, n_features = X.shape
        
        importance_scores = self._compute_feature_importance(X, y)
        
        sample_sizes = np.linspace(
            self.initial_sample_size, 
            min(self.max_sample_size, n_samples // 2),
            num=4
        ).astype(int)
        
        pairs_per_level = self.n_pairs_per_phase // len(sample_sizes)
        
        for level, sample_size in enumerate(sample_sizes):
            print(f"Level {level+1}: {sample_size:,} samples, {pairs_per_level:,} pairs")
            
            sample_idx = cp.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            
            tested_pairs = set()
            for _ in range(pairs_per_level * 2):
                if np.random.random() < 0.7:
                    top_k = min(self.n_top_features, n_features)
                    top_indices = np.argsort(importance_scores)[-top_k:]
                    i = np.random.choice(top_indices)
                    j = np.random.choice(top_indices)
                else:
                    i = np.random.randint(0, n_features)
                    j = np.random.randint(0, n_features)
                
                if i != j:
                    pair = (min(i, j), max(i, j))
                    if pair not in tested_pairs:
                        tested_pairs.add(pair)
                        
                        scores = self._compute_all_interaction_scores(
                            X_sample[:, i], X_sample[:, j], y_sample
                        )
                        
                        best_score = max(scores.values())
                        if best_score > self.correlation_threshold:
                            best_type = max(scores, key=scores.get)
                            self._update_candidate(pair, i, j, best_score, best_type, 
                                                 sample_size, feature_names)
                        
                        if len(tested_pairs) >= pairs_per_level:
                            break
            
            if level < len(sample_sizes) - 1:
                self._adaptive_pruning(0.7 + level * 0.05)
    
    def _information_theoretic_search(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Use mutual information to find complementary features"""
        n_samples, n_features = X.shape
        sample_size = min(self.max_sample_size // 5, n_samples)
        
        print("Computing mutual information...")
        
        sample_idx = cp.random.choice(n_samples, size=sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        mi_scores = cp.zeros(n_features)
        for i in range(min(n_features, self.n_top_features * 2)):
            mi_scores[i] = self._mutual_information(X_sample[:, i], y_sample)
        
        mi_scores = cp.nan_to_num(mi_scores, nan=0.0)
        
        low_mi = cp.where(mi_scores < cp.percentile(mi_scores, 30))[0]
        high_mi = cp.where(mi_scores > cp.percentile(mi_scores, 70))[0]
        
        if len(low_mi) > 0 and len(high_mi) > 0:
            n_tests = min(self.n_pairs_per_phase, len(low_mi) * len(high_mi))
            
            for _ in range(n_tests):
                i = int(np.random.choice(cp.asnumpy(low_mi)))
                j = int(np.random.choice(cp.asnumpy(high_mi)))
                
                if i != j:
                    pair = (min(i, j), max(i, j))
                    if pair not in self.candidates:
                        scores = self._compute_all_interaction_scores(
                            X_sample[:, i], X_sample[:, j], y_sample
                        )
                        best_score = max(scores.values())
                        if best_score > self.correlation_threshold:
                            best_type = max(scores, key=scores.get)
                            self._update_candidate(pair, i, j, best_score, best_type,
                                                 sample_size, feature_names)
    
    def _ensemble_importance_search(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Tree ensemble to find interactions"""
        n_samples, n_features = X.shape
        sample_size = min(self.max_sample_size // 4, n_samples)
        
        print("Training random forest...")
        
        sample_idx = cp.random.choice(n_samples, size=sample_size, replace=False)
        X_cpu = cp.asnumpy(X[sample_idx])
        y_cpu = cp.asnumpy(y[sample_idx])
        
        rf = RandomForestRegressor(
            n_estimators=30,
            max_depth=3,
            min_samples_leaf=100,
            n_jobs=self.n_jobs,
            random_state=42
        )
        rf.fit(X_cpu, y_cpu)
        
        interaction_counts = defaultdict(int)
        
        for tree in rf.estimators_:
            tree_structure = tree.tree_
            
            def extract_pairs(node=0, features=set()):
                if tree_structure.feature[node] >= 0:
                    feature = tree_structure.feature[node]
                    new_features = features | {feature}
                    
                    for f in features:
                        if f != feature:
                            pair = (min(f, feature), max(f, feature))
                            interaction_counts[pair] += 1
                    
                    extract_pairs(tree_structure.children_left[node], new_features)
                    extract_pairs(tree_structure.children_right[node], new_features)
            
            extract_pairs()
        
        top_patterns = sorted(interaction_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:self.n_pairs_per_phase]
        
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        for (i, j), _ in top_patterns:
            if (i, j) not in self.candidates and i < n_features and j < n_features:
                scores = self._compute_all_interaction_scores(
                    X_sample[:, i], X_sample[:, j], y_sample
                )
                best_score = max(scores.values())
                if best_score > self.correlation_threshold:
                    best_type = max(scores, key=scores.get)
                    self._update_candidate((i, j), i, j, best_score, best_type,
                                         sample_size, feature_names)
    
    def _evolutionary_search(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Genetic algorithm search"""
        n_samples, n_features = X.shape
        population_size = min(100, self.n_pairs_per_phase // 10)
        n_generations = 5
        
        print(f"Running genetic algorithm ({n_generations} generations)...")
        
        population = []
        if self.candidates:
            top_candidates = sorted(self.candidates.items(), 
                                  key=lambda x: x[1].score, 
                                  reverse=True)[:20]
            population.extend([pair for pair, _ in top_candidates])
        
        while len(population) < population_size:
            i, j = np.random.randint(0, n_features, 2)
            if i != j:
                population.append((min(i, j), max(i, j)))
        
        sample_size = min(self.max_sample_size // 4, n_samples)
        sample_idx = cp.random.choice(n_samples, size=sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        for gen in range(n_generations):
            fitness = []
            for pair in population:
                if pair in self.candidates:
                    fitness.append(self.candidates[pair].score)
                else:
                    score = self._compute_interaction_score(
                        X_sample[:, pair[0]], X_sample[:, pair[1]], y_sample
                    )
                    fitness.append(score)
            
            new_population = []
            while len(new_population) < population_size:
                idx1, idx2 = np.random.choice(len(population), 2)
                parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
                
                if np.random.random() < 0.2:
                    new_feat = np.random.randint(0, n_features)
                    if np.random.random() < 0.5:
                        child = (min(parent[0], new_feat), max(parent[0], new_feat))
                    else:
                        child = (min(new_feat, parent[1]), max(new_feat, parent[1]))
                else:
                    child = parent
                
                if child[0] != child[1]:
                    new_population.append(child)
            
            population = new_population
            
            best_indices = np.argsort(fitness)[-10:]
            for idx in best_indices:
                if fitness[idx] > self.correlation_threshold:
                    pair = population[idx]
                    if pair not in self.candidates:
                        scores = self._compute_all_interaction_scores(
                            X_sample[:, pair[0]], X_sample[:, pair[1]], y_sample
                        )
                        best_score = max(scores.values())
                        best_type = max(scores, key=scores.get)
                        self._update_candidate(pair, pair[0], pair[1], best_score,
                                             best_type, sample_size, feature_names)
    
    def _spectral_search(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Spectral clustering approach"""
        n_samples, n_features = X.shape
        
        print("Computing spectral affinity...")
        
        sample_size = min(5000, n_samples)
        sample_idx = cp.random.choice(n_samples, size=sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        target_corrs = cp.zeros(n_features)
        for i in range(min(n_features, self.n_top_features * 2)):
            target_corrs[i] = cp.abs(cp.corrcoef(X_sample[:, i], y_sample)[0, 1])
        
        n_tests = min(self.n_pairs_per_phase, n_features * 10)
        
        for _ in range(n_tests):
            i = np.random.randint(0, min(n_features, self.n_top_features * 2))
            j = np.random.randint(0, min(n_features, self.n_top_features * 2))
            
            if i != j and abs(float(target_corrs[i] - target_corrs[j])) > 0.1:
                pair = (min(i, j), max(i, j))
                
                if pair not in self.candidates:
                    scores = self._compute_all_interaction_scores(
                        X_sample[:, i], X_sample[:, j], y_sample
                    )
                    best_score = max(scores.values())
                    if best_score > self.correlation_threshold:
                        best_type = max(scores, key=scores.get)
                        self._update_candidate(pair, i, j, best_score, best_type,
                                             sample_size, feature_names)
    
    def _bayesian_refinement(self, X: cp.ndarray, y: cp.ndarray, feature_names: List[str]):
        """Refine top candidates with larger samples"""
        n_samples = X.shape[0]
        
        print("Bayesian refinement of top candidates...")
        
        candidates_to_refine = sorted(self.candidates.items(), 
                                    key=lambda x: x[1].score * (1 - x[1].confidence),
                                    reverse=True)[:self.n_pairs_per_phase // 10]
        
        for pair, candidate in candidates_to_refine:
            alpha = candidate.score * candidate.sample_size
            beta = (1 - candidate.score) * candidate.sample_size
            sampled_score = np.random.beta(alpha + 1, beta + 1)
            
            if sampled_score > candidate.score * 0.9:
                new_sample_size = min(self.max_sample_size, candidate.sample_size * 2)
                
                sample_idx = cp.random.choice(n_samples, size=new_sample_size, replace=False)
                X_sample = X[sample_idx]
                y_sample = y[sample_idx]
                
                scores = self._compute_all_interaction_scores(
                    X_sample[:, pair[0]], X_sample[:, pair[1]], y_sample
                )
                
                best_score = max(scores.values())
                best_type = max(scores, key=scores.get)
                
                total_samples = candidate.sample_size + new_sample_size
                weight_old = candidate.sample_size / total_samples
                weight_new = new_sample_size / total_samples
                
                candidate.score = weight_old * candidate.score + weight_new * best_score
                candidate.sample_size = total_samples
                candidate.confidence = min(1.0, candidate.confidence + 0.15)
                candidate.interaction_type = best_type
    
    def _compute_all_interaction_scores(self, feat1: cp.ndarray, feat2: cp.ndarray, 
                                       y: cp.ndarray) -> Dict[str, float]:
        """Compute all interaction types"""
        scores = {}
        
        try:
            scores['multiply'] = float(cp.abs(cp.corrcoef(feat1 * feat2, y)[0, 1]))
            scores['add'] = float(cp.abs(cp.corrcoef(feat1 + feat2, y)[0, 1]))
            scores['subtract'] = float(cp.abs(cp.corrcoef(feat1 - feat2, y)[0, 1]))
            scores['divide'] = float(cp.abs(cp.corrcoef(feat1 / (cp.abs(feat2) + 0.1), y)[0, 1]))
            scores['distance'] = float(cp.abs(cp.corrcoef(cp.abs(feat1 - feat2), y)[0, 1]))
            scores['max'] = float(cp.abs(cp.corrcoef(cp.maximum(feat1, feat2), y)[0, 1]))
            scores['min'] = float(cp.abs(cp.corrcoef(cp.minimum(feat1, feat2), y)[0, 1]))
        except:
            scores = {'multiply': 0.0}
        
        scores = {k: v if not np.isnan(v) else 0.0 for k, v in scores.items()}
        
        return scores
    
    def _compute_interaction_score(self, feat1: cp.ndarray, feat2: cp.ndarray, 
                                  y: cp.ndarray) -> float:
        """Quick score computation"""
        try:
            return float(cp.abs(cp.corrcoef(feat1 * feat2, y)[0, 1]))
        except:
            return 0.0
    
    def _compute_feature_importance(self, X: cp.ndarray, y: cp.ndarray) -> np.ndarray:
        """Fast feature importance"""
        n_features = X.shape[1]
        importance = cp.zeros(n_features)
        
        for i in range(n_features):
            try:
                importance[i] = cp.abs(cp.corrcoef(X[:, i], y)[0, 1])
            except:
                importance[i] = 0.0
        
        importance = cp.nan_to_num(importance, nan=0.0)
        importance = importance / (cp.sum(importance) + 1e-8)
        
        return cp.asnumpy(importance)
    
    def _update_candidate(self, pair: Tuple[int, int], i: int, j: int, 
                         score: float, interaction_type: str, 
                         sample_size: int, feature_names: List[str]):
        """Update or create candidate"""
        if pair in self.candidates:
            old = self.candidates[pair]
            weight = sample_size / (old.sample_size + sample_size)
            new_score = old.score * (1 - weight) + score * weight
            
            self.candidates[pair] = InteractionCandidate(
                feat1_idx=i, feat2_idx=j,
                score=new_score,
                sample_size=old.sample_size + sample_size,
                confidence=min(1.0, old.confidence + 0.1),
                exploration_count=old.exploration_count + 1,
                interaction_type=interaction_type if score > old.score else old.interaction_type
            )
        else:
            self.candidates[pair] = InteractionCandidate(
                feat1_idx=i, feat2_idx=j,
                score=score,
                sample_size=sample_size,
                confidence=0.4,
                interaction_type=interaction_type
            )
        
        self.feature_participation[i] += 1
        self.feature_participation[j] += 1
    
    def _adaptive_pruning(self, confidence_threshold: float):
        """Prune weak candidates"""
        if len(self.candidates) < 100:
            return
        
        scores = np.array([c.score for c in self.candidates.values()])
        score_threshold = np.percentile(scores, 30)
        
        new_candidates = {}
        for pair, candidate in self.candidates.items():
            if (candidate.score > score_threshold or
                (candidate.confidence < confidence_threshold and 
                 candidate.score > score_threshold * 0.7) or
                (self.feature_participation[pair[0]] < 5 or 
                 self.feature_participation[pair[1]] < 5)):
                new_candidates[pair] = candidate
        
        self.candidates = new_candidates
    
    def _mutual_information(self, x: cp.ndarray, y: cp.ndarray, n_bins: int = 10) -> float:
        """Fast MI estimation"""
        try:
            if cp.std(x) < 1e-10 or cp.std(y) < 1e-10:
                return 0.0
            
            x_bins = cp.linspace(cp.min(x), cp.max(x) + 1e-10, n_bins + 1)
            y_bins = cp.linspace(cp.min(y), cp.max(y) + 1e-10, n_bins + 1)
            
            x_idx = cp.clip(cp.digitize(x, x_bins) - 1, 0, n_bins - 1)
            y_idx = cp.clip(cp.digitize(y, y_bins) - 1, 0, n_bins - 1)
            
            hist = cp.zeros((n_bins, n_bins))
            for i in range(len(x_idx)):
                hist[x_idx[i], y_idx[i]] += 1
            
            hist = hist / cp.sum(hist)
            
            px = cp.sum(hist, axis=1)
            py = cp.sum(hist, axis=0)
            
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if hist[i, j] > 1e-10:
                        mi += hist[i, j] * cp.log(hist[i, j] / (px[i] * py[j] + 1e-10) + 1e-10)
            
            return float(max(0, mi))
        except:
            return 0.0
    
    def _final_validation_advanced(self, X: cp.ndarray, y: cp.ndarray, 
                                  feature_names: List[str]) -> pd.DataFrame:
        """Final validation on large sample"""
        n_samples = X.shape[0]
        
        final_sample_size = min(n_samples, self.max_sample_size * 2)
        sample_idx = cp.random.choice(n_samples, size=final_sample_size, replace=False)
        X_final = X[sample_idx]
        y_final = y[sample_idx]
        
        top_candidates = sorted(self.candidates.items(), 
                               key=lambda x: x[1].score, 
                               reverse=True)[:self.top_interactions]
        
        print(f"Validating {len(top_candidates)} candidates on {final_sample_size:,} samples...")
        
        results = []
        for (feat1, feat2), candidate in top_candidates:
            all_scores = self._compute_all_interaction_scores(
                X_final[:, feat1], X_final[:, feat2], y_final
            )
            
            corr1 = float(cp.abs(cp.corrcoef(X_final[:, feat1], y_final)[0, 1]))
            corr2 = float(cp.abs(cp.corrcoef(X_final[:, feat2], y_final)[0, 1]))
            
            best_score = max(all_scores.values())
            synergy = best_score - max(corr1, corr2)
            
            results.append({
                'feature_1': feature_names[feat1],
                'feature_2': feature_names[feat2],
                'interaction_score': best_score,
                'interaction_type': max(all_scores, key=all_scores.get),
                'individual_score_1': corr1,
                'individual_score_2': corr2,
                'synergy': synergy,
                'confidence': candidate.confidence,
                'samples_tested': candidate.sample_size,
                'multiply_score': all_scores.get('multiply', 0),
                'distance_score': all_scores.get('distance', 0),
                'divide_score': all_scores.get('divide', 0)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('interaction_score', ascending=False)
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total interactions: {len(results_df)}")
        print(f"High synergy (>0.02): {len(results_df[results_df['synergy'] > 0.02])}")
        print(f"Very high synergy (>0.05): {len(results_df[results_df['synergy'] > 0.05])}")
        if len(results_df) > 0:
            print(f"Best interaction score: {results_df['interaction_score'].max():.4f}")
            print(f"Best synergy: {results_df['synergy'].max():.4f}")
        
        return results_df
    
    def _emergency_results(self, feature_names: List[str]) -> pd.DataFrame:
        """Fallback if validation fails"""
        results = []
        for (feat1, feat2), candidate in list(self.candidates.items())[:self.top_interactions]:
            results.append({
                'feature_1': feature_names[feat1],
                'feature_2': feature_names[feat2],
                'interaction_score': candidate.score,
                'interaction_type': candidate.interaction_type,
                'confidence': candidate.confidence
            })
        return pd.DataFrame(results)
    
    def _cleanup_memory(self):
        """Clean GPU memory"""
        self.interaction_cache.clear()
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


def create_all_interaction_features_memory_efficient(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                                    interactions_df: pd.DataFrame, 
                                                    n_interactions: int = None,
                                                    batch_size: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Create interaction features in batches to manage memory"""
    
    # Use config value if not provided
    n_interactions = n_interactions or config.N_INTERACTIONS_TO_CREATE
    
    print(f"\nCreating {n_interactions} interaction features in batches of {batch_size}...")
    
    feature_names = list(train_df.columns)
    all_interaction_dfs_train = []
    all_interaction_dfs_test = []
    all_interaction_names = []
    
    # Process in batches
    for batch_start in range(0, n_interactions, batch_size):
        batch_end = min(batch_start + batch_size, n_interactions, len(interactions_df))
        
        print(f"Processing batch: interactions {batch_start} to {batch_end}")
        
        interaction_features_train = {}
        interaction_features_test = {}
        interaction_names = []
        
        # Create interactions for this batch
        for idx in range(batch_start, batch_end):
            row = interactions_df.iloc[idx]
            feat1, feat2 = row['feature_1'], row['feature_2']
            interaction_type = row.get('interaction_type', 'multiply')
            
            if feat1 in feature_names and feat2 in feature_names:
                feat_name = f'{feat1}_{interaction_type}_{feat2}'
                
                # Calculate interaction values based on type
                if interaction_type == 'multiply':
                    interaction_features_train[feat_name] = train_df[feat1].values * train_df[feat2].values
                    interaction_features_test[feat_name] = test_df[feat1].values * test_df[feat2].values
                elif interaction_type == 'distance':
                    interaction_features_train[feat_name] = np.abs(train_df[feat1].values - train_df[feat2].values)
                    interaction_features_test[feat_name] = np.abs(test_df[feat1].values - test_df[feat2].values)
                elif interaction_type == 'subtract':
                    interaction_features_train[feat_name] = train_df[feat1].values - train_df[feat2].values
                    interaction_features_test[feat_name] = test_df[feat1].values - test_df[feat2].values
                elif interaction_type == 'add':
                    interaction_features_train[feat_name] = train_df[feat1].values + train_df[feat2].values
                    interaction_features_test[feat_name] = test_df[feat1].values + test_df[feat2].values
                elif interaction_type == 'divide':
                    interaction_features_train[feat_name] = train_df[feat1].values / (np.abs(train_df[feat2].values) + 1e-10)
                    interaction_features_test[feat_name] = test_df[feat1].values / (np.abs(test_df[feat2].values) + 1e-10)
                elif interaction_type == 'min':
                    interaction_features_train[feat_name] = np.minimum(train_df[feat1].values, train_df[feat2].values)
                    interaction_features_test[feat_name] = np.minimum(test_df[feat1].values, test_df[feat2].values)
                elif interaction_type == 'max':
                    interaction_features_train[feat_name] = np.maximum(train_df[feat1].values, train_df[feat2].values)
                    interaction_features_test[feat_name] = np.maximum(test_df[feat1].values, test_df[feat2].values)
                else:
                    interaction_features_train[feat_name] = train_df[feat1].values * train_df[feat2].values
                    interaction_features_test[feat_name] = test_df[feat1].values * test_df[feat2].values
                
                interaction_names.append(feat_name)
        
        # Create dataframes for this batch
        if interaction_names:
            batch_df_train = pd.DataFrame(interaction_features_train, index=train_df.index)
            batch_df_test = pd.DataFrame(interaction_features_test, index=test_df.index)
            
            all_interaction_dfs_train.append(batch_df_train)
            all_interaction_dfs_test.append(batch_df_test)
            all_interaction_names.extend(interaction_names)
        
        # Clean up memory
        del interaction_features_train, interaction_features_test
        gc.collect()
    
    # Combine all batches
    if all_interaction_dfs_train:
        interactions_df_train = pd.concat(all_interaction_dfs_train, axis=1)
        interactions_df_test = pd.concat(all_interaction_dfs_test, axis=1)
    else:
        interactions_df_train = pd.DataFrame(index=train_df.index)
        interactions_df_test = pd.DataFrame(index=test_df.index)
    
    print(f"Created {len(all_interaction_names)} interaction features")
    
    return interactions_df_train, interactions_df_test, all_interaction_names


def select_and_transform_interactions(interactions_train: pd.DataFrame, 
                                     interactions_test: pd.DataFrame,
                                     train_labels: pd.Series,
                                     n_select: int = None,
                                     n_components: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Apply SelectKBest then PCA to interaction features"""
    
    # Use config values if not provided
    n_select = n_select or config.N_INTERACTIONS_TO_SELECT
    n_components = n_components or config.N_PCA_COMPONENTS
    
    print(f"\nSelecting top {n_select} interactions...")
    
    # Step 1: SelectKBest on interactions
    selector = SelectKBest(score_func=f_regression, k=min(n_select, interactions_train.shape[1]))
    selector.fit(interactions_train, train_labels)
    
    # Get selected features
    selected_mask = selector.get_support()
    selected_columns = interactions_train.columns[selected_mask]
    
    interactions_train_selected = interactions_train[selected_columns]
    interactions_test_selected = interactions_test[selected_columns]
    
    print(f"Selected {len(selected_columns)} best interactions")
    
    # Step 2: Apply PCA to selected interactions
    print(f"Applying PCA to reduce to {n_components} components...")
    
    # Scale the data first
    scaler = StandardScaler()
    interactions_train_scaled = scaler.fit_transform(interactions_train_selected)
    interactions_test_scaled = scaler.transform(interactions_test_selected)
    
    # Apply PCA
    n_components_actual = min(n_components, interactions_train_scaled.shape[1])
    pca = PCA(n_components=n_components_actual, random_state=42)
    
    pca_train = pca.fit_transform(interactions_train_scaled)
    pca_test = pca.transform(interactions_test_scaled)
    
    # Create PCA feature names
    pca_feature_names = [f'AI_PCA_{i+1}' for i in range(n_components_actual)]
    
    # Create dataframes
    pca_df_train = pd.DataFrame(pca_train, columns=pca_feature_names, index=interactions_train.index)
    pca_df_test = pd.DataFrame(pca_test, columns=pca_feature_names, index=interactions_test.index)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca_df_train, pca_df_test, pca_feature_names


def create_manual_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          original_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Create manual domain-specific features"""
    
    print("\n=== ADDING MANUAL FEATURE INTERACTIONS ===")
    
    manual_features_dict_train = {}
    manual_features_dict_test = {}
    manual_features = []

    # 1. Market microstructure interactions (limited to most important)
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    existing_market_features = [f for f in market_features if f in original_features]

    # Only create most essential interactions
    if len(existing_market_features) >= 2:
        # Just a few key interactions
        if 'bid_qty' in existing_market_features and 'ask_qty' in existing_market_features:
            feat_name = 'bid_ask_ratio'
            manual_features_dict_train[feat_name] = train_df['bid_qty'].values / (train_df['ask_qty'].values + 1e-10)
            manual_features_dict_test[feat_name] = test_df['bid_qty'].values / (test_df['ask_qty'].values + 1e-10)
            manual_features.append(feat_name)
        
        if 'buy_qty' in existing_market_features and 'sell_qty' in existing_market_features:
            feat_name = 'buy_sell_ratio'
            manual_features_dict_train[feat_name] = train_df['buy_qty'].values / (train_df['sell_qty'].values + 1e-10)
            manual_features_dict_test[feat_name] = test_df['buy_qty'].values / (test_df['sell_qty'].values + 1e-10)
            manual_features.append(feat_name)

    # 2. Create order flow imbalance features
    if 'buy_qty' in original_features and 'sell_qty' in original_features:
        manual_features_dict_train['order_imbalance'] = (train_df['buy_qty'].values - train_df['sell_qty'].values) / (train_df['buy_qty'].values + train_df['sell_qty'].values + 1e-10)
        manual_features_dict_test['order_imbalance'] = (test_df['buy_qty'].values - test_df['sell_qty'].values) / (test_df['buy_qty'].values + test_df['sell_qty'].values + 1e-10)
        manual_features.append('order_imbalance')

    if 'bid_qty' in original_features and 'ask_qty' in original_features:
        manual_features_dict_train['bid_ask_spread'] = (train_df['ask_qty'].values - train_df['bid_qty'].values) / (train_df['ask_qty'].values + train_df['bid_qty'].values + 1e-10)
        manual_features_dict_test['bid_ask_spread'] = (test_df['ask_qty'].values - test_df['bid_qty'].values) / (test_df['ask_qty'].values + test_df['bid_qty'].values + 1e-10)
        manual_features.append('bid_ask_spread')

    # Create dataframes
    if manual_features:
        manual_df_train = pd.DataFrame(manual_features_dict_train, index=train_df.index)
        manual_df_test = pd.DataFrame(manual_features_dict_test, index=test_df.index)
    else:
        manual_df_train = pd.DataFrame(index=train_df.index)
        manual_df_test = pd.DataFrame(index=test_df.index)

    print(f"Created {len(manual_features)} manual interaction features")
    
    return manual_df_train, manual_df_test, manual_features


# ============================================
# MAIN PIPELINE WITH CONFIGURABLE PARAMETERS
# ============================================

print("=== LOADING DATA ===")
train = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train = train.to_pandas()
test = pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
test = test.to_pandas()
print(f"train.shape: {train.shape}, test.shape: {test.shape}")

features = [c for c in train.columns if c != 'label']
label_col = 'label'

print("\n=== CRYPTO TRADING DATA EDA ===")
print(f"Dataset shape: {train.shape}")
print(f"Number of features: {len(features)}")

# Basic preprocessing
NUNIQUE1 = [c for c in train.columns if train[c].nunique() == 1]
train.drop(NUNIQUE1 + ['timestamp'], axis=1, inplace=True)
test.drop(NUNIQUE1 + ['label'], axis=1, inplace=True)
print(f"Removed {len(NUNIQUE1)} features with only 1 unique value")

features = [c for c in train.columns if c != 'label']

# Store original features before interaction discovery
original_features = features.copy()

# ========== STEP 1: FEATURE INTERACTION DISCOVERY ==========
print("\n" + "="*70)
print("=== AI-POWERED FEATURE INTERACTION DISCOVERY ===")
print("="*70)

# Check if we already have saved interactions
if os.path.exists(config.INTERACTIONS_FILENAME):
    print(f"Found existing {config.INTERACTIONS_FILENAME} - loading discovered interactions...")
    interaction_results = pd.read_csv(config.INTERACTIONS_FILENAME)
    print(f"Loaded {len(interaction_results)} pre-discovered interactions")
    if 'synergy' in interaction_results.columns and len(interaction_results) > 0:
        print(f"Best synergy score: {interaction_results['synergy'].max():.4f}")
else:
    print(f"Running feature interaction discovery using {config.SEARCH_N_PHASES} search phases...")
    
    # Prepare data for GPU
    X_for_search = train[features].tail(config.INTERACTION_DISCOVERY_SAMPLE_SIZE)
    y_for_search = train['label'].tail(config.INTERACTION_DISCOVERY_SAMPLE_SIZE)
    
    # Convert to GPU
    X_gpu = cp.asarray(X_for_search.values)
    y_gpu = cp.asarray(y_for_search.values)
    
    # Initialize searcher with config parameters
    searcher = WorldClassInteractionSearch()
    
    # Run search
    start_time = time.time()
    interaction_results = searcher.search(X_gpu, y_gpu, features)
    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Save results
    interaction_results.to_csv(config.INTERACTIONS_FILENAME, index=False)
    print(f"Saved interaction results to '{config.INTERACTIONS_FILENAME}'")
    
    # Clean GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

# Display top synergistic interactions
if len(interaction_results) > 0 and 'synergy' in interaction_results.columns:
    print(f"\n=== TOP {config.N_TOP_SYNERGIES_TO_DISPLAY} SYNERGISTIC INTERACTIONS DISCOVERED ===")
    top_synergy = interaction_results.nlargest(min(config.N_TOP_SYNERGIES_TO_DISPLAY, len(interaction_results)), 'synergy')
    for idx, row in top_synergy.iterrows():
        print(f"{row['feature_1']} × {row['feature_2']} ({row.get('interaction_type', 'multiply')}) - "
              f"Synergy: {row['synergy']:.4f}, Score: {row['interaction_score']:.4f}")

# ========== STEP 2: CREATE ALL INTERACTION FEATURES ==========
interactions_train, interactions_test, interaction_feature_names = create_all_interaction_features_memory_efficient(
    train, test, interaction_results
)

# ========== STEP 3: SELECT BEST INTERACTIONS AND APPLY PCA ==========
if interactions_train.shape[1] > 0:
    pca_train, pca_test, pca_feature_names = select_and_transform_interactions(
        interactions_train, interactions_test, 
        train['label']
    )
else:
    print("No interactions created, skipping PCA")
    pca_train = pd.DataFrame(index=train.index)
    pca_test = pd.DataFrame(index=test.index)
    pca_feature_names = []

# Clean up intermediate data
del interactions_train, interactions_test
gc.collect()

# ========== STEP 4: CREATE MANUAL FEATURES ==========
manual_train, manual_test, manual_feature_names = create_manual_features(
    train, test, original_features
)

# ========== STEP 5: COMBINE ALL FEATURES ==========
print("\n=== COMBINING ALL FEATURE TYPES ===")

# Combine original features, PCA components, and manual features
train_combined = pd.concat([
    train[original_features],
    pca_train,
    manual_train
], axis=1)

test_combined = pd.concat([
    test[original_features],
    pca_test,
    manual_test
], axis=1)

# Add label back to train
train_combined['label'] = train['label']

print(f"Combined feature set size: {train_combined.shape[1] - 1} features")
print(f"  - Original features: {len(original_features)}")
print(f"  - PCA components: {len(pca_feature_names)}")
print(f"  - Manual features: {len(manual_feature_names)}")

# ========== STEP 6: FINAL FEATURE SELECTION ==========
print("\n=== FINAL FEATURE SELECTION ===")

# Combine all feature names
all_feature_names = original_features + pca_feature_names + manual_feature_names

# Sample data for feature selection to save memory
sample_size = min(100000, len(train_combined))
sample_indices = np.random.choice(len(train_combined), sample_size, replace=False)

# Apply final SelectKBest on sample
k_final = min(config.N_FINAL_FEATURES, len(all_feature_names))
selector_final = SelectKBest(score_func=f_regression, k=k_final)
selector_final.fit(train_combined.iloc[sample_indices][all_feature_names], 
                  train_combined.iloc[sample_indices]['label'])

# Get selected features
selected_indices = selector_final.get_support(indices=True)
selected_features = [all_feature_names[i] for i in selected_indices]

# Extract only selected features
train_final = train_combined[selected_features + ['label']].copy()
test_final = test_combined[selected_features].copy()

print(f"\nFinal feature composition:")
print(f"Original features: {len([f for f in selected_features if f in original_features])}")
print(f"PCA components: {len([f for f in selected_features if 'AI_PCA_' in f])}")
print(f"Manual features: {len([f for f in selected_features if f in manual_feature_names])}")
print(f"Total features: {len(selected_features)}")

# Clean up memory
del train, test, train_combined, test_combined
gc.collect()

# ========== DATA SAMPLING FOR TRAINING ==========
print("\n=== TRAINING DATA SAMPLING ===")

# Apply training data percentage sampling if configured
if config.TRAIN_DATA_PERCENTAGE < 100:
    n_total_samples = len(train_final)
    n_samples_to_use = int(n_total_samples * config.TRAIN_DATA_PERCENTAGE / 100)
    
    print(f"Sampling {config.TRAIN_DATA_PERCENTAGE}% of training data...")
    print(f"Total samples: {n_total_samples:,}")
    print(f"Samples to use: {n_samples_to_use:,}")
    
    if config.TRAIN_DATA_RECENT_ONLY:
        # Use most recent data
        print("Using most recent samples...")
        train_final_sampled = train_final.tail(n_samples_to_use).copy()
    else:
        # Random sampling
        print("Using random sampling...")
        sample_indices = np.random.choice(n_total_samples, size=n_samples_to_use, replace=False)
        train_final_sampled = train_final.iloc[sample_indices].copy()
else:
    print("Using 100% of training data...")
    train_final_sampled = train_final

print(f"Final training set size: {len(train_final_sampled):,} samples")

# ========== FLAML TRAINING ==========
print("\n=== FLAML MODEL TRAINING ===")
from flaml import AutoML
model = AutoML()

# Calculate time budget in seconds
time_budget_seconds = int(config.TRAINING_TIME_HOURS * 3600)

settings = {
    "time_budget": time_budget_seconds,
    "task": config.TRAINING_TASK,
    "metric": config.TRAINING_METRIC,
    "n_concurrent_trials": config.TRAINING_N_CONCURRENT_TRIALS,
    "free_mem_ratio": config.TRAINING_FREE_MEM_RATIO,
}

print("Starting FLAML AutoML training...")
print(f"Time budget: {config.TRAINING_TIME_HOURS} hours")
print(f"Number of features: {len(selected_features)}")
print(f"Training samples: {len(train_final_sampled):,}")

model.fit(X_train=train_final_sampled[selected_features], y_train=train_final_sampled['label'], **settings)

print("\n=== FLAML RESULTS ===")
print(f"Best model: {model.best_estimator}")
print(f"Best MSE: {model.best_loss:.6f}")
print(f"Best config:")
for key, value in model.best_config.items():
    print(f"  {key}: {value}")

# Make predictions
test_preds = model.predict(test_final[selected_features])

# Save submission
sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sub['prediction'] = test_preds
sub.to_csv(config.SUBMISSION_FILENAME, index=None)
print(f"\nPredictions saved to '{config.SUBMISSION_FILENAME}'")

# Save feature importance if available
try:
    if hasattr(model.model.estimator, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.model.estimator.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== TOP {config.N_TOP_FEATURES_TO_DISPLAY} MOST IMPORTANT FEATURES ===")
        print(feature_importance.head(config.N_TOP_FEATURES_TO_DISPLAY))
        
        # Check which feature types are most important
        pca_importance = feature_importance[feature_importance['feature'].str.contains('AI_PCA_')]
        if len(pca_importance) > 0:
            print(f"\n=== TOP {config.N_TOP_PCA_TO_DISPLAY} PCA COMPONENTS ===")
            print(pca_importance.head(config.N_TOP_PCA_TO_DISPLAY))
except:
    print("\nFeature importance not available for this model type")

