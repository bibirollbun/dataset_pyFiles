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


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, differential_evolution
from datetime import datetime
import warnings
import time
import os
import glob
warnings.filterwarnings('ignore')

print("="*80)
print("HIERARCHICAL BLEND WITH OOF-BASED WEIGHT OPTIMIZATION + EARLY STOPPING")
print("="*80)

# ===========================
# CONFIGURATION
# ===========================
class Config:
    """Configuration for h-blend ensemble"""
    target = 'accident_risk'
    id_column = 'id'
    n_bins_asc_desc = 10
    random_state = 42
    
    # Optimization settings
    optimize_method = 'differential_evolution'  # or 'nelder-mead', 'powell'
    max_iterations = 1000
    
    # EARLY STOPPING
    early_stopping_enabled = True
    early_stopping_patience = 15  # Stop after 15 iterations without improvement
    early_stopping_min_delta = 1e-7  # Minimum improvement to be considered
    
    # VERBOSITY CONTROLS
    verbose_loading = True      # Show file loading progress
    verbose_optimization = 2    # 0=silent, 1=minimal, 2=detailed, 3=very detailed
    verbose_blending = False    # Show per-sample blending (slow, use for debugging)
    show_progress_every = 10    # Show progress every N iterations
    
    # OUTPUT FILES
    model_name = "HBlend"  # Name for output files
    save_npy = True  # Save .npy files for OOF and test predictions
    save_csv = True  # Save submission.csv

# ===========================
# KAGGLE PATHS
# ===========================
TRAIN_PATH = '/kaggle/input/playground-series-s5e10/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e10/test.csv'

# Auto-discover notebooks with prediction files
NOTEBOOK_PATHS = []
base_path = '/kaggle/input/'
for item in os.listdir(base_path):
    item_path = os.path.join(base_path, item)
    if os.path.isdir(item_path) and item != 'playground-series-s5e10':
        NOTEBOOK_PATHS.append(item_path)

print(f"\nDiscovered {len(NOTEBOOK_PATHS)} attached notebooks:")
for path in NOTEBOOK_PATHS:
    print(f"  â€¢ {os.path.basename(path)}")

# ===========================
# AUTO-DISCOVER PREDICTION FILES
# ===========================
def discover_prediction_files():
    """Auto-discover OOF and test prediction files from attached notebooks"""
    models = []
    
    for notebook_path in NOTEBOOK_PATHS:
        notebook_name = os.path.basename(notebook_path)
        
        # Find all .npy files
        npy_files = glob.glob(os.path.join(notebook_path, '*.npy'))
        
        # Pair OOF and test files
        oof_files = [f for f in npy_files if 'oof' in os.path.basename(f).lower()]
        test_files = [f for f in npy_files if 'test' in os.path.basename(f).lower()]
        
        # Try to match OOF with test files
        for oof_file in oof_files:
            oof_basename = os.path.basename(oof_file).replace('oof_', '').replace('oof', '')
            
            # Find matching test file
            test_file = None
            for tf in test_files:
                test_basename = os.path.basename(tf).replace('test_', '').replace('test', '')
                if test_basename == oof_basename or oof_basename in test_basename:
                    test_file = tf
                    break
            
            if test_file:
                # Extract model name from filename
                model_name = oof_basename.replace('.npy', '')
                if not model_name:
                    model_name = notebook_name
                
                models.append({
                    'name': model_name,
                    'oof_path': oof_file,
                    'test_path': test_file,
                    'notebook': notebook_name
                })
    
    return models

print("\n" + "="*80)
print("AUTO-DISCOVERING PREDICTION FILES")
print("="*80)

MODELS = discover_prediction_files()

if len(MODELS) == 0:
    print("\nâš ï¸�  WARNING: No prediction files found!")
    print("Please ensure notebooks are attached with OOF and test .npy files")
    print("\nExpected file naming pattern:")
    print("  â€¢ oof_*.npy (OOF predictions)")
    print("  â€¢ test_*.npy (test predictions)")
else:
    print(f"\nâœ“ Found {len(MODELS)} model prediction pairs:")
    for i, model in enumerate(MODELS, 1):
        print(f"\n  [{i}] {model['name']}")
        print(f"      Notebook: {model['notebook']}")
        print(f"      OOF:  {os.path.basename(model['oof_path'])}")
        print(f"      Test: {os.path.basename(model['test_path'])}")

# ===========================
# PROGRESS TRACKER WITH EARLY STOPPING
# ===========================
class ProgressTracker:
    """Track optimization progress with early stopping"""
    def __init__(self, patience=100, min_delta=1e-7):
        self.iteration = 0
        self.best_score = float('inf')
        self.best_iteration = 0
        self.best_params = None
        self.patience = patience
        self.min_delta = min_delta
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.should_stop = False
        self.stop_reason = None
        
    def update(self, score, params=None):
        self.iteration += 1
        improved = False
        
        # Check if improvement is significant
        if score < (self.best_score - self.min_delta):
            self.best_score = score
            self.best_iteration = self.iteration
            self.best_params = params.copy() if params is not None else None
            improved = True
        
        # Check early stopping condition
        if Config.early_stopping_enabled:
            iterations_without_improvement = self.iteration - self.best_iteration
            if iterations_without_improvement >= self.patience:
                self.should_stop = True
                self.stop_reason = f"No improvement for {self.patience} iterations"
                if Config.verbose_optimization >= 1:
                    print(f"\n{'='*80}")
                    print(f"âš ï¸�  EARLY STOPPING TRIGGERED")
                    print(f"{'='*80}")
                    print(f"  Reason: {self.stop_reason}")
                    print(f"  Current iteration: {self.iteration}")
                    print(f"  Best iteration: {self.best_iteration}")
                    print(f"  Best score: {self.best_score:.6f}")
                    print(f"  Time saved: ~{self.estimate_time_saved():.1f}s")
                    print(f"{'='*80}\n")
        
        return improved
    
    def estimate_time_saved(self):
        """Estimate time that would have been wasted"""
        elapsed = time.time() - self.start_time
        avg_time_per_iter = elapsed / max(self.iteration, 1)
        
        if Config.optimize_method == 'differential_evolution':
            remaining_iters = (100 * 15) - self.iteration  # maxiter * popsize
        else:
            remaining_iters = Config.max_iterations - self.iteration
        
        return remaining_iters * avg_time_per_iter
    
    def should_print(self):
        """Check if we should print based on interval"""
        current_time = time.time()
        if current_time - self.last_print_time >= 5:  # Print every 5 seconds
            self.last_print_time = current_time
            return True
        return False
    
    def get_elapsed(self):
        """Get elapsed time as string"""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"
    
    def get_summary(self):
        """Get summary of optimization"""
        return {
            'total_iterations': self.iteration,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'early_stopped': self.should_stop,
            'stop_reason': self.stop_reason,
            'elapsed_time': time.time() - self.start_time
        }

# Global tracker for callbacks
tracker = ProgressTracker(
    patience=15,  # Changed to 15 iterations
    min_delta=Config.early_stopping_min_delta
)

# ===========================
# LOAD DATA FUNCTION
# ===========================
def load_predictions(model_config):
    """Load predictions from .npy files"""
    if Config.verbose_loading:
        print(f"  Loading {model_config['name']}...", end='', flush=True)
        start_time = time.time()
    
    oof = np.load(model_config['oof_path'])
    test = np.load(model_config['test_path'])
    
    if Config.verbose_loading:
        elapsed = time.time() - start_time
        print(f" âœ“ ({elapsed:.2f}s)")
    
    return oof, test

# ===========================
# LOAD ALL MODELS
# ===========================
print("\n" + "="*80)
print("LOADING OOF AND TEST PREDICTIONS")
print("="*80)

oof_predictions = []
test_predictions = []
model_names = []

total_start = time.time()

for i, model in enumerate(MODELS, 1):
    try:
        oof, test = load_predictions(model)
        oof_predictions.append(oof)
        test_predictions.append(test)
        model_names.append(model['name'])
        if not Config.verbose_loading:
            print(f"âœ“ [{i}/{len(MODELS)}] {model['name']}")
    except Exception as e:
        print(f"âœ— [{i}/{len(MODELS)}] {model['name']} - ERROR: {str(e)}")

total_elapsed = time.time() - total_start
n_models = len(model_names)

if n_models == 0:
    print("\nâ�Œ ERROR: No models loaded successfully!")
    print("Cannot proceed with ensemble. Please check input files.")
    exit(1)

print(f"\nâœ“ Successfully loaded {n_models} models in {total_elapsed:.2f}s")

# Load target from train.csv
if Config.verbose_loading:
    print("\nLoading target from train.csv...", end='', flush=True)
    start_time = time.time()

train_df = pd.read_csv(TRAIN_PATH, index_col='id')
y_true = train_df[Config.target].values

if Config.verbose_loading:
    elapsed = time.time() - start_time
    print(f" âœ“ ({elapsed:.2f}s)")
    print(f"Target shape: {y_true.shape}")

# Load test data for IDs and shape validation
test_df = pd.read_csv(TEST_PATH, index_col='id')
test_ids = test_df.index.values

print(f"Test shape: {test_df.shape}")
print(f"Test IDs shape: {test_ids.shape}")
print()

# ===========================
# HIERARCHICAL BLEND CLASS
# ===========================
class HierarchicalBlend:
    """Hierarchical blending with rank-based weighting"""
    
    def __init__(self, n_models, asc_weight=0.5, desc_weight=0.5):
        self.n_models = n_models
        self.asc_weight = asc_weight
        self.desc_weight = desc_weight
        self.main_weights = np.ones(n_models) / n_models
        self.rank_weights = np.zeros(n_models)
        
    def set_weights(self, main_weights, rank_weights):
        """Set weights from optimization"""
        self.main_weights = np.array(main_weights)
        self.rank_weights = np.array(rank_weights)
        
    def blend_single_direction(self, predictions, ascending=True):
        """Blend predictions with rank-based weighting"""
        n_samples = len(predictions[0])
        blended = np.zeros(n_samples)
        pred_matrix = np.column_stack(predictions)
        
        if Config.verbose_blending and Config.verbose_optimization >= 3:
            print(f"    Blending {'ascending' if ascending else 'descending'}...")
        
        for i in range(n_samples):
            sample_preds = pred_matrix[i, :]
            
            if ascending:
                rank_indices = np.argsort(sample_preds)
            else:
                rank_indices = np.argsort(sample_preds)[::-1]
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for rank_pos, model_idx in enumerate(rank_indices):
                weight = self.main_weights[model_idx] + self.rank_weights[rank_pos]
                weight = max(0, weight)
                
                weighted_sum += weight * sample_preds[model_idx]
                weight_sum += weight
            
            blended[i] = weighted_sum / weight_sum if weight_sum > 0 else np.mean(sample_preds)
        
        return blended
    
    def blend(self, predictions):
        """Full hierarchical blend"""
        blend_asc = self.blend_single_direction(predictions, ascending=True)
        blend_desc = self.blend_single_direction(predictions, ascending=False)
        blended = self.asc_weight * blend_asc + self.desc_weight * blend_desc
        return blended

# ===========================
# WEIGHT OPTIMIZATION WITH EARLY STOPPING
# ===========================
print("\n" + "="*80)
print("OPTIMIZING WEIGHTS USING OOF PREDICTIONS")
print("="*80)

def objective_function(params, oof_preds, y_true, n_models):
    """Objective function to minimize (RMSE) with early stopping check"""
    
    # EARLY STOPPING: Return best score to signal convergence
    if tracker.should_stop:
        return tracker.best_score
    
    main_weights = params[0:n_models]
    rank_weights = params[n_models:2*n_models]
    asc_weight = params[2*n_models]
    desc_weight = 1 - asc_weight
    
    blender = HierarchicalBlend(n_models, asc_weight, desc_weight)
    blender.set_weights(main_weights, rank_weights)
    
    blended = blender.blend(oof_preds)
    rmse = np.sqrt(mean_squared_error(y_true, blended))
    
    # Track progress with early stopping check
    improved = tracker.update(rmse, params)
    
    # Verbose output
    if Config.verbose_optimization >= 2 and tracker.should_print():
        elapsed = tracker.get_elapsed()
        marker = "ğŸŒŸ" if improved else "  "
        patience_remaining = Config.early_stopping_patience - (tracker.iteration - tracker.best_iteration)
        print(f"{marker} Iter {tracker.iteration:4d} | RMSE: {rmse:.6f} | Best: {tracker.best_score:.6f} | Patience: {patience_remaining:3d} | Time: {elapsed}")
    elif Config.verbose_optimization >= 1 and improved:
        elapsed = tracker.get_elapsed()
        print(f"ğŸŒŸ New best at iter {tracker.iteration}: {rmse:.6f} ({elapsed})")
    
    return rmse

# Optimization callback for scipy methods
def callback_scipy(xk):
    """Callback for scipy optimization methods"""
    if tracker.should_stop:
        return True  # Signal to stop
    
    if Config.verbose_optimization >= 3:
        rmse = objective_function(xk, oof_predictions, y_true, n_models)
        elapsed = tracker.get_elapsed()
        print(f"   Callback iter {tracker.iteration}: {rmse:.6f} ({elapsed})")
    return False

# Set up bounds
bounds = []
for i in range(n_models):
    bounds.append((0.0, 1.0))
for i in range(n_models):
    bounds.append((-0.3, 0.3))
bounds.append((0.0, 1.0))

print(f"\nOptimization Configuration:")
print(f"  Method: {Config.optimize_method}")
print(f"  Verbosity Level: {Config.verbose_optimization}")
print(f"  Parameters: {len(bounds)} ({n_models} main + {n_models} rank + 1 direction)")
print(f"  Max Iterations: {Config.max_iterations if Config.optimize_method != 'differential_evolution' else '~1500 (100 gen Ã— 15 pop)'}")
print(f"  Early Stopping: {'Enabled' if Config.early_stopping_enabled else 'Disabled'}")
if Config.early_stopping_enabled:
    print(f"    Patience: {Config.early_stopping_patience} iterations")
    print(f"    Min Delta: {Config.early_stopping_min_delta}")

initial_guess = np.concatenate([
    np.ones(n_models) / n_models,
    np.zeros(n_models),
    [0.5]
])

print("\n" + "-"*80)
print("Starting optimization...")
if Config.optimize_method == 'differential_evolution':
    print("Note: With early stopping, typically converges in 2-8 minutes")
else:
    print("Note: With early stopping, typically converges in 1-5 minutes")
print("-"*80 + "\n")

optimization_start = time.time()

if Config.optimize_method == 'differential_evolution':
    result = differential_evolution(
        lambda p: objective_function(p, oof_predictions, y_true, n_models),
        bounds=bounds,
        seed=Config.random_state,
        maxiter=100,
        popsize=15,
        tol=1e-7,
        atol=1e-7,
        workers=1,
        disp=(Config.verbose_optimization >= 2),
        updating='deferred'
    )
else:
    result = minimize(
        lambda p: objective_function(p, oof_predictions, y_true, n_models),
        x0=initial_guess,
        bounds=bounds,
        method='Powell' if Config.optimize_method == 'powell' else 'Nelder-Mead',
        callback=callback_scipy if Config.verbose_optimization >= 3 else None,
        options={
            'maxiter': Config.max_iterations,
            'disp': (Config.verbose_optimization >= 2)
        }
    )

optimization_elapsed = time.time() - optimization_start

# Get optimization summary
opt_summary = tracker.get_summary()

print("\n" + "-"*80)
print(f"âœ“ Optimization complete in {optimization_elapsed:.1f}s ({optimization_elapsed/60:.1f} min)")
print(f"  Total iterations: {opt_summary['total_iterations']}")
print(f"  Best iteration: {opt_summary['best_iteration']}")
print(f"  Final RMSE: {opt_summary['best_score']:.6f}")
if opt_summary['early_stopped']:
    print(f"  Early stopped: Yes ({opt_summary['stop_reason']})")
    time_saved = tracker.estimate_time_saved()
    print(f"  Estimated time saved: ~{time_saved:.1f}s (~{time_saved/60:.1f} min)")
else:
    print(f"  Early stopped: No (ran to completion)")
print("-"*80)

# Use best parameters if early stopping was triggered
if tracker.best_params is not None:
    best_params = tracker.best_params
    print("\nâœ“ Using best parameters from early stopping")
else:
    best_params = result.x
    print("\nâœ“ Using final parameters from optimization")

# Extract optimized weights
best_main_weights = best_params[0:n_models]
best_rank_weights = best_params[n_models:2*n_models]
best_asc_weight = best_params[2*n_models]
best_desc_weight = 1 - best_asc_weight

# Normalize main weights
best_main_weights = best_main_weights / best_main_weights.sum()

print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

print(f"\nBest OOF RMSE: {opt_summary['best_score']:.6f}")
print(f"\nOptimized Main Weights:")
for i, (name, weight) in enumerate(zip(model_names, best_main_weights)):
    bar = 'â–ˆ' * int(weight * 50)
    print(f"  {name:30s}: {weight:.4f} {bar}")

print(f"\nOptimized Rank Weights (position adjustments):")
for i, weight in enumerate(best_rank_weights):
    sign = '+' if weight >= 0 else ''
    bar = 'â–ˆ' * int(abs(weight) * 100) if weight > 0 else 'â–“' * int(abs(weight) * 100)
    print(f"  Position {i+1}: {sign}{weight:.4f} {bar}")

print(f"\nDirection Weights:")
asc_bar = 'â–ˆ' * int(best_asc_weight * 50)
desc_bar = 'â–ˆ' * int(best_desc_weight * 50)
print(f"  Ascending:  {best_asc_weight:.4f} {asc_bar}")
print(f"  Descending: {best_desc_weight:.4f} {desc_bar}")

# ===========================
# CREATE FINAL ENSEMBLE
# ===========================
print("\n" + "="*80)
print("CREATING FINAL ENSEMBLE")
print("="*80)

if Config.verbose_optimization >= 1:
    print("\nGenerating OOF predictions...", end='', flush=True)
    start_time = time.time()

final_blender = HierarchicalBlend(n_models, best_asc_weight, best_desc_weight)
final_blender.set_weights(best_main_weights, best_rank_weights)

oof_ensemble = final_blender.blend(oof_predictions)
oof_rmse = np.sqrt(mean_squared_error(y_true, oof_ensemble))

if Config.verbose_optimization >= 1:
    elapsed = time.time() - start_time
    print(f" âœ“ ({elapsed:.2f}s)")
    print(f"Generating test predictions...", end='', flush=True)
    start_time = time.time()

test_ensemble = final_blender.blend(test_predictions)

if Config.verbose_optimization >= 1:
    elapsed = time.time() - start_time
    print(f" âœ“ ({elapsed:.2f}s)")

print(f"\nâœ“ OOF ensemble RMSE: {oof_rmse:.6f}")

# ===========================
# COMPARE WITH INDIVIDUAL MODELS
# ===========================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

print(f"\n{'Model':<30s} {'OOF RMSE':<12s} {'Improvement':<15s} {'Visual'}")
print("-" * 80)

individual_rmses = []
for i, (name, oof) in enumerate(zip(model_names, oof_predictions)):
    rmse = np.sqrt(mean_squared_error(y_true, oof))
    individual_rmses.append(rmse)
    improvement = rmse - oof_rmse
    sign = '+' if improvement > 0 else ''
    bar = 'â–ˆ' * int(improvement * 10000) if improvement > 0 else ''
    print(f"{name:<30s} {rmse:>10.6f}  {sign}{improvement:>10.6f}  {bar}")

best_individual = min(individual_rmses)
ensemble_improvement = best_individual - oof_rmse
bar = 'â–ˆ' * int(ensemble_improvement * 10000)

print("-" * 80)
print(f"{'H-Blend Ensemble':<30s} {oof_rmse:>10.6f}  +{ensemble_improvement:>10.6f}  {bar}")
print(f"\nâœ“ Ensemble improves by {ensemble_improvement:.6f} over best individual model")

# ===========================
# SAVE OUTPUTS WITH PROPER NAMING
# ===========================
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Create timestamp and filename components
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rmse_str = f"CV{oof_rmse:.6f}".replace('.', '_')

# Generate filenames with: Date_ModelName_CVRMSE format
base_filename = f"{timestamp}_{Config.model_name}_{rmse_str}"

saved_files = []

# Save OOF predictions (NPY)
if Config.save_npy:
    oof_npy_filename = f"oof_{base_filename}.npy"
    np.save(oof_npy_filename, oof_ensemble)
    saved_files.append(oof_npy_filename)

# Save Test predictions (NPY)
if Config.save_npy:
    test_npy_filename = f"test_{base_filename}.npy"
    np.save(test_npy_filename, test_ensemble)
    saved_files.append(test_npy_filename)

# Save submission CSV
if Config.save_csv:
    submission_filename = "submission.csv"  # Simple filename
    submission = pd.DataFrame({
        Config.id_column: test_ids,
        Config.target: test_ensemble
    })
    submission.to_csv(submission_filename, index=False)  # Changed to index=False
    saved_files.append(submission_filename)

# Save configuration and weights
config_filename = f"config_{base_filename}.txt"
with open(config_filename, 'w') as f:
    f.write("="*70 + "\n")
    f.write("HIERARCHICAL BLEND CONFIGURATION\n")
    f.write("="*70 + "\n\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Model: {Config.model_name}\n")
    f.write(f"OOF RMSE (CV Score): {oof_rmse:.6f}\n")
    f.write(f"Number of Models: {n_models}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("INPUT MODELS\n")
    f.write("-"*70 + "\n")
    for i, model in enumerate(MODELS, 1):
        f.write(f"\n[{i}] {model['name']}\n")
        f.write(f"    Notebook: {model['notebook']}\n")
        f.write(f"    OOF:  {os.path.basename(model['oof_path'])}\n")
        f.write(f"    Test: {os.path.basename(model['test_path'])}\n")
    f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("OPTIMIZATION SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(f"Method: {Config.optimize_method}\n")
    f.write(f"Total Iterations: {opt_summary['total_iterations']}\n")
    f.write(f"Best Iteration: {opt_summary['best_iteration']}\n")
    f.write(f"Optimization Time: {optimization_elapsed:.1f}s ({optimization_elapsed/60:.1f} min)\n")
    f.write(f"Early Stopping: {'Yes' if opt_summary['early_stopped'] else 'No'}\n")
    if opt_summary['early_stopped']:
        f.write(f"Stop Reason: {opt_summary['stop_reason']}\n")
        f.write(f"Time Saved: ~{tracker.estimate_time_saved():.1f}s\n")
    f.write(f"Early Stopping Patience: {Config.early_stopping_patience}\n")
    f.write(f"Min Delta: {Config.early_stopping_min_delta}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("OPTIMIZED WEIGHTS\n")
    f.write("-"*70 + "\n\n")
    
    f.write("Main Weights:\n")
    for name, weight in zip(model_names, best_main_weights):
        f.write(f"  {name:30s}: {weight:.6f}\n")
    
    f.write("\nRank Weights (position adjustments):\n")
    for i, weight in enumerate(best_rank_weights):
        f.write(f"  Position {i+1}: {weight:+.6f}\n")
    
    f.write(f"\nDirection Weights:\n")
    f.write(f"  Ascending:  {best_asc_weight:.6f}\n")
    f.write(f"  Descending: {best_desc_weight:.6f}\n\n")
    
    f.write("="*70 + "\n")
    f.write("PERFORMANCE COMPARISON\n")
    f.write("="*70 + "\n\n")
    
    for name, rmse in zip(model_names, individual_rmses):
        improvement = rmse - oof_rmse
        f.write(f"{name:30s}: {rmse:.6f} ({improvement:+.6f})\n")
    
    f.write(f"\nH-Blend Ensemble: {oof_rmse:.6f}\n")
    f.write(f"Improvement over best: {ensemble_improvement:+.6f}\n\n")
    
    f.write("="*70 + "\n")
    f.write("OUTPUT FILES\n")
    f.write("="*70 + "\n\n")
    for filename in saved_files:
        f.write(f"  {filename}\n")

saved_files.append(config_filename)

# ===========================
# LIST ALL SAVED FILES (ONCE)
# ===========================
print(f"\nâœ“ Saved {len(saved_files)} output files:")
for filename in saved_files:
    print(f"  â€¢ {filename}")

# ===========================
# STATISTICS
# ===========================
print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)

print(f"\nOOF Ensemble:")
print(f"  Mean:  {oof_ensemble.mean():.6f}")
print(f"  Std:   {oof_ensemble.std():.6f}")
print(f"  Min:   {oof_ensemble.min():.6f}")
print(f"  Max:   {oof_ensemble.max():.6f}")

print(f"\nTest Ensemble:")
print(f"  Mean:  {test_ensemble.mean():.6f}")
print(f"  Std:   {test_ensemble.std():.6f}")
print(f"  Min:   {test_ensemble.min():.6f}")
print(f"  Max:   {test_ensemble.max():.6f}")

print(f"\nSubmission DataFrame:")
print(f"  Shape: {submission.shape}")
print(f"  Expected test shape: {test_df.shape}")
print(f"  Match: {'âœ“' if submission.shape[0] == test_df.shape[0] else 'âœ—'}")
print(f"  Columns: {list(submission.columns)}")
print(f"  ID range: {submission[Config.id_column].min()} to {submission[Config.id_column].max()}")

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "="*80)
print("âœ… HIERARCHICAL BLEND COMPLETE!")
print("="*80)

total_elapsed = time.time() - total_start
print(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
print(f"  Loading: {total_elapsed - optimization_elapsed:.1f}s")
print(f"  Optimization: {optimization_elapsed:.1f}s ({optimization_elapsed/60:.1f} min)")

print(f"\nKey Results:")
print(f"  âœ“ Final CV RMSE: {oof_rmse:.6f}")
print(f"  âœ“ Improvement: +{ensemble_improvement:.6f}")
print(f"  âœ“ Total Iterations: {opt_summary['total_iterations']}")
print(f"  âœ“ Best Iteration: {opt_summary['best_iteration']}")
if opt_summary['early_stopped']:
    print(f"  âœ“ Early Stopped: Yes (saved ~{tracker.estimate_time_saved()/60:.1f} min)")

print("\n" + "="*80)
print("File naming format: YYYYMMDD_HHMMSS_ModelName_CV0_XXXXXX")
print("="*80)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set unified scientific color scheme
plt.style.use('seaborn-v0_8-whitegrid')

SCIENTIFIC_COLORS = {
    'model_palette': ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8'],
    'ascending': '#2ca02c',
    'descending': '#d62728',
    'ensemble': '#9467bd',
    'neutral': '#7f7f7f'
}

print("="*80)
print("HIERARCHICAL BLEND VISUALIZATION SUITE")
print("="*80)

# ===========================
# VISUALIZATION 1: MODEL DIVERSITY
# ===========================
def create_diversity_plot(oof_predictions, model_names, y_true):
    """Comprehensive diversity analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    n_models = len(model_names)
    corr_matrix = np.corrcoef([pred.flatten() for pred in oof_predictions])
    colors = SCIENTIFIC_COLORS['model_palette'][:n_models]
    
    # Subplot 1: Mean vs Std scatter
    ax1 = fig.add_subplot(2, 2, 1)
    avg_preds = [pred.mean() for pred in oof_predictions]
    std_preds = [pred.std() for pred in oof_predictions]
    
    scatter = ax1.scatter(avg_preds, std_preds, c=colors[:n_models], s=300, 
                         alpha=0.6, edgecolors='white', linewidth=2)
    
    for i, (x, y, name) in enumerate(zip(avg_preds, std_preds, model_names)):
        ax1.annotate(f'M{i+1}', (x, y), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        ax1.annotate(name[:12], (x, y), xytext=(10, 10), 
                    textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    ax1.set_xlabel('Mean Prediction', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Std Deviation', fontsize=11, fontweight='bold')
    ax1.set_title('Model Statistics (Mean vs Variability)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 2: Correlation heatmap (7 decimals)
    ax2 = fig.add_subplot(2, 2, 2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.7f', 
                cmap='RdYlBu_r', center=0.9, vmin=0.7, vmax=1.0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=[name[:12] for name in model_names],
                yticklabels=[name[:12] for name in model_names],
                ax=ax2, annot_kws={'size': 7})
    ax2.set_title('Correlation Matrix (7 Decimals)', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # Subplot 3: Distribution comparison
    ax3 = fig.add_subplot(2, 2, 3)
    for i, (pred, name) in enumerate(zip(oof_predictions, model_names)):
        kde = gaussian_kde(pred)
        x_range = np.linspace(pred.min(), pred.max(), 200)
        ax3.plot(x_range, kde(x_range), label=name[:15], linewidth=2.5, 
                color=colors[i], alpha=0.7)
    
    kde_true = gaussian_kde(y_true)
    x_range_true = np.linspace(y_true.min(), y_true.max(), 200)
    ax3.plot(x_range_true, kde_true(x_range_true), '--', linewidth=3, 
            label='True Risk', alpha=0.8, color=SCIENTIFIC_COLORS['neutral'])
    
    ax3.set_xlabel('Accident Risk', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title('Prediction Distribution Diversity', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 4: Metrics dashboard
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    mean_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    std_corr = np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    min_corr = np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    max_corr = np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    spread = np.max(avg_preds) - np.min(avg_preds)
    
    metrics_text = f"""
DIVERSITY METRICS
{'='*40}

Correlation Statistics:
  Mean: {mean_corr:.7f}
  Std:  {std_corr:.7f}
  Min:  {min_corr:.7f}
  Max:  {max_corr:.7f}

Prediction Statistics:
  Spread:    {spread:.7f}
  Avg Std:   {np.mean(std_preds):.7f}
  N Models:  {n_models}

Diversity Score: {(1 - mean_corr) * 100:.2f}%

Interpretation:
  Lower correlation = Higher diversity
  Optimal range: 0.80-0.95
    """
    
    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3,
                     edgecolor=SCIENTIFIC_COLORS['model_palette'][0], linewidth=2))
    
    plt.tight_layout()
    plt.savefig('model_diversity_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ===========================
# VISUALIZATION 2: ARCHITECTURE
# ===========================
def create_architecture_diagram(n_models, best_main_weights, best_rank_weights, 
                                best_asc_weight, best_desc_weight, model_names):
    """Hierarchical blend architecture"""
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    colors = SCIENTIFIC_COLORS['model_palette'][:n_models]
    
    # TOP: Architecture flow
    ax_arch = fig.add_subplot(gs[0, :])
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 10)
    ax_arch.axis('off')
    ax_arch.set_title('Hierarchical Blend Architecture', fontsize=14, fontweight='bold')
    
    # Input models
    model_y = 8
    model_spacing = 10 / (n_models + 1)
    for i in range(n_models):
        x = (i + 1) * model_spacing
        box = FancyBboxPatch((x - 0.3, model_y - 0.3), 0.6, 0.6,
                            boxstyle="round,pad=0.05", 
                            edgecolor=colors[i], facecolor=colors[i],
                            alpha=0.6, linewidth=2)
        ax_arch.add_patch(box)
        ax_arch.text(x, model_y, f'M{i+1}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        ax_arch.text(x, model_y + 0.6, f'{best_main_weights[i]:.2f}', 
                    ha='center', va='bottom', fontsize=7, style='italic',
                    color=SCIENTIFIC_COLORS['neutral'])
    
    # Branches
    asc_y = 5.5
    asc_box = FancyBboxPatch((1.5, asc_y - 0.4), 3, 0.8, boxstyle="round,pad=0.1",
                            edgecolor=SCIENTIFIC_COLORS['ascending'], 
                            facecolor=SCIENTIFIC_COLORS['ascending'],
                            alpha=0.3, linewidth=2.5)
    ax_arch.add_patch(asc_box)
    ax_arch.text(3, asc_y, f'Ascending\n{best_asc_weight:.3f}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=SCIENTIFIC_COLORS['ascending'])
    
    desc_box = FancyBboxPatch((5.5, asc_y - 0.4), 3, 0.8, boxstyle="round,pad=0.1",
                             edgecolor=SCIENTIFIC_COLORS['descending'], 
                             facecolor=SCIENTIFIC_COLORS['descending'],
                             alpha=0.3, linewidth=2.5)
    ax_arch.add_patch(desc_box)
    ax_arch.text(7, asc_y, f'Descending\n{best_desc_weight:.3f}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=SCIENTIFIC_COLORS['descending'])
    
    # Arrows
    for i in range(n_models):
        x = (i + 1) * model_spacing
        for target_x in [3, 7]:
            arrow = FancyArrowPatch((x, model_y - 0.4), (target_x, asc_y + 0.5),
                                   arrowstyle='->', mutation_scale=15,
                                   color=colors[i], alpha=0.3, linewidth=1)
            ax_arch.add_patch(arrow)
    
    # Final ensemble
    final_y = 2
    final_box = FancyBboxPatch((4, final_y - 0.4), 2, 0.8, boxstyle="round,pad=0.1",
                              edgecolor=SCIENTIFIC_COLORS['ensemble'], 
                              facecolor=SCIENTIFIC_COLORS['ensemble'],
                              alpha=0.5, linewidth=3)
    ax_arch.add_patch(final_box)
    ax_arch.text(5, final_y, 'Final Ensemble', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    for start_x, color in [(3, SCIENTIFIC_COLORS['ascending']), 
                           (7, SCIENTIFIC_COLORS['descending'])]:
        arrow = FancyArrowPatch((start_x, asc_y - 0.5), 
                               (5, final_y + 0.4),
                               arrowstyle='->', mutation_scale=20,
                               color=color, linewidth=2.5, alpha=0.7)
        ax_arch.add_patch(arrow)
    
    # BOTTOM LEFT: Rank weights
    ax_rank = fig.add_subplot(gs[1, 0])
    positions = np.arange(1, n_models + 1)
    bar_colors = [SCIENTIFIC_COLORS['ascending'] if w > 0 else SCIENTIFIC_COLORS['descending'] 
                  for w in best_rank_weights]
    ax_rank.barh(positions, best_rank_weights, color=bar_colors, alpha=0.7, 
                edgecolor='white', linewidth=1.5)
    ax_rank.axvline(0, color=SCIENTIFIC_COLORS['neutral'], linewidth=2, 
                   linestyle='--', alpha=0.5)
    ax_rank.set_xlabel('Rank Weight', fontsize=10, fontweight='bold')
    ax_rank.set_ylabel('Position', fontsize=10, fontweight='bold')
    ax_rank.set_title('Rank Adjustments', fontsize=11, fontweight='bold')
    ax_rank.set_yticks(positions)
    ax_rank.set_yticklabels([f'P{i}' for i in positions])
    ax_rank.grid(axis='x', alpha=0.3, linestyle='--')
    ax_rank.invert_yaxis()
    
    for pos, weight in zip(positions, best_rank_weights):
        x_pos = weight + (0.01 if weight > 0 else -0.01)
        ax_rank.text(x_pos, pos, f'{weight:+.3f}', ha='left' if weight > 0 else 'right', 
                    va='center', fontsize=8, fontweight='bold')
    
    # BOTTOM CENTER: Model weights pie
    ax_weights = fig.add_subplot(gs[1, 1])
    wedges, texts, autotexts = ax_weights.pie(best_main_weights, 
                                               labels=[f'{name[:12]}' for name in model_names],
                                               autopct='%1.1f%%', startangle=90,
                                               colors=colors, explode=[0.05] * n_models,
                                               textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax_weights.set_title('Model Weights', fontsize=11, fontweight='bold')
    
    # BOTTOM RIGHT: Direction weights
    ax_dir = fig.add_subplot(gs[1, 2])
    directions = ['Ascending', 'Descending']
    dir_weights = [best_asc_weight, best_desc_weight]
    dir_colors = [SCIENTIFIC_COLORS['ascending'], SCIENTIFIC_COLORS['descending']]
    bars = ax_dir.bar(directions, dir_weights, color=dir_colors, 
                      alpha=0.7, edgecolor='white', linewidth=2)
    ax_dir.set_ylabel('Weight', fontsize=10, fontweight='bold')
    ax_dir.set_title('Direction Weights', fontsize=11, fontweight='bold')
    ax_dir.set_ylim(0, 1)
    ax_dir.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, weight in zip(bars, dir_weights):
        ax_dir.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{weight:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hierarchical_blend_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ===========================
# VISUALIZATION 3: COMPARISONS
# ===========================
def create_comparison_plot(oof_predictions, test_predictions, 
                          oof_ensemble, test_ensemble, y_true, model_names):
    """Model comparison analysis"""
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    n_models = len(model_names)
    colors = SCIENTIFIC_COLORS['model_palette'][:n_models]
    
    # Residuals
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (pred, name) in enumerate(zip(oof_predictions, model_names)):
        ax1.scatter(y_true, pred - y_true, alpha=0.3, s=8, color=colors[i], label=name[:12])
    ax1.scatter(y_true, oof_ensemble - y_true, alpha=0.5, s=12,
               color=SCIENTIFIC_COLORS['ensemble'], marker='x', label='Ensemble', linewidths=1.5)
    ax1.axhline(0, color=SCIENTIFIC_COLORS['descending'], linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True Risk', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Residual', fontsize=10, fontweight='bold')
    ax1.set_title('Residual Analysis', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=6, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Prediction scatter
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (pred, name) in enumerate(zip(oof_predictions, model_names)):
        ax2.scatter(y_true, pred, alpha=0.3, s=8, color=colors[i], label=name[:12])
    ax2.scatter(y_true, oof_ensemble, alpha=0.5, s=12,
               color=SCIENTIFIC_COLORS['ensemble'], marker='x', label='Ensemble', linewidths=1.5)
    min_val = min(y_true.min(), oof_ensemble.min())
    max_val = max(y_true.max(), oof_ensemble.max())
    ax2.plot([min_val, max_val], [min_val, max_val], '--', linewidth=2, 
            alpha=0.7, label='Perfect', color=SCIENTIFIC_COLORS['neutral'])
    ax2.set_xlabel('True Risk', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Predicted Risk', fontsize=10, fontweight='bold')
    ax2.set_title('Predictions vs True', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=6, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Error distribution
    ax3 = fig.add_subplot(gs[0, 2])
    errors = [np.abs(pred - y_true) for pred in oof_predictions]
    errors.append(np.abs(oof_ensemble - y_true))
    labels = [name[:12] for name in model_names] + ['Ensemble']
    bp = ax3.boxplot(errors, labels=labels, patch_artist=True, showmeans=True, meanline=True)
    box_colors = list(colors) + [SCIENTIFIC_COLORS['ensemble']]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    ax3.set_ylabel('Absolute Error', fontsize=10, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=11, fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # RMSE comparison
    ax4 = fig.add_subplot(gs[1, 0])
    rmses = [np.sqrt(np.mean((pred - y_true)**2)) for pred in oof_predictions]
    rmses.append(np.sqrt(np.mean((oof_ensemble - y_true)**2)))
    x_pos = np.arange(len(labels))
    bars = ax4.bar(x_pos, rmses, color=box_colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    bars[-1].set_linewidth(3)
    bars[-1].set_edgecolor(SCIENTIFIC_COLORS['descending'])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('RMSE', fontsize=10, fontweight='bold')
    ax4.set_title('RMSE Comparison', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, rmse in zip(bars, rmses):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0001,
                f'{rmse:.5f}', ha='center', va='bottom', fontsize=6)
    
    # Prediction heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    sample_size = min(1000, len(y_true))
    sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
    pred_matrix = np.column_stack([pred[sample_indices] for pred in oof_predictions])
    im = ax5.imshow(pred_matrix.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax5.set_yticks(np.arange(n_models))
    ax5.set_yticklabels([name[:12] for name in model_names], fontsize=7)
    ax5.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
    ax5.set_title('Prediction Heatmap', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Risk')
    
    # Test distributions
    ax6 = fig.add_subplot(gs[1, 2])
    for i, (pred, name) in enumerate(zip(test_predictions, model_names)):
        kde = gaussian_kde(pred)
        x_range = np.linspace(pred.min(), pred.max(), 200)
        ax6.plot(x_range, kde(x_range), label=name[:12], linewidth=2,
                color=colors[i], alpha=0.6)
    kde_ens = gaussian_kde(test_ensemble)
    x_range_ens = np.linspace(test_ensemble.min(), test_ensemble.max(), 200)
    ax6.plot(x_range_ens, kde_ens(x_range_ens), '-', linewidth=3,
            label='Ensemble', alpha=0.8, color=SCIENTIFIC_COLORS['ensemble'])
    ax6.set_xlabel('Predicted Risk', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax6.set_title('Test Distributions', fontsize=11, fontweight='bold')
    ax6.legend(loc='best', fontsize=6, framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('prediction_comparison_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ===========================
# EXECUTE
# ===========================
print("\nGenerating visualizations...")

try:
    create_diversity_plot(oof_predictions, model_names, y_true)
    print("âœ“ Diversity analysis saved")
except Exception as e:
    print(f"âœ— Diversity plot error: {e}")

try:
    create_architecture_diagram(n_models, best_main_weights, best_rank_weights,
                               best_asc_weight, best_desc_weight, model_names)
    print("âœ“ Architecture diagram saved")
except Exception as e:
    print(f"âœ— Architecture plot error: {e}")

try:
    create_comparison_plot(oof_predictions, test_predictions,
                          oof_ensemble, test_ensemble, y_true, model_names)
    print("âœ“ Comparison analysis saved")
except Exception as e:
    print(f"âœ— Comparison plot error: {e}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  â€¢ model_diversity_analysis.png")
print("  â€¢ hierarchical_blend_architecture.png")
print("  â€¢ prediction_comparison_analysis.png")
print("="*80)

