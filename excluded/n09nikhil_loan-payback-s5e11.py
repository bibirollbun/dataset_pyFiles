# Essential libraries for data analysis
import numpy as np 
import pandas as pd 

# Listing files in the input directory to verify data availability
import os
for root_dir, _, file_list in os.walk('/kaggle/input'):
    for file_name in file_list:
        print(os.path.join(root_dir, file_name))


"""
Advanced Ensemble Analysis Tool
Analyzes submission diversity, correlations, and suggests optimal weights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from itertools import combinations
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# ENSEMBLE ANALYZER CLASS
# =====================================================================

class EnsembleAnalyzer:
    """Comprehensive analysis of ensemble submissions."""
    
    def __init__(self, path: str, submission_names: List[str], target_col: str = 'loan_paid_back'):
        """
        Initialize analyzer with submission files.
        
        Args:
            path: Path to submission files
            submission_names: List of submission file names (without .csv)
            target_col: Name of the target column
        """
        self.path = path
        self.submission_names = submission_names
        self.target_col = target_col
        self.submissions = {}
        self.load_submissions()
    
    def load_submissions(self):
        """Load all submission files."""
        print(f"Loading {len(self.submission_names)} submissions...")
        
        for name in self.submission_names:
            filepath = f"{self.path}{name}.csv"
            try:
                df = pd.read_csv(filepath)
                if self.target_col in df.columns:
                    self.submissions[name] = df[self.target_col].values
                else:
                    print(f"Warning: {self.target_col} not found in {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        print(f"Successfully loaded {len(self.submissions)} submissions\n")
    
    # =================================================================
    # CORRELATION ANALYSIS
    # =================================================================
    
    def analyze_correlations(self, method='pearson'):
        """
        Analyze correlations between submissions.
        
        Args:
            method: 'pearson' or 'spearman'
        """
        print("=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        
        n = len(self.submission_names)
        corr_matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(self.submission_names):
            for j, name2 in enumerate(self.submission_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    if method == 'pearson':
                        corr, _ = pearsonr(
                            self.submissions[name1],
                            self.submissions[name2]
                        )
                    else:
                        corr, _ = spearmanr(
                            self.submissions[name1],
                            self.submissions[name2]
                        )
                    corr_matrix[i, j] = corr
        
        # Create DataFrame for display
        corr_df = pd.DataFrame(
            corr_matrix,
            index=self.submission_names,
            columns=self.submission_names
        )
        
        print(f"\n{method.capitalize()} Correlation Matrix:")
        print(corr_df.round(4))
        
        # Find least correlated pairs
        print("\n" + "-" * 70)
        print("LEAST CORRELATED PAIRS (More Diverse = Better for Ensemble):")
        print("-" * 70)
        
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((
                    self.submission_names[i],
                    self.submission_names[j],
                    corr_matrix[i, j]
                ))
        
        pairs.sort(key=lambda x: x[2])
        for name1, name2, corr in pairs[:5]:
            print(f"{name1} <-> {name2}: {corr:.6f}")
        
        # Find most correlated pairs
        print("\n" + "-" * 70)
        print("MOST CORRELATED PAIRS (Less Diverse = Redundant):")
        print("-" * 70)
        
        for name1, name2, corr in pairs[-5:]:
            print(f"{name1} <-> {name2}: {corr:.6f}")
        
        print()
        return corr_df
    
    # =================================================================
    # DIVERSITY ANALYSIS
    # =================================================================
    
    def analyze_diversity(self):
        """Analyze prediction diversity across submissions."""
        print("=" * 70)
        print("DIVERSITY ANALYSIS")
        print("=" * 70)
        
        # Create matrix of all predictions
        pred_matrix = np.column_stack([
            self.submissions[name] for name in self.submission_names
        ])
        
        # Calculate statistics per row
        row_means = pred_matrix.mean(axis=1)
        row_stds = pred_matrix.std(axis=1)
        row_ranges = pred_matrix.max(axis=1) - pred_matrix.min(axis=1)
        
        print(f"\nOverall Statistics:")
        print(f"  Mean prediction range: {row_ranges.mean():.6f}")
        print(f"  Median prediction range: {np.median(row_ranges):.6f}")
        print(f"  Max prediction range: {row_ranges.max():.6f}")
        print(f"  Mean std deviation: {row_stds.mean():.6f}")
        
        # Find examples with high disagreement
        print("\n" + "-" * 70)
        print("SAMPLES WITH HIGHEST DISAGREEMENT (Top 10):")
        print("-" * 70)
        
        high_disagreement_idx = np.argsort(row_ranges)[-10:][::-1]
        
        for idx in high_disagreement_idx:
            predictions = {
                name: self.submissions[name][idx]
                for name in self.submission_names
            }
            print(f"\nSample {idx}:")
            print(f"  Range: {row_ranges[idx]:.6f}")
            for name, pred in predictions.items():
                print(f"  {name}: {pred:.5f}")
        
        # Find examples with low disagreement
        print("\n" + "-" * 70)
        print("SAMPLES WITH LOWEST DISAGREEMENT (Top 10):")
        print("-" * 70)
        
        low_disagreement_idx = np.argsort(row_ranges)[:10]
        
        for idx in low_disagreement_idx:
            predictions = {
                name: self.submissions[name][idx]
                for name in self.submission_names
            }
            print(f"\nSample {idx}:")
            print(f"  Range: {row_ranges[idx]:.6f}")
            for name, pred in predictions.items():
                print(f"  {name}: {pred:.5f}")
        
        print()
        return row_ranges, row_stds
    
    # =================================================================
    # WEIGHT OPTIMIZATION
    # =================================================================
    
    def suggest_optimal_weights(self, n_trials=5):
        """
        Suggest optimal weights using different strategies.
        
        Note: Without ground truth, we optimize for diversity and correlation
        """
        print("=" * 70)
        print("WEIGHT OPTIMIZATION SUGGESTIONS")
        print("=" * 70)
        
        n = len(self.submission_names)
        
        # Strategy 1: Equal weights
        equal_weights = [1.0/n] * n
        print(f"\nStrategy 1 - Equal Weights:")
        for name, weight in zip(self.submission_names, equal_weights):
            print(f"  {name}: {weight:.4f}")
        
        # Strategy 2: Inverse correlation weighting
        # Less correlated submissions get higher weights
        corr_matrix = self.analyze_correlations(method='pearson')
        avg_correlations = []
        for name in self.submission_names:
            others = [n for n in self.submission_names if n != name]
            avg_corr = corr_matrix.loc[name, others].mean()
            avg_correlations.append(avg_corr)
        
        # Inverse weights (lower correlation = higher weight)
        inv_weights = [1.0 - c for c in avg_correlations]
        inv_weights = [w / sum(inv_weights) for w in inv_weights]
        
        print(f"\nStrategy 2 - Inverse Correlation Weights:")
        for name, weight, corr in zip(self.submission_names, inv_weights, avg_correlations):
            print(f"  {name}: {weight:.4f} (avg corr: {corr:.4f})")
        
        # Strategy 3: Rank-based on reported LB scores (if available)
        print(f"\nStrategy 3 - Rank-Based Weights:")
        print("  (Assign higher weights to better performing models)")
        print("  You need to provide LB scores for this strategy")
        
        # Strategy 4: Diversity-weighted
        pred_matrix = np.column_stack([
            self.submissions[name] for name in self.submission_names
        ])
        
        # Calculate how "unique" each submission is
        uniqueness_scores = []
        for i, name in enumerate(self.submission_names):
            others_idx = [j for j in range(n) if j != i]
            differences = np.abs(pred_matrix[:, i:i+1] - pred_matrix[:, others_idx])
            uniqueness = differences.mean()
            uniqueness_scores.append(uniqueness)
        
        diversity_weights = [u / sum(uniqueness_scores) for u in uniqueness_scores]
        
        print(f"\nStrategy 4 - Diversity-Based Weights:")
        for name, weight, unique in zip(self.submission_names, diversity_weights, uniqueness_scores):
            print(f"  {name}: {weight:.4f} (uniqueness: {unique:.6f})")
        
        print()
        
        return {
            'equal': equal_weights,
            'inverse_correlation': inv_weights,
            'diversity': diversity_weights
        }
    
    # =================================================================
    # PAIRWISE COMBINATION ANALYSIS
    # =================================================================
    
    def analyze_pairwise_combinations(self):
        """Analyze all pairwise combinations."""
        print("=" * 70)
        print("PAIRWISE COMBINATION ANALYSIS")
        print("=" * 70)
        
        n = len(self.submission_names)
        
        print(f"\nAnalyzing {n*(n-1)//2} pairwise combinations...\n")
        
        results = []
        
        for name1, name2 in combinations(self.submission_names, 2):
            # Simple average ensemble
            ensemble = (self.submissions[name1] + self.submissions[name2]) / 2
            
            # Calculate diversity metrics
            diff = np.abs(self.submissions[name1] - self.submissions[name2])
            avg_diff = diff.mean()
            max_diff = diff.max()
            
            # Correlation
            corr, _ = pearsonr(self.submissions[name1], self.submissions[name2])
            
            results.append({
                'pair': f"{name1} + {name2}",
                'avg_diff': avg_diff,
                'max_diff': max_diff,
                'correlation': corr,
                'diversity_score': avg_diff * (1 - corr)  # Combined metric
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('diversity_score', ascending=False)
        
        print("Top 10 Most Diverse Pairs (Best for Ensemble):")
        print(results_df.head(10).to_string(index=False))
        
        print("\n" + "-" * 70)
        print("Top 10 Most Similar Pairs (Redundant):")
        print(results_df.tail(10).to_string(index=False))
        
        print()
        return results_df
    
    # =================================================================
    # PREDICTION DISTRIBUTION ANALYSIS
    # =================================================================
    
    def analyze_distributions(self):
        """Analyze prediction distributions."""
        print("=" * 70)
        print("PREDICTION DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        for name in self.submission_names:
            preds = self.submissions[name]
            print(f"\n{name}:")
            print(f"  Mean: {preds.mean():.6f}")
            print(f"  Median: {np.median(preds):.6f}")
            print(f"  Std: {preds.std():.6f}")
            print(f"  Min: {preds.min():.6f}")
            print(f"  Max: {preds.max():.6f}")
            print(f"  Q25: {np.percentile(preds, 25):.6f}")
            print(f"  Q75: {np.percentile(preds, 75):.6f}")
            print(f"  Predictions < 0.5: {(preds < 0.5).sum()} ({100*(preds < 0.5).mean():.2f}%)")
            print(f"  Predictions > 0.5: {(preds > 0.5).sum()} ({100*(preds > 0.5).mean():.2f}%)")
        
        print()
    
    # =================================================================
    # COMPREHENSIVE REPORT
    # =================================================================
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE ENSEMBLE ANALYSIS REPORT")
        print("=" * 70 + "\n")
        
        # 1. Correlation Analysis
        corr_df = self.analyze_correlations(method='pearson')
        
        # 2. Diversity Analysis
        ranges, stds = self.analyze_diversity()
        
        # 3. Distribution Analysis
        self.analyze_distributions()
        
        # 4. Pairwise Combinations
        pairwise_df = self.analyze_pairwise_combinations()
        
        # 5. Weight Suggestions
        weights = self.suggest_optimal_weights()
        
        # Final Recommendations
        print("=" * 70)
        print("FINAL RECOMMENDATIONS")
        print("=" * 70)
        
        avg_corr = corr_df.values[np.triu_indices_from(corr_df.values, k=1)].mean()
        
        print(f"\n1. Average Correlation: {avg_corr:.4f}")
        if avg_corr > 0.99:
            print("   âš ï¸�  Very high correlation - submissions are too similar!")
            print("   â†’ Try different model architectures or feature engineering")
        elif avg_corr > 0.95:
            print("   âš ï¸�  High correlation - limited diversity benefit")
            print("   â†’ Consider adding more diverse models")
        else:
            print("   âœ“ Good diversity for ensemble")
        
        print(f"\n2. Prediction Diversity: {ranges.mean():.6f}")
        if ranges.mean() < 0.01:
            print("   âš ï¸�  Low diversity - models agree too much")
        else:
            print("   âœ“ Good diversity across predictions")
        
        print("\n3. Recommended Weight Strategies:")
        print("   â†’ Start with EQUAL weights as baseline")
        print("   â†’ Try INVERSE CORRELATION weights for diversity")
        print("   â†’ Use DIVERSITY weights if models have different strengths")
        print("   â†’ Optimize based on local CV if available")
        
        print("\n4. Best Pairwise Combinations:")
        top_pairs = pairwise_df.head(3)
        for _, row in top_pairs.iterrows():
            print(f"   â†’ {row['pair']}: diversity score = {row['diversity_score']:.6f}")
        
        print("\n" + "=" * 70)
        print()


# =====================================================================
# USAGE FUNCTION
# =====================================================================

def analyze_ensemble(path: str, submission_names: List[str]):
    """
    Main function to analyze ensemble.
    
    Args:
        path: Path to submission files (with trailing / or _)
        submission_names: List of submission names without .csv
    
    Example:
        analyze_ensemble(
            "/kaggle/input/22-november-2025-ps-s5e11/submission_",
            ['015', '017', '018', '019']
        )
    """
    analyzer = EnsembleAnalyzer(path, submission_names)
    analyzer.generate_comprehensive_report()
    
    return analyzer


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    # Configure your submissions
    PATH = "/kaggle/input/22-november-2025-ps-s5e11/submission_"
    SUBMISSIONS = ['015', '017', '018', '019']
    
    # Run comprehensive analysis
    analyzer = analyze_ensemble(PATH, SUBMISSIONS)
  


"""
Complete Ensemble Workflow 
All functions included
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION DATA CLASSES
# =====================================================================

@dataclass
class SubmissionConfig:
    """Configuration for a single submission file."""
    name: str
    weight: float
    color: str
    LB: Optional[float] = None

@dataclass
class EnsembleParams:
    """Parameters for ensemble blending."""
    path: str
    id_target: List[str]
    type_sort: List[Any]
    subwts: List[float]
    subm: List[SubmissionConfig]
    
    @property
    def id_column(self) -> str:
        return self.id_target[0]
    
    @property
    def target_column(self) -> str:
        return self.id_target[1]
    
    @property
    def sort_type(self) -> str:
        return self.type_sort[0]
    
    @property
    def asc_weight(self) -> float:
        return self.type_sort[1]
    
    @property
    def desc_weight(self) -> float:
        return self.type_sort[2]

# =====================================================================
# CORE ENSEMBLE FUNCTIONS
# =====================================================================

class SubmissionLoader:
    """Handles loading and merging of submission files."""
    
    def __init__(self, params: EnsembleParams):
        self.params = params
    
    def load_single_submission(self, submission: SubmissionConfig) -> pd.DataFrame:
        """Load a single submission file."""
        filepath = f"{self.params.path}{submission.name}.csv"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Rename target column to submission name
        rename_dict = {
            'target': submission.name,
            'pred': submission.name,
            self.params.target_column: submission.name
        }
        df = df.rename(columns=rename_dict)
        
        return df
    
    def load_all_submissions(self) -> List[pd.DataFrame]:
        """Load all submission files."""
        return [self.load_single_submission(sub) for sub in self.params.subm]
    
    def merge_submissions(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all submission dataframes on ID column."""
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on=self.params.id_column),
            dfs
        )
        return merged_df


class EnsembleBlender:
    """Performs ensemble blending calculations."""
    
    def __init__(self, params: EnsembleParams):
        self.params = params
        self.submission_names = [sub.name for sub in params.subm]
        self.main_weights = [sub.weight for sub in params.subm]
    
    def calculate_sorted_order(self, row: pd.Series, reverse: bool) -> List[str]:
        """Calculate sorted order of submissions for a row."""
        values = {name: row[name] for name in self.submission_names}
        sorted_subs = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
        return [name for name, _ in sorted_subs]
    
    def calculate_weighted_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted ensemble predictions."""
        # Determine sorting function
        if self.params.sort_type == 'asc/desc':
            df['alls'] = df.apply(
                lambda x: self.calculate_sorted_order(x, reverse=True), 
                axis=1
            )
        else:
            df['alls'] = df.apply(
                lambda x: self.submission_names.copy(), 
                axis=1
            )
        
        # Weighted sum calculation
        def compute_ensemble(row):
            indices = [row['alls'].index(name) for name in self.submission_names]
            weights = [
                self.main_weights[j] + self.params.subwts[indices[j]] 
                for j in range(len(self.submission_names))
            ]
            values = [row[name] for name in self.submission_names]
            return sum(w * v for w, v in zip(weights, values))
        
        df['ensemble'] = df.apply(compute_ensemble, axis=1)
        
        return df


# =====================================================================
# MAIN ENSEMBLE FUNCTION (SIMPLIFIED)
# =====================================================================

def h_blend(
    params: Dict[str, Any],
    output_file: str = '',
    show_details: bool = False
) -> pd.DataFrame:
    """
    Simplified ensemble blending function.
    
    Args:
        params: Dictionary with ensemble configuration
        output_file: Path to save output CSV
        show_details: Show detailed output
    
    Returns:
        DataFrame with ensemble predictions
    """
    
    # Convert dict to EnsembleParams if needed
    if isinstance(params, dict):
        submissions = [
            SubmissionConfig(**sub) if isinstance(sub, dict) else sub
            for sub in params['subm']
        ]
        params = EnsembleParams(
            path=params['path'],
            id_target=params['id_target'],
            type_sort=params['type_sort'],
            subwts=params['subwts'],
            subm=submissions
        )
    
    print(f"Loading {len(params.subm)} submissions...")
    
    # Load and merge submissions
    loader = SubmissionLoader(params)
    dfs = loader.load_all_submissions()
    merged_df = loader.merge_submissions(dfs)
    
    print("Calculating ensemble predictions...")
    
    # Calculate ensemble for descending sort
    blender_desc = EnsembleBlender(params)
    df_desc = blender_desc.calculate_weighted_ensemble(merged_df.copy())
    
    # Calculate ensemble for ascending sort
    blender_asc = EnsembleBlender(params)
    df_asc = blender_asc.calculate_weighted_ensemble(merged_df.copy())
    
    # Combine with weights
    final_ensemble = (
        params.desc_weight * df_desc['ensemble'] +
        params.asc_weight * df_asc['ensemble']
    )
    
    # Create final dataframe
    result_df = pd.DataFrame({
        params.id_column: merged_df[params.id_column],
        params.target_column: final_ensemble
    })
    
    # Show details if requested
    if show_details:
        print("\nWeights used:")
        for sub in params.subm:
            print(f"  {sub.name}: {sub.weight:.4f}")
        print(f"\nSub-weights: {params.subwts}")
        print(f"Sort weights: asc={params.asc_weight:.2f}, desc={params.desc_weight:.2f}")
        
        print("\nSample predictions:")
        display_df = result_df.head(10)
        display(display_df)
    
    # Save output
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"\nâœ“ Saved to: {output_file}")
    
    print(f"Ensemble complete! Shape: {result_df.shape}")
    
    return result_df


# =====================================================================
# OPTIMIZED CONFIGURATIONS
# =====================================================================

# Configuration 1: Most Diverse Pair Only (015 + 017)
config_1_diverse_pair = {
    'path': "/kaggle/input/22-november-2025-ps-s5e11/submission_",
    'id_target': ['id', "loan_paid_back"],
    'type_sort': ['asc/desc', 0.30, 0.70],
    'subwts': [0.0, 0.0],
    'subm': [
        {'name': '015', 'weight': 0.50, 'color': 'navy'},
        {'name': '017', 'weight': 0.50, 'color': 'royalblue'},
    ]
}

# Configuration 2: Inverse Correlation Weights
config_2_inverse_corr = {
    'path': "/kaggle/input/22-november-2025-ps-s5e11/submission_",
    'id_target': ['id', "loan_paid_back"],
    'type_sort': ['asc/desc', 0.30, 0.70],
    'subwts': [-0.03, +0.03, +0.02, -0.02],
    'subm': [
        {'name': '015', 'weight': 0.36, 'color': 'navy'},
        {'name': '017', 'weight': 0.32, 'color': 'royalblue'},
        {'name': '018', 'weight': 0.18, 'color': 'deepskyblue'},
        {'name': '019', 'weight': 0.14, 'color': 'dodgerblue'},
    ]
}

# Configuration 3: LB Score Weighted (Trust best models)
config_3_lb_weighted = {
    'path': "/kaggle/input/22-november-2025-ps-s5e11/submission_",
    'id_target': ['id', "loan_paid_back"],
    'type_sort': ['asc/desc', 0.30, 0.70],
    'subwts': [-0.02, +0.02, +0.02, -0.02],
    'subm': [
        {'name': '015', 'weight': 0.23, 'color': 'navy'},
        {'name': '017', 'weight': 0.17, 'color': 'royalblue'},
        {'name': '018', 'weight': 0.28, 'color': 'deepskyblue'},
        {'name': '019', 'weight': 0.32, 'color': 'dodgerblue'},  # Best LB
    ]
}

# Configuration 4: Aggressive Sub-Weights
config_4_aggressive = {
    'path': "/kaggle/input/22-november-2025-ps-s5e11/submission_",
    'id_target': ['id', "loan_paid_back"],
    'type_sort': ['asc/desc', 0.20, 0.80],
    'subwts': [-0.08, +0.08, +0.05, -0.05],
    'subm': [
        {'name': '015', 'weight': 0.36, 'color': 'navy'},
        {'name': '017', 'weight': 0.32, 'color': 'royalblue'},
        {'name': '018', 'weight': 0.18, 'color': 'deepskyblue'},
        {'name': '019', 'weight': 0.14, 'color': 'dodgerblue'},
    ]
}

# Configuration 5: Hybrid Approach
config_5_hybrid = {
    'path': "/kaggle/input/22-november-2025-ps-s5e11/submission_",
    'id_target': ['id', "loan_paid_back"],
    'type_sort': ['asc/desc', 0.25, 0.75],
    'subwts': [-0.05, +0.05, +0.03, -0.03],
    'subm': [
        {'name': '015', 'weight': 0.30, 'color': 'navy'},
        {'name': '017', 'weight': 0.28, 'color': 'royalblue'},
        {'name': '018', 'weight': 0.24, 'color': 'deepskyblue'},
        {'name': '019', 'weight': 0.18, 'color': 'dodgerblue'},
    ]
}


# =====================================================================
# TEST MULTIPLE CONFIGURATIONS
# =====================================================================

def test_multiple_configs():
    """Test multiple configurations and create submission files."""
    
    configs = {
        'submission_diverse_pair.csv': config_1_diverse_pair,
        'submission_inverse_corr.csv': config_2_inverse_corr,
        'submission_lb_weighted.csv': config_3_lb_weighted,
        'submission_aggressive.csv': config_4_aggressive,
        'submission_hybrid.csv': config_5_hybrid,
    }
    
    print("="*70)
    print("TESTING MULTIPLE ENSEMBLE CONFIGURATIONS")
    print("="*70)
    
    results = []
    
    for filename, config in configs.items():
        print(f"\n{'â”€'*70}")
        print(f"Config: {filename}")
        print(f"Models: {len(config['subm'])}")
        print(f"{'â”€'*70}")
        
        try:
            df = h_blend(config, output_file=filename, show_details=False)
            results.append({
                'file': filename,
                'status': 'âœ“ Success',
                'shape': df.shape
            })
        except Exception as e:
            print(f"âœ— Error: {str(e)}")
            results.append({
                'file': filename,
                'status': f'âœ— Error: {str(e)}',
                'shape': None
            })
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for r in results:
        print(f"{r['status']:15} {r['file']:35} {r['shape']}")
    
    print("\nğŸ“� Submit all CSV files to Kaggle and compare scores!")
    
    return results


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    print("""
    â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
    â•‘           SIMPLIFIED ENSEMBLE BLENDING - READY TO RUN    â•‘
    â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    This notebook will create 5 different ensemble submissions:
    
    1. submission_diverse_pair.csv    - Only 2 most diverse models
    2. submission_inverse_corr.csv    - Inverse correlation weights
    3. submission_lb_weighted.csv     - Trust best LB scores
    4. submission_aggressive.csv      - Large sub-weights
    5. submission_hybrid.csv          - Balanced approach
    
   
    """)
    
    # Run all configurations
    results = test_multiple_configs()
    
    print("\nâœ“ All done! ")


# =====================================================================
# ALTERNATIVE: Run just one config
# =====================================================================

# Uncomment to run just one configuration:
# df = h_blend(config_2_inverse_corr, output_file='submission.csv', show_details=True)




