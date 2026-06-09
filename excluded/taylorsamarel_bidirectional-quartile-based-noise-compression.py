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


# Advanced Bidirectional Intelligent Compression with Quartile Strategies
# This version can expand or compress different regions of the distribution differently

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kurtosis, skew
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ADVANCED BIDIRECTIONAL COMPRESSION WITH QUARTILE STRATEGIES")
print("="*80)
print("\nThis enhanced system features:")
print("1. Different compression algorithms per quartile")
print("2. Adaptive gradient curves that can expand or compress")
print("3. Train-CV gap minimization for noise verification")
print("4. Flexible n-tile segmentation (not just quartiles)")
print("="*80)

class BidirectionalQuartileCompressor:
    """
    Advanced compressor that can apply different transformations to different
    regions of the distribution, including expansion of compressed regions
    """
    
    def __init__(self, n_segments=4, optimization_metric='cv_gap'):
        self.n_segments = n_segments
        self.optimization_metric = optimization_metric
        self.segment_profiles = {}
        self.transformation_curves = {}
        self.feature_cv_gaps = {}
        
    def segment_data(self, data):
        """Segment data into n quantile-based segments"""
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        # Calculate segment boundaries
        percentiles = np.linspace(0, 100, self.n_segments + 1)
        boundaries = np.percentile(clean_data, percentiles)
        
        # Ensure unique boundaries
        boundaries = np.unique(boundaries)
        if len(boundaries) < 2:
            boundaries = np.array([clean_data.min(), clean_data.max()])
        
        return boundaries
    
    def create_gradient_curve(self, x_points, y_points, method='spline'):
        """Create smooth transformation curve from control points"""
        if method == 'spline':
            # Cubic spline for smooth curves
            spline = UnivariateSpline(x_points, y_points, s=0.1, k=3)
            return spline
        elif method == 'pchip':
            # Monotonic cubic interpolation
            pchip = PchipInterpolator(x_points, y_points)
            return pchip
        elif method == 'isotonic':
            # Isotonic regression for monotonic transformation
            iso = IsotonicRegression()
            iso.fit(x_points, y_points)
            return iso
        else:
            # Linear interpolation as fallback
            return lambda x: np.interp(x, x_points, y_points)
    
    def generate_random_gradient(self, segment_range, expansion_allowed=True):
        """Generate random gradient parameters for a segment"""
        min_val, max_val = segment_range
        
        # Random transformation parameters
        if expansion_allowed:
            # Allow both compression and expansion
            scale_factor = np.random.uniform(0.5, 2.0)  # 0.5 = compress by half, 2.0 = expand by 2x
            shift_factor = np.random.uniform(-0.2, 0.2) * (max_val - min_val)
        else:
            # Only compression
            scale_factor = np.random.uniform(0.5, 1.0)
            shift_factor = 0
        
        # Random curve shape
        curve_type = np.random.choice(['linear', 'sigmoid', 'power', 'log'])
        curve_strength = np.random.uniform(0.1, 2.0)
        
        return {
            'scale': scale_factor,
            'shift': shift_factor,
            'curve_type': curve_type,
            'strength': curve_strength
        }
    
    def apply_segment_transformation(self, data, segment_params, segment_range):
        """Apply transformation to data within a segment"""
        min_val, max_val = segment_range
        mask = (data >= min_val) & (data <= max_val)
        
        if not np.any(mask):
            return data
        
        segment_data = data[mask]
        
        # Normalize to [0, 1] within segment
        if max_val > min_val:
            normalized = (segment_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(segment_data)
        
        # Apply curve transformation
        curve_type = segment_params['curve_type']
        strength = segment_params['strength']
        
        if curve_type == 'linear':
            transformed = normalized
        elif curve_type == 'sigmoid':
            # Sigmoid transformation
            transformed = 1 / (1 + np.exp(-strength * (normalized - 0.5) * 4))
        elif curve_type == 'power':
            # Power transformation
            transformed = np.power(normalized, strength)
        elif curve_type == 'log':
            # Log transformation
            transformed = np.log1p(normalized * strength) / np.log1p(strength)
        
        # Apply scale and shift
        transformed = transformed * segment_params['scale'] + segment_params['shift']
        
        # Denormalize back to original scale
        result = data.copy()
        result[mask] = transformed * (max_val - min_val) + min_val
        
        return result
    
    def evaluate_transformation(self, X, y, feature_idx, transformation_params):
        """Evaluate transformation quality using train-CV gap"""
        # Apply transformation
        X_transformed = X.copy()
        feature_data = X.iloc[:, feature_idx].values
        
        # Apply transformations for each segment
        for seg_idx, (seg_range, seg_params) in enumerate(transformation_params):
            feature_data = self.apply_segment_transformation(
                feature_data, seg_params, seg_range
            )
        
        X_transformed.iloc[:, feature_idx] = feature_data
        
        # Evaluate using simple model
        model = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        
        # Calculate train score
        model.fit(X_transformed, y)
        train_pred = model.predict(X_transformed)
        train_score = pearsonr(y, train_pred)[0]
        
        # Calculate CV score
        cv_scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_transformed):
            model_cv = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            model_cv.fit(X_transformed.iloc[train_idx], y.iloc[train_idx])
            val_pred = model_cv.predict(X_transformed.iloc[val_idx])
            cv_scores.append(pearsonr(y.iloc[val_idx], val_pred)[0])
        
        cv_score = np.mean(cv_scores)
        
        # Calculate gap (lower is better)
        gap = train_score - cv_score
        
        return {
            'train_score': train_score,
            'cv_score': cv_score,
            'gap': gap,
            'gap_reduction': -gap  # Negative because we minimize
        }
    
    def optimize_feature_transformation(self, X, y, feature_idx, feature_name, 
                                      n_iterations=20, visualize=True):
        """Optimize transformation for a single feature"""
        print(f"\n{'='*60}")
        print(f"ğŸ”§ OPTIMIZING TRANSFORMATION FOR: {feature_name}")
        print(f"{'='*60}")
        
        feature_data = X.iloc[:, feature_idx].values
        
        # Segment the data
        boundaries = self.segment_data(feature_data)
        segments = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        
        print(f"\nğŸ“Š Data segmented into {len(segments)} regions:")
        for i, (min_val, max_val) in enumerate(segments):
            count = np.sum((feature_data >= min_val) & (feature_data <= max_val))
            print(f"   Segment {i+1}: [{min_val:.3f}, {max_val:.3f}] - {count} samples")
        
        # Initialize with baseline (no transformation)
        baseline_params = [(seg, {'scale': 1.0, 'shift': 0, 'curve_type': 'linear', 'strength': 1}) 
                          for seg in segments]
        baseline_eval = self.evaluate_transformation(X, y, feature_idx, baseline_params)
        
        print(f"\nğŸ“ˆ Baseline performance:")
        print(f"   Train score: {baseline_eval['train_score']:.4f}")
        print(f"   CV score: {baseline_eval['cv_score']:.4f}")
        print(f"   Train-CV gap: {baseline_eval['gap']:.4f}")
        
        # Optimization loop
        best_params = baseline_params
        best_eval = baseline_eval
        history = [baseline_eval]
        
        print(f"\nğŸ”„ Running {n_iterations} optimization iterations...")
        
        for iteration in range(n_iterations):
            # Generate random variations
            candidates = []
            
            for _ in range(10):  # Test 10 random variations per iteration
                # Randomly modify segments
                new_params = []
                for seg_range, seg_params in best_params:
                    if np.random.random() < 0.7:  # 70% chance to modify segment
                        new_seg_params = self.generate_random_gradient(seg_range)
                    else:
                        new_seg_params = seg_params.copy()
                    new_params.append((seg_range, new_seg_params))
                
                # Evaluate
                eval_result = self.evaluate_transformation(X, y, feature_idx, new_params)
                candidates.append((new_params, eval_result))
            
            # Select best candidate
            if self.optimization_metric == 'cv_gap':
                best_candidate = min(candidates, key=lambda x: x[1]['gap'])
            else:  # cv_score
                best_candidate = max(candidates, key=lambda x: x[1]['cv_score'])
            
            # Update if improved
            if (self.optimization_metric == 'cv_gap' and best_candidate[1]['gap'] < best_eval['gap']) or \
               (self.optimization_metric == 'cv_score' and best_candidate[1]['cv_score'] > best_eval['cv_score']):
                best_params = best_candidate[0]
                best_eval = best_candidate[1]
                history.append(best_eval)
                
                if (iteration + 1) % 5 == 0:
                    print(f"   Iteration {iteration+1}: Gap={best_eval['gap']:.4f}, "
                          f"CV={best_eval['cv_score']:.4f}")
        
        print(f"\nâœ… Optimization complete!")
        print(f"   Final train score: {best_eval['train_score']:.4f}")
        print(f"   Final CV score: {best_eval['cv_score']:.4f}")
        print(f"   Final gap: {best_eval['gap']:.4f}")
        print(f"   Gap reduction: {baseline_eval['gap'] - best_eval['gap']:.4f}")
        
        # Store results
        self.segment_profiles[feature_name] = {
            'segments': segments,
            'parameters': best_params,
            'baseline_eval': baseline_eval,
            'optimized_eval': best_eval,
            'history': history
        }
        
        # Visualize if requested
        if visualize:
            self._visualize_transformation(feature_data, feature_name, best_params)
        
        return best_params
    
    def _visualize_transformation(self, original_data, feature_name, transformation_params):
        """Visualize the transformation curve and effects"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Bidirectional Transformation for {feature_name}', fontsize=16)
        
        # Remove NaN
        clean_data = original_data[~np.isnan(original_data)]
        
        # 1. Transformation curve
        ax = axes[0, 0]
        
        # Create smooth curve for visualization
        x_range = np.linspace(clean_data.min(), clean_data.max(), 1000)
        y_transformed = x_range.copy()
        
        for seg_range, seg_params in transformation_params:
            y_transformed = self.apply_segment_transformation(
                y_transformed, seg_params, seg_range
            )
        
        ax.plot(x_range, y_transformed, 'b-', linewidth=2, label='Transformation')
        ax.plot(x_range, x_range, 'r--', alpha=0.5, label='Identity (no change)')
        
        # Mark segment boundaries
        for seg_range, _ in transformation_params:
            ax.axvline(seg_range[0], color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Original Value')
        ax.set_ylabel('Transformed Value')
        ax.set_title('Transformation Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Segment details
        ax = axes[0, 1]
        segment_info = []
        
        for i, (seg_range, seg_params) in enumerate(transformation_params):
            segment_info.append({
                'Segment': f'S{i+1}',
                'Range': f'[{seg_range[0]:.2f}, {seg_range[1]:.2f}]',
                'Scale': seg_params['scale'],
                'Type': seg_params['curve_type'],
                'Action': 'Expand' if seg_params['scale'] > 1 else 'Compress'
            })
        
        segment_df = pd.DataFrame(segment_info)
        
        # Create text visualization
        text = "Segment Transformations:\n\n"
        for _, row in segment_df.iterrows():
            text += f"{row['Segment']}: {row['Range']}\n"
            text += f"  â€¢ Action: {row['Action']} ({row['Scale']:.2f}x)\n"
            text += f"  â€¢ Curve: {row['Type']}\n\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        # 3. Before/After distribution
        ax = axes[1, 0]
        
        # Apply transformation
        transformed_data = clean_data.copy()
        for seg_range, seg_params in transformation_params:
            transformed_data = self.apply_segment_transformation(
                transformed_data, seg_params, seg_range
            )
        
        ax.hist(clean_data, bins=50, alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(transformed_data, bins=50, alpha=0.5, label='Transformed', density=True, color='red')
        
        # Mark quartiles
        for p in [25, 50, 75]:
            orig_val = np.percentile(clean_data, p)
            trans_val = np.percentile(transformed_data, p)
            ax.axvline(orig_val, color='blue', linestyle=':', alpha=0.5)
            ax.axvline(trans_val, color='red', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        
        # 4. Optimization history
        ax = axes[1, 1]
        
        if feature_name in self.segment_profiles:
            history = self.segment_profiles[feature_name]['history']
            
            iterations = range(len(history))
            gaps = [h['gap'] for h in history]
            cv_scores = [h['cv_score'] for h in history]
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(iterations, gaps, 'b-', marker='o', label='Train-CV Gap')
            line2 = ax2.plot(iterations, cv_scores, 'r-', marker='s', label='CV Score')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Train-CV Gap', color='b')
            ax2.set_ylabel('CV Score', color='r')
            ax.set_title('Optimization Progress')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.savefig(f'bidirectional_transformation_{feature_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def fit(self, X, y, feature_subset=None, n_iterations=20):
        """Fit the compressor by optimizing each feature"""
        print("\n" + "="*80)
        print("ğŸš€ FITTING BIDIRECTIONAL QUARTILE COMPRESSOR")
        print("="*80)
        
        if feature_subset is None:
            feature_subset = list(range(X.shape[1]))
        
        for idx in feature_subset[:5]:  # Limit to first 5 for demonstration
            feature_name = X.columns[idx] if hasattr(X, 'columns') else f'Feature_{idx}'
            self.optimize_feature_transformation(X, y, idx, feature_name, n_iterations)
        
        return self
    
    def transform(self, X):
        """Apply learned transformations"""
        X_transformed = X.copy()
        
        for feature_name, profile in self.segment_profiles.items():
            if hasattr(X, 'columns') and feature_name in X.columns:
                idx = list(X.columns).index(feature_name)
            else:
                continue
            
            feature_data = X_transformed.iloc[:, idx].values
            
            for seg_range, seg_params in profile['parameters']:
                feature_data = self.apply_segment_transformation(
                    feature_data, seg_params, seg_range
                )
            
            X_transformed.iloc[:, idx] = feature_data
        
        return X_transformed

# Advanced evaluation with multiple strategies
def compare_compression_strategies_advanced(X_train, y_train):
    """Compare different advanced compression strategies"""
    print("\n" + "="*80)
    print("ğŸ”¬ COMPARING ADVANCED COMPRESSION STRATEGIES")
    print("="*80)
    
    strategies = {
        'Baseline (No Compression)': {
            'type': 'none'
        },
        'Traditional Uniform': {
            'type': 'uniform',
            'strength': 0.3
        },
        'Quartile-Based (4 segments)': {
            'type': 'bidirectional',
            'n_segments': 4,
            'metric': 'cv_gap'
        },
        'Decile-Based (10 segments)': {
            'type': 'bidirectional', 
            'n_segments': 10,
            'metric': 'cv_gap'
        },
        'CV-Score Optimized': {
            'type': 'bidirectional',
            'n_segments': 5,
            'metric': 'cv_score'
        }
    }
    
    results = {}
    
    for name, config in strategies.items():
        print(f"\n\n{'='*60}")
        print(f"Testing strategy: {name}")
        print(f"{'='*60}")
        
        if config['type'] == 'none':
            X_transformed = X_train
            
        elif config['type'] == 'uniform':
            # Simple uniform compression
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns
            )
            X_transformed = X_scaled * (1 - config['strength']) + X_train * config['strength']
            
        elif config['type'] == 'bidirectional':
            compressor = BidirectionalQuartileCompressor(
                n_segments=config['n_segments'],
                optimization_metric=config['metric']
            )
            # Fit on subset of features for speed
            compressor.fit(X_train, y_train, feature_subset=list(range(10)), n_iterations=10)
            X_transformed = compressor.transform(X_train)
        
        # Evaluate
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        # Training score
        model.fit(X_transformed, y_train)
        train_pred = model.predict(X_transformed)
        train_score = pearsonr(y_train, train_pred)[0]
        
        # CV score
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_transformed):
            model_cv = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model_cv.fit(X_transformed.iloc[train_idx], y_train.iloc[train_idx])
            val_pred = model_cv.predict(X_transformed.iloc[val_idx])
            cv_scores.append(pearsonr(y_train.iloc[val_idx], val_pred)[0])
        
        cv_score = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        gap = train_score - cv_score
        
        results[name] = {
            'train_score': train_score,
            'cv_score': cv_score,
            'cv_std': cv_std,
            'gap': gap
        }
        
        print(f"\nğŸ“Š Results:")
        print(f"   Train Score: {train_score:.4f}")
        print(f"   CV Score: {cv_score:.4f} Â± {cv_std:.4f}")
        print(f"   Train-CV Gap: {gap:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Scores comparison
    ax = axes[0]
    strategies_list = list(results.keys())
    train_scores = [results[s]['train_score'] for s in strategies_list]
    cv_scores = [results[s]['cv_score'] for s in strategies_list]
    cv_stds = [results[s]['cv_std'] for s in strategies_list]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Train Score', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_scores, width, yerr=cv_stds, 
                    label='CV Score', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_list, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gap analysis
    ax = axes[1]
    gaps = [results[s]['gap'] for s in strategies_list]
    colors = ['red' if g > 0.01 else 'green' for g in gaps]
    
    bars = ax.bar(strategies_list, gaps, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Train-CV Gap')
    ax.set_title('Overfitting Analysis (Lower is Better)')
    ax.set_xticklabels(strategies_list, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{gap:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('advanced_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("ğŸ“Š STRATEGY COMPARISON SUMMARY")
    print("="*80)
    
    summary_df = pd.DataFrame(results).T
    summary_df['gap_reduction'] = summary_df['gap'].min() - summary_df['gap']
    
    print(summary_df.round(4))
    
    best_strategy = summary_df['cv_score'].idxmax()
    print(f"\nğŸ�† Best CV Score: {best_strategy}")
    
    lowest_gap = summary_df['gap'].idxmin()
    print(f"ğŸ�¯ Lowest Gap (least overfitting): {lowest_gap}")
    
    return results

# Demonstration of segment-specific transformations
def demonstrate_segment_transformations():
    """Show how different segments can have different transformations"""
    print("\n" + "="*80)
    print("ğŸ“� DEMONSTRATING SEGMENT-SPECIFIC TRANSFORMATIONS")
    print("="*80)
    
    # Create synthetic data with different characteristics in each quartile
    np.random.seed(42)
    n_samples = 2000
    
    # Q1: Compressed data (needs expansion)
    q1 = np.random.normal(0, 0.1, n_samples // 4)
    
    # Q2: Normal data (minimal change needed)
    q2 = np.random.normal(1, 0.3, n_samples // 4)
    
    # Q3: Outlier-heavy (needs compression)
    q3 = np.concatenate([
        np.random.normal(2, 0.2, n_samples // 8),
        np.random.uniform(1.5, 2.5, n_samples // 8)
    ])
    
    # Q4: Long tail (needs log-like compression)
    q4 = np.random.exponential(1, n_samples // 4) + 2.5
    
    # Combine
    data = np.concatenate([q1, q2, q3, q4])
    np.random.shuffle(data)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Segment-Specific Transformation Examples', fontsize=16)
    
    # Define custom transformations for each quartile
    quartiles = np.percentile(data, [0, 25, 50, 75, 100])
    
    transformations = [
        {
            'range': (quartiles[0], quartiles[1]),
            'name': 'Q1: Expansion',
            'scale': 1.5,  # Expand by 1.5x
            'curve': 'power',
            'strength': 0.7
        },
        {
            'range': (quartiles[1], quartiles[2]),
            'name': 'Q2: Minimal',
            'scale': 1.0,  # No change
            'curve': 'linear',
            'strength': 1.0
        },
        {
            'range': (quartiles[2], quartiles[3]),
            'name': 'Q3: Compression',
            'scale': 0.6,  # Compress to 60%
            'curve': 'sigmoid',
            'strength': 2.0
        },
        {
            'range': (quartiles[3], quartiles[4]),
            'name': 'Q4: Log Compression',
            'scale': 0.4,  # Strong compression
            'curve': 'log',
            'strength': 3.0
        }
    ]
    
    # Visualize each transformation
    for idx, transform in enumerate(transformations):
        ax = axes[idx // 2, idx % 2]
        
        # Get data in this segment
        mask = (data >= transform['range'][0]) & (data <= transform['range'][1])
        segment_data = data[mask]
        
        # Apply transformation
        compressor = BidirectionalQuartileCompressor()
        transformed = compressor.apply_segment_transformation(
            segment_data,
            {'scale': transform['scale'], 'shift': 0, 
             'curve_type': transform['curve'], 'strength': transform['strength']},
            transform['range']
        )
        
        # Plot
        ax.hist(segment_data, bins=30, alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(transformed, bins=30, alpha=0.5, label='Transformed', density=True, color='red')
        
        ax.set_title(transform['name'])
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Add statistics
        stats_text = f"Original: Î¼={np.mean(segment_data):.2f}, Ïƒ={np.std(segment_data):.2f}\n"
        stats_text += f"Transform: Î¼={np.mean(transformed):.2f}, Ïƒ={np.std(transformed):.2f}\n"
        stats_text += f"Scale: {transform['scale']:.1f}x"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('segment_transformation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    print("ğŸ“Š Loading data...")
    
    # For demonstration, create synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20
    
    # Create features with different characteristics
    X_train = pd.DataFrame()
    
    # Some normal features
    for i in range(5):
        X_train[f'normal_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Some skewed features
    for i in range(5):
        X_train[f'skewed_{i}'] = np.random.exponential(1, n_samples)
    
    # Some features with outliers
    for i in range(5):
        base = np.random.normal(0, 1, n_samples)
        outliers = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        base[outliers] *= 10
        X_train[f'outlier_{i}'] = base
    
    # Some compressed features (low variance)
    for i in range(5):
        X_train[f'compressed_{i}'] = np.random.normal(0, 0.1, n_samples)
    
    # Create target with some relationship to features
    y_train = (
        0.3 * X_train['normal_0'] +
        0.2 * np.log1p(X_train['skewed_0']) +
        0.1 * X_train['outlier_0'].clip(-3, 3) +
        0.4 * np.random.normal(0, 1, n_samples)
    )
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Demonstrate segment transformations
    demonstrate_segment_transformations()
    
    # Compare strategies
    results = compare_compression_strategies_advanced(X_train, pd.Series(y_train))
    
    # Final summary
    print("\n" + "="*80)
    print("ğŸ�“ KEY INSIGHTS: BIDIRECTIONAL ADAPTIVE COMPRESSION")
    print("="*80)
    
    print("\n1. **Segment-Specific Transformations**")
    print("   - Different parts of the distribution need different treatments")
    print("   - Some regions need expansion (compressed data)")
    print("   - Others need compression (outliers, long tails)")
    
    print("\n2. **Optimization Based on Train-CV Gap**")
    print("   - Minimizing the gap ensures we're reducing noise, not signal")
    print("   - Prevents overfitting to training data artifacts")
    
    print("\n3. **Flexible Segmentation**")
    print("   - Not limited to quartiles - can use any number of segments")
    print("   - More segments = more flexibility but higher complexity")
    
    print("\n4. **Gradient Curves**")
    print("   - Smooth transformations prevent discontinuities")
    print("   - Different curve types (linear, sigmoid, power, log) for different patterns")
    
    print("\n5. **Bidirectional Benefits**")
    print("   - Can both compress AND expand as needed")
    print("   - Adapts to the actual data distribution")
    print("   - More effective than one-size-fits-all compression")
    
    print("\n" + "="*80)
    print("âœ… Advanced compression demonstration complete!")
    print("="*80)


# Fixed Ultra-Enhanced Outlier-Biased Adaptive Compression System
# Now with proper NaN handling, fixed parallel processing, and improved robustness

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kurtosis, skew
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter, find_peaks
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import joblib
from tqdm import tqdm
import gc
import time
from datetime import datetime
import logging
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class CompressionProfile:
    """Store compression profile for a feature"""
    feature_name: str
    compression_curve: Tuple[np.ndarray, np.ndarray]
    outlier_method: str
    compression_type: str
    performance_metrics: Dict
    feature_statistics: Dict
    optimization_history: List[Dict]
    meta_features: Dict = field(default_factory=dict)
    version: str = "3.0"
    created_at: datetime = field(default_factory=datetime.now)

class RobustOutlierDetector:
    """Robust outlier detection with NaN handling"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        
    def detect_outliers(self, values: np.ndarray, method: str = 'ensemble') -> Dict:
        """Detect outliers with proper NaN handling"""
        # Handle NaN values
        nan_mask = np.isnan(values)
        if nan_mask.all():
            return self._empty_result(values)
            
        clean_values = values[~nan_mask]
        if len(clean_values) < 10:
            return self._empty_result(values)
        
        # Calculate scores based on method
        if method == 'ensemble':
            scores = self._ensemble_detection(clean_values)
        elif method == 'mad':
            scores = self._mad_scores(clean_values)
        elif method == 'iqr':
            scores = self._iqr_scores(clean_values)
        elif method == 'isolation_forest' and len(clean_values) > 50:
            scores = self._isolation_forest_scores(clean_values)
        else:
            scores = self._mad_scores(clean_values)
        
        # Create full scores array
        full_scores = np.zeros_like(values)
        full_scores[~nan_mask] = scores
        
        # Determine threshold
        threshold = self._calculate_threshold(scores)
        outliers = full_scores > threshold
        
        return {
            'scores': full_scores,
            'outliers': outliers & ~nan_mask,  # Don't mark NaN as outliers
            'threshold': threshold,
            'method': method,
            'n_outliers': int(np.sum(outliers & ~nan_mask))
        }
    
    def _ensemble_detection(self, values):
        """Ensemble of multiple methods"""
        methods_scores = []
        
        # MAD
        methods_scores.append(self._mad_scores(values))
        
        # IQR
        methods_scores.append(self._iqr_scores(values))
        
        # Z-score
        methods_scores.append(self._zscore_scores(values))
        
        # Isolation Forest if enough data
        if len(values) > 50:
            try:
                methods_scores.append(self._isolation_forest_scores(values))
            except:
                pass
        
        # Average scores
        return np.mean(methods_scores, axis=0)
    
    def _mad_scores(self, values):
        """Median Absolute Deviation scores"""
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            mad = np.std(values) * 0.67449
        return np.abs(values - median) / (mad * 1.4826 + 1e-10)
    
    def _iqr_scores(self, values):
        """Interquartile Range scores"""
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            iqr = np.std(values) * 1.35
        
        scores = np.zeros_like(values)
        upper_mask = values > q3
        scores[upper_mask] = (values[upper_mask] - q3) / (iqr + 1e-10)
        
        lower_mask = values < q1
        scores[lower_mask] = (q1 - values[lower_mask]) / (iqr + 1e-10)
        
        return scores
    
    def _zscore_scores(self, values):
        """Z-score based outlier scores"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return np.abs(values - mean) / (std + 1e-10)
    
    def _isolation_forest_scores(self, values):
        """Isolation Forest scores"""
        iso = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=50
        )
        values_2d = values.reshape(-1, 1)
        iso.fit(values_2d)
        scores = -iso.score_samples(values_2d)
        
        # Normalize to same scale as other methods
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score) * 5
        else:
            scores = np.ones_like(scores) * 2.5
            
        return scores
    
    def _calculate_threshold(self, scores):
        """Calculate adaptive threshold"""
        if len(scores) < 10:
            return 3.0
            
        # Multiple threshold strategies
        percentile_threshold = np.percentile(scores, 95)
        std_threshold = np.mean(scores) + 2 * np.std(scores)
        
        # Use more conservative threshold
        return max(min(percentile_threshold, std_threshold), 1.5)
    
    def _empty_result(self, values):
        """Return empty result for edge cases"""
        return {
            'scores': np.zeros_like(values),
            'outliers': np.zeros_like(values, dtype=bool),
            'threshold': 3.0,
            'method': 'none',
            'n_outliers': 0
        }

class AdaptiveCompressionCurveGenerator:
    """Generate compression curves with multiple strategies"""
    
    def __init__(self):
        self.curve_types = ['progressive', 'adaptive_spline', 'quantile_based']
        
    def create_curve(self, feature_data: np.ndarray, outlier_scores: np.ndarray,
                    curve_type: str = 'auto', outlier_bias: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Create compression curve"""
        
        # Handle NaN values
        mask = ~np.isnan(feature_data) & ~np.isnan(outlier_scores)
        if not mask.any():
            return self._empty_curve()
            
        clean_data = feature_data[mask]
        clean_scores = outlier_scores[mask]
        
        if len(clean_data) < 10:
            return self._empty_curve()
        
        # Auto-select curve type
        if curve_type == 'auto':
            curve_type = self._select_curve_type(clean_data, clean_scores)
        
        # Generate curve
        if curve_type == 'progressive':
            return self._progressive_curve(clean_data, clean_scores, outlier_bias)
        elif curve_type == 'adaptive_spline':
            return self._adaptive_spline_curve(clean_data, clean_scores, outlier_bias)
        elif curve_type == 'quantile_based':
            return self._quantile_based_curve(clean_data, clean_scores, outlier_bias)
        else:
            return self._progressive_curve(clean_data, clean_scores, outlier_bias)
    
    def _select_curve_type(self, data, scores):
        """Auto-select best curve type"""
        # Simple heuristics
        if len(data) > 1000:
            return 'adaptive_spline'
        elif kurtosis(data) > 3:
            return 'quantile_based'
        else:
            return 'progressive'
    
    def _empty_curve(self):
        """Return empty curve"""
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    
    def _progressive_curve(self, data, scores, bias):
        """Progressive compression based on outlier scores"""
        compression_strengths = np.zeros_like(scores)
        
        # Define zones
        zones = [
            (0, 1, 0.0),      # Normal
            (1, 2, 0.2),      # Mild outliers
            (2, 3, 0.5),      # Moderate outliers
            (3, 5, 0.7),      # Strong outliers
            (5, np.inf, 0.9)  # Extreme outliers
        ]
        
        for min_score, max_score, strength in zones:
            mask = (scores >= min_score) & (scores < max_score)
            compression_strengths[mask] = strength * bias
        
        # Sort by data values
        sorted_idx = np.argsort(data)
        return data[sorted_idx], compression_strengths[sorted_idx]
    
    def _adaptive_spline_curve(self, data, scores, bias):
        """Smooth spline-based compression"""
        sorted_idx = np.argsort(data)
        sorted_data = data[sorted_idx]
        sorted_scores = scores[sorted_idx]
        
        # Adaptive compression function
        compression_strengths = 1 - np.exp(-sorted_scores * bias / 3)
        compression_strengths = np.clip(compression_strengths * bias, 0, 0.95)
        
        # Smooth with spline if enough points
        if len(sorted_data) > 20:
            try:
                # Subsample for spline fitting
                indices = np.linspace(0, len(sorted_data)-1, 20, dtype=int)
                spline = UnivariateSpline(
                    sorted_data[indices], 
                    compression_strengths[indices], 
                    s=0.1, k=3
                )
                compression_strengths = spline(sorted_data)
                compression_strengths = np.clip(compression_strengths, 0, 0.95)
            except:
                pass  # Keep original if spline fails
        
        return sorted_data, compression_strengths
    
    def _quantile_based_curve(self, data, scores, bias):
        """Compression based on quantiles"""
        sorted_idx = np.argsort(data)
        sorted_data = data[sorted_idx]
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, len(sorted_data))
        
        # Compression based on distance from median
        median_distance = np.abs(quantiles - 0.5) * 2
        compression_strengths = median_distance * bias
        
        return sorted_data, np.clip(compression_strengths, 0, 0.95)

class FixedUltraEnhancedCompressor:
    """Fixed version with robust NaN handling and parallel processing"""
    
    def __init__(self,
                 outlier_bias_strength: float = 0.8,
                 adaptive_rate: float = 0.3,
                 batch_size: int = 5000,
                 n_jobs: int = -1,
                 compression_strategies: List[str] = None,
                 outlier_methods: List[str] = None,
                 handle_nan: str = 'impute'):  # 'impute' or 'skip'
        
        self.outlier_bias_strength = outlier_bias_strength
        self.adaptive_rate = adaptive_rate
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.handle_nan = handle_nan
        
        # Components
        self.outlier_detector = RobustOutlierDetector()
        self.curve_generator = AdaptiveCompressionCurveGenerator()
        
        # Default strategies
        self.compression_strategies = compression_strategies or ['auto', 'progressive', 'adaptive_spline']
        self.outlier_methods = outlier_methods or ['ensemble', 'mad', 'iqr']
        
        # Storage
        self.feature_profiles: Dict[str, CompressionProfile] = {}
        self.nan_imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        # Global metrics
        self.global_metrics = {
            'total_gap_reduction': 0,
            'avg_cv_improvement': 0,
            'features_improved': 0,
            'processing_time': 0,
            'total_features': 0
        }
        
        logger.info(f"Initialized FixedUltraEnhancedCompressor with {self.n_jobs} jobs")
    
    def calculate_feature_statistics(self, values: np.ndarray) -> Dict:
        """Calculate statistics with NaN handling"""
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 2:
            return {
                'mean': 0, 'std': 0, 'median': 0,
                'skew': 0, 'kurtosis': 0,
                'min': 0, 'max': 0,
                'outlier_ratio': 0,
                'nan_ratio': 1.0 if len(values) > 0 else 0
            }
        
        stats = {
            'mean': np.mean(clean_values),
            'std': np.std(clean_values),
            'median': np.median(clean_values),
            'skew': skew(clean_values),
            'kurtosis': kurtosis(clean_values),
            'min': np.min(clean_values),
            'max': np.max(clean_values),
            'q1': np.percentile(clean_values, 25),
            'q3': np.percentile(clean_values, 75),
            'iqr': np.percentile(clean_values, 75) - np.percentile(clean_values, 25),
            'nan_ratio': np.sum(np.isnan(values)) / len(values) if len(values) > 0 else 0
        }
        
        # Outlier detection
        outlier_info = self.outlier_detector.detect_outliers(values, 'mad')
        stats['outlier_ratio'] = outlier_info['n_outliers'] / len(clean_values) if len(clean_values) > 0 else 0
        
        return stats
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data by handling NaN values"""
        # Handle NaN in target
        if y.isna().any():
            logger.warning(f"Target contains {y.isna().sum()} NaN values")
            if self.handle_nan == 'impute':
                y = y.fillna(y.median())
            else:
                # Remove rows with NaN in target
                valid_idx = ~y.isna()
                X = X[valid_idx]
                y = y[valid_idx]
                logger.info(f"Removed {(~valid_idx).sum()} rows with NaN target values")
        
        # Handle NaN in features
        if self.handle_nan == 'impute' and X.isna().any().any():
            logger.info("Imputing NaN values in features")
            X_imputed = pd.DataFrame(
                self.nan_imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            return X_imputed, y
        
        return X, y
    
    def evaluate_compression_strategy(self, X: pd.DataFrame, y: pd.Series,
                                    feature_idx: int, strategy: Dict) -> Dict:
        """Evaluate a single compression strategy"""
        try:
            feature_data = X.iloc[:, feature_idx].values
            
            # Detect outliers
            outlier_info = self.outlier_detector.detect_outliers(
                feature_data, 
                method=strategy['outlier_method']
            )
            
            # Create compression curve
            compression_curve = self.curve_generator.create_curve(
                feature_data,
                outlier_info['scores'],
                curve_type=strategy['compression_type'],
                outlier_bias=self.outlier_bias_strength
            )
            
            # Evaluate
            eval_results = self._evaluate_compression(X, y, feature_idx, compression_curve)
            
            return {
                'strategy': strategy,
                'compression_curve': compression_curve,
                'outlier_info': outlier_info,
                'evaluation': eval_results,
                'score': eval_results['gap_reduction'] + eval_results['cv_improvement']
            }
        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return None
    
    def _evaluate_compression(self, X: pd.DataFrame, y: pd.Series,
                            feature_idx: int, compression_curve: Tuple) -> Dict:
        """Evaluate compression effectiveness"""
        
        if len(X) < 30:
            return {'gap_reduction': 0, 'cv_improvement': 0}
        
        # Apply compression
        X_compressed = X.copy()
        feature_values = X.iloc[:, feature_idx].values
        compressed_values = self.apply_compression(feature_values, compression_curve)
        X_compressed.iloc[:, feature_idx] = compressed_values
        
        # Use simple model for evaluation
        model = LGBMRegressor(
            n_estimators=50,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        
        # Calculate metrics with cross-validation
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        orig_train_scores = []
        orig_val_scores = []
        comp_train_scores = []
        comp_val_scores = []
        
        for train_idx, val_idx in cv.split(X):
            if len(train_idx) < 10 or len(val_idx) < 10:
                continue
                
            # Original data
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            train_pred = model.predict(X.iloc[train_idx])
            val_pred = model.predict(X.iloc[val_idx])
            
            # Handle potential NaN in predictions
            if not np.isnan(train_pred).any() and not np.isnan(val_pred).any():
                orig_train_scores.append(r2_score(y.iloc[train_idx], train_pred))
                orig_val_scores.append(r2_score(y.iloc[val_idx], val_pred))
            
            # Compressed data
            model.fit(X_compressed.iloc[train_idx], y.iloc[train_idx])
            
            train_pred = model.predict(X_compressed.iloc[train_idx])
            val_pred = model.predict(X_compressed.iloc[val_idx])
            
            if not np.isnan(train_pred).any() and not np.isnan(val_pred).any():
                comp_train_scores.append(r2_score(y.iloc[train_idx], train_pred))
                comp_val_scores.append(r2_score(y.iloc[val_idx], val_pred))
        
        # Calculate gaps
        if orig_train_scores and orig_val_scores:
            orig_gap = np.mean(orig_train_scores) - np.mean(orig_val_scores)
            comp_gap = np.mean(comp_train_scores) - np.mean(comp_val_scores) if comp_train_scores else orig_gap
            
            gap_reduction = orig_gap - comp_gap
            cv_improvement = (np.mean(comp_val_scores) if comp_val_scores else 0) - np.mean(orig_val_scores)
        else:
            gap_reduction = 0
            cv_improvement = 0
        
        return {
            'gap_reduction': gap_reduction,
            'cv_improvement': cv_improvement
        }
    
    def optimize_single_feature(self, X: pd.DataFrame, y: pd.Series,
                              feature_idx: int, feature_name: str) -> CompressionProfile:
        """Optimize compression for a single feature"""
        
        # Calculate statistics
        feature_data = X.iloc[:, feature_idx].values
        feature_stats = self.calculate_feature_statistics(feature_data)
        
        # Skip if too many NaN
        if feature_stats['nan_ratio'] > 0.9:
            logger.warning(f"Skipping {feature_name} due to high NaN ratio")
            return self._create_empty_profile(feature_name, feature_stats)
        
        # Evaluate all strategies
        all_results = []
        
        for outlier_method in self.outlier_methods:
            for compression_type in self.compression_strategies:
                strategy = {
                    'outlier_method': outlier_method,
                    'compression_type': compression_type
                }
                
                result = self.evaluate_compression_strategy(X, y, feature_idx, strategy)
                if result is not None:
                    all_results.append(result)
        
        # Select best strategy
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            
            return CompressionProfile(
                feature_name=feature_name,
                compression_curve=best_result['compression_curve'],
                outlier_method=best_result['strategy']['outlier_method'],
                compression_type=best_result['strategy']['compression_type'],
                performance_metrics=best_result['evaluation'],
                feature_statistics=feature_stats,
                optimization_history=[r['evaluation'] for r in all_results[:3]],
                meta_features={}
            )
        else:
            return self._create_empty_profile(feature_name, feature_stats)
    
    def _create_empty_profile(self, feature_name: str, feature_stats: Dict) -> CompressionProfile:
        """Create empty profile for features that can't be optimized"""
        return CompressionProfile(
            feature_name=feature_name,
            compression_curve=(np.array([0.0, 1.0]), np.array([0.0, 0.0])),
            outlier_method='none',
            compression_type='none',
            performance_metrics={'gap_reduction': 0, 'cv_improvement': 0},
            feature_statistics=feature_stats,
            optimization_history=[],
            meta_features={}
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_subset: Optional[List[int]] = None,
            max_features: int = None,
            parallel: bool = True) -> 'FixedUltraEnhancedCompressor':
        """Fit the compressor"""
        
        start_time = time.time()
        
        print("\n" + "="*80)
        print("ğŸš€ FIXED ULTRA-ENHANCED ADAPTIVE COMPRESSOR v3.0")
        print("="*80)
        print(f"Dataset shape: {X.shape}")
        print(f"Parallel processing: {parallel} (n_jobs={self.n_jobs})")
        print(f"NaN handling: {self.handle_nan}")
        
        # Prepare data
        X_prepared, y_prepared = self._prepare_data(X, y)
        
        # Determine features to process
        if feature_subset is None:
            if max_features is None:
                max_features = min(50, X_prepared.shape[1])
            feature_subset = list(range(min(max_features, X_prepared.shape[1])))
        
        # Process features
        if parallel and len(feature_subset) > 1:
            self._process_parallel(X_prepared, y_prepared, feature_subset)
        else:
            self._process_sequential(X_prepared, y_prepared, feature_subset)
        
        # Calculate global metrics
        self._calculate_global_metrics()
        
        # Mark as fitted
        self.is_fitted = True
        
        # Performance metrics
        self.global_metrics['processing_time'] = time.time() - start_time
        
        # Print summary
        self._print_summary()
        
        return self
    
    def _process_parallel(self, X: pd.DataFrame, y: pd.Series, feature_indices: List[int]):
        """Process features in parallel"""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            
            for idx in feature_indices:
                if idx >= X.shape[1]:
                    continue
                    
                feature_name = X.columns[idx] if hasattr(X, 'columns') else f'Feature_{idx}'
                future = executor.submit(self.optimize_single_feature, X, y, idx, feature_name)
                futures[future] = feature_name
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing features"):
                feature_name = futures[future]
                try:
                    profile = future.result()
                    self.feature_profiles[feature_name] = profile
                except Exception as e:
                    logger.error(f"Error processing {feature_name}: {e}")
    
    def _process_sequential(self, X: pd.DataFrame, y: pd.Series, feature_indices: List[int]):
        """Process features sequentially"""
        for idx in tqdm(feature_indices, desc="Processing features"):
            if idx >= X.shape[1]:
                continue
                
            feature_name = X.columns[idx] if hasattr(X, 'columns') else f'Feature_{idx}'
            try:
                profile = self.optimize_single_feature(X, y, idx, feature_name)
                self.feature_profiles[feature_name] = profile
            except Exception as e:
                logger.error(f"Error processing {feature_name}: {e}")
    
    def _calculate_global_metrics(self):
        """Calculate overall metrics"""
        if not self.feature_profiles:
            return
        
        total_gap_reduction = 0
        total_cv_improvement = 0
        features_improved = 0
        
        for profile in self.feature_profiles.values():
            metrics = profile.performance_metrics
            total_gap_reduction += metrics.get('gap_reduction', 0)
            total_cv_improvement += metrics.get('cv_improvement', 0)
            
            if metrics.get('gap_reduction', 0) > 0:
                features_improved += 1
        
        n_features = len(self.feature_profiles)
        
        self.global_metrics.update({
            'total_gap_reduction': total_gap_reduction,
            'avg_gap_reduction': total_gap_reduction / n_features if n_features > 0 else 0,
            'avg_cv_improvement': total_cv_improvement / n_features if n_features > 0 else 0,
            'features_improved': features_improved,
            'improvement_rate': features_improved / n_features if n_features > 0 else 0,
            'total_features': n_features
        })
    
    def _print_summary(self):
        """Print summary report"""
        print("\n" + "="*80)
        print("ğŸ“Š COMPRESSION SUMMARY REPORT")
        print("="*80)
        
        if not self.feature_profiles:
            print("No features processed.")
            return
        
        # Feature summary
        summary_data = []
        for name, profile in list(self.feature_profiles.items())[:20]:
            metrics = profile.performance_metrics
            stats = profile.feature_statistics
            
            summary_data.append({
                'Feature': name[:20],
                'Outlier%': f"{stats.get('outlier_ratio', 0)*100:.1f}",
                'NaN%': f"{stats.get('nan_ratio', 0)*100:.1f}",
                'Method': f"{profile.outlier_method[:6]}/{profile.compression_type[:6]}",
                'Gapâ†“': f"{metrics.get('gap_reduction', 0):.4f}",
                'CVâ†‘': f"{metrics.get('cv_improvement', 0):.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nğŸ“‹ Feature Results (Top 20):")
        print(summary_df.to_string(index=False))
        
        # Global metrics
        print(f"\nğŸ“ˆ Global Metrics:")
        print(f"  â€¢ Total features: {self.global_metrics['total_features']}")
        print(f"  â€¢ Features improved: {self.global_metrics['features_improved']} "
              f"({self.global_metrics['improvement_rate']*100:.1f}%)")
        print(f"  â€¢ Avg gap reduction: {self.global_metrics['avg_gap_reduction']:.4f}")
        print(f"  â€¢ Avg CV improvement: {self.global_metrics['avg_cv_improvement']:.4f}")
        print(f"  â€¢ Processing time: {self.global_metrics['processing_time']:.2f}s")
    
    def apply_compression(self, values: np.ndarray, 
                         compression_curve: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Apply compression to values"""
        sorted_values, compression_strengths = compression_curve
        
        if len(sorted_values) < 2:
            return values
        
        # Handle NaN
        nan_mask = np.isnan(values)
        result = values.copy()
        
        if nan_mask.all():
            return result
        
        valid_values = values[~nan_mask]
        
        # Interpolate compression strengths
        compress_func = interp1d(
            sorted_values, compression_strengths,
            kind='linear', bounds_error=False,
            fill_value=(compression_strengths[0], compression_strengths[-1])
        )
        
        value_compressions = compress_func(valid_values)
        center = np.median(valid_values)
        
        # Apply compression
        compressed_values = valid_values + (center - valid_values) * value_compressions
        result[~nan_mask] = compressed_values
        
        return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using learned compressions"""
        
        if not self.is_fitted:
            raise ValueError("Compressor must be fitted before transform")
        
        X_compressed = X.copy()
        
        # Apply same NaN handling as in fit
        if self.handle_nan == 'impute' and hasattr(self, 'nan_imputer'):
            X_compressed = pd.DataFrame(
                self.nan_imputer.transform(X_compressed),
                columns=X_compressed.columns,
                index=X_compressed.index
            )
        
        # Apply compressions
        for feature_name, profile in tqdm(self.feature_profiles.items(), 
                                        desc="Applying compressions"):
            if hasattr(X_compressed, 'columns') and feature_name in X_compressed.columns:
                idx = list(X_compressed.columns).index(feature_name)
                
                feature_values = X_compressed.iloc[:, idx].values
                compressed_values = self.apply_compression(
                    feature_values, 
                    profile.compression_curve
                )
                X_compressed.iloc[:, idx] = compressed_values
        
        return X_compressed
    
    def save_model(self, filepath: str):
        """Save the fitted compressor"""
        save_data = {
            'feature_profiles': self.feature_profiles,
            'global_metrics': self.global_metrics,
            'params': {
                'outlier_bias_strength': self.outlier_bias_strength,
                'adaptive_rate': self.adaptive_rate,
                'batch_size': self.batch_size,
                'compression_strategies': self.compression_strategies,
                'outlier_methods': self.outlier_methods,
                'handle_nan': self.handle_nan
            },
            'nan_imputer': self.nan_imputer if hasattr(self, 'nan_imputer') else None,
            'version': '3.0'
        }
        
        joblib.dump(save_data, filepath, compress=3)
        file_size = os.path.getsize(filepath) / 1024 / 1024
        logger.info(f"Model saved to {filepath} ({file_size:.2f} MB)")
    
    def load_model(self, filepath: str) -> 'FixedUltraEnhancedCompressor':
        """Load a fitted compressor"""
        data = joblib.load(filepath)
        
        self.feature_profiles = data['feature_profiles']
        self.global_metrics = data['global_metrics']
        
        # Update parameters
        params = data['params']
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Load imputer if available
        if 'nan_imputer' in data and data['nan_imputer'] is not None:
            self.nan_imputer = data['nan_imputer']
        
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self

# Example usage
if __name__ == "__main__":
    print("ğŸš€ Fixed Ultra-Enhanced Compression System v3.0")
    print("="*80)
    
    # Create dataset with NaN values
    np.random.seed(42)
    n_samples = 10000
    n_features = 30
    
    print(f"Creating dataset: {n_samples:,} samples Ã— {n_features} features")
    
    # Generate features with different characteristics
    X_train = pd.DataFrame()
    
    for i in range(n_features):
        if i % 5 == 0:
            # Log-normal with outliers
            data = np.random.lognormal(0, 0.5, n_samples)
            outlier_idx = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
            data[outlier_idx] *= 10
        elif i % 5 == 1:
            # Normal distribution
            data = np.random.normal(100, 10, n_samples)
        elif i % 5 == 2:
            # Skewed distribution
            data = np.random.exponential(2, n_samples)
        elif i % 5 == 3:
            # Heavy-tailed
            from scipy.stats import t
            data = t.rvs(df=3, size=n_samples)
        else:
            # Uniform
            data = np.random.uniform(0, 100, n_samples)
        
        # Add NaN values randomly
        if np.random.random() < 0.3:
            nan_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
            data[nan_idx] = np.nan
        
        X_train[f'feature_{i}'] = data
    
    # Create target with some NaN
    y_train = pd.Series(
        0.3 * X_train['feature_0'].fillna(0) +
        0.2 * X_train['feature_5'].fillna(0) +
        0.1 * X_train['feature_10'].fillna(0) +
        0.4 * np.random.normal(0, 1, n_samples)
    )
    
    # Add some NaN to target
    nan_idx = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    y_train.iloc[nan_idx] = np.nan
    
    print(f"Dataset created with {X_train.isna().sum().sum()} NaN values in features")
    print(f"Target has {y_train.isna().sum()} NaN values")
    
    # Initialize and fit compressor
    compressor = FixedUltraEnhancedCompressor(
        outlier_bias_strength=0.8,
        batch_size=2000,
        n_jobs=4,
        compression_strategies=['auto', 'progressive', 'adaptive_spline'],
        outlier_methods=['ensemble', 'mad'],
        handle_nan='impute'
    )
    
    # Fit on subset of features
    compressor.fit(
        X_train, y_train,
        max_features=15,
        parallel=True
    )
    
    # Transform data
    print("\nğŸ“Š Applying compressions...")
    X_compressed = compressor.transform(X_train)
    
    # Compare performance
    print("\nğŸ�� Model Performance Comparison")
    print("="*80)
    
    # Clean data for evaluation
    X_clean = X_train.fillna(X_train.median())
    y_clean = y_train.fillna(y_train.median())
    X_compressed_clean = X_compressed.fillna(X_compressed.median())
    
    # Sample for faster evaluation
    sample_size = min(2000, n_samples)
    idx = np.random.choice(n_samples, sample_size, replace=False)
    
    from sklearn.model_selection import cross_val_score
    
    models = {
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False)
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name} Results:")
        try:
            # Original data
            cv_orig = cross_val_score(
                model, 
                X_clean.iloc[idx], 
                y_clean.iloc[idx], 
                cv=3, 
                scoring='r2'
            )
            print(f"  Original RÂ²: {cv_orig.mean():.4f} (Â±{cv_orig.std():.4f})")
            
            # Compressed data
            cv_comp = cross_val_score(
                model, 
                X_compressed_clean.iloc[idx], 
                y_clean.iloc[idx], 
                cv=3, 
                scoring='r2'
            )
            print(f"  Compressed RÂ²: {cv_comp.mean():.4f} (Â±{cv_comp.std():.4f})")
            print(f"  Improvement: {cv_comp.mean() - cv_orig.mean():+.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save model
    print("\nğŸ’¾ Saving model...")
    compressor.save_model('fixed_compressor_v3.pkl')
    
    print("\nâœ… Fixed compression analysis complete!")

