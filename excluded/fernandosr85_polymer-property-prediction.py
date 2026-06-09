# Install RDKit, check Python version, and perform functional test

!pip install /kaggle/input/rdkit-2025-3-3/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl --quiet

import sys
print(f"âœ… Python version: {sys.version}")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # Functional test with a simple SMILES
    mol = Chem.MolFromSmiles('CCO')
    if mol is not None:
        mol_wt = Descriptors.MolWt(mol)
        print(f"âœ… RDKit is working. Molecular Weight (CCO): {mol_wt}")
    else:
        print("â�Œ Failed to process SMILES.")
except Exception as e:
    print(f"â�Œ RDKit import error: {e}")


import os                              # OS utilities (path management, file handling)
import pickle                          # Object serialization
import warnings                        # Control warning messages
from collections import Counter        # Utility for counting frequencies

warnings.filterwarnings('ignore')

import pandas as pd                    # Data manipulation and analysis
import numpy as np                     # Numerical computing and arrays
from scipy import stats               # Statistical functions and tests

import matplotlib.pyplot as plt       # Basic plotting and visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting capabilities
import seaborn as sns                 # Statistical data visualization
import plotly.express as px          # Interactive plotting (express API)
import plotly.graph_objects as go     # Advanced interactive plots

from IPython.display import display, HTML  # Display utilities for notebooks

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from sklearn.metrics import mean_absolute_error

import torch                         # PyTorch tensor operations
import torch.nn as nn               # Neural network modules


try:
    # Attempt to load the dataset
    train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
    
    print("Data loaded successfully!")
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print(f"Sample submission: {sample_submission.shape}")
    
except FileNotFoundError:
    print("â�Œ Files not found. Generating synthetic data for exploration...")

    # Generate synthetic data for exploration
    np.random.seed(42)
    n_train = 8000
    n_test = 1500
    
    # Generate synthetic SMILES (typical polymer patterns)
    train_smiles = [f"C{i}C{i+1}N{i%3}O{i%2}" for i in range(n_train)]
    test_smiles = [f"C{i}C{i+1}N{i%3}O{i%2}" for i in range(n_test)]
    
    # Simulate properties with realistic correlations
    tg_base = np.random.normal(80, 40, n_train)
    density_base = np.random.normal(1.2, 0.3, n_train)
    
    train_df = pd.DataFrame({
        'id': range(n_train),
        'SMILES': train_smiles,
        'Tg': tg_base + np.random.normal(0, 10, n_train),  # Glass transition temperature
        'FFV': np.random.beta(2, 5, n_train) * 0.5,        # Fractional free volume (0-0.5)
        'Tc': np.random.gamma(2, 0.1, n_train),            # Thermal conductivity
        'Density': density_base + np.random.normal(0, 0.1, n_train),
        'Rg': np.random.gamma(3, 2, n_train) + 5           # Radius of gyration
    })
    
    test_df = pd.DataFrame({
        'id': range(n_test), 
        'SMILES': test_smiles
    })
    
    print("âœ… Synthetic data generated for exploration!")
    print(f"ğŸ“Š Training set: {train_df.shape}")
    print(f"ğŸ“Š Test set: {test_df.shape}")


def load_and_explore_data():
    """Load and perform initial data overview"""
    try:
        # Try to load real data
        train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
        test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
        sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
        
        print("âœ… Real data loaded successfully!")
        print(f"ğŸ“Š Training set: {train_df.shape}")
        print(f"ğŸ“Š Test set: {test_df.shape}")
        print(f"ğŸ“Š Sample submission: {sample_submission.shape}")
        
    except FileNotFoundError:
        print("ğŸ“� Data not found. Creating simulated data for exploration...")
        
        # Create simulated data based on competition description
        np.random.seed(42)
        n_train = 8000
        n_test = 1500
        
        # Simulated SMILES with realistic polymer patterns
        polymer_fragments = ['C', 'CC', 'CCC', 'c1ccc', 'COC', 'NCO', 'OCO', 'CNC']
        train_smiles = []
        test_smiles = []
        
        for i in range(n_train):
            # Simulate polymer SMILES
            fragments = np.random.choice(polymer_fragments, size=np.random.randint(3, 8))
            smiles = ''.join(fragments) + f'_{i%100}'  # Add variation
            train_smiles.append(smiles)
            
        for i in range(n_test):
            fragments = np.random.choice(polymer_fragments, size=np.random.randint(3, 8))
            smiles = ''.join(fragments) + f'_test_{i%100}'
            test_smiles.append(smiles)
        
        # Properties with realistic correlations based on literature
        # Tg: Glass transition temperature (-100 to 300Â°C typical)
        tg_values = np.random.normal(80, 50, n_train)
        
        # Density: Polymer density (0.8 to 2.0 g/cmÂ³ typical)  
        density_values = np.random.normal(1.2, 0.25, n_train)
        
        # FFV: Fractional free volume (0.1 to 0.4 typical)
        ffv_values = np.random.beta(2, 6, n_train) * 0.4 + 0.05
        
        # Tc: Thermal conductivity (0.1 to 1.0 W/mÂ·K typical)
        tc_values = np.random.gamma(2, 0.15, n_train)
        
        # Rg: Radius of gyration (5 to 50 Ã… typical)
        rg_values = np.random.gamma(3, 3, n_train) + 5
        
        # Add realistic correlations
        # Denser polymers tend to have higher Tg
        tg_values += (density_values - 1.2) * 30
        
        # FFV inversely correlated with density
        ffv_values -= (density_values - 1.2) * 0.1
        ffv_values = np.clip(ffv_values, 0.05, 0.45)
        
        # Tc correlated with density
        tc_values += (density_values - 1.2) * 0.2
        tc_values = np.clip(tc_values, 0.05, 2.0)
        
        train_df = pd.DataFrame({
            'id': range(n_train),
            'SMILES': train_smiles,
            'Tg': tg_values,
            'FFV': ffv_values, 
            'Tc': tc_values,
            'Density': density_values,
            'Rg': rg_values
        })
        
        test_df = pd.DataFrame({
            'id': range(n_test),
            'SMILES': test_smiles
        })
        
        sample_submission = pd.DataFrame({
            'id': range(n_test),
            'Tg': [0.0] * n_test,
            'FFV': [0.0] * n_test,
            'Tc': [0.0] * n_test,
            'Density': [0.0] * n_test,
            'Rg': [0.0] * n_test
        })
        
        print("âœ… Simulated data created!")
        print(f"ğŸ“Š Training set: {train_df.shape}")
        print(f"ğŸ“Š Test set: {test_df.shape}")
    
    return train_df, test_df, sample_submission

def analyze_properties(train_df):
    """Detailed statistical analysis of properties"""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("\n" + "="*50)
    print("ğŸ“Š STATISTICAL ANALYSIS OF PROPERTIES")
    print("="*50)
    
    stats_summary = train_df[properties].describe()
    print(stats_summary)
    
    # Check missing values
    print("\nğŸ“‹ MISSING VALUES:")
    missing_counts = train_df[properties].isnull().sum()
    print(missing_counts)
    
    # Check outliers (using IQR)
    print("\nğŸš¨ OUTLIER ANALYSIS (IQR Method):")
    for prop in properties:
        Q1 = train_df[prop].quantile(0.25)
        Q3 = train_df[prop].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train_df[(train_df[prop] < lower_bound) | (train_df[prop] > upper_bound)]
        print(f"{prop}: {len(outliers)} outliers ({len(outliers)/len(train_df)*100:.1f}%)")
        print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return stats_summary

def analyze_correlations(train_df):
    """Analyze correlations between properties"""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("\n" + "="*50)
    print("ğŸ”— CORRELATION MATRIX")
    print("="*50)
    
    correlation_matrix = train_df[properties].corr()
    print(correlation_matrix.round(3))
    
    # Identify strong correlations (>0.5 or <-0.5)
    print("\nğŸ”¥ STRONG CORRELATIONS (|r| > 0.5):")
    strong_correlations = []
    for i in range(len(properties)):
        for j in range(i+1, len(properties)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_correlations.append({
                    'prop1': properties[i],
                    'prop2': properties[j], 
                    'correlation': corr_val
                })
                print(f"{properties[i]} â†” {properties[j]}: {corr_val:.3f}")
    
    return correlation_matrix, strong_correlations

def analyze_smiles(train_df, test_df):
    """Detailed analysis of SMILES"""
    print("\n" + "="*50)
    print("ğŸ§ª SMILES ANALYSIS")
    print("="*50)
    
    # Basic length statistics
    train_lengths = train_df['SMILES'].str.len()
    test_lengths = test_df['SMILES'].str.len()
    
    print("ğŸ“� SMILES LENGTH:")
    print(f"Train - Mean: {train_lengths.mean():.1f}, Std: {train_lengths.std():.1f}")
    print(f"Train - Min: {train_lengths.min()}, Max: {train_lengths.max()}")
    print(f"Test - Mean: {test_lengths.mean():.1f}, Std: {test_lengths.std():.1f}")
    print(f"Test - Min: {test_lengths.min()}, Max: {test_lengths.max()}")
    
    # Analysis of most frequent characters
    print("\nğŸ”¤ MOST FREQUENT CHARACTERS:")
    all_chars_train = ''.join(train_df['SMILES'])
    char_counts = Counter(all_chars_train)
    
    for char, count in char_counts.most_common(10):
        percentage = count / len(all_chars_train) * 100
        print(f"'{char}': {count} ({percentage:.1f}%)")
    
    # Analysis of common fragments
    print("\nğŸ§© COMMON FRAGMENTS (3 characters):")
    fragments_3 = []
    for smiles in train_df['SMILES']:
        for i in range(len(smiles) - 2):
            fragments_3.append(smiles[i:i+3])
    
    fragment_counts = Counter(fragments_3)
    for fragment, count in fragment_counts.most_common(10):
        percentage = count / len(fragments_3) * 100
        print(f"'{fragment}': {count} ({percentage:.1f}%)")
    
    return {
        'train_lengths': train_lengths,
        'test_lengths': test_lengths,
        'char_counts': char_counts,
        'fragment_counts': fragment_counts
    }

def analyze_wmae_metric(train_df):
    """Analysis of weighted Mean Absolute Error metric"""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("\n" + "="*50)
    print("âš–ï¸� wMAE METRIC ANALYSIS")
    print("="*50)
    
    # Calculate wMAE metric components
    K = len(properties)
    weights = {}
    
    for prop in properties:
        values = train_df[prop].dropna()
        ni = len(values)  # number of available values
        ri = values.max() - values.min()  # range of values
        
        # Weight according to competition formula
        wi = (1/ri) * (K * (1/ni) / sum(1/len(train_df[p].dropna()) for p in properties))
        weights[prop] = wi
        
        print(f"{prop}:")
        print(f"  Available values (ni): {ni}")
        print(f"  Range (ri): {ri:.3f}")
        print(f"  Weight (wi): {wi:.6f}")
        print(f"  Normalized weight: {wi/sum(weights.values()) if weights else 0:.3f}")
    
    # Calculate normalized total weight
    total_weight = sum(weights.values())
    normalized_weights = {prop: w/total_weight for prop, w in weights.items()}
    
    print(f"\nğŸ“Š FINAL NORMALIZED WEIGHTS:")
    for prop, weight in normalized_weights.items():
        print(f"{prop}: {weight:.3f}")
    
    return weights, normalized_weights

def identify_polymer_families(train_df):
    """Identify polymer families based on SMILES patterns"""
    print("\n" + "="*50)
    print("ğŸ‘¨â€�ğŸ‘©â€�ğŸ‘§â€�ğŸ‘¦ POLYMER FAMILY IDENTIFICATION")
    print("="*50)
    
    # Patterns to identify polymer families
    polymer_patterns = {
        'Polyolefin': ['CC', 'CCC', 'CCCC'],
        'Polyester': ['COC', 'C(=O)O', 'OC(=O)'],
        'Polyamide': ['NC(=O)', 'C(=O)N', 'NCO'],
        'Polyether': ['COC', 'OCCO', 'OCO'],
        'Aromatic': ['c1ccc', 'cccc', 'c1cc'],
        'Fluoropolymer': ['CF', 'CFF', 'F']
    }
    
    family_counts = {family: 0 for family in polymer_patterns.keys()}
    family_assignments = []
    
    for smiles in train_df['SMILES']:
        assigned_family = 'Other'
        for family, patterns in polymer_patterns.items():
            if any(pattern in smiles for pattern in patterns):
                assigned_family = family
                family_counts[family] += 1
                break
        if assigned_family == 'Other':
            family_counts['Other'] = family_counts.get('Other', 0) + 1
        family_assignments.append(assigned_family)
    
    # Add family column to dataframe
    train_df_with_families = train_df.copy()
    train_df_with_families['Polymer_Family'] = family_assignments
    
    print("ğŸ“Š FAMILY DISTRIBUTION:")
    for family, count in family_counts.items():
        percentage = count / len(train_df) * 100
        print(f"{family}: {count} ({percentage:.1f}%)")
    
    # Property analysis by family
    print("\nğŸ“ˆ AVERAGE PROPERTIES BY FAMILY:")
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    family_stats = train_df_with_families.groupby('Polymer_Family')[properties].mean()
    print(family_stats.round(2))
    
    return train_df_with_families, family_counts, family_stats

def analyze_for_phase_space_modeling(train_df):
    """Specific analysis to prepare Phase Space Conformational Modeling"""
    print("\n" + "="*50)
    print("ğŸŒŒ PHASE SPACE MODELING ANALYSIS")
    print("="*50)
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # 1. Identify natural clusters in properties
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Normalize properties
    scaler = StandardScaler()
    properties_scaled = scaler.fit_transform(train_df[properties].fillna(train_df[properties].mean()))
    
    # Clustering to identify natural "attractors"
    n_clusters_range = range(3, 8)
    inertias = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(properties_scaled)
        inertias.append(kmeans.inertia_)
    
    # Elbow method for optimal number of clusters
    optimal_clusters = n_clusters_range[np.argmax(np.diff(np.diff(inertias))) + 2]
    print(f"ğŸ�¯ Optimal number of clusters (attractors): {optimal_clusters}")
    
    # Apply optimal clustering
    kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans_optimal.fit_predict(properties_scaled)
    
    print(f"\nğŸ“Š CLUSTER DISTRIBUTION:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(cluster_labels) * 100
        print(f"Cluster {cluster_id}: {count} polymers ({percentage:.1f}%)")
    
    # 2. Cluster stability analysis (pseudo-attractors)
    print(f"\nğŸ�›ï¸� ATTRACTOR CHARACTERISTICS:")
    cluster_centers = scaler.inverse_transform(kmeans_optimal.cluster_centers_)
    
    for i, center in enumerate(cluster_centers):
        print(f"\nAttractor {i}:")
        for j, prop in enumerate(properties):
            print(f"  {prop}: {center[j]:.2f}")
        
        # Calculate "stability" as inverse of intra-cluster variance
        cluster_mask = cluster_labels == i
        cluster_data = train_df[cluster_mask][properties]
        stability = 1.0 / (cluster_data.var().mean() + 1e-6)  # Inverse of variance
        print(f"  Stability: {stability:.3f}")
    
    # 3. Identify polymers in transition regions
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances_to_centers = euclidean_distances(properties_scaled, kmeans_optimal.cluster_centers_)
    min_distances = np.min(distances_to_centers, axis=1)
    
    # Polymers with high distance to centers = transition regions
    transition_threshold = np.percentile(min_distances, 90)
    transition_polymers = min_distances > transition_threshold
    
    print(f"\nğŸŒŠ POLYMERS IN TRANSITION REGIONS:")
    print(f"Threshold: {transition_threshold:.3f}")
    print(f"Transition polymers: {np.sum(transition_polymers)} ({np.sum(transition_polymers)/len(train_df)*100:.1f}%)")
    
    return {
        'optimal_clusters': optimal_clusters,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'scaler': scaler,
        'transition_polymers': transition_polymers,
        'kmeans_model': kmeans_optimal
    }

def run_complete_exploration():
    """Execute complete exploration pipeline"""
    print("ğŸš€ STARTING COMPLETE DATA EXPLORATION")
    print("="*60)
    
    # 1. Load data
    train_df, test_df, sample_submission = load_and_explore_data()
    
    # 2. Statistical analysis
    stats_summary = analyze_properties(train_df)
    
    # 3. Correlation analysis
    correlation_matrix, strong_correlations = analyze_correlations(train_df)
    
    # 4. SMILES analysis
    smiles_analysis = analyze_smiles(train_df, test_df)
    
    # 5. wMAE metric analysis
    weights, normalized_weights = analyze_wmae_metric(train_df)
    
    # 6. Family identification
    train_df_with_families, family_counts, family_stats = identify_polymer_families(train_df)
    
    # 7. Phase Space Modeling analysis
    phase_space_analysis = analyze_for_phase_space_modeling(train_df)
    
    print("\n" + "="*60)
    print("âœ… COMPLETE EXPLORATION FINISHED!")
    print("="*60)
    
    # Executive summary
    print("\nğŸ“‹ EXECUTIVE SUMMARY:")
    print(f"â€¢ Dataset: {train_df.shape[0]} training polymers, {test_df.shape[0]} test")
    print(f"â€¢ Properties: 5 physicochemical properties")
    print(f"â€¢ Strong correlations: {len(strong_correlations)} identified")
    print(f"â€¢ Polymer families: {len(family_counts)} identified") 
    print(f"â€¢ Conformational attractors: {phase_space_analysis['optimal_clusters']} identified")
    print(f"â€¢ Transition polymers: {np.sum(phase_space_analysis['transition_polymers'])} identified")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'train_df_with_families': train_df_with_families,
        'stats_summary': stats_summary,
        'correlation_matrix': correlation_matrix,
        'strong_correlations': strong_correlations,
        'smiles_analysis': smiles_analysis,
        'weights': weights,
        'normalized_weights': normalized_weights,
        'family_counts': family_counts,
        'family_stats': family_stats,
        'phase_space_analysis': phase_space_analysis
    }

def create_visualizations(exploration_results):
    """Create visualizations of exploration results"""
    train_df = exploration_results['train_df']
    correlation_matrix = exploration_results['correlation_matrix']
    
    # Configure style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Exploration - Polymers NeurIPS 2025', fontsize=16, fontweight='bold')
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # 1. Property distributions
    for i, prop in enumerate(properties):
        row, col = i // 3, i % 3
        if row < 2 and col < 3:
            # Handle missing values
            data = train_df[prop].dropna()
            if len(data) > 0:
                axes[row, col].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[row, col].set_title(f'{prop} Distribution')
                axes[row, col].set_xlabel(prop)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
    
    # 2. Correlation matrix in last subplot
    if len(properties) == 5:  # Use last subplot for correlation
        axes[1, 2].remove()
        ax_corr = fig.add_subplot(2, 3, 6)
        im = ax_corr.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(properties)))
        ax_corr.set_yticks(range(len(properties)))
        ax_corr.set_xticklabels(properties, rotation=45)
        ax_corr.set_yticklabels(properties)
        ax_corr.set_title('Correlation Matrix')
        
        # Add values to matrix
        for i in range(len(properties)):
            for j in range(len(properties)):
                text = ax_corr.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", 
                                  color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_corr, shrink=0.8)
        cbar.set_label('Correlation')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Scatter plot of strongest correlations
    strong_correlations = exploration_results['strong_correlations']
    if strong_correlations:
        fig, axes = plt.subplots(1, min(3, len(strong_correlations)), figsize=(15, 5))
        if len(strong_correlations) == 1:
            axes = [axes]
        elif min(3, len(strong_correlations)) == 1:
            axes = [axes]
        
        for i, corr_info in enumerate(strong_correlations[:3]):
            prop1, prop2 = corr_info['prop1'], corr_info['prop2']
            correlation = corr_info['correlation']
            
            if i < len(axes):
                # Remove missing values for both properties
                valid_mask = train_df[prop1].notna() & train_df[prop2].notna()
                x_data = train_df.loc[valid_mask, prop1]
                y_data = train_df.loc[valid_mask, prop2]
                
                if len(x_data) > 0 and len(y_data) > 0:
                    axes[i].scatter(x_data, y_data, alpha=0.6, color='coral')
                    axes[i].set_xlabel(prop1)
                    axes[i].set_ylabel(prop2)
                    axes[i].set_title(f'{prop1} vs {prop2}\nCorrelation: {correlation:.3f}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Trend line
                    if len(x_data) > 1:
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x_data.min(), x_data.max(), 100)
                        axes[i].plot(x_line, p(x_line), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
    
    # 4. SMILES length analysis
    smiles_analysis = exploration_results['smiles_analysis']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Length distribution
    train_lengths = smiles_analysis['train_lengths']
    test_lengths = smiles_analysis['test_lengths']
    
    # Ensure we have valid data
    if len(train_lengths) > 0 and len(test_lengths) > 0:
        axes[0].hist(train_lengths, bins=30, alpha=0.7, label='Train', color='lightblue')
        axes[0].hist(test_lengths, bins=30, alpha=0.7, label='Test', color='lightcoral')
        axes[0].set_xlabel('SMILES Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('SMILES Length Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Top 10 most frequent characters
    if 'char_counts' in smiles_analysis and smiles_analysis['char_counts']:
        top_chars = list(smiles_analysis['char_counts'].most_common(10))
        if top_chars:
            chars, counts = zip(*top_chars)
            
            axes[1].bar(range(len(chars)), counts, color='lightgreen')
            axes[1].set_xlabel('Characters')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Top 10 Most Frequent Characters')
            axes[1].set_xticks(range(len(chars)))
            axes[1].set_xticklabels(chars)
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("ğŸ“Š Visualizations created successfully!")

def prepare_phase_space_features(exploration_results):
    """Prepare specific features for Phase Space Conformational Modeling"""
    print("\n" + "="*60)
    print("ğŸŒŒ PHASE SPACE CONFORMATIONAL MODELING PREPARATION")
    print("="*60)
    
    train_df = exploration_results['train_df_with_families']
    phase_space_analysis = exploration_results['phase_space_analysis']
    
    # 1. Create features based on identified attractors
    cluster_labels = phase_space_analysis['cluster_labels']
    train_df['Attractor_ID'] = cluster_labels
    
    # 2. Calculate distances to attractor centers
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    scaler = phase_space_analysis['scaler']
    kmeans_model = phase_space_analysis['kmeans_model']
    
    properties_scaled = scaler.transform(train_df[properties].fillna(train_df[properties].mean()))
    distances_to_centers = []
    
    for i, center in enumerate(kmeans_model.cluster_centers_):
        distances = np.linalg.norm(properties_scaled - center, axis=1)
        train_df[f'Distance_to_Attractor_{i}'] = distances
        distances_to_centers.append(distances)
    
    # 3. Identify multi-stable polymers (close to multiple attractors)
    distances_array = np.array(distances_to_centers).T
    
    # Multi-stable polymers: close to at least 2 attractors
    threshold_distance = np.percentile(distances_array.min(axis=1), 50)  # Median of minimum distance
    
    multi_stable_mask = []
    for i, distances in enumerate(distances_array):
        close_attractors = np.sum(distances < threshold_distance * 1.5)
        multi_stable_mask.append(close_attractors >= 2)
    
    train_df['Multi_Stable'] = multi_stable_mask
    
    print(f"ğŸ�¯ Attractors identified: {len(kmeans_model.cluster_centers_)}")
    print(f"ğŸŒŠ Multi-stable polymers: {np.sum(multi_stable_mask)} ({np.sum(multi_stable_mask)/len(train_df)*100:.1f}%)")
    
    # 4. Create features for adaptive sampling
    adaptive_features = {}
    
    # Features based on chemical family
    family_encoding = pd.get_dummies(train_df['Polymer_Family'], prefix='Family')
    train_df = pd.concat([train_df, family_encoding], axis=1)
    
    # Features based on SMILES chemical content
    def extract_chemical_features(smiles):
        """Extract chemical features from SMILES"""
        features = {
            'carbon_content': smiles.count('C') / len(smiles) if len(smiles) > 0 else 0,
            'oxygen_content': smiles.count('O') / len(smiles) if len(smiles) > 0 else 0,
            'nitrogen_content': smiles.count('N') / len(smiles) if len(smiles) > 0 else 0,
            'aromatic_content': smiles.count('c') / len(smiles) if len(smiles) > 0 else 0,
            'fluorine_content': smiles.count('F') / len(smiles) if len(smiles) > 0 else 0,
            'ring_content': smiles.count('1') / len(smiles) if len(smiles) > 0 else 0,  # Proxy for rings
            'branch_content': smiles.count('(') / len(smiles) if len(smiles) > 0 else 0,  # Proxy for branches
            'smiles_length': len(smiles),
            'complexity_ratio': (smiles.count('(') + smiles.count('[')) / len(smiles) if len(smiles) > 0 else 0
        }
        return features
    
    # Apply chemical feature extraction
    chemical_features_list = []
    for smiles in train_df['SMILES']:
        chemical_features_list.append(extract_chemical_features(smiles))
    
    chemical_features_df = pd.DataFrame(chemical_features_list)
    train_df = pd.concat([train_df, chemical_features_df], axis=1)
    
    # 5. Create features for thermodynamic weighting
    normalized_weights = exploration_results['normalized_weights']
    
    print(f"\nâš–ï¸� THERMODYNAMIC WEIGHTS:")
    for prop, weight in normalized_weights.items():
        print(f"{prop}: {weight:.4f}")
    
    # 6. Prepare seeds for diversified ensemble
    ensemble_seeds = [84, 294, 1134, 420, 252, 1001, 3780, 756]
    temperature_params = [0.15, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.25]
    
    adaptive_features = {
        'ensemble_seeds': ensemble_seeds,
        'temperature_params': temperature_params,
        'normalized_weights': normalized_weights,
        'attractor_info': {
            'n_attractors': len(kmeans_model.cluster_centers_),
            'attractor_centers': kmeans_model.cluster_centers_,
            'scaler': scaler
        }
    }
    
    print(f"\nğŸ”§ PREPARED FEATURES:")
    print(f"â€¢ Ensemble seeds: {len(ensemble_seeds)}")
    print(f"â€¢ Temperature parameters: {len(temperature_params)}")
    print(f"â€¢ Chemical features: {len(chemical_features_df.columns)}")
    print(f"â€¢ Attractor features: {len([col for col in train_df.columns if 'Attractor' in col])}")
    print(f"â€¢ Family features: {len(family_encoding.columns)}")
    
    return train_df, adaptive_features

def generate_insights(exploration_results, prepared_features):
    """Generate insights from exploration analysis"""
    print("\n" + "="*60)
    print("ğŸ’¡ DATA INSIGHTS")
    print("="*60)
    
    train_df, adaptive_features = prepared_features
    
    # 1. Data distribution insights
    print("ğŸ“Š DATA DISTRIBUTION INSIGHTS:")
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    for prop in properties:
        values = train_df[prop].dropna()
        skewness = values.skew()
        kurtosis = values.kurtosis()
        
        print(f"\n{prop}:")
        print(f"  â€¢ Skewness: {skewness:.3f} {'(asymmetric)' if abs(skewness) > 1 else '(approximately symmetric)'}")
        print(f"  â€¢ Kurtosis: {kurtosis:.3f} {'(leptokurtic)' if kurtosis > 0 else '(platykurtic)'}")
        
        if abs(skewness) > 1:
            print(f"  â€¢ Note: Consider log or Box-Cox transformation for {prop}")
    
    # 2. Correlation insights
    strong_correlations = exploration_results['strong_correlations']
    print(f"\nğŸ”— CORRELATION INSIGHTS:")
    print(f"â€¢ {len(strong_correlations)} strong correlations identified")
    
    if strong_correlations:
        print("â€¢ Main correlations:")
        for corr in strong_correlations:
            direction = "positive" if corr['correlation'] > 0 else "negative"
            print(f"  - {corr['prop1']} â†” {corr['prop2']}: {corr['correlation']:.3f} ({direction})")
        
        print("â€¢ Modeling implications:")
        print("  - Multi-task learning can be very effective")
        print("  - Cross-property regularization may improve performance")
        print("  - Ensemble models can better capture interdependencies")
    
    # 3. Polymer family insights
    family_counts = exploration_results['family_counts']
    family_stats = exploration_results['family_stats']
    
    print(f"\nğŸ‘¨â€�ğŸ‘©â€�ğŸ‘§â€�ğŸ‘¦ FAMILY INSIGHTS:")
    dominant_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("â€¢ Dominant families:")
    for family, count in dominant_families:
        percentage = count / sum(family_counts.values()) * 100
        print(f"  - {family}: {percentage:.1f}% ({count} polymers)")
    
    print("â€¢ Significant differences between families:")
    for prop in properties:
        if prop in family_stats.columns:
            prop_range = family_stats[prop].max() - family_stats[prop].min()
            prop_std = family_stats[prop].std()
            if prop_range > 2 * prop_std:  # Significant difference
                max_family = family_stats[prop].idxmax()
                min_family = family_stats[prop].idxmin()
                print(f"  - {prop}: {max_family} ({family_stats.loc[max_family, prop]:.2f}) vs {min_family} ({family_stats.loc[min_family, prop]:.2f})")
    
    # 4. Phase Space Modeling insights
    phase_space_analysis = exploration_results['phase_space_analysis']
    n_attractors = phase_space_analysis['optimal_clusters']
    
    print(f"\nğŸŒŒ PHASE SPACE MODELING INSIGHTS:")
    print(f"â€¢ Number of attractors identified: {n_attractors}")
    print(f"â€¢ Ensemble strategy:")
    print(f"  - {len(adaptive_features['ensemble_seeds'])} base models with different seeds")
    print(f"  - {len(adaptive_features['temperature_params'])} temperature levels")
    print(f"  - Adaptive sampling based on {len(family_counts)} chemical families")
    
    # 5. wMAE optimization strategy
    normalized_weights = exploration_results['normalized_weights']
    
    print(f"\nâš–ï¸� wMAE OPTIMIZATION STRATEGY:")
    print("â€¢ Properties with highest weight in metric:")
    sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
    for prop, weight in sorted_weights:
        print(f"  - {prop}: {weight:.4f}")
    
    print("â€¢ Focus recommendations:")
    high_weight_props = [prop for prop, weight in sorted_weights if weight > 0.25]
    if high_weight_props:
        print(f"  - Extra focus on: {', '.join(high_weight_props)}")
    print("  - Implement custom loss function that mimics wMAE")
    print("  - Use adaptive weights during training")
    
    # 6. Validation pipeline
    print(f"\nâœ… VALIDATION PIPELINE:")
    print("â€¢ Stratified cross-validation by chemical family")
    print("â€¢ Temporal hold-out if temporal patterns exist")
    print("â€¢ Attractor stability validation")
    print("â€¢ Robustness testing with different temperatures")
    print("â€¢ Uncertainty calibration for active learning")
    
    return {
        'insights': {
            'data_distribution': 'Some properties are asymmetric',
            'correlations': f'{len(strong_correlations)} strong correlations identified',
            'families': f'{len(family_counts)} polymer families identified',
            'attractors': f'{n_attractors} conformational attractors found'
        }
    }

if __name__ == "__main__":
    # Execute complete exploration
    exploration_results = run_complete_exploration()
    
    # Prepare features for Phase Space Modeling
    prepared_features = prepare_phase_space_features(exploration_results)
    
    # Generate insights
    final_insights = generate_insights(exploration_results, prepared_features)
    
    # Create visualizations (optional)
    try:
        create_visualizations(exploration_results)
    except Exception as e:
        print(f"âš ï¸� Error creating visualizations: {e}")
        print("Continuing without visualizations...")
    
    print("\n" + "="*60)
    print("ğŸ�‰ COMPLETE EXPLORATION FINISHED!")
    print("Next step: Implement Phase Space Conformational Modeling")
    print("="*60)


def preview_ultra_premium_dataset():
    """Preview of the ultra-premium dataset to understand its structure"""
    print("ğŸ”� ULTRA-PREMIUM DATASET PREVIEW")
    print("=" * 60)
    
    # Correct path provided by user
    dataset_path = '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv'
    
    print(f"ğŸ“� Checking: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("â�Œ File not found!")
        
        # Try to find related CSV files
        base_dir = '/kaggle/input/tg-smiles-pid-polymer-class'
        if os.path.exists(base_dir):
            print(f"\nğŸ“‚ Available files in {base_dir}:")
            files = os.listdir(base_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            for file in csv_files:
                file_path = os.path.join(base_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  ğŸ“„ {file} ({file_size:,} bytes)")
                
                # If a similar file is found, suggest it
                if 'premium' in file.lower() or 'polymer' in file.lower():
                    print(f"    ğŸ¤” Could this be the correct file?")
        
        return None
    
    try:
        # Load dataset
        print("ğŸ“Š Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        print(f"âœ… Dataset loaded successfully!")
        print(f"ğŸ“� Dimensions: {df.shape[0]:,} rows Ã— {df.shape[1]} columns")
        
        # Column analysis
        print(f"\nğŸ“‹ DATASET STRUCTURE:")
        print(f"Available columns:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df) * 100
            unique_count = df[col].nunique()
            
            print(f"  {i:2d}. {col:<20} | {str(dtype):<10} | NaN: {null_count:5,} ({null_pct:5.1f}%) | Unique: {unique_count:6,}")
        
        # Data preview
        print(f"\nğŸ“Š DATA PREVIEW (first 3 rows):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head(3).to_string())
        
        # Detailed analysis by column
        print(f"\nğŸ”� DETAILED ANALYSIS:")
        
        # SMILES
        if 'SMILES' in df.columns:
            print(f"\nğŸ§¬ SMILES Column:")
            smiles_data = df['SMILES'].dropna()
            print(f"  Total valid: {len(smiles_data):,}")
            print(f"  Average length: {smiles_data.str.len().mean():.1f} characters")
            print(f"  Range: {smiles_data.str.len().min()}-{smiles_data.str.len().max()} characters")
            
            print(f"\n  ğŸ“� SMILES Examples:")
            for i, smiles in enumerate(smiles_data.head(5)):
                print(f"    {i+1}. {smiles}")
        
        # Important numerical properties
        target_properties = ['Tc', 'Density', 'Tg', 'FFV', 'Rg']
        
        print(f"\nğŸ�¯ TARGET PROPERTIES:")
        for prop in target_properties:
            if prop in df.columns:
                prop_data = df[prop].dropna()
                if len(prop_data) > 0:
                    print(f"\n  ğŸ“ˆ {prop}:")
                    print(f"    Valid: {len(prop_data):,}/{len(df):,} ({len(prop_data)/len(df)*100:.1f}%)")
                    print(f"    Mean: {prop_data.mean():.4f} Â± {prop_data.std():.4f}")
                    print(f"    Range: {prop_data.min():.4f} - {prop_data.max():.4f}")
                    print(f"    Median: {prop_data.median():.4f}")
                    
                    # Specific analysis for premium properties
                    if prop == 'Tc':
                        high_tc = (prop_data >= 0.7).sum()
                        ultra_tc = (prop_data >= 0.8).sum()
                        print(f"    ğŸ�¯ High Tc (â‰¥0.7): {high_tc:,} ({high_tc/len(prop_data)*100:.1f}%)")
                        print(f"    ğŸš€ Ultra Tc (â‰¥0.8): {ultra_tc:,} ({ultra_tc/len(prop_data)*100:.1f}%)")
                        
                    elif prop == 'Density':
                        high_density = (prop_data >= 0.6).sum()
                        ultra_density = (prop_data >= 0.8).sum()
                        print(f"    ğŸ�¯ High Density (â‰¥0.6): {high_density:,} ({high_density/len(prop_data)*100:.1f}%)")
                        print(f"    ğŸš€ Ultra Density (â‰¥0.8): {ultra_density:,} ({ultra_density/len(prop_data)*100:.1f}%)")
                    
                    # Distribution by quartiles
                    q25, q50, q75 = prop_data.quantile([0.25, 0.5, 0.75])
                    print(f"    Quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")
                else:
                    print(f"\n  âš ï¸� {prop}: Present but no valid data")
            else:
                print(f"\n  â�Œ {prop}: Not found")
        
        # Other important columns
        other_cols = ['Quality_Score', 'Template', 'Validation', 'Atoms', 'Rings', 'MW', 'LogP']
        
        available_others = [col for col in other_cols if col in df.columns]
        if available_others:
            print(f"\nğŸ“‹ OTHER IMPORTANT COLUMNS:")
            for col in available_others:
                print(f"\n  ğŸ“Š {col}:")
                if df[col].dtype in ['object', 'string']:
                    # Categorical data
                    value_counts = df[col].value_counts()
                    print(f"    Type: Categorical")
                    print(f"    Unique values: {len(value_counts)}")
                    print(f"    Top 5 values:")
                    for val, count in value_counts.head(5).items():
                        pct = count / len(df) * 100
                        print(f"      {val}: {count:,} ({pct:.1f}%)")
                else:
                    # Numerical data
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        print(f"    Type: Numerical")
                        print(f"    Valid: {len(col_data):,}/{len(df):,}")
                        print(f"    Mean: {col_data.mean():.3f} Â± {col_data.std():.3f}")
                        print(f"    Range: {col_data.min():.3f} - {col_data.max():.3f}")
        
        # Data quality analysis
        print(f"\nğŸ�¯ QUALITY ANALYSIS:")
        
        # Data completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        print(f"  Overall completeness: {completeness:.1f}%")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"  Duplicate rows: {duplicates:,}")
        
        if 'SMILES' in df.columns:
            smiles_duplicates = df['SMILES'].duplicated().sum()
            print(f"  Duplicate SMILES: {smiles_duplicates:,}")
        
        # Ultra-premium criteria
        print(f"\nğŸš€ ULTRA-PREMIUM CRITERIA:")
        
        premium_criteria = {}
        if 'Tc' in df.columns and 'Density' in df.columns:
            tc_data = df['Tc'].dropna()
            density_data = df['Density'].dropna()
            
            # Combined criterion: Tc â‰¥ 0.7 AND Density â‰¥ 0.6
            both_available = df[['Tc', 'Density']].dropna()
            if len(both_available) > 0:
                ultra_premium = both_available[(both_available['Tc'] >= 0.7) & (both_available['Density'] >= 0.6)]
                premium_criteria['both'] = len(ultra_premium)
                print(f"  ğŸ�¯ Ultra-premium (Tcâ‰¥0.7 AND Densityâ‰¥0.6): {len(ultra_premium):,} polymers")
                
                if len(ultra_premium) > 0:
                    print(f"    Average Tc: {ultra_premium['Tc'].mean():.3f}")
                    print(f"    Average Density: {ultra_premium['Density'].mean():.3f}")
        
        # Individual criteria
        if 'Tc' in df.columns:
            tc_premium = (df['Tc'] >= 0.7).sum()
            premium_criteria['tc'] = tc_premium
            print(f"  ğŸ�¯ High Tc (â‰¥0.7): {tc_premium:,} polymers")
            
        if 'Density' in df.columns:
            density_premium = (df['Density'] >= 0.6).sum()
            premium_criteria['density'] = density_premium
            print(f"  ğŸ�¯ High Density (â‰¥0.6): {density_premium:,} polymers")
        
        # Pipeline compatibility
        print(f"\nğŸ”§ PIPELINE COMPATIBILITY:")
        
        compatibility_score = 0
        max_score = 0
        
        # SMILES (mandatory)
        if 'SMILES' in df.columns and df['SMILES'].notna().sum() > 0:
            print(f"  âœ… SMILES: Available ({df['SMILES'].notna().sum():,} valid)")
            compatibility_score += 3
        else:
            print(f"  â�Œ SMILES: MISSING or invalid")
        max_score += 3
        
        # Target properties
        for prop in ['Tc', 'Density', 'Tg', 'FFV', 'Rg']:
            if prop in df.columns and df[prop].notna().sum() > 0:
                valid_count = df[prop].notna().sum()
                weight = {'Tc': 3, 'Density': 2, 'FFV': 1, 'Tg': 1, 'Rg': 1}[prop]
                print(f"  âœ… {prop}: {valid_count:,} valid (weight: {weight})")
                compatibility_score += weight
            else:
                weight = {'Tc': 3, 'Density': 2, 'FFV': 1, 'Tg': 1, 'Rg': 1}[prop]
                print(f"  â�Œ {prop}: Missing (weight: {weight})")
            max_score += weight
        
        compatibility_pct = compatibility_score / max_score * 100
        print(f"\n  ğŸ“Š Compatibility score: {compatibility_score}/{max_score} ({compatibility_pct:.1f}%)")
        
        # Usage recommendation
        print(f"\nğŸ�¯ USAGE RECOMMENDATION:")
        
        if compatibility_pct >= 80:
            print(f"  ğŸ�‰ EXCELLENT - Highly compatible dataset!")
            recommendation = "USE_DIRECT"
        elif compatibility_pct >= 60:
            print(f"  âœ… GOOD - Compatible dataset with minimal adaptations")
            recommendation = "USE_WITH_ADAPTATIONS"
        elif compatibility_pct >= 40:
            print(f"  âš ï¸� MODERATE - Requires significant adaptations")
            recommendation = "USE_WITH_MAJOR_ADAPTATIONS"
        else:
            print(f"  â�Œ LOW - Dataset not recommended")
            recommendation = "NOT_RECOMMENDED"
        
        # Integration strategy
        print(f"\nğŸ”§ INTEGRATION STRATEGY:")
        
        if recommendation in ["USE_DIRECT", "USE_WITH_ADAPTATIONS"]:
            print(f"  1. âœ… Use path: '/kaggle/input/polymer-smiles-dataset/ultra_premium_polymers.csv'")
            
            if 'Tc' in df.columns and 'Density' in df.columns:
                print(f"  2. âœ… Integrate Tc and Density data")
                print(f"  3. âœ… Filter premium polymers:")
                if 'both' in premium_criteria:
                    print(f"     - Ultra-premium: {premium_criteria['both']:,} polymers")
                if 'tc' in premium_criteria:
                    print(f"     - High Tc: {premium_criteria['tc']:,} polymers")
                if 'density' in premium_criteria:
                    print(f"     - High Density: {premium_criteria['density']:,} polymers")
            
            # Impact estimation
            total_premium = premium_criteria.get('both', 0) or max(premium_criteria.get('tc', 0), premium_criteria.get('density', 0))
            if total_premium > 0:
                print(f"\n  ğŸ“ˆ ESTIMATED IMPACT:")
                print(f"     Premium samples: +{total_premium:,}")
                print(f"     Benefited properties: Tc (60.95%) + Density (32.03%) = 92.98%")
                
                # Estimate based on number of samples
                if total_premium >= 200:
                    impact = "0.010-0.025"
                elif total_premium >= 100:
                    impact = "0.005-0.015"
                else:
                    impact = "0.002-0.008"
                    
                print(f"     Expected improvement: {impact} points in CV score")
                print(f"     Percentage reduction: {float(impact.split('-')[0])/0.0858*100:.1f}-{float(impact.split('-')[1])/0.0858*100:.1f}%")
        
        else:
            print(f"  â�Œ Dataset not recommended for direct integration")
            print(f"  ğŸ’¡ Consider using only the optimized pipeline (already improves 14.1%)")
        
        return {
            'path': dataset_path,
            'data': df,
            'shape': df.shape,
            'compatibility_score': compatibility_pct,
            'recommendation': recommendation,
            'premium_criteria': premium_criteria,
            'columns': list(df.columns),
            'properties_available': [prop for prop in target_properties if prop in df.columns]
        }
        
    except Exception as e:
        print(f"â�Œ Error loading dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None

# Execute preview
if __name__ == "__main__":
    result = preview_ultra_premium_dataset()
    
    if result:
        print(f"\nğŸ“Š Dataset: {result['shape'][0]:,} Ã— {result['shape'][1]} | Compatibility: {result['compatibility_score']:.1f}% | {result['recommendation']}")
        print(f"Properties: {', '.join(result['properties_available'])}")
        
        if result['premium_criteria']:
            print(f"Premium polymers: {max(result['premium_criteria'].values()):,}")
    else:
        print("â�Œ Preview failed")


def preview_smiles_extra_dataset():
    """Preview of the smiles-extra-data dataset to understand its structure"""
    print("ğŸ”� SMILES-EXTRA-DATA DATASET PREVIEW")
    print("=" * 60)
    
    # Base path for extra data
    base_path = '/kaggle/input/smiles-extra-data'
    
    print(f"ğŸ“� Checking: {base_path}")
    
    if not os.path.exists(base_path):
        print("â�Œ Directory not found!")
        return None
    
    # List all available files
    print(f"\nğŸ“‚ AVAILABLE FILES:")
    all_files = []
    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            all_files.append((file, file_size))
            print(f"  ğŸ“„ {file} ({file_size:,} bytes)")
    
    # Target files of interest
    target_files = {
        'bigsmiles': 'JCIM_sup_bigsmiles.csv',
        'dnst1': 'data_dnst1.xlsx', 
        'tg3': 'data_tg3.xlsx'
    }
    
    datasets = {}
    
    # Analyze each file
    for key, filename in target_files.items():
        file_path = os.path.join(base_path, filename)
        
        print(f"\n{'='*50}")
        print(f"ğŸ“Š ANALYZING: {filename}")
        print(f"{'='*50}")
        
        if not os.path.exists(file_path):
            print(f"â�Œ File {filename} not found!")
            continue
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                datasets[key] = {'data': df, 'sheets': None}
                analyze_dataframe(df, filename, key)
                
            elif filename.endswith('.xlsx'):
                # Read Excel file and check sheets
                excel_file = pd.ExcelFile(file_path)
                sheets = excel_file.sheet_names
                
                print(f"ğŸ“‹ Sheets found: {sheets}")
                
                sheet_data = {}
                for sheet in sheets:
                    print(f"\nğŸ“Š Analyzing sheet: '{sheet}'")
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        sheet_data[sheet] = df
                        analyze_dataframe(df, f"{filename}[{sheet}]", f"{key}_{sheet}")
                    except Exception as e:
                        print(f"â�Œ Error reading sheet '{sheet}': {e}")
                
                datasets[key] = {'data': sheet_data, 'sheets': sheets}
                
        except Exception as e:
            print(f"â�Œ Error processing {filename}: {e}")
            continue
    
    # Consolidated analysis
    print(f"\n{'='*60}")
    print(f"ğŸ�¯ CONSOLIDATED ANALYSIS")
    print(f"{'='*60}")
    
    analyze_consolidated_data(datasets)
    
    return datasets

def analyze_dataframe(df, source_name, key):
    """Analyzes a specific DataFrame"""
    
    if df.empty:
        print(f"âš ï¸� Empty DataFrame!")
        return
    
    print(f"ğŸ“� Dimensions: {df.shape[0]:,} rows Ã— {df.shape[1]} columns")
    
    # Column analysis
    print(f"\nğŸ“‹ AVAILABLE COLUMNS:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        
        print(f"  {i:2d}. {col:<25} | {str(dtype):<12} | NaN: {null_count:5,} ({null_pct:5.1f}%) | Unique: {unique_count:6,}")
    
    # Data preview
    if len(df) > 0:
        print(f"\nğŸ“Š DATA PREVIEW (first 3 rows):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 50)
        try:
            print(df.head(3).to_string())
        except:
            print("Data too wide for complete preview")
    
    # Analysis of relevant columns for polymers
    polymer_columns = {
        'SMILES': ['smiles', 'smile', 'canonical_smiles', 'polymer_smiles'],
        'BigSMILES': ['bigsmiles', 'big_smiles', 'polymer_notation'],
        'Tc': ['tc', 'critical_temperature', 'crit_temp'],
        'Tg': ['tg', 'glass_transition', 'glass_temp', 'T_g', 'Tg_K', 'Tg_C'],
        'Density': ['density', 'dens', 'rho'],
        'MW': ['mw', 'molecular_weight', 'mol_weight'],
        'Polymer': ['polymer', 'polymer_name', 'name', 'material'],
        'Monomer': ['monomer', 'monomer_smiles', 'monomer_name'],
        'Property': ['property', 'target', 'value', 'measurement']
    }
    
    found_columns = {}
    for category, possible_names in polymer_columns.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(name in col_lower for name in possible_names):
                if category not in found_columns:
                    found_columns[category] = []
                found_columns[category].append(col)
    
    if found_columns:
        print(f"\nğŸ§¬ POLYMER-RELEVANT COLUMNS:")
        for category, columns in found_columns.items():
            print(f"  {category}: {columns}")
            
            # Specific analysis by category
            for col in columns:
                analyze_column_specifics(df, col, category)
    
    # Data quality analysis
    print(f"\nğŸ“Š DATA QUALITY:")
    
    total_cells = len(df) * len(df.columns) if len(df) > 0 else 0
    missing_cells = df.isnull().sum().sum()
    completeness = (total_cells - missing_cells) / total_cells * 100 if total_cells > 0 else 0
    print(f"  Overall completeness: {completeness:.1f}%")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,}")
    
    # Estimated compatibility
    compatibility = estimate_compatibility(df, found_columns)
    print(f"  Estimated compatibility: {compatibility:.1f}%")

def analyze_column_specifics(df, col, category):
    """Specific analysis by column type"""
    
    data = df[col].dropna()
    if len(data) == 0:
        return
    
    print(f"\n    ğŸ“ˆ {col} ({category}):")
    print(f"      Valid values: {len(data):,}/{len(df):,} ({len(data)/len(df)*100:.1f}%)")
    
    if category in ['SMILES', 'BigSMILES', 'Monomer']:
        # Chemical string analysis
        if len(data) > 0:
            lengths = data.astype(str).str.len()
            print(f"      Average length: {lengths.mean():.1f} characters")
            print(f"      Range: {lengths.min()}-{lengths.max()} characters")
            
            # Examples
            print(f"      Examples:")
            for i, example in enumerate(data.head(3)):
                example_str = str(example)
                if len(example_str) > 60:
                    example_str = example_str[:57] + "..."
                print(f"        {i+1}. {example_str}")
    
    elif category in ['Tc', 'Tg', 'Density', 'MW']:
        # Numerical properties analysis
        numeric_data = pd.to_numeric(data, errors='coerce').dropna()
        if len(numeric_data) > 0:
            print(f"      Numeric values: {len(numeric_data):,}")
            print(f"      Mean: {numeric_data.mean():.4f} Â± {numeric_data.std():.4f}")
            print(f"      Range: {numeric_data.min():.4f} - {numeric_data.max():.4f}")
            print(f"      Median: {numeric_data.median():.4f}")
            
            # Property-specific analysis
            if category == 'Tg':
                # Tg usually in K or C
                if numeric_data.mean() > 200:
                    print(f"      ğŸŒ¡ï¸� Appears to be in Kelvin")
                    tg_c = numeric_data - 273.15
                    high_tg = (tg_c >= 100).sum()
                    print(f"      ğŸ�¯ High Tg (â‰¥100Â°C): {high_tg:,} ({high_tg/len(numeric_data)*100:.1f}%)")
                else:
                    print(f"      ğŸŒ¡ï¸� Appears to be in Celsius")
                    high_tg = (numeric_data >= 100).sum()
                    print(f"      ğŸ�¯ High Tg (â‰¥100Â°C): {high_tg:,} ({high_tg/len(numeric_data)*100:.1f}%)")
                    
            elif category == 'Tc':
                high_tc = (numeric_data >= 0.7).sum()
                ultra_tc = (numeric_data >= 0.8).sum()
                print(f"      ğŸ�¯ High Tc (â‰¥0.7): {high_tc:,} ({high_tc/len(numeric_data)*100:.1f}%)")
                print(f"      ğŸš€ Ultra Tc (â‰¥0.8): {ultra_tc:,} ({ultra_tc/len(numeric_data)*100:.1f}%)")
                
            elif category == 'Density':
                high_density = (numeric_data >= 0.6).sum()
                ultra_density = (numeric_data >= 0.8).sum()
                print(f"      ğŸ�¯ High Density (â‰¥0.6): {high_density:,} ({high_density/len(numeric_data)*100:.1f}%)")
                print(f"      ğŸš€ Ultra Density (â‰¥0.8): {ultra_density:,} ({ultra_density/len(numeric_data)*100:.1f}%)")
    
    else:
        # Categorical analysis
        if data.dtype == 'object':
            unique_values = data.nunique()
            print(f"      Unique values: {unique_values:,}")
            if unique_values <= 10:
                value_counts = data.value_counts()
                print(f"      Top values:")
                for val, count in value_counts.head(5).items():
                    pct = count / len(data) * 100
                    val_str = str(val)
                    if len(val_str) > 30:
                        val_str = val_str[:27] + "..."
                    print(f"        {val_str}: {count:,} ({pct:.1f}%)")

def estimate_compatibility(df, found_columns):
    """Estimates dataset compatibility with polymer pipeline"""
    
    compatibility_score = 0
    max_score = 100
    
    # SMILES/BigSMILES (weight 40)
    if 'SMILES' in found_columns or 'BigSMILES' in found_columns:
        compatibility_score += 40
    elif 'Monomer' in found_columns:
        compatibility_score += 20  # Can be converted
    
    # Target properties (weight 40)
    target_props = ['Tc', 'Tg', 'Density']
    available_props = sum(1 for prop in target_props if prop in found_columns)
    compatibility_score += (available_props / len(target_props)) * 40
    
    # Auxiliary data (weight 20)
    if 'MW' in found_columns:
        compatibility_score += 10
    if 'Polymer' in found_columns:
        compatibility_score += 10
    
    return compatibility_score

def analyze_consolidated_data(datasets):
    """Consolidated analysis of all datasets"""
    
    total_samples = 0
    total_properties = set()
    smiles_sources = []
    property_sources = {}
    
    print(f"ğŸ“Š GENERAL SUMMARY:")
    
    for key, dataset_info in datasets.items():
        data = dataset_info['data']
        
        if isinstance(data, dict):  # Excel with multiple sheets
            for sheet_name, df in data.items():
                if not df.empty:
                    total_samples += len(df)
                    total_properties.update(df.columns)
                    
                    # Check SMILES/BigSMILES
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(x in col_lower for x in ['smiles', 'bigsmiles']):
                            smiles_sources.append(f"{key}[{sheet_name}].{col}")
                        
                        # Check properties
                        if any(x in col_lower for x in ['tg', 'tc', 'density']):
                            prop_key = f"{key}[{sheet_name}].{col}"
                            if col not in property_sources:
                                property_sources[col] = []
                            property_sources[col].append(prop_key)
        else:  # CSV
            if not data.empty:
                total_samples += len(data)
                total_properties.update(data.columns)
                
                # Check SMILES/BigSMILES
                for col in data.columns:
                    col_lower = col.lower()
                    if any(x in col_lower for x in ['smiles', 'bigsmiles']):
                        smiles_sources.append(f"{key}.{col}")
                    
                    # Check properties
                    if any(x in col_lower for x in ['tg', 'tc', 'density']):
                        if col not in property_sources:
                            property_sources[col] = []
                        property_sources[col].append(f"{key}.{col}")
    
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total unique columns: {len(total_properties)}")
    
    print(f"\nğŸ§¬ SMILES SOURCES:")
    if smiles_sources:
        for source in smiles_sources:
            print(f"  âœ… {source}")
    else:
        print(f"  â�Œ No SMILES sources found")
    
    print(f"\nğŸ�¯ AVAILABLE PROPERTIES:")
    if property_sources:
        for prop, sources in property_sources.items():
            print(f"  ğŸ“Š {prop}:")
            for source in sources:
                print(f"    - {source}")
    else:
        print(f"  â�Œ No target properties found")
    
    # Integration strategy
    print(f"\nğŸ”§ INTEGRATION STRATEGY:")
    
    if smiles_sources and property_sources:
        print(f"  âœ… COMPATIBLE DATA - Integration recommended!")
        print(f"  ğŸ“‹ Suggested steps:")
        print(f"    1. Load SMILES data from: {smiles_sources[0]}")
        print(f"    2. Load properties from: {list(property_sources.keys())}")
        print(f"    3. Merge based on structure/name")
        print(f"    4. Apply premium filters")
        print(f"    5. Integrate into main pipeline")
        
        # Impact estimation
        print(f"\nğŸ“ˆ ESTIMATED IMPACT:")
        if total_samples >= 500:
            impact = "High (0.015-0.030 points)"
        elif total_samples >= 200:
            impact = "Medium (0.008-0.020 points)"
        elif total_samples >= 50:
            impact = "Low (0.003-0.010 points)"
        else:
            impact = "Minimal (0.001-0.005 points)"
        
        print(f"  Expected CV improvement: {impact}")
        
    elif smiles_sources:
        print(f"  âš ï¸� SMILES ONLY - Limited integration")
        print(f"  ğŸ’¡ Use to increase structural diversity")
        
    elif property_sources:
        print(f"  âš ï¸� PROPERTIES ONLY - Limited integration")
        print(f"  ğŸ’¡ Use for cross-validation")
        
    else:
        print(f"  â�Œ INCOMPATIBLE DATA")
        print(f"  ğŸ’¡ Consider only optimized pipeline")
    
    # Final recommendation
    print(f"\nğŸ�¯ FINAL RECOMMENDATION:")
    
    if smiles_sources and property_sources:
        print(f"  ğŸ�‰ PROCEED WITH INTEGRATION")
        print(f"  ğŸ“Š Sufficient data for significant improvement")
        print(f"  ğŸš€ Combine with enhanced pipeline for maximum impact")
    else:
        print(f"  âš ï¸� INTEGRATION NOT RECOMMENDED")
        print(f"  âœ… Keep current enhanced pipeline (14.1% improvement)")
        print(f"  ğŸ’¡ Focus on hyperparameter optimization")

# Execute preview
if __name__ == "__main__":
    result = preview_smiles_extra_dataset()
    
    if result:
        # Count valid datasets
        valid_datasets = sum(1 for v in result.values() if v['data'] is not None)
        total_sheets = 0
        for v in result.values():
            if v['sheets']:
                total_sheets += len(v['sheets'])
        
        print(f"\nğŸ“Š Datasets processed: {valid_datasets} | Sheets analyzed: {total_sheets}")
    else:
        print("â�Œ Preview failed")


import re

# Silence warnings
warnings.filterwarnings("ignore")
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

def load_competition_data():
    """Load NeurIPS Open Polymer Prediction 2025 competition data"""
    try:
        # Correct paths for NeurIPS Open Polymer Prediction 2025
        train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
        test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
        
        print(f"NeurIPS 2025 competition data loaded:")
        print(f"  Training: {train_df.shape[0]:,} samples")
        print(f"  Test: {test_df.shape[0]:,} samples")
        
        # Optional: Load supplementary datasets
        supplement_datasets = []
        for i in range(1, 5):
            try:
                supp_df = pd.read_csv(f'/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset{i}.csv')
                supplement_datasets.append(supp_df)
                print(f"  Supplement dataset{i}: {supp_df.shape[0]:,} samples")
            except FileNotFoundError:
                continue
        
        if supplement_datasets:
            print(f"  Total supplement datasets: {len(supplement_datasets)}")
        
        return train_df, test_df
        
    except FileNotFoundError:
        print("NeurIPS 2025 competition data not found, creating example data...")
        
        # Create realistic example data for testing only
        np.random.seed(42)
        n_train, n_test = 1000, 500
        
        train_smiles = [f"CCc1ccc(C)cc1{'O' if i % 2 == 0 else 'N'}C(=O){'C' * (i % 3 + 1)}" for i in range(n_train)]
        test_smiles = [f"CCc1ccc(C)cc1{'O' if i % 2 == 0 else 'N'}C(=O){'C' * (i % 3 + 1)}" for i in range(n_test)]
        
        train_df = pd.DataFrame({
            'id': range(n_train),
            'SMILES': train_smiles,
            'Tg': np.random.normal(300, 50, n_train),
            'FFV': np.random.normal(0.15, 0.05, n_train),
            'Tc': np.random.normal(0.25, 0.1, n_train), 
            'Density': np.random.normal(1.2, 0.3, n_train),
            'Rg': np.random.normal(15, 5, n_train)
        })
        
        test_df = pd.DataFrame({
            'id': range(n_test),
            'SMILES': test_smiles
        })
        
        print(f"Example data created (for testing only):")
        print(f"  Training: {train_df.shape[0]:,} samples")
        print(f"  Test: {test_df.shape[0]:,} samples")
        print("WARNING: Using synthetic data - results not valid for submission")
        
        return train_df, test_df

def clean_polymer_smiles(smiles):
    """Clean and normalize SMILES for better parsing"""
    if not isinstance(smiles, str) or not smiles.strip():
        return "C"

    s = smiles.strip()
    s = re.sub(r"\s+", "", s)

    # Replace wildcards with carbon
    s = re.sub(r"\[(?:R\d*|R'{0,2})\]", "C", s)
    s = s.replace("*", "C")

    # Fix nitro groups
    s = s.replace("[N+](=O)[O-]", "N(=O)=O")
    s = s.replace("[N+](=O)O-", "N(=O)=O")
    s = s.replace("N(=O)[O-]", "N(=O)=O")

    # Remove empty parentheses
    while "()" in s:
        s = s.replace("()", "")

    # Keep largest fragment
    if "." in s:
        parts = [p for p in s.split(".") if p]
        if parts:
            s = max(parts, key=len)

    # Balance brackets
    diff = s.count("[") - s.count("]")
    if diff > 0:
        s = s + ("]" * diff)

    s = s.strip("()[]")
    return s or "C"

class BalancedFeatureExtractor:
    """Balanced feature extractor - moderate complexity"""
    
    def __init__(self):
        self.feature_names = []
        self.embeddings_available = False
        
    def extract_features(self, smiles_list, use_embeddings=True, verbose=True):
        """Extract balanced features for polymer prediction"""
        if verbose:
            print("Balanced feature engineering...")
        
        # Clean SMILES
        cleaned_smiles = [clean_polymer_smiles(s) for s in smiles_list]
        
        # Extract core features
        base_features = self._extract_core_features(cleaned_smiles)
        if verbose:
            print(f"Core features: {base_features.shape[1]}")
        
        # Extract moderate embeddings (128-bit instead of 256)
        embeddings = None
        if use_embeddings:
            try:
                embeddings = self._extract_compact_embeddings(cleaned_smiles)
                if embeddings is not None:
                    if verbose:
                        print(f"Compact embeddings: {embeddings.shape[1]} dimensions")
                    self.embeddings_available = True
            except Exception:
                pass
        
        # Combine features
        feature_components = [base_features.values]
        feature_names = [f'base_{i}' for i in range(base_features.shape[1])]
        
        if embeddings is not None:
            feature_components.append(embeddings)
            feature_names.extend([f'emb_{i}' for i in range(embeddings.shape[1])])
        
        combined_features = np.concatenate(feature_components, axis=1)
        feature_df = pd.DataFrame(combined_features, columns=feature_names)
        
        self.feature_names = list(feature_df.columns)
        if verbose:
            total_features = feature_df.shape[1]
            complexity = "moderate" if total_features < 200 else "high"
            print(f"Total features: {total_features} ({complexity} complexity)")
            
        return feature_df
    
    def _extract_core_features(self, smiles_list):
        """Extract core set of molecular descriptors (40 features)"""
        features = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0 and i > 0:
                print(f"    Processing {i+1}/{len(smiles_list)}...")
                
            try:
                mol_features = self._extract_rdkit_features(smiles)
            except:
                mol_features = self._extract_smiles_features(smiles)
            
            features.append(mol_features)
        
        feature_names = [f'feature_{i}' for i in range(len(features[0]))]
        feature_df = pd.DataFrame(features, columns=feature_names)
        
        # Clean data
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.median())
        
        return feature_df
    
    def _extract_rdkit_features(self, smiles):
        """Extract balanced set of RDKit molecular descriptors"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            # Balanced critical features (40 total)
            features = [
                # Basic molecular properties (12)
                Descriptors.MolWt(mol), Descriptors.NumHeavyAtoms(mol),
                Descriptors.NumAtoms(mol), Descriptors.BertzCT(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.NumRigidBonds(mol),
                Descriptors.FractionCsp3(mol), Descriptors.NumAliphaticCarbocycles(mol),
                Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumSaturatedCarbocycles(mol),
                Descriptors.RingCount(mol), Descriptors.NumHeteroatoms(mol),
                
                # Surface and electronic properties (10)
                Descriptors.TPSA(mol), Descriptors.LabuteASA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Crippen.MolLogP(mol), Crippen.MolMR(mol),
                Descriptors.BalabanJ(mol), Descriptors.Chi0(mol),
                
                # Ring and aromatic features (6)
                Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
                Descriptors.NumSaturatedRings(mol), Descriptors.Chi1(mol),
                Descriptors.Kappa1(mol), Descriptors.MaxAbsEStateIndex(mol),
                
                # Functional group counts (8)
                smiles.count('C(=O)'), smiles.count('OH'), smiles.count('NH'),
                smiles.count('C=C'), smiles.count('c1ccccc1'), smiles.count('F'),
                smiles.count('Cl'), smiles.upper().count('COO'),
                
                # Derived features (4)
                len(smiles) / max(1, Descriptors.NumHeavyAtoms(mol)),
                Descriptors.NumAromaticRings(mol) / max(1, Descriptors.RingCount(mol)),
                Descriptors.NumHeteroatoms(mol) / max(1, Descriptors.NumHeavyAtoms(mol)),
                Descriptors.TPSA(mol) / max(1, Descriptors.MolWt(mol))
            ]
            
            return features
            
        except:
            raise ValueError("RDKit processing failed")
    
    def _extract_smiles_features(self, smiles):
        """Fallback SMILES string features (40 total)"""
        features = [
            len(smiles), smiles.count('C'), smiles.count('c'),
            smiles.count('O'), smiles.count('N'), smiles.count('S'),
            smiles.count('F'), smiles.count('Cl'), smiles.count('('),
            smiles.count('['), smiles.count('='), smiles.count('#'),
            smiles.count('1'), smiles.count('2'), smiles.count('C=C'),
            smiles.count('C=O'), smiles.count('OH'), smiles.count('NH'),
            smiles.count('c1ccccc1'), smiles.count('CF'), smiles.count('CCl'),
            smiles.upper().count('COO'), smiles.upper().count('SO'),
            smiles.count('C') + smiles.count('c'), smiles.count('Br'),
            smiles.count('I'), smiles.count('P'), smiles.count('Si'),
            smiles.count('c') / max(1, smiles.count('C') + smiles.count('c')),
            smiles.count('=') / max(1, len(smiles)),
            (smiles.count('O') + smiles.count('N')) / max(1, len(smiles)),
            smiles.count('F') / max(1, len(smiles)),
            sum(1 for c in smiles if c.isupper()) / max(1, len(smiles)),
            sum(1 for c in smiles if c.isdigit()) / max(1, len(smiles)),
            len(set(smiles)) / max(1, len(smiles)),
            smiles.count('(') / max(1, len(smiles)),
            (smiles.count('C(=O)') + smiles.count('OH')) / max(1, len(smiles)),
            0, 0, 0, 0  # Padding to 40 features
        ]
        
        return features[:40]  # Ensure exactly 40 features
    
    def _extract_compact_embeddings(self, smiles_list):
        """Extract compact Morgan fingerprint embeddings (128-bit)"""
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import rdFingerprintGenerator
            
            # Reduced size embeddings for better generalization
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
            
            embeddings = []
            for smiles in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        embeddings.append(np.zeros(128, dtype=np.int8))
                        continue
                    
                    fp = gen.GetFingerprint(mol)
                    arr = np.zeros(128, dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    embeddings.append(arr)
                    
                except Exception:
                    embeddings.append(np.zeros(128, dtype=np.int8))
            
            return np.array(embeddings)
            
        except Exception:
            return None

class BalancedEnsemble:
    """Balanced ensemble - 3 models per property with moderate regularization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.meta_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.is_fitted = False
        
        # Competition weights
        self.competition_weights = {
            'Tg': 0.0007, 'FFV': 0.0555, 'Tc': 0.6095, 
            'Density': 0.3203, 'Rg': 0.0140
        }
        
        self._define_models()
    
    def _define_models(self):
        """Define balanced model configurations - 3 models per property"""
        
        # Enhanced models for high-weight properties (Tc, Density)
        enhanced_models = {
            'rf_enhanced': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=3,
                min_samples_leaf=2, random_state=self.random_state
            ),
            'gb_enhanced': GradientBoostingRegressor(
                n_estimators=150, max_depth=8, learning_rate=0.08,
                subsample=0.85, random_state=self.random_state
            ),
            'ridge_enhanced': Ridge(alpha=0.5, random_state=self.random_state)
        }
        
        # Standard models for medium/low-weight properties
        standard_models = {
            'rf_standard': RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_split=4,
                min_samples_leaf=2, random_state=self.random_state
            ),
            'et_standard': ExtraTreesRegressor(
                n_estimators=120, max_depth=12, min_samples_split=4,
                min_samples_leaf=2, random_state=self.random_state
            ),
            'ridge_standard': Ridge(alpha=1.0, random_state=self.random_state)
        }
        
        # Assign models by property weight
        self.property_models = {
            'Tc': enhanced_models.copy(),        # Highest weight: 60.95%
            'Density': enhanced_models.copy(),   # High weight: 32.03%
            'FFV': standard_models.copy(),       # Medium weight: 5.55%
            'Rg': standard_models.copy(),        # Low weight: 1.40%
            'Tg': standard_models.copy()         # Lowest weight: 0.07%
        }
    
    def cross_validate(self, X, y, cv_folds=5, verbose=True):
        """Perform cross-validation"""
        if verbose:
            print(f"Balanced Cross-validation ({cv_folds} folds)...")
        
        X_array = X.values if hasattr(X, 'values') else X
        y_reset = y.reset_index(drop=True)
        
        # Stratified split based on Tc (most important property)
        try:
            tc_values = y_reset['Tc'].fillna(y_reset['Tc'].median())
            quartiles = pd.qcut(tc_values, q=4, labels=False, duplicates='drop')
            if len(np.unique(quartiles)) >= 2:
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                splits = list(kf.split(X_array, quartiles))
            else:
                raise ValueError("Fallback to KFold")
        except:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X_array))
        
        cv_scores = []
        property_scores = {prop: [] for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']}
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            if verbose:
                print(f"  Fold {fold + 1}/{cv_folds}...", end="")
            
            try:
                X_train_fold = X_array[train_idx]
                X_val_fold = X_array[val_idx]
                y_train_fold = y_reset.iloc[train_idx].reset_index(drop=True)
                y_val_fold = y_reset.iloc[val_idx].reset_index(drop=True)
                
                if hasattr(X, 'columns'):
                    X_train_fold = pd.DataFrame(X_train_fold, columns=X.columns)
                    X_val_fold = pd.DataFrame(X_val_fold, columns=X.columns)
                
                # Train ensemble
                fold_ensemble = BalancedEnsemble(random_state=self.random_state)
                fold_ensemble.fit(X_train_fold, y_train_fold, verbose=False)
                
                # Predict
                y_pred_fold = fold_ensemble.predict(X_val_fold, verbose=False)
                
                # Calculate wMAE
                wmae_score = self.calculate_wmae_score(y_val_fold, y_pred_fold)
                cv_scores.append(wmae_score)
                
                # MAE per property
                for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
                    if prop in y_val_fold.columns and prop in y_pred_fold.columns:
                        mask = ~y_val_fold[prop].isna()
                        if mask.sum() > 0:
                            mae = mean_absolute_error(y_val_fold[prop][mask], y_pred_fold[prop][mask])
                            property_scores[prop].append(mae)
                
                if verbose:
                    print(f" wMAE: {wmae_score:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f" ERROR: {e}")
                cv_scores.append(1.0)
                continue
        
        if verbose:
            print(f"Balanced CV Score: {np.mean(cv_scores):.4f} Â± {np.std(cv_scores):.4f}")
            
            print(f"Performance per property:")
            for prop, scores in property_scores.items():
                if scores:
                    prop_mean = np.mean(scores)
                    weight = self.competition_weights[prop]
                    contribution = prop_mean * weight
                    print(f"  {prop}: {prop_mean:.4f} (weight: {weight:.1%}, contribution: {contribution:.4f})")
        
        return cv_scores
    
    def fit(self, X, y, verbose=True):
        """Train the balanced ensemble"""
        if verbose:
            print("Training Balanced Ensemble...")
            print(f"  Data: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        X_array = X.values if hasattr(X, 'values') else X
        y_reset = y.reset_index(drop=True)
        
        for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
            if prop not in y_reset.columns:
                continue
                
            mask = ~y_reset[prop].isna()
            n_valid = mask.sum()
            
            if verbose:
                weight = self.competition_weights[prop]
                model_type = "enhanced" if prop in ['Tc', 'Density'] else "standard"
                print(f"  {prop}: {n_valid:,} samples, weight: {weight:.1%} ({model_type})")
            
            if n_valid < 20:
                if verbose:
                    print(f"    Insufficient data for {prop}, skipping...")
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if hasattr(X, 'columns'):
                        X_prop = pd.DataFrame(X_array[mask], columns=X.columns)
                    else:
                        X_prop = X_array[mask]
                        
                    y_prop = y_reset[prop][mask].reset_index(drop=True)
                    
                    # Balanced feature selection
                    if prop in ['Tc', 'Density']:  # High weight properties
                        n_features = min(150, X_prop.shape[1])
                    else:  # Lower weight properties
                        n_features = min(100, X_prop.shape[1])
                    
                    selector = SelectKBest(f_regression, k=n_features)
                    X_prop_selected = selector.fit_transform(X_prop, y_prop)
                    
                    # Standard scaling
                    scaler = RobustScaler()
                    X_prop_scaled = scaler.fit_transform(X_prop_selected)
                    
                    # Train base models
                    if verbose:
                        model_count = len(self.property_models[prop])
                        print(f"    Training {model_count} models...")
                    
                    prop_models = {}
                    model_configs = self.property_models[prop]
                    
                    for model_name, model_config in model_configs.items():
                        try:
                            if hasattr(model_config, 'get_params'):
                                model = model_config.__class__(**model_config.get_params())
                            else:
                                model = model_config
                                
                            model.fit(X_prop_scaled, y_prop)
                            prop_models[model_name] = model
                            if verbose:
                                print(f"      {model_name}")
                        except Exception:
                            continue
                    
                    if not prop_models:
                        continue
                    
                    # Balanced meta-learning
                    if verbose:
                        print(f"    Balanced meta-learning...")
                    
                    # 4 folds for balance between overfitting and stability
                    kf = KFold(n_splits=4, shuffle=True, random_state=self.random_state)
                    meta_features = np.zeros((len(X_prop_scaled), len(prop_models)))
                    
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_prop_scaled)):
                        X_train_fold = X_prop_scaled[train_idx]
                        X_val_fold = X_prop_scaled[val_idx]
                        y_train_fold = y_prop.iloc[train_idx] if hasattr(y_prop, 'iloc') else y_prop[train_idx]
                        
                        for model_idx, (model_name, _) in enumerate(prop_models.items()):
                            try:
                                model_config = model_configs[model_name]
                                if hasattr(model_config, 'get_params'):
                                    fold_model = model_config.__class__(**model_config.get_params())
                                else:
                                    fold_model = model_config
                                    
                                fold_model.fit(X_train_fold, y_train_fold)
                                val_pred = fold_model.predict(X_val_fold)
                                meta_features[val_idx, model_idx] = val_pred
                                
                            except Exception:
                                if model_idx > 0:
                                    meta_features[val_idx, model_idx] = np.mean(meta_features[val_idx, :model_idx])
                                continue
                    
                    # Meta-model with moderate regularization
                    if prop in ['Tc', 'Density']:
                        meta_alpha = 0.8  # Lower regularization for important properties
                    else:
                        meta_alpha = 1.5  # Higher regularization for less important properties
                        
                    meta_model = Ridge(alpha=meta_alpha)
                    meta_model.fit(meta_features, y_prop)
                    
                    # Save everything
                    self.models[prop] = prop_models
                    self.meta_models[prop] = meta_model
                    self.scalers[prop] = scaler
                    self.feature_selectors[prop] = selector
                    
                    if verbose:
                        print(f"    Balanced stacking: {len(prop_models)} base + 1 meta (Î±={meta_alpha})")
                        
            except Exception as e:
                if verbose:
                    print(f"    Error training {prop}: {e}")
                continue
        
        self.is_fitted = True
        if verbose:
            enhanced_props = len([p for p in self.models.keys() if p in ['Tc', 'Density']])
            standard_props = len(self.models) - enhanced_props
            print(f"Balanced Ensemble complete: {enhanced_props} enhanced + {standard_props} standard properties")
        
        return self
    
    def predict(self, X, verbose=True):
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be trained first")
        
        if verbose:
            print("Balanced Predictions...")
        
        predictions = {}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
                if prop in self.models:
                    try:
                        X_array = X.values if hasattr(X, 'values') else X
                        X_selected = self.feature_selectors[prop].transform(X_array)
                        X_scaled = self.scalers[prop].transform(X_selected)
                        
                        # Base predictions
                        base_predictions = np.zeros((len(X_scaled), len(self.models[prop])))
                        
                        for model_idx, (model_name, model) in enumerate(self.models[prop].items()):
                            try:
                                pred = model.predict(X_scaled)
                                base_predictions[:, model_idx] = pred
                            except Exception:
                                if model_idx > 0:
                                    base_predictions[:, model_idx] = np.mean(base_predictions[:, :model_idx], axis=1)
                                else:
                                    base_predictions[:, model_idx] = 0
                        
                        # Meta-model prediction
                        final_pred = self.meta_models[prop].predict(base_predictions)
                        predictions[prop] = final_pred
                        
                        if verbose:
                            model_type = "enhanced" if prop in ['Tc', 'Density'] else "standard"
                            print(f"  {prop}: balanced stacking ({model_type})")
                            
                    except Exception:
                        defaults = {'Tg': 25, 'FFV': 0.15, 'Tc': 0.25, 'Density': 1.0, 'Rg': 15.0}
                        predictions[prop] = [defaults[prop]] * len(X)
                        if verbose:
                            print(f"  {prop}: default value")
                else:
                    defaults = {'Tg': 25, 'FFV': 0.15, 'Tc': 0.25, 'Density': 1.0, 'Rg': 15.0}
                    predictions[prop] = [defaults[prop]] * len(X)
                    if verbose:
                        print(f"  {prop}: default value")
        
        return pd.DataFrame(predictions)
    
    def calculate_wmae_score(self, y_true, y_pred):
        """Calculate weighted MAE score"""
        total_wmae = 0
        weight_sum = 0
        
        for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
            if prop in y_true.columns and prop in y_pred.columns:
                mask = ~y_true[prop].isna()
                if mask.sum() > 0:
                    mae = mean_absolute_error(y_true[prop][mask], y_pred[prop][mask])
                    weighted_mae = mae * self.competition_weights[prop]
                    total_wmae += weighted_mae
                    weight_sum += self.competition_weights[prop]
        
        return total_wmae / weight_sum if weight_sum > 0 else float('inf')

def run_balanced_pipeline():
    """Run the balanced pipeline - middle ground approach"""
    
    print("Balanced Polymer Pipeline - Middle Ground Approach")
    print("Target: Optimal balance between complexity and generalization")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading competition data...")
    train_df, test_df = load_competition_data()
    
    # Balanced feature extraction
    print("\n2. Feature extraction...")
    feature_extractor = BalancedFeatureExtractor()
    
    X_train = feature_extractor.extract_features(
        train_df['SMILES'].tolist(), 
        use_embeddings=True,
        verbose=True
    )
    
    X_test = feature_extractor.extract_features(
        test_df['SMILES'].tolist(),
        use_embeddings=feature_extractor.embeddings_available,
        verbose=False
    )
    
    print(f"Features: Training {X_train.shape[1]}, Test {X_test.shape[1]} dimensions")
    
    # Prepare targets
    y_train = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].copy()
    
    # Cross-validation
    print("\n3. Cross-validation...")
    ensemble = BalancedEnsemble(random_state=42)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = ensemble.cross_validate(X_train, y_train, cv_folds=5, verbose=True)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Performance analysis
    baseline_score = 0.0999
    previous_enhanced_score = 0.0912
    previous_test_score = 0.076
    
    improvement_vs_baseline = baseline_score - cv_mean
    improvement_pct = improvement_vs_baseline / baseline_score * 100
    
    print(f"\nResults:")
    print(f"  CV Score: {cv_mean:.4f} Â± {cv_std:.4f}")
    print(f"  vs Baseline: {improvement_vs_baseline:+.4f} ({improvement_pct:+.1f}%)")
    print(f"  vs Enhanced: {previous_enhanced_score - cv_mean:+.4f}")
    print(f"  Target test: < 0.080 (previous: {previous_test_score})")
    
    # Train final model
    print("\n4. Training final model...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(X_train, y_train, verbose=True)
        
    # Generate predictions
    print("\n5. Generating predictions...")
    final_predictions = ensemble.predict(X_test, verbose=True)
    
    # Create submission file
    submission = pd.DataFrame({'id': test_df['id']})
    
    for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        if prop in final_predictions.columns:
            submission[prop] = final_predictions[prop]
        else:
            defaults = {'Tg': 25, 'FFV': 0.15, 'Tc': 0.25, 'Density': 1.0, 'Rg': 15.0}
            submission[prop] = defaults[prop]
    
    # Analyze prediction quality
    print("\n6. Prediction analysis...")
    realistic_predictions = 0
    
    for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        pred_values = submission[prop]
        pred_mean = pred_values.mean()
        pred_std = pred_values.std()
        
        is_realistic = pred_std > 0.001
        if is_realistic:
            realistic_predictions += 1
        
        print(f"  {prop}: {pred_mean:.3f} Â± {pred_std:.3f} {'OK' if is_realistic else 'WARN'}")
    
    # Save submission file
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    
    # Also save backup
    submission.to_csv('balanced_submission_backup.csv', index=False)
    
    print(f"\nFiles saved: {submission_file}")
    print(f"Backup saved: balanced_submission_backup.csv")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"BALANCED PIPELINE COMPLETE")
    print(f"="*60)
    
    complexity_level = f"Moderate - {X_train.shape[1]} features, 3 models per property"
    generalization_expectation = "Better" if cv_mean < previous_enhanced_score else "Similar"
    
    print(f"Complexity: {complexity_level}")
    print(f"CV Performance: {cv_mean:.4f} ({improvement_pct:.1f}% vs baseline)")
    print(f"Realistic predictions: {realistic_predictions}/5")
    print(f"Expected generalization: {generalization_expectation} than enhanced version")
    
    # Recommendation based on CV performance
    if cv_mean < 0.085:
        print("STATUS: Balanced approach successful - submit submission.csv")
        recommendation = "SUBMIT"
    elif cv_mean < 0.095:
        print("STATUS: Reasonable performance - test and compare")
        recommendation = "TEST_COMPARE"
    else:
        print("STATUS: May need further adjustment")
        recommendation = "NEEDS_TUNING"
    
    return {
        'submission': submission,
        'cv_score': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_scores,
        'baseline_score': baseline_score,
        'enhanced_score': previous_enhanced_score,
        'previous_test_score': previous_test_score,
        'improvement_vs_baseline': improvement_vs_baseline,
        'improvement_pct': improvement_pct,
        'realistic_predictions': realistic_predictions,
        'submission_file': submission_file,
        'complexity_level': 'moderate',
        'recommendation': recommendation
    }


if __name__ == "__main__":
    print("Balanced Polymer Property Prediction Pipeline")
    print("="*80)
    
    results = run_balanced_pipeline()
    
    if results:
        print(f"\nFINAL RESULT:")
        print(f"CV Score: {results['cv_score']:.4f}")

def quick_test():
    """Quick test of the balanced pipeline"""
    print("Balanced Pipeline Quick Test")
    print("="*40)
    
    try:
        # Test feature extraction
        extractor = BalancedFeatureExtractor()
        test_smiles = ["CCO", "CCc1ccccc1", "CC(=O)O"]
        features = extractor.extract_features(test_smiles, verbose=False)
        print(f"Feature extraction: {features.shape[1]} features (balanced)")
        
        # Test ensemble
        ensemble = BalancedEnsemble()
        enhanced_props = len([p for p in ['Tc', 'Density'] if p in ensemble.property_models])
        standard_props = len(ensemble.property_models) - enhanced_props
        print(f"Ensemble: {enhanced_props} enhanced + {standard_props} standard properties")
        
        print("BALANCED TEST PASSED")
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        return False

# Run quick test
quick_test()




