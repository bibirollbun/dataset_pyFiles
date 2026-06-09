!pip install rdkit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Chemical informatics libraries
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG = True

# Machine learning and statistics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

import plotly.io as pio
pio.renderers.default = 'iframe_connected'
print("ğŸ�‰ All libraries imported successfully!")
print("ğŸ“Š Ready for polymer SMILES analysis!")

# ================================================================================================
# ğŸ“� DATA LOADING AND INITIAL EXPLORATION
# ================================================================================================

# Load the datasets
print("ğŸ“‚ Loading datasets...")
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

print(f"âœ… Training data shape: {train_df.shape}")
print(f"âœ… Test data shape: {test_df.shape}")
print(f"âœ… Sample submission shape: {sample_submission.shape}")



print("\n" + "="*80)
print("ğŸ§¬ PART 1: UNDERSTANDING SMILES NOTATION FOR POLYMERS")
print("="*80)

print("""
SMILES (Simplified Molecular Input Line Entry System) Key Concepts:

ğŸ”¤ Basic Elements:
- C, N, O, S, P, F, Cl, Br, I = atoms
- () = branches
- [] = atom properties (charge, isotope)
- = triple bond, = double bond, - single bond (usually omitted)
- @ = chirality

ğŸ”— Polymer-Specific SMILES:
- *-C-C-* pattern indicates repeating units
- [*] often represents connection points
- Long chains represent polymer backbones
- Side chains shown as branches ()

Let's examine some examples from our dataset:
""")

# Display first few SMILES examples
print("ğŸ“‹ Sample SMILES from our dataset:")
for i in range(5):
    smiles = train_df.iloc[i]['SMILES']
    print(f"{i+1:2d}. {smiles}")


print("\n" + "="*80)
print("ğŸ”� PART 2: BASIC DATA EXPLORATION")
print("="*80)

# Basic dataset information
print("ğŸ“Š Dataset Overview:")
print(f"â€¢ Training samples: {len(train_df):,}")
print(f"â€¢ Test samples: {len(test_df):,}")
print(f"â€¢ Features in training: {train_df.shape[1]}")

# Target variables overview
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
print(f"\nğŸ�¯ Target Variables:")
for col in target_cols:
    print(f"â€¢ {col}: {train_df[col].dtype}")

# Check for missing values
print("\nâ�“ Missing Values Check:")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("Training set:")
for col, missing in missing_train.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(train_df)*100:.1f}%)")
    
if missing_train.sum() == 0:
    print("  âœ… No missing values in training set!")

print("Test set:")
if missing_test.sum() == 0:
    print("  âœ… No missing values in test set!")

# SMILES string basic statistics
print("\nğŸ“� SMILES String Statistics:")
train_smiles_stats = train_df['SMILES'].str.len().describe()
test_smiles_stats = test_df['SMILES'].str.len().describe()

stats_df = pd.DataFrame({
    'Training Set': train_smiles_stats,
    'Test Set': test_smiles_stats
})
print(stats_df.round(2))

# Check for duplicate SMILES
train_duplicates = train_df['SMILES'].duplicated().sum()
test_duplicates = test_df['SMILES'].duplicated().sum()
print(f"\nğŸ”„ Duplicate SMILES:")
print(f"â€¢ Training set: {train_duplicates} duplicates")
print(f"â€¢ Test set: {test_duplicates} duplicates")


print("\n" + "="*80)
print("ğŸ�¨ PART 3: TARGET VARIABLES ANALYSIS")
print("="*80)

# Create comprehensive target variable analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ğŸ�¯ Target Variables Distribution Analysis', fontsize=16, fontweight='bold')

# Define target info for better visualization
target_info = {
    'Tg': {'unit': 'Â°C', 'name': 'Glass Transition Temperature', 'color': 'red'},
    'FFV': {'unit': '', 'name': 'Fractional Free Volume', 'color': 'blue'},
    'Tc': {'unit': 'W/mÂ·K', 'name': 'Thermal Conductivity', 'color': 'green'},
    'Density': {'unit': 'gÂ·cmâ�»Â³', 'name': 'Polymer Density', 'color': 'orange'},
    'Rg': {'unit': 'Ã…', 'name': 'Radius of Gyration', 'color': 'purple'}
}

for idx, col in enumerate(target_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Histogram with KDE
    train_df[col].hist(bins=50, alpha=0.7, ax=ax, color=target_info[col]['color'], density=True)
    train_df[col].plot.kde(ax=ax, color='black', linewidth=2)
    
    ax.set_title(f'{target_info[col]["name"]}\n({col})', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{col} ({target_info[col]["unit"]})')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    ax.text(0.02, 0.98, f'Î¼ = {mean_val:.2f}\nÏƒ = {std_val:.2f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.show()

# Statistical summary of targets
print("ğŸ“ˆ Target Variables Statistical Summary:")
target_summary = train_df[target_cols].describe()
print(target_summary.round(3))

# Check for outliers using IQR method
print("\nğŸš¨ Outlier Analysis (using IQR method):")
for col in target_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((train_df[col] < lower_bound) | (train_df[col] > upper_bound)).sum()
    print(f"â€¢ {col}: {outliers} outliers ({outliers/len(train_df)*100:.1f}%)")

# Correlation analysis between targets
print("\nğŸ”— Target Variables Correlation Analysis:")
correlation_matrix = train_df[target_cols].corr()
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('ğŸ”— Target Variables Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("ğŸ“Š PART 6: ADVANCED CHEMICAL SPACE ANALYSIS")
print("="*80)

# Molecular fingerprints for chemical similarity analysis
print("ğŸ”¬ Generating molecular fingerprints for chemical space analysis...")

def generate_fingerprints(smiles_list, radius=2, nBits=2048):
    """
    Generate Morgan (ECFP) fingerprints for a list of SMILES.
    
    Morgan fingerprints are circular fingerprints that capture local
    molecular environments, making them excellent for similarity analysis.
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                # Convert to numpy array
                arr = np.zeros((nBits,))
                for bit in fp.GetOnBits():
                    arr[bit] = 1
                fingerprints.append(arr)
                valid_indices.append(idx)
        except:
            continue
    
    return np.array(fingerprints), valid_indices

# Generate fingerprints for training set
fp_array, valid_indices = generate_fingerprints(train_df['SMILES'].tolist())
print(f"âœ… Generated fingerprints for {len(valid_indices)} molecules")

# Dimensionality reduction for visualization
print("ğŸ“‰ Performing dimensionality reduction for visualization...")

# PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(fp_array)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# t-SNE for better cluster visualization
print("ğŸ�¨ Computing t-SNE embedding (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_coords = tsne.fit_transform(fp_array[:5000])  # Limit for speed

# Chemical space visualization
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('PCA of Chemical Space', 't-SNE of Chemical Space'))

# PCA plot
fig.add_trace(
    go.Scatter(x=pca_coords[:, 0], y=pca_coords[:, 1],
               mode='markers', name='Polymers',
               marker=dict(size=4, opacity=0.6, color='blue')),
    row=1, col=1
)

# t-SNE plot
fig.add_trace(
    go.Scatter(x=tsne_coords[:, 0], y=tsne_coords[:, 1],
               mode='markers', name='Polymers (subset)',
               marker=dict(size=4, opacity=0.6, color='red')),
    row=1, col=2
)

fig.update_layout(height=600, title_text="ğŸŒŒ Chemical Space Visualization")
fig.show()

# Clustering analysis
print("\nğŸ�¯ Chemical Space Clustering Analysis")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(fp_array)

# Add cluster information to training data
train_clustered = train_df.iloc[valid_indices].copy()
train_clustered['cluster'] = clusters

# Analyze cluster properties
print(f"ğŸ“Š Cluster Analysis ({n_clusters} clusters):")
for i in range(n_clusters):
    cluster_size = (clusters == i).sum()
    print(f"Cluster {i}: {cluster_size} molecules ({cluster_size/len(clusters)*100:.1f}%)")

# Visualize clusters with target property means
cluster_stats = train_clustered.groupby('cluster')[target_cols].mean()
print("\nğŸ“ˆ Average Target Properties by Cluster:")
print(cluster_stats.round(3))

# Heatmap of cluster properties
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_stats.T, annot=True, cmap='viridis', 
            cbar_kws={'label': 'Average Value'})
plt.title('ğŸ�¯ Average Target Properties by Chemical Cluster', fontweight='bold')
plt.xlabel('Cluster')
plt.ylabel('Target Property')
plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("ğŸ”§ PART 7: POLYMER-SPECIFIC ANALYSIS")
print("="*80)

def analyze_polymer_features(smiles):
    """
    Analyze polymer-specific structural features from SMILES.
    
    This function extracts features that are particularly relevant
    for polymer materials science.
    """
    features = {}
    
    # Basic polymer indicators
    features['has_polymer_notation'] = '*' in smiles
    features['smiles_length'] = len(smiles)
    
    # Backbone analysis
    features['carbon_count'] = smiles.count('C')
    features['carbon_ratio'] = smiles.count('C') / len(smiles) if len(smiles) > 0 else 0
    
    # Branching analysis
    features['branch_count'] = smiles.count('(')
    features['branch_ratio'] = smiles.count('(') / len(smiles) if len(smiles) > 0 else 0
    
    # Functional group analysis
    features['has_aromatic'] = 'c' in smiles  # lowercase c indicates aromatic carbon
    features['has_nitrogen'] = 'N' in smiles
    features['has_oxygen'] = 'O' in smiles
    features['has_sulfur'] = 'S' in smiles
    features['has_fluorine'] = 'F' in smiles
    
    # Bond types
    features['double_bond_count'] = smiles.count('=')
    features['triple_bond_count'] = smiles.count('#')
    
    # Ring structures
    features['ring_indicators'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
    
    return features

print("ğŸ”� Analyzing polymer-specific features...")
polymer_features = []
for smiles in train_df['SMILES']:
    features = analyze_polymer_features(smiles)
    polymer_features.append(features)

polymer_features_df = pd.DataFrame(polymer_features)
print("âœ… Polymer features analysis complete!")

# Polymer feature statistics
print("\nğŸ“Š Polymer Feature Statistics:")
print(polymer_features_df.describe().round(3))

# Visualize key polymer features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ğŸ”§ Polymer-Specific Features Analysis', fontsize=16, fontweight='bold')

polymer_viz_features = ['smiles_length', 'carbon_count', 'branch_count', 
                       'double_bond_count', 'carbon_ratio', 'branch_ratio']

for idx, feature in enumerate(polymer_viz_features):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    polymer_features_df[feature].hist(bins=30, alpha=0.7, ax=ax)
    ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Functional group analysis
functional_groups = ['has_aromatic', 'has_nitrogen', 'has_oxygen', 
                    'has_sulfur', 'has_fluorine']

print("\nğŸ§ª Functional Group Prevalence:")
for fg in functional_groups:
    count = polymer_features_df[fg].sum()
    percentage = count / len(polymer_features_df) * 100
    print(f"â€¢ {fg.replace('has_', '').title()}: {count:,} ({percentage:.1f}%)")

# Functional group impact on properties
print("\nğŸ�¯ Functional Group Impact on Target Properties:")
combined_df = pd.concat([train_df, polymer_features_df], axis=1)

for fg in functional_groups:
    print(f"\n{fg.replace('has_', '').title()}:")
    with_fg = combined_df[combined_df[fg] == True][target_cols].mean()
    without_fg = combined_df[combined_df[fg] == False][target_cols].mean()
    
    for target in target_cols:
        diff = with_fg[target] - without_fg[target]
        print(f"  {target}: {diff:+.3f}")





