!pip install rdkit-pypi
!pip install missingno ipywidgets plotly
!pip install py3Dmol
!pip install ipython

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import missingno as msno
from collections import defaultdict

# LOAD DATA here
df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')


def plot_data_overview(df):
    print(f"Dataset shape: {df.shape}")
    display(df.head())

plot_data_overview(df)


def plot_display(df):
    display(df.describe())
plot_display(df)


def missing(df):
    plt.figure(figsize=(10, 4))
    msno.matrix(df.drop('id', axis=1))
    plt.title('Missing Values Matrix', fontsize=14)
    plt.show()

missing(df)


def plot_molecule_examples(df, n=5):
    mols = []
    legends = []
    
    for i, row in df.head(n).iterrows():
        m = Chem.MolFromSmiles(row['SMILES'])
        if m:
            mols.append(m)
            legends.append(f"ID: {row['id']}")
    
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(300, 200))
    display(img)
plot_molecule_examples(df)


def plot_property_distributions(df):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Histograms
    plt.figure(figsize=(15, 10))
    for i, prop in enumerate(properties, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[prop].dropna(), kde=True)
        plt.title(f'Distribution of {prop}')
        plt.xlabel(prop)
    plt.tight_layout()
    plt.show()
    
plot_property_distributions(df)


def property_value(df):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[properties])
    plt.title('Property Value Ranges')
    plt.xticks(rotation=45)
    plt.show()
    
property_value(df)


def correlation(df):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    corr_df = df.dropna(subset=properties, how='all')[properties]
    if len(corr_df) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Property Correlations')
        plt.show()
correlation(df)


def advanced_polymer_analysis(df):
    if 'Tg' in df.columns and 'FFV' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='FFV', y='Tg')
        plt.title('Tg vs FFV Relationship')
        plt.show()
advanced_polymer_analysis(df)


def advanced_polymer(df):    
    # Molecular weight estimation and property relationship
    df['mol_weight'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms() if Chem.MolFromSmiles(x) else np.nan)
    
    if 'mol_weight' in df.columns:
        for prop in ['Tg', 'FFV', 'Tc']:
            if prop in df.columns:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x='mol_weight', y=prop)
                plt.title(f'{prop} vs Molecular Weight')
                plt.show()
advanced_polymer(df)



def missing_data_co(df):
    msno.heatmap(df.drop('id', axis=1))
    plt.title('Missing Data Correlation', fontsize=14)
    plt.show()

missing_data_co(df)


def missing_data_analysis(df):    
    # Data availability
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    availability = df[properties].notna().mean() * 100
    
    plt.figure(figsize=(10, 4))
    availability.plot(kind='bar')
    plt.title('Data Availability by Property (%)')
    plt.ylabel('Percentage Available')
    plt.ylim(0, 100)
    plt.show()

missing_data_analysis(df)


def plot_polymer_length_analysis(df):
    
    df['polymer_length'] = df['SMILES'].apply(lambda x: len(x.split('.')) if isinstance(x, str) else 0)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='polymer_length', y='Tg', data=df)
    plt.title('Glass Transition Temperature (Tg) vs Polymer Chain Length')
    plt.xlabel('Number of Monomers')
    plt.ylabel('Tg (K)')
    plt.show()

plot_polymer_length_analysis(df)



from rdkit.Chem import Descriptors, Lipinski

def plot_functional_group_analysis(df, sample_size=100):

    sample_df = df.dropna(subset=['SMILES']).sample(min(sample_size, len(df)))
    
    # Calculate functional group properties
    func_groups = {
        'HydrogenDonors': lambda m: Lipinski.NumHDonors(m),
        'HydrogenAcceptors': lambda m: Lipinski.NumHAcceptors(m),
        'RotatableBonds': lambda m: Lipinski.NumRotatableBonds(m),
        'AromaticRings': lambda m: Lipinski.NumAromaticRings(m)
    }
    
    results = defaultdict(list)
    for smiles in sample_df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for name, func in func_groups.items():
                results[name].append(func(mol))
    
    # Plot functional group distributions
    plt.figure(figsize=(15, 10))
    for i, (name, values) in enumerate(results.items(), 1):
        plt.subplot(2, 2, i)
        sns.histplot(values, bins=20, kde=True)
        plt.title(f'Distribution of {name}')
        plt.xlabel('Count')
    plt.tight_layout()
    plt.show()

plot_functional_group_analysis(df)



from rdkit.Chem import AllChem
from ipywidgets import interact
import py3Dmol

def visualize_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        viewer = py3Dmol.view(width=800, height=300)
        viewer.addModel(Chem.MolToMolBlock(mol), 'mol')
        viewer.setStyle({'stick': {}, 'sphere': {'scale':0.25}})
        viewer.zoomTo()
        return viewer.show()
    
# Create interactive widget
if len(df) > 0:
    interact(visualize_3d_molecule, smiles=df['SMILES'].head(10));


def plot_prediction_difficulty(df):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
   
    difficulty = pd.DataFrame({
        'Property': properties,
        'Available Samples': [df[prop].notna().sum() for prop in properties],
        'Value Range': [df[prop].max() - df[prop].min() for prop in properties],
        'StdDev': [df[prop].std() for prop in properties]
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sample size
    sns.barplot(data=difficulty, x='Property', y='Available Samples', ax=axes[0])
    axes[0].set_title('Data Availability')
    
    # Value range
    sns.barplot(data=difficulty, x='Property', y='Value Range', ax=axes[1])
    axes[1].set_title('Value Range')
    
    # Standard deviation
    sns.barplot(data=difficulty, x='Property', y='StdDev', ax=axes[2])
    axes[2].set_title('Value Variability')
    
    plt.tight_layout()
    plt.show()

plot_prediction_difficulty(df)


from collections import Counter

def plot_smiles_character_frequency(df):
    # Analyze SMILES character frequency
    all_smiles = ''.join(df['SMILES'].dropna().values)
    char_counts = Counter(all_smiles)
    
    # Plot top 30 most common characters
    common_chars = char_counts.most_common(30)
    chars, counts = zip(*common_chars)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(chars), y=list(counts))
    plt.title('Top 30 Most Common SMILES Characters')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.show()

plot_smiles_character_frequency(df)


from pandas.plotting import parallel_coordinates

def plot_parallel_coordinates(df, sample_size=100):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_props = [p for p in properties if p in df.columns]
    
    if len(available_props) < 2:
        print("Need at least 2 available properties to plot parallel coordinates")
        return
    
    # Get rows that have at least one property (not all NA)
    valid_rows = df[available_props].dropna(how='all')
    if len(valid_rows) == 0:
        print("No rows with any property values available")
        return
    
    # Sample the data (handle case where sample_size > available rows)
    sample_size = min(sample_size, len(valid_rows))
    sample_df = valid_rows.sample(sample_size)
    
    # Normalize for parallel coordinates
    norm_df = sample_df[available_props].copy()
    norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
    norm_df['ID'] = sample_df.index.astype(str)  # Using index since we sampled from valid_rows
    
    plt.figure(figsize=(15, 8))
    parallel_coordinates(norm_df, 'ID', alpha=0.3)
    plt.title('Parallel Coordinates Plot of Polymer Properties')
    plt.xticks(rotation=45)
    plt.legend().remove()
    plt.show()

plot_parallel_coordinates(df)


import plotly.express as px

def interactive_property_explorer(df):
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_props = [p for p in properties if p in df.columns]
    
    if len(available_props) >= 2:
        fig = px.scatter_matrix(
            df,
            dimensions=available_props,
            color=available_props[0],
            hover_name='id',
            title='Interactive Polymer Property Explorer'
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

interactive_property_explorer(df)  # For Jupyter notebooks

