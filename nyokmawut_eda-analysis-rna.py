# Install PyTorch Geometric and dependencies
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-geometric




!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html



# Install additional visualization packages
!pip install matplotlib missingno seaborn plotly networkx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


# Load and visualize initial data
df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')

print("\n=== Initial Data Preview ===")
print(df.head())

print("\n=== Data Statistics ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

# Missing data visualization
plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title('Missing Value Matrix', fontsize=16)
plt.show()

# Distribution of coordinates (non-missing values)
coord_cols = ['x_1', 'y_1', 'z_1']
df[coord_cols].dropna().hist(bins=50, figsize=(15, 5), layout=(1, 3))
plt.suptitle('Coordinate Distributions', y=1.05)
plt.tight_layout()
plt.show()

# 3D scatter plot of coordinates (sample)
sample_df = df.dropna().sample(n=1000, random_state=42)
fig = px.scatter_3d(sample_df, x='x_1', y='y_1', z='z_1', 
                    color='resname', title='3D RNA Structure (Sample)')
fig.update_layout(scene=dict(aspectmode="cube"))
fig.show()


import pandas as pd
import numpy as np
from pathlib import Path

# Load the dataset
input_path = Path('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
df = pd.read_csv(input_path)

# Define maximum gap length for interpolation
MAX_GAP = 5

def interpolate_group(group):
    grp = group.copy()
    # Identify where any coordinate is missing
    missing_mask = grp[['x_1', 'y_1', 'z_1']].isna().any(axis=1)

    # Identify continuous missing segments
    gaps = []
    current_gap = []
    for idx, miss in zip(grp.index, missing_mask):
        if miss:
            current_gap.append(idx)
        else:
            if current_gap:
                gaps.append(current_gap)
                current_gap = []
    if current_gap:
        gaps.append(current_gap)

    # Fill large gaps with group mean
    mean_vals = grp[['x_1', 'y_1', 'z_1']].mean()
    for gap in gaps:
        if len(gap) > MAX_GAP:
            grp.loc[gap, ['x_1', 'y_1', 'z_1']] = mean_vals.values

    # Interpolate remaining missing values by 'resid'
    grp = grp.set_index('resid')
    grp[['x_1', 'y_1', 'z_1']] = grp[['x_1', 'y_1', 'z_1']].interpolate(
        method='index', limit=MAX_GAP, limit_direction='both'
    )
    return grp.reset_index()

# Apply interpolation per ID (modified to avoid warning)
df_imputed = (
    df.groupby('ID', group_keys=False)
      .apply(lambda x: interpolate_group(x))
      .reset_index(drop=True)
)

# Final fallback: fill any remaining NaNs with global mean
df_imputed[['x_1', 'y_1', 'z_1']] = df_imputed[['x_1', 'y_1', 'z_1']].fillna(
    df[['x_1', 'y_1', 'z_1']].mean()
)

# Check remaining missing
remaining_na = df_imputed[['x_1', 'y_1', 'z_1']].isna().sum()
print("Remaining missing after final cleaning:", remaining_na.to_dict())

# Save the cleaned dataset to a writable directory
output_path = Path('/kaggle/working/train_labels_cleaned.csv')
df_imputed.to_csv(output_path, index=False)
print("Cleaned data saved to:", output_path)



# Load and visualize initial data
df = pd.read_csv('/kaggle/working/train_labels_cleaned.csv')

print("\n=== Initial Data Preview ===")
print(df.head())

print("\n=== Data Statistics ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

# Missing data visualization
plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title('Missing Value Matrix', fontsize=16)
plt.show()

# Distribution of coordinates (non-missing values)
coord_cols = ['x_1', 'y_1', 'z_1']
df[coord_cols].dropna().hist(bins=50, figsize=(15, 5), layout=(1, 3))
plt.suptitle('Coordinate Distributions', y=1.05)
plt.tight_layout()
plt.show()

# 3D scatter plot of coordinates (sample)
sample_df = df.dropna().sample(n=1000, random_state=42)
fig = px.scatter_3d(sample_df, x='x_1', y='y_1', z='z_1', 
                    color='resname', title='3D RNA Structure (Sample)')
fig.update_layout(scene=dict(aspectmode="cube"))
fig.show()

