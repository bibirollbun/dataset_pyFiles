import pandas as pd

train = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
train.head()


train = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")
train.head()


!pip install -q plotly
!pip install -q ipywidgets


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
train = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')

# Merge the datasets
merged_data = pd.merge(train, star_info, on='planet_id')




print("Train data shape:", train.shape)
print("Star info shape:", star_info.shape)
print("\nFirst few rows of merged data:")
display(merged_data.head())



wl_columns = [col for col in train.columns if col.startswith('wl_')]

plt.figure(figsize=(15, 8))
for i in range(5):  # Plot first 5 planets' wavelength data
    plt.plot(range(len(wl_columns)), train.iloc[i][wl_columns], label=f'Planet {train.iloc[i]["planet_id"]}')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.title('Wavelength Distribution for Different Planets')
plt.legend()
plt.grid(True)
plt.show()



@interact(planet_id=widgets.Dropdown(options=train['planet_id'].unique(), description='Select Planet:'))
def plot_wavelength(planet_id):
    planet_data = train[train['planet_id'] == planet_id].iloc[0]
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(wl_columns)), planet_data[wl_columns])
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.title(f'Wavelength Distribution for Planet {planet_id}')
    plt.grid(True)
    plt.show()



fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.scatterplot(data=merged_data, x='Rs', y='Ms', ax=axes[0, 0])
sns.scatterplot(data=merged_data, x='Ts', y='Rs', ax=axes[0, 1])
sns.scatterplot(data=merged_data, x='Ts', y='Ms', ax=axes[0, 2])
sns.scatterplot(data=merged_data, x='P', y='sma', ax=axes[1, 0])
sns.scatterplot(data=merged_data, x='Mp', y='sma', ax=axes[1, 1])
sns.scatterplot(data=merged_data, x='Mp', y='P', ax=axes[1, 2])
plt.tight_layout()
plt.suptitle('Star and Planet Characteristics Relationships', y=1.02)
plt.show()


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Normalize values for better visualization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma']])

ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], 
           c=merged_data['Ts'], cmap='viridis', s=merged_data['Mp']*50)

ax.set_xlabel('Star Radius (Rs)')
ax.set_ylabel('Star Mass (Ms)')
ax.set_zlabel('Star Temperature (Ts)')
ax.set_title('3D Visualization of Star-Planet Systems\n(Color: Temperature, Size: Planet Mass)')
plt.show()



fig = px.scatter_3d(merged_data, x='Rs', y='Ms', z='Ts',
                    color='Mp', size='P',
                    hover_name='planet_id',
                    title='Interactive 3D Visualization of Star-Planet Systems')
fig.show()



pca = PCA(n_components=3)
wl_data = train[wl_columns]
pca_result = pca.fit_transform(wl_data)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=merged_data['Ts'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Star Temperature (Ts)')
plt.title('PCA of Wavelength Data (PC1 vs PC2)')

plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 2], c=merged_data['Mp'], cmap='plasma')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.colorbar(label='Planet Mass (Mp)')
plt.title('PCA of Wavelength Data (PC1 vs PC3)')
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10))
corr = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Star and Planet Characteristics')
plt.show()



fig = px.parallel_coordinates(merged_data, 
                             color='Ts',
                             dimensions=['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma'],
                             labels={'Rs': 'Star Radius', 'Ms': 'Star Mass', 
                                     'Ts': 'Star Temp', 'Mp': 'Planet Mass',
                                     'P': 'Orbital Period', 'sma': 'Semi-major Axis'},
                             title='Parallel Coordinates Plot of System Characteristics')
fig.show()



fig = plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(merged_data['P'], bins=20, kde=True)
plt.title('Orbital Period Distribution')

plt.subplot(1, 3, 2)
sns.scatterplot(data=merged_data, x='P', y='sma', hue='Mp', size='Ms')
plt.title('Orbital Period vs Semi-major Axis')

plt.subplot(1, 3, 3)
sns.scatterplot(data=merged_data, x='i', y='e', hue='Ts', size='Rs')
plt.title('Inclination vs Eccentricity')
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


wl_columns = [col for col in train.columns if col.startswith('wl_')]

!pip install -q PyWavelets
import pywt



def create_3d_explorer():
    fig = go.Figure()
    
    # Add star systems
    fig.add_trace(go.Scatter3d(
        x=merged_data['Rs'],
        y=merged_data['Ms'],
        z=merged_data['Ts'],
        mode='markers',
        marker=dict(
            size=merged_data['Mp']*2,
            color=merged_data['P'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Orbital Period')
        ),
        text=[f"Planet ID: {pid}<br>Mass: {mp} Mj<br>Period: {p} days" 
              for pid, mp, p in zip(merged_data['planet_id'], merged_data['Mp'], merged_data['P'])],
        hoverinfo='text',
        name='Star Systems'
    ))
    
    # Add orbital circles
    for _, row in merged_data.iterrows():
        theta = np.linspace(0, 2*np.pi, 100)
        x = row['Rs'] + row['sma'] * np.cos(theta)
        y = row['Ms'] + row['sma'] * np.sin(theta) * np.cos(np.radians(row['i']))
        z = row['Ts'] + row['sma'] * np.sin(theta) * np.sin(np.radians(row['i']))
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=1, color='rgba(150,150,150,0.5)'),
            showlegend=False,
            hoverinfo='none'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Star Radius (Rs)',
            yaxis_title='Star Mass (Ms)',
            zaxis_title='Star Temp (Ts)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        title='Interactive 3D Star System Explorer with Orbits',
        height=800
    )
    return fig

fig_3d = create_3d_explorer()
fig_3d.show()


def plot_wavelet(planet_id):
    data = train[train['planet_id'] == planet_id][wl_columns].values.flatten()
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(data, scales, 'morl')
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coefficients), extent=[0, len(data), 1, 128], 
               cmap='viridis', aspect='auto', vmax=abs(coefficients).max(), 
               vmin=-abs(coefficients).max())
    plt.colorbar(label='Magnitude')
    plt.title(f'Continuous Wavelet Transform - Planet {planet_id}')
    plt.ylabel('Scale')
    plt.xlabel('Wavelength Index')
    plt.show()

plot_wavelet(merged_data['planet_id'].iloc[0])


!pip install -q umap-learn

import umap

def plot_manifold_projections():
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # <-- Add this import
    
    scaler = StandardScaler()
    wl_scaled = scaler.fit_transform(train[wl_columns])
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(wl_scaled)
    
    # UMAP
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(wl_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    sc1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                     c=merged_data['Ts'], cmap='plasma', s=merged_data['Mp']*20)
    ax1.set_title('t-SNE Projection (Colored by Star Temperature)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Create divider for existing axes
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc1, cax=cax, label='Star Temperature (K)')
    
    sc2 = ax2.scatter(umap_results[:, 0], umap_results[:, 1], 
                     c=merged_data['Mp'], cmap='viridis', s=merged_data['Rs']*20)
    ax2.set_title('UMAP Projection (Colored by Planet Mass)')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    # Create divider for second axes
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc2, cax=cax2, label='Planet Mass (Mj)')
    
    plt.tight_layout()
    plt.show()

plot_manifold_projections()


def plot_radial_chart(planet_id):
    selected = merged_data[merged_data['planet_id'] == planet_id].iloc[0]
    
    categories = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
    values = selected[categories].values
    values_normalized = values / values.max()
    
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values_normalized = np.append(values_normalized, values_normalized[:1])
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values_normalized, 'o-', linewidth=2)
    ax.fill(angles, values_normalized, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'Radial Chart of System Parameters - Planet {planet_id}', size=15, y=1.1)
    ax.set_rlabel_position(30)
    plt.tight_layout()
    plt.show()

plot_radial_chart(merged_data['planet_id'].iloc[0])


!pip install -q holoviews

import holoviews as hv
hv.extension('bokeh')  # This enables Bokeh rendering

def create_holoviews_dashboard():
    # Prepare data
    scatter_data = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i', 'planet_id']]
    
    # Create scatter plots
    scatter1 = hv.Scatter(scatter_data, 'Rs', 'Ms').opts(
        width=400, height=400, tools=['hover'], size='Mp', color='Ts', cmap='fire', 
        title='Star Radius vs Mass (size=Planet Mass, color=Star Temp)')
    
    scatter2 = hv.Scatter(scatter_data, 'P', 'sma').opts(
        width=400, height=400, tools=['hover'], size='Mp', color='i', cmap='viridis',
        title='Orbital Period vs SMA (size=Planet Mass, color=Inclination)')
    
    # Create histogram
    hist = hv.Histogram(np.histogram(merged_data['Ts'], bins=20)).opts(
        width=400, height=400, title='Star Temperature Distribution')
    
    # Combine into dashboard
    dashboard = (scatter1 + scatter2 + hist).cols(2)
    return dashboard

create_holoviews_dashboard()



def plot_phase_space():
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
    
    # Main scatter plot
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(merged_data['P'], merged_data['sma'], 
                   c=merged_data['Ts'], s=merged_data['Mp']*50, 
                   cmap='plasma', alpha=0.7)
    ax.set_xlabel('Orbital Period (days)')
    ax.set_ylabel('Semi-major Axis (AU)')
    ax.set_title('Phase Space of Exoplanet Systems')
    
    # Marginal distributions
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    sns.kdeplot(data=merged_data, x='P', color='blue', ax=ax_top, fill=True)
    ax_top.set_yticks([])
    ax_top.set_ylabel('Density')
    
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    sns.kdeplot(data=merged_data, y='sma', color='red', ax=ax_right, fill=True)
    ax_right.set_xticks([])
    ax_right.set_xlabel('Density')
    
    # Colorbar
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    plt.colorbar(sc, cax=cax, label='Star Temperature (K)')
    
    plt.tight_layout()
    plt.show()
print("\n## 7. Phase Space Visualization ##")
plot_phase_space()



@interact(
    x_axis=widgets.Dropdown(options=['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i'], value='Rs'),
    y_axis=widgets.Dropdown(options=['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i'], value='Ms'),
    color_by=widgets.Dropdown(options=['Ts', 'Mp', 'P', 'sma', 'i', 'e'], value='Ts'),
    size_by=widgets.Dropdown(options=['None', 'Mp', 'Rs', 'Ms', 'P', 'sma'], value='Mp')
)
def interactive_scatter(x_axis, y_axis, color_by, size_by):
    size = merged_data[size_by]*20 if size_by != 'None' else 50
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(merged_data[x_axis], merged_data[y_axis], 
                    c=merged_data[color_by], s=size, 
                    cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label=color_by)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} vs {x_axis} (Color: {color_by}, Size: {size_by if size_by != "None" else "Fixed"})')
    plt.grid(True)
    plt.show()

print("\n## 8. Interactive Parameter Explorer - Use the widgets below ##")

