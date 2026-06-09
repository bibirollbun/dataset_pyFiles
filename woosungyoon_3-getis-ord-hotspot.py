import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Optional
import matplotlib.pyplot as plt


class RadialExponentialWeights:
    """Row-standardized exponential decay weights with linear source strength."""
    
    def __init__(self, influence_radius: float, decay_alpha: Optional[float] = None):
        if influence_radius <= 0:
            raise ValueError("influence_radius must be positive")
        
        self.influence_radius = influence_radius
        self.decay_alpha = decay_alpha if decay_alpha is not None else influence_radius / 3.0
        
        if self.decay_alpha <= 0:
            raise ValueError("decay_alpha must be positive")
    
    def calculate(self, coordinates: np.ndarray, source_counts: Optional[np.ndarray] = None) -> np.ndarray:
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must be shape (n, 2)")
        
        n = len(coordinates)
        
        if source_counts is None:
            source_counts = np.ones(n)
        else:
            source_counts = np.asarray(source_counts)
            if len(source_counts) != n:
                raise ValueError("source_counts length must match coordinates length")
            if np.any(source_counts < 0):
                raise ValueError("source_counts must be non-negative")
        
        distances = squareform(pdist(coordinates))
        within_influence = distances <= self.influence_radius
        spatial_decay = np.exp(-distances / self.decay_alpha)
        source_matrix = np.broadcast_to(source_counts, (n, n))
        
        raw_weights = source_matrix * spatial_decay * within_influence
        
        # Row standardization
        row_sums = np.sum(raw_weights, axis=1)
        return np.divide(raw_weights, row_sums[:, np.newaxis], 
                        out=np.zeros_like(raw_weights), 
                        where=row_sums[:, np.newaxis] != 0)



def calculate_gi_star(weights_matrix, attribute_values):
    """Calculate Getis-Ord Gi* statistics."""
    n = len(attribute_values)
    global_mean = np.mean(attribute_values)
    global_std = np.std(attribute_values, ddof=1)
    
    weighted_sums = weights_matrix @ attribute_values
    weights_squared_sums = np.sum(weights_matrix ** 2, axis=1)
    
    numerators = weighted_sums - global_mean
    variance_terms = weights_squared_sums - (1.0 / (n - 1))
    denominators = global_std * np.sqrt(variance_terms)
    
    return np.divide(numerators, denominators, 
                    out=np.zeros_like(numerators), 
                    where=denominators != 0)


df = pd.read_csv("/kaggle/input/3-analysis-hotspot-results/hotspot_results.csv")
coordinates = df[['lon', 'lat']].values  
attribute_values = df['avg_score'].values
source_counts = df['avg_score'].values

weight_calc = RadialExponentialWeights(
    influence_radius=0.2,  
    decay_alpha=0.03
)

weights = weight_calc.calculate(coordinates, source_counts=source_counts)
gi_stats = calculate_gi_star(weights, attribute_values)


#Check
#print(np.allclose(gi_stats - df['gi_statistic'].to_numpy(), 0.0))

df['gi_statistic'] = gi_stats
df['is_hotspot'] = gi_stats > 1.96
df['is_coldspot'] = gi_stats < -1.96

print(f"Hot spots: {df['is_hotspot'].sum()}")
print(f"Cold spots: {df['is_coldspot'].sum()}")


def visualize_getis_ord_results(df):
    """Simple visualization of Getis-Ord Gi* analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Original Data
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['lon'], df['lat'], c=df['avg_score'], 
                          cmap='RdYlBu_r', s=20, alpha=0.7)
    ax1.set_title('Original Data: Average Score')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(scatter1, ax=ax1, label='Average Score')
    
    # 2. Gi* Statistics
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['lon'], df['lat'], c=df['gi_statistic'], 
                          cmap='RdBu_r', s=20, alpha=0.7)
    ax2.set_title('Gi* Statistics (Z-scores)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Gi* Z-score')
    cbar2.ax.axhline(-1.96, color='blue', linestyle='--', alpha=0.8)
    cbar2.ax.axhline(1.96, color='red', linestyle='--', alpha=0.8)
    
    # 3. Hot/Cold Spots
    ax3 = axes[1, 0]
    
    # Simple classification
    normal = ~(df['is_hotspot'] | df['is_coldspot'])
    
    if np.any(normal):
        ax3.scatter(df.loc[normal, 'lon'], df.loc[normal, 'lat'], 
                   c='lightgray', s=20, alpha=0.5, label='Normal')
    if np.any(df['is_hotspot']):
        ax3.scatter(df.loc[df['is_hotspot'], 'lon'], df.loc[df['is_hotspot'], 'lat'], 
                   c='red', s=30, alpha=0.8, label='Hot spots')
    if np.any(df['is_coldspot']):
        ax3.scatter(df.loc[df['is_coldspot'], 'lon'], df.loc[df['is_coldspot'], 'lat'], 
                   c='blue', s=30, alpha=0.8, label='Cold spots')
    
    ax3.set_title('Hot/Cold Spots Classification')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.legend()
    
    # 4. Gi* Histogram
    ax4 = axes[1, 1]
    ax4.hist(df['gi_statistic'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(-1.96, color='blue', linestyle='--', label='p < 0.05')
    ax4.axvline(1.96, color='red', linestyle='--')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title('Gi* Distribution')
    ax4.set_xlabel('Gi* Z-score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    n_hotspots = df['is_hotspot'].sum()
    n_coldspots = df['is_coldspot'].sum()
    n_total = len(df)
    
    print(f"\nRESULTS SUMMARY")
    print(f"Total points: {n_total:,}")
    print(f"Hot spots: {n_hotspots} ({n_hotspots/n_total*100:.1f}%)")
    print(f"Cold spots: {n_coldspots} ({n_coldspots/n_total*100:.1f}%)")
    print(f"Gi* range: [{df['gi_statistic'].min():.3f}, {df['gi_statistic'].max():.3f}]")
    
    return fig


_ = visualize_getis_ord_results(df)


df[df.is_hotspot == True]

