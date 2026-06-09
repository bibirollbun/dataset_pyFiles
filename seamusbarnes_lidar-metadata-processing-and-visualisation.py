!pip install laspy -q
!pip install tqdm -q
!pip install folium -q
!pip install geopandas -q


# Install requirements as needed (quiet mode)
!pip install laspy folium geopandas tqdm -q

# --- Core imports ---
import os, sys, glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# LiDAR
import laspy

# Progress/UX
from tqdm import tqdm

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping
import geopandas as gpd
import folium
import branca.colormap as cm

import warnings
warnings.filterwarnings('ignore')

# Print versions - helpful for reproducibility
def print_versions():
    import laspy, geopandas
    print("Versions:")
    for m in [np, pd, laspy, gpd]:
        print(f"  {m.__name__}: {m.__version__}")

print_versions()


# ==== FILE LOCATIONS ====
PATH_TO_PRECOMPUTED = "/kaggle/input/lidar-survey-of-brazil-laz-metadata"

filenames = {
    "inventory": "cms_brazil_lidar_tile_inventory.csv",
    "metadata": "cms_brazil_lidar_tile_metadata.csv",
    "full": "lidar_metadata_full.csv",
    "class7": "lidar_metadata_class7_summary.csv",
    "class12": "lidar_metadata_class12_summary.csv",
    "extents": "lidar_extents.gpkg"
}
paths = {k: os.path.join(PATH_TO_PRECOMPUTED, v) for k,v in filenames.items()}

# Existence check
for name, path in paths.items():
    print(f"{name:<12}: {'FOUND' if os.path.exists(path) else '!!! NOT FOUND'}")

# ==== ABBREVIATION MAPPING ====
abbr = {
    'density_total_per_m2': 'DTPM',
    'density_ground_per_m2': 'DGPM',
    'ground_fraction': 'GF',
    'area_km2': 'AKM2',
    'mean_scan_angle': 'MSAN',
    'max_scan_angle': 'MXSAN'
}

# ==== LOAD DATAFRAMES ====
df_inventory = pd.read_csv(paths['inventory'])
df_metadata  = pd.read_csv(paths['metadata'])
df_full      = pd.read_csv(paths['full'])
df_class7    = pd.read_csv(paths['class7'])
df_class12   = pd.read_csv(paths['class12'])


print("Inventory columns:")
display(df_inventory.head(1))
print("-"*50)
print("-"*50)

print("Metadata columns:")
display(df_metadata.head(1))
print("-"*50)
print("-"*50)

print("Metadata full sample:")
display(df_full.head(1))
print("-"*50)
print("-"*50)


plt.figure(figsize=(18,6))
sns.violinplot(
    data=df_full, x="class", y="density_ground_per_m2", 
    inner="quart", cut=0, palette="muted")
plt.xticks(rotation=90)
plt.title("Ground Point Density Distribution by Class")
plt.xlabel("Class (Survey Area)")
plt.ylabel("Ground Point Density (points/m²)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(18,6))
sns.violinplot(
    data=df_full, x="class", y="mean_scan_angle", 
    inner="quart", cut=0, palette="muted")
plt.xticks(rotation=90)
plt.title("Mean Scan Angle (deg) per Class")
plt.xlabel("Class (Survey Area)")
plt.ylabel("Mean Scan Angle (deg)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(
    df_full['density_ground_per_m2'], bins=50, kde=True, color="teal")
plt.xlabel("Ground Point Density (/m²)")
plt.ylabel("Number of Tiles")
plt.title("Distribution of Ground Point Density (All Tiles)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
long_df = pd.melt(
    df_full, 
    id_vars=["filename", "class"], 
    value_vars=["density_total_per_m2", "density_ground_per_m2"],
    var_name="type", value_name="density"
)
long_df['type'] = long_df['type'].map({'density_total_per_m2': 'All classes', 'density_ground_per_m2': 'Ground only'})
sns.violinplot(data=long_df, x="type", y="density", palette="pastel", inner="quart", scale="count")
plt.title("Violin Plot: Total vs Ground Point Density (All Tiles)")
plt.yscale("log")
plt.xlabel("")
plt.ylabel("Point Density (/m²)")
plt.tight_layout()
plt.show()


cls7_bar = df_class7.sort_values("DGPM_mean", ascending=False)
plt.figure(figsize=(18,5))
sns.barplot(data=cls7_bar, x="class", y="DGPM_mean", color="skyblue")
plt.xticks(rotation=90)
plt.title("Class Mean Ground Point Density (/m²)")
plt.xlabel("Class")
plt.ylabel("Mean Ground Point Density")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,7))
sns.scatterplot(
    data=df_full, x="density_ground_per_m2", y="ground_fraction",
    hue="class", palette="tab20", alpha=0.7, legend=False
)
plt.title("Ground Point Density vs. Fraction (by Class)")
plt.xlabel("Ground Density (/m²)")
plt.ylabel("Ground Fraction")
plt.tight_layout()
plt.show()


low_density_classes = (
    df_full.groupby("class")["density_ground_per_m2"]
    .mean().sort_values().head(10)
)
print("Classes with lowest mean ground density (/m²):")
display(low_density_classes.to_frame())


outlier_threshold = 0.1
outliers = df_full[df_full['density_ground_per_m2'] < outlier_threshold]
print(f"Tiles with ground point density < {outlier_threshold} /m²:")
display(outliers[['filename', 'class', 'density_ground_per_m2', 'ground_fraction']])


files_of_interest = [
    "BON_A01_2013_laz_13.laz",
    "BON_A01_2018_LAS_11.laz",
    "RIB_A01_2014_laz_2.laz",
    "RIB_A01_2018_LAS_11.laz"
]
cols_to_show = ['filename', 'class', 'density_ground_per_m2', 'ground_fraction', 'mean_scan_angle', 'area_km2']
interesting_subset = df_full[df_full['filename'].isin(files_of_interest)]
display(interesting_subset[cols_to_show])


# Load tile extents, merge ground density
PATH_TO_LIDAR_EXTENTS = paths["extents"]
gdf = gpd.read_file(PATH_TO_LIDAR_EXTENTS)
if "filename" not in gdf.columns:
    # Try guess column for the filename
    gdf["filename"] = gdf["filename_short"] if "filename_short" in gdf.columns else None
gdf = gdf.merge(df_full[["filename", "density_ground_per_m2"]], on="filename", how="left")
gdf = gdf.rename(columns={'density_ground_per_m2':'ground_density'})

# --- Location of interest for centering ---
location_filename = "RIB_A01_2014_laz_2.laz"  # change to your focus tile

center_tile = gdf[gdf["filename"] == location_filename]
if not center_tile.empty:
    centroid = center_tile.geometry.iloc[0].centroid
    center = [centroid.y, centroid.x]
else:
    centroids = gdf.geometry.centroid
    center = [centroids.y.mean(), centroids.x.mean()]

# --- Color scale
min_density, max_density = gdf['ground_density'].min(), gdf['ground_density'].max()
colormap = cm.linear.YlOrRd_09.scale(min_density, max_density)
colormap.caption = 'Ground Point Density (per m²)'

def style_function(feature):
    val = feature['properties']['ground_density']
    return {
        'fillColor': colormap(val) if val is not None else '#888888',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7}

m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")
folium.GeoJson(
    gdf,
    style_function=style_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['filename', 'ground_density'],
        aliases=['Tile', 'Density (/m²)'],
        localize=True
    )
).add_to(m)
colormap.add_to(m)
m


data = np.array(gdf["ground_density"])
data = data[~np.isnan(data)]
q50 = np.percentile(data, 50)  # divides 50/50 (median)
q25 = np.percentile(data, 25)  # 25% below, 75% above
q10 = np.percentile(data, 10)  # 10% below, 90% above

dens_max = max(data)
bins = np.arange(0, dens_max + 0.1, 0.1)

fig, ax = plt.subplots(figsize=(10,6))
n, bins, patches = ax.hist(data, bins=bins, color='skyblue', edgecolor='black')
ax.set_xlabel('Ground Point Density (per m²)')
ax.set_ylabel('Number of Tiles')
ax.set_title('Distribution of Ground Point Densities (all tiles)')
ax.grid(axis='y', alpha=0.3)

# Force layout update BEFORE annotation
fig.canvas.draw()

quantiles = [(q50, 'red', 'Median (Top 50% threshold)'),
             (q25, 'orange', 'Top 75% threshold'),
             (q10, 'green', 'Top 90% threshold')]

for val, color, label in quantiles:
    ax.axvline(val, color=color, linestyle='--', label=label)
    # Find the bin index for the quantile
    binidx = np.digitize(val, bins) - 1
    if 0 <= binidx < len(n):
        y = n[binidx]
        orig_binidx = binidx
        while y == 0 and binidx > 0:
            binidx -= 1
            y = n[binidx]
        if y == 0:
            y = 0
        ax.text(val, y + max(n)*0.03, f'{val:.2f}', color=color, rotation=90, ha='right', va='bottom', fontsize=10)

ax.legend()
fig.tight_layout()
plt.show()

