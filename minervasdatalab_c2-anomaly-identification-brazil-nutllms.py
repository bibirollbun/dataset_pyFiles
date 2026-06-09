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











!pip install rasterio


!pip install localtileserver


import rasterio
import numpy as np
import os
from rasterio.merge import merge
import matplotlib.pyplot as plt


import rasterio
import numpy as np
import os
from glob import glob

# Step 1: Define file paths (sorted for consistency)
bioclim_paths = sorted(glob("/kaggle/input/bioclims/bio*.tif"))

# Step 2: Read and stack rasters
arrays = []
meta = None

for path in bioclim_paths:
    with rasterio.open(path) as src:
        if meta is None:
            meta = src.meta.copy()
        arrays.append(src.read(1))  # read first band

# Step 3: Convert to numpy stack (shape: bands x height x width)
bioclim_stack = np.stack(arrays, axis=0)

# Update metadata to reflect stacked bands
meta.update(count=len(bioclim_paths))

print("âœ… Loaded stack shape:", bioclim_stack.shape)
print("Bands:", [os.path.basename(p) for p in bioclim_paths])



import matplotlib.pyplot as plt

# Step 1: Use band_dict and band_names from earlier
# (redefine if necessary)
band_names = [os.path.basename(p).replace(".tif", "") for p in bioclim_paths]
band_dict = dict(zip(band_names, bioclim_stack))

# Step 2: Plot all bands
n = len(band_names)
cols = 4
rows = int(np.ceil(n / cols))

plt.figure(figsize=(4 * cols, 4 * rows))
for i, name in enumerate(band_names):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(band_dict[name], cmap='viridis')
    plt.title(name)
    plt.axis('off')

plt.suptitle("Bioclimatic Variables", fontsize=16)
plt.tight_layout()
plt.show()






treerich_path='/kaggle/input/amazontrees/treerich1.tif'


import rasterio

with rasterio.open(treerich_path) as src:
    treerich_data = src.read(1)  # Read the first band
    treerich_meta = src.meta     # Get metadata (crs, transform, dtype, etc.)

print("âœ… Shape:", treerich_data.shape)
print("âœ… CRS:", treerich_meta['crs'])



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(treerich_data, cmap='YlGn', vmin=0)
plt.title("ğŸŒ³ Tree Species Richness")
plt.colorbar(label='Species Richness')
plt.axis('off')
plt.show()






brzl_path='/kaggle/input/brazilnut-sdm/mergedBrazilNut_sdm.tif'


import rasterio

with rasterio.open(brzl_path) as src:
    brzl_data = src.read(1)  # Read the first band
    brzl_meta = src.meta     # Get metadata (crs, transform, dtype, etc.)

print("âœ… Shape:", brzl_data.shape)
print("âœ… CRS:", brzl_meta['crs'])


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(brzl_data, cmap='YlGn', vmin=0)
plt.title("Brazil Tree")
plt.colorbar(label='HSM')
plt.axis('off')
plt.show()






dem='/kaggle/input/elevation/SRTM_Elevation_Amazon.tif'


import rasterio

with rasterio.open(dem) as src:
    dem_data = src.read(1)  # Read the first band
    dem_meta = src.meta     # Get metadata (crs, transform, dtype, etc.)

print("âœ… Shape:", dem_data.shape)
print("âœ… CRS:", dem_meta['crs'])


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(dem_data, cmap='YlGn', vmin=0)
plt.title("DEM")
plt.colorbar(label='dem')
plt.axis('off')
plt.show()








import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# Paths
bio_paths = [
    '/kaggle/input/bioclims/bio02.tif',
    '/kaggle/input/bioclims/bio03.tif',
    '/kaggle/input/bioclims/bio04.tif',
    '/kaggle/input/bioclims/bio05.tif',
    '/kaggle/input/bioclims/bio07.tif',
    '/kaggle/input/bioclims/bio15.tif',
    '/kaggle/input/bioclims/bio18.tif',
    '/kaggle/input/bioclims/bio19.tif'
]
suitability_path = '/kaggle/input/brazilnut-sdm/mergedBrazilNut_sdm.tif'
treerich_path = '/kaggle/input/amazontrees/treerich1.tif'
dem_path = '/kaggle/input/elevation/SRTM_Elevation_Amazon.tif'

# Step 1: Load bioclim stack
bio_stack = []
for path in bio_paths:
    with rasterio.open(path) as src:
        bio_stack.append(src.read(1))
        ref_meta = src.meta  # Use first bio raster as reference

bio_stack = np.stack(bio_stack)

# Step 2: Function to resample to match reference raster
def resample_to_match(src_path, ref_meta):
    with rasterio.open(src_path) as src:
        raw = src.read(1)
        dst = np.empty((ref_meta['height'], ref_meta['width']), dtype=np.float32)

        reproject(
            source=raw,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta['transform'],
            dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear
        )
    return dst

# Step 3: Resample all additional layers
brzl_resampled = resample_to_match(suitability_path, ref_meta)
treerich_resampled = resample_to_match(treerich_path, ref_meta)
dem_resampled = resample_to_match(dem_path, ref_meta)

# Step 4: Stack everything together
combined_stack = np.concatenate([
    bio_stack,
    brzl_resampled[np.newaxis, ...],
    treerich_resampled[np.newaxis, ...],
    dem_resampled[np.newaxis, ...]
], axis=0)

# Step 5: Visualize key layers
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
titles = ['bio19', 'bio18', 'Suitability', 'Tree Richness', 'DEM']
band_idx = [7, 6, 8, 9, 10]  # Adjust indices accordingly

for i, ax in enumerate(axes):
    im = ax.imshow(combined_stack[band_idx[i]], cmap='viridis')
    ax.set_title(titles[i])
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.show()









import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Get shape
bands, height, width = combined_stack.shape

# Step 2: Flatten the stack: (n_pixels, n_features)
flat_data = combined_stack.reshape(bands, -1).T  # Shape: (height * width, bands)

# Step 3: Create mask for valid pixels
valid_mask = ~np.any(np.isnan(flat_data), axis=1) & (np.any(flat_data != 0, axis=1))

# Step 4: Apply KMeans on valid pixels
X = flat_data[valid_mask]
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
labels = np.full(flat_data.shape[0], fill_value=-1, dtype=np.int32)  # Use -1 for nodata
labels[valid_mask] = kmeans.fit_predict(X)

# Step 5: Reshape labels to 2D map
cluster_map = labels.reshape(height, width)

# Step 6: Visualize result
plt.figure(figsize=(10, 7))
plt.imshow(cluster_map, cmap='tab10')
plt.title("KMeans Clusters (k=5)")
plt.axis("off")
plt.colorbar(label="Cluster ID")
plt.tight_layout()
plt.show()









import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Inspect stack shape (bands, height, width)
bands, height, width = combined_stack.shape

# Step 2: Flatten raster stack into (n_pixels, n_features)
flat_pixels = combined_stack.reshape(bands, -1).T  # (H*W, bands)

# Step 3: Mask invalid pixels (NaNs or all zero bands)
valid_mask = ~np.any(np.isnan(flat_pixels), axis=1) & (np.any(flat_pixels != 0, axis=1))
X_valid = flat_pixels[valid_mask]

# Step 4: Run KMeans clustering on valid pixels
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
cluster_labels = np.full(flat_pixels.shape[0], -1, dtype=np.int32)
cluster_labels[valid_mask] = kmeans.fit_predict(X_valid)

# Step 5: Reshape back to 2D cluster map
cluster_map = cluster_labels.reshape(height, width)

# Step 6: Visualize
plt.figure(figsize=(10, 7))
im = plt.imshow(cluster_map, cmap='tab10')
plt.title("KMeans Clusters (k=5)")
plt.axis("off")
plt.colorbar(im, label="Cluster ID")
plt.tight_layout()
plt.show()






import pandas as pd

# Step 1: Get number of features (bands)
num_features = combined_stack.shape[0]

# Step 2: Reshape for per-pixel access
# Shape: (n_pixels, n_features)
flat_features = combined_stack.reshape(num_features, -1).T  # (n_pixels, n_features)

# Step 3: Reuse cluster labels
valid_features = flat_features[valid_mask]
valid_labels = labels[valid_mask]

# Step 4: Compute mean feature values per cluster
cluster_profiles = []

for cluster_id in np.unique(valid_labels):
    cluster_pixels = valid_features[valid_labels == cluster_id]
    mean_profile = np.nanmean(cluster_pixels, axis=0)
    cluster_profiles.append((cluster_id, mean_profile))

# Step 5: Display as table
df_profiles = pd.DataFrame(
    [profile for _, profile in cluster_profiles],
    index=[f"Cluster {cid}" for cid, _ in cluster_profiles],
    columns=[f"Feature {i}" for i in range(num_features)]
)

# Optional: round and display
print(df_profiles.round(2))






df_profiles.columns = [
    "BIO02 (Diurnal Range)",
    "BIO03 (Isothermality)",
    "BIO04 (Temp Seasonality)",
    "BIO05 (Max Temp Warmest Month)",
    "BIO07 (Temp Annual Range)",
    "BIO15 (Precip Seasonality)",
    "BIO18 (Precip Warmest Quarter)",
    "BIO19 (Precip Coldest Quarter)",
    "Brazil Nut Suitability",
    "Tree Species Richness",
    "DEM"
]



markdown_text = df_profiles.round(2).to_markdown()
print(markdown_text)









!pip install openai --upgrade



import openai


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OpenAI")


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# Access OpenAI key from Kaggle Secrets
user_secrets = UserSecretsClient()
openai_key = user_secrets.get_secret("OpenAI")  # "OpenAI" should be the secret label

# Create OpenAI client
client = OpenAI(api_key=openai_key)





prompt = """
You're a geospatial archaeologist analyzing 5 landscape clusters across the Amazon basin. Each cluster represents a unique combination of environmental and ecological conditions, derived from:

- Bioclimatic variables (BIOCLIM)
- Brazil nut habitat suitability
- Tree species richness
- Elevation (DEM)

Below are the definitions of the features used to characterize each cluster:

- Feature 0: BIO02 â€“ Mean Diurnal Temperature Range
- Feature 1: BIO03 â€“ Isothermality
- Feature 2: BIO04 â€“ Temperature Seasonality
- Feature 3: BIO05 â€“ Max Temperature of Warmest Month
- Feature 4: BIO07 â€“ Temperature Annual Range
- Feature 5: BIO15 â€“ Precipitation Seasonality
- Feature 6: BIO18 â€“ Precipitation of Warmest Quarter
- Feature 7: BIO19 â€“ Precipitation of Coldest Quarter
- Feature 8: Brazil Nut Habitat Suitability (0â€“1)
- Feature 9: Tree Species Richness
- Feature 10: DEM (Digital Elevation Model, in meters)

Your task is to analyze the mean profile of each cluster (below) and answer the following:

1. Which clusters are ecologically unusual or anomalous?
2. Which clusters might correspond to anthropogenic or archaeologically relevant zones (e.g., suitable for agroforestry, terra preta, geoglyph contexts)?
3. Which clusters likely represent natural vs. human-managed forests, and why?

Here are the cluster profiles:

Cluster 0:
  BIO02: 12.47, BIO03: 70.07, BIO04: 13.40, BIO05: 31.34, BIO07: 17.88
  BIO15: 69.79, BIO18: 425.22, BIO19: 111.67
  Brazil Nut Suitability: 0.05, Tree Richness: -11.08, DEM: 687.53

Cluster 1:
  BIO02: 11.39, BIO03: 74.64, BIO04: 8.66, BIO05: 32.12, BIO07: 15.39
  BIO15: 65.53, BIO18: 370.88, BIO19: 379.27
  Brazil Nut Suitability: 0.07, Tree Richness: -9883.33, DEM: 483.04

Cluster 2:
  BIO02: 10.60, BIO03: 78.07, BIO04: 6.41, BIO05: 32.72, BIO07: 13.73
  BIO15: 53.80, BIO18: 398.39, BIO19: 573.06
  Brazil Nut Suitability: 0.12, Tree Richness: -4648.52, DEM: 224.14

Cluster 3:
  BIO02: 9.74, BIO03: 81.01, BIO04: 4.83, BIO05: 32.44, BIO07: 12.01
  BIO15: 44.60, BIO18: 413.31, BIO19: 765.29
  Brazil Nut Suitability: 0.06, Tree Richness: 70.90, DEM: 153.13

Cluster 4:
  BIO02: 14.21, BIO03: 76.89, BIO04: 10.78, BIO05: 17.61, BIO07: 18.68
  BIO15: 73.74, BIO18: 351.01, BIO19: 119.93
  Brazil Nut Suitability: 0.00, Tree Richness: -9998.95, DEM: 3540.54
"""




!pip install --upgrade openai






response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a geospatial archaeologist."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.4
)

print(response.choices[0].message.content)









df_profiles["Tree Species Richness"] = df_profiles["Tree Species Richness"].clip(lower=0)



df_profiles.head()





cluster_labels = {
    0: "Possible managed agroforestry zone",
    1: "Natural forest (low richness)",
    2: "Natural forest (low richness)",
    3: "Possible human-modified rich zone",
    4: "Natural forest (low richness)"
}



from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# Access API key securely
user_secrets = UserSecretsClient()
openai_key = user_secrets.get_secret("OpenAI")

client = OpenAI(api_key=openai_key)

# Final GPT prompt with correct values
prompt = """
You're a geospatial archaeologist analyzing 5 landscape clusters across the Amazon basin. Each cluster represents a unique combination of environmental and ecological conditions, derived from:

- Bioclimatic variables (BIOCLIM)
- Brazil nut habitat suitability
- Tree species richness
- Elevation (DEM)

Below are the definitions of the features used to characterize each cluster:

- Feature 0: BIO02 â€“ Mean Diurnal Temperature Range
- Feature 1: BIO03 â€“ Isothermality
- Feature 2: BIO04 â€“ Temperature Seasonality
- Feature 3: BIO05 â€“ Max Temperature of Warmest Month
- Feature 4: BIO07 â€“ Temperature Annual Range
- Feature 5: BIO15 â€“ Precipitation Seasonality
- Feature 6: BIO18 â€“ Precipitation of Warmest Quarter
- Feature 7: BIO19 â€“ Precipitation of Coldest Quarter
- Feature 8: Brazil Nut Habitat Suitability (0â€“1)
- Feature 9: Tree Species Richness
- Feature 10: DEM (Digital Elevation Model, in meters)

Your task is to analyze the mean profile of each cluster (below) and answer the following:

1. Which clusters are ecologically unusual or anomalous?
2. Which clusters might correspond to anthropogenic or archaeologically relevant zones (e.g., suitable for agroforestry, terra preta, geoglyph contexts)?
3. Which clusters likely represent natural vs. human-managed forests, and why?

Here are the cluster profiles:

Cluster 0:
  BIO02: 12.47, BIO03: 70.07, BIO04: 13.40, BIO05: 31.34, BIO07: 17.88
  BIO15: 69.79, BIO18: 425.22, BIO19: 111.67
  Brazil Nut Suitability: 0.05, Tree Richness: -11.08, DEM: 687.53

Cluster 1:
  BIO02: 11.39, BIO03: 74.64, BIO04: 8.66, BIO05: 32.12, BIO07: 15.39
  BIO15: 65.53, BIO18: 370.88, BIO19: 379.27
  Brazil Nut Suitability: 0.07, Tree Richness: -9883.33, DEM: 483.04

Cluster 2:
  BIO02: 10.60, BIO03: 78.07, BIO04: 6.41, BIO05: 32.72, BIO07: 13.73
  BIO15: 53.80, BIO18: 398.39, BIO19: 573.06
  Brazil Nut Suitability: 0.12, Tree Richness: -4648.52, DEM: 224.14

Cluster 3:
  BIO02: 9.74, BIO03: 81.01, BIO04: 4.83, BIO05: 32.44, BIO07: 12.01
  BIO15: 44.60, BIO18: 413.31, BIO19: 765.29
  Brazil Nut Suitability: 0.06, Tree Richness: 70.90, DEM: 153.13

Cluster 4:
  BIO02: 14.21, BIO03: 76.89, BIO04: 10.78, BIO05: 17.61, BIO07: 18.68
  BIO15: 73.74, BIO18: 351.01, BIO19: 119.93
  Brazil Nut Suitability: 0.00, Tree Richness: -9998.95, DEM: 3540.54
"""


# Query GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a geospatial archaeologist."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

# Print GPT's interpretation
print(response.choices[0].message.content)






import pandas as pd
gdf = pd.read_csv('/kaggle/input/archeologicaltype/filtered_points.csv')



gdf.head()





import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the CSV
gdf = pd.read_csv('/kaggle/input/archeologicaltype/filtered_points.csv')

# Create geometry from POINT_X and POINT_Y
gdf['geometry'] = gdf.apply(lambda row: Point(row['POINT_X'], row['POINT_Y']), axis=1)

# Convert to GeoDataFrame with proper CRS
gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

# Preview
print(gdf.head())









import rasterio

with rasterio.open('/kaggle/input/bioclims/bio02.tif') as src:
    transform = src.transform
    raster_crs = src.crs




# Reproject site GeoDataFrame to match raster CRS
gdf = gdf.to_crs(raster_crs)

# Convert point coords to raster pixel indices
def coords_to_rc(x, y, transform):
    col, row = ~transform * (x, y)
    return int(row), int(col)

# Generate (row, col) for each site
rows_cols = [coords_to_rc(x, y, transform) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]

# Extract cluster ID from in-memory cluster_map
gdf['cluster_id'] = [
    cluster_map[r, c] if 0 <= r < cluster_map.shape[0] and 0 <= c < cluster_map.shape[1] else -1
    for r, c in rows_cols
]



import matplotlib.pyplot as plt
from rasterio.transform import rowcol

# 1. Make sure gdf is in the same CRS as the raster
gdf_proj = gdf.to_crs(raster_crs)

# 2. Convert lat/lon â†’ raster pixel (row, col)
def lonlat_to_pixel(x, y, transform):
    col, row = ~transform * (x, y)
    return int(col), int(row)

pixel_coords = [lonlat_to_pixel(x, y, transform) for x, y in zip(gdf_proj.geometry.x, gdf_proj.geometry.y)]
cols, rows = zip(*pixel_coords)  # flip for plotting

# 3. Plot raster with scatter
plt.figure(figsize=(10, 8))
plt.imshow(cluster_map, cmap='tab10')
plt.scatter(cols, rows, c='black', s=10, label='Field Sites (pixels)')
plt.title('Cluster Map with Correctly Positioned Field Sites')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Reproject to raster CRS
gdf_proj = gdf.to_crs(raster_crs)

# Convert point geometry to pixel indices
def lonlat_to_pixel(x, y, transform):
    col, row = ~transform * (x, y)
    return int(col), int(row)

# Get pixel coordinates
pixel_coords = [lonlat_to_pixel(x, y, transform) for x, y in zip(gdf_proj.geometry.x, gdf_proj.geometry.y)]
cols, rows = zip(*pixel_coords)

# Map types to colors
unique_types = gdf_proj['type'].unique()
type_to_color = {t: c for t, c in zip(unique_types, plt.cm.tab10.colors[:len(unique_types)])}
colors = [type_to_color[t] for t in gdf_proj['type']]

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(cluster_map, cmap='tab10')
plt.scatter(cols, rows, c=colors, s=20, edgecolor='k', linewidth=0.5)

# Legend
for t, color in type_to_color.items():
    plt.scatter([], [], color=color, label=t, s=30)
plt.legend(title='Site Type', loc='lower left', bbox_to_anchor=(1, 0))

plt.title('Cluster Map with Archaeological Sites Colored by Type')
plt.axis('off')
plt.tight_layout()
plt.show()



plt.savefig("cluster_sites_overlay2.png", dpi=300, bbox_inches='tight')



import os
os.listdir("/kaggle/working/")






gdf['cluster_id'].value_counts().sort_index()



df_summary = df_profiles.copy()
df_summary["Site Count"] = gdf["cluster_id"].value_counts().reindex(df_summary.index).fillna(0).astype(int)
df_summary = df_summary.round(2)




print(df_summary.to_markdown(tablefmt="github"))






import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import os

# Save the image (if not already saved)
plt.savefig("cluster_sites_overlay2.png", dpi=300, bbox_inches="tight")

# Combine df_profiles and gdf['cluster_id'] count
df_summary = df_profiles.copy()
df_summary["Site Count"] = gdf["cluster_id"].value_counts().reindex(df_profiles.index).fillna(0).astype(int)
df_summary = df_summary.round(2)

# Optional: rename index
df_summary.index = [f"Cluster {i}" for i in df_summary.index]

# Convert to markdown table string
table_md = df_summary.reset_index().to_markdown(index=False)

# Create full markdown block
markdown_text = f"""
## ğŸ“� Cluster Map with Archaeological Site Overlays

This map shows the 5 KMeans-derived environmental clusters across Acre, Brazil.  
Each **black dot** represents a known archaeological site (e.g., platform mounds), overlaid using georeferenced field data.

![Cluster Overlay](cluster_sites_overlay2.png)

---

## ğŸ“Š Cluster Summary Table

{table_md}

---

## ğŸ¤– GPT Prompt for Interpretation

> Based on the table and spatial pattern of sites, which clusters are most strongly associated with known archaeological presence?  
> Which ones suggest evidence of ancient human land use (e.g., agroforestry, terra preta)?  
> Which are likely natural background zones?  
> Recommend priority clusters for predictive modeling of new archaeological sites.
"""

# Display in notebook
display(Markdown(markdown_text))






from openai import OpenAI

# Authenticate your OpenAI client
client = OpenAI(api_key=openai_key)  # assumes you've loaded your key already

# Send the markdown text (generated above) to GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a geospatial archaeologist."},
        {"role": "user", "content": markdown_text}  # send your markdown block directly
    ],
    temperature=0.3,
    max_tokens=1500  # optional, increase if response cuts off
)

# Output GPT's interpretation
print(response.choices[0].message.content)






prompt = f"""
You are a geospatial archaeologist analyzing five environmental landscape clusters in Acre, Brazil. Each cluster was derived from KMeans clustering of ecological raster data, including bioclimatic variables, Brazil nut habitat suitability, and tree species richness.

The figure attached below â€” *cluster_sites_overlay.png* â€” shows the spatial distribution of these clusters. **Black points** represent known archaeological sites (e.g., platform mounds) based on field survey data.

---

### Known Archaeological Site Counts:
- Cluster 0 â†’ **395 sites**
- Cluster 1 â†’ **72 sites**
- Cluster 2 â†’ **82 sites**
- Cluster 3 â†’ **371 sites**
- Cluster 4 â†’ **0 sites**

---

### Please Analyze:

1. Which clusters show the strongest known archaeological presence?
2. Which clusters are likely influenced by ancient human activity (e.g., agroforestry, terra preta)?
3. Which clusters reflect natural or background ecological zones?
4. Based on this, which clusters should be prioritized for predictive modeling of new archaeological sites?

Note: High Brazil nut suitability and tree species richness are potential proxies for ancient agroforestry.

---

### Environmental Profiles by Cluster:

{table_md}
"""



response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a geospatial archaeologist."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    max_tokens=1500
)

print(response.choices[0].message.content)












# Group by cluster and site type, count occurrences
site_matrix = gdf.groupby(['cluster_id', 'type']).size().unstack(fill_value=0)

# Optional: sort by cluster index
site_matrix = site_matrix.sort_index()

# Display or export
print(site_matrix)






prompt = """
## ğŸ“� Cluster Map with Archaeological Site Overlays

This map shows five KMeans-derived environmental clusters across Acre, Brazil.  
Each point represents a known archaeological site, overlaid based on georeferenced survey data.

---

## ğŸ“Š Archaeological Site Matrix (per Cluster Ã— Type)

This table summarizes the count of known archaeological sites per **cluster** and **site type**:

| cluster_id |  ADE | ceremonial centres | fortified settlements | geoglyphs | large platform mounds | mounded ring villages |
|------------|------|--------------------|------------------------|-----------|------------------------|------------------------|
| 0          |   91 |                 20 |                     26 |       247 |                      0 |                     11 |
| 1          |   56 |                  0 |                      9 |         0 |                      7 |                      0 |
| 2          |   71 |                  0 |                     11 |         0 |                      0 |                      0 |
| 3          |  361 |                  0 |                      2 |         8 |                      0 |                      0 |
| 4          |    0 |                  0 |                      0 |         0 |                      0 |                      0 |

---

## ğŸ¤– GPT Prompt for Interpretation

You are a geospatial archaeologist analyzing environmental clusters in Acre, Brazil.  
Each cluster was derived from KMeans on bioclimatic conditions, tree species richness, and Brazil nut suitability.

Here is your task:

1. **Which clusters show strong association with specific archaeological site types (e.g., geoglyphs, ADEs)?**
2. **Which clusters may reflect ancient human land use (e.g., agroforestry, terra preta)?**
3. **Which clusters appear to be ecologically natural or background zones?**
4. **Based on both spatial pattern and table, which clusters should be prioritized for predictive modeling of *undiscovered* archaeological sites?**

Note:
- **High Brazil nut suitability** and **high tree richness** are proxies for anthropogenic forests.
- Cluster 4 has **no known sites** â€“ does that make it background or an unexplored anomaly?

Please explain based on the matrix above and ecological reasoning.
"""



response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a geospatial archaeologist."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    max_tokens=1500
)

print(response.choices[0].message.content)






import pandas as pd
import numpy as np

# Step 1: Observed matrix
obs = pd.DataFrame({
    'ADE': [91, 56, 71, 361, 0],
    'ceremonial centres': [20, 0, 0, 0, 0],
    'fortified settlements': [26, 9, 11, 2, 0],
    'geoglyphs': [247, 0, 0, 8, 0],
    'large platform mounds': [0, 7, 0, 0, 0],
    'mounded ring villages': [11, 0, 0, 0, 0]
}, index=[0,1,2,3,4])

# Step 2: Row totals (cluster), column totals (type), grand total
row_totals = obs.sum(axis=1)
col_totals = obs.sum(axis=0)
grand_total = col_totals.sum()

# Step 3: Expected matrix under null
expected = np.outer(row_totals, col_totals) / grand_total
expected = pd.DataFrame(expected, index=obs.index, columns=obs.columns)

# Step 4: Residual Z-scores
z_scores = (obs - expected) / np.sqrt(expected)

# Step 5: Top 5 anomalies
z_long = z_scores.stack().reset_index()
z_long.columns = ['cluster_id', 'type', 'z_score']
top_anomalies = z_long.reindex(z_long.z_score.abs().sort_values(ascending=False).index).head(5)

print(top_anomalies)



# Re-import required libraries after kernel reset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Reconstruct the observed matrix
obs = pd.DataFrame({
    'ADE': [91, 56, 71, 361, 0],
    'ceremonial centres': [20, 0, 0, 0, 0],
    'fortified settlements': [26, 9, 11, 2, 0],
    'geoglyphs': [247, 0, 0, 8, 0],
    'large platform mounds': [0, 7, 0, 0, 0],
    'mounded ring villages': [11, 0, 0, 0, 0]
}, index=[0, 1, 2, 3, 4])

# Calculate expected values under null hypothesis
row_totals = obs.sum(axis=1)
col_totals = obs.sum(axis=0)
grand_total = col_totals.sum()
expected = np.outer(row_totals, col_totals) / grand_total
expected = pd.DataFrame(expected, index=obs.index, columns=obs.columns)

# Compute standardized residuals
z_scores = (obs - expected) / np.sqrt(expected)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(z_scores, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"label": "Z-score"})
plt.title("Standardized Residuals (Z-scores) for Cluster Ã— Site Type")
plt.xlabel("Archaeological Site Type")
plt.ylabel("Cluster ID")
plt.tight_layout()
plt.show()











