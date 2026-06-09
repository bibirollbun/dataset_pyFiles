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


import requests

url = "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&south=1.1&north=1.66&west=-69.8&east=-69.4&outputFormat=GTiff&API_Key=e42e810d2864393de540e7d5326803eb"
r = requests.get(url)

with open("/kaggle/working/Amazon_tile.tif", "wb") as f:
    f.write(r.content)

print("Downloaded Amazon_tile.tif")



url = "https://portal.opentopography.org/API/globaldem?demtype=AW3D30&south=1.1&north=1.66&west=-69.8&east=-69.4&outputFormat=GTiff&API_Key=e42e810d2864393de540e7d5326803eb"
r = requests.get(url)

with open("/kaggle/working/Amazon_tile1.tif", "wb") as f:
    f.write(r.content)

print("Downloaded Amazon_tile1.tif")



import os
print("amazon_tile.tif exists:", os.path.exists("/kaggle/working/Amazon_tile.tif"))
print("amazon_tile.tif exists:", os.path.exists("/kaggle/working/Amazon_tile1.tif"))


import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with rasterio.open("/kaggle/working/Amazon_tile.tif") as src:
    elevation = src.read(1)

    plt.imshow(elevation, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title("Amazon Elevation - Tile")
    plt.show()

    rows, cols = np.indices(elevation.shape)
    longitudes, latitudes = src.xy(rows, cols)

longitudes = np.array(longitudes).flatten()
latitudes = np.array(latitudes).flatten()
elevations = elevation.flatten()

df = pd.DataFrame({
    'longitude': longitudes,
    'latitude': latitudes,
    'elevation': elevations
})

print(df.head())



import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Open the new raster file
with rasterio.open("/kaggle/working/Amazon_tile1.tif") as src:
    elevation = src.read(1)

    # Plot the elevation
    plt.imshow(elevation, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title("Amazon Elevation - Tile1")
    plt.show()

    # Create row, col indices grid
    rows, cols = np.indices(elevation.shape)

    # Convert row, col to geographic coordinates (lon, lat)
    longitudes, latitudes = src.xy(rows, cols)

# Flatten everything to 1D arrays
longitudes = np.array(longitudes).flatten()
latitudes = np.array(latitudes).flatten()
elevations = elevation.flatten()

# Create DataFrame
df = pd.DataFrame({
    'longitude': longitudes,
    'latitude': latitudes,
    'elevation': elevations
})

print(df.head())



import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("/kaggle/working/Amazon_tile.tif") as src:
    print(src.bounds, src.width, src.height, src.transform, src.crs)

with rasterio.open("/kaggle/working/Amazon_tile1.tif") as src:
    print(src.bounds, src.width, src.height, src.transform, src.crs)



import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def resample_raster(src_raster_path, target_raster_path):
    with rasterio.open(target_raster_path) as target:
        target_crs = target.crs
        target_transform = target.transform
        target_width = target.width
        target_height = target.height
    
    with rasterio.open(src_raster_path) as src:
        data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        
        resampled = np.empty(shape=(target_height, target_width), dtype=data.dtype)
        
        reproject(
            source=data,
            destination=resampled,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    return resampled

tile1_path = "/kaggle/working/Amazon_tile.tif"
tile2_path = "/kaggle/working/Amazon_tile1.tif"

with rasterio.open(tile1_path) as src1:
    tile1 = src1.read(1)
    
tile2_resampled = resample_raster(tile2_path, tile1_path)

# Now you can subtract:
diff = tile1 - tile2_resampled

print("Difference stats:")
print(f"Mean diff: {np.mean(diff):.2f}, Min diff: {np.min(diff)}, Max diff: {np.max(diff)}")

import matplotlib.pyplot as plt
plt.imshow(diff, cmap='coolwarm')
plt.colorbar(label='Elevation difference (m)')
plt.title('Difference between Amazon_tile and Amazon_tile1 (resampled)')
plt.show()



import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Load raster
with rasterio.open("/kaggle/working/Amazon_tile.tif") as src:
    elevation = src.read(1)
    transform = src.transform

# Plot grayscale heatmap
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap='gray')  # grayscale colormap
plt.colorbar(label='Elevation (m)')
plt.title("Amazon Elevation - Grayscale")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

# --- Zoom Section ---
# Define zoom window: center portion
height, width = elevation.shape
row_start, row_end = int(height * 0.4), int(height * 0.6)
col_start, col_end = int(width * 0.4), int(width * 0.6)
zoomed = elevation[row_start:row_end, col_start:col_end]

# Plot zoomed
plt.figure(figsize=(8, 6))
plt.imshow(zoomed, cmap='gray')
plt.colorbar(label='Elevation (m)')
plt.title("Zoomed Elevation (Grayscale)")
plt.xlabel("Column (Zoomed)")
plt.ylabel("Row (Zoomed)")
plt.show()



import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Load raster
with rasterio.open("/kaggle/working/Amazon_tile1.tif") as src:
    elevation = src.read(1)
    transform = src.transform

# Plot grayscale heatmap
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap='gray')  # grayscale colormap
plt.colorbar(label='Elevation (m)')
plt.title("Amazon Elevation - Grayscale")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

# --- Zoom Section ---
# Define zoom window: center portion
height, width = elevation.shape
row_start, row_end = int(height * 0.4), int(height * 0.6)
col_start, col_end = int(width * 0.4), int(width * 0.6)
zoomed = elevation[row_start:row_end, col_start:col_end]

# Plot zoomed
plt.figure(figsize=(8, 6))
plt.imshow(zoomed, cmap='gray')
plt.colorbar(label='Elevation (m)')
plt.title("Zoomed Elevation (Grayscale)")
plt.xlabel("Column (Zoomed)")
plt.ylabel("Row (Zoomed)")
plt.show()



import rasterio
import numpy as np
import plotly.graph_objects as go

# Load elevation data
with rasterio.open("/kaggle/working/Amazon_tile.tif") as src:
    elevation = src.read(1)
    transform = src.transform

# Get row, col indices
rows, cols = np.indices(elevation.shape)

# Convert to geographic coordinates
xs, ys = src.xy(rows, cols)

# Flatten all arrays for plotting
x = np.array(xs).flatten()
y = np.array(ys).flatten()
z = elevation.flatten()

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=1,
        color=z,            # color by elevation
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Elevation (m)')
    )
)])

fig.update_layout(
    title="3D Elevation Point Cloud",
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Elevation (m)'
    )
)

fig.show()



import rasterio
import numpy as np
import plotly.graph_objects as go

# Load elevation data
with rasterio.open("/kaggle/working/Amazon_tile1.tif") as src:
    elevation = src.read(1)
    transform = src.transform

# Get row, col indices
rows, cols = np.indices(elevation.shape)

# Convert to geographic coordinates
xs, ys = src.xy(rows, cols)

# Flatten all arrays for plotting
x = np.array(xs).flatten()
y = np.array(ys).flatten()
z = elevation.flatten()

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=1,
        color=z,            # color by elevation
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Elevation (m)')
    )
)])

fig.update_layout(
    title="3D Elevation Point Cloud",
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Elevation (m)'
    )
)

fig.show()



import rasterio
import numpy as np
import pandas as pd

# Load the raster tile
with rasterio.open("/kaggle/working/Amazon_tile.tif") as src:
    elevation = src.read(1)
    transform = src.transform

    # Get indices of the raster (row, col)
    rows, cols = np.indices(elevation.shape)

    # Convert row-col indices to geographic coordinates (longitude, latitude)
    lon, lat = rasterio.transform.xy(transform, rows, cols)

# Flatten everything to make it tabular
lon = np.array(lon).flatten()
lat = np.array(lat).flatten()
elev = elevation.flatten()

# Create a DataFrame
df = pd.DataFrame({
    'longitude': lon,
    'latitude': lat,
    'elevation': elev
})

# Optional: Remove no-data values (usually < 0 or a specific value like -9999)
df = df[df['elevation'] > 0]  # or use df[df['elevation'] != src.nodata] if known

# Show first few rows
print(df.head())

# Save to CSV if needed
df.to_csv("/kaggle/working/amazon_elevation_dataset_SRTMGL3.csv", index=False)



import rasterio
import numpy as np
import pandas as pd

# Load the raster tile
with rasterio.open("/kaggle/working/Amazon_tile1.tif") as src:
    elevation = src.read(1)
    transform = src.transform

    # Get indices of the raster (row, col)
    rows, cols = np.indices(elevation.shape)

    # Convert row-col indices to geographic coordinates (longitude, latitude)
    lon, lat = rasterio.transform.xy(transform, rows, cols)

# Flatten everything to make it tabular
lon = np.array(lon).flatten()
lat = np.array(lat).flatten()
elev = elevation.flatten()

# Create a DataFrame
df = pd.DataFrame({
    'longitude': lon,
    'latitude': lat,
    'elevation': elev
})

# Optional: Remove no-data values (usually < 0 or a specific value like -9999)
df = df[df['elevation'] > 0]  # or use df[df['elevation'] != src.nodata] if known

# Show first few rows
print(df.head())

# Save to CSV if needed
df.to_csv("/kaggle/working/amazon_elevation_dataset_AW3D30.csv", index=False)


