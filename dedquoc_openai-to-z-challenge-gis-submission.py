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


import warnings
warnings.filterwarnings('ignore')


!pip install rasterio
!pip install laspy rasterio scipy
!pip install -q scikit-image
!pip install geopandas
!pip install open3d
#!pip install scipy
!pip install cupy-cuda11x


!apt-get update
!apt-get install -y pdal


#recheck files for working
!ls /kaggle/working


import shutil
import os

# Define the source and destination paths
source_path = '/kaggle/input/working/autzen.las'
destination_path = '/kaggle/working/autzen.las'

# Copy the file to the writable directory
shutil.copy(source_path, destination_path)

# Verify the file has been copied
print(os.listdir('/kaggle/working'))


%%time
import laspy
import numpy as np
import matplotlib.pyplot as plt

# Load LAS file
las = laspy.read("/kaggle/input/working/autzen.las")

# Extract ground points
ground_points = las.points[las.classification == 2]

# Get coordinates and intensity
x = ground_points.x
y = ground_points.y
z = ground_points.z
intensity = ground_points.intensity

# Plot DTM
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=z, cmap='terrain', s=0.5)
plt.colorbar(label='Elevation (m)')
plt.title('Digital Terrain Model (DTM) - Autzen Dataset')
plt.axis('off')
plt.show()


%%time
plt.scatter(x, y, c=z, cmap='terrain', s=2)  # Try s=1 to s=5 for better visibility


%%time
import laspy
import numpy as np
import open3d as o3d

# File paths
source_path = '/kaggle/input/working/autzen.las'
destination_path = '/kaggle/working/autzen.las'

# Load the LAS file using laspy
las = laspy.read(source_path)

# Extract XYZ coordinates
points = np.vstack((las.x, las.y, las.z)).transpose()

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: downsample for faster preview
pcd = pcd.voxel_down_sample(voxel_size=0.5)

# Visualize (in Kaggle, visualization needs to be done with matplotlib or saved)
# Convert to numpy array and plot 2D projection
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(points[::50, 0], points[::50, 1], s=0.5, c=points[::50, 2], cmap='viridis')
plt.title("2D Projection of LiDAR Point Cloud (XY)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label='Z Height')
plt.grid(True)
plt.show()


# check this fuction : read *.las file to save to lidar_dtm.tif > meaning?
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 1. Load LAS/LAZ file (make sure it's uploaded)
#las = laspy.read("/kaggle/input/your-data/your_input.las")  # or .laz
las = laspy.read('/kaggle/input/working/autzen.las')


# 2. Extract XYZ and classification
x, y, z = las.x, las.y, las.z
try:
    cls = las.classification  # Optional: for filtering ground
    mask = cls == 2  # Ground points only
except:
    mask = np.ones_like(z, dtype=bool)  # Use all if classification missing

xg, yg, zg = x[mask], y[mask], z[mask]

# 3. Create grid
res = 1.0  # Resolution in meters
xmin, xmax, ymin, ymax = xg.min(), xg.max(), yg.min(), yg.max()
xi = np.arange(xmin, xmax, res)
yi = np.arange(ymin, ymax, res)
xi, yi = np.meshgrid(xi, yi)

# 4. Interpolate Z values
zi = griddata((xg, yg), zg, (xi, yi), method='linear')

# 5. Save GeoTIFF using rasterio
transform = from_origin(xmin, ymax, res, res)
with rasterio.open(
    "/kaggle/working/lidar_dtm.tif",
    'w',
    driver='GTiff',
    height=zi.shape[0],
    width=zi.shape[1],
    count=1,
    dtype=zi.dtype,
    crs="EPSG:4326",  # replace with actual CRS if known
    transform=transform,
) as dst:
    dst.write(np.nan_to_num(zi, nan=-9999), 1)

print("DTM written to /kaggle/working/lidar_dtm.tif")


import numpy as np
import rasterio
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt

with rasterio.open("/kaggle/working/lidar_dtm.tif") as src:
    dtm = src.read(1)

enhanced = gaussian_laplace(dtm, sigma=2)

plt.figure(figsize=(10, 8))
plt.imshow(enhanced, cmap="gray")
plt.title("LiDAR DTM Enhanced (Ïƒ=2)")
plt.colorbar()
plt.scatter([140], [120], marker='o', color='red')  # example coordinates
plt.text(145, 115, "Possible Earthwork", color='red')
plt.show()


import json

# Define your JSON data
data = {
   
      "pipeline": [
        "/kaggle/working/autzen.las",
        {
          "type": "filters.smrf"
        },
        {
          "type": "filters.range",
          "limits": "Classification[2:2]"
        },
        {
          "type": "writers.gdal",
          "filename": "/kaggle/working/lidar_dtm.tif",
          "resolution": 1.0,
          "output_type": "idw"
        }
      ]
}

# Define the filename
filename = 'dtm_pipeline.json'

# Write the JSON data to the file
with open(filename, 'w') as file:
    json.dump(data, file, indent=4)

# Print a confirmation message
print(f"JSON data has been written to {filename}")


%%time
!pdal pipeline dtm_pipeline.json


%%time
import rasterio
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

# Load the DTM raster
with rasterio.open("/kaggle/working/lidar_dtm.tif") as src:
    dtm = src.read(1)
    profile = src.profile

# Compute hillshade
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.shade(dtm, cmap=plt.cm.gray, vert_exag=1.5, blend_mode='overlay')

# Plot hillshade
plt.figure(figsize=(10, 8))
plt.title("Hillshade (Simulated Light from NW)")
plt.imshow(hillshade)
plt.axis('off')
plt.show()


import numpy as np

# Inspect elevation range
print("DTM min:", np.min(dtm))
print("DTM max:", np.max(dtm))

# Optional: mask out nodata values (e.g. -9999)
dtm = np.where(dtm < -100, np.nan, dtm)

# Replace nan with mean or interpolate as fallback
dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm))


%%time
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.shade(dtm, cmap=plt.cm.gray, vert_exag=2.0, blend_mode='overlay')

plt.imshow(hillshade, cmap='gray')
plt.title("Hillshade (Gray Overlay)")
plt.axis('off')
plt.show()


%%time
hillshade = ls.hillshade(dtm, vert_exag=2, dx=1, dy=1)
plt.imshow(hillshade, cmap='gray')
plt.title("Pure Hillshade")
plt.axis('off')
plt.show()


%%time
plt.imshow(dtm, cmap='terrain')
plt.colorbar(label="Elevation (m)")
plt.title("Raw DTM Elevation")
plt.axis('off')
plt.show()


%%time
rgb = ls.shade_rgb(plt.cm.terrain(dtm / np.nanmax(dtm)), dtm)
plt.imshow(rgb)
plt.title("DTM + Hillshade RGB")
plt.axis('off')
plt.show()


import os

print("Files in /kaggle/working:")
print(os.listdir("/kaggle/working"))


%%time
import rasterio
import matplotlib.pyplot as plt

dtm_path = "/kaggle/working/lidar_dtm.tif"  # adjust if named differently

with rasterio.open(dtm_path) as src:
    dtm = src.read(1)
    plt.figure(figsize=(10, 8))
    plt.title("Digital Terrain Model (DTM)")
    plt.imshow(dtm, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()


import numpy as np

print("DTM Stats:")
print(f"Min: {np.min(dtm):.2f} m")
print(f"Max: {np.max(dtm):.2f} m")
print(f"Mean: {np.mean(dtm):.2f} m")


%%time
row = dtm.shape[0] // 2
plt.figure(figsize=(10, 4))
plt.plot(dtm[row])
plt.title(f"Elevation Profile along Row {row}")
plt.xlabel("Column")
plt.ylabel("Elevation (m)")
plt.grid(True)
plt.show()


%%time
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.shade(dtm, cmap=plt.cm.gray, vert_exag=2.0, blend_mode='overlay')

plt.imshow(hillshade, cmap='gray')
plt.title("Hillshade (Gray Overlay)")
plt.axis('off')
plt.show()


%%time
from scipy.ndimage import sobel

# Compute gradient in x and y
dx = sobel(dtm, axis=1)
dy = sobel(dtm, axis=0)
slope = np.sqrt(dx**2 + dy**2)

plt.figure(figsize=(10, 8))
plt.title("Slope Map (Gradient Magnitude)")
plt.imshow(slope, cmap="inferno")
plt.colorbar(label="Slope")
plt.axis('off')
plt.show()


%%time
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import difference_of_gaussians

# Remove NaNs if present (replace with local mean or 0 for test)
dtm_clean = np.nan_to_num(dtm, nan=0.0)

# Apply Difference of Gaussians filter
dog = difference_of_gaussians(dtm_clean, low_sigma=1, high_sigma=10)

# Visualize the result
plt.figure(figsize=(12, 6))
plt.imshow(dog, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(label="Anomaly Score")
plt.title("DoG Anomaly Map (Elevation Changes)")
plt.axis('off')
plt.show()


%%time
import matplotlib.pyplot as plt

# Choose a threshold empirically â€” try Â±0.1 as a starting point
threshold = 0.1

# Positive (elevated mounds), Negative (depressions)
positive_features = dog > threshold
negative_features = dog < -threshold

# Combine both for a general "anomaly mask"
combined_features = np.logical_or(positive_features, negative_features)

# Plot the mask
plt.figure(figsize=(12, 6))
plt.imshow(combined_features, cmap='gray')
plt.title("Detected Anomalies (Elevation Threshold Â±0.1)")
plt.axis('off')
plt.show()


%%time
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Label connected components (features)
labeled_features = label(combined_features)

# Measure properties (area, shape, etc.)
regions = regionprops(labeled_features)

# Show number of detected features
print(f"Number of features detected: {len(regions)}")

# Optional: Visualize
plt.figure(figsize=(10, 8))
plt.imshow(labeled_features, cmap='nipy_spectral')
plt.title("Labeled Features")
plt.axis('off')
plt.show()


%%time
# Filter out very small regions
min_area = 20  # pixels
large_regions = [r for r in regions if r.area >= min_area]

print(f"Number of large features (area â‰¥ {min_area} pixels): {len(large_regions)}")


%%time
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from rasterio import transform

# Convert labeled pixel regions into polygons
polygons = []
for region in large_regions:
    minr, minc, maxr, maxc = region.bbox
    # Convert pixel bbox to spatial bbox
    top_left = src.transform * (minc, minr)
    bottom_right = src.transform * (maxc, maxr)
    xmin, ymax = top_left
    xmax, ymin = bottom_right
    polygons.append(box(xmin, ymin, xmax, ymax))

# Create GeoDataFrame
gdf = gpd.GeoDataFrame({"id": range(len(polygons))}, geometry=polygons, crs=src.crs)

# Optional: reproject to EPSG:4326 (WGS84)
gdf = gdf.to_crs("EPSG:4326")

# Export to GeoJSON
gdf.to_file("detected_features.geojson", driver="GeoJSON")
print("âœ… Exported detected_features.geojson")


!ls /kaggle/working


for i, region in enumerate(large_regions[:5]):  # preview first 5
    print(f"Feature #{i}")
    print(f" - Area (pixels): {region.area}")
    print(f" - Centroid: {region.centroid}")
    print(f" - Bounding box: {region.bbox}")
    print("-------------")


%%time
import folium
import geopandas as gpd

# Load the GeoJSON
gdf = gpd.read_file("detected_features.geojson")

# Get the center of the features (lat, lon)
center = list(gdf.geometry.unary_union.centroid.coords)[0][::-1]  # [lat, lon]

# Create a Folium map centered on the features
m = folium.Map(location=center, zoom_start=16)

# Add the GeoJSON overlay
folium.GeoJson(gdf).add_to(m)

# Display the map
m


%%time
m.save("map.html")

from IPython.display import IFrame
IFrame("map.html", width=700, height=500)


%%time
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load NIR and Red bands
with rasterio.open('/kaggle/input/sentinel2-bands/2025-05-01-00_00_2025-05-01-23_59_Sentinel-2_L2A_True_color.tiff') as nir_src:
    nir = nir_src.read(1).astype('float32')
    nir_meta = nir_src.meta

with rasterio.open('/kaggle/input/sentinel2-bands/2025-05-01-00_00_2025-05-01-23_59_Sentinel-2_L2A_B04_(Raw).tiff') as red_src:
    red = red_src.read(1).astype('float32')

# Avoid divide by zero
ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

# Plot NDVI
plt.figure(figsize=(8, 6))
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title('NDVI from Sentinel-2')
plt.axis('off')
plt.show()



%%time
import geopandas as gpd

# Load the GeoJSON file
gdf = gpd.read_file("detected_features.geojson")

# If you want to preview the data:
print(gdf.head())

# Optional: simplify geometry to WKT for CSV
gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.wkt)

# Save to CSV
gdf.to_csv("detected_features.csv", index=False)

print("âœ… Exported detected_features.csv")


%%time
import json
import csv

# Function to convert GeoJSON to CSV
def geojson_to_csv(geojson_file, csv_file):
    # Read the GeoJSON file
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Extract the features
    features = geojson_data.get('features', [])

    # Define the CSV header
    header = ['id', 'geometry_type', 'latitude', 'longitude', 'properties']

    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write the data rows
        for feature in features:
            feature_id = feature.get('id', '')
            geometry_type = feature['geometry']['type']
            coordinates = feature['geometry']['coordinates']
            properties = feature.get('properties', {})

            # Handle different geometry types
            if geometry_type == 'Point':
                latitude, longitude = coordinates
            elif geometry_type == 'MultiPoint':
                latitude, longitude = coordinates[0]
            elif geometry_type == 'LineString':
                latitude, longitude = coordinates[0]
            elif geometry_type == 'MultiLineString':
                latitude, longitude = coordinates[0][0]
            elif geometry_type == 'Polygon':
                latitude, longitude = coordinates[0][0]
            elif geometry_type == 'MultiPolygon':
                latitude, longitude = coordinates[0][0][0]
            else:
                latitude, longitude = None, None

            # Write the row
            writer.writerow([feature_id, geometry_type, latitude, longitude, properties])

# Example usage
geojson_file = '/kaggle/working/detected_features.geojson'
csv_file = '/kaggle/working/features_centroids.csv'
geojson_to_csv(geojson_file, csv_file)


# Verify the CSV file
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Display the first few rows of the DataFrame
df.head()

