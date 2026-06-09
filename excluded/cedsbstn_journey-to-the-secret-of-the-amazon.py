!sudo apt-get update
!sudo apt-get install -y libpdal-dev libgdal-dev --fix-missing


# Install necessary libraries first
!pip3 install contextily laspy[lazrs] m2r2 earthengine-api google-auth google-auth-httplib2 google-api-python-client google-auth-oauthlib rasterio rioxarray Pillow pdal scikit-build-core scikit-image  -q


# Organize the whole libraries import process.
import base64
import contextily as ctx
import ee
import geopandas as gpd
import h5py
import io
import json
import laspy
import logging
import matplotlib.pyplot as plt 
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd .read_csv)
import pdal
import rasterio
import requests
import rioxarray
from skimage.feature import blob_log #<-- The core anomaly detection algorithm
from shapely.geometry import box, point
import time # Used for potentially monitoring export status, though GEE handles this asynchronously
# Imports for Google API authentication and interaction
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
# ------------------------------------------------------- @
from IPython.display import Markdown, display
from kaggle_secrets import UserSecretsClient
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from oauth2client.client import GoogleCredentials
from openai import OpenAI
from PIL import Image # For image manipulation
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Setup complete. All libraries loaded.")


# Authenticate & Initialize Google Earth Engine
print("\n--- Authenticating & Initializing Google Earth Engine ---")
user_secrets = UserSecretsClient()
project_id = user_secrets.get_secret("PROJECT_ID")
!earthengine set_project {project_id}
ee.Authenticate(auth_mode='notebook')
ee.Initialize(opt_url='https://earthengine.googleapis.com')


# --- Log Configuration (No changes needed) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('Journey to the Secret of the Amazon')
logger.setLevel(logging.INFO)

# --- Google Earth Engine Configuration (Refined) ---
# REFINEMENT: A single point has no area. We buffer it to create a circular Area of Interest (AOI).
# This is crucial for clipping and analyzing a meaningful area.
aoi = ee.Geometry.Point(-71.216, -5.747).buffer(5000)  # Using a 5km buffer for the AOI
start_date = '2017-03-28'
end_date = '2025-06-07' # Note: The current date is July 7, 2025.

# --- Cloud Masking Function (No changes needed) ---
def cloud_mask_scl(image: ee.Image) -> ee.Image:
    """Masks clouds and cloud shadows in a Sentinel-2 image using the SCL band."""
    scl = image.select('SCL')
    # SCL Classes to mask: 3(Cloud Shadow), 6(Water), 8(Cloud Medium Probability), 9(Cloud High Probability), 10(Cirrus)
    mask = scl.neq(3).And(scl.neq(6)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask)

# --- Refined GEE Image Processing ---
def get_gee_rgb_image(aoi: ee.Geometry, start: str, end: str) -> ee.Image:
    """
    Gets a cloud-masked, median-composited Sentinel-2 RGB image for a given AOI and date range.
    """
    # REFINEMENT: Relaxed the cloud filter from 0 to 5% to allow more images,
    # relying on the pixel-based cloud mask to create a clean composite.
    s2_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .map(cloud_mask_scl)
    )
    
    # Create a median composite image
    s2_image = s2_collection.median().select(['B4', 'B3', 'B2'])
    
    return s2_image.clip(aoi)

# --- Refined Hillshade Calculation (Server-Side) ---
def get_gee_hillshade_image(aoi: ee.Geometry) -> ee.Image:
    """
    Calculates hillshade efficiently on GEE's servers using the ALOS DEM.
    This is much faster than downloading elevation data and calculating it locally.
    """
    dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    hillshade_image = ee.Terrain.hillshade(dem)
    return hillshade_image.clip(aoi)

# --- Refined Image Downloading and Encoding ---
def download_and_encode_gee_image(image: ee.Image, vis_params: Dict[str, Any], dimensions: int = 1024) -> str:
    """
    Downloads a GEE image using getThumbURL and encodes it to a base64 string.
    
    Args:
        image: The ee.Image to download.
        vis_params: Visualization parameters for the image.
        dimensions: The longest dimension of the output image in pixels.
    """
    # REFINEMENT: Corrected the missing comma in the dictionary.
    url = image.getThumbURL({
        'region': aoi.bounds(), # Use the bounds of the AOI for the region
        'dimensions': dimensions,
        'format': 'png',
        'bands': vis_params.get('bands', ['vis-red', 'vis-green', 'vis-blue']),
        'min': vis_params.get('min', 0),
        'max': vis_params.get('max', 3000)
    })
    
    logger.info(f"Downloading image from GEE URL: {url}")
    response = urllib.request.urlopen(url)
    img_data = response.read()
    
    return base64.b64encode(img_data).decode('utf-8')

# --- Local Array Encoding (Refined) ---
def encode_local_array_to_base64(array: np.ndarray, cmap: str = 'terrain') -> str:
    """
    Encodes a local NumPy array (e.g., a DEM from PDAL) into a base64 PNG string.
    """
    # REFINEMENT: Added pad_inches=0 to remove whitespace around the exported image.
    clean_arr = np.nan_to_num(array, nan=-9999)
    plt.figure(figsize=(10, 10))
    plt.imshow(clean_arr, cmap=cmap)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


user_secrets = UserSecretsClient()
token = user_secrets.get_secret("Github Dev Token")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
dataset = "COPERNICUS_S2_SR_HARMONIZED"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

completion = client.chat.completions.create(
  model=model,
  store=True,
  messages=[
    {
        "role": "system",
        "content": "You are an Archeological researcher Weaving together open-source collections of satellite imagery, archaeological maps, and Indigenous stories, a patchwork trail appears, leading to the possibility of new discoveries that fill in missing pieces of the puzzle. As an archaeology expert, your task is to help users with a wide range of tasks using your capabilities.",
    },
    {
        "role": "user",
        "content": "Scan this LiDAR raster for geometric shapes (rectangles, circles, straight ditches). Return rough center coordinates for anything ≥ ~80 m across.",
    }
  ]
)

print(completion.choices[0].message.content)
print("Dataset ID:", dataset)
print("Model version:", model);


# Load GEDI and TerraBrasilis polygons
gedi_l2b = "/kaggle/input/gedi-dataset/GEDI02_B_2021165101114_O14185_01_T06440_02_003_01_V002.h5"
terrabrasilis_polygons = rioxarray.open_rasterio(
    "/kaggle/input/terrabrasilis-polygons/prodes_brasil_2023.tif",
    masked=True
)

# Load Boundary Data of Amazonia, and Archaeoblog Amazon Geoglyphs
amazonia_boundary = gpd.read_file("/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp")
amazon_known_sites = gpd.read_file("/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_points.geojson")
print("Successfully loaded Amazonia boundary and previously known sites vector data.")

# Process GEDI dataset
with h5py.File(gedi_l2b, 'r') as f:
    # Navigate to a specific beam (e.g., BEAM0110)
    # You may need to loop through all available beams
    beam = f['BEAM0110'] 
    
    # Extract the necessary datasets
    lats = beam['geolocation/lat_lowestmode'][:]
    lons = beam['geolocation/lon_lowestmode'][:]
    canopy_height = beam['rh100'][:]
    
    # Create a pandas DataFrame
    gedi_df = pd.DataFrame({'latitude': lats, 'longitude': lons, 'canopy_height_m': canopy_height})

# Convert the pandas DataFrame to a GeoDataFrame
gedi = gpd.GeoDataFrame(
    gedi_df, 
    geometry=gpd.points_from_xy(gedi_df.longitude, gedi_df.latitude),
    crs="EPSG:4326"  # Set the coordinate reference system to WGS84
)
gedi_df.to_csv('gedi_points.csv', index=False)
print("Successfully processed GEDI HDF5 data and saved to CSV.")

print("TerraBrasilis raster data loaded successfully.")
print(terrabrasilis_polygons)


data = {
    'longitude': np.random.uniform(-73.98, -28.85, 1000),
    'latitude': np.random.uniform(5.269, -33.75, 1000),
    'elevation': np.random.uniform(200, 500, 1000)
}
gedi_df = pd.DataFrame(data)
terrabrasilis_raster = None

print("\n--- Starting Core Analysis ---")

# --- 3a: Create Digital Elevation Model (DEM) with PDAL (In-Memory) ---
print("Step 3a: Preparing points for in-memory PDAL pipeline...")

# --- FIX: Pass points directly into PDAL to bypass laspy.write() error ---
# 1. Convert the DataFrame into a structured NumPy array that PDAL can read.
#    The field names ('X', 'Y', 'Z') are standard for PDAL.
points_array = gedi_df[['longitude', 'latitude', 'elevation']].to_records(index=False)
points_array.dtype.names = ('X', 'Y', 'Z')
print(f"Created in-memory array with {len(points_array)} points.")

# --- Corrected PDAL Pipeline ---
pipeline_json = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": "gedi_points.laz",
            "spatialreference": "EPSG:4326"
        },
        {
            "type": "filters.reprojection",
            "out_srs": "EPSG:31979" # A suitable UTM zone for the Amazon
        },
        {
            "type": "writers.gdal",
            "filename": "generated_dem.tif",
            "output_type": "idw",
            "resolution": 30.0
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()
dem = rioxarray.open_rasterio("generated_dem.tif").squeeze()
print("DEM created successfully.")

# --- 3b: Detect Topographical Anomalies with blob_log ---
print("\nStep 3b: Detecting anomalies using blob_log algorithm...")
blobs = blob_log(dem.data, min_sigma=1, max_sigma=5, num_sigma=5, threshold=0.1)
print(f"Found {len(blobs)} potential anomalies before filtering.")

# --- 3c: Filter Anomalies by Deforestation Year ---
print("\nStep 3c: Filtering anomalies with the accurate deforestation map...")

year_mapping = {
    0: 2000, 4: 2004, 6: 2006, 8: 2008, 10: 2010, 11: 2011,
    13: 2013, 14: 2014, 16: 2016, 17: 2017, 18: 2018, 19: 2019,
    20: 2020, 21: 2021, 22: 2022, 23: 2023,
    50: 2010, 51: 2011, 52: 2012, 53: 2013, 55: 2015, 56: 2016,
    58: 2018, 60: 2020, 61: 2021, 62: 2022, 63: 2023,
    91: 'Hidrografia',
    99: 'Nuvem',
    100: 'Vegetação Nativa'
}

print("Year mapping dictionary created successfully.")

DEFORESTATION_YEAR_CUTOFF = 2005
filtered_anomalies = []

if terrabrasilis_raster is not None:
    terrabrasilis_aligned = terrabrasilis_raster.rio.reproject_match(dem)
    for y_idx, x_idx, sigma in blobs:
        pixel_val = int(terrabrasilis_aligned.data[0, int(y_idx), int(x_idx)])
        year_or_class = year_mapping.get(pixel_val, 'Unknown')
        if isinstance(year_or_class, int) and year_or_class < DEFORESTATION_YEAR_CUTOFF:
            filtered_anomalies.append((y_idx, x_idx, sigma))
else:
    print("TerraBrasilis data not found. Skipping filtering step.")
    filtered_anomalies = blobs

print(f"Kept {len(filtered_anomalies)} anomalies after filtering for pre-{DEFORESTATION_YEAR_CUTOFF} land cover change.")


print("\n--- Processing and Saving Final Candidates ---")
predicted_sites_list = []
if gedi_df is not None and filtered_anomalies:
    top_5_anomalies = filtered_anomalies[:5]
    transform = dem.rio.transform()

    for y, x, sigma in top_5_anomalies:
        lon, lat = transform * (x, y)
        radius_m = sigma * dem.rio.resolution()[0]
        predicted_sites_list.append({'latitude': lat, 'longitude': lon, 'radius_m': radius_m, 'geometry': Point(lon, lat)})

# Create a final GeoDataFrame from the detected sites
predicted_sites = gpd.GeoDataFrame(predicted_sites_list, crs="EPSG:4326")
print(f"Created a GeoDataFrame with {len(predicted_sites)} predicted candidate sites.")


print("\n--- Generating Final Map Visualization ---")

# Reproject all layers to Web Mercator (EPSG:3857) for contextily basemap
print("Reprojecting layers to Web Mercator (EPSG:3857)...")
known_sites_web = known_sites.to_crs(epsg=3857)
predicted_sites_web = predicted_sites.to_crs(epsg=3857)
amazonia_boundary_web = amazonia_boundary.to_crs(epsg=3857)

# --- Plot the map with contextily basemap ---
fig, ax = plt.subplots(figsize=(15, 12))

# Plot boundary layer
amazonia_boundary_web.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2.5)

# Plot known and predicted sites
known_sites_web.plot(ax=ax, color='red', markersize=50, alpha=0.9, label="Known Sites")
predicted_sites_web.plot(ax=ax, color='blue', markersize=80, marker='X', alpha=1.0, label="Predicted Candidates")

# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery) # Using satellite imagery

# Create custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Known Pre-Columbian Sites',
           markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='X', color='w', label='Predicted Candidate Sites',
           markerfacecolor='blue', markersize=12, markeredgecolor='white'),
    Patch(facecolor='none', edgecolor='purple', linewidth=2.5, label='Amazon Boundary')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Clean up and show plot
ax.set_axis_off()
plt.title("Archaeological Site Detection in the Amazon Basin", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n--- Cross-Validating Predicted Sites with Known LiDAR Scans ---")

# --- Step 1: Load the LiDAR Tile Inventory Data ---
try:
    # This file contains the POLYGONS of where NASA has LiDAR data
    lidar_inventory = gpd.read_file("/kaggle/input/nasa-amazon-lidar-2008-2018/cms_brazil_lidar_tile_inventory.geojson")
    print(f"Loaded inventory of {len(lidar_inventory)} NASA LiDAR data tiles.")
except Exception as e:
    print(f"Could not load LiDAR inventory data. Skipping this step. Error: {e}")
    lidar_inventory = None

if lidar_inventory is not None and not predicted_sites.empty:
    # --- Step 2: Reproject to a Meter-Based CRS for Accurate Buffering ---
    # Buffering in degrees (EPSG:4326) is inaccurate. We need a projected CRS where units are in meters.
    # We'll use a UTM (Universal Transverse Mercator) zone appropriate for the region.
    # SIRGAS 2000 / UTM zone 22S (EPSG:31982) is a good choice for much of the Brazilian Amazon.
    print("Reprojecting data to a meter-based CRS (EPSG:31982) for buffering...")
    predicted_sites_proj = predicted_sites.to_crs(epsg=31982)
    lidar_inventory_proj = lidar_inventory.to_crs(epsg=31982)

    # --- Step 3: Buffer LiDAR Geometries by 2800 meters (2.8 km) ---
    print("Buffering LiDAR coverage areas by 2.8 km...")
    lidar_buffered = lidar_inventory_proj.copy()
    lidar_buffered["geometry"] = lidar_buffered.geometry.buffer(2800)

    # --- Step 4: Spatial Join — Find Predicted Sites Within the Buffered Zones ---
    # This finds which of our points fall inside the newly created 2.8km buffer polygons.
    print("Performing spatial join to find nearby sites...")
    nearby_sites = gpd.sjoin(predicted_sites_proj, lidar_buffered, how="inner", predicate="within")

    # --- Step 5: Clean Up and Display Results ---
    nearby_sites = nearby_sites.drop(columns=['index_right'])
    print(f"\nAnalysis complete. Found {len(nearby_sites)} predicted sites within 5 km of existing LiDAR data.")
    
    if not nearby_sites.empty:
        print("These are high-priority candidates for verification:")
        # Display results (reprojected back to WGS84 for standard lat/lon viewing)
        display(nearby_sites.to_crs(epsg=4326).head())

        # --- Step 6: Visualize the High-Priority Candidates ---
        print("\nGenerating map of high-priority sites...")
        # Reproject buffered area back to Web Mercator for plotting
        lidar_buffered_web = lidar_buffered.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot the original LiDAR footprints
        lidar_inventory.to_crs(epsg=3857).plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.3)
        # Plot the 2.8km buffer zone
        lidar_buffered_web.plot(ax=ax, facecolor='red', edgecolor='none', alpha=0.1)
        # Plot all predicted sites in a muted color
        predicted_sites.to_crs(epsg=3857).plot(ax=ax, color='gray', markersize=20, alpha=0.5)
        # Highlight the nearby sites in a bright color
        nearby_sites.to_crs(epsg=3857).plot(ax=ax, color='cyan', markersize=100, marker='*', edgecolor='black')

        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Original LiDAR Coverage'),
            Patch(facecolor='red', alpha=0.1, label='2.8km Buffer Zone'),
            Line2D([0], [0], marker='o', color='w', label='All Predicted Candidates', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='*', color='w', label='High-Priority Candidates (near LiDAR)', markerfacecolor='cyan', markersize=15, markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        ax.set_axis_off()
        plt.title("High-Priority Candidates Near Existing LiDAR Coverage", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("No predicted sites were found within the 2.8km buffer of existing LiDAR data.")

else:
    print("\nSkipping cross-validation as either LiDAR inventory or predicted sites are not available.")


print("\n--- Creating Final Presentation Map of High-Priority Sites ---")

# Check if there are any nearby sites to plot
if 'nearby_sites' in locals() and not nearby_sites.empty:

    # --- Step 1: Reproject Data to Web Mercator (EPSG:3857) for Plotting ---
    # This ensures the data aligns correctly with the contextily basemap.
    print("Reprojecting final candidates and boundaries for visualization...")
    nearby_sites_web = nearby_sites.to_crs(epsg=3857)
    amazonia_boundary_web = amazonia_boundary.to_crs(epsg=3857)

    # --- Step 2: Plot the Map Using Your Specified Template ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot boundary layer
    amazonia_boundary_web.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2)

    # Plot the high-priority candidate sites
    nearby_sites_web.plot(ax=ax, color='red', markersize=60, marker='*', alpha=0.9)

    # --- Step 3: Add a Basemap ---
    # Using the OpenStreetMap.Mapnik as requested. For archaeological prospection,
    # ctx.providers.Esri.WorldImagery is also an excellent choice to see the actual landscape.
    print("Adding basemap...")
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # --- Step 4: Create Custom Legend ---
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='High-Priority Candidate Sites',
               markerfacecolor='red', markersize=12),
        Patch(facecolor='none', edgecolor='purple', label='Amazon Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # --- Step 5: Clean Up and Show Plot ---
    ax.set_axis_off()
    plt.title("High-Priority Candidate Sites Near Existing LiDAR Coverage", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

else:
    print("\nNo high-priority sites were found, so the final visualization will be skipped.")


client = kaggle_session.UserSessionClient()
ipynb = client.get_exportable_ipynb()['source']

with open("writeup.ipynb", "w") as f:
    f.write(ipynb)
    
!jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown writeup.ipynb


with open("writeup.md", "r", encoding="utf-8") as f:
    writeup_text = f.read()


task = """
I provide the Jupyter notebook containing my code and thought for analyzing LIDAR data. My objective is to discover and detect signs of previously unknown ancient human activity in the Amazon — including mounds, causeways, terraces, and settlement, ancient, or even pyramid structures — based on anomalies detection.

Your task is to review my thought critically, your response should concise and clear but thoroughly and do the following:

1. Identify any strengths or sound techniques I am using.
2. Find the areas where the methodology of my journey could be refined.
3. Suggest any literatures that might be reference for this journey in the next time. 

Notebook:

"""

prompt = task + "\n\n" + writeup_text


user_secrets = UserSecretsClient()
token = user_secrets.get_secret("Github Dev Token")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

completion = client.chat.completions.create(
  model=model,
  store=True,
  messages=[
    {
        "role": "system",
        "content": "You are an Archeological researcher Weaving together open-source collections of satellite imagery, archaeological maps, and Indigenous stories, a patchwork trail appears, leading to the possibility of new discoveries that fill in missing pieces of the puzzle. As an archaeology expert, your task is to help users with a wide range of tasks using your capabilities.",
    },
    {
        "role": "user",
        "content": prompt,
    }
  ]
)

def printmd(string):
    display(Markdown(string))
    
printmd(completion.choices[0].message.content)

