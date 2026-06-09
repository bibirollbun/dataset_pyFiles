!pip install sentence-transformers faiss-cpu -q


import ee
import geemap
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from branca.colormap import linear
import json
import cv2
import os
import io
import base64
import openai
from PIL import Image
from kaggle_secrets import UserSecretsClient
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
import re
from typing import List, Tuple, Union, Optional
from shapely.geometry import Point

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from skimage.feature import canny
from skimage.transform import hough_line
from skimage.measure import label, regionprops_table
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sentence_transformers import SentenceTransformer
import faiss
import tempfile


openai_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_key)
print("OpenAI Initialized")

iam_service_account = UserSecretsClient().get_secret('IAM_ACCOUNT')
ee_credentials_json = UserSecretsClient().get_secret('EE_CREDENTIALS')

with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
    temp_file.write(ee_credentials_json)
    temp_path = temp_file.name

ee_creds = ee.ServiceAccountCredentials(iam_service_account, temp_path)
ee.Initialize(ee_creds)
print("EE Connected")


console = Console()

def print_formatted(text: str, tag: str):
    """
    Prints text to the console with rich formatting.

    Args:
        text (str): The string to print.
        tag (str): 'h1' (main heading), 'h2' (sub-heading),
                   'p' (paragraph), 'success', 'warning', 'error'.
    """
    tag = tag.lower()

    if tag == 'h1':
        # A horizontal Rule with text is more compatible across terminals than a Panel.
        console.print(Rule(f"[bold bright_yellow]{text.upper()}[/]", style="white"))
        
    elif tag == 'h2':
        # A Markdown-style heading with a rule line above it for steps.
        console.print(Rule(style="dim white"))
        console.print(Markdown(f"## {text}"))
        
    elif tag == 'p':
        # An indented message for progress updates.
        console.print(f"  - {text}")
        
    elif tag == 'success':
        # Green, bold text with a checkmark.
        console.print(f"[bold green]âœ… {text}[/bold green]")

    elif tag == 'warning':
        # Yellow text with a warning sign.
        console.print(f"[yellow]âš ï¸� Warning:[/] {text}")

    elif tag == 'error':
        # Bold red text with a cross mark.
        console.print(f"[bold red]â�Œ Error:[/] {text}")
        
    else:
        # If the tag is unrecognized, just print the text as is.
        console.print(text)


def parse_dms_string(dms_string: str) -> Optional[List[float]]:
    """
    Parses a single string containing latitude and longitude in DMS format.
    Example input: "9Â°02'28\\"S 70Â°37'59\\"W"
    """
    # This regex is designed to find two DMS-formatted coordinates in a string
    pattern = re.compile(
        r"""
        (\d{1,3})[Â°\s]+(\d{1,2})['\s]+(\d{1,2}(?:\.\d+)?)["\s]*([NS])  # Latitude part
        \s*[,]?\s* # Optional separator
        (\d{1,3})[Â°\s]+(\d{1,2})['\s]+(\d{1,2}(?:\.\d+)?)["\s]*([EW])  # Longitude part
        """, re.VERBOSE | re.IGNORECASE)

    match = pattern.search(dms_string)

    if not match:
        return None

    groups = match.groups()

    # --- Latitude Calculation ---
    lat_deg = float(groups[0])
    lat_min = float(groups[1])
    lat_sec = float(groups[2])
    lat_dir = groups[3].upper()
    
    # DD = Degrees + (Minutes/60) + (Seconds/3600)
    latitude_dd = lat_deg + (lat_min / 60) + (lat_sec / 3600)
    if lat_dir == 'S':
        latitude_dd *= -1

    # --- Longitude Calculation ---
    lon_deg = float(groups[4])
    lon_min = float(groups[5])
    lon_sec = float(groups[6])
    lon_dir = groups[7].upper()

    longitude_dd = lon_deg + (lon_min / 60) + (lon_sec / 3600)
    if lon_dir == 'W':
        longitude_dd *= -1
        
    return [latitude_dd, longitude_dd]


def format_map_center(coordinate: Union[str, list, tuple]) -> Optional[List[float]]:
    """
    Converts a coordinate from various formats into a [latitude, longitude] list.
    Handles lists, tuples, simple strings, and complex DMS strings.
    """
    # 1. Handle lists and tuples
    if isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
        try:
            lat, lon = float(coordinate[0]), float(coordinate[1])
            if abs(lat) > 90 and abs(lon) <= 90:
                 print("Warning: Input order appears to be [longitude, latitude]. Swapping.")
                 return [lon, lat]
            return [lat, lon]
        except (ValueError, TypeError):
             pass # Will be handled by other parsers or the final error message

    # 2. Handle strings
    if isinstance(coordinate, str):
        # Try the DMS parser first if degree/quote symbols are present
        if 'Â°' in coordinate or "'" in coordinate:
            result = parse_dms_string(coordinate)
            if result:
                return result
        
        # Fallback to simple number extraction for formats like "-9.04, -70.65"
        numbers = re.findall(r'-?\d+\.?\d*', coordinate)
        if len(numbers) == 2:
            lat, lon = float(numbers[0]), float(numbers[1])
            if abs(lat) > 90 and abs(lon) <= 90:
                print("Warning: Input order appears to be [longitude, latitude]. Swapping.")
                return [lon, lat]
            return [lat, lon]

    # 3. If all parsing fails
    print(f"Error: Could not parse coordinate: {coordinate}")
    return None


print("\n" + "="*80)
print("PHASE 1: DATA CURATION & PREPROCESSING")
print("="*80)

# 1. Input string in any format
dms_string = '-9.0425, -70.6575'
radius_m = 1900

# 2. Use the function to parse it. This returns [latitude, longitude]
formatted_coords = format_map_center(dms_string)
print_formatted(f"Formatted coordinates [lat, lon]: {formatted_coords}", 'p')

# 3. To use with Earth Engine, reverse the list to [longitude, latitude]
ee_ready_coords = formatted_coords[::-1] 
print(f"Reversed for Earth Engine [lon, lat]: {ee_ready_coords}")

# 4. Now create the Earth Engine Point object
poi = ee.Geometry.Point(ee_ready_coords)
aoi = poi.buffer(radius_m)
print(f"âœ… Focused AOI created with a radius of {radius_m}.")

print("\n--- VISUAL 1.1: Area of Interest ---")
map_center = formatted_coords
m_aoi = folium.Map(location=map_center, zoom_start=13)
folium.GeoJson(
    geemap.ee_to_geojson(aoi),
    style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.1, 'weight': 2},
    name='Area of Interest (AOI)'
).add_to(m_aoi)
folium.LayerControl().add_to(m_aoi)
display(m_aoi)
print("A map showing the circular Area of Interest (AOI) has been displayed.")


print("\n--> Step 1.2: Acquiring and Preparing Data...")
gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
search_space = gfc.select('lossyear').gt(0).clip(aoi).selfMask()
print("    - Historical deforestation data (Hansen GFC) loaded.")

s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterDate('2023-01-01', '2025-01-01') \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))

print(f"    - Found {s2_collection.size().getInfo()} potential Sentinel-2 images with <15% cloud cover.")

sentinel2_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
s2_projection = sentinel2_image.select('B4').projection()
image_id = sentinel2_image.id().getInfo()
image_date = ee.Date(sentinel2_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print(f"    - Selected best Sentinel-2 image: {image_id} (Date: {image_date})")

srtm_dem = ee.Image('USGS/SRTMGL1_003')
srtm_reprojected = srtm_dem.reproject(crs=s2_projection)
print("    - Acquired and reprojected SRTM DEM (Elevation data).")

sentinel2_unmasked = sentinel2_image.clip(aoi)
print("âœ… Core image data acquired and prepared.")

print("\n--> Step 1.3: Grid Creation and Feature Extraction...")
sentinel2_masked = sentinel2_image.updateMask(sentinel2_image.select('B2').mask())
srtm_masked = srtm_reprojected.updateMask(sentinel2_image.select('B2').mask())

# Derived features
ndvi = sentinel2_masked.normalizedDifference(['B8', 'B4']).rename('ndvi')
terrain = ee.Terrain.products(srtm_masked)

combined_image = ee.Image.cat([
    sentinel2_masked.select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'nir']),
    ndvi,
    srtm_masked.select('elevation'),
    terrain.select(['slope', 'aspect'])
])
print("    - Created combined image with 7 feature layers for analysis.")

# Grid definition
grid = aoi.coveringGrid(proj=s2_projection, scale=50)
grid_with_ids = grid.map(lambda feature: feature.set('grid_cell_id', feature.id()))
print(f"    - Generated a 50x50m grid with {grid.size().getInfo()} potential cells.")


print("\n--- VISUAL 1.3: Analysis Grid and Initial Data ---")
print("\n")
map_with_grid = folium.Map(location=map_center, zoom_start=14)
# Add satellite basemap
map_id_s2 = sentinel2_unmasked.getMapId({'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']})
folium.TileLayer(
    tiles=map_id_s2['tile_fetcher'].url_format,
    attr='Google Earth Engine',
    name='Sentinel-2 RGB',
    overlay=True,
    control=True
).add_to(map_with_grid)

# Add deforestation mask
map_id_mask = search_space.getMapId({'palette': 'FF0000'})
folium.TileLayer(
    tiles=map_id_mask['tile_fetcher'].url_format,
    attr='Google Earth Engine',
    name='Deforestation Mask',
    overlay=True,
    control=True
).add_to(map_with_grid)

# Add grid layer
folium.GeoJson(
    geemap.ee_to_geojson(grid),
    style_function=lambda x: {'color': 'yellow', 'fillOpacity': 0.0, 'weight': 0.5},
    name='50m Grid'
).add_to(map_with_grid)
folium.LayerControl().add_to(map_with_grid)
display(map_with_grid)
print("A map showing the analysis grid and deforestation mask overlaid on the satellite image has been displayed.")


print("\n" + "="*80)
print("PHASE 0: OPENAI SETUP")
print("="*80)

print("\n--> Step 1.6: Initial Scene Description by LLM...")

if client:
    try:
        print("    - Preparing full AOI image for initial analysis...")
        # Download the full AOI image as a numpy array
        aoi_rgb_array = geemap.ee_to_numpy(sentinel2_unmasked.select(['B4', 'B3', 'B2']), region=aoi)
        
        # Scale the image to 0-255 for JPEG encoding
        aoi_rgb_array_scaled = (np.clip(aoi_rgb_array, 0, 3000) / 3000 * 255).astype(np.uint8)
        img = Image.fromarray(aoi_rgb_array_scaled)
        
        # Encode image to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = "Describe the surface features of this satellite image of a region in the Amazon basin in plain English. Note the presence of rivers, forests, and any signs of human activity like deforestation or agriculture."
        
        print(f"    - Sending prompt to GPT-4.1 -    ")
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at analyzing satellite images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        response_text = completion.choices[0].message.content
        
        print("\n--- GPT-4.1 Initial Scene Analysis ---")
        print(f"Model Version: {completion.model}")
        print(f"Dataset ID: {image_id}")
        print("\nLLM Description:")
        print(response_text)
        print("-" * 35)

    except Exception as e:
        print(f"    - â�Œ Failed to get initial LLM analysis. Error: {e}")
else:
    print("    - Skipping initial LLM analysis because OpenAI client is not available.")



# Feature extraction
reducers = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
feature_vectors = combined_image.reduceRegions(collection=grid_with_ids, reducer=reducers, scale=10)
print("    - Defined feature extraction process (mean and stdDev per cell).")

print("\n--> Step 1.4: Downloading and Cleaning Data...")
print("    - Downloading features from Earth Engine... (This may take a moment)")
geojson_features = geemap.ee_to_geojson(feature_vectors)
s2_crs = sentinel2_image.select('B4').projection().crs().getInfo()
feature_gdf = gpd.GeoDataFrame.from_features(geojson_features['features'], crs=s2_crs)
feature_gdf['grid_cell_id'] = [f['id'] for f in geojson_features['features']]
print(f"    - Download complete. Created GeoDataFrame with CRS: {s2_crs}")

feature_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_gdf.dropna(inplace=True)
print("    - Data cleaned (removed NaN/infinite values).")

if feature_gdf.empty:
    raise ValueError("CRITICAL ERROR: The final DataFrame is empty after cleaning. Cannot proceed.")
else:
    print(f"\nâœ… Successfully created final GeoDataFrame with {len(feature_gdf)} valid cells.")
    print("    - Sample of the final feature data:")
    display(feature_gdf.head())

print("\n--- VISUAL 1.4: Distribution of Key Features ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
feature_gdf['ndvi_mean'].hist(bins=50, ax=axes[0], color='lightgreen')
axes[0].set_title('NDVI Distribution')
feature_gdf['elevation_mean'].hist(bins=50, ax=axes[1], color='tan')
axes[1].set_title('Elevation Distribution (m)')
feature_gdf['slope_mean'].hist(bins=50, ax=axes[2], color='lightblue')
axes[2].set_title('Slope Distribution (degrees)')
plt.tight_layout()
plt.show()
print("Histograms showing the distribution of NDVI, elevation, and slope across the grid cells.")
print("âœ… Phase 1 Complete.")


print("\n" + "="*80)
print("PHASE 2: THE PARALLEL ANALYSIS ENGINE")
print("="*80)

print("\n--> Step 2.1 (Path B): Statistical Anomaly Detection...")
features_for_stats = feature_gdf.drop(columns=['geometry', 'grid_cell_id'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_stats)
print(f"    - Standardized {features_scaled.shape[1]} features for {features_scaled.shape[0]} cells.")

iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
iforest_scores_inverted = -1 * iso_forest.fit(features_scaled).score_samples(features_scaled)
print("    - Calculated Isolation Forest scores.")

lof = LocalOutlierFactor(n_neighbors=50, contamination='auto', n_jobs=-1)
lof_scores_inverted = -1 * lof.fit_predict(features_scaled)
print("    - Calculated Local Outlier Factor scores.")

minmax_scaler = MinMaxScaler()
feature_gdf['iforest_score'] = minmax_scaler.fit_transform(iforest_scores_inverted.reshape(-1, 1))
feature_gdf['lof_score'] = minmax_scaler.fit_transform(lof_scores_inverted.reshape(-1, 1))
feature_gdf['anomaly_score'] = (feature_gdf['iforest_score'] + feature_gdf['lof_score']) / 2
print("âœ… Statistical anomaly scores calculated and added to GeoDataFrame.")

print("\n--- VISUAL 2.1: Statistical Anomaly Score Distributions ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.histplot(feature_gdf['iforest_score'], bins=50, ax=axes[0], color='coral', kde=True).set_title('Isolation Forest Scores')
sns.histplot(feature_gdf['lof_score'], bins=50, ax=axes[1], color='skyblue', kde=True).set_title('Local Outlier Factor Scores')
sns.histplot(feature_gdf['anomaly_score'], bins=50, ax=axes[2], color='mediumpurple', kde=True).set_title('Combined Anomaly Scores')
plt.tight_layout()
plt.show()
print("Histograms showing the distribution of the calculated anomaly scores.")


print("\n--> Step 2.2 (Path B): Geometric & Segmentation Analysis...")
# Use a quantile of 0.95 to select the top 5% of statistically anomalous cells
anomaly_threshold = feature_gdf['anomaly_score'].quantile(0.95)
candidate_gdf = feature_gdf[feature_gdf['anomaly_score'] >= anomaly_threshold].sort_values(by='anomaly_score', ascending=False)
print(f"    - Selected {len(candidate_gdf)} candidate cells for image analysis (top 5%).")

feature_gdf['segmentation_score'] = 0.0
feature_gdf['geometric_score'] = 0.0

# Define how many of the top candidates to process for deep analysis
NUM_TO_PROCESS = int(len(candidate_gdf) * 0.05)
candidate_ids_to_process = candidate_gdf['grid_cell_id'].tolist()[:NUM_TO_PROCESS]
print(f"    - Starting deep analysis for the top {len(candidate_ids_to_process)} candidates...")

gdf_crs = feature_gdf.crs.to_string()

print("\n--- VISUAL 2.2: Image Patch Analysis for Top Candidates ---")

for i, cell_id in enumerate(candidate_ids_to_process):
    print(f"\n    Processing Candidate {i+1}/{len(candidate_ids_to_process)} (ID: {cell_id})...")
    try:
        row = candidate_gdf[candidate_gdf['grid_cell_id'] == cell_id].iloc[0]
        local_geometry = row.geometry
        centroid = local_geometry.centroid
        ee_centroid = ee.Geometry.Point([centroid.x, centroid.y], proj=gdf_crs)
        patch_region = ee_centroid.buffer(40).bounds() # Slightly larger patch

        image_for_patch = sentinel2_unmasked.select(['B4', 'B3', 'B2']).unmask(0)
        patch_array = geemap.ee_to_numpy(image_for_patch, region=patch_region)

        if patch_array is None or np.all(patch_array == 0):
            print(f"    - Skipped cell {cell_id}: Patch falls in a no-data area.")
            continue

        patch_array_scaled = patch_array.astype(np.float32)
        max_val = patch_array_scaled.max()
        if max_val == 0: continue
        patch_array_scaled /= max_val
        patch_array_scaled *= 255.0
        patch_array_uint8 = patch_array_scaled.astype(np.uint8)

        # KMeans Segmentation
        pixels = patch_array_uint8.reshape(-1, 3)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto').fit(pixels)
        segmented_patch = kmeans.labels_.reshape(patch_array.shape[0], patch_array.shape[1])
        labeled_segments = label(segmented_patch)
        props = regionprops_table(labeled_segments, properties=('solidity',))
        seg_score = pd.DataFrame(props)['solidity'].max() if props and props['solidity'].size > 0 else 0

        # Canny & Hough Line Detection
        gray_patch = cv2.cvtColor(patch_array_uint8, cv2.COLOR_RGB2GRAY)
        edges = canny(gray_patch, sigma=1.5)
        h_lines, _, _ = hough_line(edges)
        geo_score = len(h_lines) * 0.1 # Scaled score

        feature_gdf.loc[feature_gdf['grid_cell_id'] == cell_id, 'segmentation_score'] = seg_score
        feature_gdf.loc[feature_gdf['grid_cell_id'] == cell_id, 'geometric_score'] = geo_score
        print(f"    - Scores calculated: Segmentation={seg_score:.2f}, Geometric={geo_score:.2f}")

        # Visualization for this patch
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Visual Analysis for Cell ID: {cell_id}', fontsize=16)
        axes[0].imshow(patch_array_uint8)
        axes[0].set_title('Original Patch')
        axes[0].axis('off')
        axes[1].imshow(segmented_patch, cmap='viridis')
        axes[1].set_title(f'KMeans Segmentation\nScore: {seg_score:.2f}')
        axes[1].axis('off')
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title(f'Canny Edges & Hough Lines\nScore: {geo_score:.2f}')
        axes[2].axis('off')
        plt.show()

    except Exception as e:
        print(f"    - FAILED to process cell {cell_id}. Error: {e}")
        continue

feature_gdf['segmentation_score_norm'] = minmax_scaler.fit_transform(feature_gdf[['segmentation_score']])
feature_gdf['geometric_score_norm'] = minmax_scaler.fit_transform(feature_gdf[['geometric_score']])
feature_gdf['structural_anomaly_score'] = (
    feature_gdf['anomaly_score'] + feature_gdf['segmentation_score_norm'] + feature_gdf['geometric_score_norm']
) / 3
print("âœ… Step 2.2 Complete.")


print("\n--> Step 2.3 (Path A): CNN Visual Analysis (Optimized Batch Version)...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
cnn_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

print(f"    - Step 1: Fetching all {len(candidate_ids_to_process)} image patches for CNN...")
patch_list = []
valid_cell_ids = []
for cell_id in candidate_ids_to_process:
    try:
        row = candidate_gdf[candidate_gdf['grid_cell_id'] == cell_id].iloc[0]
        centroid = row.geometry.centroid
        ee_centroid = ee.Geometry.Point([centroid.x, centroid.y], proj=gdf_crs)
        patch_region = ee_centroid.buffer(30).bounds()
        image_for_patch = sentinel2_unmasked.select(['B4', 'B3', 'B2']).unmask(0)
        patch_array = geemap.ee_to_numpy(image_for_patch, region=patch_region)
        if patch_array is None or np.all(patch_array == 0): continue
        patch_array_scaled = patch_array.astype(np.float32)
        max_val = patch_array_scaled.max()
        if max_val == 0: continue
        patch_array_scaled /= max_val
        patch_array_scaled *= 255.0
        patch_array_uint8 = patch_array_scaled.astype(np.uint8)
        patch_resized = cv2.resize(patch_array_uint8, (32, 32))
        patch_list.append(patch_resized)
        valid_cell_ids.append(cell_id)
    except Exception as e:
        print(f"    - Could not process cell {cell_id} for CNN. Error: {e}")
        continue

if patch_list:
    print(f"    - Step 2: Running batch prediction on {len(patch_list)} valid patches...")
    batch_array = np.array(patch_list)
    batch_for_model = preprocess_input(batch_array)
    batch_feature_vectors = cnn_model.predict(batch_for_model, batch_size=len(patch_list))
    print("    - Step 3: Calculating CNN anomaly scores...")
    iforest_cnn = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    scores_inverted = -1 * iforest_cnn.fit(batch_feature_vectors).score_samples(batch_feature_vectors)
    cnn_anomaly_scores = minmax_scaler.fit_transform(scores_inverted.reshape(-1, 1))
    cnn_scores_series = pd.Series(cnn_anomaly_scores.flatten(), index=valid_cell_ids)
    feature_gdf['cnn_anomaly_score'] = feature_gdf['grid_cell_id'].map(cnn_scores_series).fillna(0)
    print("    - CNN scores calculated and saved.")
else:
    print("    - No valid patches found to process for CNN analysis.")
print("âœ… Step 2.3 Complete.")


print("\n" + "="*80)
print("PHASE 3: SYNTHESIS & OUTPUT GENERATION")
print("="*80)

print("\n--> Step 3.1: Final Scoring...")
weights = {'structural': 0.5, 'cnn': 0.3, 'context': 0.2}
feature_gdf['cnn_anomaly_score'] = feature_gdf['cnn_anomaly_score'].fillna(0.0)
feature_gdf['final_confidence_score'] = (
    feature_gdf['structural_anomaly_score'] * weights['structural'] +
    feature_gdf['cnn_anomaly_score'] * weights['cnn']
) * 100
print("    - Final confidence scores calculated using weighted average.")

print("\n--- VISUAL 3.1: Final Score Analysis ---")
plt.figure(figsize=(10, 4))
sns.histplot(feature_gdf['final_confidence_score'], bins=50, color='gold', kde=True)
plt.title('Final Confidence Score Distribution')
plt.show()

print("\n--- Top 10 Highest Confidence Grid Cells ---")
top_10_anomalies = feature_gdf.sort_values(by='final_confidence_score', ascending=False).head(10)
display(top_10_anomalies[['grid_cell_id', 'anomaly_score', 'structural_anomaly_score', 'cnn_anomaly_score', 'final_confidence_score']].round(2))

print("\n--> Step 3.2: Generating Final Heatmap...")
heatmap_gdf = feature_gdf[['geometry', 'final_confidence_score']].copy()
final_map = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB positron")
folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite', overlay=True, control=True).add_to(final_map)

colormap = linear.YlOrRd_09.scale(0, 100)
colormap.caption = "Final Confidence Score (%)"
final_map.add_child(colormap)

folium.GeoJson(
    heatmap_gdf.to_crs("EPSG:4326"),
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['final_confidence_score']) if feature['properties']['final_confidence_score'] > 0 else 'transparent',
        'color': 'transparent',
        'weight': 0,
        'fillOpacity': 0.6
    },
    name="Anomaly Heatmap"
).add_to(final_map)
print("    - Heatmap layer created.")

# Add markers for top 5 anomalies
print("    - Adding markers for top 5 anomalies.")
for idx, row in top_10_anomalies.head(5).iterrows():
    geom_series = gpd.GeoSeries([row.geometry], crs=gdf_crs)
    centroid = geom_series.to_crs("EPSG:4326").iloc[0].centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        popup=f"Rank #{top_10_anomalies.index.get_loc(idx)+1}<br>Score: {row.final_confidence_score:.1f}%",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(final_map)


# --- VISUAL 3.2: Final Anomaly Heatmap --- 319,1989
print("\n--- VISUAL 3.2: Final Anomaly Heatmap ---")
folium.LayerControl().add_to(final_map)
display(final_map)
print("âœ… Phase 3 Complete.")


print("\n" + "="*80)
print("PHASE 4: GEOSPATIAL ANOMALY ANALYSIS")
print("="*80)

print("\n--> Step 4.0: Preparing Data for Advanced Analysis...")
# Filter for high-confidence anomalies to analyze
confidence_threshold = 50
high_confidence_gdf = feature_gdf[feature_gdf['final_confidence_score'] >= confidence_threshold].copy()
if high_confidence_gdf.empty:
    print(f"    - No anomalies found above the confidence threshold of {confidence_threshold}%. Skipping advanced analysis.")
else:
    print(f"    - Found {len(high_confidence_gdf)} high-confidence cells (score >= {confidence_threshold}) for further analysis.")
    # Calculate centroids for point-based analysis
    high_confidence_gdf['centroid'] = high_confidence_gdf.geometry.centroid

    print("\n--> Step 4.1: Hotspot Analysis (Kernel Density Estimation)...")
    # Get centroid coordinates in a simple array
    coords = np.array([point.coords[0] for point in high_confidence_gdf['centroid']])
    # Create a 2D density plot
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    kde_plot = sns.kdeplot(x=coords[:, 0], y=coords[:, 1], cmap="Reds", fill=True, thresh=0.05, bw_adjust=0.5)
    kde_plot.set_title("Anomaly Hotspot Density (KDE)")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    print("    - KDE plot shows where the density of high-confidence anomalies is greatest.")

    print("\n--> Step 4.2: Spatial Clustering with DBSCAN...")
    # DBSCAN parameters: eps is the search radius, min_samples is the minimum number of points to form a cluster
    # We set eps to 75m to connect adjacent or diagonally-adjacent 50m cells
    clustering = DBSCAN(eps=75, min_samples=3).fit(coords)
    high_confidence_gdf['cluster_id'] = clustering.labels_
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise = list(clustering.labels_).count(-1)
    print(f"    - DBSCAN found {n_clusters} distinct clusters and {n_noise} noise points (isolated anomalies).")

    print("\n--- VISUAL 4.2: Map of Clustered Anomaly Sites ---")
    top_anomaly_idx = high_confidence_gdf['final_confidence_score'].idxmax()
    # Get the geometry object for that specific anomaly
    top_anomaly_geom = high_confidence_gdf.loc[top_anomaly_idx, 'geometry']
    
    # Convert its centroid to Latitude/Longitude for Folium
    center_point_series = gpd.GeoSeries([top_anomaly_geom.centroid], crs=gdf_crs)
    center_wgs84 = center_point_series.to_crs("EPSG:4326").iloc[0]
    new_map_center = [center_wgs84.y, center_wgs84.x]
    cluster_map = folium.Map(location=new_map_center, zoom_start=17, tiles="CartoDB positron")
    folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite', overlay=True, control=True).add_to(cluster_map)
    # Define a color palette for clusters
    if n_clusters > 0:
        cluster_colors = sns.color_palette('Paired', n_clusters).as_hex()
    def get_style(feature):
        cluster_id = feature['properties']['cluster_id']
        if cluster_id == -1:
            return {'fillColor': '#808080', 'color': 'red', 'weight': 0.5, 'fillOpacity': 0.5} # Noise in grey
        else:
            return {'fillColor': cluster_colors[cluster_id], 'color': 'black', 'weight': 1, 'fillOpacity': 0.5}
    viz_gdf = high_confidence_gdf[['geometry', 'cluster_id', 'final_confidence_score']]
    folium.GeoJson(
        viz_gdf.to_crs("EPSG:4326"),
        style_function=get_style,
        tooltip=folium.GeoJsonTooltip(fields=['final_confidence_score', 'cluster_id'], aliases=['Score:', 'Cluster ID:'])
    ).add_to(cluster_map)
    cluster_map.add_child(folium.LayerControl())
    display(cluster_map)
    print("    - Map showing high-confidence cells colored by their spatial cluster ID.")

    print("\n--> Step 4.3: Zonal Statistics for Identified Clusters...")
    if n_clusters > 0:
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_gdf = high_confidence_gdf[high_confidence_gdf['cluster_id'] == cluster_id]
            # Create a single unified polygon for the whole cluster
            cluster_polygon = cluster_gdf.geometry.unary_union
            # Convert to an Earth Engine geometry
            ee_cluster_geom = geemap.geopandas_to_ee(gpd.GeoDataFrame([{'geometry': cluster_polygon}], crs=gdf_crs))
            # Define reducers for elevation and slope
            reducers_zonal = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
            # Extract stats from the combined image
            stats = combined_image.select(['elevation', 'slope']).reduceRegion(
                reducer=reducers_zonal,
                geometry=ee_cluster_geom.geometry(),
                scale=10,
                maxPixels=1e9
            ).getInfo()
            cluster_stats.append({
                'cluster_id': cluster_id,
                'num_cells': len(cluster_gdf),
                'avg_confidence': cluster_gdf['final_confidence_score'].mean(),
                'avg_elevation': stats.get('elevation_mean'),
                'std_elevation': stats.get('elevation_stdDev'),
                'avg_slope': stats.get('slope_mean'),
                'std_slope': stats.get('slope_stdDev')
            })
        stats_df = pd.DataFrame(cluster_stats)
        print("    - Summary of landscape characteristics for each cluster:")
        display(stats_df.round(2))
    else:
        print("    - No clusters found to analyze.")

    print("âœ… Phase 4 Complete.")


print("\n" + "="*80)
print("PHASE 5: AI EXPERT ANALYSIS & REPORTING")
print("="*80)

# Ensure OpenAI client is available
if 'client' not in locals() or client is None:
    print("â�Œ OpenAI client not initialized. Skipping Phase 5.")
    llm_analysis_results = []
else:
    llm_analysis_results = []

# Check if prerequisites from previous phases are met
if client and 'high_confidence_gdf' in locals() and not high_confidence_gdf.empty and 'candidate_ids_to_process' in locals():
    # Use the top candidates identified in Phase 2 for consistency
    top_anomalies_for_llm = feature_gdf[feature_gdf['grid_cell_id'].isin(candidate_ids_to_process)]
    print(f"--> Preparing to send {len(top_anomalies_for_llm)} top anomalies for expert LLM analysis...")

    for idx, row in top_anomalies_for_llm.iterrows():
        cell_id = row['grid_cell_id']
        print(f"\n{'='*25} Analyzing Anomaly ID: {cell_id} {'='*25}")

        # 1. Gather data and image patch (code remains the same)
        print("    - Gathering numerical and contextual data...")
        cluster_info = "Isolated Anomaly"
        if cell_id in high_confidence_gdf['grid_cell_id'].values:
            cell_cluster_id = high_confidence_gdf.loc[high_confidence_gdf['grid_cell_id'] == cell_id, 'cluster_id']
            if not cell_cluster_id.empty:
                cid = cell_cluster_id.iloc[0]
                if cid != -1:
                    num_in_cluster = (high_confidence_gdf['cluster_id'] == cid).sum()
                    cluster_info = f"Part of Cluster #{cid} (contains {num_in_cluster} cells)"
        print("    - Preparing visual evidence (image patch)...")
        centroid = row.geometry.centroid
        ee_centroid = ee.Geometry.Point([centroid.x, centroid.y], proj=gdf_crs)
        patch_region = ee_centroid.buffer(60).bounds()
        patch_array = geemap.ee_to_numpy(sentinel2_unmasked.select(['B4', 'B3', 'B2']), region=patch_region)
        patch_scaled = (np.clip(patch_array, 0, 3000) / 3000 * 255).astype(np.uint8)
        patch_img = Image.fromarray(patch_scaled)
        buffered = io.BytesIO()
        patch_img.save(buffered, format="JPEG")
        patch_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. Construct the prompt (code remains the same)
        prompt_text = f"""
        **Expert Geoarchaeological Analysis Request**
        **1. CASE SUMMARY**
        - **Case ID:** {cell_id}
        - **Location:** {row.geometry.centroid.y:.5f}, {row.geometry.centroid.x:.5f} (UTM Zone 19S)
        - **Our Model's Final Confidence Score:** {row.final_confidence_score:.1f}/100
        **2. COMPREHENSIVE DATA SHEET**
        **Analysis Scores (0-1 Scale):**
        - **Statistical Anomaly (Combined):** {row.anomaly_score:.3f}
        - **Structural & Geometric Score:** {row.structural_anomaly_score:.3f}
        - **CNN Visual Anomaly Score:** {row.cnn_anomaly_score:.3f}
        **Extracted Feature Statistics (from Satellite & DEM):**
        - **Optical (Sentinel-2):**
          - Red Band (Mean, StdDev):   ({row.red_mean:.1f}, {row.red_stdDev:.1f})
          - Green Band (Mean, StdDev): ({row.green_mean:.1f}, {row.green_stdDev:.1f})
          - Blue Band (Mean, StdDev):  ({row.blue_mean:.1f}, {row.blue_stdDev:.1f})
          - NIR Band (Mean, StdDev):   ({row.nir_mean:.1f}, {row.nir_stdDev:.1f})
        - **Vegetation Index:**
          - NDVI (Mean, StdDev):       ({row.ndvi_mean:.3f}, {row.ndvi_stdDev:.3f})
        - **Terrain (SRTM):**
          - Elevation (Mean, StdDev):  ({row.elevation_mean:.1f}m, {row.elevation_stdDev:.1f}m)
          - Slope (Mean, StdDev):      ({row.slope_mean:.1f}Â°, {row.slope_stdDev:.1f}Â°)
          - Aspect (Mean, StdDev):     ({row.aspect_mean:.1f}Â°, {row.aspect_stdDev:.1f}Â°)
        **Spatial Context:**
        - **Cluster Information:** {cluster_info}
        **3. INSTRUCTIONS FOR YOUR IN-DEPTH ANALYSIS**
        As a geoarchaeologist, provide a thorough, evidence-based analysis of the provided data sheet and the satellite image below. Structure your response as a single, valid JSON object with the exact keys defined below.
        - **"morphological_description"**: Describe the shape, structure, and texture of the primary feature(s) in the image. Note any geometric regularity, linearity, curvature, and its relationship to the immediate landscape.
        - **"archaeological_assessment"**: Provide your main assessment. Is this likely to be an archaeological feature (e.g., geoglyph, settlement, ancient agriculture)? State your reasoning, explicitly referencing visual evidence from the image and patterns in the numerical data sheet. For example, 'The high geometric score combined with low NDVI variance suggests...'
        - **"alternative_hypotheses"**: Provide a list of dictionaries, where each dictionary represents an alternative explanation and has two keys: 'hypothesis' (e.g., 'Modern Agriculture') and 'evaluation' (your argument for or against it).
        - **"confidence_score_anomaly"**: Your confidence score (integer 0-100) that this **specific anomaly cell** is a true archaeological feature.
        - **"confidence_score_cluster"**: (If applicable) Your confidence score (integer 0-100) for the **entire cluster** being a significant archaeological site. Set to 0 if the anomaly is isolated.
        - **"recommended_next_steps"**: Provide a prioritized list of strings representing concrete next steps for verification.
        """

        print("\n--- Sending Prompt to LLM ---")
        print(prompt_text)
        print("-----------------------------")

        # 3. Call the API and process the response
        try:
            completion = client.chat.completions.create(model="gpt-4.1", response_format={"type": "json_object"}, messages=[{"role": "system", "content": "You are an expert geoarchaeologist specializing in satellite imagery analysis of the Amazon basin. Provide your analysis in the requested JSON format, ensuring all fields are populated."}, { "role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{patch_base64}"}}]}], max_tokens=1500)
            response_json = json.loads(completion.choices[0].message.content)

            print(f"\n--- Received Formatted Response for {cell_id} ---")
            # Print main text fields
            print("\n1. Morphological Description:")
            print(f"   {response_json.get('morphological_description', 'N/A')}")
            print("\n2. Archaeological Assessment:")
            print(f"   {response_json.get('archaeological_assessment', 'N/A')}")
            
            # Print formatted list of dictionaries for hypotheses
            print("\n3. Alternative Hypotheses:")
            alt_hypotheses = response_json.get('alternative_hypotheses', [])
            if isinstance(alt_hypotheses, list) and alt_hypotheses:
                for i, item in enumerate(alt_hypotheses):
                    print(f"  - Hypothesis {i+1}: {item.get('hypothesis', 'N/A')}")
                    print(f"    Evaluation: {item.get('evaluation', 'N/A')}")
            else:
                print("   N/A")

            # Print formatted list for next steps
            print("\n4. Recommended Next Steps:")
            next_steps = response_json.get('recommended_next_steps', [])
            if isinstance(next_steps, list) and next_steps:
                for step in next_steps:
                    print(f"  - {step}")
            else:
                print("   N/A")

            # Print confidence scores
            print("\n5. Confidence Scores:")
            print(f"  - Anomaly Confidence: {response_json.get('confidence_score_anomaly', 'N/A')} / 100")
            if response_json.get('confidence_score_cluster', 0) > 0:
                print(f"  - Cluster Confidence: {response_json.get('confidence_score_cluster')} / 100")

            print("-------------------------------------------------")

            # 4. Store results for the final report
            result = {"cell_id": cell_id, "prompt": prompt_text, "llm_response": response_json, "llm_anomaly_confidence": response_json.get("confidence_score_anomaly", 0), "our_model_confidence": row.final_confidence_score}
            llm_analysis_results.append(result)

        except Exception as e:
            print(f"    - â�Œ Failed to get analysis for {cell_id}. Error: {e}")
else:
    print("    - Skipping LLM expert analysis because prerequisites were not met.")



print("\n" + "="*80)
print("AI EXPERT ANALYSIS FINAL REPORT")
print("=" * 80)
if llm_analysis_results:
    # Find the best result based on the LLM's confidence score
    best_result = max(llm_analysis_results, key=lambda x: x['llm_anomaly_confidence'])
    res = best_result['llm_response']
    
    # Get full data row for the top anomaly
    top_anomaly_row = feature_gdf[feature_gdf['grid_cell_id'] == best_result['cell_id']].iloc[0]
    top_anomaly_geom_wgs84 = gpd.GeoSeries([top_anomaly_row.geometry.centroid], crs=gdf_crs).to_crs("EPSG:4326").iloc[0]

    print(f"\nğŸ�† HIGHEST CONFIDENCE ANOMALY (According to LLM)")
    print("-" * 50)
    print(f"Grid Cell ID:             {best_result['cell_id']}")
    print(f"Coordinates (Lat, Lon):   {top_anomaly_geom_wgs84.y:.5f}, {top_anomaly_geom_wgs84.x:.5f}")
    print(f"Our Model's Score:        {best_result['our_model_confidence']:.1f} / 100")
    print(f"LLM's Confidence Score:     {best_result['llm_anomaly_confidence']} / 100")
    if res.get('confidence_score_cluster', 0) > 0:
        print(f"LLM's Cluster Confidence:   {res.get('confidence_score_cluster')} / 100")
    
    print("\n" + "-"*25 + " DETAILED LLM ANALYSIS " + "-"*25)
        
    print("\n1. Morphological Description:")
    print(f"   {res.get('morphological_description', 'N/A')}")
    
    print("\n2. Archaeological Assessment:")
    print(f"   {res.get('archaeological_assessment', 'N/A')}")

    print("\n3. Alternative Hypotheses:")
    alt_hypotheses = res.get('alternative_hypotheses', [])
    if isinstance(alt_hypotheses, list) and alt_hypotheses:
        for i, item in enumerate(alt_hypotheses):
            hypothesis = item.get('hypothesis', 'N/A')
            evaluation = item.get('evaluation', 'N/A')
            print(f"  - Hypothesis {i+1}: {hypothesis}")
            print(f"    Evaluation: {evaluation}")
    else:
        print("   N/A")
    
    print("\n4. Recommended Next Steps:")
    next_steps = res.get('recommended_next_steps', [])
    if isinstance(next_steps, list) and next_steps:
        for step in next_steps:
            print(f"  - {step}")
    else:
        print("   N/A")
    
    print("\n" + "-"*70)

    print("\n--- VISUAL REPORT: Map of Highest Confidence Anomaly ---")
    top_anomaly_map_center = [top_anomaly_geom_wgs84.y, top_anomaly_geom_wgs84.x]
    top_anomaly_map = folium.Map(location=top_anomaly_map_center, zoom_start=17, tiles="CartoDB positron")
    folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite', overlay=True, control=True, show=True).add_to(top_anomaly_map)
    
    top_anomaly_poly_wgs84 = gpd.GeoSeries([top_anomaly_row.geometry], crs=gdf_crs).to_crs("EPSG:4326")
    folium.GeoJson(top_anomaly_poly_wgs84, style_function=lambda x: {'color': 'cyan', 'weight': 3, 'fillColor': 'cyan', 'fillOpacity': 0.2}, name=f"Top Anomaly: {best_result['cell_id']}").add_to(top_anomaly_map)

    cluster_id_of_top = high_confidence_gdf.loc[high_confidence_gdf['grid_cell_id'] == best_result['cell_id'], 'cluster_id']
    if not cluster_id_of_top.empty and cluster_id_of_top.iloc[0] != -1:
        cluster_id = cluster_id_of_top.iloc[0]
        cluster_gdf = high_confidence_gdf[high_confidence_gdf['cluster_id'] == cluster_id]
        cluster_viz_gdf = cluster_gdf[['geometry']]
        folium.GeoJson(cluster_viz_gdf.to_crs("EPSG:4326"), style_function=lambda x: {'color': 'yellow', 'weight': 1, 'fillOpacity': 0.1}, name=f"Cluster #{cluster_id} Context").add_to(top_anomaly_map)
    
    folium.Marker(location=top_anomaly_map_center, popup=f"LLM Confidence: {best_result['llm_anomaly_confidence']}%", icon=folium.Icon(color='red', icon='star')).add_to(top_anomaly_map)
    
    folium.LayerControl().add_to(top_anomaly_map)
    display(top_anomaly_map)

else:
    print("No results were returned from the LLM analysis.")

