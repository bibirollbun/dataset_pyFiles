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


!pip install --upgrade pip
!pip install rasterio
!pip install openai
!pip install Pillow



import ee
import os
import json
import geemap
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from IPython.display import FileLink, display






SERVICE_ACCOUNT = 'id-id-ee-mayakaripel-iam-gserv@ee-mayakaripel.iam.gserviceaccount.com'
KEY_FILE = '/kaggle/input/earth-engine-auth-key/ee-mayakaripel-0c55759f4697.json'

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials)

print("âœ… Earth Engine initialized successfully on Kaggle!")



roi = ee.Geometry.Rectangle([-65, -5, -64.9, -4.9])

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Current year NDVI
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2022-01-01', '2022-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .map(add_ndvi) \
    .select('NDVI')

median_ndvi = sentinel2.median()

# Baseline NDVI
baseline = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2017-01-01', '2021-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .map(add_ndvi) \
    .select('NDVI') \
    .median()

# Calculate the anomaly
ndvi_anomaly = median_ndvi.subtract(baseline).rename('NDVI_Anomaly')



# Define visualization parameters for the visual export
vis_params = {
    'min': -0.4,
    'max': 0.4,
    'palette': ['red', 'white', 'green']
}

# --- Export 1: The Raw Data (Recommended for Analysis) ---
# This will be a single-band GeoTIFF with the actual floating-point anomaly values.
out_file_data = '/kaggle/working/ndvi_anomaly_DATA.tif'
print("Attempting to export raw anomaly data from Earth Engine...")

try:
    geemap.ee_export_image(
        ndvi_anomaly,
        filename=out_file_data,
        region=roi,
        scale=1000
    )
    print("Export task submitted and download initiated.")
except Exception as e:
    print(f"An error occurred during the GEE export command: {e}")
    # Set the filename to None so the next part of the code knows it failed
    out_file_data = None

# The CRUCIAL Check ---
# This part only runs if the export command itself didn't crash.
# Now we check if the file actually arrived.

if out_file_data and os.path.exists(out_file_data):
    print(f"\nâœ… SUCCESS: File '{out_file_data}' was created successfully.")
    print(f"File size: {os.path.getsize(out_file_data)} bytes.")


# --- Export 2: The Visual Representation (for Reports/Display) ---
# This will be an 8-bit, 3-band (RGB) GeoTIFF. It's just a picture.
out_file_visual = '/kaggle/working/ndvi_anomaly_VISUAL.tif'
print("\nExporting visualized anomaly map...")
geemap.ee_export_image(
    ndvi_anomaly.visualize(**vis_params), # <-- Use .visualize() here
    filename=out_file_visual,
    region=roi,
    scale=1000
)
print(f"âœ… Visual NDVI anomaly map saved: {out_file_visual}")


# Check if the DATA file was created
if os.path.exists(out_file_data):
    print(f"\n--- Analyzing {out_file_data} ---")
    with rasterio.open(out_file_data) as src:
        anomaly_array = src.read(1) # Read the first (and only) band

        # Plotting the raw data
        plt.figure(figsize=(10, 8))
        # We can create our own colormap to match the GEE one
        plt.imshow(anomaly_array, cmap='RdYlGn', vmin=-0.4, vmax=0.4)
        plt.colorbar(label='NDVI Anomaly')
        plt.title('NDVI Anomaly (2022 vs 2017-2021 Baseline)')
        plt.xlabel('Column #')
        plt.ylabel('Row #')
        plt.show()
else:
    print(f"\nData file {out_file_data} not found. Export may have failed.")


# --- DISPLAY THE VISUAL IMAGE ---
with rasterio.open(out_file_visual) as src:
    # Read all 3 bands (R, G, B)
    # This will have a shape of (bands, height, width), e.g., (3, 250, 250)
    rgb_image = src.read()

    # Matplotlib's imshow expects (height, width, bands)
    # We need to move the first axis (bands) to be the last axis
    rgb_image_display = np.moveaxis(rgb_image, 0, -1)

    plt.figure(figsize=(8, 8))
    # NO cmap is needed because the image is already colored
    plt.imshow(rgb_image_display)
    plt.title("Visualized NDVI Anomaly (Exported as RGB)")
    plt.axis('off')
    # A colorbar is not meaningful here as the pixel values are just 0-255 RGB.
    plt.show()


# Threshold: flag NDVI drop more than 0.2  # selfMask() masks non-true values
anomaly_mask = ndvi_anomaly.lt(-0.02).selfMask()
# ...repeat vectorization and counting...
# Convert the mask to vectors (polygons)
anomaly_vectors = anomaly_mask.reduceToVectors(
    geometry=roi,
    scale=1000,
    geometryType='polygon',
    eightConnected=False,
    labelProperty='zone'
)



# Create a map to display your results
Map = geemap.Map()
Map.centerObject(roi, 12) # Zoom into your ROI

# Add your original NDVI anomaly raster for context
vis_params = {'min': -0.4, 'max': 0.4, 'palette': ['red', 'white', 'green']}
Map.addLayer(ndvi_anomaly, vis_params, 'NDVI Anomaly Raster')

# Add the vector polygons you just created on top
# We can style them to be easily visible (e.g., a bright yellow outline)
Map.addLayer(anomaly_vectors, {'color': 'FFFF00'}, 'Anomaly Polygons')

# Display the map
Map


# Create an image where each pixel's value is its area in square meters
pixel_area = ee.Image.pixelArea()

# Multiply your mask by the area image...
# AND EXPLICITLY RENAME the resulting band to 'area'. This is good practice.
anomaly_area_image = anomaly_mask.multiply(pixel_area).rename('area')

# Sum up all the pixel areas within your ROI
total_anomaly_area_stats = anomaly_area_image.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=roi,
    scale=1000,
    maxPixels=1e9
)

# Now you can confidently get the key 'area' because you just named it.
total_area_sq_meters = total_anomaly_area_stats.get('area').getInfo()

# Convert to a more readable unit
total_area_sq_km = total_area_sq_meters / 1_000_000

print(f"ğŸ“Š Total area of negative anomaly: {total_area_sq_km:.2f} sq. km")


# -- DEBUGGING STEP --
# Temporarily add this line before the line that crashes:
print("Inspecting the dictionary from reduceRegion:")
print(total_anomaly_area_stats.getInfo())


# Define an output file for the vectors
out_vector_file = '/kaggle/working/anomaly_polygons.geojson'

# Export the FeatureCollection
geemap.ee_export_vector(
    anomaly_vectors,
    filename=out_vector_file
)

print(f"âœ… Vector polygons saved to: {out_vector_file}")


# You can give the reducer more specific instructions
custom_reducer = ee.Reducer.histogram(
    maxBuckets=100,      # Set the maximum number of bins
    minBucketWidth=0.01  # Set the minimum width of each bin
)

ndvi_hist_custom = ndvi_anomaly.reduceRegion(
    reducer=custom_reducer,
    geometry=roi,
    scale=1000,
    maxPixels=1e13
).getInfo()

# The plotting code remains the same
hist_data = ndvi_hist_custom['NDVI_Anomaly']
plt.bar(hist_data['bucketMeans'], hist_data['histogram'], width=0.01) # Match width for clarity
plt.title("NDVI Anomaly Distribution (Custom Bins)")
plt.xlabel("NDVI Anomaly")
plt.ylabel("Pixel Count")
plt.grid(True)
plt.show()


# Based on your histogram, you decide on a threshold for "severe" loss
severe_loss_threshold = -0.2

# Create a mask for only these pixels
severe_anomaly_mask = ndvi_anomaly.lt(severe_loss_threshold).selfMask()

# Visualize it on a map
Map = geemap.Map()
Map.centerObject(roi, 12)
Map.addLayer(ndvi_anomaly, {'min': -0.4, 'max': 0.4, 'palette': ['red', 'white', 'green']}, 'Full Anomaly')
Map.addLayer(severe_anomaly_mask, {'palette': 'black'}, 'Severe Anomaly Areas')
Map


vectors = anomaly_mask.reduceToVectors(
    geometry=roi,
    scale=1000,
    geometryType='polygon',
    eightConnected=False,
    labelProperty='anomaly',
    maxPixels=1e13
)



# Add the original raster anomaly for context
vis_params = {'min': -0.4, 'max': 0.4, 'palette': ['red', 'white', 'green']}
Map.addLayer(ndvi_anomaly, vis_params, 'NDVI Anomaly Raster')

# Add your new vector polygons on top. Style them to be visible.
Map.addLayer(vectors, {'color': 'yellow', 'fillColor': 'yellow, 0.3'}, 'Anomaly Polygons')

Map


# Define a function to calculate the area of a single feature.
# The .area() function returns area in square meters.
# We add maxError=1 to handle complex geometries.
def add_area(feature):
    return feature.set({'area_sq_meters': feature.area(maxError=1)})

# Use .map() to apply this robust function to every feature in your collection.
vectors_with_area = vectors.map(add_area)

# Now, this .getInfo() call should succeed.
print("Calculating area for each polygon and retrieving the first 5...")
first_five_features = vectors_with_area.limit(5).getInfo()

# Print the properties of the first few features to see the new 'area_sq_meters' property
print("\nResults:")
print(json.dumps(first_five_features, indent=2))


largest_anomaly = vectors_with_area.sort('area_sq_meters', False).first()
ee.FeatureCollection([largest_anomaly])

# Create a transparent, empty image.
empty_image = ee.Image().byte()

# Paint the outline. Wrap the single feature in a FeatureCollection using a list.
largest_anomaly_outline = empty_image.paint(
    featureCollection=ee.FeatureCollection([largest_anomaly]),  # <-- wrap in a list
    color=1,  # Use an integer for color (e.g., 1 for red in visualization)
    width=3
)

# Add the new IMAGE to the map with visualization parameters for color.
Map.addLayer(
    largest_anomaly_outline, 
    {'palette': ['red'], 'min': 0, 'max': 1}, 
    'Largest Anomaly (Painted)'
)

Map




# Create a map centered on ROI
Map = geemap.Map(center=[-4.95, -64.95], zoom=10)
Map.addLayer(vectors, {'color': 'red'}, 'NDVI Anomaly Polygons')
Map.add_basemap('SATELLITE')
Map.save('/kaggle/working/anomaly_map.html')
print("âœ… Interactive map saved to: /kaggle/working/anomaly_map.html")


# Export GeoJSON file
geojson = geemap.ee_to_geojson(vectors)
with open('/kaggle/working/anomalies.geojson', 'w') as f:
    json.dump(geojson, f)
print("âœ… Exported: anomalies.geojson (for QGIS)")


tiles = roi.coveringGrid(ee.Projection('EPSG:4326').atScale(10000))  # ~10km tiles
tiles_list = tiles.toList(tiles.size())

for i in range(tiles.size().getInfo()):
    # ... (loop contents) ...
    tile = ee.Feature(tiles_list.get(i)).geometry()
    
    # Reduce anomaly mask to vectors within this tile
    tile_vectors = anomaly_mask.reduceToVectors(
        geometry=tile,
        scale=1000,
        geometryType='polygon',
        labelProperty='anomaly',
        maxPixels=1e13
    )
    
    # Count features
    count = tile_vectors.size().getInfo()
    
    if count > 0:
        # Convert to GeoJSON
        geojson = geemap.ee_to_geojson(tile_vectors)
        
        # Export to GeoJSON file
        filename = f"/kaggle/working/anomalies_tile_{i}.geojson"
        with open(filename, "w") as f:
            json.dump(geojson, f)
        
        print(f"âœ… Exported tile {i} to GeoJSON with {count} anomaly polygons")
    else:
        print(f"âš ï¸� Skipped tile {i} (no anomalies found)")



# Get centroid of the ROI
centroid = roi.centroid().coordinates().getInfo()
lat, lon = centroid[1], centroid[0]


# Create and display map
Map = geemap.Map(center=[lat, lon], zoom=8)

# Add the mask layer, but give it a solid color (e.g., bright red) to make it stand out.
# Also, double-check your layer name matches your threshold (e.g., lt(-0.02))
visualization_params = {'palette': ['red']}
Map.addLayer(anomaly_mask, visualization_params, 'NDVI Anomaly Areas (lt -0.02)')

Map.addLayerControl()
Map


print("Submitting export task to Google Drive...")
geemap.ee_export_image_to_drive(
    image=ndvi_anomaly,
    description='NDVI_anomaly_export_10m', # Good to include scale in description
    folder='GEE_Exports',                  # Choose a folder name
    region=roi,
    scale=10,                              # <-- CRITICAL: Specify the resolution in meters
    fileFormat='GeoTIFF',
    maxPixels=1e13                         # Good practice to avoid "Too many pixels" errors
)
print("âœ… Task submitted! Check the 'Tasks' tab in the GEE Code Editor to monitor progress.")



# --- INTEGRATED EXPORT AND SUMMARY LOOP ---

# Assume `anomaly_mask` and `roi` are already defined
tiles = roi.coveringGrid(ee.Projection('EPSG:4326').atScale(10000))
num_tiles = tiles.size().getInfo() # <-- Get the number of tiles dynamically
tiles_list = tiles.toList(num_tiles)

# --- Initialize summary and file link lists before the loop ---
summary = {}
exported_files = []

print(f"Starting export process for {num_tiles} tiles...")

for i in range(num_tiles):
    tile = ee.Feature(tiles_list.get(i)).geometry()
    
    # Reduce anomaly mask to vectors within this tile
    tile_vectors = anomaly_mask.reduceToVectors(
        geometry=tile,
        scale=1000,
        geometryType='polygon',
        labelProperty='anomaly',
        maxPixels=1e13
    )
    
    # Get the feature count for this tile
    count = tile_vectors.size().getInfo()
    
    # Immediately add the count to your summary dictionary
    summary[f'tile_{i}'] = count
    
    if count > 0:
        # Convert to GeoJSON
        geojson = geemap.ee_to_geojson(tile_vectors)
        
        # Define filename
        filename = f"/kaggle/working/anomalies_tile_{i}.geojson"
        
        # Export to GeoJSON file
        with open(filename, "w") as f:
            json.dump(geojson, f)
        
        print(f"âœ… Exported tile {i} to GeoJSON with {count} anomaly polygons")
        
        # Add the filename to a list for creating links later
        exported_files.append(filename)
    else:
        print(f"âš ï¸� Skipped tile {i} (no anomalies found)")

# --- FINAL REPORT SECTION (after the loop finishes) ---

print("\n--- Export Complete! ---")
print("ğŸ“Š Anomaly summary per tile:", summary)

# Display download links for all the files that were actually created
if exported_files:
    print("\nâ¬‡ï¸� Downloadable files:")
    for file_path in exported_files:
        display(FileLink(file_path))
else:
    print("\nNo files were exported as no anomalies were found.")


#  Vectorize ALL anomalies first 
print("Step 1: Vectorizing all anomaly polygons...")
all_vectors = anomaly_mask.reduceToVectors(
    geometry=roi,
    scale=1000,
    geometryType='polygon',
    labelProperty='anomaly',
    maxPixels=1e13
)
print(f"Found {all_vectors.size().getInfo()} initial polygons.")


# Calculate Area and Filter by Size 
print("\nStep 2: Filtering polygons by a plausible size range...")

# Define your size thresholds in square meters
# Example: Keep sites larger than a small house (200 sq m) but smaller
# than a very large modern farm field (500,000 sq m = 0.5 sq km).
# *** You should adjust these values based on the type of site you're looking for! ***
MIN_AREA_SQ_METERS = 200
MAX_AREA_SQ_METERS = 6000000 


# Define a function to calculate the area of a single feature.
# We include maxError=1 to handle complex geometries robustly.
def add_area(feature):
    return feature.set({'area': feature.area(maxError=1)})

# Apply the area calculation to all vectors
vectors_with_area = all_vectors.map(add_area)

# DEBUGGING BLOCK: 

# Let's get the information for all 7 features to inspect their properties.
# .getInfo() will pull the data from GEE to your notebook.
all_vectors_info = vectors_with_area.getInfo()['features']

print("\n--- DEBUGGING: Inspecting the 7 polygons BEFORE filtering ---")
if not all_vectors_info:
    print("No vector features were found at all.")
else:
    # Loop through each feature and print its calculated area
    for i, feature in enumerate(all_vectors_info):
        # The area is stored in the 'properties' dictionary of the feature
        area = feature['properties']['area']
        print(f"Polygon {i}: Area = {area:.2f} sq. meters")

# --- End of DEBUGGING BLOCK ---

# Now, apply the size filter
plausibly_sized_vectors = vectors_with_area.filter(
    ee.Filter.And(
        ee.Filter.gte('area', MIN_AREA_SQ_METERS),
        ee.Filter.lte('area', MAX_AREA_SQ_METERS)
    )
)

print(f"Found {plausibly_sized_vectors.size().getInfo()} plausibly-sized polygons.")


# --- Step 3: Calculate Shape Compactness and Filter ---
print("\nStep 3: Filtering polygons by shape (compactness)...")

# Define a function to calculate the Polsby-Popper compactness score
def add_compactness(feature):
    # Formula: (4 * PI * area) / (perimeter^2)
    area = feature.get('area')  # We already calculated this!
    perimeter = feature.geometry().perimeter(maxError=1)
    
    # Calculate compactness. Use ee.Number to ensure server-side math.
    compactness = ee.Number(4).multiply(3.14159).multiply(area).divide(perimeter.pow(2))
    
    return feature.set({'compactness': compactness})

# Apply the compactness calculation to our size-filtered vectors
vectors_with_shape = plausibly_sized_vectors.map(add_compactness)

# Define your compactness threshold.
# Values > 0.5 tend to be blob-like or squarish.
# Values < 0.2 are very irregular.
# Let's keep things that are moderately to very compact.
COMPACTNESS_THRESHOLD = 0.5

# Apply the shape filter
candidate_vectors = vectors_with_shape.filter(
    ee.Filter.gte('compactness', COMPACTNESS_THRESHOLD)
)

# --- Final Result ---
final_count = candidate_vectors.size().getInfo()
print(f"\n--- FINAL RESULT ---")
print(f"Found {final_count} high-quality candidate sites after size and shape filtering.")


# ===================================================================
# THE "BULLETPROOF SAVE" CELL (with simplification)
# ===================================================================
import ee
import geemap
import os
from IPython.display import FileLink

print("--- Running Robust Save Process ---")

# --- Step 1: Define the collections (as before) ---
candidate_vectors = vectors_with_shape.filter(ee.Filter.gte('compactness', 0.5))
irregular_vectors = vectors_with_shape.filter(ee.Filter.lt('compactness', 0.5))
print(f"Found {candidate_vectors.size().getInfo()} candidates and {irregular_vectors.size().getInfo()} irregular shapes.")


# --- Step 2: THE CRITICAL FIX - Simplify the geometries ---
# We reduce the complexity of the polygons before trying to save them.
# A higher number means more simplification. 100 meters is a good starting point.
print("Simplifying polygon geometries to reduce file size...")
simplified_candidates = candidate_vectors.map(lambda f: f.simplify(maxError=100))
simplified_irregulars = irregular_vectors.map(lambda f: f.simplify(maxError=100))


# --- Step 3: Create the map using the SIMPLIFIED data ---
Map = geemap.Map()
Map.centerObject(roi, 10)
Map.add_basemap('SATELLITE')

# Add the simplified layers to the map
Map.addLayer(simplified_irregulars, {'color': 'blue'}, 'Irregular Shapes (Simplified)')
Map.addLayer(simplified_candidates, {'color': 'red'}, 'High-Quality Candidates (Simplified)')
Map.addLayerControl()


# --- Step 4: Save the map to an HTML file ---
output_html_path = '/kaggle/working/final_map_simplified.html'
print(f"Attempting to save simplified map to {output_html_path}...")
Map.save(output_html_path)


# --- Step 5: Verify that the file was actually created ---
if os.path.exists(output_html_path) and os.path.getsize(output_html_path) > 0:
    print(f"\nâœ… SUCCESS! Map saved successfully.")
    print("Please refresh the file browser on the right and download the file.")
    display(FileLink(output_html_path))
else:
    print(f"\nâ�Œ FAILURE. The file could not be created, even after simplification.")
    print("This suggests a more fundamental issue with the environment or data.")


# ===================================================================
# PHASE 3: IMAGE CHIP EXPORT (Corrected - GeoTIFF to Google Drive)
# ===================================================================

import ee
import geemap
import os
import time

print("--- Starting AI Data Preparation: Exporting GeoTIFF Image Chips to Google Drive ---")

# --- Step 1: Create a high-quality, cloud-free image for visualization ---
print("Creating a cloud-free satellite image...")
image_to_export = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi) \
    .filterDate('2022-01-01', '2022-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median()

# This visualization step creates a 3-band RGB image.
# The GeoTIFF will store these 0-255 RGB values.
visual_rgb_image = image_to_export.visualize(
    bands=['B4', 'B3', 'B2'],
    min=0,
    max=3000,
    gamma=1.4
)

# --- Step 2: Get the list of features ---
candidate_list = candidate_vectors.toList(candidate_vectors.size())
num_candidates = candidate_vectors.size().getInfo()
print(f"Found {num_candidates} candidate sites to submit for export as GeoTIFF.")

# --- Step 3: Loop through each candidate and submit an export task to Drive ---
drive_folder_name = 'AI_Challenge_Image_Chips'


for i in range(num_candidates):
    feature = ee.Feature(candidate_list.get(i))
    region = feature.geometry().bounds()
    
    task_description = f'candidate_site_{i}_geotiff_export'
    drive_filename = task_description # .tif will be added by GEE
    
    print(f"-> Submitting GeoTIFF export task {i+1}/{num_candidates} for site {i}: {task_description}")
    
    task = geemap.ee_export_image_to_drive(
        image=visual_rgb_image,
        description=task_description,
        folder=drive_folder_name,
        fileNamePrefix=drive_filename,
        region=region,
        scale=10,
        fileFormat='GeoTIFF', # <--- ENSURED THIS IS GeoTIFF
        maxPixels=1e10
    )
    time.sleep(5)


print("\nâœ… All GeoTIFF export tasks submitted successfully!")
print(f"Please go to the Google Earth Engine Code Editor (https://code.earthengine.google.com/)")
print(f"and check the 'Tasks' tab on the right-hand panel.")
print(f"You will need to manually click 'RUN' on each of the {num_candidates} tasks.")
print(f"The GeoTIFF files will appear in a folder named '{drive_folder_name}' in your Google Drive.")


# ===================================================================
# THE BULLETPROOF VISUALIZATION CELL (with diagnostics)
# ===================================================================


# vectors_with_shape: The collection of 7 polygons with area and compactness calculated.
# roi: Your region of interest.

print("--- Running Final Visualization & Diagnostics ---")

# --- Step 1: Re-calculate the final collections to be 100% sure ---

# Define the candidate vectors based on the compactness score
candidate_vectors = vectors_with_shape.filter(ee.Filter.gte('compactness', 0.5))

# Define the irregular vectors (the ones that were filtered out)
irregular_vectors = vectors_with_shape.filter(ee.Filter.lt('compactness', 0.5))


# --- Step 2: DIAGNOSTICS - Print the size of each collection ---
# We will use .getInfo() to get the numbers from the server before we map.

candidate_count = candidate_vectors.size().getInfo()
irregular_count = irregular_vectors.size().getInfo()

print(f"Number of 'High-Quality Candidates' to draw in RED: {candidate_count}")
print(f"Number of 'Irregular Shapes' to draw in BLUE: {irregular_count}")


# --- Step 3: Create and build the map ---

# Create a fresh map object
Map = geemap.Map()
Map.centerObject(roi, 10)

# Add the satellite basemap FIRST
Map.add_basemap('SATELLITE')

# Add the layers only if they contain features
if irregular_count > 0:
    Map.addLayer(irregular_vectors, {'color': 'blue'}, 'Irregular Shapes (Filtered Out)')
    print("-> Added BLUE layer to map.")
else:
    print("-> SKIPPED blue layer (no features).")

if candidate_count > 0:
    Map.addLayer(candidate_vectors, {'color': 'red'}, 'High-Quality Candidates')
    print("-> Added RED layer to map.")
else:
    print("-> SKIPPED red layer (no features).")

# Add the layer control widget so we can see the layer names
Map.addLayerControl()

# Display the final, complete map
# Display the map live
#print("\nRendering map...")
#Map # Un-comment this line to display the map




import openai
import os
import base64
from PIL import Image # Pillow library for image handling
from io import BytesIO # For in-memory file-like objects
from kaggle_secrets import UserSecretsClient
import json # <-- IMPORT THIS
import time # <-- IMPORT THIS
import pandas as pd # <-- IMPORT THIS (for the DataFrame at the end)

# --- 1. Configure OpenAI API Key ---
try:
    user_secrets = UserSecretsClient()
    openai.api_key = user_secrets.get_secret("OPENAI_API_KEY")
    print("âœ… OpenAI API Key loaded successfully.")
except Exception as e:
    print(f"â�Œ Error loading OpenAI API Key: {e}")
    raise SystemExit("API Key not found.")

# --- 2. Function to Open GeoTIFF, Convert to PNG, and Base64 Encode ---
def encode_geotiff_to_png_base64(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_byte_arr = buffered.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

# --- 3. Define Image Source Folder and the Expert Prompt (ONCE before the loop) ---
# Ensure this path is correct for where your uploaded TIFFs are in Kaggle
image_chips_folder = "/kaggle/input/ai-candidate-chips/" 
# If you uploaded them to /kaggle/working/ instead, change the path:
# image_chips_folder = "/kaggle/working/ai_input_chips/" 

prompt_text = """
You are a world-class expert in satellite remote sensing for archaeology, specializing in identifying subtle traces of ancient human settlements and earthworks in challenging environments like the Amazon rainforest.

Analyze the provided satellite image. Look for the following specific indicators of potential archaeological sites:
- Geometric patterns in vegetation or soil (e.g., circles, squares, rectangles, linear features) that are unlikely to be natural.
- Cropmarks or soil marks that reveal buried structures.
- Faint earthworks, mounds, ditches, or enclosures.
- Anomalous clearings or vegetation patterns that deviate from the surrounding natural landscape and do not appear to be modern agriculture or recent deforestation for other purposes.

Ignore:
- Obvious modern roads, buildings, or large-scale recent agricultural fields unless they intersect or overlay potential ancient features.
- Natural geological formations or riverine features unless they show signs of human modification.

Based on your analysis, provide your response in a JSON format with the following keys:
- "is_archaeology_candidate": A boolean (true or false).
- "confidence_score": A float between 0.0 (no confidence) and 1.0 (very high confidence).
- "observed_features": A brief list of specific visual features in the image that led to your decision (e.g., "faint rectangular outline in vegetation", "circular soil mark").
- "rationale": A concise explanation (1-2 sentences) for your overall assessment.
"""

# --- 4. Loop Through Image Chips and Analyze with OpenAI ---
results_list = [] # To store the analysis for each image

# Check if the image chips folder exists
if not os.path.exists(image_chips_folder) or not os.listdir(image_chips_folder):
    print(f"â�Œ ERROR: Image chips folder '{image_chips_folder}' is empty or does not exist.")
    print("Please ensure you have uploaded your GeoTIFF files to this location and that the path is correct.")
else:
    all_image_files = os.listdir(image_chips_folder)
    print(f"Found {len(all_image_files)} files in '{image_chips_folder}'. Processing .tif files...")

    for image_filename in all_image_files:
        if image_filename.lower().endswith(".tif"): # Process only .tif files (case-insensitive)
            image_path = os.path.join(image_chips_folder, image_filename)
            print(f"\n--- Processing: {image_filename} ---")
            
            print(f"Converting '{image_filename}' to PNG and encoding to base64...")
            base64_png_image = encode_geotiff_to_png_base64(image_path)
            print("Image conversion and encoding complete.")
            
            print(f"Sending image '{image_filename}' (as PNG) to OpenAI GPT-4o for analysis...")
            
            try:
                # This is the API call block, no changes here
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_png_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500 
                )
                
                ai_analysis_raw_str = response.choices[0].message.content
                print("Raw AI Response:", ai_analysis_raw_str) # Good for debugging

                # --- START OF IMPROVED JSON HANDLING ---
                # Attempt to find and extract JSON block if markdown is present
                json_string_to_parse = ai_analysis_raw_str
                if ai_analysis_raw_str.strip().startswith("```json"):
                    # Extract content between ```json and ```
                    try:
                        json_string_to_parse = ai_analysis_raw_str.split("```json")[1].split("```")[0].strip()
                    except IndexError:
                        # Fallback if splitting fails unexpectedly
                        print("Warning: Markdown JSON indicators found, but extraction failed. Trying to parse raw string.")
                        pass # Try parsing raw string anyway
                elif ai_analysis_raw_str.strip().startswith("```"): # Simpler ``` ``` case
                     try:
                        json_string_to_parse = ai_analysis_raw_str.split("```")[1].strip()
                     except IndexError:
                        pass


                try:
                    analysis_data = json.loads(json_string_to_parse)
                    analysis_data['filename'] = image_filename
                    # Add a flag to indicate if the response was as expected
                    analysis_data['response_type'] = 'structured_json' 
                    results_list.append(analysis_data)
                    print("âœ… Successfully parsed JSON response.")

                except json.JSONDecodeError:
                    print(f"âš ï¸� AI did not return valid JSON. Response was: '{ai_analysis_raw_str}'")
                    results_list.append({
                        'filename': image_filename,
                        'is_archaeology_candidate': None,
                        'confidence_score': None,
                        'observed_features': ['Non-JSON AI Response'],
                        'rationale': ai_analysis_raw_str, # Store the raw text response
                        'response_type': 'plain_text_fallback'
                    })
                # --- END OF IMPROVED JSON HANDLING ---
                
            except Exception as e:
                # This catches errors from the API call itself (network, auth, etc.)
                print(f"â�Œ API Call Error for {image_filename}: {e}")
                results_list.append({
                    'filename': image_filename,
                    'is_archaeology_candidate': None,
                    'confidence_score': None,
                    'observed_features': ['API Call Error'],
                    'rationale': str(e),
                    'response_type': 'api_call_error'
                })
            
            time.sleep(3) # Increased sleep slightly, 3-5 seconds is safer for loops
            
            # --- create Summary Report ---
if results_list: # Only proceed if some results were gathered
    results_df = pd.DataFrame(results_list)
    results_df_sorted = results_df.sort_values(by='confidence_score', ascending=False, na_position='last')

    print("\n--- AI Analysis Summary ---")
    print(results_df_sorted)

    output_csv_path = "/kaggle/working/ai_archaeology_analysis_results.csv"
    results_df_sorted.to_csv(output_csv_path, index=False)
    print(f"\nâœ… Results saved to {output_csv_path}")
else:
    print("\nNo image files were processed or no results were obtained.")

