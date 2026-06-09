# ğŸ”§ Install missing dependencies manually (for Kaggle environment)
!pip install rasterio openai geopandas shapely folium skyfield contextily --quiet

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
from matplotlib.colors import LightSource

# âœ… Load OpenAI API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# âœ… OpenTopography LiDAR Tile â€” TapajÃ³s region (TAP_A04_15_DTM)
tif_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"
dataset_id = "OpenTopography Tile: TAP_A04_15_DTM (TapajÃ³s, ORNL DAAC)"

# âœ… Load the DTM and clean no-data values
with rasterio.open(tif_path) as dataset:
    elevation = dataset.read(1)
    bounds = dataset.bounds

elevation = np.where(elevation == -999.0, np.nan, elevation)

# âœ… Display elevation stats
print("\n--- Elevation Statistics (Cleaned) ---")
print("Elevation shape:", elevation.shape)
print("Elevation min:", np.nanmin(elevation))
print("Elevation max:", np.nanmax(elevation))
print("Number of NaNs:", np.isnan(elevation).sum())
print("Unique Values (sample):", np.unique(elevation[~np.isnan(elevation)])[:10])
print("\n")

# âœ… Normalize for visualization
np.random.seed(42)
elev_norm = 255 * (elevation - np.nanmin(elevation)) / (np.nanmax(elevation) - np.nanmin(elevation))
elev_norm = np.nan_to_num(elev_norm, nan=0).astype(np.uint8)

# âœ… Hillshade rendering
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elevation, vert_exag=10, dx=1, dy=1)

# Clip hillshade to [0, 255] range and handle NaNs
hillshade_clean = np.nan_to_num(hillshade, nan=0)
hillshade_clean = np.clip(hillshade_clean, 0, 255)

plt.figure(figsize=(8, 6))
plt.imshow(hillshade_clean, cmap='gray')
plt.title("TapajÃ³s LiDAR Hillshade")
plt.axis("off")
plt.savefig("hillshade_tapajos.png", dpi=300, bbox_inches="tight")
plt.show()

# âœ… Construct GPT prompt with grounded observations
prompt = f"""
You are analyzing a high-resolution LiDAR-derived Digital Terrain Model (DTM) from the TapajÃ³s region in Brazil.
The dataset is {dataset_id} and covers an area of ~10 kmÂ².

Here are the details:
- Elevation ranges: {np.nanmin(elevation):.2f} to {np.nanmax(elevation):.2f} meters
- Terrain shows >30 circular depressions in upper right quadrant
- Some features are aligned and clustered
- Possible erosional valleys run diagonally in lower areas

1. Describe what surface features may be visible.
2. Evaluate if this could include anthropogenic features (e.g. mounds, fields, ring villages).
3. Suggest how to follow up with additional tools (e.g. LIDAR segmentation or fieldwork).
"""

# âœ… Call GPT-4.1
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

# âœ… Print Logs
print("\U0001f9e0 Model Used:", response.model)
print("ğŸ“‚ Dataset ID:", dataset_id)
print("\nğŸ“œ GPT Response:\n", response.choices[0].message.content)

# âœ… Save GPT response for reproducibility
with open("gpt_response.txt", "w") as f:
    f.write(response.choices[0].message.content)


# ğŸ“¦ Dependencies
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from shapely.geometry import Point
import geopandas as gpd
from rasterio.plot import show
import rasterio.warp

# === Step 1: Load LiDAR DTM ===
dtm_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"
with rasterio.open(dtm_path) as src:
    dtm_crs = src.crs
    transform = src.transform
    dtm_raw = src.read(1)
    bounds = src.bounds
    width = src.width
    height = src.height
    print("ğŸ“Œ CRS of DTM:", dtm_crs)

# Mask no-data
dtm = np.where(dtm_raw == src.nodata, np.nan, dtm_raw)

# === Step 2: Hillshade for Visualization ===
def compute_hillshade(elevation, azimuth=315, angle_altitude=45):
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=azimuth, altdeg=angle_altitude)
    return ls.hillshade(elevation, vert_exag=1, dx=1, dy=1)

# Reproject DTM to EPSG:4326
reprojected, dst_transform = rasterio.warp.reproject(
    source=dtm,
    src_crs=dtm_crs,
    src_transform=transform,
    dst_crs="EPSG:4326",
    resampling=rasterio.warp.Resampling.bilinear
)

# === ğŸ›°ï¸�  Step 3: Automated Detection of Terrain Anomalies and Detect Depressions ===

# Generate visual hillshade from reprojected DTM
hillshade = compute_hillshade(reprojected[0])  

# Normalize elevation and invert to highlight depressions
norm = (dtm - np.nanmin(dtm)) / (np.nanmax(dtm) - np.nanmin(dtm))
inverted = 1 - norm

# Apply Otsu's method to find threshold separating depressions
thresh = filters.threshold_otsu(inverted[np.isfinite(inverted)])
binary = inverted > thresh

# Remove small artifacts and label connected regions
binary = morphology.remove_small_objects(binary, min_size=100)
label_image = measure.label(binary)

# Extract region properties for anomaly analysis
regions = measure.regionprops(label_image)

# === Step 5: Extract Features and Save ===
anomaly_data = []
for region in regions:
    y, x = region.centroid
    lon, lat = rasterio.transform.xy(transform, y, x)
    anomaly_data.append({
        "geometry": Point(lon, lat),
        "area": region.area,
        "eccentricity": region.eccentricity,
        "solidity": region.solidity,
        "bbox_area": region.bbox_area
    })

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(anomaly_data, crs=dtm_crs).to_crs("EPSG:4326")
gdf.to_file("tapajos_anomalies.geojson", driver="GeoJSON")

# === Plot with Hillshade and Anomalies ===
left = dst_transform[2]
top = dst_transform[5]
right = left + dst_transform[0] * width
bottom = top + dst_transform[4] * height
extent = [left, right, bottom, top]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Detected Anomalies over TapajÃ³s DTM Hillshade")
ax.imshow(hillshade, cmap='gray', extent=extent, origin='upper')
gdf.plot(ax=ax, color='red', markersize=5)
plt.show()

# === Optional: Print top anomalies by area ===
print(gdf.sort_values("area", ascending=False).head())



import folium
import math
from geopy.distance import geodesic
from folium.plugins import MousePosition, Fullscreen, MiniMap, MeasureControl
from kaggle_secrets import UserSecretsClient

# Mapbox token
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("map_box_api_key")

# Anomaly coordinates with labels and descriptions
anomalies = [
    {"label": "A", "lat": -3.02287, "lon": -54.95535, "desc": "Large, slightly elongated, possible platform"},
    {"label": "B", "lat": -3.01634, "lon": -54.97305, "desc": "Rounder and more solid â€” promising signature"},
    {"label": "C", "lat": -3.02819, "lon": -54.96735, "desc": "Smaller but highly solid â€” possible mound"}
]

# Extract coordinates
A = (anomalies[0]["lat"], anomalies[0]["lon"])
B = (anomalies[1]["lat"], anomalies[1]["lon"])
C = (anomalies[2]["lat"], anomalies[2]["lon"])

# Helper function to compute angle at point a given triangle sides
def angle_from_sides(a, b, c):
    return math.degrees(math.acos((b**2 + c**2 - a**2) / (2*b*c)))

# Compute side lengths in meters
a = geodesic(B, C).meters
b = geodesic(A, C).meters
c = geodesic(A, B).meters

# Compute angles
angle_A = angle_from_sides(a, b, c)
angle_B = angle_from_sides(b, a, c)
angle_C = 180 - angle_A - angle_B

# Print angles
print("\U0001F4D0 Triangle Internal Angles (degrees):")
print(f"âˆ A (Anomaly #1) â‰ˆ {angle_A:.2f}Â°")
print(f"âˆ B (Anomaly #2) â‰ˆ {angle_B:.2f}Â°")
print(f"âˆ C (Anomaly #3) â‰ˆ {angle_C:.2f}Â°")

# Compute azimuths (forward bearings)
def compute_azimuth(p1, p2):
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.atan2(x, y)
    return (math.degrees(angle) + 360) % 360

azimuth_AB = compute_azimuth(A, B)
azimuth_BC = compute_azimuth(B, C)
azimuth_CA = compute_azimuth(C, A)

# Print azimuths
print("\U0001F9ED Cardinal Azimuths of Triangle Sides:")
print(f"AB: {azimuth_AB:.2f}Â°")
print(f"BC: {azimuth_BC:.2f}Â°")
print(f"CA: {azimuth_CA:.2f}Â°")

# Compute side length ratios
ratio_ab_ac = round(a / c, 3)
ratio_bc_ab = round(a / b, 3)
ratio_ca_bc = round(b / a, 3)

print("\nâ†” Side Length Ratios:")
print(f"AB / AC = {ratio_ab_ac}")
print(f"BC / AB = {ratio_bc_ab}")
print(f"CA / BC = {ratio_ca_bc}")

# Map centered at triangle centroid
centroid_lat = sum(anom["lat"] for anom in anomalies) / 3
centroid_lon = sum(anom["lon"] for anom in anomalies) / 3
m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=16, tiles=None)

# Add satellite layer
folium.TileLayer(
    tiles=f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={secret_value_0}',
    attr='Mapbox Satellite',
    name='Mapbox Satellite',
    overlay=False,
    control=True,
    max_zoom=22,
    tile_size=512,
    zoom_offset=-1
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Imagery Â© Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
    name='Esri Satellite'
).add_to(m)

# Draw triangle edges
triangle_points = [(anom["lat"], anom["lon"]) for anom in anomalies]
folium.Polygon(locations=triangle_points, color="yellow", weight=2, fill=False).add_to(m)

# Add lines and directional arrows at midpoints
edges = [(A, B, azimuth_AB), (B, C, azimuth_BC), (C, A, azimuth_CA)]
colors = ["red", "green", "blue"]
labels = ["AB", "BC", "CA"]
for i, (start, end, az) in enumerate(edges):
    folium.PolyLine(locations=[start, end], color=colors[i], weight=3).add_to(m)
    mid_lat = (start[0] + end[0]) / 2
    mid_lon = (start[1] + end[1]) / 2
    folium.RegularPolygonMarker(
        location=(mid_lat, mid_lon),
        number_of_sides=3,
        radius=8,
        rotation=az,
        color=colors[i],
        fill_color=colors[i],
        fill_opacity=1
    ).add_to(m)

# Markers with angle labels as text annotations
for i, anom in enumerate(anomalies):
    angle = [angle_A, angle_B, angle_C][i]
    folium.map.Marker(
        [anom["lat"], anom["lon"]],
        icon=folium.DivIcon(
            html=f'<div style="font-size: 14pt; font-weight: bold; color: white;">{anom["label"]}</div>'
        )
    ).add_to(m)

# Controls
folium.LayerControl().add_to(m)
Fullscreen().add_to(m)
MiniMap().add_to(m)
MeasureControl().add_to(m)
MousePosition(position='bottomright', separator=' | ', prefix='Coordinates:',
              lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
              lng_formatter="function(num) {return L.Util.formatNum(num, 6);}").add_to(m)

# Save and show
m.save("anomaly_triangle_with_azimuths.html")
m




import h5py

file_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"  
output_file = "gedi_beam0000_structure.txt"
beam = "BEAM0000"

def save_dataset_structure(file_path, beam, output_file):
    dataset_paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_paths.append(name)

    with h5py.File(file_path, 'r') as f:
        f[beam].visititems(visitor)

    with open(output_file, 'w') as out:
        out.write(f"Datasets under {beam}:\n")
        for path in dataset_paths:
            out.write(f"{path}\n")

    print(f"âœ… Saved {len(dataset_paths)} dataset paths to '{output_file}'")

save_dataset_structure(file_path, beam, output_file)


import h5py
import numpy as np
import rasterio
from pyproj import Transformer

# === GEDI bounds extraction ===
gedi_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"

with h5py.File(gedi_path, "r") as f:
    beam = f["BEAM0000"]
    lat = beam["lat_lowestmode"]
    lon = beam["lon_lowestmode"]

    chunk_size = 1_000_000
    lat_min, lat_max = 90, -90
    lon_min, lon_max = 180, -180

    for i in range(0, len(lat), chunk_size):
        lat_chunk = lat[i:i+chunk_size]
        lon_chunk = lon[i:i+chunk_size]

        valid = (lat_chunk > -90) & (lat_chunk < 90) & (lon_chunk > -180) & (lon_chunk < 180)
        if np.any(valid):
            lat_min = min(lat_min, np.min(lat_chunk[valid]))
            lat_max = max(lat_max, np.max(lat_chunk[valid]))
            lon_min = min(lon_min, np.min(lon_chunk[valid]))
            lon_max = max(lon_max, np.max(lon_chunk[valid]))

print(f"ğŸ“¡ GEDI Spatial Bounds:")
print(f"Latitude:  {lat_min:.5f} to {lat_max:.5f}")
print(f"Longitude: {lon_min:.5f} to {lon_max:.5f}")

# === Load TapajÃ³s DTM and reproject bounds ===
tapajos_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"
with rasterio.open(tapajos_path) as src:
    bounds = src.bounds
    crs = src.crs

transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

print("\nğŸŒ� TapajÃ³s DTM Bounds (Reprojected):")
print(f"Latitude:  {min_lat:.5f} to {max_lat:.5f}")
print(f"Longitude: {min_lon:.5f} to {max_lon:.5f}")

# === Check coverage ===
within_lat = (lat_min <= min_lat) and (lat_max >= max_lat)
within_lon = (lon_min <= min_lon) and (lon_max >= max_lon)

if within_lat and within_lon:
    print("\nâœ… TapajÃ³s DTM lies fully within GEDI spatial coverage.")
else:
    print("\nâš ï¸� Warning: TapajÃ³s DTM is not fully within GEDI coverage.")




import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point
import matplotlib.pyplot as plt

# === Paths ===
gedi_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"
dtm_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"

# === Get TapajÃ³s DTM bounds in EPSG:4326 ===
with rasterio.open(dtm_path) as src:
    dtm_bounds = src.bounds
    dtm_crs = src.crs
    left, bottom = rasterio.warp.transform(dtm_crs, "EPSG:4326", [dtm_bounds.left], [dtm_bounds.bottom])
    right, top = rasterio.warp.transform(dtm_crs, "EPSG:4326", [dtm_bounds.right], [dtm_bounds.top])
    LAT_MIN, LAT_MAX = min(bottom[0], top[0]), max(bottom[0], top[0])
    LON_MIN, LON_MAX = min(left[0], right[0]), max(left[0], right[0])

print("ğŸ“� Using TapajÃ³s bounding box:")
print(f"Latitude:  {LAT_MIN:.5f} to {LAT_MAX:.5f}")
print(f"Longitude: {LON_MIN:.5f} to {LON_MAX:.5f}")

# === Extract subset from GEDI ===
def extract_gedi_subset(path):
    features = []
    with h5py.File(path, 'r') as f:
        for beam in [b for b in f.keys() if b.startswith("BEAM")]:
            try:
                lat = f[f"{beam}/geolocation/lat_lowestmode"][:]
                lon = f[f"{beam}/geolocation/lon_lowestmode"][:]
                rh = f[f"{beam}/rh"][:, -1]  # RH100

                mask = (
                    (lat >= LAT_MIN) & (lat <= LAT_MAX) &
                    (lon >= LON_MIN) & (lon <= LON_MAX)
                )
                count = np.sum(mask)
                print(f"{beam}: {count} points within TapajÃ³s bounds")

                for la, lo, rh_val in zip(lat[mask], lon[mask], rh[mask]):
                    features.append({
                        "geometry": Point(lo, la),
                        "rh100": rh_val,
                        "beam": beam
                    })

            except Exception as e:
                print(f"Skipped {beam} due to error: {e}")
                continue

    if features:
        return gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    else:
        print("âš ï¸� No points found in TapajÃ³s bounding box.")
        return None

# === Run and Plot ===
gedi_subset = extract_gedi_subset(gedi_path)

if gedi_subset is not None:
    anomalies = gpd.read_file("tapajos_anomalies.geojson")
    fig, ax = plt.subplots(figsize=(10, 10))
    gedi_subset.plot(ax=ax, color='blue', markersize=5, label='GEDI Footprints')
    anomalies.plot(ax=ax, color='red', markersize=10, label='Anomalies')
    plt.title("GEDI L2A Footprints Near TapajÃ³s Terrain Anomalies")
    plt.legend()
    plt.show()



import h5py
import geopandas as gpd
from shapely.geometry import Point

gedi_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"
output_geojson = "gedi_beam0000_footprints.geojson"

points = []

with h5py.File(gedi_path, 'r') as f:
    beam = f["BEAM0000"]

    lat = beam["lat_highestreturn"][:]
    lon = beam["lon_highestreturn"][:]
    elev = beam["elev_highestreturn"][:]
    rh = beam["rh"][:]  # shape: (N, 101)

    rh100 = rh[:, -1]  # Last column is RH100

    for la, lo, rh_val, el in zip(lat, lon, rh100, elev):
        if -90 < la < 90 and -180 < lo < 180:
            points.append({
                "geometry": Point(lo, la),
                "rh100": float(rh_val),
                "elev": float(el)
            })

gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
gdf.to_file(output_geojson, driver="GeoJSON")

print(f"âœ… Saved {len(gdf)} GEDI footprints to: {output_geojson}")




import geopandas as gpd

# Load datasets
gedi = gpd.read_file("gedi_beam0000_footprints.geojson")
anomalies = gpd.read_file("tapajos_anomalies.geojson")

# Buffer anomalies by 50 meters for spatial matching (adjust if needed)
anomalies_utm = anomalies.to_crs(epsg=32721)  # UTM Zone 21S for TapajÃ³s
gedi_utm = gedi.to_crs(epsg=32721)

anomaly_buffer = anomalies_utm.copy()
anomaly_buffer["geometry"] = anomaly_buffer.buffer(50)

# Spatial join
joined = gpd.sjoin(gedi_utm, anomaly_buffer, how="inner", predicate="within")

# Summarize GEDI metrics for each anomaly
summary = joined.groupby("index_right").agg({
    "rh100": ["mean", "std", "min", "max"],
    "elev": ["mean", "std"]
}).round(2)

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()
summary = summary.merge(anomalies.reset_index(), left_on="index_right", right_on="index")

# Save or display
summary.to_csv("anomaly_gedi_summary.csv", index=False)
print(summary[["geometry", "rh100_mean", "elev_mean"]])



import geopandas as gpd
from shapely.geometry import Point

# Load GeoDataFrames
gedi = gpd.read_file("gedi_beam0000_footprints.geojson").to_crs(epsg=32721)
anomalies = gpd.read_file("tapajos_anomalies.geojson").to_crs(epsg=32721)

# Coordinates of Rank 2 and 3 anomalies (convert to UTM)
coords = [(-54.97305, -3.01634), (-54.96735, -3.02819)]
points = [Point(lon, lat) for lon, lat in coords]
targets = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(epsg=32721)

# Buffer radius in meters
buffer_radius = 50
buffers = targets.buffer(buffer_radius)

# Join and summarize GEDI data for each anomaly
for i, buffer in enumerate(buffers):
    subset = gedi[gedi.geometry.within(buffer)]
    if not subset.empty:
        rh_mean = subset["rh100"].mean()
        elev_mean = subset["elev"].mean()
        print(f"Anomaly #{i+2} â€” RH100 mean: {rh_mean:.2f} m, Elevation mean: {elev_mean:.2f} m")
    else:
        print(f"Anomaly #{i+2} â€” âš ï¸� No GEDI footprints found within {buffer_radius} m.")



import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import rasterio
from rasterio import features
from skimage import morphology, measure, filters
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# === Load DTM and extract anomalies using inversion + Otsu method ===
dtm_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"

with rasterio.open(dtm_path) as src:
    dtm = src.read(1)
    transform = src.transform
    dtm_crs = src.crs
    dtm[dtm == src.nodata] = np.nan

    # Normalize and invert elevation
    norm = (dtm - np.nanmin(dtm)) / (np.nanmax(dtm) - np.nanmin(dtm))
    inverted = 1 - norm

    # Threshold using Otsu's method
    thresh = filters.threshold_otsu(inverted[np.isfinite(inverted)])
    binary = inverted > thresh

    # Clean small objects
    cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=100)

    # Label connected regions
    labeled = measure.label(cleaned)
    props = measure.regionprops(labeled)

    # Extract polygons
    shapes = features.shapes(labeled.astype(np.int16), mask=cleaned, transform=transform)
    polygons = [Polygon(shape[0]['coordinates'][0]) for shape in shapes if shape[1] != 0]

# Construct GeoDataFrame
anomalies = gpd.GeoDataFrame(geometry=polygons, crs=dtm_crs).to_crs("EPSG:4326")
anomalies = anomalies[anomalies.geometry.is_valid & anomalies.geometry.notnull()]

if anomalies.empty:
    print("âš ï¸� No valid polygon geometries found in the anomalies dataset.")
else:
    # Define shape metrics
    def compute_shape_metrics(row):
        geom = row.geometry
        area = geom.area
        perimeter = geom.length
        convex_hull_area = geom.convex_hull.area

        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0

        # Convexity
        convexity = area / convex_hull_area if convex_hull_area else 0

        # Bounding rectangle
        min_rect = geom.minimum_rotated_rectangle
        x, y = min_rect.exterior.xy
        edges = [(x[i+1] - x[i], y[i+1] - y[i]) for i in range(4)]
        lengths = [np.hypot(dx, dy) for dx, dy in edges]
        major_axis = max(lengths)
        minor_axis = min(lengths)

        # Elongation & Aspect Ratio
        elongation = major_axis / minor_axis if minor_axis else 0
        aspect_ratio = minor_axis / major_axis if major_axis else 0

        return pd.Series({
            "shape_circularity": round(circularity, 3),
            "shape_convexity": round(convexity, 3),
            "shape_elongation": round(elongation, 3),
            "shape_aspect_ratio": round(aspect_ratio, 3)
        })

    # Apply
    shape_metrics = anomalies.apply(compute_shape_metrics, axis=1)

    # Drop any duplicate geometry columns before creating GeoDataFrame
    if "geometry" in shape_metrics.columns:
        shape_metrics = shape_metrics.drop(columns=["geometry"])

    # Drop conflicting columns before joining
    anomalies = anomalies.drop(columns=shape_metrics.columns.intersection(anomalies.columns), errors="ignore")

    # Join metrics
    anomalies = pd.concat([anomalies.reset_index(drop=True), shape_metrics.reset_index(drop=True)], axis=1)

    # Ensure single geometry column
    anomalies = gpd.GeoDataFrame(anomalies, geometry="geometry", crs="EPSG:4326")

    # Export
    anomalies.to_file("tapajos_anomalies_shape_metrics.geojson", driver="GeoJSON")
    print("âœ… Shape metrics added and saved to 'tapajos_anomalies_shape_metrics.geojson'")
    print(anomalies[["geometry", "shape_circularity", "shape_convexity", "shape_elongation", "shape_aspect_ratio"]].head())

    # === Rank Top Candidates Based on Shape Metrics ===
    anomalies_sorted = anomalies.sort_values(by=["shape_circularity", "shape_convexity"], ascending=False)
    top_anomalies = anomalies_sorted.head(5)
    print("\nğŸ�† Top 5 Ranked Anomalies by Shape Circularity & Convexity:")
    print(top_anomalies[["geometry", "shape_circularity", "shape_convexity", "shape_elongation", "shape_aspect_ratio"]])



import geopandas as gpd

# Load GEDI L2A footprint data (replace with your actual file path)
gedi = gpd.read_file("gedi_beam0000_footprints.geojson")

# Load the enriched anomaly file with shape metrics
anomalies = gpd.read_file("tapajos_anomalies_shape_metrics.geojson")

# Sort and select top 5 anomalies
top_anomalies = anomalies.sort_values(by=["shape_circularity", "shape_convexity"], ascending=False).head(5)

# Reproject to UTM for accurate buffering and intersection (TapajÃ³s is UTM Zone 21S)
gedi_utm = gedi.to_crs(epsg=32721)
anomalies_utm = top_anomalies.to_crs(epsg=32721)

# Buffer anomalies slightly to improve footprint capture (e.g., 50m)
anomalies_buffered = anomalies_utm.copy()
anomalies_buffered["geometry"] = anomalies_buffered.buffer(50)

# Spatial join: GEDI points within buffered anomalies
joined = gpd.sjoin(gedi_utm, anomalies_buffered, how="inner", predicate="within")

# Summarize RH100 and elevation statistics per anomaly
summary = joined.groupby("index_right").agg({
    "rh100": ["mean", "min", "max"],
    "elev": ["mean", "min", "max"]
}).round(2)

# Clean column names
summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.reset_index()
summary = summary.merge(top_anomalies.reset_index(), left_on="index_right", right_on="index")

# Save to CSV and display key fields
summary.to_csv("gedi_anomaly_top5_summary.csv", index=False)
print(summary[["geometry", "rh100_mean", "elev_mean", "shape_circularity", "shape_convexity"]])



import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt
from folium.plugins import MousePosition, Fullscreen, MiniMap, MeasureControl
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("map_box_api_key")

# Load GEDI-confirmed anomaly summary
summary_df = pd.read_csv("gedi_anomaly_top5_summary.csv")
summary_df["geometry"] = summary_df["geometry"].apply(wkt.loads)
summary_gdf = gpd.GeoDataFrame(summary_df, geometry="geometry", crs="EPSG:4326")

# Pick the top GEDI-confirmed anomaly
target = summary_gdf.iloc[0]
centroid = target.geometry.centroid
lat, lon = centroid.y, centroid.x
print(f"ğŸ“� GEDI-Confirmed Anomaly Centroid: Lat {lat:.6f}, Lon {lon:.6f}")

# Initialize map
m = folium.Map(location=[lat, lon], zoom_start=17, tiles=None, control_scale=True)

# Add base layers
folium.TileLayer(
    tiles=f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={secret_value_0}',
    attr='Mapbox Satellite',
    name='Mapbox Satellite',
    overlay=False,
    control=True,
    max_zoom=22,
    tile_size=512,
    zoom_offset=-1
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Imagery Â© Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
    name='Esri Satellite'
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles Â© Esri â€” Source: USGS, NOAA",
    name="Esri Hillshade",
    overlay=True,
    control=True,
    show=False
).add_to(m)

# Add GEDI-confirmed polygon as red border only
folium.GeoJson(
    target.geometry.__geo_interface__,
    name="GEDI-Confirmed Anomaly",
    style_function=lambda x: {
        'fillColor': 'transparent',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.9
    }
).add_to(m)


# Add centroid as a subtle red circle
folium.CircleMarker(
    location=[lat, lon],
    radius=4,
    color='red',
    fill=True,
    fill_color='red',
    fill_opacity=0.9,
    tooltip="GEDI Centroid"
).add_to(m)


# Controls
folium.LayerControl().add_to(m)
Fullscreen().add_to(m)
MiniMap().add_to(m)
MeasureControl().add_to(m)
MousePosition(position='bottomright', separator=' | ', prefix='Coordinates:',
              lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
              lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

m.save("gedi_confirmed_anomaly_map.html")
m




import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from shapely import wkt
from matplotlib.colors import LightSource
from rasterio.mask import mask
from rasterio.plot import show

# Read CSV and parse WKT geometry column
anomaly_df = pd.read_csv("gedi_anomaly_top5_summary.csv")
anomaly_df["geometry"] = anomaly_df["geometry"].apply(wkt.loads)
confirmed = gpd.GeoDataFrame(anomaly_df, geometry="geometry", crs="EPSG:4326")


# === Load DTM ===
dtm_path = "/kaggle/input/tap-a04-15-dtm-tif/TAP_A04_15_DTM.tif"
with rasterio.open(dtm_path) as src:
    dtm_crs = src.crs
    confirmed_utm = confirmed.to_crs(src.crs)  # match projection
    shapes = [mapping(geom) for geom in confirmed_utm.geometry]
    dtm_clip, transform = mask(src, shapes, crop=True)
    meta = src.meta.copy()

# Clean nodata and normalize
dtm_clip = dtm_clip[0]
dtm_clip[dtm_clip == src.nodata] = np.nan

# === Compute Hillshade ===
def compute_hillshade(elevation, azimuth=315, angle_altitude=45):
    ls = LightSource(azdeg=azimuth, altdeg=angle_altitude)
    return ls.hillshade(elevation, vert_exag=1, dx=1, dy=1)

hillshade = compute_hillshade(dtm_clip)

# === Compute Slope ===
from scipy.ndimage import sobel

x = sobel(dtm_clip, axis=1, mode='nearest')
y = sobel(dtm_clip, axis=0, mode='nearest')
slope = np.sqrt(x**2 + y**2)

# === Plot Outputs ===
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(hillshade, cmap='gray')
axs[0].set_title("Hillshade of GEDI-Confirmed Anomaly")
axs[0].axis('off')

slope_im = axs[1].imshow(slope, cmap='magma')
axs[1].set_title("Slope Map (Gradient)")
axs[1].axis('off')
fig.colorbar(slope_im, ax=axs[1], shrink=0.6, label="Gradient")

plt.tight_layout()
plt.show()


import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show
from shapely.geometry import mapping
from shapely import wkt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# === Load GEDI points and confirmed anomaly ===
gedi = gpd.read_file("gedi_beam0000_footprints.geojson")
anomaly_df = pd.read_csv("gedi_anomaly_top5_summary.csv")
anomaly_df["geometry"] = anomaly_df["geometry"].apply(wkt.loads)
confirmed = gpd.GeoDataFrame(anomaly_df, geometry="geometry", crs="EPSG:4326")

# Rename conflicting columns if they exist
gedi = gedi.rename(columns=lambda x: x if x not in ['index_left', 'index_right'] else f"{x}_gedi")
confirmed = confirmed.rename(columns=lambda x: x if x not in ['index_left', 'index_right'] else f"{x}_confirmed")

# Buffer confirmed anomaly to 200m in UTM for better GEDI capture
confirmed_utm = confirmed.to_crs(epsg=32721)
confirmed_utm["geometry"] = confirmed_utm.buffer(200)
confirmed = confirmed_utm.to_crs(epsg=4326)

# Intersect GEDI points with buffered anomaly polygon
gedi = gedi.to_crs("EPSG:4326")
inside = gpd.sjoin(gedi, confirmed, predicate="within")

# Extract coordinates and RH100 values
coords = np.array([[geom.x, geom.y] for geom in inside.geometry])
rh100_vals = inside["rh100"].values
elev_vals = inside["elev"].values

# === Grid setup ===
xmin, ymin, xmax, ymax = confirmed.total_bounds
x_res, y_res = 0.0002, 0.0002  # ~20m resolution
grid_x, grid_y = np.mgrid[xmin:xmax:x_res, ymin:ymax:y_res]

# === Interpolation of RH100 and Elevation using 'nearest' method ===
method_rh = "nearest"
canopy_grid = griddata(coords, rh100_vals, (grid_x, grid_y), method=method_rh)
print(f"âœ… RH100 Interpolation method used: {method_rh}")

method_elv = "nearest"
elev_grid = griddata(coords, elev_vals, (grid_x, grid_y), method=method_elv)
print(f"âœ… Elevation Interpolation method used: {method_elv}")

# === Save RH100 raster ===
transform = from_origin(xmin, ymax, x_res, y_res)
rh100_dataset = rasterio.open(
    "canopy_height_rh100_extended.tif",
    "w",
    driver="GTiff",
    height=canopy_grid.shape[1],
    width=canopy_grid.shape[0],
    count=1,
    dtype=str(canopy_grid.dtype),
    crs="EPSG:4326",
    transform=transform,
)
rh100_dataset.write(np.flipud(canopy_grid.T), 1)
rh100_dataset.close()

# === Save Elevation raster ===
elev_dataset = rasterio.open(
    "gedi_elevation_extended.tif",
    "w",
    driver="GTiff",
    height=elev_grid.shape[1],
    width=elev_grid.shape[0],
    count=1,
    dtype=str(elev_grid.dtype),
    crs="EPSG:4326",
    transform=transform,
)
elev_dataset.write(np.flipud(elev_grid.T), 1)
elev_dataset.close()

# === Visualize RH100 Canopy Height ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.flipud(canopy_grid.T), cmap="YlGn", extent=(xmin, xmax, ymin, ymax))
plt.title("Interpolated GEDI RH100 Canopy Height")
plt.colorbar(label="RH100 Height (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)

# === Visualize Elevation ===
plt.subplot(1, 2, 2)
plt.imshow(np.flipud(elev_grid.T), cmap="plasma", extent=(xmin, xmax, ymin, ymax))
plt.title("Interpolated GEDI Elevation")
plt.colorbar(label="Elevation (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)

plt.tight_layout()
plt.show()



import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

# Load RH100 canopy height raster
with rasterio.open("canopy_height_rh100_extended.tif") as src_rh:
    rh100 = src_rh.read(1)
    rh_profile = src_rh.profile
    rh_transform = src_rh.transform
    rh_bounds = src_rh.bounds

# Load elevation raster
with rasterio.open("gedi_elevation_extended.tif") as src_elev:
    elev = src_elev.read(1)
    elev_transform = src_elev.transform
    elev_bounds = src_elev.bounds

# Sanity check: ensure same shape and transform
assert rh100.shape == elev.shape, "Raster dimensions must match"
assert rh_transform == elev_transform, "Transforms must match"

# Compute difference (RH100 - Elevation)
diff = rh100 - elev

# === Export to GeoTIFF ===
diff_profile = rh_profile.copy()
diff_profile.update(dtype=rasterio.float32, count=1)

with rasterio.open("gedi_rh100_minus_elevation.tif", "w", **diff_profile) as dst:
    dst.write(diff.astype(rasterio.float32), 1)

print("âœ… Exported differential raster to 'gedi_rh100_minus_elevation.tif'")

# === Plot ===
plt.figure(figsize=(10, 6))
extent = [rh_bounds.left, rh_bounds.right, rh_bounds.bottom, rh_bounds.top]
plt.imshow(diff, cmap="coolwarm", extent=extent, origin='upper')
plt.colorbar(label="Canopy Height âˆ’ Elevation (m)")
plt.title("GEDI RH100 âˆ’ Elevation Differential Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()



import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.enums import Resampling
from shapely import wkt
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt

# âœ… Load CSV correctly, then convert geometry column from WKT
anomaly_df = pd.read_csv("gedi_anomaly_top5_summary.csv")
anomaly_df["geometry"] = anomaly_df["geometry"].apply(wkt.loads)
confirmed = gpd.GeoDataFrame(anomaly_df, geometry="geometry", crs="EPSG:4326")

# === Clip elevation raster ===
with rasterio.open("gedi_elevation_extended.tif") as src:
    clipped, transform = mask(src, confirmed.geometry, crop=True)
    elev_clipped = clipped[0]
    meta = src.meta.copy()
    meta.update({"height": elev_clipped.shape[0], "width": elev_clipped.shape[1], "transform": transform})

# === Compute terrain roughness (std dev in 3x3 window) ===
roughness = generic_filter(elev_clipped, np.std, size=3)

# === Compute Terrain Rugosity Index (TRI) ===
def compute_tri(values):
    center = values[4]
    neighbors = np.delete(values, 4)
    return np.sum(np.abs(neighbors - center))

tri = generic_filter(elev_clipped, compute_tri, size=3)

# === Save roughness raster ===
with rasterio.open("terrain_roughness.tif", "w", **meta) as dst:
    dst.write(roughness, 1)

# === Save TRI raster ===
with rasterio.open("terrain_tri.tif", "w", **meta) as dst:
    dst.write(tri, 1)

# === Also save TRI with expected filename ===
with rasterio.open("tri_gedi_confirmed.tif", "w", **meta) as dst:
    dst.write(tri, 1)

# === Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
extent = [transform[2], transform[2] + transform[0] * elev_clipped.shape[1],
          transform[5] + transform[4] * elev_clipped.shape[0], transform[5]]

axes[0].imshow(roughness, cmap="viridis", extent=extent, origin="upper")
axes[0].set_title("Terrain Roughness (StDev)")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

im = axes[1].imshow(tri, cmap="plasma", extent=extent, origin="upper")
axes[1].set_title("Terrain Rugosity Index (TRI)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.colorbar(im, ax=axes.ravel().tolist(), label="TRI")
plt.show()



import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt

# === Load GEDI-confirmed anomaly polygon ===
summary = pd.read_csv("gedi_anomaly_top5_summary.csv")
summary["geometry"] = summary["geometry"].apply(wkt.loads)
confirmed = gpd.GeoDataFrame(summary, geometry="geometry", crs="EPSG:4326")

# === Load and clip the elevation raster ===
with rasterio.open("gedi_elevation_extended.tif") as src:
    out_image, out_transform = mask(src, confirmed.geometry, crop=True)
    out_meta = src.meta.copy()
    elevation = out_image[0]

# === Compute curvature derivatives ===
xres = out_transform[0]
yres = -out_transform[4]

dy, dx = np.gradient(elevation, yres, xres)
dyy, _ = np.gradient(dy, yres, xres)
_, dxx = np.gradient(dx, yres, xres)
dxy = np.gradient(dx, yres, axis=0)

# General curvature
curvature = dxx + dyy

# Profile curvature (affects flow acceleration)
grad_mag = np.sqrt(dx**2 + dy**2)
profile_curvature = ((dx**2 * dxx + 2*dx*dy*dxy + dy**2 * dyy) / (grad_mag**2 + 1e-6))

# Planform curvature (affects flow direction)
plan_curvature = ((dy**2 * dxx - 2*dx*dy*dxy + dx**2 * dyy) / (grad_mag**2 + 1e-6))

# === Plot all curvatures ===
extent = [out_transform[2], 
          out_transform[2] + elevation.shape[1] * xres,
          out_transform[5] + elevation.shape[0] * out_transform[4], 
          out_transform[5]]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ["General Curvature", "Profile Curvature", "Planform Curvature"]
curvatures = [curvature, profile_curvature, plan_curvature]
cmaps = ["RdBu", "coolwarm", "PiYG"]

for ax, data, title, cmap in zip(axs, curvatures, titles, cmaps):
    im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle("Terrain Curvature Analysis (GEDI-Confirmed Anomaly)", fontsize=16)
plt.tight_layout()
plt.show()

# === Save curvature rasters ===
out_meta.update({
    "driver": "GTiff",
    "height": curvature.shape[0],
    "width": curvature.shape[1],
    "transform": out_transform,
    "count": 1
})

with rasterio.open("curvature_general.tif", "w", **out_meta) as dst:
    dst.write(curvature, 1)

with rasterio.open("curvature_profile.tif", "w", **out_meta) as dst:
    dst.write(profile_curvature, 1)

with rasterio.open("curvature_planform.tif", "w", **out_meta) as dst:
    dst.write(plan_curvature, 1)




import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from shapely import wkt
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster, reshape_as_image

# === Load GEDI-confirmed anomaly polygon ===
df = pd.read_csv("gedi_anomaly_top5_summary.csv")
df["geometry"] = df["geometry"].apply(wkt.loads)
confirmed = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# Calculate the gedi confirmed slope

# Clip the elevation raster using polygon
with rasterio.open("gedi_elevation_extended.tif") as src:
    if src.crs != confirmed.crs:
        confirmed = confirmed.to_crs(src.crs)
    out_image, out_transform = mask(src, confirmed.geometry, crop=True)
    elev = out_image[0]
    meta = src.meta.copy()
    meta.update({
        "height": elev.shape[0],
        "width": elev.shape[1],
        "transform": out_transform
    })

# Handle potential all-NaN output
if np.isnan(elev).all():
    raise ValueError("Clipped elevation raster is all NaNs. Check polygon overlap.")

# Compute slope from elevation
xres = out_transform[0]
yres = -out_transform[4]
dy, dx = np.gradient(elev, yres, xres)
slope = np.sqrt(dx**2 + dy**2)

# Save slope raster
with rasterio.open("slope_gedi_confirmed.tif", "w", **meta) as dst:
    dst.write(slope, 1)

# Function to extract stats from raster within polygon
def zonal_stats_manual(raster_path, polygon_gdf):
    with rasterio.open(raster_path) as src:
        polygon_gdf = polygon_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, polygon_gdf.geometry, crop=True)
        data = out_image[0]
        data = data[data != src.nodata]
        data = data[np.isfinite(data)]

        return {
            "mean": round(np.nanmean(data), 2),
            "min": round(np.nanmin(data), 2),
            "max": round(np.nanmax(data), 2),
            "std": round(np.nanstd(data), 2),
            "range": round(np.nanmax(data) - np.nanmin(data), 2)
        }

# === Compute statistics for each terrain metric ===
canopy_stats = zonal_stats_manual("canopy_height_rh100_extended.tif", confirmed)
elev_stats = zonal_stats_manual("gedi_elevation_extended.tif", confirmed)
slope_stats = zonal_stats_manual("slope_gedi_confirmed.tif", confirmed)
tri_stats = zonal_stats_manual("tri_gedi_confirmed.tif", confirmed)
curv_gen_stats = zonal_stats_manual("curvature_general.tif", confirmed)
curv_prof_stats = zonal_stats_manual("curvature_profile.tif", confirmed)
curv_plan_stats = zonal_stats_manual("curvature_planform.tif", confirmed)

# === Combine into one DataFrame ===
summary = pd.DataFrame({
    "Canopy Height (RH100)": canopy_stats,
    "Elevation": elev_stats,
    "Slope": slope_stats,
    "TRI": tri_stats,
    "Curvature General": curv_gen_stats,
    "Curvature Profile": curv_prof_stats,
    "Curvature Planform": curv_plan_stats
})

# Transpose for readability
summary = summary.T.rename(columns={
    "mean": "Mean", "min": "Min", "max": "Max",
    "std": "Std Dev", "range": "Range"
})

# Export
summary.to_csv("terrain_canopy_morphometrics_summary.csv")
print(summary)



import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
import json
from geojson import Feature, FeatureCollection, Polygon as GJPolygon

# === Step 1: Open GEDI file ===
file_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"

with h5py.File(file_path, 'r') as f:
    beams = [b for b in f.keys() if b.startswith("BEAM")]
    print("Beams found:", beams)

    all_coords = []
    for beam in beams:
        lat_key = f"{beam}/geolocation/lat_lowestmode_a1"
        lon_key = f"{beam}/geolocation/lon_lowestmode_a1"
        if lat_key in f and lon_key in f:
            lat = f[lat_key][:]
            lon = f[lon_key][:]
            mask = (~np.isnan(lat)) & (~np.isnan(lon))
            lat, lon = lat[mask], lon[mask]
            for la, lo in zip(lat, lon):
                all_coords.append((beam, la, lo))

print(f"âœ… Total valid footprint points: {len(all_coords)}")
print("ğŸ§­ Sample coordinates:")
for b, la, lo in all_coords[:5]:
    print(f"Beam: {b}, Lat: {la:.6f}, Lon: {lo:.6f}")

# === Step 2: Bin footprints into grid cells ===
df = pd.DataFrame(all_coords, columns=["beam", "lat", "lon"])

# Match your GEDI granuleâ€™s real bounds
df = df[
    (df["lat"] >= 1.0) & (df["lat"] <= 3.0) &
    (df["lon"] >= -59.5) & (df["lon"] <= -57.0)
]

lat_grid_size = 0.05
lon_grid_size = 0.05
df["lat_bin"] = (df["lat"] // lat_grid_size) * lat_grid_size
df["lon_bin"] = (df["lon"] // lon_grid_size) * lon_grid_size

counts = df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")
top_cells = counts.sort_values("count", ascending=False).head(5)

# === Step 3: Create 5 bounding boxes ===
bbox_list = []
geojson_features = []

print("\nâœ… 5 High-Density Footprint Bounding Boxes (WKT):")
for i, row in enumerate(top_cells.itertuples(), 1):
    lat, lon = row.lat_bin, row.lon_bin
    poly = box(lon, lat, lon + lon_grid_size, lat + lat_grid_size)
    bbox_list.append(poly)
    centroid = poly.centroid
    print(f"Footprint {i}: {poly.wkt}")
    print(f" - Center: Lat={centroid.y:.4f}, Lon={centroid.x:.4f}")

    # GeoJSON formatting (lon, lat)
    gj_coords = [[
        [lon, lat],
        [lon, lat + lat_grid_size],
        [lon + lon_grid_size, lat + lat_grid_size],
        [lon + lon_grid_size, lat],
        [lon, lat]  # close
    ]]
    feature = Feature(geometry=GJPolygon(gj_coords), properties={"id": f"GEDI_Footprint_{i}"})
    geojson_features.append(feature)

# === Step 4: Save GeoJSON ===
fc = FeatureCollection(geojson_features)
with open("footprints.geojson", "w") as f:
    json.dump(fc, f, indent=2)
print("âœ… Saved 5 bounding boxes as 'footprints.geojson'")



import geopandas as gpd
import matplotlib.pyplot as plt

# âœ… Load existing GeoJSON
gdf = gpd.read_file("footprints.geojson")

# âœ… Add IDs if missing
if "id" not in gdf.columns:
    gdf["id"] = [f"Footprint_{i+1}" for i in range(len(gdf))]

# âœ… Assign colors
colors = ['blue', 'green', 'red', 'orange', 'purple']
gdf['color'] = colors[:len(gdf)]

# âœ… Save WKT to text file for checkpoint logs
with open("footprints.txt", "w") as f:
    for poly in gdf.geometry:
        f.write(poly.wkt + "\n")

# âœ… Plot each boundary with its assigned color
fig, ax = plt.subplots(figsize=(8, 6))

for i, row in gdf.iterrows():
    gpd.GeoSeries(row.geometry.boundary).plot(ax=ax, edgecolor=row['color'], linewidth=2, label=row['id'])

plt.title("GEDI Footprint Bounding Boxes (Color-coded)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.legend()
plt.show()




import folium
from folium import Map, Marker
from folium.plugins import MousePosition
import geopandas as gpd
from shapely.ops import unary_union
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("map_box_api_key")

# âœ… Load the footprints GeoJSON
gdf = gpd.read_file("footprints.geojson")

# âœ… Color palette
colors = ['blue', 'green', 'red', 'orange', 'purple']

# âœ… Initialize map centered over Amazon
m = folium.Map(location=[-3, -56], zoom_start=5, control_scale=True, tiles=None)

# Coordinates of the curicform pattern spot
lat, lon = 1.155203, -57.921038

folium.CircleMarker(
   location=[lat, lon],
   radius=2,
   color='red',
   fill=True,
   fill_color='red',
   fill_opacity=1.0,
   tooltip="ğŸ“� Candidate (Cruciform Pattern) â€” GEDI Footprint 2 (1.155203, -57.921038)",
   popup="ğŸŒŸ Cruciform Candidate Site!"
).add_to(m)

# Base layers
folium.TileLayer(
    tiles=f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={secret_value_0}',
    attr='Mapbox Satellite',
    name='Mapbox Satellite',
    overlay=False,
    control=True,
    max_zoom=22,
    tile_size=512,
    zoom_offset=-1
).add_to(m)

# âœ… Add base layers with explicit attributions
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="Â© OpenStreetMap contributors",
    name="OpenStreetMap"
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles Â© Esri â€” Source: USGS, NOAA",
    name="Esri Hillshade",
    overlay=True,
    control=True,
    show=False
).add_to(m)

folium.TileLayer(
    tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='Map data: Â© OpenStreetMap contributors, SRTM | Map style: Â© OpenTopoMap (CC-BY-SA)',
    name='OpenTopoMap'
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Imagery Â© Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
    name='Esri Satellite'
).add_to(m)

# Add a MousePosition plugin to show coordinates as you hover
MousePosition(
    position='bottomright',
    separator=' | ',
    prefix='Coordinates:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

# âœ… Add GEDI footprint polygons
all_bounds = []
for i, row in gdf.iterrows():
    coords = list(row.geometry.exterior.coords)
    latlon_coords = [[lat, lon] for lon, lat in coords]

    folium.Polygon(
        locations=latlon_coords,
        color=colors[i % len(colors)],
        fill=False,
        weight=2,
        opacity=0.8,
        tooltip=f"GEDI Footprint {i+1}"
    ).add_to(m)

    all_bounds.append(row.geometry)

# âœ… Fit map to bounds
combined = unary_union(all_bounds)
minx, miny, maxx, maxy = combined.bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

# âœ… Add layer controls
folium.LayerControl().add_to(m)

# âœ… Show the map
m



from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# âœ… Load OpenAI API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

anomaly_description = """
Coordinates: Latitude 1.155203, Longitude -57.921038
Shape: Cruciform clearing with four orthogonal arms extending from a central circular hub.
Diameter: Approx. 180 meters across the full structure.
Topography: Slight central rise (~1â€“2m elevation), with symmetric terrain depressions along arm axes.
Surroundings: Upland forest zone, between two rivers, remote from modern development.
Pattern: Radial symmetry and alignment that may correspond to the Southern Cross constellation.
"""

prompt = f"""
Based on the following anomaly detected in the Amazon rainforest via GEDI LiDAR and visual analysis, suggest what type of ancient Amazonian or Andean cultural or ceremonial structure this might represent. Include historical or archaeological parallels where relevant.

{anomaly_description}
"""

# âœ… Call GPT-4.1
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

# âœ… Print Logs
print("\U0001f9e0 Model Used:", response.model)
print("\nğŸ“œ GPT Response:\n", response.choices[0].message.content)

# âœ… Save response for reproducibility
with open("gpt_response_checkpoint1.txt", "w") as f:
    f.write(response.choices[0].message.content)



import h5py
import numpy as np
import pandas as pd

# Bounding box for Footprint 2
lat_min, lat_max = 1.1500, 1.2000
lon_min, lon_max = -57.9500, -57.9000

# Open the GEDI file
file_path = "/kaggle/input/gedi02-a-2024282205519-o32987-04/GEDI02_A_2024282205519_O32987_04_T06355_02_004_02_V002.h5"
f = h5py.File(file_path, 'r')

beam = 'BEAM0000'
group = f[beam]

# Use verified dataset paths
lat = group['geolocation/lat_lowestmode_a1'][:]
lon = group['geolocation/lon_lowestmode_a1'][:]
ground_elev = group['geolocation/elev_lowestmode_a1'][:]
rh_array = group['geolocation/rh_a1'][:]  # 101-element array per shot
shot_num = group['shot_number'][:]

# RH100 is the last column in rh_a1 (index 100)
rh100 = rh_array[:, 100]

# Filter by bounding box
mask = (
    (lat >= lat_min) & (lat <= lat_max) &
    (lon >= lon_min) & (lon <= lon_max)
)

# Create DataFrame
df_subset = pd.DataFrame({
    'shot_number': shot_num[mask],
    'latitude': lat[mask],
    'longitude': lon[mask],
    'ground_elev': ground_elev[mask],
    'rh100': rh100[mask]
})

# Output preview
print(df_subset.head())
print(f"âœ… Extracted {len(df_subset)} GEDI shots in Footprint 2.")

# Save to CSV
df_subset.to_csv("gedi_subset_footprint2.csv", index=False)
f.close()


import pandas as pd
import matplotlib.pyplot as plt

# Load GEDI footprint data
df = pd.read_csv("gedi_subset_footprint2.csv")  # Ensure this CSV is in your working directory

# Plot: Ground Elevation Histogram
plt.figure(figsize=(10, 4))
plt.hist(df['ground_elev'], bins=20, color='saddlebrown', edgecolor='black')
plt.title('Distribution of Ground Elevation from GEDI Footprint 2')
plt.xlabel('Ground Elevation (m)')
plt.ylabel('Number of Shots')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: RH100 (Canopy Height) Histogram
plt.figure(figsize=(10, 4))
plt.hist(df['rh100'], bins=20, color='darkgreen', edgecolor='black')
plt.title('Distribution of RH100 (Max Canopy Height) from GEDI Footprint 2')
plt.xlabel('RH100 (m)')
plt.ylabel('Number of Shots')
plt.grid(True)
plt.tight_layout()
plt.show()



# ğŸ“¦ Required Imports
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ğŸ“„ Load the filtered GEDI data (assumes previous step created this file)
df = pd.read_csv("gedi_subset_footprint2.csv")

# ğŸ—ºï¸� Rasterize RH100 into a grid (adjust resolution as needed)
res = 0.0005  # ~55m
lat_bins = np.arange(df['latitude'].min(), df['latitude'].max(), res)
lon_bins = np.arange(df['longitude'].min(), df['longitude'].max(), res)

raster = np.full((len(lat_bins), len(lon_bins)), np.nan)
lat_idx = np.digitize(df['latitude'], lat_bins) - 1
lon_idx = np.digitize(df['longitude'], lon_bins) - 1

for y, x, rh in zip(lat_idx, lon_idx, df['rh100']):
    if 0 <= y < raster.shape[0] and 0 <= x < raster.shape[1]:
        if np.isnan(raster[y, x]):
            raster[y, x] = rh
        else:
            raster[y, x] = (raster[y, x] + rh) / 2  # average RH100

# ğŸ§ª Cruciform Pattern Detection from GEDI RH100 Raster

# Apply Gaussian filter and normalize
image = np.nan_to_num(raster, nan=0.0)
image_smooth = gaussian_filter(image, sigma=1)
img_uint8 = cv2.normalize(image_smooth, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Binary threshold to isolate high canopy zones
_, binary = cv2.threshold(img_uint8, np.percentile(img_uint8, 90), 255, cv2.THRESH_BINARY)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

cruciform_count = 0

for cnt in contours:
    if len(cnt) < 5:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h else 0
    if 0.7 < aspect_ratio < 1.3 and 10 < w < 100 and 10 < h < 100:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if 4 <= len(approx) <= 12:
            cv2.drawContours(canvas, [cnt], -1, (0, 0, 255), 2)
            cruciform_count += 1


## Plot detected cruciform-like features
plt.figure(figsize=(8, 6))
plt.imshow(canvas)
plt.title("Detected Cruciform-like Features (GEDI RH100)")
plt.axis("off")
plt.show()

print(f"ğŸ”� Total Cruciform-Like Features Detected: {cruciform_count}")



# âœ… Reproducible Notebook Code for OpenAI to Z Submission
# Generated with the assistance of OpenAI's Magellan GPT for the OpenAI to Z Challenge.
# This notebook loads 25m resolution TanDEM-X & GEDI-derived rasters,
# crops to a defined region based on GEDI detection coordinates,
# and applies visual + geometric pattern detection for archaeological features.

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import cv2
from scipy.ndimage import gaussian_filter

# === Step 1: Parameters ===
# Candidate region center (from GEDI footprint 2)
candidate_lat = 1.155203
candidate_lon = -57.921038
crop_size_deg = 0.05  # roughly 5.5 km x 5.5 km

# === Step 2: Define Bounding Box ===
lat_min = candidate_lat - crop_size_deg / 2
lat_max = candidate_lat + crop_size_deg / 2
lon_min = candidate_lon - crop_size_deg / 2
lon_max = candidate_lon + crop_size_deg / 2

# === Step 3: Load and Crop GeoTIFF ===
def crop_raster(path):
    with rasterio.open(path) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        data = src.read(1, window=window)
        data = np.where(data == src.nodata, np.nan, data)
        transform = src.window_transform(window)
    return data, transform

height_data, height_transform = crop_raster("/kaggle/input/height-amazon/height_amazon_25m.tif")
biomass_data, _ = crop_raster("/kaggle/input/biomass-amazon/biomass_amazon_25m.tif")
disturb_data, _ = crop_raster("/kaggle/input/disturbance-amazon/disturbance_amazon_25m.tif")

# === Step 4: Visualize ===
def plot_layer(data, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.show()

plot_layer(height_data, "Canopy Height (25m)", cmap="terrain")
plot_layer(biomass_data, "Biomass (25m)", cmap="YlGn")
plot_layer(disturb_data, "Disturbance (25m)", cmap="hot")

# === Step 5: Pattern Detection (Contour) ===
def detect_patterns(data):
    image = np.nan_to_num(data, nan=0.0)
    image_smooth = gaussian_filter(image, sigma=1)
    normed = cv2.normalize(image_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(normed, np.percentile(normed, 90), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h else 0
        if 0.5 < aspect < 2.0 and 10 < w < 100 and 10 < h < 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if 4 <= len(approx) <= 12:
                cv2.drawContours(result, [cnt], -1, (0, 0, 255), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(result)
    plt.title("Detected Structural Patterns (Contours)")
    plt.axis("off")
    plt.show()

# Run pattern detection on height layer
detect_patterns(height_data)



import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from rasterio.windows import from_bounds
from affine import Affine

# === Reuse bounding box and raster ===
with rasterio.open("/kaggle/input/height-amazon/height_amazon_25m.tif") as src:
    window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
    transform = src.window_transform(window)

# === Helper: Convert pixel to lat/lon ===
def px_to_latlon(x, y, transform):
    lon, lat = transform * (x, y)
    return (lon, lat)

# === Initialize feature list ===
features = []

# === Extract contour polygons ===
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h else 0
    approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    print(f"â†’ w: {w}, h: {h}, aspect: {aspect:.2f}, vertices: {len(approx)}")
    if 0.4 < aspect < 2.5 and 5 < w < 150 and 5 < h < 150:
        if len(approx) >= 4:
            coords = [px_to_latlon(p[0][0], p[0][1], transform) for p in approx]
            print("Polygon coords sample:", coords[:3])  # print first few points
            poly = Polygon(coords)
            if not poly.is_valid:
               print("â�Œ Invalid polygon detected")
            else:
               features.append(poly)


# === Save to GeoJSON ===
gdf = gpd.GeoDataFrame(geometry=features, crs="EPSG:4326")
gdf.to_file("gedi_detected_polygons.geojson", driver="GeoJSON")
print(f"âœ… Accepted contour with {len(approx)} vertices")
print("âœ… Saved gedi_detected_polygons.geojson")

# Log metadata for reproducibility
print(f"ğŸ§¾ Polygon detection executed on: {pd.Timestamp.now()}")
print(f"Total polygons detected: {len(features)}")

# Print centroid of the first polygon (if any)
if features:
    centroid = features[0].centroid
    print(f"Centroid of first polygon: ({centroid.y:.6f}, {centroid.x:.6f})")
else:
    print("âš ï¸� No valid polygon detected in this run.")



import folium
from folium.plugins import Draw, MousePosition

# === Center on Cruciform Anomaly ===
candidate_lat = 1.155203
candidate_lon = -57.921038

# === Create Folium Map with Esri Satellite ===
m = folium.Map(location=[candidate_lat, candidate_lon], zoom_start=16, control_scale=True)

# Add Esri Satellite Tile
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Esri Satellite',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

# Coordinates of the curicform pattern spot
lat, lon = 1.155203, -57.921038

folium.CircleMarker(
   location=[lat, lon],
   radius=1,
   color='red',
   fill=True,
   fill_color='red',
   fill_opacity=1.0,
   tooltip="ğŸ“� Candidate (Cruciform Pattern) â€” GEDI Footprint 2 (1.155203, -57.921038)",
   popup="ğŸŒŸ Cruciform Candidate Site!"
).add_to(m)

# Add MousePosition to see coordinates
MousePosition(
    position='bottomright',
    separator=' | ',
    prefix='Coordinates:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

# === Add Drawing Tool with GeoJSON Export ===
Draw(
    export=True,
    filename='drawn_cruciform_crux_oriented.geojson',
    draw_options={
        'polyline': True,
        'rectangle': True,
        'circle': False,
        'circlemarker': False,
        'marker': False,
        'polygon': {
            'shapeOptions': {
                'color': 'yellow',
                'fillOpacity': 0.3
            }
        }
    },
    edit_options={'edit': True}
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# === Save and Show ===
m.save("draw_cruciform_polyline_template.html")
m



import folium
import geopandas as gpd
from folium.plugins import MousePosition

# === Load Drawn Cruciform Template ===
# We have uploaded drawn cruicoform from the file in the dataset as we created this before via the interactive drawing tool
crux_template_gdf = gpd.read_file("/kaggle/input/drawn-cruicoform-crux/drawn_cruciform_crux_oriented.geojson")
# === Reproject to UTM Zone appropriate for Amazon (~Zone 21S or 22S for western Brazil)
projected = crux_template_gdf.to_crs(epsg=32721)  # Use appropriate EPSG if needed
centroid_proj = projected.geometry.centroid.iloc[0]

template_latlon = (1.155203, -57.921038)

# === Create Map Centered at Cruciform Template ===
m = folium.Map(location=template_latlon, zoom_start=15)

# Add Esri Satellite Basemap
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Esri Satellite',
    name='Esri Satellite',
    overlay=False
).add_to(m)

# Add Template Polygon in Yellow
folium.GeoJson(
    crux_template_gdf,
    name="ğŸ“� Crux-Inspired Cruciform Template",
    style_function=lambda x: {"color": "gold", "weight": 0.6, "fillOpacity": 0.1}
).add_to(m)

# Coordinates of the curicform pattern spot
lat, lon = 1.155203, -57.921038

folium.CircleMarker(
   location=template_latlon,
   radius=1,
   color='red',
   fill=True,
   fill_color='red',
   fill_opacity=1.0,
   tooltip="ğŸ“� Candidate (Cruciform Pattern) â€” GEDI Footprint 2 (1.155203, -57.921038)",
   popup="ğŸŒŸ Cruciform Candidate Site!"
).add_to(m)


# Optional: Add GEDI detection (if exists)
try:
    gedi_gdf = gpd.read_file("gedi_detected_polygons.geojson")
    folium.GeoJson(gedi_gdf, name="GEDI Detected Polygon").add_to(m)
except:
    pass

# Coordinate hover plugin
MousePosition(
    position="bottomright",
    separator=" | ",
    prefix="Coordinates:",
    lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

folium.LayerControl().add_to(m)
m.save("cruciform_template_map.html")
m



# âœ… Geometric Signature Matching with Manual Template (drawn_cruciform.geojson)
# Compares GEDI-derived polygon to user-drawn cruciform using Procrustes and Hu Moments

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.affinity import scale, translate
from scipy.spatial import procrustes
import cv2
import json

# === Load Detected Polygon ===
with open("gedi_detected_polygons.geojson") as f:
    gedi_geom = shape(json.load(f)["features"][0]["geometry"])
polygon = gedi_geom

# === Load Drawn Cruciform Template ===
with open("/kaggle/input/drawn-cruicoform-crux/drawn_cruciform_crux_oriented.geojson") as f:
    cruciform_geom = shape(json.load(f)["features"][0]["geometry"])
cruciform = cruciform_geom

# === Normalize and Sample ===
def normalize_and_sample(poly, num_points=100):
    bounds = poly.bounds
    norm = translate(scale(poly, xfact=1/(bounds[2]-bounds[0]), yfact=1/(bounds[3]-bounds[1]),
                           origin='center'), xoff=-poly.centroid.x, yoff=-poly.centroid.y)
    line = norm.boundary
    equally_spaced = [line.interpolate(i / num_points, normalized=True).coords[0] for i in range(num_points)]
    return np.array(equally_spaced)

pts1 = normalize_and_sample(polygon)
pts2 = normalize_and_sample(cruciform)

# === Procrustes Alignment ===
m1, m2, disparity = procrustes(pts1, pts2)
print(f"Procrustes disparity: {disparity:.3f}")

# === Plot Procrustes Alignment ===
plt.figure(figsize=(6, 6))
plt.plot(m1[:, 0], m1[:, 1], label='GEDI Polygon', lw=2)
plt.plot(m2[:, 0], m2[:, 1], label='Drawn Cruciform Template', lw=2, linestyle='--')
plt.legend()
plt.title("Procrustes Shape Alignment")
plt.axis('equal')
plt.grid(True)
plt.show()

# === Hu Moments ===
def rasterize_shape(shape, size=256):
    minx, miny, maxx, maxy = shape.bounds
    scale_x = scale_y = size / max(maxx - minx, maxy - miny)
    norm_shape = translate(scale(shape, xfact=scale_x, yfact=scale_y, origin=(minx, miny)),
                           xoff=5, yoff=5)
    img = np.zeros((size+10, size+10), dtype=np.uint8)
    coords = np.array(norm_shape.exterior.coords, np.int32)
    cv2.fillPoly(img, [coords], 255)
    return img

img1 = rasterize_shape(polygon)
img2 = rasterize_shape(cruciform)

moments1 = cv2.HuMoments(cv2.moments(img1)).flatten()
moments2 = cv2.HuMoments(cv2.moments(img2)).flatten()

hu1 = -np.sign(moments1) * np.log10(np.abs(moments1) + 1e-10)
hu2 = -np.sign(moments2) * np.log10(np.abs(moments2) + 1e-10)

dist = np.linalg.norm(hu1 - hu2)
print("Hu Moment Vector Distance:", dist)

# === Plot Hu Moments ===
plt.figure(figsize=(6, 4))
plt.plot(hu1, label='GEDI Polygon')
plt.plot(hu2, label='Drawn Cruciform Template', linestyle='--', marker='o')
plt.title("Hu Moment Shape Descriptors")
plt.legend()
plt.grid(True)
plt.ylim(-12, 12)  # Force range to show both
plt.show()




# âœ… Multi-Resolution Cruciform Detection (100m Raster)
# Re-runs the shape detection pipeline on lower-res data to test persistence

import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.windows import from_bounds
from affine import Affine
import rasterio
import pandas as pd
from rasterio.plot import show
import matplotlib.pyplot as plt

# === Load 100m Canopy Height Raster ===
raster_path = "/kaggle/input/height-amazon-100m/height_amazon_100m.tif"
with rasterio.open(raster_path) as src:
    data = src.read(1)
    transform = src.transform
    profile = src.profile

# === Preprocess: Normalize and Threshold for Contour Extraction ===
norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary = cv2.threshold(norm, 130, 255, cv2.THRESH_BINARY)  # You may need to tune this

# === Find Contours ===
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Convert pixel coordinates to lat/lon ===
def px_to_latlon(x, y, transform):
    lon, lat = transform * (x, y)
    return (lon, lat)

# === Extract valid polygons ===
features = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h else 0
    approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    if 0.4 < aspect < 2.5 and 5 < w < 150 and 5 < h < 150:
        if len(approx) >= 4:
            coords = [px_to_latlon(p[0][0], p[0][1], transform) for p in approx]
            poly = Polygon(coords)
            if poly.is_valid:
                features.append(poly)

# === Save Detected Polygons to GeoJSON ===
gdf = gpd.GeoDataFrame(geometry=features, crs="EPSG:4326")
gdf.to_file("gedi_detected_polygons_100m.geojson", driver="GeoJSON")
print(f"âœ… Saved {len(gdf)} polygons to gedi_detected_polygons_100m.geojson")



# âœ… Enhanced Visualization: Overlay Detected Polygons on 100m Raster

import matplotlib.pyplot as plt
from rasterio.plot import show
import geopandas as gpd
import rasterio
from shapely.geometry import Point

# === Load Raster and Detected Polygons ===
raster_path = "/kaggle/input/height-amazon-100m/height_amazon_100m.tif"
geojson_path = "gedi_detected_polygons_100m.geojson"

with rasterio.open(raster_path) as src:
    data = src.read(1)
    transform = src.transform

# Load GeoJSON with detected polygons
gdf = gpd.read_file(geojson_path)

# Create target point
target = gpd.GeoSeries([Point(-57.921038, 1.155203)], crs="EPSG:4326")

# === Plot: Raster + Polygon Overlay ===
fig, ax = plt.subplots(figsize=(10, 6))

# Show raster as background
show(data, transform=transform, cmap="inferno", ax=ax, alpha=0.7)

# Overlay detected polygons
gdf.boundary.plot(ax=ax, edgecolor="cyan", linewidth=0.6, label="Detected Polygons")

# Overlay target point
target.plot(ax=ax, color="red", markersize=20, label="Target Cruciform")

# Customize plot
plt.title("Overlay of Detected Cruciform Candidates on 100m Canopy Raster")
plt.axis("off")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()




import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic

# === Parameters ===
known_centroid = (1.155203, -57.921038)  # Latitude, Longitude

# === Load detected polygons from 100m raster analysis ===
gdf_100m = gpd.read_file("gedi_detected_polygons_100m.geojson")

# === Reproject to UTM for accurate centroid calculation ===
gdf_utm = gdf_100m.to_crs(epsg=32721)
gdf_utm['centroid_utm'] = gdf_utm.geometry.centroid

# === Reproject centroids back to WGS84 for distance calc ===
centroids_geo = gpd.GeoSeries(gdf_utm['centroid_utm'], crs=32721).to_crs(epsg=4326)

# === Compute geodesic distance to known cruciform centroid ===
gdf_utm['distance_km'] = centroids_geo.apply(
    lambda pt: geodesic((pt.y, pt.x), known_centroid).km
)

# === Find the closest polygon ===
closest_polygon = gdf_utm.loc[gdf_utm['distance_km'].idxmin()]
print(f"âœ… Closest polygon is {closest_polygon['distance_km']:.2f} km from target")

# === Save it as a single-feature GeoJSON for follow-up comparison ===
closest_gdf = gpd.GeoDataFrame([closest_polygon], geometry='geometry', crs=gdf_utm.crs).to_crs(epsg=4326)
closest_gdf.drop(columns='centroid_utm').to_file("closest_polygon_100m.geojson", driver="GeoJSON")




# âœ… 100m Biomass & Disturbance Profiling (Masked to Cruciform Polygon)
# Magellan (OpenAI to Z Challenge assistant)

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore", message=".*'partition' will ignore the 'mask' of the MaskedArray.*")

# === Load Cruciform Detection Polygon (closest match at 25m) ===
gedi_gdf = gpd.read_file("gedi_detected_polygons.geojson")
focus_geom = gedi_gdf.geometry.iloc[0:1]  # Use only the best polygon match

# === Load and Mask Biomass 100m ===
with rasterio.open("/kaggle/input/biomass-amazon-100m/biomass_amazon_100m.tif") as src:
    biomass_data, _ = mask(src, focus_geom, crop=True)
    biomass_masked = np.ma.masked_equal(biomass_data[0], src.nodata)

# === Load and Mask Disturbance 100m ===
with rasterio.open("/kaggle/input/disturbance-amazon-100m/disturbance_amazon_100m.tif") as src:
    disturb_data, _ = mask(src, focus_geom, crop=True)
    disturb_masked = np.ma.masked_equal(disturb_data[0], src.nodata)

# === Diagnostics for Disturbance Data ===
unique_vals = np.unique(disturb_masked.compressed()) if disturb_masked.compressed().size > 0 else []

# === Plot with Contrast Stretching ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biomass
biomass_norm = Normalize(vmin=np.percentile(biomass_masked, 5), vmax=np.percentile(biomass_masked, 95))
im1 = axes[0].imshow(biomass_masked, cmap="YlGn", norm=biomass_norm)
axes[0].set_title("Biomass (100m)")
axes[0].axis("off")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Disturbance
if len(unique_vals) > 1:
    disturb_norm = Normalize(vmin=np.percentile(disturb_masked, 5), vmax=np.percentile(disturb_masked, 95))
    im2 = axes[1].imshow(disturb_masked, cmap="coolwarm", norm=disturb_norm)
    axes[1].set_title("Disturbance Index (100m)")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
else:
    axes[1].text(0.5, 0.5, "No measurable disturbance\ndata in this region",
                 ha='center', va='center', fontsize=12)
    axes[1].set_title("Disturbance Index (100m)")
axes[1].axis("off")

plt.tight_layout()
plt.show()





from skyfield.api import load, Topos, Star
from skyfield.data import hipparcos

# Load the timescale and planetary ephemeris
ts = load.timescale()
eph = load('de406.bsp')

# Load the Hipparcos star catalog
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)

# Define observer location and time
latitude = 1.155203
longitude = -57.921038
location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
t = ts.utc(1000, 6, 21, 0, 0)  # Approximate solstice midnight, 1000 CE

# Define the observer
observer = eph['earth'] + location

# List of Hipparcos IDs for Crux stars: Acrux, Mimosa, Gacrux, Delta Crux
crux_hip_ids = [60718, 62434, 59747, 61084]

# Compute azimuths for each Crux star
crux_azimuths_deg = []
for hip_id in crux_hip_ids:
    star = Star.from_dataframe(df.loc[hip_id])
    astrometric = observer.at(t).observe(star).apparent()
    alt, az, distance = astrometric.altaz()
    crux_azimuths_deg.append(az.degrees)

# Output the sorted azimuths
crux_azimuths_deg.sort()
print("Crux Star Azimuths at Rise (degrees):", crux_azimuths_deg)




# âœ… Step 3: Azimuthal Limb Analysis â€” Polygon Arm Directions vs Celestial Azimuths

# === Part 1: Polygon Azimuths from Detected Cruciform Geometry ===
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from math import atan2, degrees
from skyfield.api import load, Star, Topos
from skyfield.almanac import find_discrete, risings_and_settings

# Load polygon
gdf = gpd.read_file("gedi_detected_polygons.geojson")
polygon = gdf.geometry.iloc[0]
coords = list(polygon.exterior.coords)

# Compute azimuths between consecutive vertices
azimuths = []
for i in range(len(coords) - 1):
    dx = coords[i+1][0] - coords[i][0]
    dy = coords[i+1][1] - coords[i][1]
    angle = atan2(dx, dy)
    azimuth = (degrees(angle) + 360) % 360
    azimuths.append(azimuth)

# === Part 2: Simulate Crux Azimuths from Year 1500 CE ===
latitude = 1.155203
longitude = -57.921038
location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
eph = load('de406.bsp')
ts = load.timescale()
start_time = ts.utc(1500, 6, 21)
end_time = ts.utc(1500, 6, 22)
observer = eph['earth'] + location

# Define Crux stars using SIMBAD coordinates
acrux = Star(ra_hours=(12, 26, 35.9), dec_degrees=(-63, 5, 56))
hadar = Star(ra_hours=(14, 3, 49.4), dec_degrees=(-60, 22, 22))

# Compute Crux azimuths
azimuths_deg = {}

def get_azimuths(star, name):
    f = risings_and_settings(eph, star, location)
    t, y = find_discrete(start_time, end_time, f)
    for ti, yi in zip(t, y):
        label = 'Rise' if yi == 1 else 'Set'
        astrometric = observer.at(ti).observe(star).apparent()
        alt, az, _ = astrometric.altaz()
        azimuths_deg[f"{name} {label}"] = az.degrees

get_azimuths(acrux, "Acrux")
get_azimuths(hadar, "Hadar")

# === Plot: Polar Histogram of Polygon Azimuths and Crux Overlays ===
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Plot polygon arm azimuths
radians = [np.deg2rad(a) for a in azimuths]
ax.hist(radians, bins=36, color="gold", alpha=0.7)

# Overlay Crux azimuths
for a in azimuths_deg.values():
    ax.plot([np.deg2rad(a)] * 2, [0, 1], color="red", lw=2, linestyle="--")

plt.title("Polygon Arm Azimuths vs Crux Star Azimuths (Year 1500 CE)")
plt.show()



import folium
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from PIL import Image
import json
from shapely.geometry import shape
from geopy.distance import geodesic

# === Parameters ===
candidate_lat = 1.155203
candidate_lon = -57.921038
delta = 0.025
lat_min, lat_max = candidate_lat - delta, candidate_lat + delta
lon_min, lon_max = candidate_lon - delta, candidate_lon + delta

# === Function to process raster ===
def create_overlay(raster_path, output_image, cmap, name):
    with rasterio.open(raster_path) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        data = src.read(1, window=window)
        data = np.where(data == src.nodata, np.nan, data)
    
    # Normalize and apply colormap
    vmin, vmax = np.nanpercentile(data, [5, 95])
    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(output_image)

# === Create Overlays ===
create_overlay(
    "/kaggle/input/disturbance-amazon/disturbance_amazon_25m.tif",
    "disturbance_overlay.png",
    plt.cm.hot,
    "Disturbance"
)

create_overlay(
    "/kaggle/input/biomass-amazon/biomass_amazon_25m.tif",
    "biomass_overlay.png",
    plt.cm.YlGn,
    "Biomass"
)

# === Load polygon ===
gdf = gpd.read_file("gedi_detected_polygons.geojson")

# === Create Map ===
m = folium.Map(location=[candidate_lat, candidate_lon], zoom_start=14, tiles="CartoDB positron")

# Load polygon and compute centroid
with open("gedi_detected_polygons.geojson") as f:
    poly_geom = shape(json.load(f)["features"][0]["geometry"])
centroid = poly_geom.centroid
centroid_latlon = (centroid.y, centroid.x)

# GEDI footprint center
gedi_latlon = (1.155203, -57.921038)

# Compute distance
offset_km = geodesic(gedi_latlon, centroid_latlon).meters / 1000

folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr='&copy; <a href="https://carto.com/">CARTO</a>',
    name='CartoDB Dark Matter',
    control=True
).add_to(m)

# Add a MousePosition plugin to show coordinates as you hover
MousePosition(
    position='bottomright',
    separator=' | ',
    prefix='Coordinates:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

# Add overlays
folium.raster_layers.ImageOverlay(
    image="disturbance_overlay.png",
    name="Disturbance (25m)",
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.5
).add_to(m)

folium.raster_layers.ImageOverlay(
    image="biomass_overlay.png",
    name="Biomass (25m)",
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.5
).add_to(m)

# Add centroid marker
folium.CircleMarker(
    location=centroid_latlon,
    radius=4,
    color='blue',
    fill=True,
    fill_opacity=1,
    popup=f"ğŸ”µ Polygon Centroid {centroid_latlon} (~{offset_km:.2f} km from GEDI)"
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    attr='Map: Â© National Geographic Society, Esri, DeLorme, HERE, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, increment P Corp.',
    name="Esri NatGeo",
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Imagery Â© Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
    name='Esri Satellite'
).add_to(m)


# Add GEDI marker
folium.CircleMarker(
    location=gedi_latlon,
    radius=4,
    color='red',
    fill=True,
    fill_opacity=1,
    popup=f"ğŸ”´ Cruciform Candidate Site! at ({gedi_latlon})"
).add_to(m)

# Connect with line
folium.PolyLine(
    locations=[gedi_latlon, centroid_latlon],
    color="orange",
    weight=2.5,
    dash_array="5,5",
    tooltip=f"Offset â‰ˆ {offset_km:.2f} km"
).add_to(m)


# Add detected polygons
folium.GeoJson(gdf, name="Detected Polygon").add_to(m)

# Add bounding box
folium.Rectangle([[lat_min, lon_min], [lat_max, lon_max]], color="blue", fill=False).add_to(m)

# Add controls
folium.LayerControl().add_to(m)

# Display and save
m.save("map_with_toggle_layers.html")
m



from skyfield.api import load, Star, Topos
from skyfield.almanac import find_discrete, risings_and_settings
import numpy as np

# âœ… Load long-range ephemeris
eph = load('de406.bsp')
ts = load.timescale()

# âœ… Star coordinates from SIMBAD
acrux = Star(ra_hours=(12, 26, 35.9), dec_degrees=(-63, 5, 56))
hadar = Star(ra_hours=(14, 3, 49.4), dec_degrees=(-60, 22, 22))

# âœ… Observation location
location = Topos(latitude_degrees=1.155, longitude_degrees=-57.921)

# âœ… Time range for August 1, 1500
start_time = ts.utc(1500, 8, 1)
end_time = ts.utc(1500, 8, 2)

# âœ… Function for rise/set/transit
def analyze_star_events(star, name):
    f = risings_and_settings(eph, star, location)
    t, y = find_discrete(start_time, end_time, f)

    labels = {1: "Rise", 0: "Set"}
    print(f"\n--- {name} ---")
    for ti, yi in zip(t, y):
        print(f"{labels[yi]}: {ti.utc_datetime()} UTC")

    # Transit estimation
    times = ts.utc(1500, 8, 1, range(24))
    astrometric = (eph["earth"] + location).at(times).observe(star).apparent()
    altitudes = astrometric.altaz()[0].degrees
    max_i = np.argmax(altitudes)
    print(f"Transit: {times[max_i].utc_datetime()} UTC â€” Max altitude: {altitudes[max_i]:.2f}Â°")

# âœ… Run for both stars
analyze_star_events(acrux, "Acrux (Alpha Crucis)")
analyze_star_events(hadar, "Hadar (Beta Centauri)")


# Combined Script: Calculate Rise/Set Azimuths and Plot for Acrux & Hadar

# ğŸ“¦ Import necessary libraries
from skyfield.api import load, Star, Topos
from skyfield.almanac import find_discrete, risings_and_settings
import numpy as np
import matplotlib.pyplot as plt

# ğŸ§­  Load astronomical ephemeris and time scale
eph = load('de406.bsp')  # Extended DE406 planetary ephemeris
ts = load.timescale()

# ğŸŒŒ  Define stars of interest (coordinates from SIMBAD)
acrux = Star(ra_hours=(12, 26, 35.9), dec_degrees=(-63, 5, 56))
hadar = Star(ra_hours=(14, 3, 49.4), dec_degrees=(-60, 22, 22))

# ğŸ“�  Define observation site (GEDI Footprint 2 center)
location = Topos(latitude_degrees=1.155203, longitude_degrees=-57.921038)
observer = eph['earth'] + location

# ğŸ•’  Define observation window (24-hour period)
start_time = ts.utc(1500, 8, 1)
end_time = ts.utc(1500, 8, 2)

# ğŸ”­  Extract rise/set azimuths from Skyfield
azimuths_deg = {}

def get_azimuths(star, name):
    # Compute rise/set events
    f = risings_and_settings(eph, star, location)
    t, y = find_discrete(start_time, end_time, f)

    # Loop over events and calculate azimuth angles
    for ti, yi in zip(t, y):
        label = 'Rise' if yi == 1 else 'Set'
        astrometric = observer.at(ti).observe(star).apparent()
        alt, az, _ = astrometric.altaz()
        azimuths_deg[f"{name} {label}"] = az.degrees
        print(f"{name} {label} â€” {ti.utc_iso()} UTC â€” Azimuth: {az.degrees:.2f}Â°")

# Run for both stars
get_azimuths(acrux, "Acrux")
get_azimuths(hadar, "Hadar")

# ğŸ“Š  Create polar plot of azimuth directions
az_rads = {label: np.deg2rad(az) for label, az in azimuths_deg.items()}

# Set up polar plot
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.set_theta_direction(-1)  # Clockwise rotation
ax.set_theta_zero_location('N')  # 0Â° = North

# Plot radial lines for each azimuth
for label, az in az_rads.items():
    ax.plot([az, az], [0, 1], label=label)

# Style the plot
ax.set_title('Southern Crux Star Azimuths (from 1.155Â° N, 57.921Â° W)')
ax.set_rmax(1.2)
ax.set_rticks([])
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Save and display
plt.tight_layout()
plt.savefig('celestial_azimuth_alignment.png', dpi=300)
plt.show()



# Combined Script: Calculate Rise/Set Azimuths and Overlay on Terrain with Folium Layers

from skyfield.api import load, Star, Topos
from skyfield.almanac import find_discrete, risings_and_settings
from geopy.distance import geodesic
import folium
from folium.plugins import MousePosition
import geopandas as gpd
from shapely.ops import unary_union
import math

# --- Load ephemeris and timescale ---
eph = load('de406.bsp')
ts = load.timescale()

# --- Star coordinates from SIMBAD ---
acrux = Star(ra_hours=(12, 26, 35.9), dec_degrees=(-63, 5, 56))
hadar = Star(ra_hours=(14, 3, 49.4), dec_degrees=(-60, 22, 22))

# --- Candidate Site Coordinates ---
lat_center = 1.155203
lon_center = -57.921038
location = Topos(latitude_degrees=lat_center, longitude_degrees=lon_center)
observer = eph['earth'] + location

# --- Time window for 1500 AD ---
start_time = ts.utc(1500, 8, 1)
end_time = ts.utc(1500, 8, 2)

# --- Extract azimuths from Skyfield ---
azimuths_deg = {}

def get_azimuths(star, name):
    f = risings_and_settings(eph, star, location)
    t, y = find_discrete(start_time, end_time, f)
    for ti, yi in zip(t, y):
        label = 'Rise' if yi == 1 else 'Set'
        astrometric = observer.at(ti).observe(star).apparent()
        alt, az, _ = astrometric.altaz()
        azimuths_deg[f"{name} {label}"] = az.degrees
        print(f"{name} {label} â€” {ti.utc_iso()} UTC â€” Azimuth: {az.degrees:.2f}Â°")

get_azimuths(acrux, "Acrux")
get_azimuths(hadar, "Hadar")

# --- Initialize Folium Map with Base Layers ---
m = folium.Map(location=[-3, -56], zoom_start=18, control_scale=True, tiles=None)

# --- Add tile layers ---
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="Â© OpenStreetMap contributors",
    name="OpenStreetMap"
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles Â© Esri â€” Source: USGS, NOAA",
    name="Esri Hillshade",
    overlay=True,
    control=True,
    show=False
).add_to(m)

folium.TileLayer(
    tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='Map data: Â© OpenStreetMap contributors, SRTM | Map style: Â© OpenTopoMap (CC-BY-SA)',
    name='OpenTopoMap'
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Imagery Â© Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
    name='Esri Satellite'
).add_to(m)

# --- Add GEDI footprint polygons from GeoJSON ---
gdf = gpd.read_file("footprints.geojson")
colors = ['blue', 'green', 'red', 'orange', 'purple']
all_bounds = []
for i, row in gdf.iterrows():
    coords = list(row.geometry.exterior.coords)
    latlon_coords = [[lat, lon] for lon, lat in coords]

    folium.Polygon(
        locations=latlon_coords,
        color=colors[i % len(colors)],
        fill=False,
        weight=2,
        opacity=0.8,
        tooltip=f"GEDI Footprint {i+1}"
    ).add_to(m)

    all_bounds.append(row.geometry)

# --- Add Cruciform Site Marker ---
folium.CircleMarker(
   location=[lat_center, lon_center],
   radius=1,
   color='red',
   fill=True,
   fill_color='red',
   fill_opacity=1.0,
   tooltip="ğŸ“� Candidate (Cruciform Pattern) â€” GEDI Footprint 2 (1.155203, -57.921038)",
   popup="ğŸŒŸ Cruciform Candidate Site!"
).add_to(m)

# --- Function to compute destination from azimuth ---
def destination_point(lat, lon, azimuth_deg, distance_km):
    az_rad = math.radians(azimuth_deg)
    origin = (lat, lon)
    destination = geodesic(kilometers=distance_km).destination(origin, az_rad * 180 / math.pi)
    return destination.latitude, destination.longitude

# --- Add Celestial Azimuth Lines ---
for label, az in azimuths_deg.items():
    lat_end, lon_end = destination_point(lat_center, lon_center, az, distance_km=2)
    folium.PolyLine(
        [(lat_center, lon_center), (lat_end, lon_end)],
        tooltip=label,
        color='blue',
        weight=2
    ).add_to(m)

# --- Add Mouse Position Plugin ---
MousePosition(
    position='bottomright',
    separator=' | ',
    prefix='Coordinates:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    lng_formatter="function(num) {return L.Util.formatNum(num, 6);}"
).add_to(m)

# --- Fit map to all footprints ---
combined = unary_union(all_bounds)
minx, miny, maxx, maxy = combined.bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

# --- Add Layer Control ---
folium.LayerControl().add_to(m)

# --- Display map ---
m



from skyfield.api import load, Star, Topos
from skyfield.almanac import find_discrete, risings_and_settings
from pyproj import Geod
import numpy as np

# Load ephemeris and timescale
eph = load('de406.bsp')
ts = load.timescale()

# Define star positions
acrux = Star(ra_hours=(12, 26, 35.9), dec_degrees=(-63, 5, 56))
hadar = Star(ra_hours=(14, 3, 49.4), dec_degrees=(-60, 22, 22))

# Observation site
lat0, lon0 = 1.155203, -57.921038
location = Topos(latitude_degrees=lat0, longitude_degrees=lon0)
observer = eph['earth'] + location

# Define time window
start_time = ts.utc(1500, 8, 1)
end_time = ts.utc(1500, 8, 2)

# Function to extract rise and set azimuths
def get_azimuths(star, name):
    f = risings_and_settings(eph, star, location)
    t, y = find_discrete(start_time, end_time, f)
    azimuths = {}
    for ti, yi in zip(t, y):
        event = "Rise" if yi == 1 else "Set"
        astrometric = observer.at(ti).observe(star).apparent()
        alt, az, _ = astrometric.altaz()
        azimuths[event] = az.degrees
    return azimuths

# Get azimuths
az_acrux = get_azimuths(acrux, "Acrux")
az_hadar = get_azimuths(hadar, "Hadar")

# Initialize geodetic calculator
geod = Geod(ellps="WGS84")

# Define direction vectors in ENU plane
def enu_vector(az_deg):
    az_rad = np.radians(az_deg)
    return np.array([np.sin(az_rad), np.cos(az_rad)])

v1 = enu_vector(az_acrux['Rise'])
v2 = enu_vector(az_hadar['Set'])

# Solve for intersection of lines
A = np.column_stack((v1, -v2))
b = np.zeros(2)
t = np.linalg.lstsq(A, b, rcond=None)[0]

# Approximate offset in meters from candidate center
offset = np.linalg.norm(t[0] * v1[::-1]) * 111139  # degrees to meters

print(f"Acrux Rise Azimuth: {az_acrux['Rise']:.2f}Â°")
print(f"Hadar Set Azimuth: {az_hadar['Set']:.2f}Â°")
print(f"Offset from site center to intersection point: {round(offset, 2)} meters")


