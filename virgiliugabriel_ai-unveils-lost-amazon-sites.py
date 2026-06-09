diary_text = """
July 7, 1910 – We have left the main river and entered a tributary heading east. 
After two days' journey, the guide showed us ruins of an 'ancient village'. 
We saw vestiges of earthen walls in a square layout and a broad ditch encircling the site. 
It lies about 5 leagues north of the fork where we turned, near a large Brazil-nut grove.
"""

import re

# Simple extraction: find sentences mentioning keywords of interest
keywords = ["ruins", "ancient village", "walls", "ditch", "leagues", "mile"]
clue_sentences = [sent.strip() for sent in diary_text.split('\n') 
                  if any(k in sent for k in keywords)]

print("Extracted clue sentences:")
for sent in clue_sentences:
    print("-", sent)



# Pseudo-code for using OpenAI API (not executed here)
# import openai
# openai.api_key = "YOUR_OPENAI_API_KEY"
# 
# prompt = "Extract any locations or structures mentioned:\n" + diary_text + "\n\nList of clues:"
# response = openai.ChatCompletion.create(
#     model="gpt-4", 
#     messages=[{"role": "user", "content": prompt}],
#     max_tokens=200,
#     temperature=0
# )
# print(response.choices[0].message.content)



import geopandas as gpd
import fiona

# Enable KML driver via Fiona
fiona.supported_drivers['KML'] = 'rw'

gdf_geoglyphs = gpd.read_file(
    "/kaggle/input/archaeoblog-amazon-geoglyphs/amazon_geoglyphs.kml",
    driver='KML'
)
print(f"Total geoglyph sites loaded: {len(gdf_geoglyphs)}")
print(gdf_geoglyphs.head(3))



# Quick summary of geoglyph locations
minx, miny, maxx, maxy = gdf_geoglyphs.total_bounds
print(f"Geoglyphs extent: Latitude from {miny:.2f} to {maxy:.2f}, Longitude from {minx:.2f} to {maxx:.2f}")



import numpy as np

# Simulated environmental raster (100x100 grid for example)
np.random.seed(42)
shape = (100, 100)
# Simulate annual precipitation (mm) across region
annual_precip = np.random.uniform(1000, 3000, size=shape)
# Simulate elevation (m) across region
elevation = np.random.uniform(50, 400, size=shape)
# Simulate soil type map (categorical: 0=Other, 1=Ferralsol, 2=Plinthosol)
soil_map = np.random.choice([0, 1, 2], size=shape, p=[0.8, 0.1, 0.1])

# Define desired ranges/values based on known site conditions
precip_range = (1500, 2500)     # mm/year
elevation_range = (100, 300)    # meters
desired_soils = {1, 2}          # target soil categories (1=Ferralsol, 2=Plinthosol)

# Apply masks for each criterion
precip_mask = (annual_precip >= precip_range[0]) & (annual_precip <= precip_range[1])
elev_mask   = (elevation >= elevation_range[0]) & (elevation <= elevation_range[1])
soil_mask   = np.isin(soil_map, list(desired_soils))

# Combined mask: areas meeting all criteria
candidate_area_mask = precip_mask & elev_mask & soil_mask

print(f"Candidate area pixels: {np.sum(candidate_area_mask)} / {candidate_area_mask.size}")



import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.imshow(candidate_area_mask, cmap="Greens")
plt.title("Candidate Area Mask (Environmental Filtering)")
plt.axis("off")
plt.show()



from shapely.geometry import Point

# Define the candidate site point from the diary clue
site_alpha = Point(-68.5678, -11.1234)  # (Longitude, Latitude)
# Compute distance from Site Alpha to each known geoglyph (in meters, using a projection)
gdf_geo_merc = gdf_geoglyphs.to_crs(epsg=3857)        # project to Web Mercator (meters)
site_alpha_merc = gpd.GeoSeries([site_alpha], crs="EPSG:4326").to_crs(epsg=3857)
distances = gdf_geo_merc.distance(site_alpha_merc[0])
min_dist = distances.min()
nearest_idx = distances.idxmin()
nearest_site = gdf_geoglyphs.iloc[nearest_idx]

print(f"Nearest known geoglyph to Site Alpha is {min_dist/1000:.1f} km away.")
print("Nearest known site details:", nearest_site.to_dict())



import matplotlib.pyplot as plt

# Project to metric CRS for geometric calculations (Web Mercator)
gdf_projected = gdf_geoglyphs.to_crs(epsg=3857)

# Compute centroids in meters (so they're correct!)
centroids_projected = gdf_projected.geometry.centroid

# Convert centroids back to geographic CRS (lat/lon) for plotting
centroids_latlon = gpd.GeoSeries(centroids_projected, crs="EPSG:3857").to_crs(epsg=4326)

# Extract X/Y for plotting
x_vals = centroids_latlon.x
y_vals = centroids_latlon.y

plt.figure(figsize=(6, 6))
plt.scatter(x_vals, y_vals, s=5, color='brown', label='Known Geoglyphs')
plt.scatter(site_alpha.x, site_alpha.y, marker='*', color='red', s=100, label='Site Alpha (candidate)')
plt.title("Distribution of Known Geoglyph Sites (brown) and the New Candidate Site (red)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()



# Create a synthetic NDVI image (100x100 pixels)
forest_ndvi = 0.8  # typical NDVI for dense green vegetation
clearing_ndvi = 0.2  # typical NDVI for bare soil or sparse regrowth
ndvi = np.full((100, 100), forest_ndvi)
# Introduce a synthetic "geometric clearing": a 20x20 square of lower NDVI
ndvi[40:60, 40:60] = clearing_ndvi

# Detect anomalies: define a threshold to pick out low-NDVI regions
threshold = 0.4
anomaly_mask = ndvi < threshold

# Label connected anomaly regions
from scipy.ndimage import label
label_im, num_regions = label(anomaly_mask)
print("Number of low-NDVI anomaly regions detected:", num_regions)



plt.figure(figsize=(4,4))
plt.imshow(ndvi, cmap='RdYlGn')  # red = low NDVI, green = high NDVI
plt.colorbar(label="NDVI")
plt.title("Simulated NDVI Image with Anomalous                  Clearing")
plt.axis("off")
plt.show()



# Create a synthetic DTM (200x200 grid) with a mostly flat terrain plus noise
terrain = np.random.normal(loc=0, scale=0.2, size=(200, 200))

# Insert a square earthwork: a raised platform with a ditch around it
x0, x1 = 80, 120  # square from index 80 to 119 (40m across)
y0, y1 = 80, 120
terrain[x0:x1, y0:y1] += 4.0   # raise the platform by 4 m
ditch_width = 5
# Carve a ditch (2 m deep) just outside the platform on all four sides
terrain[x0-ditch_width:x0, y0-ditch_width:y1+ditch_width] -= 2.0  # top edge ditch
terrain[x1:x1+ditch_width, y0-ditch_width:y1+ditch_width] -= 2.0  # bottom edge ditch
terrain[x0:x1, y0-ditch_width:y0] -= 2.0                          # left edge ditch
terrain[x0:x1, y1:y1+ditch_width] -= 2.0                          # right edge ditch

# Smooth the terrain to simulate natural erosion
from scipy.ndimage import gaussian_filter
terrain = gaussian_filter(terrain, sigma=1)

# Detection: find significant high or low areas relative to background
mean_elev = terrain.mean()
high_mask = terrain > (mean_elev + 2.0)   # higher than mean by 2 m (potential mounds)
low_mask  = terrain < (mean_elev - 1.0)   # lower than mean by 1 m (potential ditches)
high_labels, n_high = label(high_mask)
low_labels,  n_low  = label(low_mask)
print(f"High regions detected: {n_high}, Low regions detected: {n_low}")

# Calculate centroids of detected regions
centroids = []
for label_id in range(1, n_high+1):
    coords = np.argwhere(high_labels == label_id)
    if coords.size:
        cy, cx = coords.mean(axis=0)  # note: coords are (row=y, col=x)
        centroids.append((cx, cy, 'high'))
for label_id in range(1, n_low+1):
    coords = np.argwhere(low_labels == label_id)
    if coords.size:
        cy, cx = coords.mean(axis=0)
        centroids.append((cx, cy, 'low'))

print("Detected feature centroids (x, y, type):", centroids)



plt.figure(figsize=(5,5))
plt.imshow(terrain, cmap='terrain')
plt.colorbar(label="Elevation (m)")
# Mark the detected center of the feature
cx, cy, _ = centroids[0]
plt.scatter([cx], [cy], marker='X', color='red', s=100, label="Detected feature")
plt.title("Synthetic DTM with detected earthwork                        feature")
plt.legend()
plt.axis("off")
plt.show()


