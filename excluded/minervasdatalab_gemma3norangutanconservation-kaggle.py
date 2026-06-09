!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git





import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
prompt = "Why are there so many Geese on Kaggle?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(result)


#from IPython.display import Image
#IMAGE_URL="https://storage.googleapis.com/kaggle-media/competitions/question_goose.png"
#Image(url=IMAGE_URL,height=250,width=250)


#from transformers import AutoProcessor, AutoModelForImageTextToText

#processor = AutoProcessor.from_pretrained(GEMMA_PATH)
#model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

#messages = [
#    {
#        "role": "user",
#        "content": [
#            {"type": "image", "image": IMAGE_URL},
#            {"type": "text", "text": "Describe this image in detail."}
#        ]
#    }
#]

#inputs = processor.apply_chat_template(
#    messages,
#    add_generation_prompt=True,
#    tokenize=True,
#    return_dict=True,
#    return_tensors="pt"
#).to(model.device, dtype=model.dtype)
#input_len = inputs["input_ids"].shape[-1]

#outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
#text = processor.batch_decode(
#    outputs[:, input_len:],
#    skip_special_tokens=True,
#    clean_up_tokenization_spaces=True
#)


#print(text[0])


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











import geopandas as gpd




# Read the GeoJSON file
gdf = gpd.read_file("/kaggle/input/indonesiabounds/indonesia-province-simple.json")

# Display first few rows
print(gdf.head())

# Plot the provinces
gdf.plot(edgecolor='black', figsize=(10, 10))



# Filter Kalimantan provinces (case-insensitive contains check)
kalimantan_gdf = gdf[gdf["Propinsi"].str.contains("KALIMANTAN", case=False, na=False)]

# Display filtered provinces
print(kalimantan_gdf["Propinsi"])

# Plot
kalimantan_gdf.plot(edgecolor='black', figsize=(8, 8))


print(kalimantan_gdf.crs)



# Read the GeoJSON file
kalimantan_adm4 = gpd.read_file("/kaggle/input/kalimantanadm4/Kalimantan.shp")


kalimantan_adm4.plot(edgecolor='black', figsize=(8, 8))


 kalimantan_adm4.head()





import geopandas as gpd
import matplotlib.pyplot as plt


# Load rivers
rivers = gpd.read_file("/kaggle/input/kalimantanrivers250k/River_line_250k_Clip.shp")

# Reproject rivers to match kalimantan_adm4 CRS if needed
if rivers.crs != kalimantan_adm4.crs:
    rivers = rivers.to_crs(kalimantan_adm4.crs)

# Plot overlay
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
rivers.plot(ax=ax, color='blue', linewidth=1)

ax.set_title("Kalimantan Administrative Boundaries with Rivers")
plt.axis('equal')
plt.show()





# Load rivers
rds = gpd.read_file("/kaggle/input/kalimantanrd/Roads_Clip.shp")

# Reproject rivers to match kalimantan_adm4 CRS if needed
if rds.crs != kalimantan_adm4.crs:
    rds = rds.to_crs(kalimantan_adm4.crs)

# Plot overlay
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
rds.plot(ax=ax, color='red', linewidth=1)

ax.set_title("Kalimantan Administrative Boundaries with Roads")
plt.axis('equal')
plt.show()





# Load rivers
ops = gpd.read_file("/kaggle/input/oilpalm-kalimantan/OPinKalimantan.shp")

# Reproject rivers to match kalimantan_adm4 CRS if needed
if ops.crs != kalimantan_adm4.crs:
    ops = ops.to_crs(kalimantan_adm4.crs)

# Plot overlay
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ops.plot(ax=ax, color='purple', linewidth=1)

ax.set_title("Kalimantan Administrative Boundaries with Oil Palms")
plt.axis('equal')
plt.show()








!pip install rasterio


import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from rasterio.transform import from_bounds

# Step 1: Set raster resolution and bounds
resolution = 0.01  # degrees/pixel (adjust as needed)
minx, miny, maxx, maxy = kalimantan_adm4.total_bounds
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Step 2: Rasterize river geometries (1 where river, 0 elsewhere)
rivers_mask = rasterize(
    [(geom, 1) for geom in rivers.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    all_touched=False,
    dtype=np.uint8
)

# Step 3: Compute Euclidean distance (in pixels)
river_distance = distance_transform_edt(1 - rivers_mask) * resolution

# Step 4: Plot
plt.figure(figsize=(10, 8))
plt.imshow(river_distance, cmap='viridis', extent=[minx, maxx, miny, maxy])
plt.title("Euclidean Distance to Nearest River")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Distance (degrees)")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

# 1. Project to meters (e.g., EPSG:3857 for global, or use UTM for local precision)
projected_crs = "EPSG:3857"
rivers_proj = rivers.to_crs(projected_crs)
bounds = rivers_proj.total_bounds

# 2. Define raster resolution (e.g., 100 m per pixel)
resolution = 100
width = int((bounds[2] - bounds[0]) / resolution)
height = int((bounds[3] - bounds[1]) / resolution)
transform = from_bounds(*bounds, width, height)

# 3. Rasterize river lines (1 = river, 0 = background)
river_mask = rasterize(
    ((geom, 1) for geom in rivers_proj.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# 4. Compute Euclidean distance (in pixels), then convert to meters
dist_pixels = distance_transform_edt(1 - river_mask)
dist_meters = dist_pixels * resolution

# 5. Mask very small distances (e.g., <30 meters)
dist_meters[dist_meters < 30] = np.nan

# 6. Plot
fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(dist_meters, cmap='viridis', origin='upper')
ax.set_title("Euclidean Distance to Nearest River (in meters)")
plt.colorbar(img, ax=ax, label="Distance (m)")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.tight_layout()
plt.show()






import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show


import numpy as np











# Load Protected Areas shapefile
pas = gpd.read_file("/kaggle/input/kalimantan-pas/kalimantanPAs.shp")

# Inspect the first few rows
print(pas.head())

# Check CRS
print("CRS:", pas.crs)

# Plot to verify
pas.plot(edgecolor='black', figsize=(8, 8))





# Reproject all layers to the same CRS
target_crs = kalimantan_adm4.crs
rivers = rivers.to_crs(target_crs)
pas = pas.to_crs(target_crs)

# Plot overlay
fig, ax = plt.subplots(figsize=(12, 12))
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, label="Admin Boundaries")
rivers.plot(ax=ax, color='blue', linewidth=0.8, label="Rivers")
pas.plot(ax=ax, facecolor='green', edgecolor='darkgreen', alpha=0.4, label="Protected Areas")

# Add title and legend
ax.set_title("Kalimantan: Administrative Boundaries, Rivers & Protected Areas", fontsize=14)
plt.legend()
plt.axis('equal')
plt.show()





# Reproject all layers to the same CRS
target_crs = kalimantan_adm4.crs
rivers = rivers.to_crs(target_crs)
rds = rds.to_crs(target_crs)
pas = pas.to_crs(target_crs)
ops = ops.to_crs(target_crs)

# Plot overlay
fig, ax = plt.subplots(figsize=(12, 12))
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, label="Admin Boundaries")
rivers.plot(ax=ax, color='blue', linewidth=0.8, label="Rivers")
pas.plot(ax=ax, facecolor='green', edgecolor='darkgreen', alpha=0.4, label="Protected Areas")
rds.plot(ax=ax, color='red', linewidth=0.3, label="Roads")
ops.plot(ax=ax, color='purple', linewidth=0.3, label="Roads")

# Add title and legend
ax.set_title("Kalimantan: Administrative Boundaries, Rivers, Roads & Protected Areas", fontsize=14)
plt.legend()
plt.axis('equal')
plt.show()








!pip install pygbif geopandas shapely






from pygbif import occurrences, species
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 1. Get bounding box from your Kalimantan shapefile
#
minx, miny, maxx, maxy = kalimantan_gdf.total_bounds
bbox_wkt = f"POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))"



# 2. Get the taxon key for orangutans
pongos = species.name_backbone(name="Pongo")
orangutan_key = pongos["usageKey"]

# 3. Fetch GBIF occurrences (limit=300 per page)
occ_list = []
limit = 300
offset = 0
max_records = 3000  # Adjust as needed

while offset < max_records:
    result = occurrences.search(
    taxonKey=orangutan_key,
    hasCoordinate=True,
    geometry=bbox_wkt,  # must be POLYGON WKT
    year="2016,2025",
    basisOfRecord="HUMAN_OBSERVATION",
    offset=offset,
    limit=limit
)

    for r in result["results"]:
        if r.get("occurrenceStatus", "PRESENT").upper() != "ABSENT":
            if r.get("decimalLatitude") and r.get("decimalLongitude"):
                occ_list.append({
                    "lon": r["decimalLongitude"],
                    "lat": r["decimalLatitude"],
                    "eventDate": r.get("eventDate"),
                    "scientificName": r.get("scientificName"),
                    "basisOfRecord": r.get("basisOfRecord")
                })

    offset += limit
    if len(result["results"]) < limit:
        break  # No more pages

# 4. Convert to GeoDataFrame and clip to Kalimantan
df = pd.DataFrame(occ_list)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf_clipped = gpd.clip(gdf, kalimantan_gdf)

# 5. Export to GeoJSON or Shapefile
#gdf_clipped.to_file("orangutan_presence_2016on.geojson", driver="GeoJSON")



# Extract lat/lon from geometry
# gdf_clipped["latitude"] = gdf_clipped.geometry.y
# gdf_clipped["longitude"] = gdf_clipped.geometry.x

# Select relevant columns
# df_coords = gdf_clipped[["eventDate", "scientificName", "basisOfRecord", "latitude", "longitude"]]

# Save to CSV in Kaggle working directory
# df_coords.to_csv("/kaggle/working/orangutan_gbif_2016_2025.csv", index=False)






import matplotlib.pyplot as plt

# Ensure both GeoDataFrames use the same CRS
kalimantan_gdf = kalimantan_gdf.to_crs("EPSG:4326")
gdf_clipped = gdf_clipped.to_crs("EPSG:4326")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
gdf_clipped.plot(ax=ax, markersize=10, color='red', alpha=0.7, label='Orangutan sightings (post-2015)')

# Decorations
ax.set_title("Orangutan Occurrences in Kalimantan (2016+)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.tight_layout()
plt.show()








# 2. Ensure CRS matches
gdf_clipped = gdf_clipped.to_crs(kalimantan_adm4.crs)

# 3. Plot
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_adm4.plot(ax=ax, color="lightgrey", edgecolor="black")
gdf_clipped.plot(ax=ax, color="red", markersize=10, alpha=0.7)

plt.title("Orangutan Occurrences on Kalimantan ADM4 Map")
plt.axis("off")
plt.tight_layout()
plt.show()








import matplotlib.pyplot as plt

# Extract coordinates
x = gdf_clipped.geometry.x
y = gdf_clipped.geometry.y

# Plot hexbin over Kalimantan base
fig, ax = plt.subplots(figsize=(10, 10))
kalimantan_gdf.plot(ax=ax, color='lightgrey', edgecolor='black')

hb = ax.hexbin(x, y, gridsize=30, cmap='Reds', bins='log', alpha=0.8)

# Add colorbar and labels
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Log-scaled observation count')
ax.set_title("Orangutan Observation Hotspots (Hexbin Density)")

plt.tight_layout()
plt.show()



# Assume hexbin was created using: gridsize=30
gridsize = 30

# Axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# Estimate hex width and height
width = (xmax - xmin) / gridsize
height = (ymax - ymin) / gridsize / np.sqrt(3/4)

# Radius for each hex (horizontal)
hex_radius = width / 2



from shapely.geometry import Polygon

def make_hex(xc, yc, radius):
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = xc + radius * np.cos(angles)
    y_hex = yc + radius * np.sin(angles)
    return Polygon(zip(x_hex, y_hex))

counts = hb.get_array()
verts = hb.get_offsets()

hexes = [make_hex(x, y, hex_radius) for x, y in verts]
hexbin_gdf = gpd.GeoDataFrame({'count': counts}, geometry=hexes, crs=gdf_clipped.crs)







from shapely.geometry import Point
import geopandas as gpd

# Assume your GeoDataFrame with hexbins is called `hexbin_gdf`
# and has 'count' as the observation count in each hex

summaries = []

for _, row in hexbin_gdf.iterrows():
    center = row.geometry.centroid
    count = int(row['count'])
    
    if count == 0:
        continue
    
    summary = (
        f"At approx lat {center.y:.2f}, lon {center.x:.2f}, there are "
        f"{count} recorded orangutan sightings."
    )
    summaries.append(summary)

# Combine summaries
density_context = "\n".join(summaries[:30])  # Keep top 30 for brevity



prompt = f"""
Based on orangutan survey data, here are key observations:
{density_context}

Can you summarize the high-density regions and suggest conservation priorities?
"""



inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
config = GenerationConfig(max_new_tokens=200, temperature=0.7)
outputs = model.generate(**inputs, generation_config=config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)






# 1. Spatial join: Assign each observation to a desa
gdf_clipped = gdf_clipped.to_crs(kalimantan_adm4.crs)
joined = gpd.sjoin(gdf_clipped, kalimantan_adm4, how='inner', predicate='within')

# 2. Count sightings per desa
desa_counts = (
    joined.groupby(['DESA', 'KECAMATAN', 'KABUPATEN', 'PROPINSI'])
    .size()
    .reset_index(name='sighting_count')
)

# 3. Merge counts back into ADM4 GeoDataFrame
kalimantan_adm4_counts = kalimantan_adm4.merge(desa_counts, on=['DESA', 'KECAMATAN', 'KABUPATEN', 'PROPINSI'], how='left')
kalimantan_adm4_counts['sighting_count'] = kalimantan_adm4_counts['sighting_count'].fillna(0).astype(int)



# Choropleth
fig, ax = plt.subplots(figsize=(12, 10))
kalimantan_adm4_counts.plot(column='sighting_count', ax=ax, cmap='Reds', edgecolor='black', legend=True)
plt.title("Orangutan Sightings per Desa")
plt.axis("off")
plt.tight_layout()
plt.show()






# 4. Create summaries for prompt
summaries = []

for _, row in kalimantan_adm4_counts[kalimantan_adm4_counts['sighting_count'] > 0].iterrows():
    center = row.geometry.centroid
    summary = (
        f"In Desa '{row['DESA']}', Kecamatan '{row['KECAMATAN']}', Kabupaten '{row['KABUPATEN']}', "
        f"there are {row['sighting_count']} recorded orangutan sightings "
        f"(approx lat {center.y:.2f}, lon {center.x:.2f})."
    )
    summaries.append(summary)

# Format final prompt
desa_context = "\n".join(summaries[:30])
prompt = f"""
Based on orangutan survey data, here are the sightings at the village level:
{desa_context}

Can you identify the key conservation hotspots and suggest targeted protection strategies?
"""

# 5. Generate with Gemma
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
config = GenerationConfig(max_new_tokens=300, temperature=0.7)
outputs = model.generate(**inputs, generation_config=config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)






# 6. Show top 5 desa with highest sightings
top5 = (
    kalimantan_adm4_counts[kalimantan_adm4_counts['sighting_count'] > 0]
    .sort_values(by='sighting_count', ascending=False)
    .head(5)
)

print("\nTop 5 Desa by Orangutan Sightings:\n")
for i, row in top5.iterrows():
    print(
        f"{row['DESA']} in {row['KECAMATAN']}, {row['KABUPATEN']}: "
        f"{row['sighting_count']} sightings"
    )












import pandas as pd

# Read the capture-release centroid CSV
file_path = "/kaggle/input/orangutan-capture-relese/capture_release_centroids.csv"
df_release = pd.read_csv(file_path)


df_release.head()





import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# --- 1. Create GeoDataFrames for capture & release points ---
gdf_capture = gpd.GeoDataFrame(
    df_release,
    geometry=gpd.points_from_xy(df_release["capture_centroid_x"], df_release["capture_centroid_y"]),
    crs="EPSG:4326"
)

gdf_release = gpd.GeoDataFrame(
    df_release,
    geometry=gpd.points_from_xy(df_release["release_centroid_x"], df_release["release_centroid_y"]),
    crs="EPSG:4326"
)

# --- 2. Plot on kalimantan_adm4 base map ---
fig, ax = plt.subplots(figsize=(12, 10))

kalimantan_adm4.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)

gdf_capture.plot(ax=ax, color="red", markersize=40, label="Capture Location")
gdf_release.plot(ax=ax, color="green", markersize=40, label="Release Location")

plt.title("Orangutan Capture and Release Sites")
plt.legend()
plt.axis("off")
plt.tight_layout()
plt.show()






capture_counts = df_release.groupby(['desa'])['capture_centroid_x'].count().reset_index(name='capture_count')
release_counts = df_release.groupby(['release_name'])['release_centroid_x'].count().reset_index(name='release_count')

summary_lines = []

for _, row in capture_counts.iterrows():
    summary_lines.append(f"- Desa {row['desa']} had {row['capture_count']} orangutan capture(s).")

for _, row in release_counts.iterrows():
    summary_lines.append(f"- Release site {row['release_name']} received {row['release_count']} orangutan(s).")

desa_release_summary = "\n".join(summary_lines)



prompt = f"""
You are analyzing orangutan conservation data in Kalimantan.

Below is a summary of recorded capture and release locations by desa:

{desa_release_summary}

Please identify which desa had the most orangutan captures and which release sites received the highest number of orangutans. Provide insights about possible hotspots of orangutan movement or conservation focus.
"""






prompt = f"""
Anda sedang menganalisis data konservasi orangutan di Kalimantan.

Berikut adalah ringkasan lokasi tangkapan dan pelepasan berdasarkan desa:

{desa_release_summary}

Tolong identifikasi desa dengan jumlah tangkapan orangutan terbanyak dan lokasi pelepasan yang menerima jumlah orangutan tertinggi. Berikan juga wawasan mengenai wilayah penting untuk konservasi atau pergerakan orangutan.
"""



from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



final_prompt = f"""
You are analyzing orangutan conservation data in Kalimantan.

Below is a summary of recorded capture and release locations by desa:

{desa_release_summary}

Please identify which desa had the most orangutan captures and which release sites received the highest number of orangutans. Provide insights about possible hotspots of orangutan movement or conservation focus.
"""


inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
config = GenerationConfig(max_new_tokens=300, temperature=0.7)
outputs = model.generate(**inputs, generation_config=config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)






# Group and count captures
capture_counts = (
    df_release
    .groupby(['desa', 'capture_centroid_x', 'capture_centroid_y'])
    .size()
    .reset_index(name='capture_count')
)

# Group and count releases
release_counts = (
    df_release
    .groupby(['release_name', 'release_centroid_x', 'release_centroid_y'])
    .size()
    .reset_index(name='release_count')
)

# Create summary lines
lines = []

for _, row in capture_counts.iterrows():
    lines.append(
        f"Capture – Desa '{row['desa']}' at lat {row['capture_centroid_y']:.4f}, lon {row['capture_centroid_x']:.4f}: {row['capture_count']} orangutans"
    )

for _, row in release_counts.iterrows():
    lines.append(
        f"Release – Site '{row['release_name']}' at lat {row['release_centroid_y']:.4f}, lon {row['release_centroid_x']:.4f}: {row['release_count']} orangutans"
    )

# Join lines into full summary
desa_release_summary = "\n".join(lines)

# Compose prompt
final_prompt = f"""
You are analyzing orangutan conservation data in Kalimantan.

Below is a summary of recorded capture and release locations by desa, including coordinates:

{desa_release_summary}

Please identify:
1. Which desa had the most orangutan captures.
2. Which release site received the most orangutans.
3. Any insights into likely translocation routes or conservation hotspots.
"""



# 1. Desa with most captures
top_capture = capture_counts.sort_values(by="capture_count", ascending=False).iloc[0]
top_capture_result = (
    f"Desa with most captures: '{top_capture['desa']}' "
    f"({top_capture['capture_count']} orangutans) at "
    f"lat {top_capture['capture_centroid_y']:.4f}, lon {top_capture['capture_centroid_x']:.4f}"
)

# 2. Release site with most orangutans
top_release = release_counts.sort_values(by="release_count", ascending=False).iloc[0]
top_release_result = (
    f"Release site with most orangutans: '{top_release['release_name']}' "
    f"({top_release['release_count']} orangutans) at "
    f"lat {top_release['release_centroid_y']:.4f}, lon {top_release['release_centroid_x']:.4f}"
)

# 3. Print results
print(top_capture_result)
print(top_release_result)






import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure all GeoDataFrames use the same CRS ---
target_crs = kalimantan_adm4.crs
rivers = rivers.to_crs(target_crs)
pas = pas.to_crs(target_crs)
gdf_clipped = gdf_clipped.to_crs(target_crs)

# Capture and Release GeoDataFrames
#df_release = pd.read_csv("/kaggle/input/orangutan-capture-relese/capture_release_centroids.csv")

gdf_capture = gpd.GeoDataFrame(
    df_release,
    geometry=gpd.points_from_xy(df_release["capture_centroid_x"], df_release["capture_centroid_y"]),
    crs="EPSG:4326"
).to_crs(target_crs)

gdf_release_points = gpd.GeoDataFrame(
    df_release,
    geometry=gpd.points_from_xy(df_release["release_centroid_x"], df_release["release_centroid_y"]),
    crs="EPSG:4326"
).to_crs(target_crs)

# --- Plot all layers ---
fig, ax = plt.subplots(figsize=(14, 14))

# Base layers
kalimantan_adm4.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, label="Admin Boundaries")
rivers.plot(ax=ax, color='blue', linewidth=0.8, label="Rivers")
pas.plot(ax=ax, facecolor='green', edgecolor='darkgreen', alpha=0.3, label="Protected Areas")
rds.plot(ax=ax, facecolor='pink', edgecolor='pink', alpha=0.3, label="Roads")

# Occurrence points
gdf_clipped.plot(ax=ax, color='orange', markersize=10, alpha=0.7, label="Orangutan Occurrences")

# Capture & Release points
gdf_capture.plot(ax=ax, color='red', markersize=50, marker='o', label="Capture Sites")
gdf_release_points.plot(ax=ax, color='purple', markersize=50, marker='^', label="Release Sites")

# Map settings
ax.set_title("Kalimantan: Occurrences, Capture & Release Sites with Rivers and Protected Areas", fontsize=14)
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()









##trimmed_summary = "\n".join(lines[:20])  # top 20 for brevity

#prompt = f"""
#Here is the spatial context:
#- Rivers: shown in blue lines
#- Protected Areas (PAs): shaded green
#- Orangutan Occurrences: {len(gdf_clipped)} points
#- Capture Sites: {len(gdf_capture)} locations
#- Release Sites: {len(gdf_release)} locations

#Below is a summary of capture and release data with coordinates:
#{trimmed_summary}

#Please provide:
#1. Which unprotected areas (outside PAs) with high orangutan sightings should be prioritized for anti-poaching interventions.
#2. Which protected areas currently have the most sightings or captures nearby and whether these PAs appear to be effective or need reinforcement.
#3. Recommendations for improving connectivity between capture and release sites via protected corridors.

#Respond in 3 sections:
#(1) Unprotected Priority Areas
#(2) Protected Area Performance
#(3) Connectivity Recommendations
#"""

#inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#config = GenerationConfig(max_new_tokens=800, do_sample=True, top_p=0.9)
#outputs = model.generate(**inputs, generation_config=config)
#result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(result)






trimmed_summary = "\n".join(lines[:20])  # top 20 for brevity

prompt = f"""
Here is the spatial context:
- Rivers: shown in blue lines
- Roads: shown in pink lines
- Protected Areas (PAs): shaded green
- Orangutan Occurrences: {len(gdf_clipped)} points
- Capture Sites: {len(gdf_capture)} locations
- Release Sites: {len(gdf_release)} locations

Below is a summary of capture and release data with coordinates:
{trimmed_summary}

Please provide:
1. Which unprotected areas (outside PAs) with high orangutan sightings should be prioritized for anti-poaching interventions.
2. Which protected areas currently have the most sightings or captures nearby and whether these PAs appear to be effective or need reinforcement.
3. Recommendations for improving connectivity between capture and release sites via protected corridors and avoiding roads.

Respond in 3 sections:
(1) Unprotected Priority Areas
(2) Protected Area Performance
(3) Connectivity Recommendations
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
config = GenerationConfig(max_new_tokens=800, do_sample=True, top_p=0.9)
outputs = model.generate(**inputs, generation_config=config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)








from rasterio.transform import from_bounds
import numpy as np

# --- 1. Project desa to match the distance raster CRS (projected_crs from your code) ---
desa_proj = kalimantan_adm4_counts.to_crs(projected_crs)

# Compute centroid for sampling
desa_proj['centroid'] = desa_proj.geometry.centroid

# --- 2. Define function to sample Euclidean distance raster ---
def sample_distance(geom, transform, dist_array):
    x, y = geom.x, geom.y
    col, row = ~transform * (x, y)  # Convert to pixel indices
    row, col = int(row), int(col)
    if 0 <= row < dist_array.shape[0] and 0 <= col < dist_array.shape[1]:
        return dist_array[row, col]
    return np.nan

# --- 3. Add river distance values ---
desa_proj['river_dist_m'] = desa_proj['centroid'].apply(lambda g: sample_distance(g, transform, dist_meters))

# --- 4. Reproject desa back to PA CRS for spatial join ---
desa_proj = desa_proj.to_crs(pas.crs)

# --- 5. Spatial join with PAs to add is_protected ---
desa_with_pa = gpd.sjoin(desa_proj, pas, how='left', predicate='intersects')
desa_with_pa['is_protected'] = ~desa_with_pa['index_right'].isna()

# --- 6. Final cleaned GeoDataFrame ---
desa_proj = desa_with_pa[['DESA', 'KECAMATAN', 'KABUPATEN', 'PROPINSI',
                          'sighting_count', 'river_dist_m', 'is_protected', 'geometry']]






unprotected = desa_proj[(desa_proj['sighting_count'] > 0) & (~desa_proj['is_protected'])]
top_unprotected = unprotected.sort_values(['sighting_count']).head(5)

hydro_context = "\n".join([
    f"- {row['DESA']}: {row['sighting_count']} sightings, {row['river_dist_m']:.0f} m from river"
    for _, row in top_unprotected.iterrows()
])






# Assuming your target projected CRS is stored in projected_crs (e.g., EPSG:3857)
gdf_clipped_proj = gdf_clipped.to_crs(projected_crs)
gdf_capture_proj = gdf_capture.to_crs(projected_crs)
gdf_release_proj = gdf_release.to_crs(projected_crs)




# --- 3. Trim capture & release summary for brevity ---
trimmed_summary = "\n".join(lines[:20])  # from your previous capture/release summary

prompt = f"""
### CONTEXT ###
- Rivers: blue lines
- Protected Areas: green polygons
- Orangutan Occurrences: {len(gdf_clipped_proj)} points
- Capture Sites: {len(gdf_capture_proj)} locations
- Release Sites: {len(gdf_release_proj)} locations

Hydrology context:
Top unprotected villages by sightings & river proximity:
{hydro_context}

Capture & Release Summary:
{trimmed_summary}

### TASK ###
You are a conservation analyst. Using the context above:

1. List unprotected villages with high orangutan sightings and close river access that should be prioritized for anti-poaching.
2. Evaluate protected areas (PAs): which ones show many sightings or captures nearby and do they need reinforcement?
3. Suggest river-based corridors to improve connectivity between capture and release sites.

### FORMAT ###
Respond in 3 sections only:
(1) Unprotected Priority Areas: [bulleted list]
(2) Protected Area Performance: [bulleted list]
(3) Connectivity Recommendations: [bulleted list]

### RESPONSE START ###
"""



inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
config = GenerationConfig(max_new_tokens=1200, do_sample=True, top_p=0.9)
outputs = model.generate(**inputs, generation_config=config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== GEMMA 3N RESPONSE =====\n")
print(result.split("### RESPONSE START ###")[-1].strip())












!pip install gradio



import matplotlib.pyplot as plt
import gradio as gr

# --- generate and save the map as image ---
def generate_map_image(gdf):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(column='sighting_count', ax=ax, cmap='Reds', edgecolor='black', legend=True)
    plt.title("Orangutan Sightings per Desa")
    plt.axis("off")
    plt.tight_layout()
    img_path = "desa_sightings_map.png"
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    return img_path

# --- summarize top desa sightings ---
def summarize_top_desa(gdf, top_n=5):
    top_desa = (
        gdf[gdf['sighting_count'] > 0]
        .sort_values(by='sighting_count', ascending=False)
        .head(top_n)
    )
    lines = [
        f"- {row['DESA']} ({row['sighting_count']} sightings)"
        for _, row in top_desa.iterrows()
    ]
    return "Top desa sightings:\n" + "\n".join(lines)

# --- your main Gradio prediction function ---
def predict(user_query):
    if "highest" in user_query.lower():
        summary = summarize_top_desa(kalimantan_adm4_counts)
        image_path = generate_map_image(kalimantan_adm4_counts)
        return summary, image_path
    else:
        return "Please ask a question about top sightings.", None









import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig





# Prebuilt desa summary (based on previous code)
desa_context = "\n".join(summaries[:30])  # top 30 desa summaries

# Inference function
def answer_question(user_query):
    prompt = f"""
Based on orangutan survey data, here are the sightings at the village level:
{desa_context}

User question: {user_query}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    config = GenerationConfig(max_new_tokens=300, temperature=0.7)
    outputs = model.generate(**inputs, generation_config=config)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Gradio app
gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Orangutan Survey Q&A",
    description="Ask a question about orangutan sightings in Kalimantan."
).launch()





iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Ask a question about orangutan sightings in Kalimantan."),
    outputs=[
        gr.Textbox(label="Summary Output"),
        gr.Image(label="Sightings Choropleth Map")
    ],
    title="Orangutan Survey Q&A"
)

iface.launch()









!pip install gradio langdetect






#from langdetect import detect

# Answer Q using multilingual prompt
#def answer_question(user_query):
#    lang = detect(user_query)
#    is_indonesian = lang == "id"

#    if is_indonesian:
#        prompt = f"""
#Berdasarkan data survei orangutan, berikut adalah pengamatan di tingkat desa:
#{desa_context}

#Pertanyaan pengguna: {user_query}
#"""
 #   else:
#        prompt = f"""
#Based on orangutan survey data, here are the sightings at the village level:
#{desa_context}

#User question: {user_query}
#"""

   # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   # config = GenerationConfig(max_new_tokens=300, temperature=0.7)
  #  outputs = model.generate(**inputs, generation_config=config)
 #   result = tokenizer.decode(outputs[0], skip_special_tokens=True)

 #   return result

# Combine response + map if asked
#def predict(user_query):
  #  lang = detect(user_query)
   # if "highest" in user_query.lower() or "terbanyak" in user_query.lower():
    #    summary = summarize_top_desa(kalimantan_adm4_counts)
   #     image_path = generate_map_image(kalimantan_adm4_counts)
    #    return summary, image_path
  #  else:
       # result = answer_question(user_query)
       # return result, None



#iface = gr.Interface(
 #   fn=predict,
#    inputs=gr.Textbox(label="Tanyakan tentang pengamatan orangutan di Kalimantan / Ask about orangutan sightings in Kalimantan."),
#    outputs=[
#        gr.Textbox(label="Ringkasan / Summary"),
#        gr.Image(label="Peta / Map (jika relevan)")
 #   ],
#    title="Orangutan Survey Q&A (EN/ID)",
#    description="Tanyakan dalam Bahasa Indonesia atau English. Ask in Indonesian or English."
#)

#iface.launch()









#def get_orangutan_count(village_name):
    # Normalize case for matching
   # village_name = village_name.strip().lower()
    
    # Search for matching desa
  #  match = kalimantan_adm4_counts[kalimantan_adm4_counts['DESA'].str.lower() == village_name]
    
   # if match.empty:
   #     return f"Desa '{village_name}' tidak ditemukan / not found."
    
   # count = int(match['sighting_count'].values[0])
   # kecamatan = match['KECAMATAN'].values[0]
  #  kabupaten = match['KABUPATEN'].values[0]
    
  #  return f"Desa '{village_name.title()}' di Kecamatan {kecamatan}, Kabupaten {kabupaten} memiliki {count} pengamatan orangutan."



#import gradio as gr

#iface_lookup = gr.Interface(
#    fn=get_orangutan_count,
#    inputs=gr.Textbox(label="Masukkan nama desa / Enter village name"),
#    outputs="text",
#    title="Cek Pengamatan Orangutan per Desa",
v    description="Masukkan nama desa untuk melihat jumlah pengamatan orangutan (Enter village name to see sightings count)"
#)

#iface_lookup.launch()









!pip install langdetect



!pip install gradio transformers






# Install required packages
!pip install kagglehub transformers accelerate langdetect gradio timm --upgrade

# Import libraries
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch



!pip install transformers accelerate sentencepiece gradio langdetect --upgrade







import gradio as gr
import matplotlib.pyplot as plt
from langdetect import detect

# =====================
# Precomputed Context (Assumes these exist in memory)
# =====================
# Variables: kalimantan_adm4_counts, gdf_clipped, gdf_capture, gdf_release
# Context: hydro_context, desa_context, desa_release_summary, trimmed_summary, top5

# Top 5 Desa summary text
top5_context = "\n".join([
    f"- {row['DESA']} in {row['KECAMATAN']}, {row['KABUPATEN']}: {row['sighting_count']} sightings"
    for _, row in top5.iterrows()
])

# Knowledge base cache for repeated queries
kb_cache = {}

# =====================
# 1. Map Generator
# =====================
def generate_map(region_name=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    kalimantan_adm4_counts.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    gdf_clipped.plot(ax=ax, color="red", markersize=5, alpha=0.6, label="Sightings")
    gdf_capture.plot(ax=ax, color="blue", markersize=20, marker="x", label="Capture")
    gdf_release.plot(ax=ax, color="green", markersize=20, marker="o", label="Release")

    if region_name:
        region = kalimantan_adm4_counts[kalimantan_adm4_counts["DESA"].str.lower() == region_name.lower()]
        if not region.empty:
            region.plot(ax=ax, color="yellow", alpha=0.5, edgecolor="black", label=f"Highlight: {region_name}")

    plt.legend()
    plt.title("Orangutan Sightings, Capture & Release")
    plt.axis("off")
    img_path = "map_output.png"
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    return img_path

# =====================
# 2. Generate Response
# =====================
def generate_response(user_query):
    if user_query in kb_cache:
        return kb_cache[user_query]

    lang = detect(user_query)
    is_indonesian = lang == "id"

    if is_indonesian:
        prompt = f"""
### KONTEKS ###
- Pengamatan orangutan: {len(gdf_clipped)} titik
- Situs Tangkap: {len(gdf_capture)} | Situs Lepas: {len(gdf_release)}
- Sungai (biru), Jalan (merah muda), Kawasan Lindung (hijau)

**5 Desa Teratas**:
{top5_context}

**Pengamatan tingkat desa**:
{desa_context}

**Tangkap & Lepas**:
{desa_release_summary}

**Konteks Hidrologi**:
{hydro_context}

Pertanyaan: {user_query}

Jawab dalam 3 bagian:
(1) Desa Prioritas
(2) Kinerja Kawasan Lindung
(3) Rekomendasi Konektivitas
"""
    else:
        prompt = f"""
### CONTEXT ###
- Orangutan Occurrences: {len(gdf_clipped)} points
- Capture Sites: {len(gdf_capture)} | Release Sites: {len(gdf_release)}
- Rivers (blue), Roads (pink), Protected Areas (green)

**Top 5 Villages**:
{top5_context}

**Village Sightings**:
{desa_context}

**Capture & Release Summary**:
{desa_release_summary}

**Hydrology**:
{hydro_context}

Question: {user_query}

Respond in 3 sections:
(1) Unprotected Priority Areas
(2) Protected Area Performance
(3) Connectivity Recommendations
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=700, do_sample=True, top_p=0.9)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    kb_cache[user_query] = result
    return result

# =====================
# 3. Prediction Function
# =====================
def predict(user_query):
    highlight = None
    for desa_name in kalimantan_adm4_counts["DESA"].unique():
        if desa_name.lower() in user_query.lower():
            highlight = desa_name
            break

    response_text = generate_response(user_query)
    map_path = generate_map(region_name=highlight)
    return response_text, map_path

# =====================
# 4. Gradio Interface
# =====================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Ask in English or Bahasa Indonesia"),
    outputs=[
        gr.Textbox(label="Analysis"),
        gr.Image(label="Map")
    ],
    title="Orangutan Conservation Explorer",
    description="Ask questions about orangutan sightings, captures, releases, rivers, and PAs. The system provides analysis and maps."
)

iface.launch()





