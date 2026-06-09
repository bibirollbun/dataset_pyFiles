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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("MySecret Key")


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


!pip install requests laspy matplotlib
!pip install numpy
!pip install rasterio


!pip install simplekml


import rasterio
import numpy as np
from PIL import Image
import openai
import base64
import os

# Load and normalize the B04 band
with rasterio.open('/kaggle/input/sentinel-2data/T22MCU_20250601T134721_B04_10m.jp2') as src:
    band = src.read(1)

band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
image = Image.fromarray(band_norm)
image_path = "sentinel_b04.png"
image.save(image_path)
# load and normalize the B08 band
with rasterio.open('/kaggle/input/sentinel-2data/T22MCU_20250601T134721_B08_10m.jp2') as src:
    band = src.read(1)

band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
image = Image.fromarray(band_norm)
image_path = "sentinel_b08.png"
image.save(image_path)




import rasterio
import numpy as np
from PIL import Image
import openai
import base64
import os


# Encode the image to base64
def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image_base64('/kaggle/working/sentinel_b08.png')

from openai import OpenAI
from kaggle_secrets import UserSecretsClient
secrets_client = UserSecretsClient()
api_key = secrets_client.get_secret("MySecret Key")

client = OpenAI(api_key=api_key)


response = client.chat.completions.create(
    model="gpt-4o-mini",  # Or "gpt-4o"
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the surface features in plain English.DO Zoom in and explain"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
            ],
        }
    ],
    max_tokens=300,
)

# Access results
print("Model:", response.model)
print("Scene ID:", "S2C_MSIL2A_20250601T134721_N0511_R024_T22MCU_20250601T190405.SAFE")
print("Response:\n", response.choices[0].message.content)




import h5py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
import numpy as np
import os
import json

# --- Step 1: Load GEDI HDF5 Files ---
base_folder = "/kaggle/input/gedi-terrabrassilis"
gedi_files = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.endswith(".h5")]

def extract_gedi_points(h5_file_path):
    data = []
    is_gedi01 = "GEDI01_B" in os.path.basename(h5_file_path)
    with h5py.File(h5_file_path, 'r') as f:
        for beam in f:
            if not beam.startswith("BEAM") or "geolocation" not in f[beam]:
                continue
            geo = f[beam]["geolocation"]
            try:
                lat = geo["latitude_bin0"][:] if is_gedi01 else geo["latitude_1gfit"][:]
                lon = geo["longitude_bin0"][:] if is_gedi01 else geo["longitude_1gfit"][:]
                elev = geo["elevation_bin0"][:] if is_gedi01 else geo["elevation_1gfit"][:]
                for i in range(len(lat)):
                    data.append({"latitude": lat[i], "longitude": lon[i], "elevation": elev[i]})
            except Exception:
                continue
    return pd.DataFrame(data)

gedi_df = pd.concat([extract_gedi_points(fp) for fp in gedi_files], ignore_index=True)
gedi_df.dropna(subset=["latitude", "longitude", "elevation"], inplace=True)

# --- Step 2: Convert to GeoDataFrame ---
gedi_gdf = gpd.GeoDataFrame(
    gedi_df,
    geometry=gpd.points_from_xy(gedi_df.longitude, gedi_df.latitude),
    crs="EPSG:4326"
)

# --- Step 3: Extract NDVI values ---
ndvi_path = "/kaggle/input/nvdi-dataset-for-xingu-region/2024-09-04-00_00_2024-09-04-23_59_Sentinel-2_L2A_NDVI.tiff"
ndvi_src = rasterio.open(ndvi_path)
ndvi_data = ndvi_src.read(1)
transform = ndvi_src.transform

# Detect scaling
raw_min, raw_max = np.nanmin(ndvi_data), np.nanmax(ndvi_data)
scale_factor = 1.0
if raw_max > 1000:
    scale_factor = 1 / 10000.0
elif raw_max > 1 and raw_max <= 255:
    scale_factor = 1 / 255.0

print(f"NDVI raw range: {raw_min} to {raw_max} â€” using scale factor: {scale_factor}")

# Correct for byte-scaled NDVI
def extract_ndvi(lat, lon):
    try:
        col, row = ~transform * (lon, lat)  # rasterio uses (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < ndvi_data.shape[0] and 0 <= col < ndvi_data.shape[1]:
            val = ndvi_data[row, col]
            if val != ndvi_src.nodata and val > 0:
                return val / 255.0
    except:
        return np.nan
    return np.nan



gedi_gdf["ndvi"] = gedi_gdf.apply(lambda row: extract_ndvi(row.latitude, row.longitude), axis=1)
gedi_gdf.dropna(subset=["ndvi"], inplace=True)

# --- Step 4: Calculate Anomaly Scores ---
gedi_gdf["z_elev"] = (gedi_gdf["elevation"] - gedi_gdf["elevation"].mean()) / gedi_gdf["elevation"].std()
gedi_gdf["z_ndvi"] = (gedi_gdf["ndvi"] - gedi_gdf["ndvi"].mean()) / gedi_gdf["ndvi"].std()
gedi_gdf["anomaly_score"] = np.abs(gedi_gdf["z_elev"] - gedi_gdf["z_ndvi"])

# --- Step 5: Output Top 5 Anomalies ---
top = gedi_gdf.sort_values("anomaly_score", ascending=False).head(5)
footprints = []
for i, row in top.iterrows():
    lat, lon = row.latitude, row.longitude
    bbox = box(lon - 0.0005, lat - 0.0005, lon + 0.0005, lat + 0.0005)
    footprints.append({
        "center_latlon": (lat, lon),
        "radius_m": 50,
        "bbox_wkt": bbox.wkt,
        "elevation": row.elevation,
        "ndvi": row.ndvi,
        "anomaly_score": row.anomaly_score
    })

# --- Step 6: Show Footprints ---
print(json.dumps(footprints, indent=2))

# Optional Debug
print("\nNDVI Value Range:", gedi_gdf["ndvi"].min(), "-", gedi_gdf["ndvi"].max())



import folium
from shapely import wkt
from shapely.geometry import Polygon
import geopandas as gpd

# === Input: Provided anomaly data ===
anomalies = [
  {
    "center_latlon": [
      -5.700886783080665,
      -52.39251980683522
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.392019806835215 -5.701386783080665, -52.392019806835215 -5.700386783080665, -52.39301980683522 -5.700386783080665, -52.39301980683522 -5.701386783080665, -52.392019806835215 -5.701386783080665))",
    "elevation": 455.59288993291557,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 13.338739909616542
  },
  {
    "center_latlon": [
      -5.727974589766515,
      -52.39032776552173
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.38982776552173 -5.728474589766515, -52.38982776552173 -5.727474589766516, -52.390827765521735 -5.727474589766516, -52.390827765521735 -5.728474589766515, -52.38982776552173 -5.728474589766515))",
    "elevation": 416.20562410820276,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 11.414395537855482
  },
  {
    "center_latlon": [
      -5.710148114648483,
      -52.41276672441086
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.412266724410856 -5.710648114648483, -52.412266724410856 -5.709648114648483, -52.41326672441086 -5.709648114648483, -52.41326672441086 -5.710648114648483, -52.412266724410856 -5.710648114648483))",
    "elevation": 350.30963728250936,
    "ndvi": 0.12941176470588237,
    "anomaly_score": 7.674899181188643
  },
  {
    "center_latlon": [
      -5.894366512394855,
      -52.530862813905216
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.530362813905214 -5.894866512394855, -52.530362813905214 -5.8938665123948555, -52.53136281390522 -5.8938665123948555, -52.53136281390522 -5.894866512394855, -52.530362813905214 -5.894866512394855))",
    "elevation": 200.99940341804177,
    "ndvi": 0.9333333333333333,
    "anomaly_score": 5.542345749349016
  },
  {
    "center_latlon": [
      -5.862608378689217,
      -52.51521182359307
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.51471182359307 -5.863108378689216, -52.51471182359307 -5.862108378689217, -52.51571182359307 -5.862108378689217, -52.51571182359307 -5.863108378689216, -52.51471182359307 -5.863108378689216))",
    "elevation": 190.38550852425396,
    "ndvi": 0.8627450980392157,
    "anomaly_score": 5.540894092870796
  }
]


# === Create folium map centered at the average anomaly location ===
avg_lat = sum([a["center_latlon"][0] for a in anomalies]) / len(anomalies)
avg_lon = sum([a["center_latlon"][1] for a in anomalies]) / len(anomalies)
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=17)

# === Plot each anomaly ===
for i, a in enumerate(anomalies):
    lat, lon = a["center_latlon"]
    radius = a["radius_m"]
    poly = wkt.loads(a["bbox_wkt"])

    # Add bounding box polygon
    folium.Polygon(
        locations=[(pt[1], pt[0]) for pt in poly.exterior.coords],
        color="red",
        weight=2,
        fill=False,
        tooltip=f"Anomaly {i+1} | Elev: {a['elevation']} | NDVI: {a['ndvi']} | Score: {a['anomaly_score']:.2f}"
    ).add_to(m)

    # Add center point
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="green",
        fill=True,
        popup=f"Center: {lat:.5f}, {lon:.5f}\nRadius: {radius} m",
    ).add_to(m)

# === Save and display ===
m.save("anomaly_footprints_map.html")
print("âœ… Map saved as anomaly_footprints_map.html")
m



import folium
from shapely import wkt
import json

# Replace with your actual anomaly data
anomalies =  [
  {
    "center_latlon": [
      -5.700886783080665,
      -52.39251980683522
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.392019806835215 -5.701386783080665, -52.392019806835215 -5.700386783080665, -52.39301980683522 -5.700386783080665, -52.39301980683522 -5.701386783080665, -52.392019806835215 -5.701386783080665))",
    "elevation": 455.59288993291557,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 13.338739909616542
  },
  {
    "center_latlon": [
      -5.727974589766515,
      -52.39032776552173
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.38982776552173 -5.728474589766515, -52.38982776552173 -5.727474589766516, -52.390827765521735 -5.727474589766516, -52.390827765521735 -5.728474589766515, -52.38982776552173 -5.728474589766515))",
    "elevation": 416.20562410820276,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 11.414395537855482
  },
  {
    "center_latlon": [
      -5.710148114648483,
      -52.41276672441086
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.412266724410856 -5.710648114648483, -52.412266724410856 -5.709648114648483, -52.41326672441086 -5.709648114648483, -52.41326672441086 -5.710648114648483, -52.412266724410856 -5.710648114648483))",
    "elevation": 350.30963728250936,
    "ndvi": 0.12941176470588237,
    "anomaly_score": 7.674899181188643
  },
  {
    "center_latlon": [
      -5.894366512394855,
      -52.530862813905216
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.530362813905214 -5.894866512394855, -52.530362813905214 -5.8938665123948555, -52.53136281390522 -5.8938665123948555, -52.53136281390522 -5.894866512394855, -52.530362813905214 -5.894866512394855))",
    "elevation": 200.99940341804177,
    "ndvi": 0.9333333333333333,
    "anomaly_score": 5.542345749349016
  },
  {
    "center_latlon": [
      -5.862608378689217,
      -52.51521182359307
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.51471182359307 -5.863108378689216, -52.51471182359307 -5.862108378689217, -52.51571182359307 -5.862108378689217, -52.51571182359307 -5.863108378689216, -52.51471182359307 -5.863108378689216))",
    "elevation": 190.38550852425396,
    "ndvi": 0.8627450980392157,
    "anomaly_score": 5.540894092870796
  }
]
# Create base map centered at first anomaly
m = folium.Map(location=anomalies[0]["center_latlon"], zoom_start=18, tiles=None)

# Add Esri World Imagery basemap
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri World Imagery",
    overlay=False
).add_to(m)

# Plot anomaly footprints
for anomaly in anomalies:
    bbox = wkt.loads(anomaly["bbox_wkt"])
    folium.GeoJson(bbox, style_function=lambda x: {"color": "red", "weight": 2}).add_to(m)
    folium.Marker(
        location=anomaly["center_latlon"],
        popup=f"Anomaly Score: {anomaly['anomaly_score']:.2f}",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

folium.LayerControl().add_to(m)
m



import json
import datetime
from openai import OpenAI
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
secrets_client = UserSecretsClient()
api_key = secrets_client.get_secret("MySecret Key")
client = OpenAI(api_key=api_key)

def log_and_prompt_anomalies(anomalies, dataset_ids, prompt_text, log_path="run_log.json"):
    """
    Logs anomalies, dataset info, prompt used, and calls OpenAI to analyze them.

    Returns:
        Tuple of (log_path, OpenAI response text)
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # Build the log structure
    log = {
        "timestamp": timestamp,
        "dataset_ids": dataset_ids,
        "prompt_used": prompt_text,
        "anomalies": anomalies
    }

    # Write to disk
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"ğŸ“¦ Run logged to: {log_path}")

    # Send prompt to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a geospatial anomaly analyst."},
            {"role": "user", "content": prompt_text},
            {"role": "user", "content": f"Here are the anomalies:\n{json.dumps(anomalies, indent=2)}"}
        ]
    )
    reply = response.choices[0].message.content
    print("ğŸ§  OpenAI response received.")
    return log_path, reply



prompt = """
Analyze the following anomalies â€” each with NDVI, elevation, and location.
Explain possible environmental or anthropogenic causes, and recommend what remote sensing data to fetch next.
"""

dataset_ids = ["GEDI_1B_v002","GEDI_2A_v002", "S2C_MSIL2A_20250601T134721_N0511_R024_T22MCU_20250601T190405-ql","T22MCU_20250601T134721_B04_10m","T22MCU_20250601T134721_B08_10m","2024-09-04-00_00_2024-09-04-23_59_Sentinel-2_L2A_NDVI"]

log_path, response = log_and_prompt_anomalies(anomalies, dataset_ids, prompt)

print("ğŸ”� Analysis:\n", response)



from openai import OpenAI
from kaggle_secrets import UserSecretsClient
secrets_client = UserSecretsClient()
api_key = secrets_client.get_secret("MySecret Key")
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":f"Using these {json.dumps(anomalies, indent=2)} anomalies as a reference can you suggest similar places in the Amazon Rainforest which may have been affected by anthropogenic activities by pre-coloumbian era inhabitants that remain undiscovered and their possible locations"}])
reply = response.choices[0].message.content
print("ğŸ§  OpenAI response received.")


print(reply)


pip install rasterio geopandas matplotlib numpy shapely folium



import rasterio
import matplotlib.pyplot as plt

# Load the HGT file (example)
hgt_path = '/kaggle/input/digitalelevationmodel/s06w053.hgt'
with rasterio.open(hgt_path) as src:
    elevation = src.read(1)
    plt.imshow(elevation, cmap='terrain')
    plt.title("Elevation from NASADEM")
    plt.colorbar(label='Elevation (meters)')
    plt.show()



from shapely.geometry import Point
import geopandas as gpd

# Example anomaly
anomalies = [
  {
    "center_latlon": [
      -5.700886783080665,
      -52.39251980683522
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.392019806835215 -5.701386783080665, -52.392019806835215 -5.700386783080665, -52.39301980683522 -5.700386783080665, -52.39301980683522 -5.701386783080665, -52.392019806835215 -5.701386783080665))",
    "elevation": 455.59288993291557,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 13.338739909616542
  },
  {
    "center_latlon": [
      -5.727974589766515,
      -52.39032776552173
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.38982776552173 -5.728474589766515, -52.38982776552173 -5.727474589766516, -52.390827765521735 -5.727474589766516, -52.390827765521735 -5.728474589766515, -52.38982776552173 -5.728474589766515))",
    "elevation": 416.20562410820276,
    "ndvi": 0.058823529411764705,
    "anomaly_score": 11.414395537855482
  },
  {
    "center_latlon": [
      -5.710148114648483,
      -52.41276672441086
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.412266724410856 -5.710648114648483, -52.412266724410856 -5.709648114648483, -52.41326672441086 -5.709648114648483, -52.41326672441086 -5.710648114648483, -52.412266724410856 -5.710648114648483))",
    "elevation": 350.30963728250936,
    "ndvi": 0.12941176470588237,
    "anomaly_score": 7.674899181188643
  },
  {
    "center_latlon": [
      -5.894366512394855,
      -52.530862813905216
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.530362813905214 -5.894866512394855, -52.530362813905214 -5.8938665123948555, -52.53136281390522 -5.8938665123948555, -52.53136281390522 -5.894866512394855, -52.530362813905214 -5.894866512394855))",
    "elevation": 200.99940341804177,
    "ndvi": 0.9333333333333333,
    "anomaly_score": 5.542345749349016
  },
  {
    "center_latlon": [
      -5.862608378689217,
      -52.51521182359307
    ],
    "radius_m": 50,
    "bbox_wkt": "POLYGON ((-52.51471182359307 -5.863108378689216, -52.51471182359307 -5.862108378689217, -52.51571182359307 -5.862108378689217, -52.51571182359307 -5.863108378689216, -52.51471182359307 -5.863108378689216))",
    "elevation": 190.38550852425396,
    "ndvi": 0.8627450980392157,
    "anomaly_score": 5.540894092870796
  }
]
# Convert to GeoDataFrame
points = [Point(lon, lat) for lat, lon in [a["center_latlon"] for a in anomalies]]
gdf = gpd.GeoDataFrame(anomalies, geometry=points, crs="EPSG:4326")

# Plot overlay
fig, ax = plt.subplots(figsize=(10, 8))
plt.imshow(elevation, cmap='terrain', extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
gdf.plot(ax=ax, marker='o', color='red', label='Anomalies')
plt.legend()
plt.title("Anomalies Over NASADEM")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
print("The above overlay confirms the accuracy of the anomalies,thus we can proceed further.We can clearly see the abnormal elevations near the areas where the first 3 anomalies are located whereas the last 2 anomalies show that these places are located  near a river bank thereby showing lower NDVI Values.")


!pip install fastkml


from lxml import etree
from shapely.geometry import Polygon
import geopandas as gpd

# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2024.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs.geojson", driver="GeoJSON")



from lxml import etree
from shapely.geometry import Polygon
import geopandas as gpd

# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2023.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2023.geojson", driver="GeoJSON")



from lxml import etree
from shapely.geometry import Polygon
import geopandas as gpd

# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2022.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2022.geojson", driver="GeoJSON")



from lxml import etree
from shapely.geometry import Polygon
import geopandas as gpd
# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2021.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2021.geojson", driver="GeoJSON")



# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2020.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2020.geojson", driver="GeoJSON")



# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2016.kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2016.geojson", driver="GeoJSON")



# --- 1. Parse the KML ---
with open("/kaggle/input/geoglyphs/amazon_geoglyphs-2016(2).kml", 'rb') as f:
    tree = etree.parse(f)

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# --- 2. Extract Placemarks and Coordinates ---
placemarks = tree.findall('.//kml:Placemark', namespaces=ns)

geoms = []
names = []

for pm in placemarks:
    name_el = pm.find('kml:name', namespaces=ns)
    name = name_el.text if name_el is not None else "Unnamed"

    coords_el = pm.find('.//kml:coordinates', namespaces=ns)
    if coords_el is not None:
        coord_text = coords_el.text.strip()
        coords = []
        for c in coord_text.split():
            parts = c.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if len(coords) >= 3:  # Need at least 3 points to form polygon
            geom = Polygon(coords)
            geoms.append(geom)
            names.append(name)

# --- 3. Construct GeoDataFrame ---
gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs="EPSG:4326")

# --- 4. Preview and export ---
print(f"âœ… Parsed {len(gdf)} geoglyphs")
display(gdf.head())

# Optional: Save
gdf.to_file("amazon_geoglyphs-2016(2).geojson", driver="GeoJSON")



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("amazon_geoglyphs.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2023.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2022.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2021.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2020.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2016.geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



import folium
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from folium import GeoJson

# Load GeoDataFrame
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs-2016(2).geojson")

# --- 1. Fix invalid geometries ---
gdf["geometry"] = gdf["geometry"].buffer(0)

# Optional: drop geometries that are still invalid or empty
gdf = gdf[gdf.is_valid & ~gdf.is_empty]

# --- 2. Calculate centroid for map center ---
center_geom = unary_union(gdf.geometry)
center = center_geom.centroid
map_center = [center.y, center.x]

# --- 3. Initialize map ---
m = folium.Map(location=map_center, zoom_start=7, tiles=None)

# Add Sentinel-2-like Esri imagery
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

# --- 4. Overlay the geoglyphs ---
GeoJson(
    gdf,
    name="Amazon Geoglyphs",
    tooltip=folium.GeoJsonTooltip(fields=["name"] if "name" in gdf.columns else []),
    style_function=lambda x: {
        'color': 'orange',
        'weight': 2,
        'fillColor': 'yellow',
        'fillOpacity': 0.3
    }
).add_to(m)

# --- 5. Add Layer Control ---
folium.LayerControl().add_to(m)

# Show the map
m



!pip install contextily


!pip install selenium pillow folium tqdm



import geopandas as gpd
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image
import time
import os
from tqdm import tqdm

# --- CONFIG ---
geojson_path = "/kaggle/working/amazon_geoglyphs.geojson"
output_dir = "/kaggle/working/geo_tiles"
os.makedirs(output_dir, exist_ok=True)

TILE_SIZE = 512
ZOOM = 16
WAIT = 0.5  # Seconds to wait after loading map

# --- LOAD GEOGLYPHS ---
gdf = gpd.read_file(geojson_path)
gdf = gdf.to_crs(epsg=4326)

# --- SETUP SELENIUM HEADLESS BROWSER ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument(f"--window-size={TILE_SIZE},{TILE_SIZE}")
driver = webdriver.Chrome(options=options)

# --- TILE EXPORT LOOP ---
for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
    try:
        geom = row.geometry
        centroid = geom.centroid
        lat, lon = centroid.y, centroid.x

        # Generate map
        m = folium.Map(
            location=[lat, lon],
            zoom_start=ZOOM,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri"
        )
        folium.GeoJson(geom.__geo_interface__, style_function=lambda x: {
            'fillColor': 'none',
            'color': 'red',
            'weight': 3
        }).add_to(m)

        # Save map as HTML and load it in browser
        html_path = os.path.join(output_dir, f"temp_{i}.html")
        m.save(html_path)
        driver.get(f"file://{html_path}")
        time.sleep(WAIT)

        # Screenshot full map
        img_path = os.path.join(output_dir, f"geoglyph_{i}.png")
        driver.save_screenshot(img_path)

        os.remove(html_path)  # Optional cleanup

    except Exception as e:
        print(f"[x] Failed geoglyph_{i}.png due to {e}")

driver.quit()
print("âœ… Finished all geoglyphs.")



import rasterio.features
from affine import Affine
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_masks_from_gdf(gdf, out_dir_img, out_dir_mask, dpi=150, zoom_buffer=500):
    os.makedirs(out_dir_mask, exist_ok=True)
    
    gdf = gdf.to_crs(epsg=3857)  # Project to Web Mercator

    for i, row in gdf.iterrows():
        # Match image path
        img_path = os.path.join(out_dir_img, f"geoglyph_{i}.png")
        if not os.path.exists(img_path):
            print(f"[!] Image {img_path} not found. Skipping.")
            continue
        
        img = plt.imread(img_path)
        height, width = img.shape[:2]

        geom = row.geometry

        # Get bounds used during image creation
        minx, miny, maxx, maxy = geom.bounds
        minx -= zoom_buffer
        maxx += zoom_buffer
        miny -= zoom_buffer
        maxy += zoom_buffer

        # Calculate affine transform
        xres = (maxx - minx) / width
        yres = (maxy - miny) / height
        transform = Affine.translation(minx, maxy) * Affine.scale(xres, -yres)

        # Rasterize
        mask = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Save
        out_path = os.path.join(out_dir_mask, f"geoglyph_{i}_mask.png")
        plt.imsave(out_path, mask, cmap='gray')
        print(f"âœ… Saved mask: {out_path}")
# Assuming you already saved your gdf:
gdf = gpd.read_file("/kaggle/working/amazon_geoglyphs.geojson")
generate_masks_from_gdf(
    gdf=gdf,
    out_dir_img="/kaggle/working/geo_tiles",
    out_dir_mask="/kaggle/working/masks",
    zoom_buffer=500  # Should match value used during tile generation
)



import cv2
import numpy as np
import os

# ---- SETTINGS ---- #
image_dir = "/kaggle/working/geo_tiles"  # Update this if your image directory is different
output_image_dir = "/kaggle/working/yolo_dataset/images"
output_label_dir = "/kaggle/working/yolo_dataset/labels"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ---- FUNCTION TO EXTRACT BBOX FROM RED POLYGON ---- #
def extract_yolo_label_from_red_polygon(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    H, W = img.shape[:2]
    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    width = w / W
    height = h / H

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# ---- PROCESS IMAGES ---- #
yolo_labels = 0
for filename in os.listdir(image_dir):
    if not filename.endswith(".png"):
        continue

    image_path = os.path.join(image_dir, filename)
    label = extract_yolo_label_from_red_polygon(image_path)

    if label:
        # Save image to YOLO folder
        img_out_path = os.path.join(output_image_dir, filename)
        cv2.imwrite(img_out_path, cv2.imread(image_path))

        # Write label
        label_out_path = os.path.join(output_label_dir, filename.replace(".png", ".txt"))
        with open(label_out_path, "w") as f:
            f.write(label)
        yolo_labels += 1

print(f"âœ… Created {yolo_labels} labeled YOLO training images")



data_yaml = """
train: /kaggle/working/yolo_dataset/images
val: /kaggle/working/yolo_dataset/images

nc: 1
names: ['geoglyph']
"""

with open("/kaggle/working/data.yaml", "w") as f:
    f.write(data_yaml.strip())

print("âœ… data.yaml created!")



pip install transformers timm datasets



from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import numpy as np

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval().cuda()  # Use GPU if available



pip install transformers timm datasets albumentations torch torchvision



import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# Dataset
class GeoglyphSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.img_files[idx].replace(".png", "_mask.png"))).convert("L")
        img = np.array(img)
        mask = (np.array(mask) // 255).astype(np.uint8)  # ensure 0/1 class labels

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        return img, mask



from torch.amp import autocast, GradScaler

# Feature Extractor
feat_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,# binary segmentation
    ignore_mismatched_sizes=True 
)
model = torch.nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# ----- Loss and optimizer -----
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()


train_ds = GeoglyphSegDataset("/kaggle/working/geo_tiles", "/kaggle/working/masks")
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)



num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for imgs, masks in train_loader:
        # Convert to list of PIL-compatible np arrays
        img_list = [np.array(img) if not isinstance(img, np.ndarray) else img for img in imgs]
        inputs = feat_extractor(images=img_list, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        masks = masks.unsqueeze(1).float().to(device)  # shape: [B, 1, H, W]

        optimizer.zero_grad()

    with autocast(device_type="cuda"):
        outputs = model(pixel_values=inputs["pixel_values"])
        logits = outputs.logits  # shape: [B, 1, 128, 128]
    
        # Resize ground truth to match output
        resized_masks = torch.nn.functional.interpolate(masks, size=logits.shape[-2:], mode="nearest")
    
        loss = criterion(logits, resized_masks)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} â€” Loss: {running_loss / len(train_loader):.4f}")


pip install ultralytics --quiet



from ultralytics import YOLO



!yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640



# Xingu River Basin (Southern Brazilian Amazon)
XINGU_BBOX = (-54.0, -6.0, -52.0, -4.0)


import numpy as np

BBOXES = [XINGU_BBOX]

for bbox in BBOXES:
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_steps = np.arange(bbox[0], bbox[2], 0.05)
    lat_steps = np.arange(bbox[1], bbox[3], 0.05)



!pip install mercantile


from PIL import Image
import mercantile
import requests
from io import BytesIO
import os

# Fetch a single ESRI imagery tile at a given lat/lon/zoom
def fetch_tile(lat, lon, zoom=16, save_dir="tiles"):
    tile = mercantile.tile(lon, lat, zoom)
    tile_size = 256  # Standard tile size

    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            tile_img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            tile_img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            print(f"Received status {response.status_code} for tile ({tile.x}, {tile.y})")
    except Exception as e:
        print(f"Tile fetch failed for ({tile.x}, {tile.y}): {e}")
        tile_img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))

    # Save to disk
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"tile_{lat}_{lon}_z{zoom}.png")
    tile_img.save(filename)
    print(f"Tile saved to {filename}")
    return tile_img



from tqdm import tqdm

def scan_amazon(zoom=10):
    for lat in tqdm(lat_steps, desc="Scanning latitude"):
        for lon in lon_steps:
                tile_img = fetch_tile(lat, lon, zoom)
scan_amazon()


import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

def predict_masks_in_directory(
    image_dir,
    output_dir,
    model,
    feat_extractor,
    device="cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in tqdm(image_files, desc="Predicting masks"):
        img_path = os.path.join(image_dir, img_file)
        out_path = os.path.join(output_dir, img_file.replace(".png", "_pred.png"))

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        inputs = feat_extractor(images=img, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)
            mask = (probs[0, 0] > 0.5).float().cpu().numpy() * 255
            mask = mask.astype(np.uint8)

        Image.fromarray(mask).save(out_path)

    print(f"âœ… All masks saved to: {output_dir}")



# extractor = torch.load("/kaggle/input/segmentation-using-the-segformer-model/pytorch/default/1/segformer_bundle.pth",weights_only=False)["feature_extractor"]

# Run batch prediction
predict_masks_in_directory(
    image_dir="/kaggle/working/tiles",
    output_dir="/kaggle/working/pred_masks",
    model=model,
    feat_extractor=feat_extractor,
    device="cuda"
)



from PIL import Image

def overlay_mask_on_image(image_path, mask_path, save_path, mask_color=(255, 0, 0), alpha=0.5):
    # Load original image and predicted mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)

    # Create RGBA overlay from mask
    color_mask = Image.new("RGBA", image.size, mask_color + (0,))
    mask_data = mask.point(lambda p: int(p > 128) * int(255 * alpha))
    color_mask.putalpha(mask_data)

    # Overlay
    overlay_image = Image.alpha_composite(image.convert("RGBA"), color_mask)

    # Save result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    overlay_image.save(save_path)

# Example usage
import glob
for img_path in glob.glob("/kaggle/working/tiles/*.png"):
    base_name = os.path.basename(img_path).replace(".png", "")
    mask_path = f"/kaggle/working/pred_masks/{base_name}_pred.png"
    overlay_save_path = f"/kaggle/working/test_overlay/{base_name}_overlay.png"

    overlay_mask_on_image(img_path, mask_path, overlay_save_path)
    print("Overlay saved:", overlay_save_path)



import os
from ultralytics import YOLO
import cv2

# --- Paths ---
input_dir = "/kaggle/working/test_overlay"
output_with_detections = "has_geoglyphs"
output_without_detections = "no_geoglyphs"

# Create output folders
os.makedirs(output_with_detections, exist_ok=True)
os.makedirs(output_without_detections, exist_ok=True)

# --- Load YOLOv8 Model ---
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")  

# --- Run inference ---
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, filename)

        # Run inference
        results = model(img_path, conf=0.25, save=False, verbose=False)

        # Check if any detections
        detections = results[0].boxes
        img = cv2.imread(img_path)

        if len(detections) > 0:
            # Draw boxes on image
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imwrite(os.path.join(output_with_detections, filename), img)
            print(f"[âœ“] Geoglyph detected: {filename}")
        else:
            cv2.imwrite(os.path.join(output_without_detections, filename), img)
            print(f"[ ] No geoglyph: {filename}")



import cv2
import os
import numpy as np
from shapely.geometry import box

def extract_boxes_from_overlay(image_path, geo_bounds, color=(255, 0, 0), img_size=256):
    """Detect red YOLO-style boxes and convert to geo-coordinates."""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red box range in HSV (tuned for YOLO's blue/red outline)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_lon, min_lat, max_lon, max_lat = geo_bounds
    polygons = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Convert to geo-coordinates
        lon_min = min_lon + (x / img_size) * (max_lon - min_lon)
        lat_max = max_lat - (y / img_size) * (max_lat - min_lat)
        lon_max = min_lon + ((x + w) / img_size) * (max_lon - min_lon)
        lat_min = max_lat - ((y + h) / img_size) * (max_lat - min_lat)

        polygons.append(box(lon_min, lat_min, lon_max, lat_max))

    return polygons



import folium

def overlay_boxes_on_map(polygons, save_path="geoglyph_map.html"):
    m = folium.Map(location=[-10, -60], zoom_start=5, tiles=None)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Esri World Imagery").add_to(m)

    for poly in polygons:
        folium.GeoJson(poly.__geo_interface__,
                       style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0.2}
                       ).add_to(m)
    m.add_child(folium.LatLngPopup())


    m.save(save_path)
    print(f"âœ… Map saved to: {save_path}")



import os

def get_tile_bounds_from_filename(filename, tile_size_deg=0.5):
    """
    Extract geographic bounding box from tile filename.
    Assumes filename format: tile_<lat>_<lon>_z<zoom>_overlay.jpg
    Returns: [min_lon, min_lat, max_lon, max_lat]
    """
    name = os.path.splitext(filename)[0]  # Remove .jpg
    parts = name.replace("tile_", "").replace("_overlay", "").split("_z")[0].split("_")
    
    if len(parts) < 2:
        raise ValueError(f"Filename not parseable: {filename}")

    lat, lon = float(parts[0]), float(parts[1])
    min_lat = lat
    max_lat = lat + tile_size_deg
    min_lon = lon
    max_lon = lon + tile_size_deg

    return [min_lon, min_lat, max_lon, max_lat]



all_boxes = []
image_folder = "/kaggle/working/has_geoglyphs"

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        try:
            geo_bounds = get_tile_bounds_from_filename(filename)
            img_path = os.path.join(image_folder, filename)
            boxes = extract_boxes_from_overlay(img_path, geo_bounds)
            all_boxes.extend(boxes)
        except Exception as e:
            print(f"[âš ï¸�] Failed for {filename}: {e}")

# Plot
overlay_boxes_on_map(all_boxes, save_path="/kaggle/working/fgeoglyph_overlay_map.html")



!pip install selenium


import folium
from shapely.geometry import box, mapping
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os

# === STEP 1: Use your manually selected bounding box ===
selected_bounds = (-51.9922, -3.6241, -51.8922, -3.5241)
selected_poly = box(*selected_bounds)

center_lat = (selected_bounds[1] + selected_bounds[3]) / 2
center_lon = (selected_bounds[0] + selected_bounds[2]) / 2
map_center = [center_lat, center_lon]

# === STEP 2: Setup Mapbox Tiles ===
MAPBOX_TOKEN = "pk.eyJ1Ijoic2h5YW1oZm4iLCJhIjoiY2x3czFseGYwMDBpczJqc2FuODMwcmt6MyJ9.TjkwouqkFPuEu5EqSeiKrA"
mapbox_tiles = (
    f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}"
    f"?access_token={MAPBOX_TOKEN}"
)

# === STEP 3: Create the map ===
m = folium.Map(location=map_center, zoom_start=20, tiles=None)
folium.TileLayer(
    tiles=mapbox_tiles,
    attr="Mapbox",
    name="Satellite",
    max_zoom=1200,
    tile_size=512,
    zoom_offset=-1,
).add_to(m)

# === STEP 4: Add red polygon overlay ===
folium.GeoJson(
    data=mapping(selected_poly),
    style_function=lambda x: {
        "fillColor": "red",
        "color": "yellow",
        "weight": 2,
        "fillOpacity": 0.15,
    },
    name="Selected Geoglyph Area"
).add_to(m)

folium.LayerControl().add_to(m)

# === STEP 5: Save the map ===
html_path = "/kaggle/working/best_site_geoglyph_focus.html"
m.save(html_path)
print("âœ… Map saved to:", html_path)

# === STEP 6: Save Screenshot using Selenium ===
img_path = html_path.replace(".html", ".png")

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1280x720")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
driver.get("file://" + html_path)
time.sleep(3)  # wait for map tiles to load

driver.save_screenshot(img_path)
driver.quit()

print("ğŸ–¼ï¸� Screenshot saved to:", img_path)



m


#Storing the visuals and labels of the selected map 
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# --- Paths ---
model_path = "/kaggle/working/runs/detect/train/weights/best.pt"
test_images_dir = "/kaggle/working/best_site_geoglyph_focus.png"
output_labels_dir = "/kaggle/working/best_site_label"
output_overlay_dir = "/kaggle/working/best_site_visuals"
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

# --- Load model ---
model = YOLO(model_path)

# --- Inference ---
results = model.predict(
    source=test_images_dir,
    save=False,
    conf=0.2,
    iou=0.5,
    stream=True,
)

# --- Process each result ---
for result in results:
    img_name = os.path.basename(result.path)
    base_name = os.path.splitext(img_name)[0]

    img = Image.open(result.path)
    width, height = img.size

    # Prepare label file
    label_path = os.path.join(output_labels_dir, base_name + ".txt")
    with open(label_path, "w") as f:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # Optionally: save visual overlay
    result.save(filename=os.path.join(output_overlay_dir, base_name + "_overlay.jpg"))

print("âœ… YOLO label files generated.")



import folium
from folium.plugins import DualMap
from shapely.geometry import box
import os
from PIL import Image

# ---- Helper to convert YOLO bbox to lat/lon ----
def yolo_to_geo_bbox(x_center, y_center, w, h, bounds, img_size):
    lon_min, lat_min, lon_max, lat_max = bounds
    img_w, img_h = img_size
    xc, yc = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h
    x1, y1 = xc - bw / 2, yc - bh / 2
    x2, y2 = xc + bw / 2, yc + bh / 2
    lon1 = lon_min + (x1 / img_w) * (lon_max - lon_min)
    lon2 = lon_min + (x2 / img_w) * (lon_max - lon_min)
    lat1 = lat_max - (y1 / img_h) * (lat_max - lat_min)
    lat2 = lat_max - (y2 / img_h) * (lat_max - lat_min)
    return [[lat1, lon1], [lat2, lon2]]

# ---- Constants ----
MAPBOX_TOKEN = "pk.eyJ1Ijoic2h5YW1oZm4iLCJhIjoiY2x3czFseGYwMDBpczJqc2FuODMwcmt6MyJ9.TjkwouqkFPuEu5EqSeiKrA"
tiles_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}"

# ---- Site Coordinates ----
your_site_center = [-3.5741, -51.9422]
your_bounds = [-51.9922, -3.6241, -51.8922, -3.5241]  # W, S, E, N
jaco_site_center = [-10.4466, -62.5482]
jaco_bounds = [[-10.4966, -62.5982], [-10.3966, -62.4982]]

# ---- Initialize DualMap ----
dual_map = DualMap(location=[-7.0, -57.0], zoom_start=15, control=False, synced=False)

# ---- LEFT MAP: Your Site ----
folium.TileLayer(tiles=tiles_url, attr="Mapbox", name="Mapbox Satellite",
                 max_zoom=22, tile_size=512, zoom_offset=-1).add_to(dual_map.m1)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Boundaries",
    name="Place Labels + Rivers",
    overlay=True,
    control=True
).add_to(dual_map)
folium.Marker(location=your_site_center, popup="ğŸ”´ Your Site").add_to(dual_map.m1)

# Parse YOLO detections
label_file = "/kaggle/working/best_site_label/best_site_geoglyph_focus.txt"
image_path = "/kaggle/working/best_site_visuals/best_site_geoglyph_focus_overlay.jpg"
img_size = Image.open(image_path).size

with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            _, x, y, w, h = map(float, parts)
            bbox = yolo_to_geo_bbox(x, y, w, h, your_bounds, img_size)
            folium.Rectangle(bbox, color="red", fill=True, fill_opacity=0.3).add_to(dual_map.m1)

# ---- RIGHT MAP: Jaco SÃ¡ Site ----
folium.TileLayer(tiles=tiles_url, attr="Mapbox", name="Mapbox Satellite",
                 max_zoom=22, tile_size=512, zoom_offset=-1).add_to(dual_map.m2)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Boundaries",
    name="Place Labels + Rivers",
    overlay=True,
    control=True
).add_to(dual_map)
folium.Marker(location=jaco_site_center, popup="ğŸŸ¦ Jaco SÃ¡ Site").add_to(dual_map.m2)
folium.Rectangle(bounds=jaco_bounds, color='blue', weight=2).add_to(dual_map.m2)

# Simulated geoglyph overlays (add your own coords if needed)
# Example polygon
polygon_coords = [
    [-10.445, -62.555],
    [-10.445, -62.545],
    [-10.455, -62.545],
    [-10.455, -62.555],
    [-10.445, -62.555]
]
folium.Polygon(locations=polygon_coords, color="yellow", fill=True, fill_opacity=0.2, popup="Geoglyph").add_to(dual_map.m2)

# ---- Save the Split Map ----
dual_map.save("/kaggle/working/split_geoglyph_comparison.html")
print("âœ… Comparison map saved to: split_geoglyph_comparison.html")



dual_map


from openai import OpenAI
from kaggle_secrets import UserSecretsClient
secrets_client = UserSecretsClient()
api_key = secrets_client.get_secret("MySecret Key")
client = OpenAI(api_key=api_key)
import base64

# Encode the image to base64
def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
base64_image = encode_image_base64('/kaggle/working/best_site_visuals/best_site_geoglyph_focus_overlay.jpg')

from openai import OpenAI
from kaggle_secrets import UserSecretsClient
secrets_client = UserSecretsClient()
api_key = secrets_client.get_secret("MySecret Key")

client = OpenAI(api_key=api_key)


response = client.chat.completions.create(
    model="gpt-4o-mini",  # Or "gpt-4o"
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Derive a historical and cultural cross-reference between the geoglyphs in Jaco SÃ¡ and the geoglyphs detected in the following image"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
            ],
        }
    ],
    max_tokens=300,
)



print("Response:\n", response.choices[0].message.content)


