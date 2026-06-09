# benchmark.py

from kaggle_secrets import UserSecretsClient
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

# Load OpenAI API key from Kaggle secrets or environment
try:
    user_secrets = UserSecretsClient()
    openai_key = user_secrets.get_secret("openai")
except Exception:
    import os
    openai_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) if openai_key else OpenAI()

# Define Pydantic models for structured output
class BenchmarkSite(BaseModel):
    name: str
    lat: float
    lon: float

class BenchmarkSites(BaseModel):
    sites: list[BenchmarkSite]

# Prompt for OpenAI Structured Output (expects a JSON object with a 'sites' key)
prompt = (
    "You are an archaeologist specialized in the Amazon region.\n"
    "List at least 10 known archaeological sites located in the state of Acre, Brazil, "
    "including their approximate latitude and longitude.\n"
    "Return ONLY a JSON object with a 'sites' key, which is a list of objects with fields: name (string), lat (number), lon (number).\n"
    "Example: {\"sites\": [{\"name\": \"Site Name\", \"lat\": -X.XXXX, \"lon\": -Y.YYYY}]}\n"
    "Focus on geoglyphs and earthworks documented in academic literature or official records.\n"
)

# Call OpenAI API and parse with Pydantic Structured Output
response = client.responses.parse(
    model="o3",
    input=[{"role": "user", "content": prompt}],
    text_format=BenchmarkSites,
)

sites = response.output_parsed.sites
df_benchmark = pd.DataFrame([s.model_dump() for s in sites])

# Add Google Maps column before display
def make_gmaps_link(lat, lon):
    url = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
    return f'<a href="{url}" target="_blank">View on Google Maps</a>'

df_benchmark['Google Maps'] = df_benchmark.apply(lambda row: make_gmaps_link(row['lat'], row['lon']), axis=1)

# Display DataFrame in notebook/Kaggle environment, fallback to print
try:
    from IPython.display import display, HTML
    display(HTML(df_benchmark.to_html(escape=False)))
except Exception:
    print(df_benchmark)

# Print model version used
print(f"\n[INFO] OpenAI model used: o3")

# Print token usage if available
usage = getattr(response, "usage", None)
if usage:
    print(f"\nPrompt tokens: {getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', None))}")
    print(f"Completion tokens: {getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', None))}")
    print(f"Total tokens: {getattr(usage, 'total_tokens', None)}")


# auth.py

from kaggle_secrets import UserSecretsClient
import json
import ee

# Get the Service Account key from Kaggle Secrets
user_secrets = UserSecretsClient()
gcloud_key = user_secrets.get_secret("service_account")  # or the secret name you used

# Save the key to a file (Earth Engine expects a file)
with open('gcloud_key.json', 'w') as f:
    f.write(gcloud_key)

# Load the service account email
service_account_info = json.loads(gcloud_key)
service_account = service_account_info['client_email']

# Authenticate Earth Engine with the service account
credentials = ee.ServiceAccountCredentials(service_account, 'gcloud_key.json')
ee.Initialize(credentials)


#get-benchmark-data.py

# --- DATASET IDS USED ---
DATASET_IDS = [
    'COPERNICUS/S2_SR_HARMONIZED',
    'COPERNICUS/S1_GRD',
    'USGS/SRTMGL1_003',
    'projects/mapbiomas-raisg/public/collection3/mapbiomas_raisg_panamazonia_collection3_integration_v2',
    'LARSE/GEDI/GEDI02_A_002_MONTHLY',
    'NASA/JPL/global_forest_canopy_height_2005'
]

print("[INFO] Datasets used in this script:")
for ds in DATASET_IDS:
    print(f"  - {ds}")
print()

# --- Earth Engine functions ---
# --- Authenticate with Earth Engine before running this block ---

import ee
import pandas as pd
import time

# --- Earth Engine functions ---

def get_ndvi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))
    # Get the least cloudy image
    s2_sorted = s2.sort('CLOUDY_PIXEL_PERCENTAGE')
    img = ee.Image(s2_sorted.first())
    ndvi_img = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi = ndvi_img.select('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDVI')
    # Get the image ID
    img_id = img.get('PRODUCT_ID')
    if img_id is None:
        img_id = img.get('system:index')
    return {
        'ndvi': ndvi.getInfo() if ndvi is not None else None,
        'sentinel2_id': img_id.getInfo() if img_id is not None else None
    }

def get_ndwi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .map(lambda img: img.normalizedDifference(['B3', 'B8']).rename('NDWI')))
    ndwi = s2.median().select('NDWI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDWI')
    return ndwi.getInfo() if ndwi is not None else None

def get_ndbi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .map(lambda img: img.normalizedDifference(['B11', 'B8']).rename('NDBI')))
    ndbi = s2.median().select('NDBI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDBI')
    return ndbi.getInfo() if ndbi is not None else None

def get_srtm_elevation(lat, lon, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    srtm = ee.Image("USGS/SRTMGL1_003")
    elev = srtm.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=30).get('elevation')
    return elev.getInfo() if elev is not None else None

def get_srtm_slope(lat, lon, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    elev = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(elev)
    slope_val = slope.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=30).get('slope')
    return slope_val.getInfo() if slope_val is not None else None

def get_sentinel1_vv(lat, lon, year=2023, buffer_m=1000):
    """
    Returns the average VV backscatter of Sentinel-1.
    
    buffer_m = 1000 by default (>> 50 m from other sensors)
    ──────────────────────────────────────────────────────────
    Using a larger buffer reduces the speckle noise characteristic
    of radar images when spatially averaging. 
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        roi = point.buffer(buffer_m).bounds()
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV')
        count = s1.size().getInfo()
        if count == 0:
            return {'vv': None, 'sentinel1_id': None}
        s1_img = s1.median()
        vv_value = s1_img.reduceRegion(ee.Reducer.mean(), point, 30).get('VV')
        # Get scene ID from first image
        first_img = ee.Image(s1.first())
        img_id = first_img.get('system:index')
        return {
            'vv': vv_value.getInfo() if vv_value is not None else None,
            'sentinel1_id': img_id.getInfo() if img_id is not None else None
        }
    except Exception as e:
        print(f"Sentinel-1 VV error at ({lat}, {lon}): {e}")
        return {'vv': None, 'sentinel1_id': None}

def get_sentinel1_vh(lat, lon, year=2023, buffer_m=1000):
    """
    Returns the average VH backscatter from Sentinel-1.
    
    The 1000m buffer helps to smooth out radar speckle.
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        roi = point.buffer(buffer_m).bounds()
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select('VH')
        count = s1.size().getInfo()
        if count == 0:
            return None
        s1_img = s1.median()
        vh_value = s1_img.reduceRegion(ee.Reducer.mean(), point, 30).get('VH')
        return vh_value.getInfo() if vh_value is not None else None
    except Exception as e:
        print(f"Sentinel-1 VH error at ({lat}, {lon}): {e}")
        return None

def get_mapbiomas_class(lat, lon, year=2020):
    try:
        point = ee.Geometry.Point(lon, lat)
        img = ee.Image('projects/mapbiomas-raisg/public/collection3/mapbiomas_raisg_panamazonia_collection3_integration_v2') \
            .select(f'classification_{year}')
        value = img.reduceRegion(ee.Reducer.mode(), point, 30).get(f'classification_{year}')
        return value.getInfo() if value is not None else None
    except Exception as e:
        print(f"MapBiomas error at ({lat}, {lon}): {e}")
        return None


def get_gedi_canopy_height(lat, lon):
    """
    Returns mean GEDI canopy height (rh98) for a point.
    If GEDI is not available, fallback to NASA/JPL/global_forest_canopy_height_2005.
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        gedi = (ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
                .filterBounds(point))
        if gedi.size().getInfo() == 0:
            raise ValueError("No GEDI pulses")
        gedi_img = gedi.select('rh98').median()
        value = gedi_img.reduceRegion(
            ee.Reducer.mean(), point, 25).get('rh98')
        val = value.getInfo() if value is not None else None
        if val is not None and val != 0:
            return val
        # If value is None or 0, fallback
        raise ValueError("GEDI returned None or 0")
    except Exception as e:
        print(f"GEDI error at ({lat}, {lon}): {e}. Trying NASA/JPL/global_forest_canopy_height_2005...")
        try:
            # NASA/JPL/global_forest_canopy_height_2005: altura média do dossel em metros (2005)
            point = ee.Geometry.Point([lon, lat])
            canopy_img = ee.Image('NASA/JPL/global_forest_canopy_height_2005')
            value = canopy_img.reduceRegion(
                ee.Reducer.mean(), point, 1000).get('1')
            val = value.getInfo() if value is not None else None
            return val
        except Exception as e2:
            print(f"Fallback canopy height error at ({lat}, {lon}): {e2}")
            return None

# --- Enrich DataFrame with all sensors ---

def enrich_benchmarks_with_all_sensors(
    df,
    ndvi_year=2023,
    ndwi_year=2023,
    ndbi_year=2023,
    s1_year=2023,
    mapbiomas_year=2020,
    buffer_m=50,
    delay=1
):
    ndvi_list = []
    s2_id_list = []
    ndwi_list = []
    ndbi_list = []
    elev_list = []
    slope_list = []
    vv_list = []
    s1_id_list = []
    vh_list = []
    landclass_list = []
    canopyheight_list = []

    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        print(f"Processing {row.get('name', 'site')} ({lat}, {lon})...")
        
        # Get NDVI and Sentinel-2 ID
        ndvi_result = get_ndvi(lat, lon, ndvi_year, buffer_m)
        if isinstance(ndvi_result, dict):
            ndvi_list.append(ndvi_result['ndvi'])
            s2_id_list.append(ndvi_result['sentinel2_id'])
        else:
            ndvi_list.append(ndvi_result)
            s2_id_list.append(None)
        
        ndwi_list.append(get_ndwi(lat, lon, ndwi_year, buffer_m))
        ndbi_list.append(get_ndbi(lat, lon, ndbi_year, buffer_m))
        elev_list.append(get_srtm_elevation(lat, lon, buffer_m))
        slope_list.append(get_srtm_slope(lat, lon, buffer_m))
        
        # Get Sentinel-1 VV and ID
        s1_vv_result = get_sentinel1_vv(lat, lon, s1_year, buffer_m=1000)
        if isinstance(s1_vv_result, dict):
            vv_list.append(s1_vv_result['vv'])
            s1_id_list.append(s1_vv_result['sentinel1_id'])
        else:
            vv_list.append(s1_vv_result)
            s1_id_list.append(None)
        
        vh_list.append(get_sentinel1_vh(lat, lon, s1_year, buffer_m=1000))
        landclass_list.append(get_mapbiomas_class(lat, lon, mapbiomas_year))
        canopyheight_list.append(get_gedi_canopy_height(lat, lon))
        time.sleep(delay)  # To avoid quota limits

    df['NDVI'] = ndvi_list
    df['Sentinel2_ID'] = s2_id_list
    df['NDWI'] = ndwi_list
    df['NDBI'] = ndbi_list
    df['Elevation'] = elev_list
    df['Slope'] = slope_list
    df['Sentinel1_VV'] = vv_list
    df['Sentinel1_ID'] = s1_id_list
    df['Sentinel1_VH'] = vh_list
    df['MapBiomas_Class'] = landclass_list
    df['CanopyHeight'] = canopyheight_list
    return df

# --- Usage example ---

# df_benchmark = pd.read_csv("benchmark_sites_acre.csv")  # or from previous cell
df_benchmark = enrich_benchmarks_with_all_sensors(df_benchmark)

# Remove 'Google Maps' column if present (inherited from other scripts)
if 'Google Maps' in df_benchmark.columns:
    df_benchmark.drop(columns=['Google Maps'], inplace=True)
# Keep 'Sentinel2_ID' and 'Sentinel1_ID' columns - only these sensors have individual scene IDs
# Other sensors (SRTM, MapBiomas, GEDI, NASA/JPL) are static/aggregated datasets without individual scene IDs

# Substitui infinitos por NaN para evitar warnings do pandas
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
df_benchmark.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display the main DataFrame with scene IDs included
from IPython.display import display
display(df_benchmark)


# search-candidates.py

from kaggle_secrets import UserSecretsClient
import openai
from pydantic import BaseModel
from typing import List
import pandas as pd

# Initialize OpenAI client
user_secrets = UserSecretsClient()
client = openai.OpenAI(api_key=user_secrets.get_secret("openai"))

# Prompt: ask o3 for promising but underexplored locations in Nhamini-wi territories (≤200 chars rationale)
prompt = (
    "You are an Amazon explorer and researcher.\n"
    "Based on historical legends, indigenous oral history, and published expedition records, "
    "suggest up to 5 possible locations (latitude and longitude) within the Nhamini-wi region (Upper Rio Negro, near the Brazil/Colombia/Venezuela border) "
    "that could correspond to the legendary trail or its unexplored sites. "
    "Focus on areas that remain little explored archaeologically, according to the scientific literature. "
    "For each, briefly justify your choice referencing myths, remoteness, or lack of fieldwork. "
    "Return your answer as a JSON list with the fields: name, lat, lon, rationale (≤200 characters), and radius_m (fixed value, e.g., 500). "
    "Example: [{\"name\": \"Suggested Area\", \"lat\": 1.2345, \"lon\": -67.8901, \"rationale\": \"...\", \"radius_m\": 500}, ...]"
)

# Define schema with Pydantic (now includes radius_m)
class Area(BaseModel):
    name: str
    lat: float
    lon: float
    rationale: str
    radius_m: int = 500  # default radius in meters

class SuggestedAreas(BaseModel):
    areas: List[Area]

response = client.responses.parse(
    model="o3",
    input=[{"role": "user", "content": prompt}],
    text_format=SuggestedAreas,
)

areas = response.output_parsed.areas

# Ensure full text is shown in the 'rationale' column
pd.set_option('display.max_colwidth', None)


# Helper functions for bbox and WKT
import math
def get_bbox(lat, lon, radius_m):
    # Approximate 1 degree latitude ~ 111.32 km, longitude varies with latitude
    dlat = (radius_m / 111320)
    dlon = (radius_m / (40075000 * math.cos(math.radians(lat)) / 360))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

def get_bbox_wkt(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f"POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{min_lon} {max_lat}, "
        f"{max_lon} {max_lat}, "
        f"{max_lon} {min_lat}, "
        f"{min_lon} {min_lat}"  # close polygon
        f"))"
    )

def get_circle_wkt(lat, lon, radius_m, n_points=36):
    # Approximate circle as polygon
    coords = []
    for i in range(n_points+1):
        angle = 2 * math.pi * i / n_points
        dlat = (radius_m / 111320) * math.sin(angle)
        dlon = (radius_m / (40075000 * math.cos(math.radians(lat)) / 360)) * math.cos(angle)
        coords.append(f"{lon + dlon} {lat + dlat}")
    return f"POLYGON(({', '.join(coords)}))"

# Monta DataFrame com bbox e WKT
df = pd.DataFrame([a.model_dump() for a in areas])
df['bbox'] = df.apply(lambda row: get_bbox(row['lat'], row['lon'], row['radius_m']), axis=1)
df['bbox_wkt'] = df['bbox'].apply(get_bbox_wkt)
df['circle_wkt'] = df.apply(lambda row: get_circle_wkt(row['lat'], row['lon'], row['radius_m']), axis=1)


# Display as DataFrame (organized for notebook/Kaggle, agora inclui bbox e WKT)
display(df[['name', 'lat', 'lon', 'radius_m', 'bbox', 'bbox_wkt', 'circle_wkt', 'rationale']].rename(columns={
    'name': 'Name',
    'lat': 'Latitude',
    'lon': 'Longitude',
    'radius_m': 'Radius (m)',
    'rationale': 'Rationale (≤200 chars)',
    'bbox': 'BBox [min_lon, min_lat, max_lon, max_lat]',
    'bbox_wkt': 'BBox WKT',
    'circle_wkt': 'Circle WKT'
}))

# Print model version used
print(f"\n[INFO] OpenAI model used: o3")

# Print coordinates and radius for reference
print("\nSuggested Coordinates (with radius):")
for area in areas:
    print(f"{area.name}: lat {area.lat}, lon {area.lon}, radius {area.radius_m}m")

# Display usage information safely
usage = response.usage
try:
    print("\nPrompt tokens:", getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)))
    print("Completion tokens:", getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)))
    print("Total tokens:", getattr(usage, "total_tokens", None))
except Exception:
    print("\nUsage info:", usage)


# get-candidates-data.py

# Run sensor enrichment on the new candidate areas
# The column 'CanopyHeight' is used for both GEDI and NASA/JPL fallback, matching the benchmark structure

# Log dataset IDs used for enrichment
DATASET_IDS = [
    'COPERNICUS/S2_SR_HARMONIZED',
    'COPERNICUS/S1_GRD',
    'USGS/SRTMGL1_003',
    'projects/mapbiomas-raisg/public/collection3/mapbiomas_raisg_panamazonia_collection3_integration_v2',
    'LARSE/GEDI/GEDI02_A_002_MONTHLY',
    'NASA/JPL/global_forest_canopy_height_2005'
]

print("[INFO] Datasets used in candidate enrichment:")
for ds in DATASET_IDS:
    print(f"  - {ds}")
print()

df_candidates = pd.DataFrame([a.model_dump() for a in areas])
num_areas = len(areas)
df_candidates = enrich_benchmarks_with_all_sensors(df_candidates)

# Remove 'Google Maps' column if present (inherited from other scripts)
if 'Google Maps' in df_candidates.columns:
    df_candidates.drop(columns=['Google Maps'], inplace=True)
# Keep 'Sentinel2_ID' and 'Sentinel1_ID' columns - only these sensors have individual scene IDs
# Other sensors (SRTM, MapBiomas, GEDI, NASA/JPL) are static/aggregated datasets without individual scene IDs

# Add Sentinel-2 thumbnail download column for each candidate
import ee

# Functions to generate download links for different sensors
def get_rgb_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first()
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'bands': ['B4', 'B3', 'B2'],
            'min': 500, 'max': 2500,
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">RGB</a>'
        else:
            return None
    except Exception:
        return None

def get_ndvi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B8', 'B4']).rename('NDVI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': 0, 'max': 1,
            'palette': ['blue', 'white', 'green'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDVI</a>'
        else:
            return None
    except Exception:
        return None

def get_ndwi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B3', 'B8']).rename('NDWI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': -1, 'max': 1,
            'palette': ['brown', 'beige', 'blue'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDWI</a>'
        else:
            return None
    except Exception:
        return None

def get_ndbi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B11', 'B8']).rename('NDBI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': -1, 'max': 1,
            'palette': ['white', 'gray', 'black'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDBI</a>'
        else:
            return None
    except Exception:
        return None

def get_s1_vv_download_url_html(lat, lon, year="2023"):
    try:
        point = ee.Geometry.Point(lon, lat).buffer(500)
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(point) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV')
        s1_img = s1.median().clip(point)
        url = s1_img.getThumbURL({
            'region': point, 'dimensions': 512, 'min': -25, 'max': 0,
            'palette': ['black', 'white']})
        if url:
            return f'<a href="{url}" target="_blank">Sentinel-1 VV</a>'
        else:
            return None
    except Exception:
        return None

# Download column with all available links (adds only the sensors that are actually available)
def make_download_links(row):
    links = []
    rgb = get_rgb_download_url_html(row['lat'], row['lon'])
    if rgb:
        links.append(rgb)
    ndvi = get_ndvi_download_url_html(row['lat'], row['lon'])
    if ndvi:
        links.append(ndvi)
    ndwi = get_ndwi_download_url_html(row['lat'], row['lon'])
    if ndwi:
        links.append(ndwi)
    ndbi = get_ndbi_download_url_html(row['lat'], row['lon'])
    if ndbi:
        links.append(ndbi)
    s1vv = get_s1_vv_download_url_html(row['lat'], row['lon'])
    if s1vv:
        links.append(s1vv)
    # Add other sensors here if needed
    return ' | '.join(links)

df_candidates['Download'] = df_candidates.apply(make_download_links, axis=1)

# Substitui infinitos por NaN para evitar warnings do pandas
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
df_candidates.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display the main DataFrame with scene IDs included
from IPython.display import display, HTML
display(HTML(df_candidates.to_html(escape=False)))

# Check: ensure all candidates are present after enrichment
if len(df_candidates) != num_areas:
    print(f"[ERROR] Expected {num_areas} candidates, but df_candidates has {len(df_candidates)} after enrichment!")
    print("Expected IDs:", [a['name'] if hasattr(a, 'name') else a.get('name') for a in areas])
    print("Present IDs:", df_candidates['name'].tolist() if 'name' in df_candidates.columns else df_candidates.index.tolist())
else:
    pass


# compare.py

import numpy as np
import matplotlib.pyplot as plt

sensor_cols = [col for col in df_benchmark.columns if df_benchmark[col].dtype != object]

# Select only sensors with valid values in at least one group
valid_cols = [
    col for col in sensor_cols
    if (df_benchmark[col].notna().any() and df_candidates[col].notna().any())
]

coverage = (df_candidates[valid_cols].notna().sum() / len(df_candidates)).round(2)
print("Proportion of valid values ​​in each sensor (candidates):")
print(coverage)

# --- Explicit comparison explanation ---
print("\n[INFO] The following plot and statistics provide a direct comparison between the environmental parameters of known archaeological benchmarks and the new candidates.\n"
      "The Z-score profile plot visualizes how similar or different the candidates are from the benchmarks for each sensor.\n"
      "Use this to identify which candidates most closely resemble known sites, or which parameters stand out as anomalous.")

z_bench = (df_benchmark[valid_cols] - df_benchmark[valid_cols].mean()) \
          / df_benchmark[valid_cols].std()

z_cand  = (df_candidates[valid_cols] - df_benchmark[valid_cols].mean()) \
          / df_benchmark[valid_cols].std()

means_bench = z_bench.mean().values
means_cand  = z_cand.mean().values

x = np.arange(len(valid_cols))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, means_bench, marker='o', label='Benchmark (z)', linewidth=2)
ax.plot(x, means_cand, marker='s', label='Candidates (z)', linewidth=2)

ax.set_ylabel("Z-score (σ)")
ax.set_title('Sensor Parameter Means Profile: Benchmarks vs. Nhamini-wi Candidates')
ax.set_xticks(x)
ax.set_xticklabels(valid_cols, rotation=45, ha='right')
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# analyze-candidates-data.py

from openai import OpenAI
import os
import pandas as pd
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    openai_key = user_secrets.get_secret("openai")
except Exception:
    openai_key = os.environ.get("OPENAI_API_KEY")
from pydantic import BaseModel
try:
    from IPython.display import display
except ImportError:
    display = None

# Generates the mean summary for all benchmark sensors
def generate_sensor_summary(df, label):
    lines = [f"{label} stats (mean):"]
    for col in df.columns:
        if df[col].dtype != object:
            val = df[col].mean()
            lines.append(f"{col}: {val:.3f}")
    return "\n".join(lines)

# Generates a detailed summary for all candidates
def generate_candidates_detail(df):
    lines = ["Candidates:"]
    for idx, row in df.iterrows():
        vals = []
        for col in df.columns:
            if df[col].dtype != object:
                vals.append(f"{col}: {row[col]:.3f}")
            else:
                vals.append(f"{col}: {row[col]}")
        lines.append("- " + ", ".join(vals))
    return "\n".join(lines)

summary_bench = generate_sensor_summary(df_benchmark, "Benchmark")
summary_cand = generate_candidates_detail(df_candidates)
summary = f"{summary_bench}\n\n{summary_cand}"

prompt = (
    "You are an expert in Amazonian remote sensing and archaeology.\n"
    "Below are summarized environmental parameters for known archaeological sites (benchmarks) and for new candidate locations along the Nhamini-wi trail.\n"
    "Based on this data, compare the candidates to the benchmarks and assess:\n"
    "- Which, if any, of the candidates most closely match the benchmarks?\n"
    "- Are there anomalies or promising signals in the candidate data that warrant field investigation?\n"
    "- Briefly explain the key differences and what they might mean archaeologically.\n"
    "Be concise and analytical, referencing the key parameters (NDVI, NDWI, NDBI, SRTM, slope, Sentinel-1 radar, land cover, canopy height, etc.).\n"
    "Return ONLY a JSON object with a 'matches' key, which is a list of the closest candidate(s) to the benchmarks. Each match must have: name, lat, lon, reason. If none, return an empty list.\n"
    f"\n{summary}\n"
)
    # Defines the schema for Structured Outputs
schema = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "lat":    {"type": "number"},
                    "lon":    {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["name", "lat", "lon", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["matches"],
    "additionalProperties": False
}

# --- Structured Output with Pydantic ---
client = OpenAI(api_key=openai_key) if openai_key else OpenAI()

class ClosestMatch(BaseModel):
    name: str
    lat: float
    lon: float
    reason: str

class ClosestMatches(BaseModel):
    matches: list[ClosestMatch]

model_name = "o3"
response = client.responses.parse(
    model=model_name,
    input=[{"role": "user", "content": prompt}],
    text_format=ClosestMatches,
)

matches = response.output_parsed.matches

# Print model version used
print(f"\n[INFO] OpenAI model used: {model_name}")

print("\nMatches:")
for m in matches:
    print(f"- {m.name} (lat: {m.lat}, lon: {m.lon})\n  Reason: {m.reason}\n")

# Display the result as a DataFrame in Kaggle/notebook environments (optional)

import pandas as pd

# Ensure full text is shown in the 'reason' column (rationale)
pd.set_option('display.max_colwidth', None)

df_matches = pd.DataFrame([m.model_dump() for m in matches])
try:
    from IPython.display import display
    display(df_matches)
except Exception:
    print(df_matches)

# Display usage information safely (as in search-candidates.py)
usage = getattr(response, "usage", None)
try:
    print("\nPrompt tokens:", getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)))
    print("Completion tokens:", getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)))
    print("Total tokens:", getattr(usage, "total_tokens", None))
except Exception:
    print("\nUsage info:", usage)


# get-image-for-matches.py

import ee

def plot_multiple_satellite_views(lat, lon, buffer_m=1000, year=2023):
    point = ee.Geometry.Point(lon, lat).buffer(buffer_m)
    print("[INFO] Using dataset_id: COPERNICUS/S2_SR_HARMONIZED (Sentinel-2) for RGB, Infrared (NIR), NDVI, NDWI")
    
    # Get Sentinel-2 collection and select least cloudy image
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(point)
           .filterDate(f'{year}-01-01', f'{year}-12-31')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    img = s2_collection.first().clip(point)
    
    # Get Sentinel-2 scene ID
    try:
        s2_id = img.get('PRODUCT_ID')
        if s2_id is None:
            s2_id = img.get('system:index')
        s2_scene_id = s2_id.getInfo() if s2_id is not None else "N/A"
    except Exception as e:
        print(f"[WARNING] Could not get Sentinel-2 scene ID: {e}")
        s2_scene_id = "N/A"
    
    # URLs for each composite
    urls = {}
    urls['RGB'] = img.select(['B4', 'B3', 'B2']).getThumbURL({
        'region': point, 'dimensions': 512, 'min': 500, 'max': 2500})
    urls['Infrared (NIR)'] = img.select(['B8', 'B4', 'B3']).getThumbURL({
        'region': point, 'dimensions': 512, 'min': 500, 'max': 2500})
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    urls['NDVI'] = ndvi.getThumbURL({
        'region': point, 'dimensions': 512, 'min': 0, 'max': 1,
        'palette': ['blue', 'white', 'green']})
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    urls['NDWI'] = ndwi.getThumbURL({
        'region': point, 'dimensions': 512, 'min': -1, 'max': 1,
        'palette': ['brown', 'beige', 'blue']})
    
    print("[INFO] Using dataset_id: COPERNICUS/S1_GRD (Sentinel-1) for Sentinel-1 VV")
    # Sentinel-1 VV
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(point) \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .select('VV')
    
    # Check if Sentinel-1 collection has any images
    s1_size = s1_collection.size()
    try:
        s1_count = s1_size.getInfo()
        if s1_count > 0:
            s1_img = s1_collection.median().clip(point)
            
            # Get Sentinel-1 scene ID (using first image from collection)
            try:
                s1_first = s1_collection.first()
                s1_id = s1_first.get('system:index')
                s1_scene_id = s1_id.getInfo() if s1_id is not None else "N/A"
            except Exception as e:
                print(f"[WARNING] Could not get Sentinel-1 scene ID: {e}")
                s1_scene_id = "N/A"
            
            urls['Sentinel-1 VV'] = s1_img.getThumbURL({
                'region': point, 'dimensions': 512, 'min': -25, 'max': 0,
                'palette': ['black', 'white']})
        else:
            print(f"[WARNING] No Sentinel-1 images found for this location and time period")
            s1_scene_id = "No images available"
            # Skip adding Sentinel-1 VV to urls
    except Exception as e:
        print(f"[WARNING] Error processing Sentinel-1 data: {e}")
        s1_scene_id = "Error processing"
        # Skip adding Sentinel-1 VV to urls
    
    # Plot all images
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    
    # Calculate subplot dimensions based on number of images
    num_images = len(urls)
    if num_images <= 3:
        rows, cols = 1, num_images
        figsize = (4 * num_images, 4)
    else:
        rows, cols = 2, 3
        figsize = (12, 8)
    
    plt.figure(figsize=figsize)
    
    plot_idx = 1
    for name, url in urls.items():
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            im = Image.open(BytesIO(response.content))
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(im)
            plt.title(name)
            plt.axis('off')
            plot_idx += 1
        except Exception as e:
            print(f"[WARNING] Could not load image for {name}: {e}")
            # Continue with other images
    
    plt.tight_layout()
    plt.show()
    
    # Print scene IDs for reference
    print(f"\n[INFO] Scene IDs used:")
    print(f"  Sentinel-2: {s2_scene_id}")
    print(f"  Sentinel-1: {s1_scene_id}")

# Example usage for all matches using the in-memory df_matches DataFrame:
import pandas as pd
# Log dataset ID if available in df_matches
if 'df_matches' in globals() and df_matches is not None and not df_matches.empty:
    dataset_id = None
    # Try to get dataset_id from DataFrame attribute or column
    if hasattr(df_matches, 'dataset_id'):
        dataset_id = getattr(df_matches, 'dataset_id', None)
    elif 'dataset_id' in df_matches.columns:
        dataset_id = df_matches['dataset_id'].iloc[0]
    if dataset_id:
        print(f"[INFO] Using dataset_id: {dataset_id}")
    for _, m in df_matches.iterrows():
        print(f"\n[INFO] Generating images for: {m['name']} (lat: {m['lat']}, lon: {m['lon']})")
        plot_multiple_satellite_views(m['lat'], m['lon'], buffer_m=1000, year=2023)
else:
    print("No match found to generate images.")

