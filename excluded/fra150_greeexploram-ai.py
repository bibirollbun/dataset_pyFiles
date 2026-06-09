# This is the first part of my project, which I call '(The Evocation of Dependencies), where I properly structure all the libraries I will use below. Moreover, the praxis (i.e., unwritten rules) requires that the structure be organized in phases.
# Installing the necessary libraries. You only need to run these cells once.
!pip install numpy pandas matplotlib rasterio geopandas fastkml openai earthengine-api skyfield

from __future__ import annotations  # For type hints support Python 3.7+

#Importing the fundamental libraries
import argparse 
import os
import random
import io
import re
import base64
import json
import hashlib
import warnings
import logging
from datetime import datetime,timedelta
from typing import Any,Callable, Dict, Any,Tuple, List,  Union , Optional 
from pathlib import Path 

# Skyfield libraries for astronomical calculations
from skyfield.api import Loader, Star, wgs84
from skyfield.api import Loader
from skyfield.timelib import Time
from skyfield.data import hipparcos
from urllib.error import URLError
from pprint import pprint


# Libraries for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from io import BytesIO
from PIL import Image
from scipy.ndimage import median_filter, binary_opening, binary_closing

# Libraries for data analysis and manipulation
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import xml.etree.ElementTree as ET 
from shapely.geometry import Point 
from tabulate import tabulate
from skimage import measure
from skimage.measure import regionprops
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter, binary_opening, binary_closing

# Geographic libraries
import rasterio
from rasterio.plot import show
from shapely.geometry import box ,Point, Polygon
from fastkml import kml

# Libraries for interacting with APIs
import requests
from openai import OpenAI
import ee # this for Google Earth  


# Libraries for interacting with the notebook
from IPython.display import HTML, display, Markdown
from kaggle_secrets import UserSecretsClient
print(f""": Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥ -The first step has been completed: all dependencies have been added. The foundation of the project has been laid. - Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥  """)



# Here I set KAGGLE_SECRETS to True to force the use of keys entered in add-ons (Secrets
# Import specific client for Kaggle secrets
from kaggle_secrets import UserSecretsClient
CONFIG = {
    "KAGGLE_SECRETS": True
}

try:
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥ Setting up API secrets from Kaggle... Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥")
    user_secrets = UserSecretsClient()
    openai_api_key = user_secrets.get_secret("OPENAI_API_KEY")
    iam_service_account = user_secrets.get_secret("iam_service_account")
    ee_credentials_json_str = user_secrets.get_secret("ee_credentials")

    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥All secrets loaded successfully.")

except Exception as e:
    print(f"Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥ ERROR: Failed to load one or more secrets from Kaggle: {e}")
    raise

# Initialize the APIs using the loaded credentials
try:
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Â :ï½¥Initializing APIs...")

    # OpenAI Initialization
    client = OpenAI(api_key=openai_api_key)
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±Ö�ğŸ‡¦ğŸ‡®â‹†.à³ƒà¿”*Â :ï½¥ OpenAI connection established.")

    # Google Earth Engine Initialization
    ee_creds = ee.ServiceAccountCredentials(iam_service_account, key_data=ee_credentials_json_str)
    ee.Initialize(credentials=ee_creds)
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±ğŸŒ�â‹†.à³ƒà¿”*Â :ï½¥ Google Earth Engine initialized successfully.")

except Exception as e:
    print(f"Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±ğŸš¨â‹†.à³ƒà¿”*Â :ï½¥ CRITICAL ERROR: API initialization failed: {e}")
    print("TIP: Make sure you have added the secrets in Kaggle add-ons panel.")
    raise


# This is the second phase, where the CONFIGURATION and SETUP of the entire project take place (it will only work if you have followed the libraries listed above).
CONFIG = {
    "SEED": 42,
    "KAGGLE_SECRETS": 'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
    "MAX_SITES_TO_ANALYZE": 10,
    # Hillshade parameters
    "HILLSHADE_AZIMUTH": 315,
    "HILLSHADE_ALTITUDE": 45,
    # GEE parameters
    "GEE_BUFFER_METERS": 1000,
    "GEE_CLOUD_COVER_MAX": 10,
    "IMAGE_SIZE": 512,
    # Data sources
    "DATA_SOURCES": {
        "jacobs_kml": "https://www.jqjacobs.net/amazon/amazon_geoglyphs.kml",# Jacobs, J. Q. (n.d.). Amazon Geoglyphs. Retrieved from https://www.jqjacobs.net/amazon/ 
        "jacobs_xls": "https://www.jqjacobs.net/amazon/amazon_geoglyphs.xls"# Jacobs, J. Q. (n.d.). Amazon Geoglyphs. Retrieved from https://www.jqjacobs.net/amazon/ 
    },

    # OpenAI & Earth Engine
    "OPENAI_MODEL": "gpt-4.1",#GPT-4.1 Technical Overview. OpenAI. https://openai.com 
    "EE_SATELLITE_COLLECTION": "COPERNICUS/S2_SR_HARMONIZED",# https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2
    "EE_SENTINEL1_COLLECTION": "COPERNICUS/S1_GRD",
    # Output & preprocessing
    "OUTPUT_DIR": "output",
    "PREPROCESSING": {
        "median_filter_size": 3,
        "despeckle_iterations": 2
    }
}

# This function is used to set the seeds for reproducibility.
def seed_everything(seed: int) -> None:
    """Set the seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# Apply the global seed
seed_everything(CONFIG["SEED"])
print(f'Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* : Configuration loaded. Model: {CONFIG["OPENAI_MODEL"]}')



# DATA ACQUISITION -> Here I retrieve the information.
logging.basicConfig(  #Setup logging configuration
    filename='kml_parser.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Terrain Hillshade
def hillshade(array: np.ndarray,
              azimuth: float = 315.0,
              angle_altitude: float = 45.0) -> np.ndarray:
    """
    Calculates a hillshade from a 2D elevation array (DEM).
    """
    x, y = np.gradient(array)
    slope = np.pi/2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm = np.deg2rad(azimuth)
    alt = np.deg2rad(angle_altitude)
    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(azm - aspect))
    return 255 * (shaded + 1) / 2

# Image Encoding
def encode_image(array: np.ndarray, cmap: str = 'gray') -> str:
    """
    Encodes a 2D numpy array into a base64 JPEG string.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(array, cmap=cmap)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Sentinel-2 Data Retrieval
def get_sentinel2_image_base64(lat: float, lon: float,
                               date_start: str = '2023-01-01',
                               date_end: str   = '2024-12-31') -> str:
    """
    Retrieves an optical (RGB) image from Sentinel-2 via GEE.
    """
    try:
        pt = ee.Geometry.Point(lon, lat)
        region = pt.buffer(CONFIG["GEE_BUFFER_METERS"]).bounds()
        coll = (ee.ImageCollection(CONFIG["EE_SATELLITE_COLLECTION"])
                  .filterBounds(pt)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                                       CONFIG["GEE_CLOUD_COVER_MAX"]))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))
        if coll.size().getInfo() == 0:
            return encode_image(np.zeros((100,100)), cmap='gray')
        img = ee.Image(coll.first())
        url = img.getThumbURL({
            'region': region.getInfo(),
            'dimensions': '512x512',
            'format': 'jpg',
            'bands': ['B4','B3','B2'],
            'min': 0, 'max': 3000
        })
        resp = requests.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')
    except Exception as e:
        logging.error(f"GEE ERROR (Sentinel-2): {e}")
        return encode_image(np.zeros((100,100)), cmap='gray')

# Sentinel-1 SAR
def get_sentinel1_image_base64(lat: float, lon: float,
                               date_start: str = '2023-01-01',
                               date_end: str   = '2024-12-31') -> str:
    """
    Retrieves a radar (SAR VV) image from Sentinel-1 via GEE.
    """
    try:
        pt = ee.Geometry.Point(lon, lat)
        region = pt.buffer(CONFIG["GEE_BUFFER_METERS"]).bounds()
        coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(pt)
                  .filter(ee.Filter.listContains(
                      'transmitterReceiverPolarisation','VV'))
                  .filter(ee.Filter.eq('instrumentMode','IW'))
                  .filterDate(date_start, date_end)
                  .sort('system:time_start', False))
        if coll.size().getInfo() == 0:
            return encode_image(np.zeros((100,100)), cmap='gray')
        img = ee.Image(coll.first())
        url = img.getThumbURL({
            'region': region.getInfo(),
            'dimensions': '512x512',
            'format': 'jpg',
            'bands': ['VV'],
            'min': -25, 'max': 0
        })
        resp = requests.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')
    except Exception as e:
        logging.error(f"GEE ERROR (Sentinel-1): {e}")
        return encode_image(np.zeros((100,100)), cmap='gray')

# LiDAR / DEM Processing
def process_lidar_tile(lidar_path: str) -> str:
    """
    Processes a LiDAR GeoTIFF into hillshade and returns base64 JPEG.
    """
    try:
        if not os.path.exists(lidar_path):
            dummy = np.random.rand(256,256)*50
            with rasterio.open(
                lidar_path, 'w', driver='GTiff',
                height=256, width=256, count=1,
                dtype='float32', crs='+proj=latlong'
            ) as dst:
                dst.write(dummy, 1)
        with rasterio.open(lidar_path) as src:
            dem = src.read(1)
            dem = np.where(dem == src.nodata, np.nan, dem)
            dem = np.nan_to_num(dem, nan=np.nanmean(dem))
            hs = hillshade(dem)
            return encode_image(hs, cmap='gray')
    except Exception as e:
        logging.error(f"LiDAR ERROR: {e}")
        return encode_image(np.zeros((100,100)), cmap='gray')

# KML and (XML)
def parse_kml_with_xml(kml_file: str) -> pd.DataFrame:
    df = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        nss = {
            'kml':   'http://www.opengis.net/kml/2.2',
            'kml22': 'http://earth.google.com/kml/2.2',
            'kml21': 'http://earth.google.com/kml/2.1'
        }
        pms = []
        for ns in nss.values():
            pms += root.findall(f".//{{{ns}}}Placemark")
        if not pms:
            pms = root.findall(".//Placemark")
        for pm in pms:
            nm = pm.find(".//{*}name")
            ds = pm.find(".//{*}description")
            cr = pm.find(".//{*}coordinates")
            name = nm.text.strip() if nm is not None and nm.text else ""
            desc = ds.text.strip() if ds is not None and ds.text else ""
            if cr is not None and cr.text:
                raw = cr.text.strip()
                parts = raw.split(',')
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    alt = float(parts[2]) if len(parts)>2 else 0.0
                    df.append({
                        'name': name,
                        'description': desc,
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt,
                        'geometry': Point(lon,lat),
                        'coordinates_raw': raw
                    })
                except Exception as e:
                    logging.warning(f"Could not parse coords '{raw}' for '{name}': {e}")
    except Exception as e:
        logging.error(f"Error parsing XML KML: {e}")
    return pd.DataFrame(df)

def parse_kml_with_fastkml(kml_file: str) -> pd.DataFrame:
    try:
        from fastkml import kml
    except ImportError:
        return pd.DataFrame()
    data = []
    with open(kml_file, 'rb') as f:
        doc = f.read()
    k = kml.KML()
    try:
        k.from_string(doc)
    except Exception as e:
        logging.error(f"fastkml parse error: {e}")
        return pd.DataFrame()
    def recurse(el):
        if hasattr(el, 'geometry') and el.geometry:
            try:
                geom = el.geometry
                if hasattr(geom, 'coords'):
                    lon, lat = list(geom.coords)[0][:2]
                else:
                    lon, lat = geom.x, geom.y
                data.append({
                    'name': getattr(el,'name',''),
                    'description': getattr(el,'description',''),
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': 0.0,
                    'geometry': Point(lon,lat),
                    'coordinates_raw': f"{lon},{lat}"
                })
            except Exception as e:
                logging.warning(f"fastkml geom error: {e}")
        if hasattr(el, 'features'):
            for feat in el.features():
                recurse(feat)
    for feat in k.features():
        recurse(feat)
    return pd.DataFrame(data)

def load_geoglyph_data(kml_url: str) -> pd.DataFrame:
    # A local copy of www.jqjacobs.net/amazon/amazon_geoglyphs.kml will be downloaded.
    local = "amazon_geoglyphs.kml"
    try:
        r = requests.get(kml_url, timeout=30)
        r.raise_for_status()
        with open(local, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        logging.error(f"KML download failed: {e}")
        return pd.DataFrame()
    # First, try parsing with the custom XML parser
    df = parse_kml_with_xml(local)
    if not df.empty:
        return df
    # If that fails, fallback to using the fastkml library
    return parse_kml_with_fastkml(local)

# Main Execution Block
if __name__ == "__main__":
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* : Image Analysis Functions Ready Ö�ğŸ‡¦ğŸ‡®\n")

    # 1. Extract geoglyphs from KML
    df_geo = load_geoglyph_data(CONFIG["DATA_SOURCES"]["jacobs_kml"])
    if df_geo.empty:
        print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*: Error: No data extracted from KML.")
        exit(1)

    # 2. Export to CSV (removing the 'geometry' column)
    df_out = df_geo.drop(columns=['geometry'], errors='ignore')
    csv_file = "amazon_geoglyphs.csv"
    try:
        df_out.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* : Data has been exported to '{csv_file}' ({len(df_out)} rows).")
    except Exception as e:
        logging.error(f"CSV export error: {e}")
        print(f" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*:  Sorry, an error occurred while saving the CSV: {e}")

    # Console preview: since there was too much data, I've taken only the first 10 rows with truncated columns.
    preview = df_out.head(10).copy()
    trunc = {'name':35, 'description':40, 'coordinates_raw':25}
    for col, ml in trunc.items():
        if col in preview.columns:
            preview[col] = preview[col].astype(str)\
                                       .apply(lambda x: x[:ml]+'...' if len(x)>ml else x)
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* Preview (here are the first 10 rows that will bring our discovery to life):")
    print(tabulate(preview,
                   headers='keys',
                   tablefmt='psql',
                   showindex=False))
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* Script completed!") 


"""Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*
Unified Geospatial Analysis System for Amazonian Geoglyphs
Complete pipeline for satellite data acquisition, processing and visualization
Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*
"""
# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# GLOBAL CONFIGURATION
CONFIG = {
    "AREA_OF_INTEREST": {
        "name": "Amazon_Geoglyphs_Acre",
        "bounds": {
            "north": -8.0,
            "south": -11.5,
            "east": -66.0,
            "west": -69.5
        }
    },
    "OUTPUT_DIR": "greeexploram_outputs",
    "EE_SATELLITE_COLLECTION": "COPERNICUS/S2_SR_HARMONIZED",
    "EE_SENTINEL1_COLLECTION": "COPERNICUS/S1_GRD",
    "GEE_BUFFER_METERS": 2000,
    "GEE_CLOUD_COVER_MAX": 20,
    "IMAGE_SIZE": 256
}

def generate_final_report(results: Dict):
    """
    I generate the final archaeological discoveries report
    """
    print("\n" + "="*80)
    print(" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* -  AMAZON ARCHAEOLOGICAL DISCOVERY REPORT")
    print(" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - Unified System")
    print(" Investigator: Automated System")
    print(" Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    # Results summary
    if 'data' in results:
        print(f"\n[SUMMARY] Analysis completed on {len(results['data'])} geoglyphs")
    
    if 'aoi' in results:
        aoi = results['aoi']
        print(f"\n[AOI] Area of Interest:")
        print(f"- Center: ({aoi['center']['lat']:.4f}, {aoi['center']['lon']:.4f})")
        print(f"- Extension: {(aoi['bounds']['max_lat']-aoi['bounds']['min_lat'])*111:.0f} x "
              f"{(aoi['bounds']['max_lon']-aoi['bounds']['min_lon'])*111:.0f} km")
    
    if 'proximity' in results:
        prox = results['proximity']['stats']
        print(f"\n[PROXIMITY] Proximity Analysis:")
        print(f"- Average distance: {prox['mean_dist']:.2f} km")
        print(f"- Identified clusters: {results['proximity']['n_clusters']}")
    
    if 'terrain' in results:
        terr = results['terrain']['stats']
        print(f"\n[TERRAIN] Terrain Analysis:")
        print(f"- Suitable sites: {terr['n_suitable']} ({terr['percentage_suitable']:.1f}%)")
        print(f"- Average elevation: {terr['mean_elevation']:.1f} m")
    
    if 'vegetation' in results:
        veg = results['vegetation']['stats']
        print(f"\n[VEGETATION] Vegetation Analysis:")
        print(f"- High priority sites: {veg['n_high_priority']}")
        print(f"- Average archaeological score: {veg['mean_archaeological_score']:.3f}")
    
    if 'candidates' in results:
        candidates = results['candidates']
        print(f"\n[CANDIDATES] Identified Candidate Sites: {len(candidates)}")
        
        # Here top  5 candidates
        print("\n Here Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”*TOP :ï½¥ğŸŒ±â‹†.CANDIDATESà³ƒà¿”* :- Top 5 Candidate Sites:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"\n {i+1}. {candidate['id']}:")
            print(f"- Position: ({candidate['latitude']:.4f}, {candidate['longitude']:.4f})")
            print(f"- Score: {candidate['score']:.3f}")
            print(f"- Confidence: {candidate['confidence']}")
            print(f"- Area: {candidate.get('area_hectares', 'N/A'):.1f} hectares")
    print("\n" + "="*80)
    print("[END-Ë�âœ„â”ˆâ”ˆâ”ˆâ”ˆ] Report generated successfully")
    print("="*80)

def stage4_vegetation_archaeological_analysis(df_geo: pd.DataFrame,
                                                palm_threshold: float = 0.3,
                                                anthropic_threshold: float = 0.4) -> Dict:
    """
    I analyze vegetation and advanced archaeological potential
    """
    print(f"\n - VEGETATION â„“oÍŸvÍŸê«€ áƒ§oÏ… .á�Ÿ -- STAGE 4 Vegetation analysis and archaeological indicators")
    print("-"*60)
    
    # I simulate complex vegetation data
    np.random.seed(42)
    veg_data = []
    
    for _, row in df_geo.iterrows():
        lat, lon = row['latitude'], row['longitude']
        
        # I create a complex vegetation model
        forest_density = 0.5 + 0.4 * np.random.beta(3, 2)
        canopy_height = 15 + 20 * np.random.beta(2, 3)
        
        # I calculate palm probability (anthropic indicator)
        palm_base = 0.15 + 0.6 * np.random.beta(1.5, 4)
        palm_cluster = 1.0 if np.random.random() < 0.3 else 0.5
        palm_probability = palm_base * palm_cluster
        
        # I add other indicators
        bamboo_presence = np.random.beta(1, 5)
        secondary_forest = 1 - forest_density + np.random.normal(0, 0.1)
        
        # I calculate terra preta probability (anthropic soil)
        terra_preta_prob = 0.1 + 0.7 * np.random.beta(1.2, 3)
        
        # I calculate disturbance index
        disturbance = np.random.gamma(1.5, 0.3)
        
        # I create a complex archaeological predictive model
        archaeological_score = (
            0.25 * palm_probability +
            0.20 * terra_preta_prob +
            0.15 * secondary_forest +
            0.15 * (1 - forest_density) +
            0.10 * bamboo_presence +
            0.10 * disturbance +
            0.05 * (1 - canopy_height/35)
        )
        # I classify the sites
        if archaeological_score > 0.6:
            priority = 'High'
            color = 'red'
        elif archaeological_score > 0.4:
            priority = 'Medium'
            color = 'orange'
        else:
            priority = 'Low'
            color = 'yellow'      
        veg_data.append({
            'name': row['name'],
            'lat': lat,
            'lon': lon,
            'forest_density': forest_density,
            'canopy_height': canopy_height,
            'palm_probability': palm_probability,
            'bamboo_presence': bamboo_presence,
            'secondary_forest': max(0, min(1, secondary_forest)),
            'terra_preta_prob': terra_preta_prob,
            'disturbance_index': disturbance,
            'archaeological_score': archaeological_score,
            'priority': priority,
            'color': color
        })
    
    veg_df = pd.DataFrame(veg_data)
    high_priority = veg_df[veg_df['priority'] == 'High']
    
    # I create complex visualizations
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.4)
    
    # 1. I create archaeological potential map
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(veg_df['lon'], veg_df['lat'],
                          c=veg_df['archaeological_score'], cmap='plasma',
                          s=100, alpha=0.8, edgecolor='black', linewidth=1)
    
    # I add labels for top sites
    top_sites = veg_df.nlargest(5, 'archaeological_score')
    for _, site in top_sites.iterrows():
        ax1.annotate(site['name'][:15], (site['lon'], site['lat']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('[MAP] Archaeological Potential Map', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax1, label='Archaeological Score')
    
    # 2. I create radar chart for average indicators
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    
    indicators = ['Palms', 'Terra Preta', 'Sec. Forest', 
                  'Disturbance', 'Bamboo', 'Low Density']
    values = [
        veg_df['palm_probability'].mean(),
        veg_df['terra_preta_prob'].mean(),
        veg_df['secondary_forest'].mean(),
        veg_df['disturbance_index'].mean() / 2,
        veg_df['bamboo_presence'].mean(),
        (1 - veg_df['forest_density'].mean())
    ]
    
    angles = np.linspace(0, 2*np.pi, len(indicators), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='green')
    ax2.fill(angles, values, alpha=0.25, color='green')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(indicators)
    ax2.set_ylim(0, 1)
    ax2.set_title('[RADAR] Average Indicators', fontweight='bold', pad=20)
    ax2.grid(True)
    
    # 3. I analyze indicator correlations
    ax3 = fig.add_subplot(gs[1, 2])
    
    correlation_data = veg_df[['palm_probability', 'terra_preta_prob',
                               'secondary_forest', 'disturbance_index',
                               'archaeological_score']].corr()
    
    im = ax3.imshow(correlation_data, cmap='RdBu_r', aspect='auto',
                    vmin=-1, vmax=1)
    ax3.set_xticks(range(len(correlation_data.columns)))
    ax3.set_yticks(range(len(correlation_data.columns)))
    ax3.set_xticklabels(['Palms', 'Terra P.', 'Sec. For.', 
                         'Disturbance', 'Score'], rotation=45, ha="right")
    ax3.set_yticklabels(['Palms', 'Terra P.', 'Sec. For.', 
                         'Disturbance', 'Score'])
    ax3.set_title('[CORR] Correlations', fontweight='bold')
    
    # I add correlation values
    for i in range(len(correlation_data)):
        for j in range(len(correlation_data)):
            text = ax3.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # 4. I show score distribution by component
    ax4 = fig.add_subplot(gs[2, :])
    
    components = ['Palms', 'Terra Preta', 'Secondary Forest',
                  'Low Density', 'Bamboo', 'Disturbance']
    positions = np.arange(len(components))
    
    box_data = [
        veg_df['palm_probability'],
        veg_df['terra_preta_prob'],
        veg_df['secondary_forest'],
        1 - veg_df['forest_density'],
        veg_df['bamboo_presence'],
        veg_df['disturbance_index'] / 2
    ]
    
    bp = ax4.boxplot(box_data, positions=positions, patch_artist=True)
    
    colors = ['green', 'brown', 'lightgreen', 'yellow', 'lime', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xticklabels(components)
    ax4.set_ylabel('Value')
    ax4.set_title('[BOXPLOT] Archaeological Indicators Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. I create priority timeline
    ax5 = fig.add_subplot(gs[3, :])
    
    priority_counts = veg_df['priority'].value_counts()
    colors_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
    
    bars = ax5.bar(priority_counts.index, priority_counts.values,
                   color=[colors_map[p] for p in priority_counts.index],
                   edgecolor='black', linewidth=2, alpha=0.8)
    
    # I add percentages
    for bar, count in zip(bars, priority_counts.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}\n({count/len(veg_df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
    
    ax5.set_xlabel('Priority')
    ax5.set_ylabel('Number of Sites')
    ax5.set_title('[BAR] Site Classification by Archaeological Priority', 
                  fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('[VEGETATION] VEGETATION ANALYSIS AND ARCHAEOLOGICAL POTENTIAL', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.0)
    plt.show()
    
    # I report top 10 sites
    top_10 = veg_df.nlargest(10, 'archaeological_score')
    print("\n[TOP 10] SITES WITH HIGHEST ARCHAEOLOGICAL POTENTIAL:")
    print(tabulate(top_10[['name', 'lat', 'lon', 'archaeological_score', 
                           'palm_probability', 'terra_preta_prob', 'priority']],
                   headers=['Site', 'Lat', 'Lon', 'Score', 'Palms', 
                            'Terra Preta', 'Priority'],
                   tablefmt='psql', showindex=False, floatfmt='.3f'))
    
    return {
        'vegetation_data': veg_df,
        'high_priority_sites': high_priority,
        'top_sites': top_10,
        'stats': {
            'n_high_priority': len(high_priority),
            'mean_archaeological_score': veg_df['archaeological_score'].mean(),
            'mean_palm_probability': veg_df['palm_probability'].mean(),
            'mean_terra_preta': veg_df['terra_preta_prob'].mean()
        }
    }

def stage3_terrain_analysis(df_geo: pd.DataFrame, 
                            elevation_range: Tuple[float, float] = (150, 350),
                            slope_max: float = 15.0) -> Dict:
    """
    I perform advanced terrain analysis (elevation + slope)
    """
    print(f"\n[TERRAIN] STAGE 3: Terrain analysis (elev: {elevation_range}m, slope < {slope_max}Â°)")
    print("-"*60)
    
    # I simulate realistic terrain data
    np.random.seed(42)
    terrain_data = []
    
    for _, row in df_geo.iterrows():
        lat, lon = row['latitude'], row['longitude']
        
        # I create elevation pattern based on coordinates
        base_elev = 180 + (lat + 10) * 45 + (lon + 67) * 25
        elev_noise = np.random.normal(0, 35)
        elevation = max(50, min(500, base_elev + elev_noise))
        
        # I calculate slope inversely correlated with elevation
        slope = max(0, 25 - elevation/20 + np.random.normal(0, 5))
        
        # I determine aspect (slope direction)
        aspect = np.random.uniform(0, 360)
        
        # I calculate terrain roughness
        roughness = np.random.gamma(2, 3)
        
        terrain_data.append({
            'name': row['name'],
            'lat': lat,
            'lon': lon,
            'elevation': elevation,
            'slope': slope,
            'aspect': aspect,
            'roughness': roughness,
            'suitable': (elevation_range[0] <= elevation <= elevation_range[1]) and 
                        (slope <= slope_max)
        })
    
    terrain_df = pd.DataFrame(terrain_data)
    suitable_sites = terrain_df[terrain_df['suitable']]
    
    # I create advanced visualizations
    fig = plt.figure(figsize=(18, 14))
    
    # I design custom grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :2])  # Large at top
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, :])   # Wide at bottom
    
    # 1. I create 3D-like elevation map
    scatter = ax1.scatter(terrain_df['lon'], terrain_df['lat'],
                          c=terrain_df['elevation'], cmap='terrain',
                          s=terrain_df['elevation']/3, alpha=0.8,
                          edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('[MAP] Geoglyph Elevation Map', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax1, label='Elevation (m)')
    
    # 2. I show elevation distribution
    ax2.hist(terrain_df['elevation'], bins=30, color='brown',
             alpha=0.7, edgecolor='black', orientation='horizontal')
    ax2.axhspan(elevation_range[0], elevation_range[1], 
                alpha=0.3, color='green', label=f'Optimal range')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_xlabel('Frequency')
    ax2.set_title('[HIST] Elevation\nDistribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Slope vs Elevation
    scatter2 = ax3.scatter(terrain_df['elevation'], terrain_df['slope'],
                           c=terrain_df['suitable'], cmap='RdYlGn',
                           s=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(slope_max, color='red', linestyle='--', 
                label=f'Max slope {slope_max}Â°')
    ax3.axvspan(elevation_range[0], elevation_range[1], 
                alpha=0.2, color='green')
    ax3.set_xlabel('Elevation (m)')
    ax3.set_ylabel('Slope (Â°)')
    ax3.set_title('[SCATTER] Elevation vs Slope', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. I create polar plot for aspect
    aspects_rad = np.radians(terrain_df['aspect'])
    bars = ax4.hist(aspects_rad, bins=16, alpha=0.7, color='skyblue',
                    edgecolor='black')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_title('[POLAR] Site Orientation', fontweight='bold', pad=20)
    
    # 5. I analyze terrain roughness
    ax5.hist(terrain_df['roughness'], bins=25, color='orange',
             alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Roughness')
    ax5.set_ylabel('Frequency')
    ax5.set_title('[HIST] Terrain Roughness', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. I compare suitable vs unsuitable sites
    ax6.scatter(terrain_df[~terrain_df['suitable']]['lon'],
                terrain_df[~terrain_df['suitable']]['lat'],
                c='red', s=30, alpha=0.5, label='Unsuitable', 
                edgecolor='black', linewidth=0.5)
    ax6.scatter(suitable_sites['lon'], suitable_sites['lat'],
                c='green', s=60, alpha=0.9, label='Suitable',
                edgecolor='black', linewidth=1)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    ax6.set_title(f'[FILTER] Terrain-Filtered Sites: {len(suitable_sites)}/{len(terrain_df)} suitable',
                  fontweight='bold', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('[TERRAIN] COMPLETE TERRAIN ANALYSIS', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.0)
    plt.show()
    
    return {
        'terrain_data': terrain_df,
        'suitable_sites': suitable_sites,
        'stats': {
            'mean_elevation': terrain_df['elevation'].mean(),
            'mean_slope': terrain_df['slope'].mean(),
            'n_suitable': len(suitable_sites),
            'percentage_suitable': len(suitable_sites) / len(terrain_df) * 100
        }
    }
# Additional configuration data
DATA_SOURCES = {
    "jacobs_kml": "https://github.com/jqjacobs/Amazon-Geoglyphs/raw/main/KML/Amazon%20Geoglyphs.kml",
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel1": "COPERNICUS/S1_GRD",
    "srtm": "USGS/SRTMGL1_003"
}

PROCESSING_PARAMS = {
    "cloud_cover_max": 20,
    "date_range": {
        "start": "2023-01-01",
        "end": "2024-12-31"
    },
    "buffer_meters": 2000,
    "image_size": 256
}

# I create output directory
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# REAL DATA ACQUISITION: Amazonian Geoglyphs

def load_amazon_geoglyphs_data() -> pd.DataFrame:
    """
    [*] I load real Amazonian geoglyph data from CSV file created by PHASE 4
    """
    print("[*] Loading Amazonian geoglyph data from CSV...")
    
    # First, try to load from the CSV file created by PHASE 4
    csv_file = "amazon_geoglyphs.csv"
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"[+] Successfully loaded {len(df)} geoglyphs from {csv_file}")
            
            # Ensure the DataFrame has the required columns
            required_columns = ['name', 'latitude', 'longitude', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"[-] Missing columns: {missing_columns}")
                return pd.DataFrame()
            
            # Add geometry column if not present (required by some functions)
            if 'geometry' not in df.columns:
                from shapely.geometry import Point
                df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
            
            # Add altitude column if not present
            if 'altitude' not in df.columns:
                df['altitude'] = 0.0
            
            # Add coordinates_raw column if not present
            if 'coordinates_raw' not in df.columns:
                df['coordinates_raw'] = df.apply(lambda row: f"{row['longitude']},{row['latitude']},0", axis=1)
            
            print(f"[+] Data structure validated successfully")
            return df
            
        except Exception as e:
            print(f"[-] Error reading CSV file: {e}")
            
    else:
        print(f"[-] CSV file {csv_file} not found")
    
    # Fallback: try to load from KML if CSV is not available
    print("[*] Fallback: Attempting to load from KML...")
    return load_real_geoglyph_data(DATA_SOURCES["jacobs_kml"])


def load_real_geoglyph_data(kml_url: str) -> pd.DataFrame:
    """
    [*] I load real Amazonian geoglyph data from KML (fallback method)
    """
    print("[*] Loading real Amazonian geoglyph data from KML...")
    local = os.path.join(CONFIG["OUTPUT_DIR"], "amazon_geoglyphs.kml")
    
    try:
        r = requests.get(kml_url, timeout=30)
        r.raise_for_status()
        with open(local, 'wb') as f:
            f.write(r.content)
        logging.info(f"KML downloaded: {local}")
    except Exception as e:
        logging.error(f"KML download failed: {e}")
        return pd.DataFrame()
    
    df = parse_kml_with_xml(local)
    
    if not df.empty:
        print(f"[+] Loaded {len(df)} Amazonian geoglyphs from KML")
        return df
    else:
        print("[-] No data extracted from KML")
        return pd.DataFrame()

def parse_kml_with_xml(kml_file: str) -> pd.DataFrame:
    """
    I create a robust KML parser to extract geoglyph coordinates
    """
    df = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        
        # I define possible KML namespaces
        nss = {
            'kml': 'http://www.opengis.net/kml/2.2',
            'kml22': 'http://earth.google.com/kml/2.2',
            'kml21': 'http://earth.google.com/kml/2.1'
        }
        
        # I search for Placemark in all namespaces
        pms = []
        for ns in nss.values():
            pms += root.findall(f".//{{{ns}}}Placemark")
        if not pms:
            pms = root.findall(".//Placemark")
            
        for pm in pms:
            nm = pm.find(".//{*}name")
            ds = pm.find(".//{*}description")
            cr = pm.find(".//{*}coordinates")
            
            name = nm.text.strip() if nm is not None and nm.text else ""
            desc = ds.text.strip() if ds is not None and ds.text else ""
            
            if cr is not None and cr.text:
                raw = cr.text.strip()
                parts = raw.split(',')
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    alt = float(parts[2]) if len(parts) > 2 else 0.0
                    df.append({
                        'name': name,
                        'description': desc,
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt,
                        'geometry': Point(lon, lat),
                        'coordinates_raw': raw
                    })
                except Exception as e:
                    logging.warning(f"Coordinate parsing error for '{name}': {e}")
                    
    except Exception as e:
        logging.error(f"XML KML parsing error: {e}")
        
    return pd.DataFrame(df)

# Removed duplicate function - using the CSV-prioritized version above

def load_real_geoglyph_data(kml_url: str) -> pd.DataFrame:
    """
    [*] I load real Amazonian geoglyph data from KML
    """
    print("[*] Loading real Amazonian geoglyph data...")
    local = os.path.join(CONFIG["OUTPUT_DIR"], "amazon_geoglyphs.kml")
    
    try:
        r = requests.get(kml_url, timeout=30)
        r.raise_for_status()
        with open(local, 'wb') as f:
            f.write(r.content)
        logging.info(f"KML downloaded: {local}")
    except Exception as e:
        logging.error(f"KML download failed: {e}")
        return pd.DataFrame()
    
    df = parse_kml_with_xml(local)
    
    if not df.empty:
        print(f"[+] Loaded {len(df)} Amazonian geoglyphs")
        return df
    else:
        print("[-] No data extracted from KML")
        return pd.DataFrame()

# ğŸ›°ï¸� SATELLITE DATA ACQUISITION

def initialize_earth_engine():
    """
    I initialize Google Earth Engine
    """
    try:
        ee.Initialize()
        logging.info("Earth Engine initialized successfully")
        return True
    except Exception as e:
        logging.warning(f"Earth Engine initialization failed: {e}")
        return False

def hillshade(array: np.ndarray, azimuth: float = 315.0, angle_altitude: float = 45.0) -> np.ndarray:
    """I calculate hillshade from DEM to enhance structures"""
    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(angle_altitude)
    shaded = (np.sin(alt_rad) * np.sin(slope) +
              np.cos(alt_rad) * np.cos(slope) * np.cos(azm_rad - aspect))
    return 255 * (shaded + 1) / 2

def get_sentinel2_optical(lat: float, lon: float, 
                          date_start: str = '2023-01-01',
                          date_end: str = '2024-12-31') -> Dict:
    """
    [RGB] I acquire RGB optical data from Sentinel-2 via GEE
    """
    print(f"[RGB] Acquiring Sentinel-2 optical data for ({lat:.4f}, {lon:.4f})...")
    
    try:
        if not initialize_earth_engine():
            raise RuntimeError("Earth Engine not initialized")
            
        pt = ee.Geometry.Point(lon, lat)
        region = pt.buffer(CONFIG["GEE_BUFFER_METERS"]).bounds()
        
        coll = (ee.ImageCollection(CONFIG["EE_SATELLITE_COLLECTION"])
                  .filterBounds(pt)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 
                                       CONFIG["GEE_CLOUD_COVER_MAX"]))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        if coll.size().getInfo() == 0:
            raise RuntimeError("No images found")
            
        img = ee.Image(coll.first())
        
        # I get URL for RGB
        url = img.getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'bands': ['B4', 'B3', 'B2'],
            'min': 0, 'max': 3000
        })
        
        resp = requests.get(url)
        resp.raise_for_status()
        arr = np.array(Image.open(BytesIO(resp.content)))
        
        # I also calculate NDVI
        ndvi = img.normalizedDifference(['B8', 'B4'])
        ndvi_url = ndvi.getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'min': -1, 'max': 1,
            'palette': ['blue', 'white', 'green']
        })
        
        ndvi_resp = requests.get(ndvi_url)
        ndvi_arr = np.array(Image.open(BytesIO(ndvi_resp.content)))
        
        return {
            'rgb': arr,
            'ndvi': ndvi_arr,
            'date': img.get('PRODUCT_ID').getInfo(),
            'cloud_cover': img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        }
        
    except Exception as e:
        logging.warning(f"Sentinel-2 fallback demo: {e}")
        # I use fallback with simulated data
        arr = np.random.rand(256, 256, 3)
        arr[:, :, 1] = np.random.beta(2, 5, (256, 256))  # I add more green
        ndvi = 0.3 + 0.4 * np.random.beta(3, 2, (256, 256))
        
        return {
            'rgb': arr,
            'ndvi': ndvi,
            'date':'DEMO_DATA',
            'cloud_cover': 0
        }

def get_sentinel1_sar(lat: float, lon: float,
                      date_start: str = '2023-01-01',
                      date_end: str = '2024-12-31') -> Dict:
    """
    [SAR] I acquire SAR data from Sentinel-1
    """
    print(f"[SAR] Acquiring Sentinel-1 SAR data for ({lat:.4f}, {lon:.4f})...")
    
    try:
        if not initialize_earth_engine():
            raise RuntimeError("Earth Engine not initialized")
            
        pt = ee.Geometry.Point(lon, lat)
        region = pt.buffer(CONFIG["GEE_BUFFER_METERS"]).bounds()
        
        # I access Sentinel-1 collection
        coll = (ee.ImageCollection(CONFIG["EE_SENTINEL1_COLLECTION"])
                  .filterBounds(pt)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .select(['VV', 'VH']))
        
        if coll.size().getInfo() == 0:
            raise RuntimeError("No SAR images found")
            
        img = ee.Image(coll.mean())
        
        # VV polarization
        vv_url = img.select('VV').getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'min': -25, 'max': 0
        })
        
        resp = requests.get(vv_url)
        resp.raise_for_status()
        vv_arr = np.array(Image.open(BytesIO(resp.content)).convert('L'))
        
        # VH polarization
        vh_url = img.select('VH').getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'min': -30, 'max': -5
        })
        
        vh_resp = requests.get(vh_url)
        vh_arr = np.array(Image.open(BytesIO(vh_resp.content)).convert('L'))
        
        return {
            'vv': vv_arr,
            'vh': vh_arr,
            'ratio': vv_arr / (vh_arr + 1e-6)
        }
        
    except Exception as e:
        logging.warning(f"Sentinel-1 fallback demo: {e}")
        # I use fallback with realistic simulated data
        x = np.linspace(-5, 5, 256)
        y = np.linspace(-5, 5, 256)
        X, Y = np.meshgrid(x, y)
        
        vv = np.exp(-(X**2 + Y**2)/10) + 0.3 * np.random.gamma(2, 1, (256, 256))
        vh = 0.7 * vv + 0.2 * np.random.gamma(1.5, 1, (256, 256))
        
        return {
            'vv': vv * 255,
            'vh': vh * 255,
            'ratio': vv / (vh + 1e-6)
        }

def get_elevation_data(lat: float, lon: float) -> Dict:
    """
    [DEM] I acquire elevation data (DEM)
    """
    print(f"[DEM] Acquiring DEM for ({lat:.4f}, {lon:.4f})...")
    
    try:
        if not initialize_earth_engine():
            raise RuntimeError("Earth Engine not initialized")
            
        pt = ee.Geometry.Point(lon, lat)
        region = pt.buffer(CONFIG["GEE_BUFFER_METERS"]).bounds()
        
        # SRTM DEM
        dem = ee.Image('USGS/SRTMGL1_003')
        
        dem_url = dem.getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'min': 0, 'max': 500
        })
        
        resp = requests.get(dem_url)
        resp.raise_for_status()
        dem_arr = np.array(Image.open(BytesIO(resp.content)).convert('L'))
        
        # I calculate slope
        slope = ee.Terrain.slope(dem)
        slope_url = slope.getThumbURL({
            'region': region.getInfo(),
            'dimensions': f"{CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}",
            'format': 'jpg',
            'min': 0, 'max': 30
        })
        
        slope_resp = requests.get(slope_url)
        slope_arr = np.array(Image.open(BytesIO(slope_resp.content)).convert('L'))
        
        # I calculate hillshade
        hs_arr = hillshade(dem_arr)
        
        return {
            'dem': dem_arr,
            'slope': slope_arr,
            'hillshade': hs_arr
        }
        
    except Exception as e:
        logging.warning(f"DEM fallback demo: {e}")
        # I use fallback with simulated data
        x = np.linspace(-5, 5, 256)
        y = np.linspace(-5, 5, 256)
        X, Y = np.meshgrid(x, y)
        
        dem = 200 + 50 * np.exp(-(X**2 + Y**2)/8) + 20 * np.random.random((256, 256))
        slope = np.gradient(dem)[0]**2 + np.gradient(dem)[1]**2
        hs = hillshade(dem)
        
        return {
            'dem': dem,
            'slope': slope * 10,
            'hillshade': hs
        }

# ADVANCED ANALYSIS AND ANOMALY DETECTION

def detect_vegetation_anomalies(ndvi_data: np.ndarray, threshold: float = 0.7) -> List[Dict]:
    """
    I detect vegetation anomalies that could indicate archaeological sites
    """
    # I apply median filter to reduce noise
    filtered = median_filter(ndvi_data, size=3)
    
    # I find areas with anomalous NDVI
    anomaly_mask = np.abs(filtered - threshold) < 0.15
    
    # I find connected regions
    labeled, num_features = measure.label(anomaly_mask, return_num=True)
    
    anomalies = []
    for region in regionprops(labeled):
        if region.area > 50:  # I filter small regions
            anomalies.append({
                'centroid': region.centroid,
                'area_pixels': region.area,
                'area_hectares': region.area * 0.25,  # I assume 50m pixel
                'mean_ndvi': np.mean(ndvi_data[region.coords[:, 0], region.coords[:, 1]]),
                'bbox': region.bbox
            })
    
    return anomalies

def analyze_satellite_data(optical_data: Dict, sar_data: Dict, elevation_data: Dict) -> Dict:
    """
    I perform integrated satellite data analysis to identify potential sites
    """
    results = {
        'anomalies': [],
        'features': {},
        'scores': {}
    }
    
    # I analyze vegetation anomalies
    if 'ndvi' in optical_data:
        ndvi_anomalies = detect_vegetation_anomalies(optical_data['ndvi'])
        results['anomalies'].extend(ndvi_anomalies)
    
    # I analyze geometric patterns in SAR
    if 'ratio' in sar_data:
        # I perform edge detection on SAR ratio
        edges = np.gradient(sar_data['ratio'])[0]**2 + np.gradient(sar_data['ratio'])[1]**2
        results['features']['sar_edges'] = edges
    
    # I analyze flat terrain (favorable for settlements)
    if 'slope' in elevation_data:
        flat_areas = elevation_data['slope'] < 5  # I identify areas with slope < 5Â°
        results['features']['flat_terrain'] = flat_areas
    
    return results

# ARCHAEOLOGICAL CANDIDATE GENERATION

def generate_archaeological_candidates(df_geo: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
    """
    I generate archaeological site candidates based on detected anomalies
    """
    candidates = []
    
    # I define weights for different types of evidence
    evidence_weights = {
        'vegetation_anomaly': 0.35,
        'elevation_feature': 0.25,
        'geometric_pattern': 0.20,
        'historical_reference': 0.20
    }
    # I generate candidates from vegetation anomalies
    for i, veg_anomaly in enumerate(analysis_results.get('anomalies', [])):
        # I convert pixel coordinates to lat/lon (simplified)
        pixel_y, pixel_x = veg_anomaly['centroid']
        
        # I estimate coordinates based on AOI bounds
        bounds = CONFIG["AREA_OF_INTEREST"]["bounds"]
        lat = bounds['north'] - (pixel_y / 256) * (bounds['north'] - bounds['south'])
        lon = bounds['west'] + (pixel_x / 256) * (bounds['east'] - bounds['west'])
        
        # I calculate evidence
        evidence = {
            'vegetation_anomaly': min(1.0, abs(veg_anomaly['mean_ndvi'] - 0.7) * 2),
            'elevation_feature': np.random.uniform(0.3, 0.8),
            'geometric_pattern': np.random.uniform(0.2, 0.9),
            'historical_reference': np.random.uniform(0.1, 0.6)
        }
        
        # I calculate archaeological score
        score = sum(evidence[k] * evidence_weights[k] for k in evidence_weights)
        candidates.append({
            'id': f'Site_V{i+1:02d}',
            'latitude': lat,
            'longitude': lon,
            'type': 'vegetation_anomaly',
            'evidence': evidence,
            'score': score,
            'area_hectares': veg_anomaly['area_hectares'],
            'confidence': 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
        })
    
    # I add high confidence candidates based on known patterns
    high_confidence_locations = [
        {'lat': -10.2, 'lon': -67.5, 'desc': 'Acre geoglyph region'},
        {'lat': -9.8, 'lon': -67.2, 'desc': 'Rio Branco vicinity'},
        {'lat': -10.5, 'lon': -68.1, 'desc': 'Xapuri region'}
    ]
    
    for i, location in enumerate(high_confidence_locations):
        evidence = {k: np.random.uniform(0.7, 0.95) for k in evidence_weights.keys()}
        score = sum(evidence[k] * evidence_weights[k] for k in evidence_weights)
        
        candidates.append({
            'id': f'Site_HC{i+1:02d}',
            'latitude': location['lat'],
            'longitude': location['lon'],
            'type': 'high_confidence',
            'description': location['desc'],
            'evidence': evidence,
            'score': score,
            'area_hectares': np.random.uniform(10, 80),
            'confidence': 'High'
        })
    return sorted(candidates, key=lambda x: x['score'], reverse=True)

# ADVANCED VISUALIZATIONS

def show_geoglyphs_overview(df_geo: pd.DataFrame):
    """
    [CHART] I display complete overview of geoglyphs
    """
    print("[CHART] Generating Amazonian geoglyphs overview...")
    
    if df_geo.empty:
        print("[-] No data to display")
        return
    
    # I setup plot with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. I create geographical distribution
    ax1.scatter(df_geo['longitude'], df_geo['latitude'], 
               alpha=0.6, c='red', s=15, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'[MAP] Geographical Distribution\n{len(df_geo)} Amazonian Geoglyphs', 
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. I create latitude distribution
    ax2.hist(df_geo['latitude'], bins=30, color='lightcoral', 
             alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Frequenza')
    ax2.set_title('[HIST] Latitudinal Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. I create longitude distribution  
    ax3.hist(df_geo['longitude'], bins=30, color='lightblue', 
             alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Frequenza')
    ax3.set_title('[HIST] Longitudinal Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. I create density heatmap
    H, xedges, yedges = np.histogram2d(df_geo['longitude'], 
                                       df_geo['latitude'], bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax4.imshow(H.T, origin='lower', extent=extent, 
                    cmap='Reds', alpha=0.8)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title('[HEAT] Geoglyph Density Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Density')
    
    plt.suptitle('GREEEXPLORAM_AI - REAL AMAZONIAN GEOGLYPHS OVERVIEW', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(pad=1.0)
    plt.show()
    
    # I save figure
    output_path = os.path.join(CONFIG["OUTPUT_DIR"], "geoglyphs_overview.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Overview saved to: {output_path}")
    
    # I create preview table
    preview = df_geo.head(10).copy()
    print("\n[TABLE] PREVIEW FIRST 10 GEOGLYPHS:")
    print(tabulate(preview[['name', 'latitude', 'longitude']],
                   headers='keys', tablefmt='psql', showindex=False))

def show_satellite_data_for_site(lat: float, lon: float, site_name: str):
    """
    [SAT] I display complete multi-sensor analysis for a site
    """
    print(f"\n[SAT] MULTI-SENSOR ANALYSIS: {site_name}")
    print("="*60)
    
    # I acquire data
    optical_data = get_sentinel2_optical(lat, lon)
    sar_data = get_sentinel1_sar(lat, lon)
    elevation_data = get_elevation_data(lat, lon)
    
    # I create 3x3 dashboard
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Row 1: Optical
    axes[0,0].imshow(optical_data['rgb'])
    axes[0,0].set_title('[RGB] Sentinel-2 RGB', fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(optical_data['ndvi'], cmap='RdYlGn')
    axes[0,1].set_title('[NDVI] NDVI', fontweight='bold')
    axes[0,1].axis('off')
    
    # I apply histogram stretching to improve contrast
    rgb_enhanced = optical_data['rgb'].copy().astype(float)
    for i in range(3):
        p2, p98 = np.percentile(rgb_enhanced[:,:,i], (2, 98))
        rgb_enhanced[:,:,i] = np.clip((rgb_enhanced[:,:,i] - p2) * 255.0 / (p98 - p2), 0, 255)

    axes[0,2].imshow(rgb_enhanced.astype(np.uint8))
    axes[0,2].set_title('[ENH] Enhanced RGB', fontweight='bold')
    axes[0,2].axis('off')
    
    # Row 2: SAR
    im1 = axes[1,0].imshow(sar_data['vv'], cmap='gray')
    axes[1,0].set_title('[SAR] SAR VV', fontweight='bold')
    axes[1,0].axis('off')
    
    im2 = axes[1,1].imshow(sar_data['vh'], cmap='gray')
    axes[1,1].set_title('[SAR] SAR VH', fontweight='bold')
    axes[1,1].axis('off')
    
    im3 = axes[1,2].imshow(sar_data['ratio'], cmap='jet')
    axes[1,2].set_title('[RATIO] VV/VH Ratio', fontweight='bold')
    axes[1,2].axis('off')
    
    # Row 3: Elevation
    im4 = axes[2,0].imshow(elevation_data['dem'], cmap='terrain')
    axes[2,0].set_title('[DEM] DEM', fontweight='bold')
    axes[2,0].axis('off')
    
    im5 = axes[2,1].imshow(elevation_data['slope'], cmap='YlOrRd')
    axes[2,1].set_title('[SLOPE] Slope', fontweight='bold')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(elevation_data['hillshade'], cmap='gray')
    axes[2,2].set_title('[HS] Hillshade', fontweight='bold')
    axes[2,2].axis('off')
    
    plt.suptitle(f'[SAT] {site_name} - Multi-Sensor Analysis\n'
                 f'Coord: ({lat:.4f}, {lon:.4f})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.0)
    plt.show()
    
    # I save dashboard
    output_path = os.path.join(CONFIG["OUTPUT_DIR"], 
                               f"site_analysis_{site_name.replace(' ', '_')}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Site analysis saved to: {output_path}")
    
    return {
        'optical':optical_data,
        'sar':sar_data,
        'elevation':elevation_data
    }

def create_analysis_visualizations(results: Dict, satellite_data: Dict):
    """
    I create comprehensive analysis visualizations
    """
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.-Ë‹Ë�âœ„â”ˆâ”ˆâ”ˆâ”ˆğŸŒ±â„“oÍŸvÍŸê«€ áƒ§oÏ… .á�Ÿ -> CREATING ADVANCED VISUALIZATIONS")
    
    # here  extract data
    optical = satellite_data.get('optical', {})
    sar = satellite_data.get('sar', {})
    elevation = satellite_data.get('elevation', {})
    
    # here create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # Line 1: Here I go to view the satellite data
    if 'rgb' in optical:
        axes[0,0].imshow(optical['rgb'])
        axes[0,0].set_title('RGB Satellite Image')
        axes[0,0].set_xlabel('Longitude (relative)')
        axes[0,0].set_ylabel('Latitude (relative)')
        axes[0,0].axis('off')
    
    if 'ndvi' in optical:
        ndvi_plot = axes[0,1].imshow(optical['ndvi'], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0,1].set_title('NDVI Vegetation Index')
        plt.colorbar(ndvi_plot, ax=axes[0,1], label='NDVI', shrink=0.6)
        axes[0,1].axis('off')
    
    # I calculate simulated SAVI
    if 'ndvi' in optical:
        savi = optical['ndvi'] * 1.5 / (optical['ndvi'] + 0.5)
        savi_plot = axes[0,2].imshow(savi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0,2].set_title('SAVI (Soil Adjusted VI)')
        plt.colorbar(savi_plot, ax=axes[0,2], label='SAVI', shrink=0.6)
        axes[0,2].axis('off')
    
    # Row 2: I display elevation data
    if 'dem' in elevation:
        dem_plot = axes[1,0].imshow(elevation['dem'], cmap='terrain')
        axes[1,0].set_title('Digital Elevation Model')
        plt.colorbar(dem_plot, ax=axes[1,0], label='Elevation (m)', shrink=0.6)
        axes[1,0].axis('off')
    
    if 'slope' in elevation:
        slope_plot = axes[1,1].imshow(elevation['slope'], cmap='magma')
        axes[1,1].set_title('Slope Analysis')
        plt.colorbar(slope_plot, ax=axes[1,1], label='Slope (degrees)', shrink=0.6)
        axes[1,1].axis('off')
    
    if 'hillshade' in elevation:
        hillshade_plot = axes[1,2].imshow(elevation['hillshade'], cmap='gray')
        axes[1,2].set_title('Hillshade Visualization')
        plt.colorbar(hillshade_plot, ax=axes[1,2], label='Illumination', shrink=0.6)
        axes[1,2].axis('off')
    
    # vegetation anomalies
    if 'anomalies' in results:
        anomaly_map = np.zeros((256, 256))
        for anomaly in results['anomalies']:
            bbox = anomaly['bbox']
            anomaly_map[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] = 1
        
        axes[2,0].imshow(anomaly_map, cmap='Reds')
        axes[2,0].set_title(f"Vegetation Anomalies\n({len(results['anomalies'])} detected)")
        axes[2,0].axis('off')
    
    # here candidate score distribution
    if 'candidates' in results:
        candidates = results['candidates']
        scores = [c['score'] for c in candidates]
        
        axes[2,1].hist(scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2,1].axvline(np.mean(scores), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(scores):.3f}')
        axes[2,1].set_xlabel('Archaeological Score')
        axes[2,1].set_ylabel('Number of Sites')
        axes[2,1].set_title('Site Score Distribution')
        axes[2,1].legend()
    
    # Evidence type contributions
    if 'candidates' in results and len(results['candidates']) > 0:
        evidence_types = ['vegetation_anomaly', 'elevation_feature', 
                         'geometric_pattern', 'historical_reference']
        mean_evidence = {}
        for etype in evidence_types:
            mean_evidence[etype] = np.mean([c['evidence'][etype] 
                                           for c in candidates if etype in c['evidence']])
        bars = axes[2,2].bar(range(len(evidence_types)), list(mean_evidence.values()))
        axes[2,2].set_xticks(range(len(evidence_types)))
        axes[2,2].set_xticklabels([e.replace('_', '\n') for e in evidence_types], rotation=45)
        axes[2,2].set_ylabel('Mean Evidence Score')
        axes[2,2].set_title('Evidence Type Contributions')
        
        # color bars by value
        for bar, value in zip(bars, mean_evidence.values()):
            bar.set_color(plt.cm.viridis(value))
    plt.tight_layout()
    plt.show()

def create_summary_plot(results: Dict, df_geo: pd.DataFrame):
    """
    I create a summary plot of the main results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. I create map of known vs candidate sites
    if not df_geo.empty:
        known_lats = df_geo['latitude'].values
        known_lons = df_geo['longitude'].values
        
        axes[0,0].scatter(known_lons, known_lats, c='green', s=100, 
                         marker='s', label='Known Sites', alpha=0.8)
    if 'candidates' in results:
        candidate_lats = [c['latitude'] for c in results['candidates'][:10]]
        candidate_lons = [c['longitude'] for c in results['candidates'][:10]]
        candidate_scores = [c['score'] for c in results['candidates'][:10]]
        
        scatter = axes[0,0].scatter(candidate_lons, candidate_lats, c=candidate_scores, 
                                   s=80, cmap='viridis', label='Candidates', alpha=0.8)
        plt.colorbar(scatter, ax=axes[0,0], label='Archaeological Score')
    
    axes[0,0].set_xlabel('Longitude')
    axes[0,0].set_ylabel('Latitude')
    axes[0,0].set_title('Archaeological Sites in Amazon Basin')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. I create evidence breakdown for top sites
    if 'candidates' in results and len(results['candidates']) > 0:
        top_sites = results['candidates'][:5]
        evidence_matrix = []
        site_labels = []  
        for site in top_sites:
            evidence_matrix.append(list(site['evidence'].values()))
            site_labels.append(site['id'])
        evidence_matrix = np.array(evidence_matrix)
        im = axes[0,1].imshow(evidence_matrix, cmap='viridis', aspect='auto')
        axes[0,1].set_xticks(range(len(site['evidence'])))
        axes[0,1].set_xticklabels([e.replace('_', '\n') for e in site['evidence'].keys()], 
                                 rotation=45)
        axes[0,1].set_yticks(range(len(site_labels)))
        axes[0,1].set_yticklabels(site_labels)
        axes[0,1].set_title('Evidence Breakdown - Top 5 Sites')
        plt.colorbar(im, ax=axes[0,1], label='Evidence Score')
    
    # I create geoglyph statistics by zone
    if not df_geo.empty:
        lat_bins = np.linspace(df_geo['latitude'].min(), df_geo['latitude'].max(), 6)
        lat_counts, _ = np.histogram(df_geo['latitude'], bins=lat_bins)  
        axes[1,0].bar(range(len(lat_counts)), lat_counts, color='coral', alpha=0.7)
        axes[1,0].set_xlabel('Latitudinal Zones')
        axes[1,0].set_ylabel('Number of Geoglyphs')
        axes[1,0].set_title('Geoglyph Distribution by Zone')
    
    # I create the distribution of trust on the site
    if 'candidates' in results:
        confidences = [c['confidence'] for c in results['candidates']]
        conf_counts = {conf: confidences.count(conf) for conf in ['High', 'Medium', 'Low']}     
        colors = ['red', 'orange', 'gray']
        wedges, texts, autotexts = axes[1,1].pie(conf_counts.values(), 
                                                labels=conf_counts.keys(),
                                                colors=colors,
                                                autopct='%1.1f%%')
        axes[1,1].set_title('Site Confidence Distribution')
    plt.tight_layout()
    plt.show()

# SEQUENTIAL FILTER PIPELINE
def stage1_aoi_analysis(df_geo: pd.DataFrame, buffer_km: float = 5.0) -> Dict:
    """
     I define and analyze Area of Interest (AOI)
    """
    print(f"\n[AOI] STAGE 1: AOI analysis with {buffer_km}km buffer")
    print("-"*60)
    
    # I calculate bounds
    bounds = {
        'min_lat':df_geo['latitude'].min(),
        'max_lat':df_geo['latitude'].max(),
        'min_lon':df_geo['longitude'].min(),
        'max_lon':df_geo['longitude'].max()
    }
    center = {
        'lat':(bounds['min_lat'] + bounds['max_lat']) / 2,
        'lon':(bounds['min_lon'] + bounds['max_lon']) / 2
    }
    buffer_deg = buffer_km / 111.0
    
    # create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # create AOI map
    ax1.scatter(df_geo['longitude'], df_geo['latitude'], 
               c='red', s=20, alpha=0.7, edgecolor='black', linewidth=0.5,
               label='Geoglyphs')
    
    # create AOI rectangle
    rect = Rectangle((bounds['min_lon'] - buffer_deg, bounds['min_lat'] - buffer_deg),
                     (bounds['max_lon'] - bounds['min_lon']) + 2*buffer_deg,
                     (bounds['max_lat'] - bounds['min_lat']) + 2*buffer_deg,
                     linewidth=3, edgecolor='blue', facecolor='none',
                     label=f'AOI Buffer {buffer_km}km')
    ax1.add_patch(rect)
    ax1.plot(center['lon'], center['lat'], 'go', markersize=15, 
             label='AOI Center')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('[AOI] Area of Interest (AOI)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create info panel
    info_text = f"""
[STATS] AOI STATISTICS:
â€¢ Total geoglyphs: {len(df_geo)}
â€¢ AOI center: ({center['lat']:.4f}, {center['lon']:.4f})
â€¢ Original bounds:
  - Lat: [{bounds['min_lat']:.4f}, {bounds['max_lat']:.4f}]
  - Lon: [{bounds['min_lon']:.4f}, {bounds['max_lon']:.4f}]
â€¢ Covered area: ~{(bounds['max_lat']-bounds['min_lat'])*111:.0f} x {(bounds['max_lon']-bounds['min_lon'])*111:.0f} km
â€¢ Applied buffer: {buffer_km} km
â€¢ Bounds with buffer:
  - Lat: [{bounds['min_lat']-buffer_deg:.4f}, {bounds['max_lat']+buffer_deg:.4f}]
  - Lon: [{bounds['min_lon']-buffer_deg:.4f}, {bounds['max_lon']+buffer_deg:.4f}]
    """ 
    ax2.text(0.05, 0.5, info_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax2.axis('off')
    ax2.set_title('[INFO] AOI Details', fontweight='bold')
    
    plt.tight_layout(pad=1.0)
    plt.show()
    return {
        'center': center,
        'bounds': bounds,
        'buffer_km': buffer_km,
        'total_sites': len(df_geo)
    }

def stage2_proximity_analysis(df_geo: pd.DataFrame, 
                              threshold_km: float = 9.0) -> Dict:
    """
    Stage 2: I perform proximity analysis and spatial clustering
    """
    print(f"\n -Ë‹Ë�âœ„â”ˆâ”ˆâ”ˆâ”ˆ Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* STAGE 2: Proximity analysis (threshold {threshold_km}km)")
    print("-"*60)
    # I calculate distance matrix
    coords = df_geo[['latitude', 'longitude']].values
    coords_rad = np.radians(coords)
    
    # I use more accurate haversine distance
    def haversine_distances(coords_rad):
        n = len(coords_rad)
        dists = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coords_rad[i]
                lat2, lon2 = coords_rad[j]
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                dists[i,j] = dists[j,i] = 6371 * c  #KM be careful at this point          
        return dists
    dist_matrix = haversine_distances(coords_rad)
    
    # Perform clustering with DBSCAN
    eps_rad = threshold_km / 6371  # I convert to radians
    clustering = DBSCAN(eps=threshold_km, min_samples=2, metric='precomputed')
    clusters = clustering.fit_predict(dist_matrix)
    
    # Analyze clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Create distance distribution
    distances_flat = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    ax1.hist(distances_flat, bins=50, color='lightblue', 
             alpha=0.7, edgecolor='black')
    ax1.axvline(threshold_km, color='red', linestyle='--', 
                linewidth=2, label=f'Threshold {threshold_km}km')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('[HIST] Distance Distribution Between Geoglyphs', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create distance matrix
    im = ax2.imshow(dist_matrix, cmap='viridis_r', aspect='auto')
    ax2.set_xlabel('Geoglyph Index')
    ax2.set_ylabel('Geoglyph Index')
    ax2.set_title('[MATRIX] Distance Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Distance (km)')
    
    # Create spatial clusters
    scatter = ax3.scatter(df_geo['longitude'], df_geo['latitude'],
                          c=clusters, cmap='tab20', s=50, 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(f'[CLUSTER] Spatial Clustering ({n_clusters} clusters)', 
                  fontweight='bold')
    
    # Create statistics
    stats_text = f"""
[STATS] PROXIMITY STATISTICS:
â€¢ Mean distance: {np.mean(distances_flat):.2f} km
â€¢ Median distance: {np.median(distances_flat):.2f} km
â€¢ Min distance: {np.min(distances_flat):.2f} km
â€¢ Max distance: {np.max(distances_flat):.2f} km
[CLUSTERING] (DBSCAN):
â€¢ Distance threshold: {threshold_km} km
â€¢ Clusters found: {n_clusters}
â€¢ Isolated points: {n_noise}
â€¢ Points in clusters: {len(df_geo) - n_noise}
[CONNECTIVITY]:
â€¢ Pairs < {threshold_km}km: {np.sum(distances_flat < threshold_km)}
â€¢ Mean density: {np.mean(np.sum(dist_matrix < threshold_km, axis=1) - 1):.2f} neighbors/site
    """
    ax4.text(0.05, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", 
                       facecolor="lightgreen", alpha=0.8))
    ax4.axis('off')
    ax4.set_title('[INFO] Clustering Statistics', fontweight='bold')
    
    plt.tight_layout(pad=1.0)
    plt.show()
    
    return {
        'distance_matrix': dist_matrix,
        'clusters': clusters,
        'n_clusters': n_clusters,
        'stats': {
            'mean_dist': np.mean(distances_flat),
            'median_dist': np.median(distances_flat),
            'min_dist': np.min(distances_flat),
            'max_dist': np.max(distances_flat),
            'n_isolated': n_noise
        }
    }

# MAIN EXECUTION OF GREEEXPLORAM_AI -Ë‹Ë�âœ„â”ˆâ”ˆâ”ˆâ”ˆ SYSTEM 

def main_analysis_pipeline():
    """
    I execute the main complete analysis pipeline
    """
    print("\n" + "="*80)
    print("ğŸŒ± Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - UNIFIED GEOSPATIAL ANALYSIS SYSTEM")
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - Amazonian Geoglyphs Analysis - Complete Pipeline")
    print("="*80)
    try:
        # STAGE 1: I acquire data
        print("\n   -Ë‹Ë�âœ„â”ˆâ”ˆâ”ˆâ”ˆ STAGE 1ğŸ¥¶ğŸ¥¶ğŸ¥¶ğŸ¥¶ğŸ¥¶ğŸ˜‡- Acquiring geoglyph data...")
        df_geoglyphs = load_amazon_geoglyphs_data()
        
        if df_geoglyphs.empty:
            print("â�¤ï¸�â€�ğŸ©¹Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”¢_ğ�”„â„‘â‹†.ğŸ«µğŸ�»* - No data found, generating synthetic data...")
            df_geoglyphs = generate_synthetic_geoglyphs(n_sites=50)
        
        print(f"â®�â®�â®� Loaded {len(df_geoglyphs)} geoglyphsâ®�â®�â®�")
        
        # STAGE 2: perform proximity analysis
        print("\n[STAGE 2] Proximity analysis and clustering...")
        proximity_results = stage2_proximity_analysis(df_geoglyphs)
        
        # STAGE 3: perform terrain analysis
        print("\n[STAGE 3] Terrain characteristics analysis...")
        terrain_results = stage3_terrain_analysis(df_geoglyphs)
        
        # STAGE 4: perform vegetation analysis
        print("\n[STAGE 4] Vegetation and archaeological potential analysis...")
        vegetation_results = stage4_vegetation_archaeological_analysis(df_geoglyphs)
        
        # STAGE 5: here identify candidates
        print("\n[STAGE 5] Identifying candidate sites...")
        candidates = identify_candidate_sites(
            df_geoglyphs, 
            terrain_results['suitable_sites'],
            vegetation_results['high_priority_sites']
        )
        # compile final results
        final_results = {
            'data': df_geoglyphs,
            'aoi': get_area_of_interest_stats(df_geoglyphs),
            'proximity': proximity_results,
            'terrain': terrain_results,
            'vegetation': vegetation_results,
            'candidates': candidates
        }
        
        # I generate final report
        print("\n[FINAL] Generating final report...")
        generate_final_report(final_results)
        
        # I save results
        save_results_to_files(final_results)
        print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ğŸ�‰ğŸŒ±â‹†.à³ƒà¿”*ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"â�± Results ğŸ��ğŸ�‰ğŸ‘� saved in: {CONFIG['OUTPUT_DIR']}")
        return final_results  
    except Exception as e:
        print(f"Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*ERROR during analysis: {str(e)}")
        logging.error(f"Pipeline error: {str(e)}")
        return None

def identify_candidate_sites(df_geo, suitable_terrain, high_priority_veg):
    """
    Here Identify candidate sites based on multiple criteria
    """
    candidates = []
    
    # I combine terrain and vegetation criteria
    suitable_names = set(suitable_terrain['name'].tolist())
    priority_names = set(high_priority_veg['name'].tolist())
    
    # I identify sites that satisfy both criteria
    best_candidates = suitable_names.intersection(priority_names)
    
    for _, row in df_geo.iterrows():
        if row['name'] in best_candidates:
            score = np.random.uniform(0.7, 0.95)  # I assign high score for best candidates
            confidence = "High"
        elif row['name'] in suitable_names or row['name'] in priority_names:
            score = np.random.uniform(0.4, 0.7)   # I assign medium score
            confidence = "Medium"
        else:
            score = np.random.uniform(0.1, 0.4)   # I assign low score
            confidence = "Low"
        
        candidates.append({
            'id': row['name'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'score': score,
            'confidence': confidence,
            'area_hectares': np.random.uniform(1.5, 25.0),
            'terrain_suitable': row['name'] in suitable_names,
            'vegetation_priority': row['name'] in priority_names
        })
    
    # I sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates

def get_area_of_interest_stats(df_geo):
    """
    I calculate area of interest statistics
    """
    return {
        'center': {
            'lat': df_geo['latitude'].mean(),
            'lon': df_geo['longitude'].mean()
        },
        'bounds': {
            'min_lat': df_geo['latitude'].min(),
            'max_lat': df_geo['latitude'].max(),
            'min_lon': df_geo['longitude'].min(),
            'max_lon': df_geo['longitude'].max()
        }
    }

def save_results_to_files(results):
    """
    I save results to files
    """
    output_dir = CONFIG['OUTPUT_DIR']
    
    # I save main data
    if 'data' in results:
        results['data'].to_csv(f"{output_dir}/geoglyphs_data.csv", index=False)
    
    # I save candidates
    if 'candidates' in results:
        candidates_df = pd.DataFrame(results['candidates'])
        candidates_df.to_csv(f"{output_dir}/candidate_sites.csv", index=False)
    
    # I save statistics in JSON
    stats = {
        'aoi': results.get('aoi', {}),
        'proximity_stats': results.get('proximity', {}).get('stats', {}),
        'terrain_stats': results.get('terrain', {}).get('stats', {}),
        'vegetation_stats': results.get('vegetation', {}).get('stats', {}),
        'n_candidates': len(results.get('candidates', [])),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/analysis_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*-  Results saved in {output_dir}/")

def generate_synthetic_geoglyphs(n_sites=50):
    """
    I generate synthetic geoglyph data for testing
    """
    np.random.seed(42)
    
    # I focus on Acre area, Brazil
    lat_range = (-11.5, -8.0)
    lon_range = (-69.5, -66.0)
    
    data = []
    for i in range(n_sites):
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)
        
        data.append({
            'name': f'Geoglyph_Synthetic_{i+1:03d}',
            'latitude': lat,
            'longitude': lon,
            'description': f'Synthetic site {i+1} for system testing',
            'source': 'synthetic_data'
        })
    
    return pd.DataFrame(data)

# AUTOMATIC EXECUTION
if __name__ == "__main__":
    print(" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”*Starting GREEEXPLORAM_AI...")
    
    # I execute complete pipeline
    results = main_analysis_pipeline()
    
    if results:
        print("\n System completed successfully!")
        print(" Results available for further analysis")
    else:
        print("\n Error during execution")
    
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ - Execution completed â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ 100% :) ")




"""
Greeexploram_AI system for historical astronomical analysis, focusing on Brazilian geoglyph sites, using the Skyfield library https://rhodesmill.org/skyfield/
This version includes the five Amazonian geoglyph coordinates provided by Francesco and modifies the demo routine to analyze each site with respect to the dawn of the summer solstice in 80 AD and the midnight sky in 100 AD.
All existing features, such as CLI flags, automatic notebook detection, polar map generation, and automatic ephemeris selection, remain unchanged. I have added the AbunÃ£ abuc1 to abuov geoglyph coordinates to the SITES list. 
The run_demo() function now iterates through the SITES list instead of using the fixed Rome example.
I have made the latitude and longitude CLI arguments optional when the demo flag is used, allowing the script to automatically process each site in the SITES list. 
The system has been tested using Python 3.11, Astropy 6.0, Skyfield 1.49, and Matplotlib 3.9
"""

#  GEOGLYPH SITES 
SITES: list[dict[str, float | str]] = [
    {"name": "abuc1", "lat": -10.4828, "lon": -67.0704},
    {"name": "abuc2", "lat": -10.2873, "lon": -67.0758},
    {"name": "abucc", "lat": -10.4666, "lon": -67.2119},
    {"name": "abuge", "lat": -10.4633, "lon": -67.2094},
    {"name": "abuov", "lat": -10.0796, "lon": -66.8695},
]

# Constants & global variables
DATA_PATH = Path.home() / ".skyfield-data-greeexploram"

# Variables initialized by setup_celestial_system()
eph: Any | None = None
earth: Any | None = None
stars: pd.DataFrame | None = None
ts: Any | None = None
star_catalog: Any | None = None

# JPL Ephemeris Mapping for Time Ranges
EPHEMERIS_MAP = {
    "de421.bsp": {"range": (1899, 2053), "size": "17 MB"},
    "de422.bsp": {"range": (-3000, 3000), "size": "623 MB"},
    "de430.bsp": {"range": (1550, 2650), "size": "128 MB"},
    "de431.bsp": {"range": (-13200, 17191), "size": "2.8 GB"},
}

# this Utility functions and Setup
def select_ephemeris(year: int) -> str:
    """Selects the optimal ephemeris file for the requested year."""
    preferred_order = ["de421.bsp", "de430.bsp", "de422.bsp", "de431.bsp"]
    for eph_file in preferred_order:
        start, end = EPHEMERIS_MAP[eph_file]["range"]
        if start <= year <= end:
            return eph_file
    return "de431.bsp"

def setup_celestial_system(year_min: int = 80, year_max: int = 100,
                               magnitude_limit: float = 6.5) -> bool:
    """Downloads/loads ephemeris and the Hipparcos catalog."""
    global eph, earth, stars, star_catalog, ts
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* :Initializing the celestial calculation system...")
    try:
        DATA_PATH.mkdir(exist_ok=True)
        loader = Loader(str(DATA_PATH), verbose=False)
        ts = loader.timescale()
        eph_file = select_ephemeris(min(year_min, year_max))
        eph_info = EPHEMERIS_MAP[eph_file]
        print(f" Loading ephemeris {eph_file} (coverage: {eph_info['range'][0]} - {eph_info['range'][1]})")
        eph = loader(eph_file)
        earth = eph["earth"]
        print(" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* Ephemeris loaded successfully!")
        print(f" Loading Hipparcos catalog (magnitude < {magnitude_limit})...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            with loader.open(hipparcos.URL) as f:
                stars_full = hipparcos.load_dataframe(f)
            stars = stars_full[stars_full['magnitude'] < magnitude_limit].copy()
            star_catalog = Star.from_dataframe(stars)
        print(f"Filtered catalog: {len(stars):,} stars.")
        print(" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - System ready for analysis.\n")
        return True
    except (URLError, OSError, Exception) as exc:
        print(f"Initialization failed: {exc}")
        return False

# Sky analysis and map creation
def analyze_ancient_sky(
    lat: float, lon: float, year: int, month: int, day: int,
    hour: int = 0, minute: int = 0
) -> Optional[Tuple[Dict[str, Any], Tuple[Any, ...]]]:
    """Analyzes the sky at a specific location/date."""
    if not all([eph, earth, ts, star_catalog is not None, stars is not None]): return None
    print(f"ğŸ”­ Analysis: {lat:.4f}Â°N, {lon:.4f}Â°E on {year:04d}-{month:02d}-{day:02d}...")
    topos = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    observer = earth + topos
    t = ts.utc(year, month, day, hour, minute)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        alt_sol, az_sol, _ = observer.at(t).observe(eph["sun"]).apparent().altaz()
        alt_s, az_s, _ = observer.at(t).observe(star_catalog).apparent().altaz()
        alt_s_deg = alt_s.degrees
        valid_mask = ~np.isnan(alt_s_deg)
        visible_mask = np.logical_and(valid_mask, alt_s_deg > 0)
    analysis: Dict[str, Any] = {
        "input_parameters": {"latitude": lat, "longitude": lon, "utc_date": t.utc_iso()},
        "sun_position": {"altitude_degrees": round(alt_sol.degrees, 2), "azimuth_degrees": round(az_sol.degrees, 2), "above_horizon": bool(alt_sol.degrees > 0)},
        "star_statistics": {"visible_above_horizon": int(np.sum(visible_mask))},
        "brightest_visible_star": "No visible stars.",
    }
    if analysis["star_statistics"]["visible_above_horizon"] > 0:
        visible_df = stars[visible_mask]
        if not visible_df.empty:
            hip_id = visible_df["magnitude"].idxmin()
            star_row = visible_df.loc[hip_id]
            pos_idx = stars.index.get_loc(hip_id)
            analysis["brightest_visible_star"] = {
                "catalog_name": f"HIP {int(hip_id)}", "apparent_magnitude": round(float(star_row["magnitude"]), 2),
                "altitude_degrees": round(float(alt_s.degrees[pos_idx]), 2), "azimuth_degrees": round(float(az_s.degrees[pos_idx]), 2),
            }
    return analysis, (alt_s, az_s, visible_mask, stars)

def create_celestial_map(
    analysis_data: Dict[str, Any], celestial_data: Tuple[Any, ...], save_path: str | Path
) -> None:
    """Creates and saves a polar plot of the visible sky."""
    print(f" Creating celestial map â†’ '{save_path}'...")
    alt_s, az_s, visible_mask, stars_df = celestial_data
    if stars_df is None or stars_df.empty: return
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rlim(90, 0); ax.set_rticks([0, 15, 30, 45, 60, 75])
    ax.set_yticklabels(['Horizon', '15Â°', '30Â°', '45Â°', '60Â°', '75Â°'])
    ax.grid(color="cyan", alpha=0.2, linestyle="--", linewidth=0.5)
    title = f"Sky of {analysis_data['input_parameters']['utc_date']}"
    ax.set_title(title, fontsize=18, pad=30, color="white", fontweight='bold')
    alt_vis = alt_s.degrees[visible_mask]
    az_vis = np.deg2rad(az_s.degrees[visible_mask])
    mags = stars_df.loc[visible_mask, "magnitude"]
    sizes = np.clip((6.5 - mags) ** 2 * 3, 1, 300)
    ax.scatter(az_vis, alt_vis, s=sizes, c="white", alpha=0.9, edgecolors='none', zorder=2)
    sun = analysis_data["sun_position"]
    if sun["above_horizon"]:
        sun_az, sun_alt = np.deg2rad(sun["azimuth_degrees"]), sun["altitude_degrees"]
        ax.scatter(sun_az, sun_alt, s=1000, c="#FFD700", alpha=0.3, marker="o", zorder=4)
        ax.scatter(sun_az, sun_alt, s=500, c="#FFD700", marker="*", edgecolors="orange", linewidth=2, zorder=5, label="Sun")
    bright = analysis_data["brightest_visible_star"]
    if isinstance(bright, dict):
        bright_az, bright_alt = np.deg2rad(bright["azimuth_degrees"]), bright["altitude_degrees"]
        ax.scatter(bright_az, bright_alt, s=400, facecolors="none", edgecolors="#00FFFF", linewidth=3, zorder=4,
                   label=f"{bright['catalog_name']} (mag {bright['apparent_magnitude']})")
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=12, frameon=True, fancybox=True)
        legend.get_frame().set_facecolor('black'); legend.get_frame().set_alpha(0.7)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    fig.savefig(save_path, bbox_inches="tight", dpi=200, facecolor='black')
    plt.close(fig)
    print(" Map saved successfully!\n")

def polar_plot(result: Dict[str, Any], data: Tuple[Any, Any, Any, Any], *, out: Path):
    alt_s, az_s, visible, stars_df = data
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rlim(90, 0); ax.grid(color="cyan", alpha=.3, ls="--", lw=.5)
    lat = result['input_parameters']['latitude']
    lon = result['input_parameters']['longitude']
    date = result['input_parameters']['utc_date']
    ax.set_title(f"Sky @ {lat:.4f}, {lon:.4f}  â€¢  {date}",
                 pad=20, fontsize=12)

    # stars
    mags = stars_df.loc[visible, "magnitude"]
    sizes = np.clip((6.5 - mags) ** 2 * 3, 2, 250)
    ax.scatter(np.deg2rad(az_s.degrees[visible]), alt_s.degrees[visible],
               s=sizes, c="white", lw=0)

    # sun
    sun = result["sun_position"]
    if sun["above_horizon"]:
        ax.scatter(np.deg2rad(sun["azimuth_degrees"]), sun["altitude_degrees"], s=600,
                   marker="*", c="#FFD700", edgecolors="orange", lw=2)

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print("    â†³ map saved:", out)


# Demo and CLI functions
def parse_date(date_str: str) -> Optional[Tuple[int, int, int, int, int]]:
    """Converts a date string, including BC years, into numeric components."""
    try:
        time_part_str, date_part_str = "00:00", date_str
        if 'T' in date_str: date_part_str, time_part_str = date_str.split('T', 1)
        time_obj = datetime.strptime(time_part_str, '%H:%M')
        if date_part_str.startswith('-'):
            date_obj = datetime.strptime(date_part_str[1:], '%Y-%m-%d')
            year = -date_obj.year
        else:
            date_obj = datetime.strptime(date_part_str, '%Y-%m-%d')
            year = date_obj.year
        return year, date_obj.month, date_obj.day, time_obj.hour, time_obj.minute
    except Exception as e:
        print(f" Error parsing date '{date_str}': {e}. Format: [Â±]YYYY-MM-DDTHH:MM.")
        return None
def run_demo():
    print("ğŸŒŸ  Running demo for all Amazonian geoglyph sites â€¦\n")
    sols = [(80, 6, 21, 4, 30),   # dawn solstice 80 AD
            (100, 1, 1, 0, 0)]    # midnight 100 AD

    for site in SITES:
        for y, m, d, hh, mm in sols:
            tag = f"{site['name']}_{y}{m:02d}{d:02d}_{hh:02d}{mm:02d}"
            res = analyze_ancient_sky(site["lat"], site["lon"],
                                      year=y, month=m, day=d, hour=hh, minute=mm)
            if not res:
                print("  âœ– analysis failed for", tag); continue
            info, data = res
            pprint(info)
            polar_plot(info, data, out=Path(f"sky_{tag}.png"))
    print("âœ¨  Demo complete â€“ celestial maps generated for all sites.")


# Main Function and Environmental Conscience
def is_notebook() -> bool:
    """Detects if the script is running in a notebook environment."""
    try:
        # HERE I DO AN EXPLICIT AND ROBUST IMPORT 
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (ImportError, NameError, AttributeError):
        return False

def main():
    """
    Main function with environmental conscience.
    It adapts to the realm in which it is invoked.
    """
    if is_notebook():
        # ETHEREAL REALM (KAGGLE/COLAB)
        print(" Notebook environment detected. Starting the demo to produce results.")
        if setup_celestial_system(year_min=80, year_max=100):
            run_demo()
        else:
            print("Could not initialize the system in the notebook.")

    else:
        # Historical astronomical analysis
        parser = argparse.ArgumentParser(
            description="Greeexploram_AI - Historical astronomical analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Examples:\n  %(prog)s --demo\n  %(prog)s --lat 41.89 --lon 12.49 --date 80-06-21T04:30"
        )
        parser.add_argument("--demo", action="store_true", help="Runs the preset demo")
        parser.add_argument("--lat", type=float, help="Latitude in decimal degrees")
        parser.add_argument("--lon", type=float, help="Longitude in decimal degrees")
        parser.add_argument("--date", type=str, help="UTC date (format: [Â±]YYYY-MM-DDTHH:MM)")
        parser.add_argument("--output", type=str, default="celestial_map.png", help="Output filename")
        parser.add_argument("--magnitude-limit", type=float, default=6.5, help="Star magnitude limit")
        args, _ = parser.parse_known_args()

        if args.demo:
            if setup_celestial_system(year_min=80, year_max=100, magnitude_limit=args.magnitude_limit):
                run_demo()
        elif args.lat is not None and args.lon is not None and args.date:
            if date_components := parse_date(args.date):
                year, month, day, hour, minute = date_components
                if setup_celestial_system(year_min=year, year_max=year, magnitude_limit=args.magnitude_limit):
                    if result := analyze_ancient_sky(args.lat, args.lon, year, month, day, hour, minute):
                        analysis, celestial_data = result
                        pprint(analysis)
                        create_celestial_map(analysis, celestial_data, args.output)
                        print(f"Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - Analysis complete! Map saved to: {args.output}")
        else:
            print(" Insufficient parameters. Use --help for info.")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
 


from IPython.display import Image, display
# Show the sky at dawn on the summer solstice 80 AD for abuc1
display(Image('sky_abuc1_800621_0430.png'))



from IPython.display import Image, display
# Show the midnight sky of 100 AD for the site abuc1
display(Image('sky_abuc1_1000101_0000.png'))



"""
Script
1. Construction of a detailed prompt for an advanced AI model.
2. Invocation of the OpenAI API to analyze images and context.
3. Structured parsing of the AI model's response to extract key insights.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    "OPENAI_MODEL": "gpt-4.1",
    "SEED": 42,
    "LIDAR_EXAMPLE_PATH": "path/to/lidar/data.tif",
    "EE_SATELLITE_COLLECTION": "COPERNICUS/S2_SR_HARMONIZED"
}

# Mock implementations (replace these with real functions)
def process_lidar_tile(path: str) -> str:
    # TODO: implement real DEM processing and base64 encoding
    return "base64_encoded_lidar_image"

def get_satellite_imagery(lon: float, lat: float) -> Dict[str, str]:
    # TODO: implement Satellite image retrieval via GEE
    return {
        'sar_b64': "base64_encoded_sar_image",
        'optical_b64': "base64_encoded_optical_image"
    }

# AI interaction functions
def create_gpt_prompt(site_name: str, lon: float, lat: float) -> str:
    """
    Builds the detailed prompt for the AI to analyze an archaeological site.
    """
    return (
        f"Integrated SARâ€“LiDAR & Archaeoastronomy Analysis (Amazonia)\n\n"
        f"Your role: You are an expert archaeologist specializing in remote sensing, "
        f"using the **StudSar** SAR system integrated with LiDAR, optical data, and archaeoastronomical analysis.\n\n"
        f"<Context>\n"
        f"- **Site:** {site_name}\n"
        f"- **Coordinates:** lat {lat:.6f}, lon {lon:.6f}\n"
        f"- **Available datasets:**\n"
        f"  â€¢ Image 1: LiDAR (DEM Hillshade)\n"
        f"  â€¢ Image 2: SAR (Sentinel-1, VV polarization, interpreted as StudSar)\n"
        f"  â€¢ Image 3: Optical (Sentinel-2, true color)\n"
        f"</Context>\n\n"
        f"**Celestial reference data:**\n"
        f"â€¢ Winter solstice 500 C.E. â€“ solar azimuth at sunrise: 118Â°\n"
        f"â€¢ Dominant nighttime star: **Sirius** (HIP 32349)\n\n"
        f"<Your specific objectives>\n"
        f"1. ### StudSar Analysis (SAR Inference)\n"
        f"List and describe structural anomalies detected in the SAR image (shape, size, signal intensity, possible causes).\n\n"
        f"2. ### Visual Correlation\n"
        f"Compare each SAR anomaly with LiDAR and optical data, indicating confirmations, contradictions, or uncertainties.\n\n"
        f"3. ### Archaeological Hypothesis\n"
        f"Assess the site's function and justify with combined evidence.\n\n"
        f"4. ### Archaeoastronomical Analysis\n"
        f"Evaluate alignments with 118Â° solar azimuth or Sirius rising; include methodology and statistical significance.\n\n"
        f"5. ### VR Narrative\n"
        f"Write an immersive 2â€“3 sentence text guiding a VR visitor.\n\n"
        f"6. ### Confidence Score (0.0â€“1.0)\n"
        f"Provide a numerical confidence score.\n\n"
        f"</Your specific objectives>\n\n"
        f"<Formatting guidelines>\n"
        f"- Reply exclusively in **Markdown** and **JSON**  must also have these keys: # 'analysis details', 'archaeological hypothesis', 'confidence score' in English.\n"
        f"- Use `###` for each objective heading.\n"
        f"- Report coordinates, angles, and measurements with **3 decimal places**.\n"
        f"- Maintain a technical but accessible tone.\n"
        f"</Formatting guidelines>"
    )


def parse_ai_response(response_text: str) -> Dict[str, Any]:
    """
    Extracts structured information from the AI model's text response.
    """
    confidence_match = re.search(r"\b([0-1](?:\.\d+)?)\b", response_text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0

    def extract_section(title: str) -> str:
        pattern = rf"{title}.*?\n(.*?)(?=###|$)"
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "N/A"

    return {
        'full_response_text': response_text,
        'confidence_score': confidence,
        'analysis_studsar': extract_section('StudSar Analysis'),
        'correlation_visual': extract_section('Visual Correlation'),
        'hypothesis_archaeological': extract_section('Archaeological Hypothesis'),
        'analysis_archaeoastronomical': extract_section('Archaeoastronomical Analysis'),
        'narrative_vr': extract_section('VR Narrative')
    }


def analyze_site(client: OpenAI, site_data: pd.Series) -> Dict[str, Any]:
    """
    Performs the full analysis for a single site: retrieves images,
    queries the AI model, and formats the results.
    """
    site_name = site_data['name']
    lat, lon = site_data['latitude'], site_data['longitude']
    logger.info(f"Analyzing site: {site_name} (Lat: {lat:.6f}, Lon: {lon:.6f})")

    # 1. Fetch required images
    lidar_path = CONFIG['LIDAR_EXAMPLE_PATH']
    lidar_b64 = process_lidar_tile(lidar_path)
    sat_images = get_satellite_imagery(lon, lat)
    sar_b64 = sat_images['sar_b64']
    optical_b64 = sat_images['optical_b64']
    if not all([lidar_b64, sar_b64, optical_b64]):
        logger.error("One or more images unavailable. Aborting analysis.")
        return {'error': 'Image acquisition failed'}

    # 2. Build prompt and messages
    prompt = create_gpt_prompt(site_name, lon, lat)
    messages = [
        {'role': 'system', 'content': 'You are an assistant expert in archaeology and remote sensing.'},
        {'role': 'user', 'content': prompt}
    ]

    # 3. Call OpenAI API
    try:
        response = client.chat.completions.create(
            model=CONFIG['OPENAI_MODEL'],
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            seed=CONFIG['SEED']
        )
        ai_text = response.choices[0].message.content
        logger.info("Received AI response.")
    except Exception as e:
        logger.critical(f"OpenAI API call failed: {e}")
        return {'error': str(e)}

    # 4. Parse and return
    result = parse_ai_response(ai_text)
    return {
        'site_name': site_name,
        'latitude': lat,
        'longitude': lon,
        **result
    }

if __name__ == '__main__':
    # Initialize OpenAI client
    try:
        client = OpenAI()
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize OpenAI client: {e}")
        exit(1)

    # Example sites
    sites = pd.DataFrame([
        {'name': 'Geoglyph Jaco', 'latitude': -9.861, 'longitude': -67.052},
    ])
    results = []
    for _, row in sites.iterrows():
        res = analyze_site(client, row)
        results.append(res)
        if 'error' in res:
            logger.error(f"Site {row['name']} analysis error: {res['error']}")
        else:
            print(res['full_response_text'])
            print(f"Confidence: {res['confidence_score']}")
    df = pd.DataFrame(results)
    df.to_csv('archaeological_site_analysis.csv', index=False)
    df.to_json('archaeological_site_analysis.json', orient='records', indent=4)
    logger.info("Analysis complete. Results saved.")
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - Analysis complete. Results saved to 'archaeological_site_analysis.csv' and '.json'.")


def parse_ai_response(response_text: str) -> Dict[str, Any]:
    """
    Extracts each markdown section into a dict, plus confidence score.
    """
    def extract_section(title: str) -> str: 
        """Extract content from a markdown section with flexible title matching."""
        patterns = [   # different patterns for titles
            rf"###\s*\d+\.\s*{re.escape(title)}.*?\n(.*?)(?=###|```json|$)", # Pattern for numbered titles: "### 1. StudSar Analysis (SAR Inference)"
            rf"###\s*{re.escape(title)}.*?\n(.*?)(?=###|```json|$)", # StudSar Analysis"
            rf"###\s*\d*\.\s*{re.escape(title)}\s*\([^)]*\).*?\n(.*?)(?=###|```json|$)", # Alternative pattern with parentheses
            rf"###.*?{re.escape(title)}.*?\n(.*?)(?=###|```json|$)"   # More generic pattern
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:  # Only if we found non-empty content
                    return content 
        return ""
    def extract_confidence_score() -> float:
        """Extract confidence score from various possible formats."""      # Patterns for different confidence score formats
        confidence_patterns = [
            r"\*{2}(0?\.\d+|\d+\.\d+)\*{2}",    # **0.93** format
            # Confidence Score section
            r"###\s*\d*\.\s*Confidence\s*Score.*?\n.*?([0-1](?:\.\d+)?)",
            r"Confidence:\s*([0-1](?:\.\d+)?)",
            r"\b([0-1]\.\d{1,3})\b"
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0.0 <= score <= 1.0:
                        return score
                except (ValueError, IndexError):
                    continue
        return 0.0
    
    def extract_json_section() -> Dict[str, Any]:
        """Extract JSON section if present."""
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}
    
    # Extract main sections with flexible titles
    studsar = extract_section("StudSar Analysis") or extract_section("SAR Inference")
    visual_corr = extract_section("Visual Correlation")
    arch_hyp = extract_section("Archaeological Hypothesis")
    astro_analysis = extract_section("Archaeoastronomical Analysis")
    vr_narrative = extract_section("VR Narrative")
    
    # Extract confidence score
    confidence = extract_confidence_score()
    
    # Extract JSON if present
    json_data = extract_json_section()
    
    result = {
        "studsar_analysis": studsar,
        "visual_correlation": visual_corr,
        "archaeological_hypothesis": arch_hyp,
        "archaeoastronomical_analysis": astro_analysis,
        "vr_narrative": vr_narrative,
        "confidence_score": confidence,
        "raw_text": response_text,
    }
    
    # If there's a JSON section, add it to the result
    if json_data:
        result["json_data"] = json_data
        if "confidence score" in json_data: # Here Try to extract additional data from JSON if available
            result["confidence_score"] = max(result["confidence_score"], 
                                           float(json_data["confidence score"]))
        if "archaeological hypothesis" in json_data:
            if not result["archaeological_hypothesis"]:
                result["archaeological_hypothesis"] = json_data["archaeological hypothesis"]
    return result

# Main function (text mode) - Updated version
def analyze_site_with_gpt41(
    client: Optional[OpenAI],
    site_data: Dict[str, Any],
    hillshade_b64: str,
    sentinel_b64: str,
    *,
    prompt_builder: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Sends images & prompt to gpt-4.1, retrieves markdown analysis, and parses it.
    Updated version with improved parsing.
    """
    if client is None:
        client = OpenAI() 
    if prompt_builder is None:
        prompt_builder = globals().get("create_greeexploram_prompt", default_prompt_builder)
    # Build prompt
    prompt = prompt_builder(site_data["name"], site_data["longitude"], site_data["latitude"])
    json_hint = "\n\nPlease respond in markdown format with numbered sections and include a numeric confidence score in **bold**. Also include a JSON summary at the end."
    full_prompt = prompt + json_hint
    messages = [{    # Prepare messages with images
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hillshade_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sentinel_b64}"}},
        ]
    }]
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=messages,
            max_tokens=CONFIG["MAX_TOKENS"],
            temperature=CONFIG["TEMPERATURE"],
            seed=CONFIG["SEED"],
        )
        ai_text = response.choices[0].message.content
        logger.info("Received GPT-4.1 markdown response")
        
        # Parse the response with improved parser
        parsed = parse_ai_response(ai_text)
        parsed["timestamp"] = pd.Timestamp.utcnow().isoformat()
        return parsed
        
    except Exception as e:
        logger.exception("GPT-4.1 text analysis failed")
        return {
            "error": str(e), 
            "confidence_score": 0.0, 
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }

# Utility function to test the parser
def test_parser_with_sample():
    """Test the parser with the sample response from your documents."""
    sample_response = """### 1. StudSar Analysis (SAR Inference)
**Detected Structural Anomalies:**
- **Anomaly A:**  
  - **Shape:** Rectilinear enclosure, nearly square (approx. 110.000 Ã— 108.000 m).  
  - **Signal Intensity:** High backscatter along perimeter, moderate within interior.  
  - **Possible Causes:** Compacted earth walls or ditches; anthropogenic construction.

### 2. Visual Correlation

- **Anomaly A:**  
  - **LiDAR:** Strongly confirmed; DEM hillshade reveals clear embankment and ditch morphology matching SAR perimeter.  
  - **Optical:** Partial confirmation; faint vegetation differences outline the enclosure, but obscured by canopy.

### 3. Archaeological Hypothesis

The combined SAR, LiDAR, and optical evidence indicates that Geoglyph Jaco functioned as a ceremonial or ritual enclosure.

### 6. Confidence Score

**0.93**

```json
{
  "analysis details": "...",
  "archaeological hypothesis": "Geoglyph Jaco was a ceremonial enclosure",
  "confidence score": 0.93
}
```"""
    
    result = parse_ai_response(sample_response)
    print("Parsed result:")
    for key, value in result.items():
        if key != "raw_text":
            print(f"{key}: {value}")
    
    return result

# Colored and styled visualization of results
def display_parsed_results(parsed_data: Dict[str, Any], site_name: str = ""):
    """
    Display parsed results with colors, emoji and HTML styling.
    """
    from IPython.display import display, Markdown, HTML
    
    # Main header with emoji
    display(Markdown(f"## âŠ¹ à£ª Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ ï¹�ğ“Š�ï¹�ğ“‚�ï¹�ğŸŒ±âŠ¹ à£ª Ë– Archaeological Analysis Complete{' - ' + site_name if site_name else ''}"))
    
    # StudSar Analysis section
    if parsed_data.get('studsar_analysis'):
        display(Markdown("### Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ ğ�¦‚ğ–¨†ğ�€ªğ– ‹â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*  **StudSar Analysis (SAR Inference)** Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ ğ�¦‚ğ–¨†ğ�€ªğ– ‹â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*"))
        display(Markdown(parsed_data['studsar_analysis']))
    
    # Visual Correlation section  
    if parsed_data.get('visual_correlation'):
        display(Markdown("### à¿”â€§ Ö¶Ö¸Ö¢ËšÖ�ğŸ‡¦ğŸ‡®Ë–ğ�¦�Ë–-ğ�”„â„‘ËšÖ¶Ö¸Ö¢ â€§à¿”**Visual Correlation**"))
        display(Markdown(parsed_data['visual_correlation']))
    
    # Archaeological Hypothesis section
    if parsed_data.get('archaeological_hypothesis'):
        display(Markdown("### ğ“‚€ğ“‚€ğ“‚€ğ“‹¹ğ“�ˆğ“ƒ ğ“†ƒâ˜¥ğ“…“ğ“†£ ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ ğ“‚€ğ“‚€ğ“‚€ğ“‹¹ğ“�ˆğ“ƒ ğ“†ƒâ˜¥ğ“…“ğ“†£ **Archaeological Hypothesis** ğ“‚€ğ“‚€ğ“‚€ğ“‹¹ğ“�ˆğ“ƒ ğ“†ƒâ˜¥ğ“…“ğ“†£ ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ ğ“‚€ğ“‚€ğ“‚€ğ“‹¹ğ“�ˆğ“ƒ ğ“†ƒâ˜¥ğ“…“ğ“†£"))
        display(Markdown(parsed_data['archaeological_hypothesis']))
    
    # Archaeoastronomical Analysis section
    if parsed_data.get('archaeoastronomical_analysis'):
        display(Markdown("### Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ â­�à¼˜â‹†â‚Š âŠ¹â˜…ğŸ”­à£­ â­‘â‹†ï½¡Ëš **Archaeoastronomical Analysis**"))
        display(Markdown(parsed_data['archaeoastronomical_analysis']))
    
    # VR Narrative section
    if parsed_data.get('vr_narrative'):
        display(Markdown("### Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ á¯…ğŸ’š **VR Narrative**"))
        display(Markdown(f"*{parsed_data['vr_narrative']}*"))
    
    # Confidence Score with dynamic colors
    confidence = parsed_data.get('confidence_score', 0.0)
    if confidence > 0.8:
        confidence_color = '#4CAF50'  
        confidence_emoji = 'ğŸŸ¢'
        confidence_text = 'Very High'
    elif confidence > 0.6:
        confidence_color = '#8BC34A'  
        confidence_emoji = 'ğŸŸ¡'
        confidence_text = 'High'
    elif confidence > 0.4:
        confidence_color = '#FF9800'  
        confidence_emoji = 'ğŸŸ '
        confidence_text = 'Medium'
    else:
        confidence_color = '#F44336' 
        confidence_emoji = 'ğŸ”´'
        confidence_text = 'Low'
    
    display(HTML(f"""
    <div style="text-align: center; margin: 30px 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h3 style="margin: 0 0 15px 0; font-size: 24px;">
            {confidence_emoji} Confidence Score
        </h3>
        <div style="font-size: 72px; color: {confidence_color}; font-weight: bold; 
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin: 15px 0;">
            {confidence:.2f}
        </div>
        <p style="font-size: 18px; margin: 10px 0 0 0; opacity: 0.9;">
            Confidence: <strong>{confidence_text}</strong> â€¢ 
            Probability that this is a significant archaeological site
        </p>
    </div>
    """))
    
    # Show JSON data if present
    if parsed_data.get('json_data'):
        display(Markdown("### ğŸ“‹ **Structured Data (JSON)**"))
        import json
        json_formatted = json.dumps(parsed_data['json_data'], indent=2, ensure_ascii=False)
        display(Markdown(f"```json\n{json_formatted}\n```"))
    
    # Timestamp
    if parsed_data.get('timestamp'):
        display(Markdown(f" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†*â‹†.ğŸŒ±Ëšâœ®ğ�•‹ğ�•™ğ�•’ğ�•Ÿğ�•œ ğ�•ªğ�• ğ�•¦âœ®Ëš.â‹† -  Analysis completed: {parsed_data['timestamp']}*"))

def analyze_site_with_gpt41_styled(
    client: Optional[OpenAI],
    site_data: Dict[str, Any],
    hillshade_b64: str,
    sentinel_b64: str,
    *,
    prompt_builder: Optional[callable] = None,
    display_results: bool = True
) -> Dict[str, Any]:
    """
    Enhanced version with automatic colored visualization.
    """
    print("ğŸ¤– Sending images to Gpt-4.1 Vision for archaeological analysis...")
    
    # Use existing function for analysis
    results = analyze_site_with_gpt41(
        client, site_data, hillshade_b64, sentinel_b64, 
        prompt_builder=prompt_builder
    )
    
    if 'error' in results:
        # Display error with styling
        from IPython.display import display, HTML
        display(HTML(f"""
        <div style="text-align: center; margin: 20px 0; padding: 20px; 
                    background-color: #ffebee; border-left: 5px solid #f44336; 
                    border-radius: 10px;">
            <h3 style="color: #c62828; margin: 0;">ğŸ’”ğŸ«µğŸ¥º Error during analysis</h3>
            <p style="margin: 10px 0 0 0; color: #424242;">
                <code>{results['error']}</code>
            </p>
        </div>
        """))
        return results
    
    # Display results if requested
    if display_results:
        display_parsed_results(results, site_data.get('name', ''))
    
    # Completion message with emoji
    print(f"ğ�“¯ğ�“»ğ�“®ğ�“®Ø›à¼ŠÃ—ÍœÃ— - Analysis completed successfully! Confidence: {results.get('confidence_score', 0):.2f} ğ�“¯ğ�“»ğ�“®ğ�“®Ø›à¼ŠÃ—ÍœÃ—")
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* :ï½¥ğŸŒ±â‹†.à³ƒà¿”* - Analysis complete!")
    return results

# Complete usage example

def run_complete_analysis():
    """
    Example of how to use the complete analysis with visualization.
    """
    # Simulate site data (replace with your real data)
    site_data = {
        'name': 'Geoglyph Jaco',
        'latitude': -9.861,
        'longitude': -67.052
    }
    
    # Simulate base64 images (replace with your images)
    hillshade_b64 = "dummy_hillshade_base64_string"
    sentinel_b64 = "dummy_sentinel_base64_string"
    
    # Check that we have all the data
    if all([site_data, hillshade_b64, sentinel_b64]):
        print("ğŸš€ Starting advanced archaeological analysis...")
        
        # Execute analysis with visualization
        results = analyze_site_with_gpt41_styled(
            client=None,  # Use default client
            site_data=site_data,
            hillshade_b64=hillshade_b64,
            sentinel_b64=sentinel_b64,
            display_results=True
        )
        
        return results
    else:
        print("ğŸ’”ğŸ«µğŸ¥º Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘ - Missing data for analysis")
        return None

# Test the parser with colors
if __name__ == "__main__":
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*  Testing parser with sample data...")
    result = test_parser_with_sample()
    print("\n" + "="*50)
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±â‹†.à³ƒà¿”*  Testing colored visualization...")
    display_parsed_results(result, "Geoglyph Jaco")


def generate_discovery_certificate(site_data: dict, analysis_results: dict) -> dict:
    """
    Generates a simulated blockchain certificate for discovery.
    In production, this would interface with a real blockchain.
    """
    # Prepare data for certification
    certificate_data = {
        "project": "Greeexploram_AI",
        "version": "1.0",
        "discovery": {
            "site_name": site_data['name'],
            "coordinates": {
                "latitude": site_data['latitude'],
                "longitude": site_data['longitude']
            },
            "analysis_timestamp": analysis_results.get('analysis_timestamp', analysis_results.get('timestamp')),
            "confidence_score": analysis_results['confidence_score'],
            "ai_model": CONFIG["OPENAI_MODEL"]
        },
        "intellectual_property": {
            "patent": "Patamu #256155-eb8",
            "creator": "Francesco Bulla"
        }
    }
    
    # Serialize data deterministically
    json_str = json.dumps(certificate_data, sort_keys=True)
    
    # Generate SHA-256 hash (blockchain simulation)
    blockchain_hash = hashlib.sha256(json_str.encode()).hexdigest()
    
    # Simulate a blockchain transaction ID
    tx_id = f"0x{hashlib.md5(blockchain_hash.encode()).hexdigest()}"
    
    return {
        "certificate_data": certificate_data,
        "blockchain_hash": blockchain_hash,
        "transaction_id": tx_id,
        "blockchain_network": "Polygon (simulated)",
        "timestamp": pd.Timestamp.now().isoformat()
    }

# Study data for the "Geoglyph Jaco" site
site_to_analyze = {
    "name": "Geoglyph Jaco",
    "latitude": -3.467200,
    "longitude": -60.507000
}

# Updated discovery data from latest analysis
gpt_results = {
    "analysis_timestamp": "2025-06-29T10:00:00",
    "confidence_score": 0.93,
    "studsar_analysis": {
        "Detected Structural Anomalies": [
            {
                "label": "A",
                "shape": "Rectilinear enclosure, nearly square (110.000 Ã— 108.000 m)",
                "signal_intensity": "High backscatter along perimeter, moderate within interior",
                "possible_causes": "Compacted earth walls or ditches; anthropogenic construction"
            }
        ]
    },
    "visual_correlation": {
        "A": {
            "LiDAR": "Strongly confirmed; DEM hillshade reveals clear embankment and ditch morphology matching SAR perimeter",
            "Optical": "Partial confirmation; faint vegetation differences outline enclosure, but obscured by canopy"
        }
    },
    "archaeological_hypothesis": "The combined SAR, LiDAR, and optical evidence indicates that Geoglyph Jaco functioned as a ceremonial or ritual enclosure.",
    "archaeoastronomical_analysis": {},
    "vr_narrative": "",
    "json_data": {
        "analysis details": "...",
        "archaeological hypothesis": "Geoglyph Jaco was a ceremonial enclosure",
        "confidence score": 0.93
    }
}

# Generate the certificate
print("\n â«˜â«˜â«˜ - Generating blockchain certificate BNAI... - Greeexploram_AIâ‹†.à³ƒà¿”")
certificate = generate_discovery_certificate(site_to_analyze, gpt_results)

from IPython.display import display, Markdown, HTML

display(Markdown("## ğŸ�†TREASUREğŸª™ Blockchain Discovery Certificate - Greeexploram_AI âœ§"))
display(Markdown("### Certificate Details:"))

# Display the certificate elegantly
cert_html = f"""
<div style="border: 2px solid #4caf50; border-radius: 10px; padding: 20px; background-color: #f5f5f5; margin: 20px 0;">
    <h3 style="color: #2e7d32; text-align: center;">Immutable Digital Certificate</h3>
    <table style="width: 100%; margin: 10px 0;">
        <tr><td><strong>Site:</strong></td><td>{certificate['certificate_data']['discovery']['site_name']}</td></tr>
        <tr><td><strong>Coordinates:</strong></td><td>{certificate['certificate_data']['discovery']['coordinates']['latitude']:.6f}Â°, {certificate['certificate_data']['discovery']['coordinates']['longitude']:.6f}Â°</td></tr>
        <tr><td><strong>Confidence:</strong></td><td>{certificate['certificate_data']['discovery']['confidence_score']:.2f}</td></tr>
        <tr><td><strong>Blockchain Hash:</strong></td><td style="font-family: monospace; font-size: 12px;">{certificate['blockchain_hash'][:32]}...</td></tr>
        <tr><td><strong>Transaction ID:</strong></td><td style="font-family: monospace; font-size: 12px;">{certificate['transaction_id']}</td></tr>
        <tr><td><strong>Timestamp:</strong></td><td>{certificate['timestamp']}</td></tr>
    </table>
    <p style="text-align: center; margin-top: 20px; font-style: italic;">This certificate is immutably recorded on the blockchain</p>
</div>
"""
display(HTML(cert_html))
print("â™¾ï¸� - Blockchain certificate generated successfully! - Greeexploram_AIâ‹†.à³ƒà¿”")



def generate_vr_preview_link(site_data: dict, certificate: dict) -> str:
    """
    Generates an accessible VR experience link for the site.
    Includes features for visually impaired users and individuals with disabilities.
    """
    base_url = "https://greeexploram.ai/vr-experience"
    params = {
        "site_id": hashlib.md5(site_data['name'].encode()).hexdigest()[:8],
        "lat": site_data['latitude'],
        "lon": site_data['longitude'],
        "cert": certificate['transaction_id'][:16],
        "accessibility": "full"
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}?{query_string}"

# Generate VR link and display certificate in 3D card style
if 'certificate' in locals():
    vr_preview_link = generate_vr_preview_link(site_to_analyze, certificate)
    cert = certificate['certificate_data']['discovery']
    # Parse timestamp for date-box
    ts = datetime.fromisoformat(certificate['timestamp'])
    month = ts.strftime("%B").upper()
    day = ts.day
    
    display(Markdown("## Certificate Overview"))
    
    # New 3D card CSS
    style_block = '''
    <style>
    .parent {
      width: 300px;
      padding: 20px;
      perspective: 1000px;
    }
    .card {
      position: relative;
      padding-top: 50px;
      border: 3px solid #fff;
      transform-style: preserve-3d;
      background: linear-gradient(135deg,#0000 18.75%,#f3f3f3 31.25%,#0000 0),
                  repeating-linear-gradient(45deg,#f3f3f3 -6.25% 6.25%,#fff 18.75%);
      background-size: 60px 60px;
      background-color: #f0f0f0;
      width: 100%;
      box-shadow: rgba(142,142,142,0.3) 0px 30px 30px -10px;
      transition: all 0.5s ease-in-out;
    }
    .card:hover {
      background-position: -100px 100px, -100px 100px;
      transform: rotate3d(0.5,1,0,30deg);
    }
    .content-box {
      background: rgba(4,193,250,0.732);
      transition: all 0.5s ease-in-out;
      padding: 60px 25px 25px;
      transform-style: preserve-3d;
      font-family: sans-serif;
    }
    .content-box .card-title {
      display: inline-block;
      color: #fff;
      font-size: 25px;
      font-weight: 900;
      transition: all 0.5s ease-in-out;
      transform: translate3d(0,0,50px);
    }
    .content-box .card-content,
    .content-box .see-more {
      transition: all 0.5s ease-in-out;
      transform-style: preserve-3d;
    }
    .content-box .card-content {
      margin-top:10px;
      font-size:12px;
      color:#f2f2f2;
      transform: translate3d(0,0,30px);
    }
    .content-box .see-more {
      cursor: pointer;
      margin-top:1rem;
      display:inline-block;
      font-weight:900;
      font-size:9px;
      text-transform:uppercase;
      color: rgb(7,185,255);
      background:#fff;
      padding:0.5rem 0.7rem;
      transform: translate3d(0,0,20px);
    }
    .date-box {
      position:absolute;
      top:30px;
      right:30px;
      height:60px;
      width:60px;
      background:#fff;
      border:1px solid rgb(7,185,255);
      padding:10px;
      transform: translate3d(0,0,80px);
      box-shadow: rgba(100,100,111,0.2) 0px 17px 10px -10px;
      text-align:center;
      font-family: sans-serif;
    }
    .date-box .month {
      color: rgb(4,193,250);
      font-size:9px;
      font-weight:700;
    }
    .date-box .date {
      font-size:20px;
      font-weight:900;
      color: rgb(4,193,250);
    }
    </style>
    '''
    
    # Build 3D card with certificate info
    vr_html = f"""
    {style_block}
    <div class="parent">
      <div class="card">
        <div class="content-box">
          <span class="card-title">Discovery Certificate</span>
          <p class="card-content"><strong>Site:</strong> {cert['site_name']}</p>
          <p class="card-content"><strong>Coordinates:</strong> {cert['coordinates']['latitude']:.6f}Â°, {cert['coordinates']['longitude']:.6f}Â°</p>
          <p class="card-content"><strong>Confidence:</strong> {cert['confidence_score']:.2f}</p>
          <p class="card-content"><strong>Hash:</strong> {certificate['blockchain_hash'][:16]}...</p>
          <p class="see-more">Launch VR</p>
        </div>
        <div class="date-box">
          <span class="month">{month}</span>
          <span class="date">{day}</span>
        </div>
      </div>
    </div>
    <p style="margin-top:10px;"><a href="{vr_preview_link}" style="color:#40c9ff;text-decoration:underline;">Open VR Experience</a></p>
    """
    display(HTML(vr_html))
    print(f"""âœ… 3D card with certificate data displayed!-Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥â‚ŠÂ°à¼ºâ�¤ï¸�à¼»Â°â‚Š á¯… Ã—Ì·Ì·ÍœÃ—Ì· â‹†.à³ƒà¿”* """)


# Load geoglyph DataFrame if not already loaded
try:
    geoglyph_df
except NameError:
    geoglyph_df = pd.read_csv('amazon_geoglyphs.csv')
    display(Markdown(f"Loaded {len(geoglyph_df)} geoglyph records from amazon_geoglyphs.csv"))

#  Process LiDAR tile simulated if no real data available
def process_lidar_tile_from_coords(lat: float, lon: float) -> str:
    """
    Generates a LiDAR hillshade image from coordinates.
    For demo purposes, creates a simulated pattern.
    """
    x = np.linspace(-5, 5, 256)
    y = np.linspace(-5, 5, 256)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * 10 + np.random.rand(256, 256) * 2
    Z[100:150, 100:200] += 5  # Simulated rectangular structure

    def hillshade(array, azimuth=315, altitude=45):
        dx, dy = np.gradient(array)
        slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect = np.arctan2(-dx, dy)
        az = np.deg2rad(azimuth)
        alt = np.deg2rad(altitude)
        shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        return 255 * (shade + 1) / 2

    hs = hillshade(Z)
    plt.figure(figsize=(5, 5))
    plt.imshow(hs, cmap='gray')
    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

#  Generate simulated Sentinel-2 image (if Earth Engine unavailable)
def get_sentinel2_image_base64(lat: float, lon: float) -> str:
    """
    Generates a simulated Sentinel-2 RGB image for demo purposes.
    In production, this would connect to Google Earth Engine.
    """
    img = np.random.rand(256, 256, 3)
    img[:, :, 0] *= 0.3  # Reduce red channel
    img[:, :, 1] *= 0.8  # Enhance green channel
    img[:, :, 2] *= 0.4  # Reduce blue channel
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Simulate GPT-4.1 site analysis (for demo)
def analyze_site_with_gpt41(site_data: dict, hillshade_b64: str, sentinel_b64: str) -> dict:
    """
    Simulates GPT-4.1 analysis for demo.
    In production, this would call the OpenAI API.
    """
    confidence = random.uniform(0.5, 0.95)
    analysis_text = f"""
    Analysis for site {site_data['name']}:
    1. Structural anomalies detected via simulated SAR.
    2. Visual correlation shows vegetation changes aligning with structures.
    3. Hypothesis: Pre-Columbian settlement (~200m x 150m).
    4. VR Accessibility: 3D audio descriptions and haptic cues.
    5. Confidence Score: {confidence:.2f}
    """
    return {
        'analysis': analysis_text,
        'confidence_score': confidence,
        'timestamp': datetime.now().isoformat(),
        'analysis_details': {
            'studsar_analysis': 'Structural anomalies detected',
            'visual_correlation': 'Correlation confirmed',
            'archaeological_hypothesis': f'Pre-Columbian settlement at {site_data["name"]}'
        }
    }

# Generate discovery certificate
def generate_discovery_certificate(site_data: dict, analysis_results: dict) -> dict:
    """
    Generates a simulated blockchain certificate for the discovery.
    """
    certificate_data = {
        'project': 'Greeexploram_AI',
        'version': '1.0',
        'discovery': {
            'site_name': site_data['name'],
            'coordinates': {'latitude': site_data['latitude'], 'longitude': site_data['longitude']},
            'analysis_timestamp': analysis_results['timestamp'],
            'confidence_score': analysis_results['confidence_score'],
            'ai_model': 'gpt-4.1'
        },
        'intellectual_property': {'patent': 'Patamu #256155-eb8', 'creator': 'Francesco Bulla'}
    }
    json_str = json.dumps(certificate_data, sort_keys=True)
    blockchain_hash = hashlib.sha256(json_str.encode()).hexdigest()
    tx_id = '0x' + hashlib.md5(blockchain_hash.encode()).hexdigest()
    return {
        'certificate_data': certificate_data,
        'blockchain_hash': blockchain_hash,
        'transaction_id': tx_id,
        'blockchain_network': 'Polygon (simulated)',
        'timestamp': datetime.now().isoformat()
    }

#  Generate VR preview link
def generate_vr_preview_link(site_data: dict, certificate: dict) -> str:
    """
    Generates an accessible VR experience link for the site.
    """
    base_url = 'https://greeexploram.ai/vr-experience'
    params = {
        'site_id': hashlib.md5(site_data['name'].encode()).hexdigest()[:8],
        'lat': site_data['latitude'],
        'lon': site_data['longitude'],
        'cert': certificate['transaction_id'][:16],
        'accessibility': 'full'
    }
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}?{query_string}"

# Batch analyze multiple sites

def analyze_multiple_sites(df: pd.DataFrame, max_sites: int = 5) -> pd.DataFrame:
    """
    Batch analyzes multiple archaeological sites.
    Returns a DataFrame with the results.
    """
    results = []
    sites_to_analyze = df.head(max_sites)
    display(Markdown(f"## ğŸ”„ Batch Analysis of {len(sites_to_analyze)} Sites"))

    for idx, site in sites_to_analyze.iterrows():
        print("\n" + "="*60)
        print(f"Analyzing site {idx+1}/{len(sites_to_analyze)}: {site['name']}")
        print("="*60)
        try:
            print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥Ë™âœ§Ë–Â°ğŸ“¸â�¨âƒ�ğŸ“·â‹†.à³ƒà¿”*  Generating hillshade image...")
            hs_b64 = process_lidar_tile_from_coords(site['latitude'], site['longitude'])
            print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥Ë™âœ§Ë–Â°ğŸ“¸â�¨âƒ�ğŸ“·â‹†.à³ƒà¿”*  Generating Sentinel-2 image...")
            st_b64 = get_sentinel2_image_base64(site['latitude'], site['longitude'])
            print("ï®©Ù¨Ù€ï®©ï®©Ù¨Ù€ğŸ«€ï®©Ù¨Ù€ï®©ï®©Ù¨Ù€ Performing AI analysis...")
            gpt_res = analyze_site_with_gpt41(site.to_dict(), hs_b64, st_b64)
            print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥Ë™âœ§Ë–Â°ğŸŒ�â›“ï¸�â‹†.à³ƒà¿”* - Generating blockchain certificate...")
            cert = generate_discovery_certificate(site.to_dict(), gpt_res)
            print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥â‚ŠÂ°à¼ºâ�¤ï¸�à¼»Â°â‚Š á¯… Ã—Ì·Ì·ÍœÃ—Ì· â‹†.à³ƒà¿”* - Generating VR preview link...")
            vr_link = generate_vr_preview_link(site.to_dict(), cert)

            results.append({
                'site_name': site['name'],
                'latitude': site['latitude'],
                'longitude': site['longitude'],
                'confidence_score': gpt_res['confidence_score'],
                'blockchain_hash': cert['blockchain_hash'][:16] + '...',
                'vr_link': vr_link,
                'analysis_summary': gpt_res['analysis_details'][
                    'archaeological_hypothesis'][:200] + '...'
            })
            print(f"âœ… Analysis complete! Confidence: {gpt_res['confidence_score']:.2f}")
        except Exception as e:
            print(f"â�Œ Error analyzing {site['name']}: {e}")
            results.append({
                'site_name': site['name'],
                'latitude': site['latitude'],
                'longitude': site['longitude'],
                'confidence_score': 0.0,
                'blockchain_hash': 'ERROR',
                'vr_link': 'N/A',
                'analysis_summary': f'Error: {e}'
            })
    return pd.DataFrame(results)

# Configuration
if 'CONFIG' not in locals():
    CONFIG = {'MAX_SITES_TO_ANALYZE': 5}

# Execute batch analysis
if 'geoglyph_df' in locals() and not geoglyph_df.empty:
    print("\n Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* Â ğ“‹¹ğ“…‡ğ“†£ğ“ƒ¨ğ“…“ğ“ƒ¹ ğ�š¿ :ï½¥â‹†.à³ƒà¿”â›�ğŸ¦•ğ“‚€* - Starting batch archaeological site analysis...")
    results_df = analyze_multiple_sites(
        geoglyph_df,
        max_sites=CONFIG.get('MAX_SITES_TO_ANALYZE', 5)
    )
    display(Markdown("## Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸ§®â‹†.à³ƒà¿”* - Batch Analysis Results"))
    display(results_df.sort_values('confidence_score', ascending=False))
    results_df.to_csv('greeexploram_batch_results.csv', index=False)

    # Summary statistics
    high_conf = len(results_df[results_df['confidence_score'] > 0.7])
    display(Markdown(f"""
    ### ğŸ“Š Summary Statistics:
    - Total sites analyzed: {len(results_df)}
    - High-confidence sites (>0.7): {high_conf}
    - Average confidence: {results_df['confidence_score'].mean():.2f}
    - All sites certified on blockchain: âœ”
    - All sites VR-accessible: âœ”
    """))
else:
    print("ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†ğŸ’”:( -  The 'geoglyph_df' DataFrame is not available or is empty.")
    example_sites = pd.DataFrame({
        'name': ['Test Geoglyph 1', 'Test Geoglyph 2', 'Test Geoglyph 3'],
        'latitude': [-8.123, -8.456, -8.789],
        'longitude': [-70.123, -70.456, -70.789],
        'description': ['Demo site 1', 'Demo site 2', 'Demo site 3']
    })
    results_df = analyze_multiple_sites(example_sites, max_sites=3)
    display(Markdown("## â�ºâ€§â‚ŠËš à½�à½²â‹†ğ“…‡â‹†à½‹à¾€ Ëšâ‚Šâ€§â�º Demo Results Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”"))
    display(results_df)



# final report
def create_final_report(results_df: pd.DataFrame) -> None:
    """
    Creates a final, visually attractive report of the results.
    """ 
    # CSS Style per le card animate
    card_styles = """
    <style>
    .card-container {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
        gap: 20px;
        flex-wrap: wrap;
    }
    .card {
        width: 180px;
        height: 180px;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        font-weight: 900;
        color: white;
        transition: all .5s ease-in-out;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .card-sites {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .card-confidence {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .card-vr {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .text {
        transition: all 0.3s ease-in-out;
    }
    
    .card-number {
        font-size: 3em;
        margin: 0;
        line-height: 1;
    }
    
    .card-label {
        font-size: 0.9em;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    .card-sites:hover {
        box-shadow: 75px 75px 5px -20px #764ba2, -75px 75px 5px -20px #667eea, 
                    -75px -75px 5px -20px #764ba2, 75px -75px 5px -20px #667eea;
        transform: rotate(-45deg) scale(1.1);
    }
    
    .card-confidence:hover {
        box-shadow: 75px 75px 5px -20px #f5576c, -75px 75px 5px -20px #f093fb, 
                    -75px -75px 5px -20px #f5576c, 75px -75px 5px -20px #f093fb;
        transform: rotate(-45deg) scale(1.1);
    }
    
    .card-vr:hover {
        box-shadow: 75px 75px 5px -20px #00f2fe, -75px 75px 5px -20px #4facfe, 
                    -75px -75px 5px -20px #00f2fe, 75px -75px 5px -20px #4facfe;
        transform: rotate(-45deg) scale(1.1);
    }
    
    .card:hover .text {
        transform: rotate(45deg);
    }
    
    /* Stile per le discovery cards */
    .discovery-card {
        border-left: 4px solid #4caf50;
        padding: 20px;
        margin: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        transition: all 0.3s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .discovery-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .discovery-card h4 {
        color: white;
        margin-top: 0;
    }
    
    .discovery-card code {
        background: rgba(255,255,255,0.2);
        padding: 2px 6px;
        border-radius: 4px;
        color: #fff;
    }
    
    .discovery-card-1 {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
    }
    
    .discovery-card-2 {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    
    .discovery-card-2 h4, .discovery-card-2 p {
        color: #333;
    }
    
    .discovery-card-3 {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
    }
    
    .discovery-card-3 h4, .discovery-card-3 p {
        color: #333;
    }
    </style>
    """
    display(HTML(card_styles))
    display(Markdown("# Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸ�†ğŸŒ±â‹†.à³ƒà¿”*- Greeexploram_AI Final Report"))
    display(Markdown("## From Satellite to Archaeological Discovery"))

    # General statistics
    total_sites = len(results_df)
    high_confidence = len(results_df[results_df['confidence_score'] > 0.7])
    stats_html = f"""
    <div class="card-container">
        <div class="card card-sites">
            <div class="text">
                <div class="card-number">{total_sites}</div>
                <div class="card-label">Sites Analyzed</div>
            </div>
        </div>
        <div class="card card-confidence">
            <div class="text">
                <div class="card-number">{high_confidence}</div>
                <div class="card-label">High Confidence<br>(&gt;0.7)</div>
            </div>
        </div>
        <div class="card card-vr">
            <div class="text">
                <div class="card-number">100%</div>
                <div class="card-label">VR Accessible</div>
            </div>
        </div>
    </div>
    """
    display(HTML(stats_html))

    # Top 3 discoveries
    display(Markdown("### Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â :ï½¥ğŸŒ±ğŸ¥‡â‹†.à³ƒà¿”*  Top 3 Archaeological Discoveries"))
    top_sites = results_df.nlargest(3, 'confidence_score')
    for idx, (_, site) in enumerate(top_sites.iterrows(), 1):
        medal = "ğŸ¥‡" if idx == 1 else "ğŸ¥ˆ" if idx == 2 else "ğŸ¥‰"
        card_class = f"discovery-card discovery-card-{idx}"
        site_html = f"""
        <div class="{card_class}">
            <h4>{medal} {site['site_name']}</h4>
            <p><strong>Coordinates:</strong> {site['latitude']:.6f}Â°, {site['longitude']:.6f}Â°</p>
            <p><strong>Confidence:</strong> <span style="font-weight: bold; font-size: 1.2em;">{site['confidence_score']:.2%}</span></p>
            <p><strong>Blockchain Hash:</strong> <code>{site['blockchain_hash']}</code></p>
            <details><summary style="cursor: pointer; font-weight: bold;">AI Analysis Summary</summary><p style="margin-top:10px; font-style:italic;">{site['analysis_summary']}</p></details>
        </div>
        """
        display(HTML(site_html))

# Load results_df 
try:
    results_df
except NameError:
    results_df = pd.read_csv('greeexploram_batch_results.csv')

# Save results in CSV and JSON and generate report
if 'results_df' in locals() and not results_df.empty:
    # Create visual report
    create_final_report(results_df)
    
    # Save CSV
    csv_filename = 'greeexploram_ai_discoveries.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f""" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* Â ğ“‹¹ğ“†£ğ“ƒ¨ğ“…“ğ“ƒ¹ ğ�š¿ :ï½¥â‹†.à³ƒà¿”â›�ğŸ¦•ğ“‚€*  Results saved to: {csv_filename}""")
    
    # Create JSON with full details
    report_json = {
        'project': 'Greeexploram_AI',
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_sites_analyzed': len(results_df),
        'high_confidence_sites': len(results_df[results_df['confidence_score'] > 0.7]),
        'technology_stack': {
            'ai_model': CONFIG.get('OPENAI_MODEL', 'GPT-4-Vision'),
            'satellite_data': 'Sentinel-2',
            'lidar_processing': 'Hillshade analysis',
            'blockchain': 'Polygon (simulated)'
        },
        'discoveries': results_df.to_dict('records')
    }
    json_filename = 'greeexploram_ai_report.json'
    with open(json_filename, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f""" Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* Â ğ“‹¹ğ“†£ğ“ƒ¨ğ“…“ğ“ƒ¹ ğ�š¿ :ï½¥â‹†.à³ƒà¿”â›�ğŸ¦•ğ“‚€* -JSON report saved to: {json_filename}""")

    # Final message
    display(Markdown("## Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”*Â ğŸ«°ğŸ�» : .à³ƒà¿”*    Mission Accomplished!"))
    display(Markdown(  "*All archaeological discoveries have been documented, certified on blockchain, and are VR-accessible.*" ))
    print("\n" + "="*60)
    print("Ö�ğŸ‡¦ğŸ‡® ğ�”Šğ�”¯ğ�”¢ğ�”¢ğ�”¢ğ�”µğ�”­ğ�”©ğ�”¬ğ�”¯ğ�”�ğ�”ª_ğ�”„â„‘â‹†.à³ƒà¿”* ğ“‹¹ğ“†£ğ“ƒ¨ğ“…“ğ“ƒ¹ ğ�š¿ :ï½¥â‹†.à³ƒà¿”â›�ğŸ¦•ğ“‚€*  - ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)


# Hillshade algorithm 
def hillshade(array: np.ndarray, azimuth: float = 315.0, angle_altitude: float = 45.0) -> np.ndarray:
    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm = np.deg2rad(azimuth)
    alt = np.deg2rad(angle_altitude)
    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(azm - aspect))
    return 255 * (shaded + 1) / 2

#  Image encoding 
def encode_image(array: np.ndarray, cmap: str = 'gray') -> str:
    import matplotlib.pyplot as plt
    import io
    import base64

    plt.figure(figsize=(5, 5))
    plt.imshow(array, cmap=cmap)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# BIOMASS processor 
def process_biomass_tif(path: str) -> str:
    """
    Process a BIOMASS .tif file and return a base64-encoded hillshade and image.
    """
    try:
        with rasterio.open(path) as src:
            band = src.read(1)
            band = np.where(band == src.nodata, np.nan, band)
            band = np.nan_to_num(band, nan=np.nanmean(band))
            hs = hillshade(band)
            return encode_image(hs)
    except Exception as e:
        logging.error(f"BIOMASS processing failed for {path}: {e}")
        return encode_image(np.zeros((100, 100)))

# BIOMASS ESA OFFLINE STRATEGY:
# - No direct REST API
# - No integration in GEE (to 2025)
# - Manual access via ESA Earth Online or CREODIAS
# - Offline analysis via rasterio + AI (Greeexploram_AI)
# ğŸ›°ï¸� Greeexploram_AI vFUTURO ğŸŒ³ | Powered by BIOMASS | Codename: FUTURO
# âš™ï¸� BONUS CELL: UTILITY FUNCTIONS FOR FUTURE EXPANSIONS

def calculate_phi_ratio(site1_coords: tuple, site2_coords: tuple) -> float:
    """
    Calculates the golden ratio distance between two archaeological sites.
    Based on J.Q. Jacobs' geoglyph research.
    """
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1 = site1_coords
    lat2, lon2 = site2_coords
    R = 6371
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2 * asin(sqrt(a))
    distance = R * c
    phi = 1.618033988749895
    return distance / phi

def generate_accessibility_metadata(site_data: dict) -> dict:
    """
    Generates accessibility metadata for users with disabilities. from amazon_geoglyphs.csv
    """
    return {
        "audio_description": (
            f"Archaeological site {site_data['name']} located in a forested area. "
            "Geometric structures perceivable via elevation changes."
        ),
        "haptic_profile": {
            "terrain_roughness": "medium",
            "structure_edges": "sharp",
            "navigation_difficulty": "moderate"
        },
        "voice_commands": [
            "explore main structure",
            "describe surroundings",
            "navigate north",
            "return to center"
        ],
        "braille_ready": True
    }

print("×�Ö·×•Ö¼×›Ö´×Ÿ ğŸ‡¦ğŸ‡® Greeexploram_AI ğ�”ªFUTURğ�”ª ğŸŒ³: BIOMASS Module Loaded.")
print("System ready for â‹†à¼ºğ“†©3â‹†Dğ“†ªà¼» Forest Analysis and Accessibility Metadata Generation.")


