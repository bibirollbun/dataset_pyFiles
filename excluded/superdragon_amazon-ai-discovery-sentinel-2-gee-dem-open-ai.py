# This cell should be run first to install all necessary packages.
print("Starting library installation process...")

!pip install --upgrade pip -q
print("Step 1/3: pip upgraded.")

!pip install uv -q
print("Step 2/3: uv package manager installed.")

# Using uv to install the bulk of the packages.
# Ensure versions are compatible if issues arise.
!uv pip install --system --no-cache-dir \
    "earthengine-api>=0.1.300" \
    "openai>=1.10.0" \
    "pystac-client>=0.7.0" \
    "rioxarray>=0.14.0" \
    "rasterio>=1.3.0" \
    "elevation>=1.1.0" \
    "folium>=0.15.0" \
    "geopandas>=0.13.0" \
    "shapely>=2.0.0" \
    "Pillow>=9.5.0" \
    "tqdm>=4.65.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.7.0" \
    "requests>=2.30.0" \
    "pyproj>=3.6.0" # For CRS transformations

print("Step 3/3: Main geospatial and AI libraries installation command executed.")


import os
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
import re
import base64
import io
import urllib.request
import time 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import rasterio
from rasterio.windows import Window
from rasterio.plot import show as rio_show
import geopandas as gpd
from pystac_client import Client as PystacClient
from PIL import Image
from tqdm.notebook import tqdm 
import folium
from pyproj import Transformer as PyProjTransformer
from IPython.display import display, IFrame, HTML
import requests
from folium import plugins

# Google Earth Engine API
import ee

# OpenAI API
from openai import OpenAI

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress some common geopandas/shapely warnings

# --- Global Path Configuration (Kaggle Specific) ---
BASE_WORK_DIR_CONFIG = Path('/kaggle/working/')
RAW_DATA_DIR_CONFIG = BASE_WORK_DIR_CONFIG / 'data' / 'raw'
TILE_DIR_CONFIG = BASE_WORK_DIR_CONFIG / 'data' / 'tiles' # For full pipeline
OUTPUT_DIR_CONFIG = BASE_WORK_DIR_CONFIG / 'outputs'

for d_path in (RAW_DATA_DIR_CONFIG, TILE_DIR_CONFIG, OUTPUT_DIR_CONFIG):
    d_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Data directories created/ensured at {BASE_WORK_DIR_CONFIG / 'data'}")

# --- AOI and Date Configuration ---
# Main AOI for wider search 
MAIN_AOI_BBOX_CONFIG = {"west": -72.5, "south": -11.0, "east": -66.0, "north": -7.5}

GEE_CP1_POINT_LON_LAT_CONFIG = [-58.36623265568083, -6.981918953145955]
GEE_CP1_POINT_NAME_CONFIG = 'Amazon Checkpoint 1 Target (GEE)'

GLOBAL_TODAY_CONFIG = date.today()
# Date range for GEE Checkpoint 1 (e.g., last 6 months for clearer image)
GEE_CP1_END_DATE_CONFIG = GLOBAL_TODAY_CONFIG.isoformat()
GEE_CP1_START_DATE_CONFIG = (GLOBAL_TODAY_CONFIG - timedelta(days=180)).isoformat()
# Date range for wider STAC search (e.g., last 2 years)
STAC_END_DATE_CONFIG = GLOBAL_TODAY_CONFIG.isoformat()
STAC_START_DATE_CONFIG = (GLOBAL_TODAY_CONFIG - timedelta(days=2*365)).isoformat()


# --- GEE Specific Configuration for Checkpoint 1 ---
GEE_CP1_BUFFER_RADIUS_METERS_CONFIG = 2500  # Buffer around point for thumbnail region
GEE_CP1_IMAGE_COLLECTION_ID_CONFIG = 'COPERNICUS/S2_SR_HARMONIZED' # Surface Reflectance
GEE_CP1_MAX_CLOUD_COVERAGE_CONFIG = 15 # Target lower cloud cover for single image
GEE_CP1_VIS_PARAMS_CONFIG = {
    'bands': ['B4', 'B3', 'B2'],  # True color (Red, Green, Blue)
    'min': 0.0,                   # Sentinel-2 SR data is usually 0-10000, GEE scales it.
    'max': 3000,                  # Typical visualization max for SR scaled values. Adjust as needed.
    'gamma': 1.4                  # Gamma correction for visual appeal
}
GEE_CP1_THUMBNAIL_DIMENSIONS_CONFIG = '768' # Max pixels for width/height
GEE_CP1_THUMBNAIL_FORMAT_CONFIG = 'png'

# --- Full Pipeline Configuration (Tiling, NDVI, etc.) ---
FULL_PIPELINE_TILE_SIZE_METERS_CONFIG = 1000 # 1km tiles
FULL_PIPELINE_NDVI_THRESHOLD_CONFIG = 0.25
FULL_PIPELINE_BLACK_TILE_THRESHOLD_CONFIG = 0.98 # Exclude if >98% black/NoData

# --- File Path Definitions (derived from above) ---
# Checkpoint 1 GEE files (primarily for logging, actual image is URL)
GEE_CP1_THUMBNAIL_INFO_JSON_CONFIG = OUTPUT_DIR_CONFIG / 'gee_cp1_thumbnail_info.json'

# Full Pipeline files
S2_RGBNIR_COMPOSITE_PATH_CONFIG = RAW_DATA_DIR_CONFIG / 's2_rgbnir_composite_stac.tif'
S2_FCC_PATH_CONFIG = RAW_DATA_DIR_CONFIG / 's2_fcc_stac.tif'
NDVI_PATH_CONFIG = RAW_DATA_DIR_CONFIG / 's2_ndvi_stac.tif'
DEM_PATH_CONFIG = RAW_DATA_DIR_CONFIG / 'dem_srtmgl1.tif'
HILLSHADE_PATH_CONFIG = RAW_DATA_DIR_CONFIG / 'hillshade.tif'
TILES_INDEX_CSV_CONFIG = OUTPUT_DIR_CONFIG / 'stac_tiles_index.csv'
CANDIDATES_CSV_CONFIG = OUTPUT_DIR_CONFIG / 'stac_candidate_tiles_for_openai.csv'
OPENAI_RESULTS_CSV_CONFIG = OUTPUT_DIR_CONFIG / 'stac_openai_tile_analysis_results.csv'
FOLIUM_MAP_HTML_CONFIG = OUTPUT_DIR_CONFIG / 'stac_openai_analysis_interactive_map.html'

logger.info("Global configurations and paths defined.")


# Configuration Manager Class

class ConfigurationManager:
    """Manages API keys and critical configuration settings, including paths."""
    def __init__(self):
        # API Keys and Project ID
        self.openai_api_key: Optional[str] = None
        self.opentopo_api_key: Optional[str] = None
        self.gee_project_id: Optional[str] = None
        
        self.base_work_dir: Path = BASE_WORK_DIR_CONFIG
        self.raw_data_dir: Path = RAW_DATA_DIR_CONFIG
        self.tile_dir: Path = TILE_DIR_CONFIG
        self.output_dir: Path = OUTPUT_DIR_CONFIG

        # Specific file paths
        self.S2_RGBNIR_COMPOSITE_PATH: Path = S2_RGBNIR_COMPOSITE_PATH_CONFIG
        self.S2_FCC_PATH: Path = S2_FCC_PATH_CONFIG
        self.NDVI_PATH: Path = NDVI_PATH_CONFIG
        self.DEM_PATH: Path = DEM_PATH_CONFIG
        self.HILLSHADE_PATH: Path = HILLSHADE_PATH_CONFIG
        self.TILES_INDEX_CSV_PATH: Path = TILES_INDEX_CSV_CONFIG 
        self.CANDIDATES_CSV_PATH: Path = CANDIDATES_CSV_CONFIG   

        self.OPENAI_RESULTS_CSV_CONFIG: Path = OPENAI_RESULTS_CSV_CONFIG # use OPENAI_RESULTS_CSV_PATH 
        self.GEE_CP1_THUMBNAIL_INFO_JSON_PATH: Path = GEE_CP1_THUMBNAIL_INFO_JSON_CONFIG
        self.FOLIUM_MAP_HTML_CONFIG: Path = FOLIUM_MAP_HTML_CONFIG 

        # Pipeline settings constants
        self.FULL_PIPELINE_TILE_SIZE_METERS_CONFIG: int = FULL_PIPELINE_TILE_SIZE_METERS_CONFIG
        self.FULL_PIPELINE_NDVI_THRESHOLD_CONFIG: float = FULL_PIPELINE_NDVI_THRESHOLD_CONFIG
        self.FULL_PIPELINE_BLACK_TILE_THRESHOLD_CONFIG: float = FULL_PIPELINE_BLACK_TILE_THRESHOLD_CONFIG

        
        logger.info("ConfigurationManager initialized with paths. Attempting to load keys/IDs...")
        
        self._load_credentials() # API key load
        self.is_valid = self._validate_configuration() # check 
        
        if not self.is_valid:
            logger.error("One or more critical configurations are missing. Please check Kaggle Secrets or environment variables.")
            print(" CRITICAL ERROR: API Keys or GEE Project ID not configured. Check logs and Kaggle Secrets. ðŸš¨")
        else:
            logger.info("ConfigurationManager setup complete and validated.")

  
    def _load_credentials(self) -> None:
        """Load credentials from Kaggle Secrets or environment variables."""
        try:
            from kaggle_secrets import UserSecretsClient
            secrets_client = UserSecretsClient()
            self.openai_api_key = secrets_client.get_secret("OPENAI_API_KEY")
            logger.info("Successfully loaded OPENAI_API_KEY from Kaggle Secrets.")
            
            try:
                self.gee_project_id = secrets_client.get_secret("GEE_PROJECT_ID")
                logger.info("Successfully loaded GEE_PROJECT_ID from Kaggle Secrets.")
            except Exception as e_gee:
                logger.warning(f"GEE_PROJECT_ID not found in Kaggle Secrets: {e_gee}. Trying environment variable.")
                self.gee_project_id = os.environ.get("GEE_PROJECT_ID")
                if self.gee_project_id: logger.info("Loaded GEE_PROJECT_ID from environment variable.")
                else: logger.error("GEE_PROJECT_ID is also not set as an environment variable.")

            try:
                self.opentopo_api_key = secrets_client.get_secret("OPENTOPO_API_KEY")
                logger.info("Successfully loaded OPENTOPO_API_KEY from Kaggle Secrets.")
            except Exception as e_topo:
                logger.warning(f"OPENTOPO_API_KEY not found in Kaggle Secrets: {e_topo}. Trying environment variable.")
                self.opentopo_api_key = os.environ.get("OPENTOPO_API_KEY")
                if self.opentopo_api_key: logger.info("Loaded OPENTOPO_API_KEY from environment variable.")
                else: logger.info("OPENTOPO_API_KEY is also not set as an environment variable. DEM download will be skipped if key remains unavailable.")

        except ImportError:
            logger.warning("KaggleSecretsClient not found. Assuming local environment. Loading from os.environ.")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            self.gee_project_id = os.environ.get("GEE_PROJECT_ID")
            self.opentopo_api_key = os.environ.get("OPENTOPO_API_KEY")
            if self.openai_api_key: logger.info("Loaded OPENAI_API_KEY from environment.")
            if self.gee_project_id: logger.info("Loaded GEE_PROJECT_ID from environment.")
            if self.opentopo_api_key: logger.info("Loaded OPENTOPO_API_KEY from environment.")
        except Exception as e_main: 
            logger.error(f"An error occurred while trying to load secrets via KaggleSecretsClient: {e_main}")
            logger.info("Falling back to environment variables for all credentials.")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            self.gee_project_id = os.environ.get("GEE_PROJECT_ID")
            self.opentopo_api_key = os.environ.get("OPENTOPO_API_KEY")

    def _validate_configuration(self) -> bool:
        """Validates that essential configurations are loaded."""
        valid_config = True
        if not self.openai_api_key:
            logger.error("OpenAI API Key is MISSING.")
            valid_config = False
        if not self.gee_project_id: # GEE Project ID
            logger.error("Google Earth Engine Project ID is MISSING.")
            valid_config = False
        if not self.opentopo_api_key:
             logger.warning("OpenTopography API Key is MISSING. DEM download will be skipped.")
        # Note: DEM download can be optional, so opentopo_api_key missing might not make config invalid for all purposes
        return valid_config
        
config = ConfigurationManager()


gee_initialized_successfully = False

# Check if GEE_PROJECT_ID was loaded by ConfigurationManager
if config.is_valid and config.gee_project_id:
    logger.info(f"Attempting to initialize Google Earth Engine with Project ID: {config.gee_project_id}")
    try:
        # Authenticate if necessary. ee.Authenticate() guides through the process.
        # In many Kaggle environments, if you've done it once, it might use cached credentials
        # or a default credential mechanism without interactive prompts on subsequent runs
        # within the same session or if service accounts are configured.
        # However, for the first time or new sessions, interactive auth is common.
        
        # A common pattern is to try Initialize first, and Authenticate only if it fails with auth error.
        # However, explicit Authenticate() is often clearer for users first time.
        # We will try to initialize first, and if it fails due to auth, then guide for Authenticate().
        try:
            if not ee.data._credentials: # Check if already authenticated
                 logger.info("GEE credentials not found, attempting ee.Authenticate()...")
                 ee.Authenticate() # This will trigger the interactive authentication flow if needed.
            else:
                 logger.info("GEE credentials found, proceeding to initialize.")
            
            ee.Initialize(project=config.gee_project_id, opt_url='https://earthengine-highvolume.googleapis.com')
            logger.info(f"Google Earth Engine initialized successfully with Project ID: {config.gee_project_id}.")
            gee_initialized_successfully = True

        except ee.EEException as e:
            logger.error(f"Google Earth Engine initialization failed: {e}")
            if "Please authorize access to your Earth Engine account" in str(e) or \
               "gelt_bearer" in str(e) or "Ø­Ø¯ÛŒØ«" in str(e): # Common error message parts
                logger.info("Authentication might be required or token expired. If ee.Authenticate() did not run or failed, "
                            "try running 'import ee; ee.Authenticate()' in a new cell and follow instructions, then re-run this cell.")
            print(f" GEE Initialization Error: {e}. Make sure you have authenticated and your GEE_PROJECT_ID ('{config.gee_project_id}') is correct and has the Earth Engine API enabled in Google Cloud Console.")
        except Exception as e_other: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during GEE initialization: {e_other}", exc_info=True)
            print(f" Unexpected GEE Initialization Error: {e_other}.")

    except Exception as e_outer:
        logger.error(f"Outer exception during GEE setup: {e_outer}", exc_info=True)
        print(f" GEE Setup failed. Message: {e_outer}")
else:
    logger.error("GEE_PROJECT_ID not available from ConfigurationManager or config is invalid. GEE cannot be initialized.")
    print(" GEE_PROJECT_ID is missing. Cannot initialize Google Earth Engine.")

if gee_initialized_successfully:
    print("\n Google Earth Engine setup appears successful.")
else:
    print("\n Google Earth Engine setup failed or was skipped. GEE-dependent features will not work.")


class DataSourceManager:
    """Manages data acquisition from GEE, STAC (for S2), and OpenTopography (for DEM)."""

    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.stac_client: Optional[PystacClient] = None
        try:
            self.stac_client = PystacClient.open("https://earth-search.aws.element84.com/v1", timeout=30)
            logger.info("DataSourceManager initialized with PystacClient for AWS Earth Search.")
        except Exception as e:
            logger.error(f"Failed to initialize PystacClient for AWS STAC: {e}")

        self.selected_s2_stac_scene_info: Optional[Dict[str, Any]] = None
        self.gee_s2_image_info: Optional[Dict[str, Any]] = None
        logger.info("DataSourceManager instance created.")

    # --- GEE Methods for Checkpoint 1 ---
    def _get_least_cloudy_s2_image_gee(self,
                                      point_lon_lat: List[float],
                                      start_date_str: str,
                                      end_date_str: str,
                                      cloud_filter_percentage: float,
                                      buffer_radius_meters: int,
                                      collection_id: str
                                     ) -> Tuple[Optional[ee.Image], Optional[ee.Geometry.Polygon], Optional[str]]:
        if not gee_initialized_successfully: # Check global flag
            logger.error("GEE not initialized. Cannot fetch GEE image.")
            return None, None, None
        try:
            target_point = ee.Geometry.Point(point_lon_lat)
            # Define region for filtering and thumbnail by buffering and getting bounds
            region_for_filtering_and_thumb = target_point.buffer(buffer_radius_meters).bounds()

            image_collection = ee.ImageCollection(collection_id) \
                .filterBounds(region_for_filtering_and_thumb) \
                .filterDate(ee.Date(start_date_str), ee.Date(end_date_str)) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter_percentage)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE')

            least_cloudy_image = ee.Image(image_collection.first())
            
            try:
                image_id_info = least_cloudy_image.id().getInfo()
                if image_id_info is None: # No image found
                    logger.warning(f"No GEE S2 image found for point {point_lon_lat} with criteria.")
                    return None, None, None
            except ee.EEException as e_id: # Handles cases where .first() might return something not having .id() if empty
                logger.warning(f"Could not retrieve ID from GEE image (collection might be empty): {e_id}")
                return None, None, None

            logger.info(f"GEE: Found least cloudy S2 image: {image_id_info}")
            return least_cloudy_image, region_for_filtering_and_thumb, image_id_info
        except Exception as e:
            logger.error(f"Error getting least cloudy S2 image from GEE: {e}", exc_info=True)
            return None, None, None

    def _get_gee_image_thumbnail_url(self,
                                    image: ee.Image,
                                    region: ee.Geometry.Polygon, # Expecting GEE Geometry object
                                    dimensions: str,
                                    img_format: str,
                                    vis_params: Dict
                                   ) -> Optional[str]:
        if not gee_initialized_successfully or not image or not region:
            logger.error("GEE not initialized or GEE image/region missing for thumbnail URL generation.")
            return None
        try:
            # Ensure region is a server-side object or its coordinates for getThumbURL
            region_payload = region.getInfo()['coordinates'] if isinstance(region, ee.Geometry) else region
            
            visualized_image = image.visualize(**vis_params)
            thumbnail_url = visualized_image.getThumbURL({
                'region': region_payload,
                'dimensions': dimensions,
                'format': img_format
            })
            logger.info(f"GEE: Thumbnail URL generated: {thumbnail_url[:100]}...") # Log partial URL
            return thumbnail_url
        except Exception as e:
            logger.error(f"Error generating GEE image thumbnail URL: {e}", exc_info=True)
            return None

    def get_s2_image_info_via_gee(self,
                                 point_lon_lat: List[float],
                                 point_name: str,
                                 buffer_radius_m: int,
                                 start_date_iso: str,
                                 end_date_iso: str,
                                 max_cloud_perc: float,
                                 collection_id: str,
                                 vis_params: Dict,
                                 thumb_dimensions: str,
                                 thumb_format: str
                                ) -> Optional[Dict[str, Any]]:
        """Fetches S2 image info and thumbnail URL from GEE for a point."""
        logger.info(f"Fetching S2 image from GEE for: {point_name} at {point_lon_lat}")
        gee_s2_image, region_bounds_gee, image_id_str = self._get_least_cloudy_s2_image_gee(
            point_lon_lat, start_date_iso, end_date_iso, max_cloud_perc, buffer_radius_m, collection_id
        )

        if not gee_s2_image or not region_bounds_gee or not image_id_str:
            self.gee_s2_image_info = None
            return None

        thumbnail_url = self._get_gee_image_thumbnail_url(
            gee_s2_image, region_bounds_gee, thumb_dimensions, thumb_format, vis_params
        )
        if not thumbnail_url:
            self.gee_s2_image_info = None
            return None
            
        self.gee_s2_image_info = {
            "source": "Google Earth Engine", "image_id": image_id_str,
            "collection_id": collection_id, "point_coordinates_lon_lat": point_lon_lat,
            "point_name": point_name, "thumbnail_url": thumbnail_url,
            "timestamp": datetime.now().isoformat(),
            "visualization_params_used": vis_params
        }
        logger.info(f"Successfully retrieved S2 image info and thumbnail URL from GEE for '{point_name}'.")
        return self.gee_s2_image_info

    def download_gee_thumbnail_to_pil(self, url: str) -> Optional[Image.Image]:
        """Downloads an image from a GEE thumbnail URL and returns as PIL Image."""
        if not url: return None
        try:
            with urllib.request.urlopen(url, timeout=60) as response: # Added timeout
                img_data = response.read()
            pil_image = Image.open(io.BytesIO(img_data))
            logger.info(f"Successfully downloaded GEE thumbnail from URL to PIL Image.")
            return pil_image
        except urllib.error.URLError as e:
            logger.error(f"URL Error downloading image from GEE URL {url}: {e}")
        except Exception as e:
            logger.error(f"Error processing GEE image from URL {url}: {e}", exc_info=True)
        return None

    # --- STAC Methods for Full Pipeline (Sentinel-2 from AWS) ---
    def search_best_sentinel2_scene_stac(self, 
                                         aoi_bbox: Dict[str, float], 
                                         start_date_iso: str, 
                                         end_date_iso: str, 
                                         max_cloud_cover: int = 20, 
                                         max_items_search: int = 20) -> Optional[Dict[str, Any]]:
        if not self.stac_client:
            logger.error("STAC client not initialized. Cannot search via STAC.")
            return None
        logger.info(f"Searching STAC for Sentinel-2 L2A scenes in AOI: {aoi_bbox} for {start_date_iso}/{end_date_iso}")
        try:
            search = self.stac_client.search(
                collections=["sentinel-2-l2a"],
                bbox=[aoi_bbox[k] for k in ("west", "south", "east", "north")],
                datetime=f"{start_date_iso}/{end_date_iso}",
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                max_items=max_items_search
            )
            items_collection = search.item_collection()
            if not items_collection or not items_collection.items:
                logger.warning("No STAC S2 L2A scenes found matching criteria.")
                return None
            
            sorted_items = sorted(
                items_collection.items,
                key=lambda it: (it.properties.get("eo:cloud_cover", 101), -it.datetime.timestamp())
            )
            best_item = sorted_items[0]
            self.selected_s2_stac_scene_info = {
                "id": best_item.id, "datetime": best_item.datetime.isoformat(),
                "cloud_cover": best_item.properties.get("eo:cloud_cover"),
                "assets": {key: asset.href for key, asset in best_item.assets.items()},
                "bbox": best_item.bbox, "geometry": best_item.geometry,
                "properties": best_item.properties
            }
            logger.info(f"STAC: Selected S2 scene: {best_item.id} (Cloud: {best_item.properties.get('eo:cloud_cover')})")
            return self.selected_s2_stac_scene_info
        except Exception as e:
            logger.error(f"Error during STAC Sentinel-2 scene search: {e}", exc_info=True)
            return None

    def _get_stac_asset_href(self, band_common_name: str) -> Optional[str]:
        """Gets asset href from selected STAC scene's assets."""
        if not self.selected_s2_stac_scene_info or 'assets' not in self.selected_s2_stac_scene_info:
            logger.error("No STAC scene selected or assets not available.")
            return None
        
        asset_prefs = {
            'R': ['B04', 'red'], 'G': ['B03', 'green'], 'B': ['B02', 'blue'],
            'N': ['B08', 'nir', 'nir08'] # NIR
        }
        target_keys = asset_prefs.get(band_common_name.upper()) # Use uppercase for map key
        if not target_keys:
            logger.error(f"Common band name '{band_common_name}' not recognized for STAC assets.")
            return None

        scene_assets_dict = self.selected_s2_stac_scene_info['assets']
        for key_option in target_keys: # Check exact keys first
            if key_option in scene_assets_dict: return scene_assets_dict[key_option]
        for key_option_fallback in target_keys: # Fallback to partial, case-insensitive
            for actual_asset_key, href in scene_assets_dict.items():
                if key_option_fallback.lower() in actual_asset_key.lower(): return href
        logger.warning(f"STAC: Asset for band '{band_common_name}' not found.")
        return None

    def create_rgb_nir_composite_from_stac(self, output_path: Path) -> bool:
        """Creates a 4-band (R,G,B,NIR) float32 reflectance composite from STAC S2 assets."""
        if not self.selected_s2_stac_scene_info:
            logger.error("No STAC S2 scene selected. Cannot create RGB-NIR composite.")
            return False
        
        logger.info(f"Creating RGB-NIR composite from STAC assets to: {output_path}")
        band_keys_in_order = ['R', 'G', 'B', 'N'] # Red, Green, Blue, NIR
        urls = {key: self._get_stac_asset_href(key) for key in band_keys_in_order}

        if not all(urls.values()):
            missing = [key for key, url in urls.items() if not url]
            logger.error(f"STAC: Missing URLs for bands {missing}. Cannot create composite.")
            return False
        try:
            # Use NIR band (B08, typically 10m) as reference for profile
            ref_url = urls['N'] 
            with rasterio.open(ref_url) as src_ref:
                dst_profile = src_ref.profile.copy()
                dst_profile.update({
                    'count': 4, 'driver': 'GTiff', 'compress': 'lzw',
                    'dtype': 'float32', 'photometric': 'MINISBLACK'
                })
            
            stacked_bands_data = []
            for band_key in band_keys_in_order:
                with rasterio.open(urls[band_key]) as src:
                    # Assuming 10m bands, co-registered. Otherwise, resample.
                    data = src.read(1).astype(np.float32) / 10000.0 # Reflectance 0-1
                    data = np.clip(data, 0, 1.0)
                    stacked_bands_data.append(data)

            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for i in range(4): dst.write(stacked_bands_data[i], i + 1)
            logger.info(f"STAC: RGB-NIR (R,G,B,NIR order, float32) composite saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating STAC RGB-NIR composite: {e}", exc_info=True)
            return False

    def create_fcc_and_ndvi_from_rgbnir(self, rgb_nir_path: Path, fcc_path: Path, ndvi_path: Path) -> bool:
        """Creates FCC (uint8) and NDVI (float32) from a 4-band RGB-NIR (float32) GeoTIFF."""
        if not rgb_nir_path.exists():
            logger.error(f"RGB-NIR composite not found at {rgb_nir_path}. Cannot create FCC/NDVI.")
            return False
        logger.info(f"Creating FCC and NDVI from {rgb_nir_path}...")
        try:
            with rasterio.open(rgb_nir_path) as src: # R=1, G=2, B=3, NIR=4
                red_band = src.read(1); green_band = src.read(2); nir_band = src.read(4)

                # FCC (NIR,R,G -> RGB channels) uint8
                fcc_bands_for_rgb = [nir_band, red_band, green_band]
                fcc_uint8_bands = []
                for band_data in fcc_bands_for_rgb:
                    p_low,p_high = np.percentile(band_data[np.isfinite(band_data)&(band_data>0)], (2,98)) # Avoid zeros in percentile
                    scaled = np.clip(band_data, p_low, p_high)
                    scaled = ((scaled - p_low) / (p_high - p_low + 1e-7)) * 255
                    fcc_uint8_bands.append(scaled.astype(np.uint8))
                
                fcc_profile = src.profile.copy(); fcc_profile.update({'count':3, 'dtype':'uint8', 'photometric':'RGB'})
                with rasterio.open(fcc_path, 'w', **fcc_profile) as dst:
                    for i in range(3): dst.write(fcc_uint8_bands[i], i+1)
                logger.info(f"FCC (uint8) image saved to: {fcc_path}")

                # NDVI (float32)
                num = nir_band - red_band; den = nir_band + red_band
                ndvi = np.full_like(num, np.nan, dtype=np.float32)
                np.divide(num, den, out=ndvi, where=den!=0)
                ndvi = np.clip(ndvi, -1.0, 1.0)
                
                ndvi_profile = src.profile.copy(); ndvi_profile.update({'count':1, 'dtype':'float32', 'photometric':'MINISBLACK'})
                with rasterio.open(ndvi_path, 'w', **ndvi_profile) as dst: dst.write(ndvi, 1)
                logger.info(f"NDVI (float32) image saved to: {ndvi_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating FCC/NDVI from RGB-NIR: {e}", exc_info=True)
            return False

    # --- DEM Method (OpenTopography) ---
    def download_opentopo_dem(self, aoi_bbox: Dict[str, float], output_path: Path) -> bool:
        """Downloads SRTMGL1 DEM from OpenTopography for the AOI."""
        if not self.config.opentopo_api_key or self.config.opentopo_api_key == "YOUR_API_KEY_HERE_IF_NOT_USING_SECRETS_FALLBACK":
            logger.warning("OpenTopography API Key not configured. Skipping DEM download.")
            return False
        if output_path.exists():
            logger.info(f"DEM file already exists at {output_path}. Skipping download.")
            return True

        w,s,e,n = aoi_bbox['west'],aoi_bbox['south'],aoi_bbox['east'],aoi_bbox['north']
        api_url = (f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
                   f"&south={s}&north={n}&west={w}&east={e}&outputFormat=GTiff"
                   f"&API_Key={self.config.opentopo_api_key}")
        logger.info(f"Requesting DEM from OpenTopography (key redacted): "
                    f"{api_url.replace(self.config.opentopo_api_key, self.config.opentopo_api_key[:7] + '...')}")
        try:
            response = requests.get(api_url, stream=True, timeout=300)
            response.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            logger.info(f"âœ” DEM downloaded successfully to: {output_path}")
            return True
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"OpenTopography API HTTP error: {http_err}. Response: {response.text[:500] if hasattr(response, 'text') else 'N/A'}")
        except Exception as e:
            logger.error(f"Error during DEM download: {e}", exc_info=True)
        return False

    def calculate_hillshade(self, dem_path: Path, hillshade_path: Path) -> bool:
        """Calculates Hillshade from a DEM GeoTIFF."""
        if not dem_path.exists():
            logger.error(f"DEM file not found at {dem_path}. Cannot calculate hillshade.")
            return False
        logger.info(f"Calculating Hillshade from {dem_path} to {hillshade_path}")
        try:
            with rasterio.open(dem_path) as src_dem:
                elev = src_dem.read(1).astype('float32')
                if np.all(np.isnan(elev)) or (np.nanmax(elev)==np.nanmin(elev)):
                    logger.warning("DEM is flat or all NoData. Hillshade will be uniform.")
                    hs_data = np.full_like(elev, 128, dtype='uint8')
                else:
                    y_res, x_res = abs(src_dem.res[1]), abs(src_dem.res[0])
                    if x_res == 0 or y_res == 0: raise ValueError("DEM resolution is zero.")
                    gy, gx = np.gradient(elev, y_res, x_res)
                    slope = np.arctan(np.sqrt(gx**2 + gy**2))
                    aspect = np.arctan2(-gx, gy)
                    az, alt = np.deg2rad(315), np.deg2rad(45)
                    hs_float = np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az-aspect)
                    hs_min, hs_max = np.nanmin(hs_float), np.nanmax(hs_float)
                    if hs_max == hs_min: hs_data = np.full_like(hs_float, 128, dtype='uint8')
                    else: hs_data = (((hs_float-hs_min)/(hs_max-hs_min))*255).astype('uint8')
                    hs_data[np.isnan(hs_float)] = 0 # Handle NaNs

                meta = src_dem.meta.copy()
                meta.update(dtype='uint8', count=1, compress='lzw', nodata=0)
                with rasterio.open(hillshade_path, 'w', **meta) as dst_hs: dst_hs.write(hs_data, 1)
                logger.info(f"âœ” Hillshade saved to: {hillshade_path}")
            return True
        except Exception as e:
            logger.error(f"Error calculating hillshade: {e}", exc_info=True)
            return False

# Instantiate DataSourceManager
data_manager = DataSourceManager(config)


class AnomalyDetector:
    """Handles tiling of raster imagery and filtering of tiles based on criteria like NDVI."""

    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.tile_info_list: List[Dict[str, Any]] = [] 
        logger.info("AnomalyDetector initialized.")

    def generate_tiles_from_fcc(self, fcc_raster_path: Path) -> bool:
        """
        Generates PNG tiles from the input False-Colour Composite (FCC) GeoTIFF.
        Saves tile information (path, UTM coordinates, source offsets) to a CSV file.
        Returns True if successful (even if no tiles were generated but process completed), False on critical error.
        """
        logger.info(f"AnomalyDetector: Received FCC path for tiling: {fcc_raster_path}")
        if not fcc_raster_path.exists():
            logger.error(f"AnomalyDetector: FCC raster file not found at '{fcc_raster_path}'. Cannot generate tiles.")
            return False

        self.tile_info_list = [] # Reset for this run
        # Ensure the output directory for tiles exists
        try:
            self.config.tile_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"AnomalyDetector: Could not create tile directory at '{self.config.tile_dir}': {e}", exc_info=True)
            return False
        
        logger.info(f"Generating tiles from FCC: {fcc_raster_path} into {self.config.tile_dir}")
        logger.info(f"Tile size configured in METERS: {self.config.FULL_PIPELINE_TILE_SIZE_METERS_CONFIG} meters.")

        processed_tile_count = 0
        try:
            with rasterio.open(fcc_raster_path) as src_fcc:
                pixel_size_x, pixel_size_y = src_fcc.res[0], abs(src_fcc.res[1])
                logger.info(f"AnomalyDetector DEBUG: FCC Metada - CRS: {src_fcc.crs}, Count: {src_fcc.count}, Width: {src_fcc.width}, Height: {src_fcc.height}")
                logger.info(f"AnomalyDetector DEBUG: FCC pixel_size_x (X-resolution from src_fcc.res[0]) = {pixel_size_x}")
                logger.info(f"AnomalyDetector DEBUG: FCC pixel_size_y (Y-resolution from src_fcc.res[1]) = {pixel_size_y}")

                if pixel_size_x <= 1e-9 or pixel_size_y <= 1e-9: # Check for effectively zero or negative resolution
                    logger.error(f"AnomalyDetector ERROR: Invalid pixel resolution read from FCC: X={pixel_size_x}, Y={pixel_size_y}. Cannot tile.")
                    return False

                tile_size_meters = self.config.FULL_PIPELINE_TILE_SIZE_METERS_CONFIG
                logger.info(f"AnomalyDetector DEBUG: Using TILE_SIZE_METERS_CONFIG = {tile_size_meters}")

                tile_px_width = int(round(tile_size_meters / pixel_size_x)) # Round to nearest int
                tile_px_height = int(round(tile_size_meters / pixel_size_y)) # Round to nearest int
                logger.info(f"AnomalyDetector DEBUG: Calculated tile_px_width = {tile_px_width} pixels")
                logger.info(f"AnomalyDetector DEBUG: Calculated tile_px_height = {tile_px_height} pixels")

                if tile_px_width <= 0 or tile_px_height <= 0:
                    logger.error(f"AnomalyDetector ERROR: Calculated tile pixel dimensions are zero or negative "
                                 f"(Width={tile_px_width}px, Height={tile_px_height}px). "
                                 f"Check TILE_SIZE_METERS_CONFIG ({tile_size_meters}) and "
                                 f"raster resolutions (X={pixel_size_x}, Y={pixel_size_y}). Aborting tiling.")
                    return False
                
                logger.info(f"AnomalyDetector: Starting tiling loop with tile dimensions {tile_px_width}x{tile_px_height} pixels for an image of {src_fcc.width}x{src_fcc.height} pixels.")
                
                for r_offset in tqdm(range(0, src_fcc.height, tile_px_height), desc="Tiling FCC Image"):
                    for c_offset in range(0, src_fcc.width, tile_px_width):
                        # Individual tile processing wrapped in try-except
                        try:
                            current_w = min(tile_px_width, src_fcc.width - c_offset)
                            current_h = min(tile_px_height, src_fcc.height - r_offset)

                            if current_w < tile_px_width / 4 or current_h < tile_px_height / 4:
                                # logger.debug(f"Skipping very small edge tile at r={r_offset}, c={c_offset}, w={current_w}, h={current_h}")
                                continue

                            window = Window(c_offset, r_offset, current_w, current_h)
                            tile_raster_data = src_fcc.read(window=window) # FCC is typically 3-band Uint8

                            # Ensure data was actually read
                            if tile_raster_data is None or tile_raster_data.shape[1] == 0 or tile_raster_data.shape[2] == 0:
                                logger.warning(f"AnomalyDetector WARNING: Read empty data for tile at r_offset={r_offset}, c_offset={c_offset}. Skipping.")
                                continue

                            img_array_for_pil = np.transpose(tile_raster_data, (1, 2, 0))
                            pil_img = Image.fromarray(img_array_for_pil.astype(np.uint8))

                            tile_file_name = f"tile_fcc_{r_offset}_{c_offset}.png" # Consistent naming
                            tile_full_path = self.config.tile_dir / tile_file_name
                            pil_img.save(tile_full_path, optimize=True)
                            
                            center_map_x, center_map_y = src_fcc.xy(
                                r_offset + current_h / 2.0,
                                c_offset + current_w / 2.0
                            )
                            tile_utm_crs = src_fcc.crs.to_string() if src_fcc.crs else None

                            self.tile_info_list.append({
                                "path": str(tile_full_path.relative_to(BASE_WORK_DIR_CONFIG)),
                                "absolute_path": str(tile_full_path),
                                "utm_x": center_map_x, "utm_y": center_map_y, "utm_crs": tile_utm_crs,
                                "src_fcc_row_offset": r_offset, "src_fcc_col_offset": c_offset,
                                "tile_pixel_width": current_w, "tile_pixel_height": current_h
                            })
                            processed_tile_count += 1
                        except Exception as e_tile:
                            logger.error(f"AnomalyDetector ERROR: Failed to process or save tile at r_offset={r_offset}, c_offset={c_offset}: {e_tile}", exc_info=True)
                
                logger.info(f"AnomalyDetector: Tiling loop completed. Successfully processed and added info for {processed_tile_count} tiles.")
                if not self.tile_info_list and processed_tile_count == 0 :
                     logger.warning("AnomalyDetector WARNING: Tiling loop ran, but no valid tiles were added to the list. Check image dimensions or skipping logic.")
 

            # Attempt to save CSV even if list is empty 
            logger.info(f"AnomalyDetector: Attempting to save {len(self.tile_info_list)} tile records to CSV: {self.config.TILES_INDEX_CSV_PATH}")
            tiles_df = pd.DataFrame(self.tile_info_list) # Create DF from potentially empty list
            tiles_df.to_csv(self.config.TILES_INDEX_CSV_PATH, index=False)
            logger.info(f"AnomalyDetector: Tile index CSV (potentially empty if no tiles) saved to {self.config.TILES_INDEX_CSV_PATH}")
            return True 

        except Exception as e_outer:
            logger.error(f"AnomalyDetector CRITICAL ERROR in generate_tiles_from_fcc (outer try-except): {e_outer}", exc_info=True)
            return False # Critical error during file open or pre-loop setup


    def filter_tiles_by_ndvi(self, ndvi_raster_path: Path) -> Optional[pd.DataFrame]:
        """
        Filters tiles (from the generated tile_info_list or its CSV) based on NDVI values.
        Reads the corresponding window from the NDVI raster for each tile.
        Returns a DataFrame of candidate tiles, or None if a critical error occurs.
        """
        # Determine source of tile information
        if self.config.TILES_INDEX_CSV_PATH.exists():
            try:
                tiles_to_filter_df = pd.read_csv(self.config.TILES_INDEX_CSV_PATH)
                logger.info(f"Filtering {len(tiles_to_filter_df)} tiles from CSV {self.config.TILES_INDEX_CSV_PATH} using NDVI...")
            except Exception as e_csv:
                logger.error(f"Could not read tiles index CSV {self.config.TILES_INDEX_CSV_PATH}: {e_csv}. Will try in-memory list.")
                if not self.tile_info_list: 
                    logger.error("No tile information in CSV and in-memory tile_info_list is also empty. Cannot filter.")
                    return None
                tiles_to_filter_df = pd.DataFrame(self.tile_info_list)
                logger.info(f"Filtering {len(tiles_to_filter_df)} tiles from in-memory list using NDVI...")
        elif self.tile_info_list: 
            tiles_to_filter_df = pd.DataFrame(self.tile_info_list)
            logger.info(f"Filtering {len(tiles_to_filter_df)} tiles from in-memory list using NDVI (CSV not found).")
        else:
            logger.error("No tile information available (neither CSV nor in-memory list). Generate tiles first.")
            return None # Critical: no tile data to filter

        if not ndvi_raster_path.exists():
            logger.error(f"NDVI raster not found at {ndvi_raster_path}. Cannot filter tiles.")
            return None # Critical: no NDVI data

        if tiles_to_filter_df.empty:
            logger.info("No tiles found in the index to filter by NDVI. Returning empty candidate list.")
            # Save an empty candidate CSV for consistency
            pd.DataFrame().to_csv(self.config.CANDIDATES_CSV_PATH, index=False)
            return pd.DataFrame()

        candidate_flags = []
        mean_ndvi_values = []
        fraction_black_pixels = []

        try:
            with rasterio.open(ndvi_raster_path) as src_ndvi:
                logger.info(f"AnomalyDetector DEBUG: NDVI Metada - CRS: {src_ndvi.crs}, Count: {src_ndvi.count}, Width: {src_ndvi.width}, Height: {src_ndvi.height}")
                for _, tile_row in tqdm(tiles_to_filter_df.iterrows(), total=len(tiles_to_filter_df), desc="Filtering Tiles by NDVI"):
                    try: # Individual tile filtering error handling
                        col_off = int(tile_row["src_fcc_col_offset"])
                        row_off = int(tile_row["src_fcc_row_offset"])
                        width_px = int(tile_row["tile_pixel_width"])
                        height_px = int(tile_row["tile_pixel_height"])

                        # Check if window is within NDVI raster bounds
                        if col_off + width_px > src_ndvi.width or row_off + height_px > src_ndvi.height:
                            logger.warning(f"Tile window for {tile_row['path']} exceeds NDVI raster dimensions. Skipping.")
                            mean_ndvi = np.nan
                            frac_black = 1.0 # Treat as invalid
                            is_candidate = False
                        else:
                            window = Window(col_off, row_off, width_px, height_px)
                            ndvi_tile_data = src_ndvi.read(1, window=window)

                            if ndvi_tile_data.size == 0:
                                logger.warning(f"Empty NDVI tile data read for window {window}, path {tile_row['path']}. Skipping.")
                                mean_ndvi = np.nan
                                frac_black = 1.0
                                is_candidate = False
                            else:
                                nodata_mask_ndvi = np.isnan(ndvi_tile_data) # Assuming NoData is NaN in NDVI float
                                near_zero_mask_ndvi = np.isclose(ndvi_tile_data, 0.0, atol=1e-5)
                                
                                black_or_nodata_count_ndvi = np.sum(nodata_mask_ndvi | near_zero_mask_ndvi)
                                frac_black = black_or_nodata_count_ndvi / float(ndvi_tile_data.size) if ndvi_tile_data.size > 0 else 1.0
                                is_not_mostly_black = frac_black < self.config.FULL_PIPELINE_BLACK_TILE_THRESHOLD_CONFIG

                                if np.all(nodata_mask_ndvi):
                                    mean_ndvi = np.nan
                                else:
                                    mean_ndvi = np.nanmean(ndvi_tile_data[~nodata_mask_ndvi])
                                
                                is_low_ndvi = (not np.isnan(mean_ndvi)) and \
                                              (mean_ndvi < self.config.FULL_PIPELINE_NDVI_THRESHOLD_CONFIG)
                                is_candidate = is_low_ndvi and is_not_mostly_black
                        
                        candidate_flags.append(is_candidate)
                        mean_ndvi_values.append(mean_ndvi if not np.isnan(mean_ndvi) else None)
                        fraction_black_pixels.append(frac_black)

                    except Exception as e_filter_tile:
                        logger.error(f"Error filtering tile {tile_row.get('path', 'Unknown Path')}: {e_filter_tile}", exc_info=True)
                        candidate_flags.append(False)
                        mean_ndvi_values.append(None)
                        fraction_black_pixels.append(1.0) # Treat as problematic

            tiles_to_filter_df['mean_ndvi'] = mean_ndvi_values
            tiles_to_filter_df['fraction_black'] = fraction_black_pixels
            tiles_to_filter_df['is_candidate_tile'] = candidate_flags
            
            candidate_df = tiles_to_filter_df[tiles_to_filter_df['is_candidate_tile']].copy()
            
            # Ensure 'absolute_path' is present for AI analysis if 'path' was relative
            if 'path' in candidate_df.columns and 'absolute_path' not in candidate_df.columns:
                 candidate_df['absolute_path'] = candidate_df['path'].apply(lambda x: str(BASE_WORK_DIR_CONFIG / x))
            elif 'absolute_path' not in candidate_df.columns and not candidate_df.empty: # If even 'path' is missing
                 logger.error("Critical: 'path' or 'absolute_path' column missing in candidate_df after filtering.")
                 return None # Cannot proceed if paths are missing

            candidate_df.to_csv(self.config.CANDIDATES_CSV_PATH, index=False)
            logger.info(f"NDVI Filtering complete. Found {len(candidate_df)} candidate tiles out of {len(tiles_to_filter_df)}. "
                        f"Saved to: {self.config.CANDIDATES_CSV_PATH}")
            return candidate_df

        except Exception as e_outer_filter:
            logger.error(f"Critical error during NDVI tile filtering (outer try-except): {e_outer_filter}", exc_info=True)
            return None # Indicate critical failure


# Helper to encode local image file to base64
def encode_local_image_to_base64(image_path: Path) -> Optional[str]:
    """Encodes an image file at the given path to a Base64 string."""
    if not image_path.exists():
        logger.error(f"Image file not found for encoding: {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path} to Base64: {e}", exc_info=True)
        return None

class AIAnalyst:
    """Handles AI-powered image analysis using OpenAI vision models."""

    def __init__(self, config: ConfigurationManager, openai_model_name: str = "gpt-4o-mini"):
        self.config = config # Provides API key
        self.openai_model_name = openai_model_name
        self.client: Optional[OpenAI] = None
        self.analysis_history: List[Dict[str, Any]] = []

        if self.config.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.config.openai_api_key)
                logger.info(f"OpenAI client initialized for model: {self.openai_model_name}.")
                # self._test_openai_connection() # Test connection can be called explicitly later
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        else:
            logger.error("OpenAI API key not found in configuration. AIAnalyst (OpenAI) cannot function.")

    def test_openai_connection(self) -> bool:
        """Tests the OpenAI API connection with a simple request."""
        if not self.client:
            logger.error("OpenAI client not initialized. Cannot test connection.")
            return False
        try:
            logger.info(f"Testing OpenAI API connection with model: {self.openai_model_name}...")
            self.client.chat.completions.create(
                model=self.openai_model_name, # Use the configured model
                messages=[{"role": "user", "content": "Briefly say hello as an AI assistant."}],
                max_tokens=15
            )
            logger.info("OpenAI API connection test successful.")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection test failed for model {self.openai_model_name}: {e}", exc_info=False) # Less verbose for test
            return False

    def _create_openai_vision_prompt(self, context_text: str) -> str:
        """Creates a standardized prompt for archaeological feature detection from satellite imagery."""
        return (
            f"You are an expert archaeological analyst specializing in Amazon rainforest remote sensing, "
            f"tasked with identifying potential pre-Columbian earthworks from satellite imagery tiles. "
            f"The provided image is a tile from a Sentinel-2 False-Colour Composite (NIR-Red-Green), unless specified otherwise by context.\n\n"
            f"{context_text}\n\n"
            f"Examine the image carefully. Describe any signs of man-made or geometric earthworks larger "
            f"than approximately 80-100 meters (unless the image scale is clearly different). Look for:\n"
            f"- Unusual geometric patterns (circles, squares, rectangles, linear alignments, geoglyphs).\n"
            f"- Straight or unnaturally regular lines that could be ditches, embankments, causeways, or ancient roads.\n"
            f"- Mounds or depressions forming regular or distinct shapes.\n"
            f"- Distinct clearings or vegetation patterns that suggest deliberate, long-term land modification "
            f"rather than recent deforestation or natural features.\n\n"
            f"Provide a concise assessment. If no clear or significant signs of such features are visible, "
            f"state that clearly (e.g., 'No distinct man-made or geometric earthworks are readily apparent in this tile.'). "
            f"If there are potential signs, describe them specifically, their characteristics, and your confidence. "
            f"Mention if image quality or cloud cover limits your analysis."
        )

    def analyze_image_url_with_openai(self, image_url: str, image_context_text: str,
                                      max_tokens: int = 400, detail_level: str = "auto") -> Optional[str]:
        """Analyzes an image from a URL using the configured OpenAI Vision model."""
        if not self.client: return "OpenAI client not available."
        if not image_url: return "Image URL missing."

        full_prompt = self._create_openai_vision_prompt(image_context_text)
        logger.info(f"Analyzing image URL ({image_url[:60]}...) with OpenAI ({self.openai_model_name}).")
        
        messages_payload = [{"role": "user", "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}]
        
        if "mini" not in self.openai_model_name.lower() and self.openai_model_name not in ["gpt-3.5-turbo"]: # Models that support detail
            messages_payload[0]["content"][1]["image_url"]["detail"] = detail_level
        
        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model_name, messages=messages_payload, max_tokens=max_tokens
            )
            response_text = completion.choices[0].message.content
            usage = completion.usage
            self.analysis_history.append({
                "type": "image_url_analysis", "image_source": image_url, "prompt": full_prompt,
                "response": response_text, "model_used": completion.model or self.openai_model_name,
                "usage": dict(usage) if usage else None, "timestamp": datetime.now().isoformat()
            })
            logger.info(f"OpenAI URL analysis successful. Usage: {usage}")
            return response_text
        except Exception as e:
            logger.error(f"Error analyzing image URL with OpenAI: {e}", exc_info=True)
            self.analysis_history.append({ "type": "error_image_url_analysis", "image_source": image_url, "error": str(e)})
            return f"Error: OpenAI analysis failed for URL. {e}"

    def analyze_local_tile_with_openai(self, tile_path: Path, image_context_text: str,
                                       max_tokens: int = 400, detail_level: str = "auto") -> Optional[str]:
        """Analyzes a local image tile (PNG/JPG) by base64 encoding it for OpenAI Vision."""
        if not self.client: return "OpenAI client not available."
        if not tile_path.exists(): return f"Tile image not found: {tile_path}"

        base64_image = encode_local_image_to_base64(tile_path)
        if not base64_image: return "Failed to encode image."

        full_prompt = self._create_openai_vision_prompt(image_context_text)
        image_mime_type = f"image/{tile_path.suffix[1:].lower()}" # e.g., image/png
        
        logger.info(f"Analyzing local tile {tile_path.name} with OpenAI ({self.openai_model_name}).")
        
        messages_payload = [{"role": "user", "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:{image_mime_type};base64,{base64_image}"}
            }
        ]}]

        if "mini" not in self.openai_model_name.lower() and self.openai_model_name not in ["gpt-3.5-turbo"]:
             messages_payload[0]["content"][1]["image_url"]["detail"] = detail_level

        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model_name, messages=messages_payload, max_tokens=max_tokens
            )
            response_text = completion.choices[0].message.content
            usage = completion.usage
            self.analysis_history.append({
                "type": "local_tile_analysis", "image_source": str(tile_path.name), "prompt": full_prompt,
                "response": response_text, "model_used": completion.model or self.openai_model_name,
                "usage": dict(usage) if usage else None, "timestamp": datetime.now().isoformat()
            })
            logger.info(f"OpenAI local tile analysis successful. Usage: {usage}")
            return response_text
        except Exception as e:
            logger.error(f"Error analyzing local tile {tile_path.name} with OpenAI: {e}", exc_info=True)
            self.analysis_history.append({"type": "error_local_tile_analysis", "image_source": str(tile_path.name), "error": str(e)})
            return f"Error: OpenAI analysis failed for local tile. {e}"

    def analyze_candidate_tiles_batch_openai(self, candidate_tiles_df: pd.DataFrame,
                                             max_tokens_per_tile: int = 400,
                                             detail_level: str = "auto") -> pd.DataFrame:
        """Performs batch analysis on candidate tiles using OpenAI (sequentially)."""
        if not self.client:
            logger.error("OpenAI client not available. Skipping batch analysis.")
            if 'absolute_path' in candidate_tiles_df.columns or 'path' in candidate_tiles_df.columns:
                 candidate_tiles_df['openai_answer'] = "OpenAI client not available."
            return candidate_tiles_df # Return original or empty if path missing
        
        if candidate_tiles_df.empty:
            logger.info("No candidate tiles for OpenAI batch analysis.")
            return candidate_tiles_df

        logger.info(f"Starting OpenAI Vision batch analysis on {len(candidate_tiles_df)} tiles...")
        openai_answers = []
        
        # Determine path column to use
        path_col = 'absolute_path' if 'absolute_path' in candidate_tiles_df.columns else 'path'
        if path_col not in candidate_tiles_df.columns:
            logger.error(f"Path column ('{path_col}' or 'path') not found in candidate_tiles_df.")
            candidate_tiles_df['openai_answer'] = "Path column missing in DataFrame."
            return candidate_tiles_df

        for _, row in tqdm(candidate_tiles_df.iterrows(), total=len(candidate_tiles_df), desc="OpenAI Batch Tile Analysis"):
            tile_path_str = row[path_col]
            tile_path = Path(tile_path_str)
            context = f"This tile is located at approx. UTM X: {row.get('utm_x', 'N/A')}, UTM Y: {row.get('utm_y', 'N/A')}."
            if 'mean_ndvi' in row and pd.notna(row['mean_ndvi']):
                context += f" The mean NDVI for this tile is {row['mean_ndvi']:.3f}."
            
            answer = self.analyze_local_tile_with_openai(tile_path, context, max_tokens_per_tile, detail_level)
            openai_answers.append(answer)
            time.sleep(1) 

        results_df = candidate_tiles_df.copy()
        results_df['openai_answer'] = openai_answers
        
        # Save results
        results_df.to_csv(self.config.OPENAI_RESULTS_CSV_CONFIG, index=False) # Use global path
        logger.info(f"OpenAI batch analysis results for {len(results_df)} tiles saved to {self.config.OPENAI_RESULTS_CSV_CONFIG}")
        return results_df


class VisualizationManager:
    """Handles data visualization including rasters and interactive Folium maps."""

    def __init__(self, config: ConfigurationManager):
        self.config = config
        logger.info("VisualizationManager initialized.")

    def display_pil_image(self, pil_image: Optional[Image.Image], title: str, figsize: Tuple[int, int] = (8, 8)):
        """Displays a PIL Image object using matplotlib."""
        if pil_image is None:
            logger.warning(f"Cannot display PIL image for '{title}', image is None.")
            return
        try:
            plt.figure(figsize=figsize)
            plt.imshow(pil_image)
            plt.title(title, fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            logger.info(f"Displayed PIL Image: {title}")
        except Exception as e:
            logger.error(f"Error displaying PIL image '{title}': {e}", exc_info=True)

    def display_raster_image(self, raster_path: Path, title: str, 
                             cmap: Optional[str] = None, 
                             display_bands: Optional[List[int]] = None,
                             figsize: Tuple[int, int] = (10, 8),
                             stretch_percentiles: Optional[Tuple[float, float]] = (2, 98)) -> None:
        """Displays a geospatial raster file using rasterio and matplotlib."""
        if not raster_path.exists():
            logger.warning(f"Raster file not found for display: {raster_path}")
            print(f" File not found: {raster_path}. Cannot display '{title}'.")
            return
        logger.info(f"Attempting to display raster: {title} from {raster_path}")
        try:
            with rasterio.open(raster_path) as src:
                if display_bands: 
                    if any(b > src.count for b in display_bands):
                        logger.error(f"Invalid bands requested {display_bands} for raster with {src.count} bands.")
                        return
                    img_data_bands = src.read(display_bands)
                    img_display = np.transpose(img_data_bands, (1, 2, 0))
                    
                    # Basic contrast stretch for uint8 RGB-like images
                    if img_display.dtype == np.uint8:
                        # Assumes it's already well-scaled for display
                        pass
                    elif np.issubdtype(img_display.dtype, np.floating): 
                        # Apply percentile stretch to each band individually then stack
                        stretched_bands = []
                        for i in range(img_display.shape[2]): # Iterate over bands
                            band = img_display[:,:,i]
                            if stretch_percentiles:
                                vmin, vmax = np.nanpercentile(band, stretch_percentiles)
                                band = np.clip(band, vmin, vmax)
                                if vmax > vmin:
                                    band = (band - vmin) / (vmax - vmin) # Normalize 0-1
                                else: # Flat band
                                    band = np.zeros_like(band) 
                            else: # No stretch, just clip 0-1
                                band = np.clip(band, 0, 1)
                            stretched_bands.append(band)
                        img_display = (np.dstack(stretched_bands) * 255).astype(np.uint8)
                    else: # Other data types might need specific handling
                         logger.warning(f"Unsupported dtype {img_display.dtype} for multi-band display without specific scaling.")


                else: # Single-band image
                    img_display = src.read(1)
                    if nodata_val := src.nodata: # Mask nodata if present
                        img_display = np.ma.masked_equal(img_display, nodata_val)
                    if cmap is None: cmap = 'viridis' # Default cmap for single band
                    if stretch_percentiles and np.issubdtype(img_display.dtype, np.floating) :
                         vmin, vmax = np.nanpercentile(img_display, stretch_percentiles)
                         norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    else:
                         norm = None


                plt.figure(figsize=figsize)
                if len(img_display.shape) == 2 or img_display.shape[2] == 1: # Single band
                    plt.imshow(img_display, cmap=cmap, norm=norm if 'norm' in locals() else None)
                    plt.colorbar(label=f'{title} Value')
                else: # Multi-band (already (H,W,C) uint8)
                    plt.imshow(img_display)
                
                plt.title(title, fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                logger.info(f"Displayed raster: {title}")

        except Exception as e:
            logger.error(f"Failed to display raster {raster_path}: {e}", exc_info=True)

    def create_folium_map_with_openai_results(self, 
                                              results_df: pd.DataFrame, 
                                              map_output_path: Path,
                                              aoi_bbox: Dict[str,float]) -> bool:
        """Creates a Folium interactive map from a DataFrame with OpenAI analysis results."""
        if results_df.empty or not all(col in results_df.columns for col in ['utm_x', 'utm_y', 'utm_crs', 'openai_answer']):
            logger.warning("Results DataFrame is empty or missing required columns for Folium map. Map not created.")
            if results_df.empty: return True
            return False

        logger.info(f"Creating Folium map for {len(results_df)} OpenAI-analyzed tiles, will save to {map_output_path}")

        map_center_lat = (aoi_bbox['south'] + aoi_bbox['north']) / 2
        map_center_lon = (aoi_bbox['west'] + aoi_bbox['east']) / 2
        folium_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=8, tiles=None) # Start with no base tiles

        # Add base layers
        folium.TileLayer("CartoDB positron", name="CartoDB Positron (Light)").add_to(folium_map)
        folium.TileLayer("Esri.WorldImagery", name="Esri Satellite View", attr="Esri/Maxar").add_to(folium_map)
        
        df_copy = results_df.copy()
        df_copy['latitude_wgs84'] = np.nan
        df_copy['longitude_wgs84'] = np.nan
        
        path_col_to_use = 'absolute_path' if 'absolute_path' in df_copy.columns else 'path'


        unique_crs_list = df_copy['utm_crs'].dropna().unique()
        for utm_crs_str in unique_crs_list:
            try:
                transformer = PyProjTransformer.from_crs(utm_crs_str, "epsg:4326", always_xy=True) # ensure lon,lat order
                crs_mask = (df_copy['utm_crs'] == utm_crs_str)
                valid_coords_mask = crs_mask & df_copy['utm_x'].notna() & df_copy['utm_y'].notna()
                if not valid_coords_mask.any(): continue

                lons_reproj, lats_reproj = transformer.transform(
                    df_copy.loc[valid_coords_mask, 'utm_x'].values,
                    df_copy.loc[valid_coords_mask, 'utm_y'].values
                )
                df_copy.loc[valid_coords_mask, 'longitude_wgs84'] = lons_reproj
                df_copy.loc[valid_coords_mask, 'latitude_wgs84'] = lats_reproj
            except Exception as e:
                logger.error(f"Error reprojecting coordinates for CRS {utm_crs_str}: {e}")
        
        df_plottable = df_copy.dropna(subset=['latitude_wgs84', 'longitude_wgs84'])
        if df_plottable.empty and not results_df.empty:
            logger.warning("No tiles could be reprojected. Folium map will not have markers.")
        
        ## marker_cluster = folium.plugins.MarkerCluster(name="Analyzed Tiles").add_to(folium_map)
        marker_cluster = plugins.MarkerCluster(name="Analyzed Tiles").add_to(folium_map)

        for _, row in tqdm(df_plottable.iterrows(), total=len(df_plottable), desc="Adding markers to Folium map"):
            tile_display_path = Path(row[path_col_to_use]).name # Show only filename
            
            # Create popup HTML content
            html_popup = f"<h4>Tile: {tile_display_path}</h4>"
            html_popup += f"<p><b>Approx. Center (Lat, Lon):</b> {row['latitude_wgs84']:.5f}, {row['longitude_wgs84']:.5f}<br>"
            html_popup += f"<b>Mean NDVI (if available):</b> {row.get('mean_ndvi', 'N/A'):.3f}</p><hr>"
            html_popup += f"<p><b>OpenAI Analysis:</b></p>"
            html_popup += f"<div style='max-height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 5px; white-space: pre-wrap; font-size: smaller;'>{row['openai_answer']}</div>"
            
            # Add Base64 encoded image to popup
            try:
                image_to_encode_path = Path(row.get('absolute_path', BASE_WORK_DIR_CONFIG / row['path']))
                if image_to_encode_path.exists():
                    base64_img = encode_local_image_to_base64(image_to_encode_path)
                    if base64_img:
                        html_popup += f"<br><img src='data:image/png;base64,{base64_img}' alt='Tile Image' style='width:100%; max-width:300px; height:auto; margin-top:5px;'>"
                else:
                     logger.warning(f"Image for popup not found at {image_to_encode_path}")
            except Exception as e_img:
                logger.warning(f"Could not add image to popup for {tile_display_path}: {e_img}")

            iframe = folium.IFrame(html_popup, width=350, height=400)
            popup = folium.Popup(iframe, max_width=350)
            
            folium.Marker(
                location=[row['latitude_wgs84'], row['longitude_wgs84']],
                popup=popup,
                tooltip=f"{tile_display_path} - Click for OpenAI analysis",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)

        folium.LayerControl().add_to(folium_map)
        try:
            folium_map.save(str(map_output_path))
            logger.info(f"âœ” Folium interactive map saved to: {map_output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Folium map: {e}", exc_info=True)
            return False


class AmazonArchaeologyExplorer:
    """Orchestrates the archaeological discovery pipeline using various managers and analysts."""

    def __init__(self, openai_model_name_cp1: str = "gpt-4o-mini", 
                 openai_model_name_pipeline: str = "gpt-4o-mini"): # Can use different models
        
        if 'config' not in globals() or not isinstance(config, ConfigurationManager):
            logger.error("Global 'config' (ConfigurationManager) not found or invalid. Please run Cell 3.")
            raise ValueError("ConfigurationManager 'config' is required.")
        self.config = config

        self.data_manager = DataSourceManager(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        # AIAnalyst for Checkpoint 1
        self.ai_analyst_cp1 = AIAnalyst(self.config, openai_model_name=openai_model_name_cp1)
        # AIAnalyst for full pipeline
        self.ai_analyst_pipeline = AIAnalyst(self.config, openai_model_name=openai_model_name_pipeline)
        self.visualizer = VisualizationManager(self.config)
        
        self.gee_s2_cp1_info: Optional[Dict[str, Any]] = None
        self.stac_s2_pipeline_info: Optional[Dict[str, Any]] = None
        self.dem_pipeline_downloaded: bool = False
        self.fcc_pipeline_path: Optional[Path] = None
        self.ndvi_pipeline_path: Optional[Path] = None
        self.hillshade_pipeline_path: Optional[Path] = None
        self.candidate_tiles_pipeline_df: Optional[pd.DataFrame] = None
        self.analyzed_tiles_pipeline_df: Optional[pd.DataFrame] = None

        logger.info(f"AmazonArchaeologyExplorer initialized. CP1 AI: {openai_model_name_cp1}, Pipeline AI: {openai_model_name_pipeline}")

    def _ensure_global_gee_initialized(self) -> bool:
        """Checks the global GEE initialization flag."""
        if 'gee_initialized_successfully' not in globals() or not gee_initialized_successfully:
            logger.error("Global GEE flag 'gee_initialized_successfully' is false or not set. "
                         "Run GEE Authentication/Initialization cell (Cell 4).")
            return False
        return True

    def _ensure_openai_clients_ready(self) -> bool:
        """Checks if OpenAI clients in AIAnalysts are ready."""
        cp1_ready = self.ai_analyst_cp1.client is not None
        pipeline_ready = self.ai_analyst_pipeline.client is not None
        if not cp1_ready: logger.error("AIAnalyst for CP1 (OpenAI client) is not ready.")
        if not pipeline_ready: logger.error("AIAnalyst for Pipeline (OpenAI client) is not ready.")
        return cp1_ready and pipeline_ready

    def run_checkpoint_1_workflow(self) -> bool:
        """Executes the Checkpoint 1 workflow: GEE S2 image + OpenAI analysis."""
        logger.info(" Starting Checkpoint 1 Workflow...")
        if not self._ensure_global_gee_initialized() or not self._ensure_openai_clients_ready():
            logger.error("Prerequisites for Checkpoint 1 (GEE/OpenAI) not met.")
            return False

        # 1. Fetch Sentinel-2 image info and thumbnail URL from GEE
        self.gee_s2_cp1_info = self.data_manager.get_s2_image_info_via_gee(
            point_lon_lat=GEE_CP1_POINT_LON_LAT_CONFIG,
            point_name=GEE_CP1_POINT_NAME_CONFIG,
            buffer_radius_m=GEE_CP1_BUFFER_RADIUS_METERS_CONFIG,
            start_date_iso=GEE_CP1_START_DATE_CONFIG,
            end_date_iso=GEE_CP1_END_DATE_CONFIG,
            max_cloud_perc=GEE_CP1_MAX_CLOUD_COVERAGE_CONFIG,
            collection_id=GEE_CP1_IMAGE_COLLECTION_ID_CONFIG,
            vis_params=GEE_CP1_VIS_PARAMS_CONFIG,
            thumb_dimensions=GEE_CP1_THUMBNAIL_DIMENSIONS_CONFIG,
            thumb_format=GEE_CP1_THUMBNAIL_FORMAT_CONFIG
        )

        if not self.gee_s2_cp1_info or not self.gee_s2_cp1_info.get("thumbnail_url"):
            logger.error(" CP1: Failed to retrieve Sentinel-2 image thumbnail URL from GEE.")
            return False
        
        thumbnail_url = self.gee_s2_cp1_info["thumbnail_url"]
        gee_image_id = self.gee_s2_cp1_info["image_id"]
        logger.info(f"CP1: GEE Image ID for analysis: {gee_image_id}")
        print(f"\n--- Checkpoint 1: GEE Details ---")
        print(f"GEE Image ID: {gee_image_id}")
        print(f"Thumbnail URL: {thumbnail_url[:100]}...") # Print partial URL

        # 2. Download and display the GEE thumbnail
        pil_thumbnail = self.data_manager.download_gee_thumbnail_to_pil(thumbnail_url)
        if pil_thumbnail:
            plot_title_cp1 = (f'{GEE_CP1_POINT_NAME_CONFIG} - Sentinel-2 (True Color via GEE)\n'
                              f'Image ID: {gee_image_id} (<{GEE_CP1_MAX_CLOUD_COVERAGE_CONFIG}% cloud)')
            self.visualizer.display_pil_image(pil_thumbnail, plot_title_cp1)
        else:
            logger.warning("CP1: Failed to download or display GEE thumbnail, but proceeding with URL for AI.")

        # 3. Analyze the image using OpenAI Vision via the thumbnail URL
        openai_prompt_context_cp1 = (
            f"The GEE Image ID for this Sentinel-2 true-color thumbnail is '{gee_image_id}'. "
            f"It covers a point named '{GEE_CP1_POINT_NAME_CONFIG}' at approximately Lon/Lat: {GEE_CP1_POINT_LON_LAT_CONFIG}. "
            f"The visualization uses bands {GEE_CP1_VIS_PARAMS_CONFIG['bands']}."
        )
        
        openai_analysis_result = self.ai_analyst_cp1.analyze_image_url_with_openai(
            image_url=thumbnail_url,
            image_context_text=openai_prompt_context_cp1,
            detail_level="auto" if "mini" not in self.ai_analyst_cp1.openai_model_name.lower() else "low"
        )

        if not openai_analysis_result or "Error:" in openai_analysis_result :
            logger.error(f" CP1: OpenAI analysis failed or returned an error: {openai_analysis_result}")
            return False

        logger.info("CP1: OpenAI Analysis for GEE Image Thumbnail successful.")
        print(f"\n--- Checkpoint 1: OpenAI Analysis ---")
        print(f"OpenAI Model Used: {self.ai_analyst_cp1.openai_model_name}")
        print(f"GEE Dataset (Image) ID for this analysis: {gee_image_id}")
        print("\nAnalysis Result:")
        print(openai_analysis_result)
        print("--- End of OpenAI Analysis ---")
        
        logger.info(" Checkpoint 1 (GEE + OpenAI) workflow executed successfully.")
        return True

    # --- Methods for Full Pipeline (STAC S2, Local DEM, Tiling, Batch AI) ---
    def _pipeline_step_1_acquire_data(self) -> bool:
        logger.info("PIPELINE STEP 1: Acquiring STAC Sentinel-2 and OpenTopography DEM...")
        # Acquire STAC Sentinel-2
        self.stac_s2_pipeline_info = self.data_manager.search_best_sentinel2_scene_stac(
            MAIN_AOI_BBOX_CONFIG, STAC_START_DATE_CONFIG, STAC_END_DATE_CONFIG
        )
        if not self.stac_s2_pipeline_info:
            logger.error("Pipeline: Failed to find STAC S2 scene for main AOI. Cannot proceed with S2 processing.")

        else:
            logger.info(f"Pipeline: STAC S2 Scene selected: {self.stac_s2_pipeline_info.get('id', 'N/A')}")

        # Acquire DEM
        self.dem_pipeline_downloaded = self.data_manager.download_opentopo_dem(
            MAIN_AOI_BBOX_CONFIG, DEM_PATH_CONFIG
        )
        if not self.dem_pipeline_downloaded:
            logger.warning("Pipeline: Failed to download DEM. Hillshade and DEM-based analysis will be skipped.")
        else:
            logger.info(f"Pipeline: DEM downloaded to {DEM_PATH_CONFIG}")
        
        if not self.stac_s2_pipeline_info and not self.dem_pipeline_downloaded:
            logger.error("Pipeline: Both S2 STAC and DEM acquisition failed. Aborting.")
            return False
        return True

    def _pipeline_step_2_preprocess_rasters(self) -> bool:
        logger.info(" PIPELINE STEP 2: Preprocessing Local Rasters (RGB-NIR, FCC, NDVI, Hillshade)...")
        success_s2_proc = False
        if self.stac_s2_pipeline_info:
            if self.data_manager.create_rgb_nir_composite_from_stac(S2_RGBNIR_COMPOSITE_PATH_CONFIG):
                if self.data_manager.create_fcc_and_ndvi_from_rgbnir(
                    S2_RGBNIR_COMPOSITE_PATH_CONFIG, S2_FCC_PATH_CONFIG, NDVI_PATH_CONFIG
                ):
                    self.fcc_pipeline_path = S2_FCC_PATH_CONFIG
                    self.ndvi_pipeline_path = NDVI_PATH_CONFIG
                    logger.info("Pipeline: S2 FCC and NDVI created successfully.")
                    self.visualizer.display_raster_image(self.fcc_pipeline_path, "Pipeline FCC (STAC)", display_bands=[1,2,3])
                    self.visualizer.display_raster_image(self.ndvi_pipeline_path, "Pipeline NDVI (STAC)", cmap='RdYlGn')
                    success_s2_proc = True
                else: logger.error("Pipeline: Failed to create FCC/NDVI from STAC RGB-NIR composite.")
            else: logger.error("Pipeline: Failed to create STAC RGB-NIR composite.")
        else: logger.info("Pipeline: No STAC S2 scene selected, skipping S2 preprocessing.")

        success_dem_proc = False
        if self.dem_pipeline_downloaded and DEM_PATH_CONFIG.exists():
            if self.data_manager.calculate_hillshade(DEM_PATH_CONFIG, HILLSHADE_PATH_CONFIG):
                self.hillshade_pipeline_path = HILLSHADE_PATH_CONFIG
                logger.info("Pipeline: Hillshade created successfully.")
                self.visualizer.display_raster_image(DEM_PATH_CONFIG, "Pipeline DEM (OpenTopo)")
                self.visualizer.display_raster_image(self.hillshade_pipeline_path, "Pipeline Hillshade", cmap='gray')
                success_dem_proc = True
            else: logger.error("Pipeline: Failed to calculate hillshade from DEM.")
        else: logger.info("Pipeline: DEM not downloaded or path invalid, skipping hillshade.")
        
        # Proceed if at least one type of data was processed
        return success_s2_proc or success_dem_proc


    def _pipeline_step_3_tile_and_filter(self) -> bool:
        logger.info(" PIPELINE STEP 3: Tiling FCC and Filtering by NDVI...")
        if not self.fcc_pipeline_path or not self.fcc_pipeline_path.exists():
            logger.error("Pipeline: FCC raster for tiling not available. Cannot proceed with tiling.")
            return False
        if not self.ndvi_pipeline_path or not self.ndvi_pipeline_path.exists():
            logger.error("Pipeline: NDVI raster for filtering not available. Cannot filter tiles effectively.")
            return False 

        if not self.anomaly_detector.generate_tiles_from_fcc(self.fcc_pipeline_path):
            logger.error("Pipeline: Failed to generate tiles from FCC.")
            return False
        
        self.candidate_tiles_pipeline_df = self.anomaly_detector.filter_tiles_by_ndvi(self.ndvi_pipeline_path)
        if self.candidate_tiles_pipeline_df is None: # None indicates error
            logger.error("Pipeline: Error occurred during NDVI tile filtering.")
            return False
        if self.candidate_tiles_pipeline_df.empty:
            logger.warning("Pipeline: No candidate tiles found after NDVI filtering.")
 
        else:
            logger.info(f"Pipeline: Found {len(self.candidate_tiles_pipeline_df)} candidate tiles for AI analysis.")
            # Visualize a few sample candidate tiles
            num_samples = min(4, len(self.candidate_tiles_pipeline_df))
            if num_samples > 0:
                sample_df = self.candidate_tiles_pipeline_df.sample(num_samples, random_state=42)
                fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
                if num_samples == 1: axes = [axes] 
                for i_ax, (_, tile_row) in enumerate(sample_df.iterrows()):
                    try:
                        tile_img_path = Path(tile_row.get('absolute_path', BASE_WORK_DIR_CONFIG / tile_row['path']))
                        self.visualizer.display_pil_image(Image.open(tile_img_path), Path(tile_img_path).name, ax=axes[i_ax]) # Modify display_pil_image to take ax
                    except Exception as e_disp: logger.error(f"Error displaying sample tile: {e_disp}")
                plt.suptitle(f"Sample Candidate Tiles for Pipeline ({num_samples} of {len(self.candidate_tiles_pipeline_df)})")
                plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

        return True

    def _pipeline_step_4_batch_analyze_openai(self, max_tiles_to_analyze: Optional[int] = None) -> bool:
        logger.info("PIPELINE STEP 4: Batch Analyzing Candidate Tiles with OpenAI Vision...")
        if self.candidate_tiles_pipeline_df is None or self.candidate_tiles_pipeline_df.empty:
            logger.info("Pipeline: No candidate tiles to analyze with OpenAI. Skipping.")
            self.analyzed_tiles_pipeline_df = pd.DataFrame() # Ensure it's an empty DF
            return True 

        df_to_analyze = self.candidate_tiles_pipeline_df
        if max_tiles_to_analyze is not None and len(df_to_analyze) > max_tiles_to_analyze:
            logger.info(f"Pipeline: Sampling {max_tiles_to_analyze} tiles for OpenAI analysis from {len(df_to_analyze)} candidates.")
            df_to_analyze = df_to_analyze.sample(n=max_tiles_to_analyze, random_state=42).copy()
        
        self.analyzed_tiles_pipeline_df = self.ai_analyst_pipeline.analyze_candidate_tiles_batch_openai(
            df_to_analyze
        )
        if self.analyzed_tiles_pipeline_df is None: # Error during analysis
            logger.error("Pipeline: OpenAI batch analysis encountered an error.")
            return False
        
        logger.info(f"Pipeline: OpenAI batch analysis completed for {len(self.analyzed_tiles_pipeline_df)} tiles.")
        return True

    def _pipeline_step_5_report_and_map(self) -> None:
        logger.info(" PIPELINE STEP 5: Generating Final Report and Interactive Map...")
        print("\n" + "="*70)
        print("      AMAZON ARCHAEOLOGICAL AI DISCOVERY - FULL PIPELINE REPORT")
        print("="*70)
        print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"OpenAI Model Used for Tile Analysis: {self.ai_analyst_pipeline.openai_model_name}")

        if self.stac_s2_pipeline_info:
            print("\n--- STAC Sentinel-2 Scene ---")
            print(f"  ID: {self.stac_s2_pipeline_info.get('id', 'N/A')}, Date: {self.stac_s2_pipeline_info.get('datetime','N/A')[:10]}, Cloud: {self.stac_s2_pipeline_info.get('cloud_cover','N/A')}%")
        if self.dem_pipeline_downloaded:
            print(f"\n--- DEM Data ---")
            print(f"  Source: OpenTopography SRTMGL1, Path: {DEM_PATH_CONFIG.name}")
        
        if self.analyzed_tiles_pipeline_df is not None and not self.analyzed_tiles_pipeline_df.empty:
            print(f"\n--- OpenAI Analyzed Tiles ({len(self.analyzed_tiles_pipeline_df)}) ---")
            # Display a snippet of the results
            df_display = self.analyzed_tiles_pipeline_df.copy()
            df_display['tile_filename'] = df_display.get('absolute_path', df_display.get('path', 'N/A')).apply(lambda x: Path(x).name)
            print(df_display[['tile_filename', 'utm_x', 'utm_y', 'mean_ndvi', 'openai_answer']].head().to_string())
            
            self.visualizer.create_folium_map_with_openai_results(
                self.analyzed_tiles_pipeline_df,
                FOLIUM_MAP_HTML_CONFIG, # Use global path
                MAIN_AOI_BBOX_CONFIG
            )
            logger.info(f"Pipeline: Folium map generated at {FOLIUM_MAP_HTML_CONFIG}")
            # Display map inline
            display(IFrame(src=str(FOLIUM_MAP_HTML_CONFIG.relative_to(BASE_WORK_DIR_CONFIG)), width='100%', height=600))

        else:
            print("\n--- OpenAI Analyzed Tiles ---")
            print("No tiles were analyzed or analysis produced no results.")
        
        print("\n" + "="*70)
        logger.info("Pipeline: Final report generated.")

    def run_full_pipeline(self, max_tiles_for_ai_analysis: Optional[int] = 20) -> bool:
        """Executes the complete data processing and analysis pipeline for a wider AOI."""
        logger.info("INITIATING FULL ARCHAEOLOGICAL PROSPECTION PIPELINE")
        if not self._ensure_global_gee_initialized() or not self._ensure_openai_clients_ready(): # GEE might not be used here but good to check if config wants it
            logger.error("Full Pipeline: Prerequisites (OpenAI client) not met. Aborting.")
            return False # OpenAI client is definitely needed
        
        if not self._pipeline_step_1_acquire_data(): return False
        if not self._pipeline_step_2_preprocess_rasters(): return False # Can proceed if one is True
        if not self._pipeline_step_3_tile_and_filter(): return False
        if not self._pipeline_step_4_batch_analyze_openai(max_tiles_to_analyze=max_tiles_for_ai_analysis): return False
        self._pipeline_step_5_report_and_map()
        
        logger.info("FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        return True


# Ensure GEE is initialized 
if 'gee_initialized_successfully' not in globals() or not gee_initialized_successfully:
    logger.error("GEE was not set up correctly in Cell 4. Please run Cell 4 and authenticate if prompted.")
    print(" GEE Setup Incomplete. Cannot run Checkpoint 1 with GEE.")
    cp1_completed_flag = False
else:
    logger.info("Proceeding with Checkpoint 1 execution using GEE and OpenAI...")
    
    # Instantiate the explorer for Checkpoint 1
    explorer_cp1 = AmazonArchaeologyExplorer(
        openai_model_name_cp1=GEE_CP1_VIS_PARAMS_CONFIG.get("openai_model", "gpt-4o-mini") 
    )
    
    if not config.is_valid: # Check if ConfigurationManager loaded keys properly
        print(" Configuration is invalid (missing API keys or GEE Project ID). Cannot run Checkpoint 1.")
        cp1_completed_flag = False
    else:
        cp1_completed_flag = explorer_cp1.run_checkpoint_1_workflow()

        if cp1_completed_flag:
            logger.info("Checkpoint 1 workflow completed successfully!")
        else:
            logger.error("Checkpoint 1 workflow failed. Please review logs above.")
            print("Ensure GEE is authenticated, GEE_PROJECT_ID is correct and has Earth Engine API enabled, and OPENAI_API_KEY is valid.")



if 'config' not in globals() or not isinstance(config, ConfigurationManager) or not config.is_valid:
    logger.error("Global 'config' is not valid or not found. Please run Cell(ConfigurationManager) and ensure keys are set.")
    print(" Critical Error: ConfigurationManager is not properly set up. Cannot proceed.")
    pipeline_step1_success = False
else:
    logger.info(" Initializing Explorer for Full AOI Pipeline...")

    explorer_pipeline = AmazonArchaeologyExplorer(
        openai_model_name_cp1="gpt-4o-mini", # Not used in this pipeline run
        openai_model_name_pipeline="gpt-4o-mini" # Model for batch tile analysis
    )


    if not explorer_pipeline.ai_analyst_pipeline.client:
        logger.error("OpenAI client for pipeline analysis is not initialized. Check API Key.")
        pipeline_step1_success = False
    elif not explorer_pipeline.ai_analyst_pipeline.test_openai_connection():
        logger.error("OpenAI connection test failed for pipeline AI analyst. Check API Key and network.")
        pipeline_step1_success = False
    else:
        logger.info("OpenAI connection for pipeline AI analyst successful.")
        # --- Execute Pipeline Step 1: Data Acquisition ---
        pipeline_step1_success = explorer_pipeline._pipeline_step_1_acquire_data()

        if pipeline_step1_success:
            logger.info(" Full Pipeline - Step 1 (Data Acquisition) completed.")
            if explorer_pipeline.stac_s2_pipeline_info:
                 print(f"  STAC Sentinel-2 Scene ID selected for pipeline: {explorer_pipeline.stac_s2_pipeline_info.get('id', 'N/A')}")
            if explorer_pipeline.dem_pipeline_downloaded:
                 print(f"  DEM data for pipeline downloaded to: {DEM_PATH_CONFIG}")
            elif config.opentopo_api_key: 
                 print(f"   DEM data download for pipeline failed or was skipped, though API key was present.")
            else:
                 print(f"   DEM data download for pipeline skipped as OpenTopography API key was not available.")
        else:
            logger.error(" Full Pipeline - Step 1 (Data Acquisition) failed. See logs for details.")
            print("  Data acquisition for the full pipeline failed. Check logs. Subsequent steps will be affected.")


pipeline_step2_success = False
if 'pipeline_step1_success' in locals() and pipeline_step1_success:
    logger.info(" Executing Full Pipeline - Step 2: Preprocess Local Rasters...")
    pipeline_step2_success = explorer_pipeline._pipeline_step_2_preprocess_rasters()
    if pipeline_step2_success:
        logger.info(" Full Pipeline - Step 2 (Raster Preprocessing) completed.")
        print("\n--- Full Pipeline: Raster Preprocessing Summary ---")
        if explorer_pipeline.fcc_pipeline_path and explorer_pipeline.fcc_pipeline_path.exists():
            print(f"  FCC (for tiling) generated at: {explorer_pipeline.fcc_pipeline_path}")
        else:
            print("  FCC (for tiling) was NOT generated.")
        if explorer_pipeline.ndvi_pipeline_path and explorer_pipeline.ndvi_pipeline_path.exists():
            print(f"  NDVI (for filtering) generated at: {explorer_pipeline.ndvi_pipeline_path}")
        else:
            print("  NDVI (for filtering) was NOT generated.")
        if explorer_pipeline.hillshade_pipeline_path and explorer_pipeline.hillshade_pipeline_path.exists():
            print(f"  Hillshade generated at: {explorer_pipeline.hillshade_pipeline_path}")
        else:
            print("  Hillshade was NOT generated (DEM might have been unavailable).")
    else:
        logger.error(" Full Pipeline - Step 2 (Raster Preprocessing) failed or produced no outputs. See logs.")
        print("   Raster preprocessing for the full pipeline failed. Check logs.")
else:
    logger.warning("Skipping Full Pipeline - Step 2 (Raster Preprocessing) due to failure in Step 1.")
    print("  Skipping raster preprocessing as data acquisition (Step 1) was not successful.")



pipeline_step3_success = False 

if 'pipeline_step2_success' in locals() and pipeline_step2_success: 

    if hasattr(explorer_pipeline, 'fcc_pipeline_path') and explorer_pipeline.fcc_pipeline_path and explorer_pipeline.fcc_pipeline_path.exists() and \
       hasattr(explorer_pipeline, 'ndvi_pipeline_path') and explorer_pipeline.ndvi_pipeline_path and explorer_pipeline.ndvi_pipeline_path.exists():
        
        logger.info("Executing Full Pipeline - Step 3: Tile FCC and Filter by NDVI...")
        
        pipeline_step3_success = explorer_pipeline._pipeline_step_3_tile_and_filter() 
        
        # Summary information output section
        if pipeline_step3_success:
            logger.info("Full Pipeline - Step 3 (Tiling and Filtering) completed.")
            print("\n--- Full Pipeline: Tiling and Filtering Summary ---")
            if explorer_pipeline.candidate_tiles_pipeline_df is not None:
                print(f"  Number of candidate tiles identified for AI analysis: {len(explorer_pipeline.candidate_tiles_pipeline_df)}")
                if not explorer_pipeline.candidate_tiles_pipeline_df.empty:
                    print(f"  Candidate tiles CSV saved to: {config.CANDIDATES_CSV_PATH}") 
                    print("  Sample of candidate tiles (if any) displayed above.") 
                else:
                    print("  No candidate tiles met the filtering criteria.")
            else: 

                print("  Tiling/filtering step reported success, but candidate_tiles_pipeline_df is None (unexpected).")
        else: 
            logger.error("Full Pipeline - Step 3 (Tiling and Filtering) failed. See logs for details from within the method.")
            print("  Tiling and/or filtering for the full pipeline failed. Check logs from the executed function.")
    else:
        logger.warning("Skipping Full Pipeline - Step 3: Necessary FCC or NDVI raster from Step 2 is missing or path attribute not found in explorer_pipeline.")
        print("  INFO: Skipping tiling and filtering as necessary FCC or NDVI rasters were not properly generated or available in Step 2.")

else:
    logger.warning("Skipping Full Pipeline - Step 3 (Tiling and Filtering) due to failure in prior steps (Step 1 or 2).")
    print("  INFO: Skipping tiling and filtering as prior pipeline steps were not successful.")

# Sample tile visualization part 
if pipeline_step3_success and hasattr(explorer_pipeline, 'candidate_tiles_pipeline_df') and \
   explorer_pipeline.candidate_tiles_pipeline_df is not None and \
   not explorer_pipeline.candidate_tiles_pipeline_df.empty:
    
    logger.info("Displaying sample candidate tiles (if any):")
    samples_to_show = min(4, len(explorer_pipeline.candidate_tiles_pipeline_df))
    if samples_to_show > 0:
        sample_df = explorer_pipeline.candidate_tiles_pipeline_df.sample(samples_to_show, random_state=42)

        fig, axes_list = plt.subplots(1, samples_to_show, figsize=(5 * samples_to_show, 5.5)) # Slightly adjusted figsize
        if samples_to_show == 1:
            axes_list = [axes_list] 
            
        for i_ax, (_, tile_row) in enumerate(sample_df.iterrows()):
            ax_current = axes_list[i_ax] # Current subplot axis to use
            try:
                path_col_to_use_sample = 'absolute_path' if 'absolute_path' in tile_row and pd.notna(tile_row['absolute_path']) else 'path'
                tile_img_path_str = tile_row.get(path_col_to_use_sample)

                if tile_img_path_str:
                    tile_img_path = Path(tile_img_path_str)
                    if not tile_img_path.is_absolute():
                        tile_img_path = BASE_WORK_DIR_CONFIG / tile_img_path

                    if tile_img_path.exists():
                        img = Image.open(tile_img_path)
                        ax_current.imshow(img)
                        ax_current.set_title(Path(tile_img_path).name, fontsize=8)
                        ax_current.axis('off')
                    else:
                        logger.warning(f"Sample tile image not found at: {tile_img_path}")
                        ax_current.set_title("Img not found", fontsize=8); ax_current.axis('off')
                else:
                    logger.warning("Path column missing or NaN in sample tile row.")
                    ax_current.set_title("Path N/A", fontsize=8); ax_current.axis('off')

            except Exception as e_disp: 
                logger.error(f"Error displaying sample tile: {e_disp}")
                ax_current.set_title("Error loading", fontsize=8); ax_current.axis('off')
                
        plt.suptitle(f"Sample Candidate Tiles for Pipeline ({samples_to_show} of {len(explorer_pipeline.candidate_tiles_pipeline_df)})", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()
else:

    if 'pipeline_step3_success' in locals() and pipeline_step3_success:
         logger.info("No candidate tiles to display samples for (candidate list might be empty).")



pipeline_step4_success = False
if 'pipeline_step3_success' in locals() and pipeline_step3_success:

    if explorer_pipeline.candidate_tiles_pipeline_df is not None and not explorer_pipeline.candidate_tiles_pipeline_df.empty:
        logger.info(" Executing Full Pipeline - Step 4: Batch AI Analysis of Candidate Tiles...")

        num_tiles_to_analyze_this_run = min(20, len(explorer_pipeline.candidate_tiles_pipeline_df)) 
        logger.info(f"Will attempt to analyze up to {num_tiles_to_analyze_this_run} candidate tiles with OpenAI.")

        pipeline_step4_success = explorer_pipeline._pipeline_step_4_batch_analyze_openai(
            max_tiles_to_analyze=num_tiles_to_analyze_this_run
        )
        if pipeline_step4_success:
            logger.info("Full Pipeline - Step 4 (Batch AI Analysis) completed.")
            print("\n--- Full Pipeline: Batch AI Analysis Summary ---")
            if explorer_pipeline.analyzed_tiles_pipeline_df is not None:
                 analyzed_count = len(explorer_pipeline.analyzed_tiles_pipeline_df[explorer_pipeline.analyzed_tiles_pipeline_df['openai_answer'].notna() & ~explorer_pipeline.analyzed_tiles_pipeline_df['openai_answer'].str.contains("Error", na=False)])
                 print(f"  Successfully received OpenAI analysis for {analyzed_count} out of {len(explorer_pipeline.analyzed_tiles_pipeline_df)} submitted tiles.")
                 print(f"  AI analysis results saved to: {config.OPENAI_RESULTS_CSV_CONFIG}") # Using global config path
                 if not explorer_pipeline.analyzed_tiles_pipeline_df.empty:
                    print("\n  Sample of AI Analysis Results:")
                    display_ai_results = explorer_pipeline.analyzed_tiles_pipeline_df.copy()
                    display_ai_results['tile_filename'] = display_ai_results.get('absolute_path', display_ai_results.get('path', 'N/A')).apply(lambda x: Path(x).name)
                    display_ai_results['openai_answer_snippet'] = display_ai_results['openai_answer'].str.slice(0, 100) + "..."
                    print(display_ai_results[['tile_filename', 'mean_ndvi', 'openai_answer_snippet']].head())
            else:
                print("  AI analysis step completed, but no results DataFrame was generated (e.g. if input was empty or error in method).")
        else:
            logger.error(" Full Pipeline - Step 4 (Batch AI Analysis) failed. See logs.")
            print("  Batch AI analysis for the full pipeline failed. Check logs.")
    else:
        logger.info("No candidate tiles from Step 3 to analyze with AI. Skipping Step 4.")
        print("  No candidate tiles were available for AI analysis. Skipping this step.")
        pipeline_step4_success = True # Not a failure if there's no input
else:
    logger.warning("Skipping Full Pipeline - Step 4 (Batch AI Analysis) due to failure in prior steps.")
    print("  Skipping batch AI analysis as prior pipeline steps were not successful.")


if 'pipeline_step4_success' in locals() and pipeline_step4_success: 
    logger.info(" Executing Full Pipeline - Step 5: Final Report and Interactive Map...")

    explorer_pipeline._pipeline_step_5_report_and_map()
    logger.info(" Full Pipeline - Step 5 (Reporting and Mapping) completed.")
    print("\n--- Full Pipeline: Final Report and Map Generation ---")
    print(f"  Final summary report printed above.")
    if config.FOLIUM_MAP_HTML_CONFIG.exists(): # Check global path
        print(f"  Interactive Folium map saved to: {config.FOLIUM_MAP_HTML_CONFIG}")
        print(f"  The map should also be displayed inline above this message if successful.")
    else:
        print(f"  Interactive Folium map was NOT generated or saved to the expected path: {config.FOLIUM_MAP_HTML_CONFIG}")
else:
    logger.warning("Skipping Full Pipeline - Step 5 (Reporting and Mapping) due to failure in prior steps or no data to report.")
    print("   Skipping final report and map generation as prior pipeline steps were not successful or produced no data for mapping.")

logger.info("Full AOI Archaeological Discovery Pipeline run attempt finished.")
print("\n Full AOI Pipeline processing attempt is complete. Check logs and outputs for details.")

