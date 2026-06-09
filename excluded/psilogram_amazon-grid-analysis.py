!pip install -q rasterio


# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# PURPOSE: Load required libraries and initialize paths and parameters
# ==============================================================================

print("--- Loading Libraries ---")

# --------------------------------------------------------------------------
# Core libraries for mapping, data handling, and geospatial analysis
# --------------------------------------------------------------------------
import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, box
from branca.colormap import linear
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore, gaussian_kde

# --------------------------------------------------------------------------
# Folium (interactive mapping)
# --------------------------------------------------------------------------
import folium
from folium.plugins import FeatureGroupSubGroup, FastMarkerCluster
from folium.utilities import JsCode

print("--- Initializing Configuration ---")

# ==============================================================================
# CONFIGURATION: Input Paths and Parameters
# ==============================================================================

# Root path for all inputs (update for local or Kaggle use)
KAGGLE_INPUT_ROOT = "/kaggle/input/"

# --------------------------------------------------------------------------
# Input file paths
# --------------------------------------------------------------------------
arch_sites_csv_path = os.path.join(
    KAGGLE_INPUT_ROOT, "collect-amazon-sites", "geoglyphs.csv"
)
grouped_lidar_metadata_csv_path = os.path.join(
    KAGGLE_INPUT_ROOT,
    "nasa-lidar-grouped-pngs/preprocessed_lidar_groups",
    "grouped_dtm_hillshade_metadata.csv"
)
original_rivers_geojson_path = os.path.join(
    KAGGLE_INPUT_ROOT, "amazon-river-map", "Amazon River.geojson"
)
corrected_rivers_geojson_path = os.path.join(
    KAGGLE_INPUT_ROOT, "amazon-river-map-processed", "amazon_rivers_wgs84.geojson"
)

# --------------------------------------------------------------------------
# Spatial parameters
# --------------------------------------------------------------------------
TARGET_PROJECTED_CRS = "EPSG:31980"  # Use an appropriate UTM or SIRGAS projection
CELL_SIZE_KM = 20.0                  # Base analysis grid resolution
RIVER_BUFFER_METERS = 50            # Buffer width for rivers (left/right bank)

# ==============================================================================
# STEP: Load and Prepare River Feature Points (Sources & Intersections)
# ==============================================================================

PROCESSED_RIVER_DATASET_SLUG = "amazon-river-map-processed"
RIVER_FEATURES_FILENAME = "amazon_river_features.geojson"
river_features_path = os.path.join(KAGGLE_INPUT_ROOT, PROCESSED_RIVER_DATASET_SLUG, RIVER_FEATURES_FILENAME)

# Create empty containers
sources_gdf_proj = gpd.GeoDataFrame()
intersections_gdf_proj = gpd.GeoDataFrame()

if os.path.exists(river_features_path):
    try:
        river_features_gdf = gpd.read_file(river_features_path)
        print(f"Loaded {len(river_features_gdf)} river feature points.")
        
        # Separate source points and intersection points
        sources_gdf_wgs84 = river_features_gdf[river_features_gdf['feature_type'] == 'source']
        intersections_gdf_wgs84 = river_features_gdf[river_features_gdf['feature_type'] == 'intersection']
        
        # Project both feature sets to analysis CRS
        if not sources_gdf_wgs84.empty:
            sources_gdf_proj = sources_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
        if not intersections_gdf_wgs84.empty:
            intersections_gdf_proj = intersections_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
        
        print(f"Prepared {len(sources_gdf_proj)} river sources and {len(intersections_gdf_proj)} intersections for analysis.")
    
    except Exception as e:
        print(f"â�Œ Error preparing river features data: {e}")



# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# PURPOSE: Provide reusable geospatial utility functions for loading,
#          projecting, and generating grid structures across the Amazon Basin.
# ==============================================================================

# --------------------------------------------------------------------------
# Function: Safely convert any value to string
# --------------------------------------------------------------------------
def get_str_val(value, default_if_empty="N/A"):
    """
    Safely converts a value to string, handling cases like None, NaN, or "nan" string.

    Parameters:
    ----------
    value : Any
        Input value to convert to string.
    default_if_empty : str
        Value to return if input is invalid or empty.

    Returns:
    -------
    str
        Cleaned and converted string or fallback default.
    """
    if pd.isna(value): return default_if_empty
    s_value = str(value).strip()
    return default_if_empty if not s_value or s_value.lower() == 'nan' else s_value


# --------------------------------------------------------------------------
# Function: Reproject a GeoJSON file to WGS84 and save output
# --------------------------------------------------------------------------
def reproject_geojson_to_wgs84(
    input_geojson_path: str, 
    output_geojson_path: str, 
    default_source_crs: str = "EPSG:3857"
) -> bool:
    """
    Reprojects a GeoJSON file to WGS84 (EPSG:4326) and saves the result.
    
    Parameters:
    ----------
    input_geojson_path : str
        Filepath to the original GeoJSON.
    output_geojson_path : str
        Filepath where the WGS84-reprojected GeoJSON will be saved.
    default_source_crs : str
        CRS to assume if the input file lacks CRS metadata.

    Returns:
    -------
    bool
        True if success, False if failure (e.g. file not found or projection error).
    """
    print(f"\n--- Reprojecting GeoJSON: {os.path.basename(input_geojson_path)} ---")
    if not os.path.exists(input_geojson_path):
        print(f"ERROR: Input GeoJSON not found at {input_geojson_path}.")
        return False

    try:
        gdf_original = gpd.read_file(input_geojson_path)
        print(f"Successfully read original GeoJSON. Number of features: {len(gdf_original)}")

        # Determine and assign CRS
        if gdf_original.crs:
            print(f"GeoJSON file's detected internal CRS: {gdf_original.crs}")
        else:
            print(f"Warning: No CRS found. Assuming default source CRS: {default_source_crs}")
            gdf_original.crs = default_source_crs

        # Skip reprojection if already in WGS84
        if gdf_original.crs.to_epsg() == 4326:
            print("GeoJSON is already in EPSG:4326. Copying file.")
            shutil.copy2(input_geojson_path, output_geojson_path)
        else:
            print(f"Reprojecting to WGS84...")
            gdf_wgs84 = gdf_original.to_crs("EPSG:4326")
            gdf_wgs84.to_file(output_geojson_path, driver="GeoJSON")

        print(f"Processed GeoJSON saved to: {output_geojson_path}")
        return output_geojson_path

    except Exception as e:
        print(f"â�Œ Error reprojecting GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return False


# --------------------------------------------------------------------------
# Function: Create a uniform analysis grid clipped to a basin polygon
# --------------------------------------------------------------------------
def create_analysis_grid(basin_polygon_wgs84, cell_size_km):
    """
    Generates a rectangular grid of polygons (cells) covering the bounding
    box of a basin polygon, then clips the grid to only cells intersecting
    the polygon.

    Parameters:
    ----------
    basin_polygon_wgs84 : shapely geometry
        A polygon (or multipolygon) defining the Amazon basin boundary.
    cell_size_km : float
        Size of each cell in kilometers (both height and width).

    Returns:
    -------
    GeoDataFrame
        Filtered grid clipped to the basin extent.
    """
    if basin_polygon_wgs84 is None or basin_polygon_wgs84.is_empty:
        print("ERROR: Basin polygon is not valid. Cannot create grid.")
        return None

    # Determine bounds and convert degrees to kilometers
    min_lon, min_lat, max_lon, max_lat = basin_polygon_wgs84.bounds
    print(f"Basin bounds (WGS84): MinLon {min_lon:.2f}, MinLat {min_lat:.2f}, MaxLon {max_lon:.2f}, MaxLat {max_lat:.2f}")

    center_lat_rad = np.deg2rad((min_lat + max_lat) / 2)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(center_lat_rad)
    if km_per_deg_lon == 0:
        return None

    delta_lat_approx = cell_size_km / km_per_deg_lat
    delta_lon_approx = cell_size_km / km_per_deg_lon
    print(f"Approx. cell size in degrees: dLat={delta_lat_approx:.4f}, dLon={delta_lon_approx:.4f}")

    # Create bounding box grid
    lon_coords = np.arange(min_lon, max_lon, delta_lon_approx)
    lat_coords = np.arange(min_lat, max_lat, delta_lat_approx)
    grid_cells = [
        box(lon, lat, lon + delta_lon_approx, lat + delta_lat_approx)
        for lon in lon_coords for lat in lat_coords
    ]
    if not grid_cells:
        return None

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    print(f"Generated {len(grid_gdf)} raw grid cells. Filtering by intersection with basin polygon...")
    grid_gdf_in_basin = grid_gdf[grid_gdf.intersects(basin_polygon_wgs84)]
    print(f"Retained {len(grid_gdf_in_basin)} grid cells intersecting the basin.")
    return grid_gdf_in_basin


# --------------------------------------------------------------------------
# Function: Load and simplify Amazon basin polygon geometry
# --------------------------------------------------------------------------
def load_amazon_basin_polygon(geojson_path, target_crs="EPSG:4326", simplify_tol=0.05):
    """
    Loads a GeoJSON file representing the Amazon basin boundary and optionally
    simplifies and reprojects it to a target CRS.

    Parameters:
    ----------
    geojson_path : str
        Path to the basin polygon GeoJSON file.
    target_crs : str
        Desired coordinate reference system (default is EPSG:4326).
    simplify_tol : float
        Tolerance for simplifying geometry (in degrees).

    Returns:
    -------
    shapely geometry
        Unified (multi)polygon representing the simplified basin area.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Polygon file not found: {geojson_path}")

    basin_gdf = gpd.read_file(geojson_path)

    if basin_gdf.crs != target_crs:
        print(f"ğŸ”„ Reprojecting basin polygon from {basin_gdf.crs} to {target_crs}")
        basin_gdf = basin_gdf.to_crs(target_crs)

    unified = basin_gdf.unary_union

    if simplify_tol:
        unified = unified.simplify(simplify_tol)

    return unified



# ==============================================================================
# SECTION 1: GEOSPATIAL PREPROCESSING â€“ BASIN, SITES, RIVERS, LIDAR, GRID
# PURPOSE : Load and reproject all relevant geospatial inputs for Amazon analysis
# ==============================================================================

# -------------------------------------------------------------------
# Utility function: Load and simplify Amazon basin polygon from GeoJSON
# -------------------------------------------------------------------
def load_amazon_basin_polygon(geojson_path, target_crs="EPSG:4326", simplify_tol=0.05):
    """
    Load Amazon basin boundary from a GeoJSON file and optionally reproject and simplify it.

    Parameters:
    ----------
    geojson_path : str
        Path to the input basin polygon file (GeoJSON).
    target_crs : str
        Target CRS for output geometry (default = EPSG:4326).
    simplify_tol : float
        Simplification tolerance (degrees); 0 = no simplification.

    Returns:
    -------
    shapely.geometry.Polygon or MultiPolygon
        Unified simplified polygon.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Polygon file not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)

    if gdf.crs != target_crs:
        print(f"ğŸ”„ Reprojecting basin from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    union = gdf.unary_union
    return union.simplify(simplify_tol) if simplify_tol else union


# ==============================================================================
# BEGIN PHASE 1
# PURPOSE: Load and project all geospatial data to a unified analysis CRS
# ==============================================================================

print("\n--- PHASE 1: Preparing and Projecting All Geospatial Data ---")

# -------------------------------------------------------------------
# Load and prepare the Amazon basin polygon
# -------------------------------------------------------------------
print("\n--- Preparing Amazon Basin Polygon and River Data ---")
amazon_basin_geom = None
try:
    basin_fp = "/kaggle/input/make-amazon-polygon/amazon_basin_polygon.geojson"
    amazon_basin_geom = load_amazon_basin_polygon(basin_fp, target_crs="EPSG:4326")
    print("âœ… Amazon basin polygon loaded and simplified.")
except Exception as e:
    print(f"âš ï¸� Failed to load basin polygon: {e}")

# -------------------------------------------------------------------
# Load, filter, and project archaeological site locations
# -------------------------------------------------------------------
sites_gdf_proj = gpd.GeoDataFrame()
try:
    if os.path.exists(arch_sites_csv_path):
        sites_df = pd.read_csv(arch_sites_csv_path).dropna(subset=['latitude', 'longitude'])

        sites_gdf_wgs84 = gpd.GeoDataFrame(
            sites_df,
            geometry=gpd.points_from_xy(sites_df.longitude, sites_df.latitude),
            crs="EPSG:4326"
        ).dropna(subset=['geometry'])

        if amazon_basin_geom:
            sites_gdf_in_basin = sites_gdf_wgs84[sites_gdf_wgs84.geometry.within(amazon_basin_geom)].copy()
        else:
            sites_gdf_in_basin = sites_gdf_wgs84

        if not sites_gdf_in_basin.empty:
            print(f"Projecting {len(sites_gdf_in_basin)} sites to {TARGET_PROJECTED_CRS}...")
            sites_gdf_proj = sites_gdf_in_basin.to_crs(TARGET_PROJECTED_CRS)
        print(f"Prepared {len(sites_gdf_proj)} sites for analysis.")
    else:
        print(f"âš ï¸� Sites CSV not found at {arch_sites_csv_path}")
except Exception as e:
    print(f"â�Œ Error preparing sites data: {e}")

# -------------------------------------------------------------------
# Load, filter, and project LiDAR bounding box geometries
# -------------------------------------------------------------------
lidar_areas_gdf_proj = gpd.GeoDataFrame()
if os.path.exists(grouped_lidar_metadata_csv_path):
    try:
        lidar_df = pd.read_csv(grouped_lidar_metadata_csv_path)
        coord_cols = ['group_id', 'lon_min_wgs84', 'lat_min_wgs84', 'lon_max_wgs84', 'lat_max_wgs84']
        lidar_df.dropna(subset=coord_cols, inplace=True)

        for col in coord_cols[1:]:
            lidar_df[col] = pd.to_numeric(lidar_df[col], errors='coerce')
        lidar_df.dropna(subset=coord_cols, inplace=True)

        if not lidar_df.empty:
            lidar_geoms = [
                box(r['lon_min_wgs84'], r['lat_min_wgs84'], r['lon_max_wgs84'], r['lat_max_wgs84'])
                for _, r in lidar_df.iterrows()
            ]
            lidar_gdf_wgs84 = gpd.GeoDataFrame(lidar_df, geometry=lidar_geoms, crs="EPSG:4326")

            if amazon_basin_geom:
                lidar_gdf_wgs84 = lidar_gdf_wgs84[lidar_gdf_wgs84.geometry.intersects(amazon_basin_geom)].copy()

            if not lidar_gdf_wgs84.empty:
                print(f"Projecting {len(lidar_gdf_wgs84)} LiDAR areas to {TARGET_PROJECTED_CRS}...")
                lidar_areas_gdf_proj = lidar_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
            print(f"Prepared {len(lidar_areas_gdf_proj)} LiDAR area polygons.")
    except Exception as e:
        print(f"â�Œ Error preparing LiDAR data: {e}")

# -------------------------------------------------------------------
# Load and project river feature points (sources, intersections)
# -------------------------------------------------------------------
sources_gdf_proj = gpd.GeoDataFrame()
intersections_gdf_proj = gpd.GeoDataFrame()
if os.path.exists(river_features_path):
    try:
        river_features_gdf = gpd.read_file(river_features_path)
        print(f"\nğŸ“„ Loaded {len(river_features_gdf)} river feature points.")

        sources_gdf_wgs84 = river_features_gdf[river_features_gdf['feature_type'] == 'source']
        intersections_gdf_wgs84 = river_features_gdf[river_features_gdf['feature_type'] == 'intersection']

        if not sources_gdf_wgs84.empty:
            sources_gdf_proj = sources_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
            print(f"Prepared {len(sources_gdf_proj)} river source points.")

        if not intersections_gdf_wgs84.empty:
            intersections_gdf_proj = intersections_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
            print(f"Prepared {len(intersections_gdf_proj)} river intersection points.")
    except Exception as e:
        print(f"â�Œ Error preparing river feature data: {e}")

# -------------------------------------------------------------------
# Project river geometry (lines) and generate analysis grid
# -------------------------------------------------------------------
rivers_gdf_proj = gpd.GeoDataFrame()
rivers_gdf_wgs84 = river_features_gdf[
    river_features_gdf['feature_type'].isin(['source', 'intersection', 'river'])
]

if 'rivers_gdf_wgs84' in locals() and rivers_gdf_wgs84 is not None and not rivers_gdf_wgs84.empty:
    print(f"Projecting {len(rivers_gdf_wgs84)} river geometries to {TARGET_PROJECTED_CRS}...")
    rivers_gdf_proj = rivers_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)

# -------------------------------------------------------------------
# Generate and project analysis grid from basin polygon
# -------------------------------------------------------------------
grid_gdf_proj = gpd.GeoDataFrame()
if amazon_basin_geom and not amazon_basin_geom.is_empty:
    grid_gdf_wgs84 = create_analysis_grid(amazon_basin_geom, CELL_SIZE_KM)
    if grid_gdf_wgs84 is not None and not grid_gdf_wgs84.empty:
        print(f"Projecting {len(grid_gdf_wgs84)} grid cells to {TARGET_PROJECTED_CRS}...")
        grid_gdf_proj = grid_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
        print("ğŸ§± Analysis grid created and projected.")




# ==============================================================================
# SCRIPT: align_rasters_to_reference
# PURPOSE: Align a collection of input rasters to match the shape, resolution,
#          CRS, and transform of a specified reference raster.
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from typing import Dict

# --------------------------------------------------------------------------
# Function: align_rasters_to_reference
# --------------------------------------------------------------------------
def align_rasters_to_reference(
    reference_raster: str,
    input_rasters: Dict[str, str],
    output_dir: str = "aligned_rasters",
    force: bool = False,
    verbose: bool = True,
    dtype: str = "float32",
    nodata: float = -9999.0,
    tile_size: int = 512
) -> Dict[str, str]:
    """
    Aligns multiple rasters to match the spatial grid of a reference raster.

    Parameters:
    ----------
    reference_raster : str
        Path to the raster used as the alignment anchor.
    input_rasters : Dict[str, str]
        Dictionary of rasters to align (e.g. {"slope": "slope.tif"}).
    output_dir : str
        Directory to save aligned rasters (default = "aligned_rasters").
    force : bool
        If True, overwrite existing aligned outputs (default = False).
    verbose : bool
        Print status messages during execution (default = True).
    dtype : str
        Output raster data type (default = "float32").
    nodata : float
        Value to use for nodata in output (default = -9999.0).
    tile_size : int
        Tile block size in pixels (default = 512 for tiling efficiency).

    Returns:
    -------
    Dict[str, str]
        Dictionary with same keys as input_rasters, pointing to output paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Read reference raster properties (CRS, shape, transform)
    # ----------------------------------------------------------------------
    with rasterio.open(reference_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    aligned_paths = {}

    # ----------------------------------------------------------------------
    # Loop through each raster to be aligned
    # ----------------------------------------------------------------------
    for name, in_path in input_rasters.items():
        out_path = os.path.join(output_dir, f"{name}.tif")
        aligned_paths[name] = out_path

        # Skip reprocessing if file exists and not forced
        if os.path.exists(out_path) and not force:
            if verbose:
                print(f"ğŸ”� Using cached: {out_path}")
            continue

        if verbose:
            print(f"ğŸ”„ Aligning {name}...")

        # ------------------------------------------------------------------
        # Open and reproject raster to match reference geometry
        # ------------------------------------------------------------------
        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update({
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_shape[1],
                "height": ref_shape[0],
                "dtype": dtype,
                "nodata": nodata,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": tile_size,
                "blockysize": tile_size
            })

            with rasterio.open(out_path, 'w', **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )

        if verbose:
            print(f"âœ… Saved: {out_path}")

    return aligned_paths

# ==============================================================================
# RUN: Align a predefined set of rasters to a reference mask raster
# ==============================================================================
aligned = align_rasters_to_reference(
    reference_raster="/kaggle/input/amazon-water-data/jrc_occurrence_amazon/amazon_occurrence_mask.tif",
    input_rasters={
        "dem": "/kaggle/input/elevation-above-river-amazon/amazon_merit_dem_downsampled.tif",
        "slope": "/kaggle/input/merit-dem-slope-curv-prof-tpi-tri/merit_pre/merit_slope_3857.tif",
        "tpi": "/kaggle/input/merit-dem-slope-curv-prof-tpi-tri/merit_pre/tpi.tif",
        "tri": "/kaggle/input/merit-dem-slope-curv-prof-tpi-tri/merit_pre/tri.tif",
        "plan_curv": "/kaggle/input/merit-dem-slope-curv-prof-tpi-tri/merit_pre/curv.tif",
        "prof_curv": "/kaggle/input/merit-dem-slope-curv-prof-tpi-tri/merit_pre/prof.tif",
        "biomass": "/kaggle/input/amazon-biomas-map/MapBiomas_Amazon_2018_EPSG4326.tif",
        "elev_river": "/kaggle/input/elevation-above-river-amazon/amazon_elevation_above_river.tif"
    },
    output_dir="aligned_rasters",
    force=False,
    verbose=True
)



# ==============================================================================
# SCRIPT: prospecting_pipeline.py
# PURPOSE: Main class and utilities for building and running the archaeological
#          site prospecting model using geospatial vector and raster layers.
# ==============================================================================

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================

# --- Standard Libraries ---
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter

# --- Geospatial Libraries ---
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# --- RasterIO (for raster handling and projection) ---
import rasterio
from rasterio import features, transform
from rasterio.enums import Resampling as REnum
from rasterio.warp import reproject, calculate_default_transform
from rasterio.crs import CRS

# --- Scikit-learn (for modeling and preprocessing) ---
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.base import TransformerMixin

# --- SciPy utilities ---
from scipy.ndimage import binary_opening, binary_dilation
from scipy.spatial import cKDTree


# ==============================================================================
# CLASS WRAPPER: Sklearn-compatible model wrapper for .predict()
# ==============================================================================
class DecisionFunctionWrapper:
    """
    Wrapper around sklearn decision_function-based models to expose .predict() 
    interface required by permutation_importance.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.decision_function(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def fit(self, X, y=None):
        return self.model.fit(X, y)


# ==============================================================================
# CLASS: ProspectingPipeline
# PURPOSE: Encapsulates geospatial data preparation, model training, and scoring
# ==============================================================================
class ProspectingPipeline:
    def __init__(
        self,
        sites_gdf: gpd.GeoDataFrame,
        rivers_gdf: gpd.GeoDataFrame,
        sources_gdf: gpd.GeoDataFrame,
        intersections_gdf: gpd.GeoDataFrame,
        flood_tif: str,
        dem_tif: str,
        slope_tif: str,
        tpi_tif: str,
        tri_tif: str,
        plan_curv_tif: str,
        prof_curv_tif: str,
        biomass_tif: str,
        elev_river_tif: str,
        river_buffer: float = 50.0,
    ):
        """
        Initialize the pipeline with all input vector and raster paths.
        Raster alignment and CRS projection occur automatically on setup.
        """
        t0 = time.time()

        # --- Vector inputs ---
        self.sites = sites_gdf
        self.rivers = rivers_gdf
        self.sources = sources_gdf
        self.intersections = intersections_gdf

        # --- Raster inputs (as file paths) ---
        self.flood_tif = flood_tif
        self.dem_tif = dem_tif
        self.slope_tif = slope_tif
        self.tpi_tif = tpi_tif
        self.tri_tif = tri_tif
        self.plan_curv_tif = plan_curv_tif
        self.prof_curv_tif = prof_curv_tif
        self.biomass_tif = biomass_tif
        self.elev_river_tif = elev_river_tif
        self.RIVER_BUFFER_METERS = river_buffer

        # --- Internal state (pipeline, model, features) ---
        self._pipeline = None
        self._model = None
        self._scored_grid = None
        self._feats = None
        self.feat_desc = {}

        # --- Use flood raster as reference grid (CRS, transform, shape) ---
        with rasterio.open(self.flood_tif) as src:
            self._ref_crs = src.crs
            self._ref_transform = src.transform
            self._ref_shape = (src.height, src.width)

        # --- Enforce a canonical projected CRS for all vector operations ---
        self.grid_crs = CRS.from_string(TARGET_PROJECTED_CRS)

        # --- Align CRS across all loaded vector layers ---
        self._align_all_crs()

        print(f"Initialize class: {(time.time() - t0):.2f}s")


    def _align_all_crs(self):
        """
        Force all vector layers (sites, rivers, sources, intersections)
        to match the reference projected CRS.
        """
        for attr in ['sites', 'rivers', 'sources', 'intersections']:
            layer = getattr(self, attr, None)
            if layer is not None and hasattr(layer, 'crs') and layer.crs != self.grid_crs:
                setattr(self, attr, layer.to_crs(self.grid_crs))

        for layer in [self.sites, self.rivers, self.sources, self.intersections]:
            if layer is not None and 'geometry' in layer.columns:
                layer.set_geometry('geometry', inplace=True)


    def _pct(self, arr, cond, cell_ids):
        """
        Compute the percentage of valid raster pixels that meet a condition within each cell.
        Efficiently implemented using numpy masking and Counter accumulation.

        Parameters:
        ----------
        arr : ndarray
            Raster array of values.
        cond : function
            Boolean test applied to each pixel (e.g., lambda x: x > 0.1).
        cell_ids : ndarray
            Same shape as arr, with cell index ID for each pixel.

        Returns:
        -------
        pd.Series
            Fraction of pixels per cell satisfying the condition.
        """
        mask = cond(arr) & np.isfinite(arr) & (cell_ids >= 0)
        valid = np.isfinite(arr) & (cell_ids >= 0)

        ids_masked = cell_ids[mask]
        ids_valid = cell_ids[valid]

        if len(ids_valid) == 0:
            return pd.Series(dtype=float)

        count_masked = Counter(ids_masked.ravel())
        count_valid = Counter(ids_valid.ravel())

        result = {
            cid: count_masked.get(cid, 0) / count_valid[cid]
            for cid in count_valid
        }

        return pd.Series(result)
    
    def create_fixed_grid(self, bounds, cell_size: float) -> gpd.GeoDataFrame:
        """
        Create a uniform grid (GeoDataFrame of square polygons) over a bounding box.

        Parameters:
        ----------
        bounds : GeoDataFrame or shapely geometry
            The region to cover; can be a polygon or a GeoDataFrame with .geometry.
        cell_size : float
            Size of each grid cell (in projected units, e.g., meters).

        Returns:
        -------
        GeoDataFrame
            Grid of square polygons clipped to the input boundary.
        """
        from shapely.geometry import box

        # Handle bounds input as polygon or GeoDataFrame
        mask = bounds.geometry.unary_union if hasattr(bounds, 'geometry') else bounds
        minx, miny, maxx, maxy = mask.bounds

        # Create grid using x/y intervals
        xs = np.arange(minx, maxx, cell_size)
        ys = np.arange(miny, maxy, cell_size)
        polys = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]

        # Construct grid GeoDataFrame
        grid = gpd.GeoDataFrame(geometry=polys, crs=bounds.crs if hasattr(bounds, 'crs') else None)

        # Filter to cells within input mask
        grid = grid[grid.geometry.centroid.within(mask)].copy()
        grid.index.name = 'cell_id'

        return grid

    def read_raster_band_safe(self, path, ref_shape, ref_transform=None, resampling=REnum.nearest):
        """
        Read a raster file and resample to match a reference shape and transform.
        Nodata and common fill values (-9999, 9999) are converted to NaN.

        Parameters:
        ----------
        path : str
            Path to the raster file.
        ref_shape : tuple[int, int]
            Target array shape (height, width).
        ref_transform : Affine
            Optional affine transform to match output grid.
        resampling : rasterio.enums.Resampling
            Resampling method (default = nearest).

        Returns:
        -------
        ndarray
            2D float32 array of raster values, aligned to ref_shape.
        """
        with rasterio.open(path) as src:
            # Read first band and mask nodata values
            src_arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                src_arr[src_arr == nodata] = np.nan
            src_arr[src_arr == 9999] = np.nan
            src_arr[src_arr == -9999] = np.nan

            # Return immediately if no resampling is needed
            if src_arr.shape == ref_shape:
                return src_arr

            # Reproject to match reference shape and transform
            dst_arr = np.full(ref_shape, np.nan, dtype=float)
            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform or src.transform,
                dst_crs=src.crs,
                resampling=resampling
            )
            return dst_arr

    def _prepare_grid(self, grid):
        """
        Project the input grid to the pipelineâ€™s target CRS and add cell area.

        Parameters:
        ----------
        grid : GeoDataFrame
            Grid to be projected and processed.

        Returns:
        -------
        GeoDataFrame
            Grid with cell_area column in square meters.
        """
        if grid.crs != self.grid_crs:
            print(f"ğŸŒ� Reprojecting grid to {self.grid_crs}")
            grid = grid.to_crs(self.grid_crs)

        # Ensure all reference layers match the pipeline CRS
        self._align_all_crs()

        grid = grid.copy()
        grid['cell_area'] = grid.geometry.area
        self.feat_desc['cell_area'] = "area of each grid cell (mÂ²)"

        return grid

    def _add_hydrology_features(self, grid):
        """
        Compute hydrology-related features for each grid cell:
        - Area overlapping buffered river polygons (e.g., floodplain area)
        - Distance from cell centroid to nearest:
            - river
            - river source
            - river intersection/confluence

        Updates `grid` in-place with new columns:
            - water_area_sq_m
            - distance_to_river_m
            - distance_to_source_m
            - distance_to_confluence_m

        Also updates `self.feat_desc` with human-readable descriptions.
        """
        # --- Floodplain buffer intersection ---
        if not self.rivers.empty:
            buf = self.rivers.copy()
            buf.geometry = buf.geometry.buffer(self.RIVER_BUFFER_METERS)
            inter = gpd.overlay(grid, buf, how='intersection')
            inter['area'] = inter.geometry.area

            # Sum buffered water area per cell
            wa = inter.groupby(level=0)['area'].sum()
            grid['water_area_sq_m'] = wa.reindex(grid.index, fill_value=0)
        else:
            grid['water_area_sq_m'] = 0

        self.feat_desc['water_area_sq_m'] = "floodplain buffer area intersecting the cell (mÂ²)"

        # --- Distance to river features ---
        cent = gpd.GeoDataFrame(geometry=grid.geometry.centroid, crs=grid.crs)

        for label, src, col in [
            ("river", self.rivers, "distance_to_river_m"),
            ("source", self.sources, "distance_to_source_m"),
            ("confluence", self.intersections, "distance_to_confluence_m"),
        ]:
            if not src.empty:
                joined = gpd.sjoin_nearest(cent, src, how='left', distance_col=col)
                dist = joined.groupby(joined.index)[col].first()
                grid[col] = dist.reindex(grid.index, fill_value=np.nan)
            else:
                grid[col] = np.nan
            self.feat_desc[col] = f"distance from centroid to nearest {label}"

    def _add_site_features(self, grid):
        """
        Add binary and count-based features for archaeological sites.

        For each grid cell:
        - Counts how many known sites fall within it
        - Flags whether it contains at least one site

        Updates `grid` in-place with:
            - site_count
            - has_site

        Also updates `self.feat_desc` with column descriptions.
        """
        if not self.sites.empty:
            joined = gpd.sjoin(grid, self.sites, how='left', predicate='intersects')
            counts = joined.groupby(level=0)['index_right'].count()
            grid['site_count'] = counts.reindex(grid.index, fill_value=0)
        else:
            grid['site_count'] = 0

        # Binary presence flag
        grid['has_site'] = grid['site_count'] > 0

        self.feat_desc['site_count'] = "number of sites in cell"
        self.feat_desc['has_site'] = "binary flag: 1 if site exists in cell"


    def _add_raster_overlay_features(self, grid):
        """
        Adds terrain-related features to each grid cell based on slope and flood mask rasters.

        Features computed:
        - `flood_pct`:    % of cell area classified as flooded
        - `flat_pct`:     % of cell with slope < 5 degrees
        - `relief_pct`:   % of cell with slope > 15 degrees

        All features are computed using raster masks and pixel-level aggregation.

        Updates `grid` in-place and records descriptions in `self.feat_desc`.
        """
        # Load raster metadata
        with rasterio.open(self.flood_tif) as src:
            ref_transform, ref_crs = src.transform, src.crs
            ref_shape = (src.height, src.width)

        # Rasterize grid cell IDs to match raster dimensions
        grid_proj = grid.to_crs(ref_crs)
        shapes = ((geom, cid) for cid, geom in zip(grid_proj.index, grid_proj.geometry))
        cell_ids = features.rasterize(
            shapes, out_shape=ref_shape, transform=ref_transform, fill=-1, dtype='int32'
        )

        # Load raster bands
        arr_flood = self.read_raster_band_safe(self.flood_tif, ref_shape)
        arr_slope_raw = self.read_raster_band_safe(self.slope_tif, ref_shape)

        # Prepare clean slope array
        valid_mask = np.isfinite(arr_slope_raw) & (cell_ids >= 0)
        arr_slope = np.full_like(arr_slope_raw, np.nan)
        arr_slope[valid_mask] = arr_slope_raw[valid_mask]

        # Count valid pixels per cell
        cell_id_flat = cell_ids[valid_mask].ravel()
        total_counts = Counter(cell_id_flat)

        # Define masks
        flood_mask = valid_mask & (arr_flood > 0)
        flat_mask = np.where(np.isfinite(arr_slope), arr_slope < 5, False)
        relief_mask = np.where(np.isfinite(arr_slope), arr_slope > 15, False)

        # Count masked pixels by cell
        flood_counts = Counter(cell_ids[flood_mask].ravel())
        flat_counts = Counter(cell_ids[flat_mask].ravel())
        relief_counts = Counter(cell_ids[relief_mask].ravel())

        # Convert to percent-of-cell series
        def ratio_series(sub):
            return pd.Series({k: sub.get(k, 0) / total_counts[k] for k in total_counts})

        grid['flood_pct'] = ratio_series(flood_counts).reindex(grid.index, fill_value=0).astype("float32")
        grid['flat_pct'] = ratio_series(flat_counts).reindex(grid.index, fill_value=0).astype("float32")
        grid['relief_pct'] = ratio_series(relief_counts).reindex(grid.index, fill_value=0).astype("float32")

        self.feat_desc.update({
            'flood_pct':  "fraction of cell area classified as flood",
            'flat_pct':   "fraction with slope < 5Â°",
            'relief_pct': "fraction with slope > 15Â°"
        })

    def _add_elevation_band_features(self, grid):
        """
        Adds elevation-based features from the elevation-above-river raster.
        Each cell is classified into bands based on how much of its area falls into:
        - < 1m
        - 1â€“5m
        - 5â€“10m
        - 10â€“20m
        - > 20m

        Adds 5 columns to `grid` and updates self.feat_desc with labels.
        """
        # Load elevation-above-river raster
        arr_above = self.read_raster_band_safe(
            self.elev_river_tif,
            self._ref_shape,
            self._ref_transform,
            resampling=Resampling.nearest
        )

        # Rasterize cell ID map
        shapes = ((geom, cid) for cid, geom in zip(grid.index, grid.geometry))
        cell_ids = features.rasterize(
            shapes, out_shape=self._ref_shape, transform=self._ref_transform, fill=-1, dtype='int32'
        )

        # Flatten valid raster data
        valid_mask = np.isfinite(arr_above) & (cell_ids >= 0)
        cid_flat = cell_ids[valid_mask].ravel()
        val_flat = arr_above[valid_mask].ravel()
        count_total = Counter(cid_flat)

        # Define elevation band masks
        bands = {
            "pct_elev_below_1m":     val_flat < 1,
            "pct_elev_1_5m":         (val_flat >= 1) & (val_flat < 5),
            "pct_elev_5_10m":        (val_flat >= 5) & (val_flat < 10),
            "pct_elev_10_20m":       (val_flat >= 10) & (val_flat < 20),
            "pct_elev_above_20m":    val_flat >= 20
        }

        # Compute ratios for each band
        for name, mask in bands.items():
            cid_band = cid_flat[mask]
            count_band = Counter(cid_band)
            pct = {k: count_band.get(k, 0) / count_total[k] for k in count_total}
            grid[name] = pd.Series(pct).reindex(grid.index, fill_value=0).astype("float32")

        self.feat_desc.update({
            "pct_elev_below_1m":    "fraction of cell with elevation < 1â€¯m above river",
            "pct_elev_1_5m":        "fraction of cell with elevation 1â€“5â€¯m above river",
            "pct_elev_5_10m":       "fraction of cell with elevation 5â€“10â€¯m above river",
            "pct_elev_10_20m":      "fraction of cell with elevation 10â€“20â€¯m above river",
            "pct_elev_above_20m":   "fraction of cell with elevation > 20â€¯m above river"
        })


    def _add_elevation_stats_features(self, grid):
        """
        Computes summary statistics of elevation-above-river within each grid cell.

        Uses a raster aligned with the grid to calculate:
        - Mean, median, min, max, and standard deviation of elevation values

        Adds the following columns to `grid`:
        - elev_river_mean
        - elev_river_median
        - elev_river_min
        - elev_river_max
        - elev_river_std

        Descriptions are added to self.feat_desc.
        """
        # Project grid to match raster CRS
        grid_proj = grid.to_crs(self._ref_crs)
        shapes = ((geom, cid) for cid, geom in zip(grid_proj.index, grid_proj.geometry))

        # Read raster
        arr_above = self.read_raster_band_safe(
            self.elev_river_tif,
            self._ref_shape,
            self._ref_transform,
            resampling=Resampling.nearest
        )

        # Rasterize cell IDs
        cell_ids = features.rasterize(
            shapes,
            out_shape=self._ref_shape,
            transform=self._ref_transform,
            fill=-1,
            dtype='int32'
        )

        # Extract valid values and their cell IDs
        valid_mask = np.isfinite(arr_above) & (cell_ids >= 0)
        ids = cell_ids[valid_mask].ravel()
        vals = arr_above[valid_mask].ravel()

        # Compute grouped stats
        df = pd.DataFrame({'id': ids, 'val': vals})
        grp = df.groupby('id')['val']
        stats = {
            'elev_river_mean': grp.mean(),
            'elev_river_median': grp.median(),
            'elev_river_min': grp.min(),
            'elev_river_max': grp.max(),
            'elev_river_std': grp.std()
        }

        # Assign stats back to grid
        for name, series in stats.items():
            grid[name] = grid.index.map(lambda i: float(series.get(i, np.nan)))

        self.feat_desc.update({
            'elev_river_mean':   "mean elevation above river within cell (m)",
            'elev_river_median': "median elevation above river within cell (m)",
            'elev_river_min':    "minimum elevation above river in cell (m)",
            'elev_river_max':    "maximum elevation above river in cell (m)",
            'elev_river_std':    "standard deviation of elevation above river (m)"
        })

    def _add_zonal_stats(self, grid):
        """
        Computes zonal statistics (mean, std, range) for multiple terrain rasters:
        - slope
        - TPI (topographic position index)
        - TRI (terrain ruggedness index)
        - plan curvature
        - profile curvature

        Each stat is calculated over pixels within each grid cell.
        Columns added include:
        - slope_mean, slope_std, slope_range
        - tpi_mean, tpi_std, tpi_range
        - ...

        Descriptions are stored in self.feat_desc.
        """
        raster_layers = [
            ('slope', self.slope_tif),
            ('tpi', self.tpi_tif),
            ('tri', self.tri_tif),
            ('plan_curv', self.plan_curv_tif),
            ('prof_curv', self.prof_curv_tif)
        ]

        # Rasterize cell IDs
        shapes = ((geom, cid) for cid, geom in zip(grid.index, grid.geometry))
        cell_ids = features.rasterize(
            shapes,
            out_shape=self._ref_shape,
            transform=self._ref_transform,
            fill=-1,
            dtype='int32'
        )

        for name, path in raster_layers:
            # Read raster aligned with reference
            arr = self.read_raster_band_safe(
                path,
                self._ref_shape,
                self._ref_transform,
                resampling=Resampling.nearest
            )

            # Prepare DataFrame of valid values
            ids, vals = cell_ids.ravel(), arr.ravel()
            df = pd.DataFrame({'id': ids, 'val': vals})
            df = df[df['id'] >= 0].dropna(subset=['val'])
            grp = df.groupby('id')['val']

            # Compute zonal stats
            stats = {
                f"{name}_mean": grp.mean(),
                f"{name}_std":  grp.std(),
                f"{name}_range": grp.max() - grp.min()
            }

            # Assign results back to grid and record descriptions
            for col, series in stats.items():
                grid[col] = grid.index.map(lambda i: float(series.get(i, np.nan)))
                self.feat_desc[col] = f"{col.split('_')[-1]} of {name} in cell"

    def _add_derived_features(self, grid):
        """
        Computes additional derived terrain and hydrology features from previously added columns.

        Includes:
        - Elevation variability measures (range, IQR, relative std)
        - Stream proximity and hydrology ratios
        - Terrain band dominance and terrace index
        - Binary flags for flood-prone and probable terrace cells
        - Interaction metrics (e.g. flood Ã— slope)

        Updates `grid` in-place and appends descriptions to `self.feat_desc`.
        """
        try:
            # --- Elevation variability ---
            grid["elev_river_range"] = grid["elev_river_max"] - grid["elev_river_min"]
            self.feat_desc["elev_river_range"] = "elevation range above river in cell (max - min, m)"

            elev_q75 = grid["elev_river_median"] + 0.674 * grid["elev_river_std"]
            elev_q25 = grid["elev_river_median"] - 0.674 * grid["elev_river_std"]
            grid["elev_river_iqr"] = elev_q75 - elev_q25
            self.feat_desc["elev_river_iqr"] = "interquartile range of elevation above river (Q3â€“Q1, approx)"

            grid["elev_river_rel_std"] = grid["elev_river_std"] / (grid["elev_river_mean"] + 1e-6)
            self.feat_desc["elev_river_rel_std"] = "relative std dev of elevation above river (std / mean)"

            # --- Stream hierarchy and proximity ---
            grid["dist_to_midstream_m"] = grid[["distance_to_confluence_m", "distance_to_source_m"]].min(axis=1)
            self.feat_desc["dist_to_midstream_m"] = "minimum of source/confluence distance (stream center proxy, m)"

            grid["stream_hierarchy_ratio"] = grid["distance_to_confluence_m"] / (grid["distance_to_source_m"] + 1)
            self.feat_desc["stream_hierarchy_ratio"] = "confluence/source distance ratio (stream hierarchy index)"

            grid["is_headwater"] = (grid["distance_to_source_m"] < grid["distance_to_confluence_m"]).astype(int)
            self.feat_desc["is_headwater"] = "binary flag: 1 if closer to source than confluence"

            # --- Normalized slope and elevation by stream distance ---
            grid["elev_div_midstream"] = grid["elev_river_mean"] / (grid["dist_to_midstream_m"] + 1e-6)
            self.feat_desc["elev_div_midstream"] = "mean elev. above river divided by distance to midstream (m/m)"

            grid["slope_div_midstream"] = grid["slope_mean"] / (grid["dist_to_midstream_m"] + 1e-6)
            self.feat_desc["slope_div_midstream"] = "mean slope divided by distance to midstream (Â°/m)"

            # --- Elevation band analysis ---
            elev_band_cols = [
                "pct_elev_below_1m", "pct_elev_1_5m", "pct_elev_5_10m",
                "pct_elev_10_20m", "pct_elev_above_20m"
            ]
            band_names = ["<1m", "1â€“5m", "5â€“10m", "10â€“20m", ">20m"]

            band_array = grid[elev_band_cols].values
            grid["dominant_elev_band"] = np.argmax(band_array, axis=1)
            self.feat_desc["dominant_elev_band"] = "index (0â€“4) of dominant elevation-above-river band"
            grid["dominant_elev_band_label"] = grid["dominant_elev_band"].map(dict(enumerate(band_names)))

            # --- Composite terrain features ---
            grid["terrace_index"] = (
                grid["pct_elev_5_10m"] + grid["pct_elev_10_20m"] - grid["pct_elev_below_1m"]
            )
            self.feat_desc["terrace_index"] = (
                "composite terrace signal: (5â€“20m band) minus flood-prone area (<1m)"
            )

            grid["elev_consistency"] = (grid["elev_river_mean"] - grid["elev_river_median"]).abs()
            self.feat_desc["elev_consistency"] = "abs difference between mean and median elevation above river (m)"

            # --- Flood and terrace flags ---
            grid["is_probable_flood"] = (grid["flood_pct"] > 0.3).astype(int)
            self.feat_desc["is_probable_flood"] = "1 if >30% of cell is flooded"

            grid["is_probable_terrace"] = (
                (grid["elev_river_mean"] > 5) & (grid["flood_pct"] < 0.05)
            ).astype(int)
            self.feat_desc["is_probable_terrace"] = "1 if >5m above river and <5% flood area"

            # --- Interaction features ---
            grid["flood_slope_contrast"] = grid["flood_pct"] * grid["slope_mean"]
            self.feat_desc["flood_slope_contrast"] = (
                "interaction of flood percent and slope â€” detects abrupt terrain transitions near flood zones"
            )

        except Exception as e:
            print(f"âš ï¸� Skipping derived features due to error: {e}")


    def _add_raster_overlay_features(self, grid):
        """
        Adds floodplain and slope classification metrics to each grid cell using raster overlays.

        This method computes the fraction of each cell that:
        - overlaps with floodplain pixels (`flood_pct`)
        - has flat terrain (slope < 5Â°) â†’ `flat_pct`
        - has steep terrain (slope > 15Â°) â†’ `relief_pct`

        Each metric is computed using raster alignment with the floodplain and slope rasters.

        Updates `grid` in-place with 3 new columns and adds descriptions to `self.feat_desc`.
        """
        # --- Load flood raster metadata (shape, transform, CRS) ---
        with rasterio.open(self.flood_tif) as src:
            ref_transform, ref_crs = src.transform, src.crs
            ref_shape = (src.height, src.width)

        # --- Project grid to raster CRS and rasterize cell IDs ---
        grid_proj = grid.to_crs(ref_crs)
        shapes = ((geom, cid) for cid, geom in zip(grid_proj.index, grid_proj.geometry))
        cell_ids = features.rasterize(
            shapes,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=-1,
            dtype='int32'
        )

        # --- Load flood and slope rasters aligned with reference shape ---
        arr_flood = self.read_raster_band_safe(self.flood_tif, ref_shape)
        arr_slope_raw = self.read_raster_band_safe(self.slope_tif, ref_shape)

        # --- Clean slope array by masking invalid entries ---
        valid_mask = np.isfinite(arr_slope_raw) & (cell_ids >= 0)
        arr_slope = np.full_like(arr_slope_raw, np.nan)
        arr_slope[valid_mask] = arr_slope_raw[valid_mask]

        # --- Prepare flattened views for frequency counting ---
        cell_id_flat = cell_ids[valid_mask].ravel()
        total_counts = Counter(cell_id_flat)

        # --- Mask definitions ---
        flood_mask = valid_mask & (arr_flood > 0)

        # Use robust NumPy comparisons to define slope categories
        arr_slope_clean = np.where(np.isfinite(arr_slope), arr_slope, np.nan)
        flat_mask = np.less(arr_slope_clean, 5, where=np.isfinite(arr_slope_clean), out=np.zeros_like(arr_slope_clean, dtype=bool))
        relief_mask = np.greater(arr_slope_clean, 15, where=np.isfinite(arr_slope_clean), out=np.zeros_like(arr_slope_clean, dtype=bool))

        # --- Count pixels meeting each condition per cell ---
        flood_counts = Counter(cell_ids[flood_mask].ravel())
        flat_counts = Counter(cell_ids[flat_mask & valid_mask].ravel())
        relief_counts = Counter(cell_ids[relief_mask & valid_mask].ravel())

        # --- Convert counts to per-cell fractions ---
        def ratio_series(sub):
            return pd.Series({k: sub.get(k, 0) / total_counts[k] for k in total_counts})

        grid['flood_pct']  = ratio_series(flood_counts).reindex(grid.index, fill_value=0).astype("float32")
        grid['flat_pct']   = ratio_series(flat_counts).reindex(grid.index, fill_value=0).astype("float32")
        grid['relief_pct'] = ratio_series(relief_counts).reindex(grid.index, fill_value=0).astype("float32")

        # --- Document features ---
        self.feat_desc.update({
            'flood_pct':  "fraction of cell area classified as flood",
            'flat_pct':   "fraction with slope < 5Â°",
            'relief_pct': "fraction with slope > 15Â°"
        })

    def _add_oxbow_features(self, grid):
        """
        Identifies low-slope, low-elevation depressions (oxbow-like features) in the terrain,
        then computes distance from each grid cell centroid to the nearest such depression.

        This is useful for detecting paleo-channels or relict meanders often associated with
        human settlement or floodplain dynamics.

        Adds a new column to `grid`:
            - 'dist_to_oxbow': distance (in meters) to nearest oxbow depression
        Updates `self.feat_desc` with the corresponding feature description.
        """
        try:
            from scipy.ndimage import binary_opening, binary_dilation
            from scipy.spatial import cKDTree
            from rasterio import transform

            # --- Load smoothed DEM and raw slope raster ---
            dem_arr = self.read_raster_band_safe(self.dem_tif, self._ref_shape, resampling=Resampling.bilinear)
            arr_slope_raw = self.read_raster_band_safe(self.slope_tif, self._ref_shape)

            # --- Clean raster values ---
            arr_slope = np.where(np.isfinite(arr_slope_raw), arr_slope_raw, np.nan)
            dem_arr = np.where(np.isfinite(dem_arr), dem_arr, np.nan)

            # --- Create boolean masks for low slope and low elevation ---
            finite_slope = np.isfinite(arr_slope)
            with np.errstate(invalid='ignore'):
                low_slope = np.less(arr_slope, 3, where=finite_slope, out=np.zeros_like(arr_slope, dtype=bool))

            finite_dem = np.isfinite(dem_arr)
            finite_values = dem_arr[finite_dem].ravel()
            if finite_values.size == 0:
                raise ValueError("DEM contains no valid values for oxbow detection")

            threshold = np.nanpercentile(finite_values, 25)
            with np.errstate(invalid='ignore'):
                low_elev = np.less(dem_arr, threshold, where=finite_dem, out=np.zeros_like(dem_arr, dtype=bool))

            # --- Identify combined oxbow mask and clean with morphology ---
            oxbow_mask = low_slope & low_elev & finite_dem
            oxbow_mask = binary_opening(oxbow_mask, structure=np.ones((3, 3)))
            oxbow_mask = binary_dilation(oxbow_mask, iterations=1)

            # --- Convert raster coordinates to spatial coordinates ---
            rows, cols = np.meshgrid(
                np.arange(self._ref_shape[0]), np.arange(self._ref_shape[1]), indexing='ij'
            )
            xs, ys = transform.xy(self._ref_transform, rows, cols)
            xs = np.asarray(xs).reshape(self._ref_shape)
            ys = np.asarray(ys).reshape(self._ref_shape)

            # --- Stack XY coords of all oxbow pixels ---
            oxbow_coords = np.column_stack((xs[oxbow_mask], ys[oxbow_mask]))
            if len(oxbow_coords) == 0:
                raise ValueError("No oxbow-like depressions found")

            # --- Build spatial search index from oxbow pixels ---
            tree = cKDTree(oxbow_coords)

            # --- Project grid to UTM and prepare centroids ---
            grid_proj = grid.to_crs(grid.estimate_utm_crs())
            dist_to_oxbow = pd.Series(np.nan, index=grid.index)

            # --- Validate geometries and bounds ---
            geoms = grid_proj.geometry
            basic_mask = geoms.is_valid & ~geoms.is_empty
            try:
                bounds = geoms.bounds
                coord_mask = bounds['minx'].notna() & bounds['miny'].notna()
                valid_mask = basic_mask & coord_mask
            except Exception as e:
                print(f"âš ï¸� Skipping bounds-based validation due to error: {e}")
                valid_mask = basic_mask

            print("ğŸ”� Geometry filter:")
            print(f"  Total: {len(geoms)}")
            print(f"  Valid + non-empty: {basic_mask.sum()}")
            print(f"  Valid coordinates: {valid_mask.sum()}")

            valid_geoms = grid_proj.loc[valid_mask, 'geometry']
            centroids = valid_geoms.centroid
            cell_coords = np.column_stack((centroids.x.values, centroids.y.values))

            # --- Query tree for distance to nearest oxbow pixel ---
            dists, _ = tree.query(cell_coords)
            dist_to_oxbow.loc[valid_mask] = dists

            # --- Save result back to grid ---
            grid['dist_to_oxbow'] = dist_to_oxbow.values
            self.feat_desc['dist_to_oxbow'] = "distance in meters to nearest oxbow-like depression"

        except Exception as e:
            print(f"âš ï¸� Skipping oxbow features due to error: {e}")

    
    def _add_neighbor_delta_features(self, grid, feat_cols):
        """
        Computes the difference between each cell's feature values and the mean of its neighboring cells.

        Adds new columns to `grid` named 'delta_<feature>' for each feature in `feat_cols`.

        Parameters:
            grid (GeoDataFrame): The input grid of cells.
            feat_cols (list): List of feature column names to compute neighbor differences for.

        Returns:
            GeoDataFrame: Updated grid with delta features added.
        """
        try:
            # Initialize self._feats if not already done
            if not hasattr(self, '_feats') or self._feats is None:
                self._feats = feat_cols.copy()

            # Prepare left-right GeoDataFrames for spatial join
            left = grid.reset_index()[['cell_id', 'geometry']].rename(columns={'cell_id': 'cell'})
            right = left.rename(columns={'geometry': 'nbr_geom', 'cell': 'nbr'})

            # Perform spatial join to find neighboring cell pairs (cells that touch)
            nbrs = gpd.sjoin(
                left.set_geometry('geometry'),
                right.set_geometry('nbr_geom'),
                predicate='touches'
            )

            # Remove self-joins (i.e., same cell matched to itself)
            nbrs = nbrs[nbrs['cell'] != nbrs['nbr']][['cell', 'nbr']]

            delta_data = {}

            for col in feat_cols:
                vals = grid[col]

                # Compute neighbor mean for each cell
                nei = nbrs.groupby('cell')['nbr'].apply(lambda ids: vals.loc[ids].mean())

                # Map neighbor means back to full index
                nei_series = grid.index.map(lambda i: float(nei.get(i, vals.loc[i])))

                # Compute delta from neighbor mean
                delta_col = f"delta_{col}"
                delta_data[delta_col] = grid[col] - nei_series
                self.feat_desc[delta_col] = f"difference from neighbors' mean of {col}"
                self._feats.append(delta_col)

            # Merge delta columns into the grid
            all_new_data = pd.DataFrame(delta_data, index=grid.index)
            grid = pd.concat([grid, all_new_data], axis=1)

            # Update internal feature list
            delta_feats = list(delta_data.keys())
            self._feats = feat_cols + delta_feats

            return grid

        except Exception as e:
            print(f"âš ï¸� Skipping neighbor and delta features due to error: {e}")


    def _select_feature_columns(self, grid):
        """
        Selects a list of candidate numeric columns from the grid to be used as model features.

        Filters out:
        - Geometry and categorical columns
        - Explicitly excluded columns (e.g., prospect_score, site_count, etc.)
        - Columns starting with 'delta_' (these will be added separately)

        Parameters:
            grid (GeoDataFrame): The input grid of cell features.

        Returns:
            list: Selected numeric column names for modeling.
        """
        exclude = [
            'geometry', 'cell_area', 'water_area_sq_m', 'site_count',
            'has_site', 'forest_pct', 'prospect_score', 'prev_score',
            'dominant_elev_band_label', 'parent_id', 'non_forest_pct'
        ]

        # Keep only numeric, non-excluded columns that are not already delta features
        feat_cols = [
            c for c in grid.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(grid[c])
            and not c.startswith('delta_')
        ]
        return feat_cols


    def generate_features(self, grid: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
        """
        Orchestrates the full feature generation pipeline on a spatial analysis grid.

        This method applies all core feature engineering steps in sequence:
            - CRS alignment and grid preparation
            - Hydrology, site, raster overlay, and elevation features
            - Zonal statistics and derived metrics
            - Oxbow proximity analysis
            - Spatial neighbor deltas

        Parameters:
            grid (GeoDataFrame): The analysis grid on which features will be computed.

        Returns:
            tuple:
                - Updated GeoDataFrame with added feature columns.
                - List of final feature column names (self._feats).
        """
        t_start = time.time()
        self.feat_desc = {}  # Clear previous feature descriptions
        self._feats = []     # Clear feature column list

        print("â–¶ï¸� Starting feature generation")

        # Step 1: Reproject grid and add area column
        t = time.time()
        grid = self._prepare_grid(grid)
        print(f"âœ… _prepare_grid in {time.time() - t:.2f}s")

        # Step 2: Add river distance, confluence distance, etc.
        t = time.time()
        self._add_hydrology_features(grid)
        print(f"âœ… _add_hydrology_features in {time.time() - t:.2f}s")

        # Step 3: Add archaeological site counts and binary presence
        t = time.time()
        self._add_site_features(grid)
        print(f"âœ… _add_site_features in {time.time() - t:.2f}s")

        # Step 4: Flood and slope masks
        t = time.time()
        self._add_raster_overlay_features(grid)
        print(f"âœ… _add_raster_overlay_features in {time.time() - t:.2f}s")

        # Step 5: Percent of elevation above river within defined bands
        t = time.time()
        self._add_elevation_band_features(grid)
        print(f"âœ… _add_elevation_band_features in {time.time() - t:.2f}s")

        # Step 6: Mean, std, range, etc. of elevation above river
        t = time.time()
        self._add_elevation_stats_features(grid)
        print(f"âœ… _add_elevation_stats_features in {time.time() - t:.2f}s")

        # Step 7: Mean, std, range of slope, TPI, TRI, curvature
        t = time.time()
        self._add_zonal_stats(grid)
        print(f"âœ… _add_zonal_stats in {time.time() - t:.2f}s")

        # Step 8: Derived and composite terrain features
        t = time.time()
        self._add_derived_features(grid)
        print(f"âœ… _add_derived_features in {time.time() - t:.2f}s")

        # Step 9: Add oxbow proximity
        t = time.time()
        self._add_oxbow_features(grid)
        print(f"âœ… _add_oxbow_features in {time.time() - t:.2f}s")

        # Step 10: Compute delta from neighbor means
        t = time.time()
        feat_cols = self._select_feature_columns(grid)
        grid = self._add_neighbor_delta_features(grid, feat_cols)
        print(f"âœ… _add_neighbor_delta_features in {time.time() - t:.2f}s")

        print(f"ğŸ�� Feature generation complete in {time.time() - t_start:.2f}s")
        return grid

    def train_model(
        self,
        grid: gpd.GeoDataFrame,
        method: str = 'svm',
        **svm_params
    ) -> tuple[object, gpd.GeoDataFrame]:
        """
        Trains a one-class SVM or Isolation Forest model on positive samples (cells with known archaeological sites),
        transforms the entire feature space, and assigns a normalized "prospect_score" to all grid cells.

        Parameters:
            grid (GeoDataFrame): The input feature grid.
            method (str): The model type ('svm' for OneClassSVM, anything else for IsolationForest).
            **svm_params: Additional keyword arguments for model instantiation.

        Returns:
            model (sklearn model): Trained SVM or IsolationForest model.
            grid (GeoDataFrame): Updated grid with 'prospect_score' column.
        """

        # Custom wrapper to retain feature names
        class FunctionTransformerWithNames(FunctionTransformer, TransformerMixin):
            def get_feature_names_out(self, input_features=None):
                return np.array(input_features)

        t0 = time.time()

        # ---------------------------------------------
        # 1. Sample weighting: use 'weight' column if available
        # ---------------------------------------------
        grid['site_count'] = grid.get('site_count', 0)
        if 'weight' in self.sites.columns:
            join = gpd.sjoin(grid, self.sites[['geometry', 'weight']], how='left', predicate='intersects')
            wts = join.groupby(join.index)['weight'].sum()
            grid['sample_weight'] = wts.reindex(grid.index, fill_value=0).clip(lower=1)
        else:
            grid['sample_weight'] = grid['site_count'].clip(lower=1)

        pos = grid['site_count'] > 0  # mask for positive training samples

        # ---------------------------------------------
        # 2. Feature preparation and outlier clipping
        # ---------------------------------------------
        X_df = grid[self._feats].replace([np.inf, -np.inf], np.nan)
        clip_val = np.nanpercentile(X_df.values, 99)
        X_df = X_df.clip(-clip_val, clip_val)

        # ---------------------------------------------
        # 3. Preprocessing pipeline (impute â†’ log transform â†’ scale)
        # ---------------------------------------------
        numeric_pipeline = Pipeline([
            ('imp1', SimpleImputer(strategy='median')),
            ('log', FunctionTransformerWithNames(
                func=lambda X: np.sign(X) * np.log1p(np.abs(X)),
                validate=False
            )),
            ('imp2', SimpleImputer(strategy='median')),
            ('sca', StandardScaler())
        ])

        col_transform = ColumnTransformer(
            transformers=[('num', numeric_pipeline, self._feats)],
            remainder='drop',
            verbose_feature_names_out=False
        )

        pipeline = Pipeline([
            ('pre', col_transform)
        ])
        self._pipeline = pipeline  # store for reuse

        # ---------------------------------------------
        # 4. Fit and transform entire dataset
        # ---------------------------------------------
        try:
            X = pipeline.fit_transform(X_df)
        except Exception as e:
            raise RuntimeError(f"Pipeline transformation failed: {e}")

        self._feats_final = pipeline.named_steps['pre'].get_feature_names_out()

        if not np.all(np.isfinite(X)):
            bad_rows = (~np.all(np.isfinite(X), axis=1)).sum()
            print(f"âš ï¸� train_model: {bad_rows} rows in full dataset have invalid features after pipeline")

        # ---------------------------------------------
        # 5. Fit model on positive cells only
        # ---------------------------------------------
        Xp = X[pos.values]
        w = grid.loc[pos, 'sample_weight'].values

        # Filter out invalid training rows
        mask = np.all(np.isfinite(Xp), axis=1)
        if not mask.all():
            dropped = (~mask).sum()
            print(f"âš ï¸� train_model: dropping {dropped} invalid positive samples")
            Xp, w = Xp[mask], w[mask]

        t1 = time.time()
        if method == 'svm':
            model = OneClassSVM(**svm_params)
            model.fit(Xp, sample_weight=w)
        else:
            model = IsolationForest(**svm_params)
            model.fit(Xp, sample_weight=w)

        raw = model.decision_function(X)
        raw = np.asarray(raw)

        # Handle NaNs from decision function
        if np.isnan(raw).any():
            print(f"âš ï¸� train_model: decision_function returned {np.isnan(raw).sum()} NaNs; filling with fallback")
            raw[np.isnan(raw)] = -9999

        # ---------------------------------------------
        # 6. Normalize decision scores â†’ 1â€“100 range
        # ---------------------------------------------
        rmin, rmax = np.nanmin(raw), np.nanmax(raw)
        if rmax == rmin:
            print("âš ï¸� train_model: rmax == rmin; assigning uniform prospect_score = 50")
            grid['prospect_score'] = 50.0
        else:
            norm_score = 1 + 99 * (raw - rmin) / (rmax - rmin)
            norm_score[~np.isfinite(norm_score)] = np.nan
            grid['prospect_score'] = norm_score

        print(f"âœ… train_model: model built in {time.time()-t1:.2f}s (total {time.time()-t0:.2f}s)")

        # ---------------------------------------------
        # 7. Store outputs and return
        # ---------------------------------------------
        self._model = model
        self._scored_grid = grid
        self._X_train_df = X_df.copy()
        self._raw_decision = raw.copy()
        self._X_train = Xp.copy()
        self._y_train = raw[pos.values]

        return model, grid


    def compute_importance(
            self,
            model,
            scored_grid: gpd.GeoDataFrame,
            n_repeats: int = 5,
            random_state: int = 0
        ) -> pd.DataFrame:
        """
        Computes permutation-based feature importance using a custom scoring function
        that penalizes disruption of prediction variance (e.g., for one-class models).

        Parameters:
            model: Trained model (SVM or Isolation Forest)
            scored_grid (GeoDataFrame): Grid with prospect scores assigned
            n_repeats (int): Number of permutations per feature
            random_state (int): Seed for reproducibility

        Returns:
            DataFrame: Ranked feature importance summary including average values
                       for top-scoring cells and descriptions.
        """
        import time
        import numpy as np
        import pandas as pd
        from sklearn.inspection import permutation_importance

        # Ensure training data exists from prior call to train_model()
        if not hasattr(self, "_X_train") or not hasattr(self, "_y_train"):
            raise RuntimeError("ğŸš¨ Missing training data. Ensure train_model() has been called.")

        X = self._X_train
        y = self._y_train

        wrapped_model = DecisionFunctionWrapper(model)

        # Counter to track failed importance attempts
        self._importance_nan_count = 0

        # --- Custom scoring: lower variance = better
        def variance_disruption_score(estimator, X, y_true):
            pred = estimator.predict(X)
            pred = np.asarray(pred)
            pred = pred[np.isfinite(pred)]

            if len(pred) == 0:
                self._importance_nan_count += 1
                return 0

            std = np.std(pred)
            if not np.isfinite(std):
                self._importance_nan_count += 1
                return 0

            return -std  # goal is to penalize disruption

        # --- Run permutation importance ---
        t0 = time.time()
        with np.errstate(invalid="ignore"):
            imp = permutation_importance(
                wrapped_model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=variance_disruption_score
            )

        # Sanitize NaNs in importance outputs
        imp.importances_mean = np.nan_to_num(imp.importances_mean, nan=0.0)
        imp.importances_std = np.nan_to_num(imp.importances_std, nan=0.0)

        feat_names = self._feats_final

        # --- Safe column mean calculation ---
        def safe_mean(col, mask=None):
            try:
                if mask is not None:
                    return scored_grid.loc[mask, col].mean()
                return scored_grid[col].mean()
            except KeyError:
                return np.nan

        # --- Thresholding for top-cell filtering ---
        pos_mask = scored_grid['site_count'] > 0
        prospect_scores = scored_grid.loc[pos_mask, 'prospect_score']
        valid_scores = prospect_scores.dropna()

        if valid_scores.empty:
            print("âš ï¸� No valid prospect scores for thresholding; skipping importance computation.")
            return pd.DataFrame()

        cutoff5 = np.percentile(valid_scores, 95)
        cutoff10 = np.percentile(valid_scores, 90)

        # Pre-fill prospect scores to avoid invalid comparisons
        scored_grid['prospect_score_filled'] = scored_grid['prospect_score'].fillna(-1e6)

        with np.errstate(invalid='ignore'):
            top5 = pd.Series(np.greater_equal(scored_grid['prospect_score_filled'].values, cutoff5), index=scored_grid.index)
            top10 = pd.Series(np.greater_equal(scored_grid['prospect_score_filled'].values, cutoff10), index=scored_grid.index)

        # --- Assemble final importance summary ---
        fi = pd.DataFrame({
            "feature": feat_names,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
            "mean_all_cells": [safe_mean(f) for f in feat_names],
            "mean_top_5pct": [safe_mean(f, top5) for f in feat_names],
            "mean_top_10pct": [safe_mean(f, top10) for f in feat_names],
            "description": [self.feat_desc.get(f, "") for f in feat_names]
        })

        print(f"ğŸ“Š compute_importance: permutation took {time.time() - t0:.2f}s")
        return fi.sort_values("importance_mean", ascending=False)

    def _check_num_neighbors(self, gdf: gpd.GeoDataFrame):
        """
        Diagnostic utility to compute the number of neighbors (adjacent cells) for each grid cell.

        - A neighbor is defined as a cell that *touches* the current one.
        - Uses spatial join to identify neighbors.
        - Prints summary statistics of neighbor counts.
        - Adds a new column: `num_neighbors` to the input GeoDataFrame.

        Raises:
            AssertionError if any cell has zero neighbors (which may indicate an isolated polygon).
        """
        # Create pairwise combinations of all cells
        left = gdf.reset_index()[['cell_id', 'geometry']]
        right = left.rename(columns={'geometry': 'nbr_geom', 'cell_id': 'nbr'})

        # Perform spatial join to find touching neighbors
        nbrs = gpd.sjoin(
            left.set_geometry('geometry'),
            right.set_geometry('nbr_geom'),
            predicate='touches',
            how='left'
        )

        # Remove self-matches
        nbrs = nbrs[nbrs['cell_id'] != nbrs['nbr']]

        # Count neighbors per cell
        counts = nbrs.groupby('cell_id')['nbr'].count()
        gdf['num_neighbors'] = gdf.index.map(counts).fillna(0).astype(int)

        # Print summary
        print("Neighbor count summary:")
        print(gdf['num_neighbors'].describe())
        print(f"Cells with fewer than 8 neighbors: {(gdf['num_neighbors'] < 8).sum()}")

        # Sanity check: no cells should be completely isolated
        assert (gdf['num_neighbors'] >= 1).all(), "ERROR: Found a cell with zero neighbors!"

    def compute_forest_pct_from_tif(
        self,
        tif_path: str,
        grid: gpd.GeoDataFrame,
    ) -> pd.Series:
        """
        Computes the fraction of each grid cell that is potentially forested based on categorical raster input.

        Forest classification rules:
        - Forest pixels = class 3 (forest) or class 0 (unspecified)
        - Non-forest = all other valid class codes
        - NoData values are excluded from both numerator and denominator

        Parameters:
            tif_path (str): path to land cover classification raster (e.g. MapBiomas)
            grid (GeoDataFrame): analysis grid with polygon geometries

        Returns:
            Series: forest_pct values indexed by grid cell
        """
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            raster_crs = src.crs

            # Replace nodata values with NaN for masking
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)

            # Define masks
            possible_forest_mask = np.isin(arr, [0, 3])  # 0 = unknown, 3 = forest
            valid_mask = ~np.isnan(arr)

            # Project grid to match raster CRS
            grid_proj = grid.to_crs(raster_crs)

            # Rasterize grid cell IDs to match raster shape
            shapes = ((geom, cid) for cid, geom in zip(grid_proj.index, grid_proj.geometry))
            cell_ids = features.rasterize(
                shapes,
                out_shape=arr.shape,
                transform=src.transform,
                fill=-1,
                dtype='int32'
            )

            # Flatten arrays to vector format for grouping
            ids = cell_ids.ravel()
            forest_flat = possible_forest_mask.ravel()
            valid_flat = valid_mask.ravel()

            # Build DataFrame for aggregation
            df = pd.DataFrame({
                'id': ids,
                'forest': forest_flat.astype(int),
                'valid': valid_flat.astype(int)
            }).query('id >= 0')  # exclude background

            # Compute forest percentage
            num = df.groupby('id')['forest'].sum()
            den = df.groupby('id')['valid'].sum()
            forest_pct = num / den
            forest_pct = forest_pct.fillna(0)

        return forest_pct


    def _diagnostic_report(
        self,
        grid: gpd.GeoDataFrame,
        prev_scores: pd.Series,
        prev_threshold: float,
        nbrs: gpd.GeoDataFrame
    ):
        """
        Diagnostic tool that prints a multi-part report on model context, coverage, and filtering logic.

        Includes:
        ğŸ”¹ Site diagnostics:
            - Total number of sites
            - How many cells contain sites
            - How many cells are neighbors of a site
        ğŸŒ± Non-forest diagnostics (breakdown of conditions aâ€“e):
            - Non-forest dominant cells
            - Cells with no sites or neighboring sites
            - Cells isolated in terms of forest and archaeology
        ğŸ§ª Safe filtering diagnostics:
            - How many cells meet "safe" criteria via current model
            - How many are recoverable via neighboring "good" cell
            - Final safe coverage

        Returns:
            Boolean Series indicating which cells passed the composite "safe filter"
        """
        print("\n=== ğŸ“Œ Full Diagnostic Report ===")

        # ----------------------------------------------------------------------
        # ğŸ”¹ SITE COVERAGE STATS
        # ----------------------------------------------------------------------
        num_sites_total = len(self.sites)
        num_cells_with_sites = (grid['site_count'] > 0).sum()

        # Build neighbor linkages for site checking
        left = grid.reset_index()[['cell_id', 'geometry']]
        right = left.rename(columns={'cell_id': 'nbr', 'geometry': 'nbr_geom'})
        nbr_sites = nbrs.groupby('cell_id')['nbr'].apply(
            lambda ids: (grid.loc[ids, 'site_count'] > 0).any()
        ).reindex(grid.index, fill_value=False)

        num_cells_with_neighbor_sites = nbr_sites.sum()
        num_cells_with_site_or_neighbor = ((grid['site_count'] > 0) | nbr_sites).sum()

        print(f"ğŸ”¹ Total sites in self.sites: {num_sites_total}")
        print(f"ğŸ”¹ Grid cells with site_count > 0: {num_cells_with_sites} / {len(grid)}")
        print(f"ğŸ”¹ Grid cells with neighbor site: {num_cells_with_neighbor_sites}")
        print(f"ğŸ”¹ Grid cells with site OR neighbor site: {num_cells_with_site_or_neighbor}")

        # ----------------------------------------------------------------------
        # ğŸŒ± NON-FOREST DIAGNOSTICS
        # ----------------------------------------------------------------------
        mask_a = grid['non_forest_pct'] > 0.5
        mask_b = mask_a & (grid['site_count'] == 0)
        mask_c = mask_b & ~nbr_sites

        nbr_has_forest = nbrs.groupby('cell_id')['nbr'].apply(
            lambda ids: (grid.loc[ids, 'non_forest_pct'] < 0.5).any()
        ).reindex(grid.index, fill_value=False)

        mask_d = mask_a & ~nbr_has_forest
        mask_e = mask_d & ~nbr_sites

        print("\n--- Non-Forest Breakdown ---")
        print(f"(a) non-forest > 50%: {mask_a.sum()}")
        print(f"(b) + no sites: {mask_b.sum()}")
        print(f"(c) + no neighbor sites: {mask_c.sum()}")
        print(f"(d) + no neighbor forest: {mask_d.sum()}")
        print(f"(e) + no neighbor forest AND no neighbor sites: {mask_e.sum()}")

        # ----------------------------------------------------------------------
        # ğŸ§ª SAFE FILTER LOGIC (basic + neighbor coverage)
        # ----------------------------------------------------------------------
        basic_mask = (
            (grid['site_count'] > 0) |
            (prev_scores >= prev_threshold) |
            (grid['forest_pct'] >= 0.5)
        )

        nbrs = nbrs.copy()
        nbrs['nbr_is_good'] = nbrs['nbr'].map(basic_mask)
        neighbor_has_good = nbrs.groupby('cell_id')['nbr_is_good'].any()
        neighbor_has_good = neighbor_has_good.reindex(grid.index, fill_value=False)

        safe_mask = basic_mask | neighbor_has_good

        print("\n--- Safe Filter Breakdown ---")
        print(f"Basic mask TRUE: {basic_mask.sum()} / {len(basic_mask)}")
        print(f"Neighbor has good: {neighbor_has_good.sum()} / {len(neighbor_has_good)}")
        print(f"Safe filter keeps: {safe_mask.sum()} / {len(safe_mask)}")

        return safe_mask


    def _ensure_sites_crs(self, crs):
        """
        Ensures the internal 'sites' GeoDataFrame is projected to the specified CRS.

        Parameters:
            crs: A target coordinate reference system (CRS) to check against.
                 Can be a pyproj CRS object, EPSG string (e.g., 'EPSG:4326'), or dict.

        Returns:
            GeoDataFrame: The 'sites' layer reprojected to the given CRS if needed,
                          otherwise returned unchanged.
        """
        # If CRS does not match, reproject and return a copy
        if self.sites.crs != crs:
            return self.sites.to_crs(crs)
        
        # If already aligned, return as-is
        return self.sites

    def _run_single_resolution(
        self,
        basin_bounds: gpd.GeoDataFrame,
        size: float,
        prev_scored: gpd.GeoDataFrame | None,
        prev_threshold: float,
        top_pct: float,
        forest_pct: float,
        method: str,
        **svm_params
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, object, pd.DataFrame, float]:
        """
        Executes a single resolution pass for the quad-tree modeling pipeline.

        Parameters:
            basin_bounds (GeoDataFrame): Polygon or bounds to restrict the grid.
            size (float): Grid cell size in projected units (e.g., meters).
            prev_scored (GeoDataFrame | None): Output from previous coarser resolution, or None if first pass.
            prev_threshold (float): Score threshold used in previous resolution to select promising cells.
            top_pct (float): Proportion (0â€“1) of cells to retain for next level.
            forest_pct (float): Minimum forest cover proportion required to include a cell.
            method (str): Either 'svm' (OneClassSVM) or 'iforest' (IsolationForest).
            **svm_params: Parameters passed to the selected model constructor.

        Returns:
            tuple containing:
                - strict_grid (GeoDataFrame): Grid used for model training and scoring
                - next_region (GeoDataFrame): Subset of grid to retain for next resolution
                - model (object): Trained scikit-learn model
                - fi (DataFrame): Feature importance table
                - new_threshold (float): Updated score threshold based on top_pct
        """
        print(f"\n=== Running resolution {size} ===")

        # --- Step 1: Create or reuse grid ---
        if prev_scored is None:
            # First pass: create fresh grid
            grid = self.create_fixed_grid(basin_bounds, size)
            grid['prev_score'] = 0
        else:
            # Subsequent pass: reuse subdivided grid with prior scores
            grid = prev_scored.copy()
            if 'prospect_score' in grid.columns:
                grid['prev_score'] = grid['prospect_score']
            else:
                raise ValueError("Expected 'prospect_score' column for pass > 1")

        # --- Step 2: Compute forest percentage from raster ---
        forest_pct_series = self.compute_forest_pct_from_tif(self.biomass_tif, grid)
        grid['forest_pct'] = grid.index.map(lambda i: float(forest_pct_series.get(i, 0)))
        grid['non_forest_pct'] = 1.0 - grid['forest_pct']

        # --- Step 3: Assign site counts per grid cell ---
        sites_crs = self._ensure_sites_crs(grid.crs)
        sj = gpd.sjoin(sites_crs, grid, how='left', predicate='within')
        cnt = sj.groupby('index_right').size()
        grid['site_count'] = cnt.reindex(grid.index, fill_value=0)

        # --- Step 4: Build neighbor map (touching polygons) ---
        left = grid.reset_index()[['cell_id', 'geometry']]
        right = left.rename(columns={'cell_id': 'nbr', 'geometry': 'nbr_geom'})
        nbrs = gpd.sjoin(
            left.set_geometry('geometry'),
            right.set_geometry('nbr_geom'),
            predicate='touches',
            how='inner'
        )
        nbrs = nbrs[nbrs['cell_id'] != nbrs['nbr']]

        # --- Step 5: Construct 'safe' training mask ---
        basic_mask = (
            (grid['site_count'] > 0) |
            (grid['prev_score'] >= prev_threshold) |
            (grid['forest_pct'] >= forest_pct)
        )
        nbrs['nbr_is_good'] = nbrs['nbr'].map(basic_mask)
        neighbor_has_good = nbrs.groupby('cell_id')['nbr_is_good'].any().reindex(grid.index, fill_value=False)
        safe_mask = basic_mask | neighbor_has_good

        print("\n=== ğŸ“Œ Full Diagnostic Report ===")
        print(f"Basic mask TRUE: {basic_mask.sum()} / {len(basic_mask)}")
        print(f"Neighbor has good: {neighbor_has_good.sum()} / {len(neighbor_has_good)}")
        print(f"Safe filter keeps: {safe_mask.sum()} / {len(safe_mask)}")

        safe_grid = grid.loc[safe_mask].copy()

        # --- Step 6: Feature generation on filtered cells ---
        grid_feats = self.generate_features(safe_grid)

        # --- Step 7: Strict mask for training ---
        if prev_scored is None:
            strict_mask = (
                (grid_feats['site_count'] > 0) |
                (grid_feats['forest_pct'] >= forest_pct)
            )
        else:
            strict_mask = (
                (grid_feats['site_count'] > 0) |
                (
                    (grid_feats['forest_pct'] >= forest_pct) &
                    (grid_feats['prev_score'] >= prev_threshold)
                )
            )
        strict_grid = grid_feats.loc[strict_mask].copy()
        print(f"Strict filter keeps: {strict_grid.shape[0]} / {grid_feats.shape[0]}")

        # --- Step 8: Train model and score all ---
        model, scored_strict = self.train_model(strict_grid, method=method, **svm_params)
        strict_grid['prospect_score'] = scored_strict['prospect_score']

        # --- Step 9: Threshold to determine top N% ---
        new_threshold = strict_grid['prospect_score'].quantile(1 - top_pct)
        fi = self.compute_importance(model, strict_grid)

        # --- Step 10: Select region for next resolution ---
        pass_mask = (
            (strict_grid['site_count'] > 0) |
            (
                (strict_grid['forest_pct'] >= forest_pct) &
                (strict_grid['prospect_score'] >= new_threshold)
            )
        )
        next_region = strict_grid.loc[pass_mask, ['geometry', 'prospect_score']].copy()

        return strict_grid, next_region, model, fi, new_threshold

    def subdivide_grid(self, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Subdivide each grid cell into 4 equal quadrants (children).

        Each child:
          - Inherits the geometry of its sub-box (split evenly)
          - Retains its parent cell ID (as 'parent_id')
          - Copies the parent's prospect score (if present)

        This function is used in hierarchical modeling workflows where promising
        cells are recursively subdivided and re-scored.

        Parameters:
            grid (GeoDataFrame): The original coarser-resolution grid.

        Returns:
            GeoDataFrame with 4x as many rows as input, each representing a sub-cell.
        """
        from shapely.geometry import box

        # Containers for subdivided cells
        new_polys = []
        parent_ids = []
        parent_scores = []

        # Loop through each cell and subdivide
        for idx, row in grid.iterrows():
            xmin, ymin, xmax, ymax = row.geometry.bounds
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2

            # Create 4 quadrants (top-left, top-right, bottom-left, bottom-right)
            children = [
                box(xmin, ymid, xmid, ymax),   # top-left
                box(xmid, ymid, xmax, ymax),   # top-right
                box(xmin, ymin, xmid, ymid),   # bottom-left
                box(xmid, ymin, xmax, ymid),   # bottom-right
            ]

            # Store geometry + metadata for each child
            for child in children:
                new_polys.append(child)
                parent_ids.append(idx)
                parent_scores.append(row.get('prospect_score', 0))

        # Create GeoDataFrame for output
        subdivided = gpd.GeoDataFrame({
            'parent_id': parent_ids,
            'prospect_score': parent_scores,
            'geometry': new_polys
        }, crs=grid.crs)

        subdivided.index.name = 'cell_id'
        return subdivided

    def multi_resolution_prospecting(
        self,
        basin_bounds: gpd.GeoDataFrame,
        initial_cell_size: float,
        num_levels: int = 3,
        top_pct_min_max: list = [0.25, 0.25],
        forest_pct_min_max: list = [0.5, 0.5],
        method: str = 'svm',
        **svm_params
    ) -> dict:
        """
        Run a multi-resolution prospecting analysis using recursive grid subdivision.

        At each resolution:
          - The grid is scored using a one-class SVM or Isolation Forest.
          - Only high-scoring and forested regions are retained for the next level.
          - Features are recomputed at each level for increased precision.
          - The grid is subdivided to focus on finer spatial detail in promising areas.

        This implements a stable quad-tree refinement pipeline with inherited prospect scores.

        Parameters:
            basin_bounds (GeoDataFrame): The target region to grid and score.
            initial_cell_size (float): Starting grid cell size in CRS units (e.g., meters).
            num_levels (int): Number of zoom levels (each halves the resolution).
            top_pct_min_max (list): [min, max] top scoring percentiles per level.
            forest_pct_min_max (list): [min, max] forest coverage thresholds per level.
            method (str): Either 'svm' or 'iforest' for modeling.
            **svm_params: Parameters passed to the underlying model.

        Returns:
            Dictionary with keys = resolution levels and values = {
                'grid': scored GeoDataFrame,
                'importances': feature importance DataFrame,
                'threshold': cutoff score for that level
            }
        """
        results = {}

        # === First pass: generate coarse grid over the basin ===
        coarse_grid = self.create_fixed_grid(basin_bounds, initial_cell_size)
        coarse_grid['prospect_score'] = 0

        prev_grid = coarse_grid
        prev_threshold = 0.0  # no filtering on first pass

        for i, level in enumerate(range(num_levels)):
            resolution = initial_cell_size / (2 ** level)

            # Interpolate forest/top-pct thresholds for this level
            top_pcts = np.linspace(top_pct_min_max[0], top_pct_min_max[1], num_levels)
            forest_pcts = np.linspace(forest_pct_min_max[0], forest_pct_min_max[1], num_levels)
            top_pct = top_pcts[i]
            forest_pct = forest_pcts[i]

            print(f"\n=== Multi-resolution pass {level+1} of {num_levels}: {resolution}, top_pct: {top_pct}, forest_pct: {forest_pct} ===")

            # Run full pipeline: feature generation, training, scoring, thresholding
            scored, next_region, model, fi, new_threshold = self._run_single_resolution(
                basin_bounds=basin_bounds,
                size=resolution,
                prev_scored=prev_grid,
                prev_threshold=prev_threshold,
                top_pct=top_pct,
                forest_pct=forest_pct,
                method=method,
                **svm_params
            )

            # Store results for this resolution level
            results[resolution] = {
                'grid': scored,
                'importances': fi,
                'threshold': new_threshold
            }

            # Prepare next pass input: subdivide retained region
            prev_grid = self.subdivide_grid(next_region)
            prev_threshold = new_threshold

        return results



%%time
# ==============================================================================
# STEP 3: Run Multi-Resolution Prospecting Model
# ==============================================================================

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Initialize the ProspectingPipeline with all relevant spatial layers ---
print("\n--- Calculating prospecting scores for grid cells ---")

pp = ProspectingPipeline(
    sites_gdf = sites_gdf_proj,                   # Archaeological sites (projected)
    rivers_gdf = rivers_gdf_proj,                 # River lines (projected)
    sources_gdf = sources_gdf_proj,               # River sources (projected)
    intersections_gdf = intersections_gdf_proj,   # River confluences (projected)
    flood_tif="/kaggle/input/amazon-water-data/jrc_occurrence_amazon/amazon_occurrence_mask.tif",
    dem_tif="/kaggle/working/aligned_rasters/dem.tif",
    slope_tif = "/kaggle/working/aligned_rasters/slope.tif",
    tpi_tif="/kaggle/working/aligned_rasters/tpi.tif",
    tri_tif="/kaggle/working/aligned_rasters/tri.tif",
    plan_curv_tif ="/kaggle/working/aligned_rasters/plan_curv.tif",
    prof_curv_tif="/kaggle/working/aligned_rasters/prof_curv.tif",
    biomass_tif = "/kaggle/working/aligned_rasters/biomass.tif",
    elev_river_tif = "/kaggle/working/aligned_rasters/elev_river.tif"
)

# --- 2. Set analysis parameters ---
CELL_SIZE = 64000         # Starting grid resolution in meters
LEVELS = 7                # Number of refinement levels (each halves resolution)
FOREST_PCT = [0.25, 0.9]  # Min/max forest thresholds across levels
TOP_PCT = [0.25, 0.05]    # Min/max top scoring cell thresholds

# --- 3. Model configuration for one-class SVM ---
svm_params = {
    'nu': 0.05,           # Controls false positive rate / boundary tightness
    'kernel': 'rbf',      # Radial basis function
    'gamma': 'scale'      # Kernel coefficient
}

# --- 4. Run multi-resolution prospecting model ---
results = pp.multi_resolution_prospecting(
    basin_bounds=grid_gdf_proj,             # Region to analyze (already projected)
    initial_cell_size = CELL_SIZE,          # Start at 64km grid
    num_levels=LEVELS,                      # How many resolution levels to process
    top_pct_min_max = TOP_PCT,              # Thresholds for selecting top cells
    forest_pct_min_max = FOREST_PCT,        # Thresholds for forest coverage
    method='svm',                           # Use one-class SVM
    **svm_params                            # Pass in model hyperparameters
)



# ==============================================================================
# STEP 4: Save Model Results and Feature Importances
# ==============================================================================

# Save the complete multi-resolution results dictionary as a pickle file
# This includes all grids, feature importances, models, and thresholds per resolution level.
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("âœ… Saved full results dictionary to 'results.pkl'")

# Export the feature importance table for each resolution to a separate CSV file
# Each file is named using the corresponding cell size (e.g., featimp_64000.csv)
for n in range(LEVELS):
    cell_size = CELL_SIZE / 2**n
    output_path = f'featimp_{cell_size}.csv'
    results[cell_size]['importances'].to_csv(output_path)
    print(f"ğŸ“„ Feature importances saved to {output_path}")


