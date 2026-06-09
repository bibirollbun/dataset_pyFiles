!pip install -q rasterio
!pip install -q selenium


# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================
print("--- Loading Libraries ---")

# --------------------------------------------------------------------------
# Core Python
# --------------------------------------------------------------------------
import base64
import math
import os
import shutil
import time
from io import BytesIO

# --------------------------------------------------------------------------
# Data Handling
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd

# --------------------------------------------------------------------------
# Visualization & Mapping
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import folium
from folium import GeoJson, GeoJsonTooltip, MacroElement
from folium.plugins import FeatureGroupSubGroup, FastMarkerCluster
from folium.raster_layers import ImageOverlay
from folium.utilities import JsCode
from branca.colormap import linear
from branca.element import MacroElement

# --------------------------------------------------------------------------
# Web Automation (Selenium)
# --------------------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --------------------------------------------------------------------------
# Raster I/O and Processing
# --------------------------------------------------------------------------
import rasterio
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from rasterio.transform import array_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds
)
from rasterio.crs import CRS


# --------------------------------------------------------------------------
# Geometry and Spatial Operations
# --------------------------------------------------------------------------
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping
from shapely.ops import unary_union
from scipy.ndimage import label

# --------------------------------------------------------------------------
# Statistics and Scaling
# --------------------------------------------------------------------------
from scipy.stats import gaussian_kde, zscore
from sklearn.preprocessing import minmax_scale

# --------------------------------------------------------------------------
# HTML Templating
# --------------------------------------------------------------------------
from jinja2 import Template

# --------------------------------------------------------------------------
# Image Processing
# --------------------------------------------------------------------------
from PIL import Image


# --------------------------------------------------------------------------
# Initialize file paths and runtime configuration
# --------------------------------------------------------------------------
print("--- Initializing Configuration ---")

# Root input folder (adjustable for local vs. Kaggle execution)
KAGGLE_INPUT_ROOT = "/kaggle/input/"

# --- Input data paths ---
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
# Analysis Parameters
# --------------------------------------------------------------------------

# Projected CRS for geospatial analysis â€” must use meters. 
# Example: SIRGAS 2000 / UTM Zone 20S (suitable for central-western Amazon).
TARGET_PROJECTED_CRS = "EPSG:31980"

# Optional override:
CELL_SIZE_KM = 20.0  # Uncomment to specify cell size in kilometers

# Buffer distance (in meters) around river lines to simulate river width
RIVER_BUFFER_METERS = 50

# --------------------------------------------------------------------------
# Define land cover category color palette (MapBiomas legend)
# --------------------------------------------------------------------------
mapbiomas_colors = {
    #0: '#FFFFFF',   # Non-observed / background
    3: '#006400',   # Forest
    4: '#22B14C',   # Flooded forest / VÃ¡rzea
    5: '#7CFC00',   # Grassland / Natural savanna
    6: '#228B22',   # Shrubland
    11: '#66FF66',  # Wetlands / Permanently flooded non-forest
    12: '#E2E2E2',  # Pasture / Managed grassland
    13: '#FFFFCC',  # Cropland
    21: '#FFCC99',  # Urban / Built-up
    22: '#999999',  # Mining / bare ground
    33: '#B3B3B3',  # Rocky outcrop
    34: '#0000FF'   # Water bodies
}


# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

def get_str_val(value, default_if_empty="N/A"):
    """
    Safely converts a value to a clean string, handling missing, null, or placeholder values.

    Parameters:
    ----------
    value : any
        The input value to convert to a string.
    default_if_empty : str, optional
        The default string to return if the value is None, NaN, or 'nan'.

    Returns:
    -------
    str
        Cleaned string representation of the input or a default fallback.
    """
    # Check for NaN or None using pandas
    if pd.isna(value): 
        return default_if_empty

    # Convert to string and trim whitespace
    s_value = str(value).strip()

    # Return default if the result is empty or a stringified 'nan'
    return default_if_empty if not s_value or s_value.lower() == 'nan' else s_value


def reproject_geojson_to_wgs84(
    input_geojson_path: str, 
    output_geojson_path: str, 
    default_source_crs: str = "EPSG:3857"
) -> bool:
    """
    Reads a GeoJSON file, reprojects it to WGS84 (EPSG:4326) if necessary, 
    and saves the output. It uses the GeoJSON's internal CRS if available.

    Parameters:
    ----------
    input_geojson_path : str
        Path to the source GeoJSON file.
    output_geojson_path : str
        Destination path to save the reprojected GeoJSON.
    default_source_crs : str, optional
        CRS to assume if input file has no CRS defined.

    Returns:
    -------
    bool
        True if the output file is created successfully, False otherwise.
    """
    print(f"\n--- Reprojecting GeoJSON: {os.path.basename(input_geojson_path)} ---")

    # Check if input exists
    if not os.path.exists(input_geojson_path):
        print(f"ERROR: Input GeoJSON not found at {input_geojson_path}.")
        return False

    try:
        # Load GeoJSON
        gdf_original = gpd.read_file(input_geojson_path)
        print(f"Successfully read original GeoJSON. Number of features: {len(gdf_original)}")

        # Determine CRS from file or fallback
        source_crs_for_reprojection = None
        if gdf_original.crs:
            print(f"GeoJSON file's detected internal CRS: {gdf_original.crs}")
            source_crs_for_reprojection = gdf_original.crs
        else:
            # Fallback CRS if not defined
            print(f"Warning: No CRS found in GeoJSON file. Assuming source CRS is {default_source_crs}.")
            gdf_original.crs = default_source_crs
            source_crs_for_reprojection = gdf_original.crs

        target_crs_wgs84 = "EPSG:4326"

        # If already WGS84, just copy the file
        if source_crs_for_reprojection.to_epsg() == 4326:
            print("GeoJSON is already in WGS84 (EPSG:4326). Copying file directly.")
            shutil.copy2(input_geojson_path, output_geojson_path)
        else:
            # Reproject and save
            print(f"Reprojecting from CRS ({source_crs_for_reprojection}) to {target_crs_wgs84}...")
            gdf_wgs84 = gdf_original.to_crs(target_crs_wgs84)
            gdf_wgs84.to_file(output_geojson_path, driver="GeoJSON")

        print(f"Processed GeoJSON saved to: {output_geojson_path}")
        return output_geojson_path

    except Exception as e:
        # Catch and print full stack trace
        print(f"An error occurred during GeoJSON reprojection: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_analysis_grid(basin_polygon_wgs84, cell_size_km):
    """
    Creates a regular rectangular analysis grid (GeoDataFrame) over a WGS84 Amazon basin polygon.

    The function:
    - Estimates degree spacing for the requested cell size (in kilometers)
    - Builds a lat/lon grid of bounding boxes
    - Filters out cells that do not intersect the basin polygon

    Parameters:
    ----------
    basin_polygon_wgs84 : shapely Polygon or MultiPolygon
        The basin polygon geometry in WGS84 (EPSG:4326).
    cell_size_km : float
        Desired cell size in kilometers.

    Returns:
    -------
    GeoDataFrame or None
        A GeoDataFrame of valid grid cells intersecting the basin, or None on error.
    """
    if basin_polygon_wgs84 is None or basin_polygon_wgs84.is_empty:
        print("ERROR: Basin polygon is not valid. Cannot create grid.")
        return None

    # Get lat/lon bounding box of the basin polygon
    min_lon, min_lat, max_lon, max_lat = basin_polygon_wgs84.bounds
    print(f"Basin bounds (WGS84): MinLon {min_lon:.2f}, MinLat {min_lat:.2f}, MaxLon {max_lon:.2f}, MaxLat {max_lat:.2f}")

    # Estimate approximate degrees per km
    center_lat_rad = np.deg2rad((min_lat + max_lat) / 2)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(center_lat_rad)
    if km_per_deg_lon == 0:
        return None

    # Convert desired km cell size to degree steps
    delta_lat_approx = cell_size_km / km_per_deg_lat
    delta_lon_approx = cell_size_km / km_per_deg_lon
    print(f"Approx. cell size in degrees: dLat={delta_lat_approx:.4f}, dLon={delta_lon_approx:.4f}")

    # Build grid coordinates
    lon_coords = np.arange(min_lon, max_lon, delta_lon_approx)
    lat_coords = np.arange(min_lat, max_lat, delta_lat_approx)

    # Generate rectangular cells (bounding boxes)
    grid_cells = [
        box(lon, lat, lon + delta_lon_approx, lat + delta_lat_approx)
        for lon in lon_coords for lat in lat_coords
    ]
    if not grid_cells:
        return None

    # Construct GeoDataFrame and filter cells inside basin
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    print(f"Generated {len(grid_gdf)} raw grid cells. Filtering by intersection with basin polygon...")
    grid_gdf_in_basin = grid_gdf[grid_gdf.intersects(basin_polygon_wgs84)]
    print(f"Retained {len(grid_gdf_in_basin)} grid cells intersecting the basin.")

    return grid_gdf_in_basin


def load_amazon_basin_polygon(geojson_path, target_crs="EPSG:4326", simplify_tol=0.05):
    """
    Loads a GeoJSON file containing the Amazon basin polygon,
    reprojects it to the desired CRS, optionally simplifies it,
    and returns the unified shapely geometry.

    Parameters:
    ----------
    geojson_path : str
        Path to the input GeoJSON file.
    target_crs : str, optional
        Desired coordinate reference system (default is "EPSG:4326").
    simplify_tol : float, optional
        Simplification tolerance in CRS units. Set to 0 to skip simplification.

    Returns:
    -------
    shapely.geometry.Polygon or MultiPolygon
        Unified geometry of the basin area.
    """
    import geopandas as gpd

    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Polygon file not found: {geojson_path}")

    # Read GeoJSON as GeoDataFrame
    basin_gdf = gpd.read_file(geojson_path)

    # Reproject if needed
    if basin_gdf.crs != target_crs:
        print(f"ğŸ”„ Reprojecting basin polygon from {basin_gdf.crs} to {target_crs}")
        basin_gdf = basin_gdf.to_crs(target_crs)

    # Merge into one unified polygon
    unified = basin_gdf.unary_union

    # Optionally simplify geometry
    if simplify_tol:
        unified = unified.simplify(simplify_tol)

    return unified



def load_amazon_basin_polygon(geojson_path, target_crs="EPSG:4326", simplify_tol=0.05):
    """
    Loads a GeoJSON file containing the Amazon basin polygon,
    reprojects it to a specified CRS, simplifies it if requested,
    and returns the unified geometry as a shapely object.

    Parameters:
    ----------
    geojson_path : str
        Path to the input GeoJSON file containing basin boundaries.
    target_crs : str, optional
        Target CRS to reproject the geometry to (default: "EPSG:4326").
    simplify_tol : float, optional
        Simplification tolerance. If 0, simplification is skipped.

    Returns:
    -------
    shapely.geometry.Polygon or MultiPolygon
        Unified (and optionally simplified) basin geometry.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Polygon file not found: {geojson_path}")

    # Load GeoJSON into a GeoDataFrame
    gdf = gpd.read_file(geojson_path)

    # Reproject if CRS doesn't match target
    if gdf.crs != target_crs:
        print(f"ğŸ”„ Reprojecting basin from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Merge all geometries into a single unified polygon/multipolygon
    union = gdf.unary_union

    # Return simplified geometry if requested
    return union.simplify(simplify_tol) if simplify_tol else union

# -------------------------------------------------------------------
# PHASE 1: Preparing and Projecting All Geospatial Data
# -------------------------------------------------------------------
print("\n--- PHASE 1: Preparing and Projecting All Geospatial Data ---")

# -------------------------------------------------------------------
# Load and simplify Amazon Basin polygon
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
# Load, filter, and project archaeological site data
# -------------------------------------------------------------------
sites_gdf_proj = gpd.GeoDataFrame()
try:
    if os.path.exists(arch_sites_csv_path):
        # Read sites from CSV and convert to WGS84 GeoDataFrame
        sites_df = pd.read_csv(arch_sites_csv_path).dropna(subset=['latitude', 'longitude'])
        sites_gdf_wgs84 = gpd.GeoDataFrame(
            sites_df,
            geometry=gpd.points_from_xy(sites_df.longitude, sites_df.latitude),
            crs="EPSG:4326"
        ).dropna(subset=['geometry'])

        # Filter sites to those inside the basin polygon
        if amazon_basin_geom:
            sites_gdf_in_basin = sites_gdf_wgs84[sites_gdf_wgs84.geometry.within(amazon_basin_geom)].copy()
        else:
            sites_gdf_in_basin = sites_gdf_wgs84

        # Project to analysis CRS if any sites remain
        if not sites_gdf_in_basin.empty:
            print(f"Projecting {len(sites_gdf_in_basin)} sites to {TARGET_PROJECTED_CRS}...")
            sites_gdf_proj = sites_gdf_in_basin.to_crs(TARGET_PROJECTED_CRS)

        print(f"Prepared {len(sites_gdf_proj)} sites for analysis.")
    else:
        print(f"âš ï¸� Sites CSV not found at {arch_sites_csv_path}")
except Exception as e:
    print(f"â�Œ Error preparing sites data: {e}")

# -------------------------------------------------------------------
# Load, filter, and project LiDAR survey area polygons
# -------------------------------------------------------------------
lidar_areas_gdf_proj = gpd.GeoDataFrame()
if os.path.exists(grouped_lidar_metadata_csv_path):
    try:
        lidar_df = pd.read_csv(grouped_lidar_metadata_csv_path)

        # Ensure all required bounding box columns are present and numeric
        coord_cols = ['group_id', 'lon_min_wgs84', 'lat_min_wgs84', 'lon_max_wgs84', 'lat_max_wgs84']
        lidar_df.dropna(subset=coord_cols, inplace=True)
        for col in coord_cols[1:]:
            lidar_df[col] = pd.to_numeric(lidar_df[col], errors='coerce')
        lidar_df.dropna(subset=coord_cols, inplace=True)

        if not lidar_df.empty:
            # Convert bounding box rows to rectangular geometries
            lidar_geoms = [
                box(r['lon_min_wgs84'], r['lat_min_wgs84'], r['lon_max_wgs84'], r['lat_max_wgs84'])
                for _, r in lidar_df.iterrows()
            ]
            lidar_gdf_wgs84 = gpd.GeoDataFrame(lidar_df, geometry=lidar_geoms, crs="EPSG:4326")

            # Clip LiDAR polygons to basin boundary
            if amazon_basin_geom:
                lidar_gdf_wgs84 = lidar_gdf_wgs84[
                    lidar_gdf_wgs84.geometry.intersects(amazon_basin_geom)
                ].copy()

            # Project LiDAR geometries if any remain
            if not lidar_gdf_wgs84.empty:
                print(f"Projecting {len(lidar_gdf_wgs84)} LiDAR areas to {TARGET_PROJECTED_CRS}...")
                lidar_areas_gdf_proj = lidar_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)

            print(f"Prepared {len(lidar_areas_gdf_proj)} LiDAR area polygons.")
    except Exception as e:
        print(f"â�Œ Error preparing LiDAR data: {e}")

# -------------------------------------------------------------------
# Create and project analysis grid
# -------------------------------------------------------------------
grid_gdf_proj = gpd.GeoDataFrame()
if amazon_basin_geom and not amazon_basin_geom.is_empty:
    # Generate grid covering the basin polygon (in WGS84)
    grid_gdf_wgs84 = create_analysis_grid(amazon_basin_geom, CELL_SIZE_KM)

    # Reproject to analysis CRS
    if grid_gdf_wgs84 is not None and not grid_gdf_wgs84.empty:
        print(f"Projecting {len(grid_gdf_wgs84)} grid cells to {TARGET_PROJECTED_CRS}...")
        grid_gdf_proj = grid_gdf_wgs84.to_crs(TARGET_PROJECTED_CRS)
        print("ğŸ§± Analysis grid created and projected.")




import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import mapping

def reproject_to_web_mercator(input_path, output_path, amazon_basin_geom=None, force=False):
    """
    Reprojects a raster to EPSG:3857 and optionally clips it to the Amazon basin polygon.

    Parameters:
    ----------
    input_path : str
        Path to the input raster file.
    output_path : str
        Path to the reprojected output file.
    amazon_basin_geom : shapely.geometry (optional)
        Amazon basin polygon (EPSG:4326) for clipping.
    force : bool
        If True, reprocess even if output already exists.

    Returns:
    -------
    str : Path to the reprojected and optionally clipped raster.
    """
    if os.path.exists(output_path) and not force:
        print(f"ğŸ”� Using existing file: {output_path}")
        return output_path

    dst_crs = CRS.from_epsg(3857)

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'nodata': 0
        })

        # Reproject to in-memory file
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as tmp:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(tmp, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

            with memfile.open() as tmp:
                if amazon_basin_geom:
                    basin_gdf = gpd.GeoDataFrame(geometry=[amazon_basin_geom], crs="EPSG:4326").to_crs(dst_crs)
                    shapes = [mapping(geom) for geom in basin_gdf.geometry]

                    clipped_data, clipped_transform = mask(tmp, shapes=shapes, crop=True, nodata=0)

                    profile.update({
                        'height': clipped_data.shape[1],
                        'width': clipped_data.shape[2],
                        'transform': clipped_transform
                    })

                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(clipped_data)
                else:
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(tmp.read())

    print(f"âœ… Saved reprojected raster: {output_path}")
    return output_path


# --------------------------------------------------------------------------
# Reproject elevation-above-river raster to EPSG:3857 (Web Mercator)
# --------------------------------------------------------------------------
reproject_to_web_mercator(
    input_path="/kaggle/input/elevation-above-river-amazon/amazon_elevation_above_river.tif",
    output_path="/kaggle/working/elevation_above_river_3857.tif",
    amazon_basin_geom=amazon_basin_geom,
    force=True
)

# --------------------------------------------------------------------------
# Reproject MapBiomas 2018 land cover map to EPSG:3857 for web display
# --------------------------------------------------------------------------
reproject_to_web_mercator(
    input_path="/kaggle/input/amazon-biomas-map/MapBiomas_Amazon_2018_EPSG4326.tif",
    output_path="/kaggle/working/MapBiomas_Amazon_2018_EPSG3857.tif",
    amazon_basin_geom=amazon_basin_geom,
    force=True
)


# ---------------------------------------------------------------------------------
# Macro: Click-to-GoogleEarth (opens one window and reuses it)
# ---------------------------------------------------------------------------------
class ClickToGoogleEarth(MacroElement):
    """
    A Folium extension that enables Google Earth navigation from clicked map features.

    When a GeoJSON layer is clicked, this macro:
    - Calculates the clicked featureâ€™s center coordinates
    - Estimates zoom altitude based on the featureâ€™s vertical span
    - Opens (or reuses) a browser tab pointing to the featureâ€™s location in Google Earth Web

    Parameters:
    ----------
    geojson_name : str
        The name of the GeoJSON layer to attach click events to.
    """
    def __init__(self, geojson_name):
        super().__init__()
        self._name = "ClickToGoogleEarth"
        self.geojson_name = geojson_name

        # JavaScript template to handle click-to-Google-Earth behavior
        self._template = Template("""
        {% macro script(this, kwargs) %}
        var gj = {{ this.geojson_name }};
        gj.eachLayer(function(layer) {
            layer.on('click', function() {
                var bounds = layer.getBounds();
                var center = bounds.getCenter();
                var lat = center.lat.toFixed(6);
                var lng = center.lng.toFixed(6);

                // Estimate vertical span and use it to determine altitude
                var spanLat = Math.abs(bounds.getNorth() - bounds.getSouth());
                var approx_km = spanLat * 111.0;
                var alt_m = Math.max(3000, approx_km * 1500);

                // Construct the Google Earth Web URL
                var url = `https://earth.google.com/web/@${lat},${lng},${alt_m}a`;

                // âœ… Open in or reuse a browser tab named 'earthTab'
                window.open(url, 'earthTab');  // works in Safari + Chrome
            });
        });
        {% endmacro %}
        """)



# ---------------------------------------------------------------------------------
# Cluster styling & JS (for grouped site icons)
# ---------------------------------------------------------------------------------
custom_cluster_styles_css = """
<style>
/* White text with bold font for cluster labels */
.marker-cluster div span {
    color: white;
    font-weight: bold;
}

/* Consistent static color for all archaeological site clusters */
.marker-cluster.sites-cluster-static-color div {
    background-color: rgba(40, 167, 69, 0.85) !important;
}
</style>
"""


site_cluster_icon_creator_js = """
<script>
/**
 * Creates a Leaflet DivIcon based on cluster size.
 * Adds a static color and adjusts size class dynamically.
 */
function site_cluster_icon_creator_func(cluster) {
    var count = cluster.getChildCount();
    var c = ' marker-cluster-';

    if (count < 10) {
        c += 'small';
    } else if (count < 100) {
        c += 'medium';
    } else {
        c += 'large';
    }

    return new L.DivIcon({
        html: '<div><span>' + count + '</span></div>',
        className: 'marker-cluster' + c + ' sites-cluster-static-color',
        iconSize: null
    });
}
</script>
"""


# ---------------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------------

def ensure_wgs84(gdf):
    """
    Ensures that a GeoDataFrame is in WGS84 (EPSG:4326) CRS.

    If the GeoDataFrame is not already in EPSG:4326, it is reprojected.

    Parameters:
    ----------
    gdf : GeoDataFrame
        The input GeoDataFrame.

    Returns:
    -------
    GeoDataFrame
        A GeoDataFrame reprojected to EPSG:4326, if needed.
    """
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        return gdf.to_crs("EPSG:4326")
    return gdf


def inject_map_assets(m):
    """
    Injects custom JavaScript and CSS assets into a Folium map.

    Assets include:
    - Custom cluster styling (green background, white text)
    - A JavaScript icon creator for consistent cluster appearance

    This function prevents duplicate injection using an internal flag.

    Parameters:
    ----------
    m : folium.Map
        The Folium map object to enhance.
    """
    if getattr(m, '_assets_injected', False):
        return  # Already injected

    # Add style and JS to map header
    m.get_root().header.add_child(folium.Element(custom_cluster_styles_css))
    m.get_root().header.add_child(folium.Element(site_cluster_icon_creator_js))
    m._assets_injected = True


# ---------------------------------------------------------------------------------
# Layer: Prospect Score Choropleth
# ---------------------------------------------------------------------------------

def add_prospect_choropleth(m, grid, size_km, show, legend_name=None):
    """
    Adds a choropleth layer to the Folium map, colored by 'prospect_score',
    and shows a tooltip with score, site count, and centroid coordinates.

    Parameters:
    ----------
    m : folium.Map
        The map object to which the choropleth will be added.
    grid : GeoDataFrame
        Grid containing 'prospect_score' and 'site_count' columns.
    size_km : int
        Cell size in kilometers (used in the layer name).
    show : bool
        Whether to show the layer by default.
    legend_name : str, optional
        Caption to display alongside the colormap legend.

    Returns:
    -------
    None
        Adds a colored layer and tooltip to the map in-place.
    """
    # Create color scale based on valid prospect scores
    scores = grid['prospect_score'].dropna()
    cmap = linear.YlOrRd_09.scale(scores.min(), scores.max())

    # Ensure centroid lat/lon columns exist
    if 'lat_center' not in grid.columns or 'lon_center' not in grid.columns:
        # Safely compute lat/lon centroids
        projected = grid.to_crs(grid.estimate_utm_crs())  # or use a fixed EPSG like 3857
        centroids_projected = projected.geometry.centroid
        centroids_wgs84 = centroids_projected.to_crs("EPSG:4326")
        grid['lat_center'] = centroids_wgs84.y
        grid['lon_center'] = centroids_wgs84.x


    # Define how each grid cell is styled
    style_fn = lambda f: {
        'fillColor': cmap(f['properties']['prospect_score'])
            if f['properties']['prospect_score'] is not None else 'rgba(0,0,0,0)',
        'fillOpacity': 0.7,
        'color': '#444',
        'weight': 0.5,
        'opacity': 0.3
    }

    # Create the GeoJson overlay layer
    gj = GeoJson(
        grid,
        name=f"Prospect {size_km} km",
        style_function=style_fn,
        tooltip=GeoJsonTooltip(
            fields=["prospect_score", "site_count", "lat_center", "lon_center"],
            aliases=["Prospect Score:", "Site Count:", "Latitude:", "Longitude:"],
            localize=True,
            sticky=False
        ),
        show=show
    ).add_to(m)

    # Enable Google Earth click-to-zoom integration
    m.get_root().add_child(ClickToGoogleEarth(gj.get_name()))

    # Add optional legend
    if legend_name:
        cmap.caption = legend_name
        cmap.add_to(m)



def add_cluster_outlines(
    m, grid, size_km, min_cluster_size=10, show=False, label_clusters=True
):
    """
    Identify and visualize clusters of contiguous high-prospect cells on a Folium map.

    A cluster is defined as a group of contiguous grid cells (based on 8-way connectivity)
    where site_count == 0. Clusters are:
      - Filtered to a minimum size (default: 10 cells)
      - Ranked by their maximum prospect score (cluster_rank)
      - Outlined on the map
      - Optionally labeled with their rank

    Additionally:
    - For cells in ranked clusters, saves centroid coordinates in lat/lon as columns
      'lat_center' and 'lon_center', and writes to clustered_cells.csv

    Returns:
    -------
    GeoDataFrame
        The updated grid with 'cluster_id' and 'cluster_rank' columns added.
        Also writes a CSV of all clustered cells: "clustered_cells.csv"
    """
    from scipy.ndimage import label
    from shapely.ops import unary_union
    import numpy as np
    import geopandas as gpd
    import pandas as pd
    from folium import GeoJson, GeoJsonTooltip, Marker, DivIcon
    from branca.colormap import linear

    scores = grid['prospect_score'].dropna()
    if scores.empty:
        return grid

    cmap = linear.YlOrRd_09.scale(scores.min(), scores.max())

    if grid.crs is None or grid.crs.to_epsg() in (4326, 4269):
        print("âš ï¸� Grid is in geographic CRS â€” converting to UTM for clustering...")
        grid = grid.to_crs(grid.estimate_utm_crs())

    grid = grid.copy()
    grid['cluster_candidate'] = (grid['site_count'] == 0).astype(int)

    resolution = np.sqrt(grid.geometry.area.median())
    bounds = grid.total_bounds
    cols = int((bounds[2] - bounds[0]) / resolution)
    rows = int((bounds[3] - bounds[1]) / resolution)

    centroids = grid.geometry.centroid
    grid['row'] = ((centroids.y - bounds[1]) / resolution).astype(int)
    grid['col'] = ((centroids.x - bounds[0]) / resolution).astype(int)

    mat = np.zeros((rows + 1, cols + 1), dtype=int)
    for idx, row in grid.iterrows():
        if 0 <= row.row <= rows and 0 <= row.col <= cols:
            mat[row.row, row.col] = row.cluster_candidate

    structure = np.ones((3, 3))
    labeled, _ = label(mat, structure=structure)

    label_grid = np.full_like(mat, -1)
    label_grid[mat == 1] = labeled[mat == 1]

    grid['cluster_id'] = grid.apply(
        lambda r: label_grid[r.row, r.col]
        if 0 <= r.row <= rows and 0 <= r.col <= cols
        else -1,
        axis=1,
    )

    cluster_sizes = grid.groupby('cluster_id').size()
    valid_ids = cluster_sizes[cluster_sizes >= min_cluster_size].index
    valid_ids = valid_ids[valid_ids >= 0]

    cluster_info = []
    for cid in valid_ids:
        cells = grid[grid['cluster_id'] == cid]
        max_score = cells['prospect_score'].max()
        geom = unary_union(cells.geometry)
        centroid = geom.centroid
        centroid_ll = gpd.GeoSeries([centroid], crs=grid.crs).to_crs("EPSG:4326").iloc[0]

        cluster_info.append({
            'cluster_id': cid,
            'geometry': geom,
            'max_score': max_score,
            'centroid_lat': centroid_ll.y,
            'centroid_lon': centroid_ll.x
        })

    if not cluster_info:
        return grid

    cluster_info = sorted(cluster_info, key=lambda x: x['max_score'], reverse=True)
    for rank, c in enumerate(cluster_info, start=1):
        c['rank'] = rank

    cluster_id_to_rank = {c['cluster_id']: c['rank'] for c in cluster_info}
    grid['cluster_rank'] = grid['cluster_id'].map(cluster_id_to_rank).fillna(-1).astype(int)

    cluster_gdf = gpd.GeoDataFrame(cluster_info, crs=grid.crs).to_crs(epsg=4326)

    style_fn = lambda f: {
        'fillOpacity': 0,
        'color': '#5e3c99',
        'weight': 5,
        'opacity': 1.0,
        'dashArray': '3'
    }

    cluster_layer = GeoJson(
        cluster_gdf,
        name=f"Clusters ({size_km} km)",
        style_function=style_fn,
        tooltip=GeoJsonTooltip(
            fields=["rank", "max_score"],
            aliases=["Cluster Rank:", "Max Score in Cluster:"],
            localize=True
        ),
        show=show
    )
    cluster_layer.add_to(m)

    if label_clusters:
        for _, row in cluster_gdf.iterrows():
            label = Marker(
                location=[row['centroid_lat'], row['centroid_lon']],
                icon=DivIcon(
                    html=f"""<div style="font-size:14px; font-weight:bold; color:#5e3c99;">{row['rank']}</div>"""
                )
            )
            label.add_to(m)

    clustered_cells = grid[grid['cluster_rank'] > 0].copy()

    clustered_proj = clustered_cells.to_crs(clustered_cells.estimate_utm_crs())
    centers_proj = clustered_proj.geometry.centroid
    centers_wgs84 = gpd.GeoSeries(centers_proj, crs=clustered_proj.crs).to_crs("EPSG:4326")
    clustered_cells["lat_center"] = centers_wgs84.y
    clustered_cells["lon_center"] = centers_wgs84.x

    clustered_cells.to_csv("clustered_cells.csv", index=False)
    return grid


def add_site_density(m, grid, size_km, show):
    """
    Adds a shaded layer to the Folium map, highlighting grid cells with archaeological sites.

    Cells with `site_count > 0` are colored using a sequential blue scale.  
    The color intensity reflects the number of known sites in each cell.

    Parameters:
    ----------
    m : folium.Map
        The map object to which the site density layer will be added.
    grid : GeoDataFrame
        Grid containing 'site_count' and 'prospect_score' columns.
    size_km : int
        Cell size in kilometers (used in the layer name).
    show : bool
        Whether the layer is visible by default.

    Returns:
    -------
    None
        The layer and optional legend are added to the map in-place.
    """
    # Filter grid to cells containing at least one site
    subset = grid[grid['site_count'] > 0]
    if subset.empty:
        return  # Nothing to show

    # Create a color scale based on site count
    max_count = int(subset['site_count'].max())
    cmap = linear.Blues_09.scale(1, max_count)

    # Create GeoJson overlay for cells with site data
    gj = GeoJson(
        subset,
        name=f"Sites {size_km} km",
        style_function=lambda f: {
            'fillColor': cmap(f['properties']['site_count']),
            'fillOpacity': 1.0,
            'color': '#000',
            'weight': 0.5,
        },
        tooltip=GeoJsonTooltip(
            fields=["site_count", "prospect_score"],
            aliases=["Site Count:", "Prospect Score:"],
            localize=True
        ),
        show=show
    ).add_to(m)

    # Optionally add a legend if this layer is shown by default
    if show:
        cmap.caption = 'Site Count'
        cmap.add_to(m)


def add_all_sites_fast_clustered(map_obj, sites_gdf):
    """
    Adds all archaeological sites to the Folium map using a fast marker cluster.

    Each site is visualized with a `CircleMarker`, styled and sized according to its
    `weight` value (if available). Markers are grouped using a `MarkerCluster`
    with a custom icon renderer (`site_cluster_icon_creator_func`), and each includes
    a popup showing available site metadata.

    Parameters:
    ----------
    map_obj : folium.Map
        The Folium map object to which the cluster of sites will be added.
    sites_gdf : GeoDataFrame
        GeoDataFrame containing site point geometries and metadata columns such as:
        ['site', 'Name', 'Shape', 'size', 'weight', 'source'].

    Returns:
    -------
    None
        Modifies the map object in-place by adding a clustered marker layer.
    """
    from folium.plugins import MarkerCluster

    # Ensure input GeoDataFrame is in WGS84
    sites = ensure_wgs84(sites_gdf)

    # Create clustered marker group with custom icon renderer
    cluster = MarkerCluster(
        name="Archaeological Sites",
        overlay=True,
        control=True,
        icon_create_function="site_cluster_icon_creator_func"
    )
    cluster.add_to(map_obj)

    # Loop through each site row and add a marker with a popup
    for _, row in sites.iterrows():
        lat, lon = row.geometry.y, row.geometry.x

        # Build popup content from metadata
        popup_parts = []
        for label in ["site", "Name", "Shape", "size", "weight", "source"]:
            val = row.get(label, None)
            if val is not None and (not isinstance(val, float) or not np.isnan(val)):
                popup_parts.append(f"<b>{label}:</b> {val}")
        popup_html = "<br>".join(popup_parts)

        # Normalize weight (default = 1)
        weight = row.get("weight", 1)
        if weight is None or (isinstance(weight, float) and np.isnan(weight)):
            weight = 1

        # Generate RGB fill color based on weight scale (1â€“10)
        t = (weight - 1) / 9  # Scale to [0, 1]
        r = round(153 * (1 - t))
        g = round(255 * (1 - t) + 100 * t)
        b = round(153 * (1 - t))
        color = f"rgb({r},{g},{b})"

        # Add the site marker to the cluster
        folium.CircleMarker(
            location=[lat, lon],
            radius=4 + weight * 2,
            color="#003300",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup_html
        ).add_to(cluster)




def add_rivers_layer(m, river_fp, show=True):
    """
    Loads a river network GeoJSON and adds it as a styled vector layer to a Folium map.

    The layer is reprojected to WGS84 (EPSG:4326) if needed and displayed using a 
    light blue line style. This function adds the layer to a Folium FeatureGroup 
    named "Rivers", which appears in the layer control.

    Parameters:
    ----------
    m : folium.Map
        The Folium map object to which the river layer will be added.
    river_fp : str
        Path to the GeoJSON file containing river geometries.
    show : bool, optional
        Whether the layer is visible by default (default is True).

    Returns:
    -------
    None
        Adds the river layer directly to the map.
    """
    try:
        # Attempt to load the river file into a GeoDataFrame
        rivers = gpd.read_file(river_fp)
    except Exception as e:
        print(f"â�Œ Failed to load river file '{river_fp}': {e}")
        return

    # Ensure the river data is in WGS84 for compatibility with Leaflet/Folium
    if rivers.crs is None or rivers.crs.to_string() != "EPSG:4326":
        print("ğŸ”„ Reprojecting rivers to EPSG:4326")
        rivers = rivers.to_crs("EPSG:4326")

    # Create a named layer group for toggling in Folium
    river_group = folium.FeatureGroup(name="Rivers", show=show)

    # Add river geometries with a blue stroke style
    GeoJson(
        rivers,
        name="Rivers",
        style_function=lambda f: {
            'color': "#0066CC",  # Light blue stroke
            'weight': 1.5,
            'opacity': 0.7
        },
        show=show
    ).add_to(river_group)

    # Attach to map
    river_group.add_to(m)

    print("âœ… River layer added from file.")


def add_biomass_overlay(m, tif_path, color_map, opacity=0.7, show=True):
    """
    Adds a land cover raster overlay to a Folium map using a color-coded PNG image.
    Only applies colors to values in `color_map` to avoid rendering background.
    """
    with rasterio.open(tif_path) as src:
        dst_crs = "EPSG:3857"

        # Reproject to Web Mercator
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
                data = dst.read(1)

                # Mask only valid color values
                valid_values = set(color_map.keys())
                keep_mask = np.isin(data, list(valid_values))

                # Build RGBA image
                rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
                for val, hexcol in color_map.items():
                    mask = data == val
                    if np.any(mask):
                        r, g, b = tuple(int(hexcol[i:i+2], 16) for i in (1, 3, 5))
                        rgba[mask] = (r, g, b, int(opacity * 255))

                # Make everything else transparent
                rgba[~keep_mask] = (0, 0, 0, 0)

                # Save PNG
                image = Image.fromarray(rgba, mode="RGBA")
                png_path = "/kaggle/working/biomass_overlay.png"
                image.save(png_path)

                # Convert bounds to WGS84
                bounds_3857 = array_bounds(data.shape[0], data.shape[1], dst.transform)
                bounds_wgs84 = transform_bounds(dst_crs, "EPSG:4326", *bounds_3857)
                sw = [bounds_wgs84[1], bounds_wgs84[0]]
                ne = [bounds_wgs84[3], bounds_wgs84[2]]

                # Add image overlay
                ImageOverlay(
                    image=png_path,
                    bounds=[sw, ne],
                    name="MapBiomas Land Cover",
                    opacity=opacity,
                    interactive=False,
                    cross_origin=False,
                    zindex=2,
                    show=show
                ).add_to(m)

    print("âœ… Biomass overlay added with no background.")


def add_above_river_overlay(m, dem_tif_path, grid_gdf, show=False, name="Elevation Above River"):
    """
    Adds a colored raster overlay showing elevation above river (in meters) to a Folium map.

    The function:
    - Clips the DEM to the grid extent
    - Colorizes elevation using the `terrain` matplotlib colormap
    - Adds a PNG overlay to the map
    - Displays a gradient legend when the overlay is toggled on

    Parameters:
    ----------
    m : folium.Map
        The map to which the elevation overlay will be added.
    dem_tif_path : str
        Path to the elevation-above-river GeoTIFF.
    grid_gdf : GeoDataFrame
        The analysis grid used to clip the overlay extent.
    show : bool, optional
        Whether to display the overlay by default.
    name : str, optional
        The name of the overlay layer (used in the legend toggle).

    Returns:
    -------
    None
        The overlay and interactive legend are added to the map in-place.
    """
    if grid_gdf.empty:
        print("âš ï¸� No grid cells found for elevation overlay.")
        return

    with rasterio.open(dem_tif_path) as src:
        raster_crs = src.crs

        # Reproject grid to match raster CRS for clipping
        grid_proj = grid_gdf.to_crs(raster_crs)
        geometry = [mapping(grid_proj.geometry.unary_union)]

        try:
            # Mask the DEM to the extent of the grid
            elev_data, elev_transform = mask(src, geometry, crop=True, filled=True)
        except ValueError as e:
            print("â�Œ Overlay clipping failed:", e)
            return

        # Extract the elevation array and replace nodata with NaN
        elev_data = elev_data[0]
        elev_data = np.where(elev_data == src.nodata, np.nan, elev_data)

        # Normalize elevation values and apply colormap
        vmin, vmax = 0, 30
        norm = np.clip((elev_data - vmin) / (vmax - vmin), 0, 1)
        rgba = (plt.cm.terrain(norm) * 255).astype("uint8")

        # Convert RGBA array to PNG and save
        image = Image.fromarray(rgba, mode="RGBA")
        png_path = "/kaggle/working/elevation_overlay.png"
        image.save(png_path, format="PNG")

        # Transform raster bounds to WGS84 for mapping
        bounds = array_bounds(elev_data.shape[0], elev_data.shape[1], elev_transform)
        bounds_wgs84 = transform_bounds(raster_crs, "EPSG:4326", *bounds)
        sw = [bounds_wgs84[1], bounds_wgs84[0]]
        ne = [bounds_wgs84[3], bounds_wgs84[2]]

        # Add raster as a semi-transparent image overlay
        ImageOverlay(
            image=png_path,
            bounds=[sw, ne],
            name=name,
            opacity=0.65,
            interactive=True,
            cross_origin=False,
            zindex=999,
            show=show,
        ).add_to(m)

        print("âœ… Elevation overlay added.")

        # Add a custom HTML+JS legend that appears when the overlay is visible
        legend = MacroElement()
        legend._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <div id="elevation-legend" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 180px;
            height: 110px;
            z-index:9999;
            background-color: white;
            border:2px solid grey;
            padding: 10px;
            font-size: 14px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            display: none;
        ">
            <b>Elevation Above River (m)</b><br>
            <svg width="160" height="10">
              <defs>
                <linearGradient id="elevGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%"  stop-color="#1a9641" />
                  <stop offset="25%" stop-color="#a6d96a" />
                  <stop offset="50%" stop-color="#ffffbf" />
                  <stop offset="75%" stop-color="#fdae61" />
                  <stop offset="100%" stop-color="#d7191c" />
                </linearGradient>
              </defs>
              <rect x="0" y="0" width="160" height="10" fill="url(#elevGrad)" />
            </svg><br>
            <div style="display: flex; justify-content: space-between;">
                <span>0m</span><span>15m</span><span>30m</span>
            </div>
        </div>
        <script>
        var legend = document.getElementById('elevation-legend');
        map.on('overlayadd', function(e) {{
            if (e.name === '{name}') legend.style.display = 'block';
        }});
        map.on('overlayremove', function(e) {{
            if (e.name === '{name}') legend.style.display = 'none';
        }});
        </script>
        {{% endmacro %}}
        """)
        m.get_root().add_child(legend)



def get_zoom_center_for_clustered_cell(grid, zoom_best=1, zoom_factor=10000):
    """
    Returns the zoom level and center coordinates (lat, lon) for the highest scoring cell
    within a specific ranked cluster (e.g., cluster_rank == 1 for the best cluster).

    Parameters:
    ----------
    grid : GeoDataFrame
        The grid with prospect scores and cluster rankings. Must include 'cluster_rank'.
    zoom_best : int, optional
        Rank of the cluster to zoom into (1 = best cluster, 2 = second-best, etc.).
    zoom_factor : int, optional
        Controls how much area around the cell should be visible (larger = more context).

    Returns:
    -------
    tuple
        (zoom_level : int, center_coords : tuple(lat, lon))
    """
    if grid.empty or "cluster_rank" not in grid.columns:
        raise ValueError("Grid must include a 'cluster_rank' column.")

    # Filter for the requested cluster
    cluster_cells = grid[grid["cluster_rank"] == zoom_best]
    if cluster_cells.empty:
        raise ValueError(f"No cells found for cluster_rank = {zoom_best}")

    # Get the best cell (highest score) within the cluster
    selected_cell = cluster_cells.sort_values("prospect_score", ascending=False).iloc[0]

    # Convert centroid to WGS84 lat/lon
    centroid_wgs = (
        gpd.GeoSeries([selected_cell.geometry.centroid], crs=grid.crs)
        .to_crs(epsg=4326)
        .iloc[0]
    )
    center_coords = (centroid_wgs.y, centroid_wgs.x)

    # Estimate size of cell in meters
    bounds = selected_cell.geometry.bounds
    cell_width_m = bounds[2] - bounds[0]
    cell_height_m = bounds[3] - bounds[1]
    max_cell_dimension = max(cell_width_m, cell_height_m)
    desired_area_m = max_cell_dimension * zoom_factor

    # Estimate zoom level from area size
    def area_to_zoom(area_m):
        zoom = round(18 - math.log(area_m / 40075016.686 * 256, 2))
        return max(min(zoom, 18), 1)

    zoom_level = area_to_zoom(desired_area_m)

    return zoom_level, center_coords



def make_prospecting_map(grid_dict, cell_sizes, center=None, zoom=None,
                         rivers_fp=None, biomass_tif=None, biomass_colors=None,
                         sites_gdf=None, show_biomass=True, show_rivers=True,
                         show_sites=True, show_density=True, show_clusters=True,
                         legend=True):
    """
    Builds and returns a Folium map showing multi-resolution prospect scores and optional overlays.

    This function supports rendering a rich, interactive map with:
    - Prospect score choropleths (per cell size)
    - Archaeological site clusters
    - LiDAR survey density
    - Rivers and land cover overlays (optional)
    - Zoom and center controls

    Parameters:
    ----------
    grid_dict : dict
        Dictionary of cell size â†’ result dicts containing 'grid' (GeoDataFrame).
    cell_sizes : list of int
        List of cell sizes (in meters), from coarse to fine resolution.
    center : tuple of float, optional
        Map center coordinates as (lat, lon). If None, defaults to central Amazon.
    zoom : int, optional
        Initial map zoom level. If None, defaults to 5.
    rivers_fp : str, optional
        File path to river GeoJSON layer to display.
    biomass_tif : str, optional
        File path to categorical biomass/land cover TIFF.
    biomass_colors : dict, optional
        Color mapping for land cover values (e.g., MapBiomas codes).
    sites_gdf : GeoDataFrame, optional
        Point features representing archaeological sites.
    show_biomass : bool, optional
        Whether to render biomass overlay.
    show_rivers : bool, optional
        Whether to show river polylines.
    show_sites : bool, optional
        Whether to add site markers.
    show_density : bool, optional
        Whether to show per-cell site count shading.
    show_clusters : bool, optional
        Whether to highlight clusters of high-score, site-free cells.
    legend : bool, optional
        Whether to display a color legend on the final resolution layer.

    Returns:
    -------
    folium.Map
        The final interactive map object, ready for display or export.
    """
    # Initialize base map centered on the Amazon, or user-defined location
    m = folium.Map(location=center or [-5, -60], zoom_start=zoom or 5, tiles="CartoDB positron")

    # Inject custom JS/CSS for marker clustering and styling
    inject_map_assets(m)

    # Optional: Add river network layer
    if show_rivers and rivers_fp:
        add_rivers_layer(m, rivers_fp, show=True)

    # Optional: Add biomass land cover overlay
    if show_biomass and biomass_tif:
        add_biomass_overlay(m, biomass_tif, biomass_colors, show=True)

    # Loop over resolutions and add data layers
    for size in cell_sizes:
        grid = grid_dict[size]['grid']
        if grid.empty:
            continue  # Skip empty resolution levels

        is_final = (size == cell_sizes[-1])  # Last level = finest resolution
        size_km = size // 1000
        grid_wgs = ensure_wgs84(grid).reset_index()

        # Optionally add cluster outlines
        if show_clusters:
            grid = add_cluster_outlines(m, grid, size_km, min_cluster_size=10, show=is_final)

        # Add the main choropleth layer for prospect scores
        add_prospect_choropleth(
            m, grid_wgs, size_km, show=is_final,
            legend_name="Prospect Score (1â€“100)" if (legend and is_final) else None
        )

        # Optional: overlay site count shading
        if show_density:
            add_site_density(m, grid_wgs, size_km, show=is_final)

    # Optional: plot archaeological site markers
    if show_sites and sites_gdf is not None:
        add_all_sites_fast_clustered(m, sites_gdf)

    # Enable layer toggles
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def build_prospecting_map(
    results,
    cell_sizes,
    biomass_tif,
    biomass_colors,
    sites_gdf,
    rivers_fp,
    elev_above_river_tif,
    zoom=5,
    zoom_best=0,
    show_biomass=True,
    show_rivers=True,
    show_sites=True,
    show_density=True,
    show_clusters=True,
    legend=True,
    output_path="/kaggle/working/prospecting_score_map.html"
):
    """
    Builds and saves a Folium map showing archaeological prospect scores, clusters, and overlays.

    This is the top-level map rendering function. It optionally zooms to the best cell
    in a specified ranked cluster (via `zoom_best`), adds rivers, land cover, sites,
    and cluster outlines, and writes the result to an interactive HTML map file.

    Parameters:
    ----------
    results : dict
        Dictionary of cell size â†’ result dicts containing 'grid' (GeoDataFrame).
    cell_sizes : list of int
        Ordered list of cell sizes (in meters), from coarse to fine resolution.
    biomass_tif : str
        Path to MapBiomas or other land cover raster in categorical format.
    biomass_colors : dict
        Color map (int category â†’ hex color) for rendering biomass overlay.
    sites_gdf : GeoDataFrame
        Archaeological site locations to display.
    rivers_fp : str
        Path to rivers GeoJSON file to overlay.
    elev_above_river_tif : str
        Path to DEM of elevation above river (not used directly here, but reserved).
    zoom_best : int, optional
        If >0, zooms to the highest scoring cell in the N-th best cluster.
    show_biomass : bool
        Toggle to include the biomass/land cover layer.
    show_rivers : bool
        Toggle to show hydrographic line data.
    show_sites : bool
        Toggle to show archaeological site markers.
    show_density : bool
        Toggle to enable site count heatmap per cell.
    show_clusters : bool
        Toggle to display cluster outlines and ranking.
    legend : bool
        Whether to include a color legend for the prospect choropleth.
    output_path : str
        Output file path for the saved interactive map (HTML).

    Returns:
    -------
    None
        Saves a rendered map with all selected overlays and interactivity.
    """
    # --------------------------------------------------------------------------
    # Set default view (center of Amazon basin if no zoom specified)
    # --------------------------------------------------------------------------
    center = [-5, -60]
    zoom = zoom

    # --------------------------------------------------------------------------
    # If zoom_best is set, center map on best-scoring cell in ranked cluster
    # --------------------------------------------------------------------------
    if zoom_best > 0:
        try:
            # âœ… Load clustered cells from saved CSV file
            clustered_csv_path = "/kaggle/working/clustered_cells.csv"
            clustered_df = pd.read_csv(clustered_csv_path)

            # âœ… Convert WKT geometry back to shapely objects
            grid = gpd.GeoDataFrame(
                clustered_df,
                geometry=gpd.GeoSeries.from_wkt(clustered_df["geometry"]),
                crs="EPSG:4326"
            )

            # âœ… Compute zoom and center based on highest scoring cell in cluster
            zoom, center = get_zoom_center_for_clustered_cell(grid, zoom_best)

        except Exception as e:
            print(f"âš ï¸� Zoom to cluster {zoom_best} failed: {e}. Using default map view.")

    # --------------------------------------------------------------------------
    # Build the map with selected overlays and rendering options
    # --------------------------------------------------------------------------
    m = make_prospecting_map(
        grid_dict=results,
        cell_sizes=cell_sizes,
        center=center,
        zoom=zoom,
        rivers_fp=rivers_fp,
        biomass_tif=biomass_tif,
        biomass_colors=biomass_colors,
        sites_gdf=sites_gdf,
        show_biomass=show_biomass,
        show_rivers=show_rivers,
        show_sites=show_sites,
        show_density=show_density,
        show_clusters=show_clusters,
        legend=legend
    )

    # --------------------------------------------------------------------------
    # Save the interactive map to disk
    # --------------------------------------------------------------------------
    m.save(output_path)
    print(f"âœ… Map saved: {output_path}")



def save_static_map_image(html_path, output_png="map.png", delay=3, width=1024, height=768,
                          resolution_scale=1):
    """
    Loads a Folium map (HTML file) and captures a static PNG image using Selenium.

    This function uses a headless Chrome browser to render the map and save a screenshot.
    Useful for generating report-ready map images or animation frames from dynamic layers.

    Parameters:
    ----------
    html_path : str
        Path to the saved Folium HTML file to render.
    output_png : str, optional
        Destination file path for the output PNG image.
    delay : int, optional
        Seconds to wait for the map tiles and overlays to load (default: 3 seconds).
    width : int, optional
        Width of the browser window in pixels before applying resolution scale.
    height : int, optional
        Height of the browser window in pixels before applying resolution scale.
    resolution_scale : float, optional
        Multiplier for high-DPI rendering. Use 2 for 2x resolution, etc.

    Returns:
    -------
    None
        Saves a PNG screenshot of the map in-place.
    """
    # Configure headless Chrome window with custom resolution
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"--window-size={width * resolution_scale},{height * resolution_scale}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # Launch headless browser and load map
    driver = webdriver.Chrome(options=options)
    file_url = "file://" + os.path.abspath(html_path)
    driver.get(file_url)

    # Allow time for base map and overlays to fully load
    time.sleep(delay)

    # Capture screenshot of the rendered map (entire body)
    body = driver.find_element(By.TAG_NAME, "body")
    body.screenshot(output_png)

    # Clean up browser session
    driver.quit()
    print(f"âœ… Saved screenshot to {output_png} (resolution scale: {resolution_scale}x)")



%%time
# ==============================================================================
# STEP: Load previously saved grid results and render the final interactive map
# ==============================================================================

import pickle

# --------------------------------------------------------------------------
# Load saved grid analysis results (multi-resolution scoring, features, model)
# --------------------------------------------------------------------------
with open("/kaggle/input/amazon-grid-analysis/results.pkl", "rb") as file:
    results = pickle.load(file)

# Sort available grid resolutions from largest to smallest (e.g., 64000, 32000, ...)
cell_sizes = sorted(list(results.keys()), reverse=True)


# --------------------------------------------------------------------------
# Build final interactive Folium map using the finest grid resolution
# --------------------------------------------------------------------------
build_prospecting_map(
    results=results,
    cell_sizes=[4000, 1000], 
    biomass_tif="/kaggle/working/MapBiomas_Amazon_2018_EPSG3857.tif",
    biomass_colors=mapbiomas_colors,
    sites_gdf=sites_gdf_in_basin,
    rivers_fp="/kaggle/input/amazon-river-map-processed/amazon_rivers_wgs84.geojson",
    elev_above_river_tif="/kaggle/working/elevation_above_river_3857.tif",
)



RESOLUTION = cell_sizes[-1]
build_prospecting_map(
    results=results,
    cell_sizes= [int(cell_sizes[-1])], 
    biomass_tif="/kaggle/working/MapBiomas_Amazon_2018_EPSG3857.tif",
    biomass_colors=mapbiomas_colors,
    sites_gdf = sites_gdf_in_basin,
    rivers_fp = "/kaggle/input/amazon-river-map-processed/amazon_rivers_wgs84.geojson",
    elev_above_river_tif = "/kaggle/working/elevation_above_river_3857.tif",
    show_biomass=True,
    show_rivers=False,
    show_sites=False,
    show_density=True,
    show_clusters=False,
    legend=True,
    zoom=6,
    output_path=f"/kaggle/working/prospecting_score_map_{RESOLUTION}.html"
)

save_static_map_image(
    f"/kaggle/working/prospecting_score_map_{RESOLUTION}.html",
    f"final_prediction_map_{RESOLUTION}.png",
    width=2*1048,
    height=2*756,
    resolution_scale=1
)

