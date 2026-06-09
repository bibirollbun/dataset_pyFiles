!apt-get update -qq && apt-get install -y unzip --quiet
!pip install earthaccess xarray netCDF4 geopandas rasterio --quiet
!pip install --upgrade earthaccess --quiet


# ==============================================================================
# SECTION: EARTHDATA AUTHENTICATION & LIBRARY IMPORTS
# ==============================================================================
"""
Sets up authentication for NASA Earthdata access and loads required geospatial
and remote sensing libraries for working with satellite-based elevation, land cover,
or climate products (e.g., GEDI, MODIS, SMAP).

- Loads Kaggle secrets for Earthdata login
- Initializes Earthaccess session for data discovery and download
- Prepares geospatial tools for file handling, merging, and rasterization
"""

# --------------------------------------------------------------------------
# Standard Library
# --------------------------------------------------------------------------
import glob
import os
import shutil
import tempfile
import zipfile

# --------------------------------------------------------------------------
# Scientific and Geospatial Libraries
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

# --------------------------------------------------------------------------
# Raster I/O and Transformation
# --------------------------------------------------------------------------
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.merge import merge as rio_merge

# --------------------------------------------------------------------------
# Earthdata Access Setup (requires valid credentials)
# --------------------------------------------------------------------------
import earthaccess
from kaggle_secrets import UserSecretsClient

# âœ… Load NASA Earthdata credentials securely from Kaggle Secrets
secrets = UserSecretsClient()
os.environ["EARTHDATA_USERNAME"] = secrets.get_secret("EARTHDATA_USER")
os.environ["EARTHDATA_PASSWORD"] = secrets.get_secret("EARTHDATA_PASS")

# âœ… Log in to Earthdata (uses environmental variables and caches session)
auth = earthaccess.login()

# âœ… Access internal session store (used by `earthaccess.get(...)`)
from earthaccess import __store__

os.mkdir('swot_node_files')


def process_node_tile(bbox, tile_index, temporal=("2024-06-01", "2024-06-15"), max_granules=5):
    """
    Downloads and processes SWOT L2 RiverSP Node data for a given bounding box tile.

    This function:
    - Queries Earthdata for granules in a specified temporal window
    - Downloads and unzips granule shapefiles
    - Extracts Water Surface Elevation (WSE) values from shapefiles
    - Aggregates max WSE per point
    - Rasterizes those points into a GeoTIFF using EPSG:4326
    - Saves the tile to disk

    Parameters:
    ----------
    bbox : tuple
        Bounding box (xmin, ymin, xmax, ymax) for the tile in EPSG:4326.
    tile_index : int
        Tile number used in saved output filename.
    temporal : tuple of str, optional
        Date range (start, end) in YYYY-MM-DD format.
    max_granules : int, optional
        Maximum number of SWOT granules to download and process.

    Returns:
    -------
    str or None
        Path to the saved raster tile, or None if nothing was saved.
    """
    import zipfile
    import glob

    output_dir = 'swot_node_files'
    xmin, ymin, xmax, ymax = bbox

    print(f"\nğŸ”¹ Processing tile {tile_index + 1}/36: ({xmin}, {ymin}, {xmax}, {ymax})")

    # Search SWOT granules within a fixed bounding box and date range
    results = earthaccess.search_data(
        short_name="SWOT_L2_HR_RiverSP_Node_2.0",
        temporal=(temporal[0], temporal[1]),
        bounding_box=(-80, -20, -75, -15)  # <- hardcoded bounding box
    )

    print("Granules in wet season 2024:", len(results))
    results = results[:max_granules]

    if not results:
        print("âš ï¸� No granules found.")
        return None

    all_dfs = []

    for granule in results:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"ğŸ“¥ Downloading granule to: {tmpdir}")
            try:
                files = __store__.get([granule], local_path=tmpdir)
            except Exception as e:
                print(f"â�Œ Download failed: {e}")
                continue

            # Extract ZIP contents
            for f in files:
                if f.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(f, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                    except zipfile.BadZipFile:
                        print(f"â�Œ Corrupted zip: {f}")
                        continue

            # Find shapefiles recursively
            shp_files = glob.glob(os.path.join(tmpdir, "**", "*.shp"), recursive=True)
            if not shp_files:
                print("âš ï¸� No .shp files found.")
                continue

            # Read and clean shapefile
            try:
                gdf = gpd.read_file(shp_files[0])
                gdf = gdf[['geometry', 'wse']].copy()
                gdf = gdf[np.isfinite(gdf['wse'])]     # Drop NaNs
                gdf = gdf[gdf['wse'] > -1e6]           # Filter dummy fill values

                # Extract coordinates for rasterization
                gdf['x'] = gdf.geometry.x.round(5)
                gdf['y'] = gdf.geometry.y.round(5)
                all_dfs.append(gdf[['x', 'y', 'wse']])
            except Exception as e:
                print(f"â�Œ Failed to read shapefile: {e}")
                continue

    if not all_dfs:
        print("âš ï¸� No valid data extracted.")
        return

    # Merge all extracted points and calculate max WSE per location
    merged = pd.concat(all_dfs)
    max_wse = merged.groupby(['x', 'y'])['wse'].max().reset_index()
    geometry = [Point(xy) for xy in zip(max_wse.x, max_wse.y)]
    gdf_points = gpd.GeoDataFrame(max_wse, geometry=geometry, crs="EPSG:4326")

    # Rasterization parameters
    res_deg = 0.0025
    width = int((xmax - xmin) / res_deg)
    height = int((ymax - ymin) / res_deg)
    transform = from_origin(xmin, ymax, res_deg, res_deg)

    # Rasterize WSE values
    raster = rasterize(
        [(geom, val) for geom, val in zip(gdf_points.geometry, gdf_points['wse'])],
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype='float32'
    )

    # Save to GeoTIFF
    out_fp = os.path.join(output_dir, f"tile_{tile_index}.tif")
    with rasterio.open(out_fp, 'w', driver='GTiff',
                       height=height, width=width,
                       count=1, dtype='float32',
                       crs='EPSG:4326',
                       transform=transform,
                       nodata=np.nan,
                       compress='deflate') as dst:
        dst.write(raster, 1)

    print(f"âœ… Tile saved: {out_fp}")
    return out_fp



%%time

# ==============================================================================
# STEP: Define 5Â°x5Â° Tiling Grid Over the Amazon Basin and Process Each Tile
# ==============================================================================

# --------------------------------------------------------------------------
# Define spatial bounds of the Amazon basin region (in degrees)
# --------------------------------------------------------------------------
minx, miny, maxx, maxy = -80, -20, -50, 10  # bounding box (WGS84)
tile_size = 5  # degrees per tile (both lat and lon)

# Generate X and Y coordinate breaks for tiling
x_coords = np.arange(minx, maxx, tile_size)
y_coords = np.arange(miny, maxy, tile_size)

# Generate bounding boxes for all tiles: (xmin, ymin, xmax, ymax)
tiles = [(x, y, x + tile_size, y + tile_size) for x in x_coords for y in y_coords]

# --------------------------------------------------------------------------
# Define SWOT search window (wet season example)
# --------------------------------------------------------------------------
temporal_range = ("2024-06-01", "2024-06-15")

# --------------------------------------------------------------------------
# Loop through tiles and process SWOT Node data into GeoTIFFs
# --------------------------------------------------------------------------
tile_paths = []

for i, tile in enumerate(tiles):
    out_fp = process_node_tile(tile, i, temporal=temporal_range, max_granules=2)
    if out_fp:
        tile_paths.append(out_fp)



from rasterio.merge import merge as rio_merge

# ==============================================================================
# STEP: Merge All Individual SWOT Tile Rasters into a Mosaic
# ==============================================================================

if tile_paths:
    # Open each individual tile raster
    srcs = [rasterio.open(fp) for fp in tile_paths]

    # Merge all tile rasters into one mosaic array and compute new transform
    mosaic, out_transform = rio_merge(srcs)

    # Copy metadata from the first tile and update dimensions and transform
    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "driver": "GTiff",
        "compress": "deflate",
        "nodata": np.nan
    })

    # Ensure output directory exists and define output filepath
    os.makedirs("output", exist_ok=True)
    merged_fp = "output/amazon_swot_max_wse.tif"

    # Save the merged raster to disk
    with rasterio.open(merged_fp, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"\nâœ… Merged GeoTIFF saved to: {merged_fp}")

else:
    print("âš ï¸� No tiles were generated â€” nothing to merge.")

# ==============================================================================
# STEP: Clean Up Temporary Tile Folder
# ==============================================================================
shutil.rmtree("swot_node_tiles", ignore_errors=True)
print("ğŸ§¹ Deleted swot_node_tiles folder.")


